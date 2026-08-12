"""`--sparse`: skip pyramid tasks whose input has no stored chunk.

The claim being tested is that this is **exact, not a heuristic** — the output must be
byte-identical to a full run, not merely close. That is what separates it from the
occupancy prefilters elsewhere (which need dilating because they ask a different
question): here an absent chunk object *is* all fill, so a task with no stored input
writes nothing whether it runs or not.
"""

import numpy as np
import pytest

from em_volume_tools import cli, convert, rebuild_pyramid
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.ops._multiscale import NothingStored
from em_volume_tools.profiles import StorageProfile, zarr3_create_spec


def _sparse_volume(tmp_path, name="v", *, chunk=(8, 8, 8), shape=(64, 64, 64)):
    """A volume whose data occupies one corner, so most chunks are never stored.

    Built by `create` + a direct write rather than `convert`, because that is how a
    ground-truth volume gets made: an empty frame with a few pieces written into it.
    """
    from em_volume_tools import create_volume

    dst = str(tmp_path / f"{name}.zarr")
    create_volume(dst, shape=shape, voxel_size=(8, 8, 8), dtype="uint32",
                  chunk=chunk, kind="segmentation", levels=1)
    be = open_backend({"backend": "zarr3", "path": f"{dst}/0"})
    data = np.arange(1, 8 * 8 * 8 + 1, dtype=np.uint32).reshape(8, 8, 8)
    be.write_region((slice(0, 8), slice(8, 16), slice(16, 24)), data)
    return dst


def _levels(volume, n):
    out = []
    for i in range(n):
        be = open_backend({"backend": "zarr3", "path": f"{volume}/{i}"})
        out.append(be.read_region(tuple(slice(0, s) for s in be.shape)))
    return out


def _stored_keys(volume, level):
    from em_volume_tools.location import list_keys

    return sorted(list_keys(volume, str(level)))


# --------------------------------------------------------------------------- #
# it is exact
# --------------------------------------------------------------------------- #
def test_sparse_and_full_rebuilds_are_identical(tmp_path):
    """Same voxels, same stored objects. If this ever diverges, --sparse is unsafe."""
    full = _sparse_volume(tmp_path, "full")
    lean = _sparse_volume(tmp_path, "lean")

    a = rebuild_pyramid(full, start_level=0, kind="segmentation", min_dim=8,
                        profile="local")
    b = rebuild_pyramid(lean, start_level=0, kind="segmentation", min_dim=8,
                        profile="local", sparse=True)
    assert a["num_levels"] == b["num_levels"] > 1

    for level, (x, y) in enumerate(zip(_levels(full, a["num_levels"]),
                                       _levels(lean, b["num_levels"]))):
        np.testing.assert_array_equal(x, y, err_msg=f"level {level} differs")
        assert _stored_keys(full, level) == _stored_keys(lean, level), \
            f"level {level} stores a different set of objects"


def test_sparse_skips_nearly_every_task(tmp_path):
    """The point of the exercise: the work has to actually go away."""
    vol = _sparse_volume(tmp_path)
    summary = rebuild_pyramid(vol, start_level=0, kind="segmentation", min_dim=8,
                              profile="local", sparse=True)
    ran = sum(summary["status_counts"].values())
    # level 1 alone is (32/8)^3 = 64 grid tasks, and the data touches one chunk
    assert ran < 20, f"expected a handful of tasks, ran {ran}"
    assert summary["status_counts"].get("written", 0) >= 1, "the real data was written"


def test_the_manifest_denominator_is_the_filtered_total(tmp_path):
    """Invariant 11: a reader can only trust `total`, so it must be what was dispatched.

    Recording the whole grid here would make `em-vol progress` sit at a few percent
    forever on a run that had in fact finished.
    """
    import json

    vol = _sparse_volume(tmp_path)
    progress = str(tmp_path / "p.jsonl")
    rebuild_pyramid(vol, start_level=0, kind="segmentation", min_dim=8,
                    profile="local", sparse=True, progress_path=progress)

    meta, statuses = {}, {}
    for line in open(progress):
        rec = json.loads(line)
        if "meta" in rec:
            meta[rec["group"]] = rec["meta"]
        else:
            statuses[rec["group"]] = statuses.get(rec["group"], 0) + 1
    assert meta, "levels recorded their totals"
    for level, m in meta.items():
        assert m["total"] == statuses.get(level, 0), \
            f"level {level}: total {m['total']} but {statuses.get(level, 0)} tasks ran"
        # Recorded for every filtered level, including the coarsest, whose whole grid is
        # one task that cannot be skipped — hence >=, not >.
        assert m["grid_total"] >= m["total"], "and it says what the full grid would be"
        assert m["skipped_empty"] == m["grid_total"] - m["total"]
    assert any(m["skipped_empty"] for m in meta.values()), "something was skipped"


# --------------------------------------------------------------------------- #
# the ways it must refuse rather than silently write nothing
# --------------------------------------------------------------------------- #
def test_a_seed_level_with_no_stored_chunks_is_refused(tmp_path):
    """Otherwise every task is skipped and the run reports success having done nothing.

    That is invariant 4's failure mode — a successful run that writes nothing — and it is
    exactly what a wrong --start-level or a not-yet-filled volume would produce.
    """
    from em_volume_tools import create_volume

    empty = str(tmp_path / "empty.zarr")
    create_volume(empty, shape=(64, 64, 64), voxel_size=(8, 8, 8), dtype="uint32",
                  chunk=(8, 8, 8), kind="segmentation", levels=1)
    with pytest.raises(NothingStored, match="no stored chunks"):
        rebuild_pyramid(empty, start_level=0, kind="segmentation", min_dim=8,
                        profile="local", sparse=True)


def test_a_sharded_level_filters_at_SHARD_granularity_and_stays_exact(tmp_path):
    """Sharding does not break the filter, because the cell shape is the WRITE chunk.

    On a sharded zarr level the stored object is the shard and its key is the shard's cell
    (verified: `chunks` reports (32,32,32) while `read_chunks` reports (8,8,8), and the
    only data key is `c/0/0/0`). Since the filter takes its cell shape from `chunks`, the
    occupancy grid it builds is the shard grid — coarser, so less is skipped, but still
    exact: a shard with no object means every chunk inside it is absent.

    Precomputed sharding is the case that cannot be answered this way, and
    `occupied_cells` raises there rather than guessing; it is unreachable from here
    because this package writes no sharded precomputed.
    """
    src = str(tmp_path / "s.src.zarr")
    data = np.zeros((64, 64, 64), dtype=np.uint32)
    data[:8, :8, :8] = 5
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint32",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)

    profile = StorageProfile("zarr3", chunk=(8, 8, 8), shard=(32, 32, 32))
    full, lean = str(tmp_path / "sh_full.zarr"), str(tmp_path / "sh_lean.zarr")
    for dst in (full, lean):
        convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", multiscale=False,
                profile=profile, delete_existing=True)

    a = rebuild_pyramid(full, start_level=0, kind="segmentation", min_dim=8,
                        profile=profile)
    b = rebuild_pyramid(lean, start_level=0, kind="segmentation", min_dim=8,
                        profile=profile, sparse=True)
    assert a["num_levels"] == b["num_levels"] > 1
    for level, (x, y) in enumerate(zip(_levels(full, a["num_levels"]),
                                       _levels(lean, b["num_levels"]))):
        np.testing.assert_array_equal(x, y, err_msg=f"level {level} differs")


# --------------------------------------------------------------------------- #
# convert, and the CLI
# --------------------------------------------------------------------------- #
def test_convert_filters_the_pyramid_but_not_level_zero(tmp_path):
    """Level 0 reads a foreign source, whose stored chunks are not ours to inspect."""
    src = str(tmp_path / "c.src.zarr")
    data = np.zeros((32, 32, 32), dtype=np.uint32)
    data[:8, :8, :8] = 9
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint32",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)

    dst = str(tmp_path / "c.zarr")
    summary = convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
                     min_dim=8, profile="local", sparse=True, delete_existing=True)
    # level 0 ran its whole grid: 4^3 = 64 tasks, most of them "empty"
    assert sum(summary["status_counts"].values()) >= 64
    np.testing.assert_array_equal(_levels(dst, 1)[0], data)


def test_rebuild_takes_its_level_count_from_the_volume(tmp_path, caplog):
    """How many levels a pyramid has is recorded in the volume, not a preference.

    Inheriting `convert`'s default of 8 made a rebuild plan one level fewer than the
    9-level volume it was repairing — leaving the top level serving whatever it held
    before, with every shape matching so the mismatch check saw nothing.
    """
    from em_volume_tools import create_volume, describe

    vol = str(tmp_path / "deep.zarr")
    create_volume(vol, shape=(64, 64, 64), voxel_size=(8, 8, 8), dtype="uint32",
                  chunk=(8, 8, 8), kind="segmentation", min_dim=1, max_levels=4)
    assert len(describe(vol)["levels"]) == 4, "a depth the default 8 would not produce"
    be = open_backend({"backend": "zarr3", "path": f"{vol}/0"})
    be.write_region((slice(0, 64),) * 3, np.full((64, 64, 64), 3, dtype=np.uint32))

    with caplog.at_level("INFO"):
        summary = rebuild_pyramid(vol, start_level=0, kind="segmentation", min_dim=1,
                                  profile="local")
    assert summary["num_levels"] == 4, "the rebuild covered the volume's own levels"
    assert "matching the volume's recorded levels" in caplog.text

    # and an explicit value still wins, which is how a pyramid gets EXTENDED
    assert rebuild_pyramid(vol, start_level=0, kind="segmentation", min_dim=1,
                           profile="local", max_levels=6)["num_levels"] == 6


def test_a_single_scale_volume_still_gets_a_pyramid_built(tmp_path, caplog):
    """`write` and `relabel` both end by saying "run downsample" — that must still work.

    Deriving the cap from a volume that records ONE level would make the documented next
    step a silent no-op, so there the conversion default applies instead.
    """
    vol = _sparse_volume(tmp_path, "single")            # created with levels=1
    with caplog.at_level("INFO"):
        summary = rebuild_pyramid(vol, start_level=0, kind="segmentation", min_dim=8,
                                  profile="local", sparse=True)
    assert summary["num_levels"] > 1, "a pyramid was built, not nothing"
    assert "no pyramid to match" in caplog.text


def test_the_cli_passes_sparse_through(tmp_path, capsys):
    vol = _sparse_volume(tmp_path)
    assert cli.main(["downsample", vol, "--start-level", "0", "--kind", "segmentation",
                     "--min-dim", "8", "--serial", "--sparse"]) == 0
    assert cli._parse_args(["convert", "--src", "a", "--dst", "b", "--sparse"]).sparse
