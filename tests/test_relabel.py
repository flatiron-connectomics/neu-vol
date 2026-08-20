"""Renumbering each occupied region of a sparse volume into its own id range.

The property that matters is the one the operation exists for: an id that meant two
different cells in two regions must mean one cell afterwards, and every voxel must still
be labelled — nothing dropped, nothing merged. The mapping file has to make that
reversible, because once the ids are overwritten it is the only record.
"""

import json

import numpy as np
import pytest

from neu_vol import cli, convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.ops.relabel import (_apply_map, apply_relabel, default_map_path,
                                         plan_relabel, relabel)
from neu_vol.profiles import zarr3_create_spec
from neu_vol.source_metadata import level_spec


def _volume(tmp_path, name, seg, *, profile="local-neuroglancer", chunk=(8, 8, 8)):
    """A real two-level volume holding ``seg``, built by the production op."""
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, seg.shape, str(seg.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in seg.shape), seg)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", profile=profile,
            chunk=chunk, factors=[(2, 2, 2)], min_dim=8, delete_existing=True)
    return dst


def _colliding(shape=(32, 32, 32)):
    """Two separated regions that both use ids 1 and 2 — the gt_v1 situation."""
    seg = np.zeros(shape, np.uint64)
    seg[0:8, 0:8, 0:4] = 1
    seg[0:8, 0:8, 4:8] = 2
    seg[16:24, 24:32, 8:12] = 1
    seg[16:24, 24:32, 12:16] = 2
    seg[16:24, 24:32, 12:13] = 7          # a third id, only in the second region
    return seg


def _read(volume, level=0):
    be = open_backend(level_spec(volume, "neuroglancer_precomputed", level))
    return be.read_region(tuple(slice(0, int(s)) for s in be.shape))


# --------------------------------------------------------------------------- #
# the mapping primitive
# --------------------------------------------------------------------------- #
def test_apply_map_keeps_zero_and_rewrites_the_rest():
    data = np.array([[0, 5], [9, 5]], np.uint64)
    out = _apply_map(data, np.array([5, 9], np.uint64), np.array([1, 2], np.uint64))
    np.testing.assert_array_equal(out, [[0, 1], [2, 1]])


def test_apply_map_preserves_dtype_and_shape():
    data = np.zeros((3, 4, 5), np.uint64)
    data[1, 2, 3] = 400
    out = _apply_map(data, np.array([400], np.uint64), np.array([1], np.uint64))
    assert out.shape == data.shape and out.dtype == data.dtype
    assert int(out[1, 2, 3]) == 1 and int(out.sum()) == 1


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_a_destination_must_be_chosen_explicitly(tmp_path):
    """Neither default is safe: one publishes a second volume, the other overwrites the
    ids it is derived from."""
    dst = _volume(tmp_path, "v", _colliding())
    with pytest.raises(ValueError, match="exactly one"):
        plan_relabel(dst)
    with pytest.raises(ValueError, match="exactly one"):
        plan_relabel(dst, out=str(tmp_path / "o"), in_place=True)


def test_boxes_are_chunk_aligned_not_tightened(tmp_path):
    """Tightened boxes would leave data outside them holding old ids.

    A tightened box is the bbox of the nonzero voxels seen at a COARSE level, and mode
    downsampling can drop a stray voxel — so it can exclude scale-0 data that is really
    there. Chunk-aligned boxes provably cover every stored chunk, and being on the chunk
    grid also means no write is a partial-chunk read-modify-write.
    """
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[9:11, 9:11, 9:11] = 4              # small, well inside chunk (1,1,1)
    dst = _volume(tmp_path, "v", seg)
    plan = plan_relabel(dst, out=str(tmp_path / "o"))
    r = plan["regions"][0]
    assert tuple(r["lo"]) == (8, 8, 8) and tuple(r["hi"]) == (16, 16, 16)
    assert all(v % 8 == 0 for v in list(r["lo"]) + list(r["hi"]))


def test_an_oversized_region_is_refused_with_the_number(tmp_path):
    dst = _volume(tmp_path, "v", _colliding())
    with pytest.raises(ValueError, match="over the"):
        plan_relabel(dst, out=str(tmp_path / "o"), max_region_bytes=8)


def test_a_missing_level_and_a_missing_volume_are_named(tmp_path):
    dst = _volume(tmp_path, "v", _colliding())
    with pytest.raises(ValueError, match="no level 6"):
        plan_relabel(dst, out=str(tmp_path / "o"), level=6)
    with pytest.raises(FileNotFoundError, match="no volume found"):
        plan_relabel(str(tmp_path / "nope"), in_place=True)


def test_levels_above_the_renumbered_one_are_reported_as_stale(tmp_path):
    """They keep the old ids, so they disagree until `downsample` is re-run."""
    dst = _volume(tmp_path, "v", _colliding())
    assert plan_relabel(dst, in_place=True, level=0)["stale_levels"] == [1]


# --------------------------------------------------------------------------- #
# the renumbering itself
# --------------------------------------------------------------------------- #
def test_colliding_ids_become_distinct_and_nothing_is_lost(tmp_path):
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    result = relabel(src, out=out, map_path=str(tmp_path / "m.json"))

    before, after = _read(src), _read(out)
    # every labelled voxel is still labelled, and every background voxel still isn't
    np.testing.assert_array_equal(before != 0, after != 0)
    assert result["n_distinct_in"] == 3          # ids 1, 2, 7
    assert result["n_labels_out"] == 5           # 2 in the first region, 3 in the second
    assert result["collisions_resolved"] == 2    # 1 and 2 were shared

    # the two regions' id ranges do not overlap
    lo0, hi0 = result["regions"][0]["new_id_range"]
    lo1, hi1 = result["regions"][1]["new_id_range"]
    assert [lo0, hi0] == [1, 2] and [lo1, hi1] == [3, 5]

    # what was one id spanning both regions is now two ids, one per region
    was_one = before == 1
    ids_now = set(int(v) for v in np.unique(after[was_one]))
    assert len(ids_now) == 2, f"id 1 should have split, got {ids_now}"
    # ...and each new id sits in exactly one region
    for new_id in ids_now:
        zs = np.nonzero((after == new_id).any(axis=(1, 2)))[0]
        assert zs.max() - zs.min() < 16, "a new id spans both regions"


def test_a_label_keeps_its_exact_voxels(tmp_path):
    """Relabelling is a permutation of ids, not a change of geometry."""
    seg = _colliding()
    src = _volume(tmp_path, "v", seg)
    out = str(tmp_path / "out")
    result = relabel(src, out=out, map_path=str(tmp_path / "m.json"))
    before, after = _read(src), _read(out)

    for entry in result["regions"]:
        region = tuple(slice(lo, hi) for lo, hi in zip(entry["lo_zyx"], entry["hi_zyx"]))
        for old, new in entry["map"].items():
            np.testing.assert_array_equal(before[region] == int(old),
                                          after[region] == int(new))


def test_the_mapping_is_reversible(tmp_path):
    """The mapping is the only record once the ids are gone, so it must invert exactly."""
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    path = str(tmp_path / "m.json")
    relabel(src, out=out, map_path=path)

    with open(path) as f:
        saved = json.load(f)
    before, after = _read(src), _read(out)
    restored = np.zeros_like(after)
    for entry in saved["regions"]:
        region = tuple(slice(lo, hi) for lo, hi in zip(entry["lo_zyx"], entry["hi_zyx"]))
        block = after[region]
        out_block = np.zeros_like(block)
        for old, new in entry["map"].items():
            out_block[block == int(new)] = int(old)
        restored[region] = out_block
    np.testing.assert_array_equal(restored, before)

    # new ids are unique across the whole file, which is what "own range" means
    all_new = [v for e in saved["regions"] for v in e["map"].values()]
    assert len(all_new) == len(set(all_new))


def test_the_map_goes_through_the_kvstore_not_open(tmp_path):
    """So `--map s3://...` works, letting the mapping sit beside a remote volume.

    Pinned with a path whose parents do not exist: the file driver creates them, while
    `open()` raises. That is the same code path a remote destination takes, and the only
    part of it a test can exercise without a bucket.
    """
    src = _volume(tmp_path, "v", _colliding())
    path = str(tmp_path / "no" / "such" / "dir" / "m.json")
    result = relabel(src, out=str(tmp_path / "o"), map_path=path)
    assert result["map_path"] == path
    with open(path) as f:
        assert len(json.load(f)["regions"]) == 2


def test_in_place_renumbers_the_volume_itself(tmp_path):
    src = _volume(tmp_path, "v", _colliding())
    before = _read(src)
    result = relabel(src, in_place=True, map_path=str(tmp_path / "m.json"))
    after = _read(src)
    assert result["destination"] == src and result["in_place"]
    np.testing.assert_array_equal(before != 0, after != 0)
    assert set(int(v) for v in np.unique(after) if v) == {1, 2, 3, 4, 5}


def test_out_creates_the_volume_in_the_same_frame(tmp_path):
    """A voxel index has to mean the same thing in both, so the frame is copied."""
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    relabel(src, out=out, map_path=str(tmp_path / "m.json"))

    from neu_vol.source_metadata import describe
    a, b = describe(src), describe(out)
    assert a["shape"] == b["shape"] and a["dtype"] == b["dtype"]
    assert a["format"] == b["format"]
    assert sorted(a["levels"]) == sorted(b["levels"])
    assert a["level_voxel_sizes"] == b["level_voxel_sizes"]


def test_untouched_regions_of_the_new_volume_stay_empty(tmp_path):
    """Only the occupied regions are written, so the copy stays sparse."""
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    relabel(src, out=out, map_path=str(tmp_path / "m.json"))
    from neu_vol.cli import _stored_chunks
    assert _stored_chunks(out, "neuroglancer_precomputed", 0, "8_8_8") == \
        _stored_chunks(src, "neuroglancer_precomputed", 0, "8_8_8")


def test_block_size_makes_the_region_readable_off_the_id(tmp_path):
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    result = relabel(src, out=out, block_size=1000, map_path=str(tmp_path / "m.json"))
    assert result["regions"][0]["new_id_range"] == [1, 2]
    assert result["regions"][1]["new_id_range"] == [1001, 1003]
    assert set(int(v) for v in np.unique(_read(out)) if v) == {1, 2, 1001, 1002, 1003}


def test_block_size_too_small_refuses_rather_than_colliding(tmp_path):
    """Overflowing a block would run one region's ids into the next — the exact bug the
    whole operation exists to remove."""
    src = _volume(tmp_path, "v", _colliding())
    with pytest.raises(ValueError, match="block_size"):
        relabel(src, out=str(tmp_path / "o"), block_size=2,
                map_path=str(tmp_path / "m.json"))


def test_dry_run_writes_nothing_but_still_reports_the_ranges(tmp_path):
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    result = relabel(src, out=out, dry_run=True, map_path=str(tmp_path / "m.json"))
    assert result["dry_run"] and result["n_labels_out"] == 5
    assert result["regions"][1]["new_id_range"] == [3, 5]
    from neu_vol.source_metadata import detect_backend
    assert detect_backend(out) is None, "dry run created the destination"
    assert not (tmp_path / "m.json").exists(), "dry run wrote the mapping file"


def test_a_volume_with_no_collisions_is_left_semantically_alone(tmp_path):
    """Renumbering is still a permutation, but nothing was merged to begin with."""
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[0:8, 0:8, 0:8] = 3
    seg[16:24, 24:32, 8:16] = 9
    src = _volume(tmp_path, "v", seg)
    result = relabel(src, out=str(tmp_path / "o"), map_path=str(tmp_path / "m.json"))
    assert result["collisions_resolved"] == 0
    assert result["n_labels_out"] == 2


def test_a_volume_with_nothing_stored_is_refused(tmp_path):
    from neu_vol.ops.create import create_volume

    dst = str(tmp_path / "empty")
    create_volume(dst, format="precomputed", shape=(32, 32, 32), dtype="uint64",
                  voxel_size=(8, 8, 8), chunk=(8, 8, 8), kind="segmentation")
    with pytest.raises(ValueError, match="stores no chunks"):
        plan_relabel(dst, in_place=True)


def test_zarr_volumes_work_too(tmp_path):
    """Occupancy comes from chunk keys, which the two formats spell differently."""
    src = _volume(tmp_path, "v", _colliding(), profile="local")
    out = str(tmp_path / "out")
    result = relabel(src, out=out, map_path=str(tmp_path / "m.json"))
    assert result["n_labels_out"] == 5 and result["collisions_resolved"] == 2
    be = open_backend(level_spec(out, "zarr3", 0))
    after = be.read_region(tuple(slice(0, int(s)) for s in be.shape))
    assert set(int(v) for v in np.unique(after) if v) == {1, 2, 3, 4, 5}


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_map_path_defaults_to_the_destination_name():
    assert default_map_path("s3://b/sample3/gt_v2/", 0) == "gt_v2.relabel-0.json"
    assert default_map_path("/abs/gt_v2", 2) == "gt_v2.relabel-2.json"


def test_cli_requires_a_destination():
    with pytest.raises(SystemExit):
        cli._parse_args(["relabel", "vol"])


def test_cli_runs_and_reports(tmp_path, capsys):
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    assert cli.cmd_relabel(cli._parse_args(
        ["relabel", src, "--out", out, "--map", str(tmp_path / "m.json")])) == 0
    text = capsys.readouterr().out
    assert "2 region(s)" in text
    assert "2 id(s) were shared" in text
    assert "still hold the OLD ids" in text, "the stale pyramid must be called out"
    assert "downsample" in text


def test_cli_dry_run_says_so_and_writes_nothing(tmp_path, capsys):
    src = _volume(tmp_path, "v", _colliding())
    out = str(tmp_path / "out")
    cli.cmd_relabel(cli._parse_args(
        ["relabel", src, "--out", out, "--dry-run",
         "--map", str(tmp_path / "m.json")]))
    assert "DRY RUN" in capsys.readouterr().out
    assert not (tmp_path / "m.json").exists()


def test_cli_is_wired():
    assert cli._parse_args(["relabel", "v", "--in-place"]).func is cli.cmd_relabel
