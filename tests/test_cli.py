"""The em-vol CLI: argument parsing, volume inspection, and the info/downsample plan.

The inspection tests matter most. `info` and `downsample` both have to agree with what
is actually on disk, and both used to be separate scripts that could drift.
"""

import json
import os

import numpy as np
import pytest

from em_volume_tools import cli, convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.source_metadata import read_level_voxel_sizes
from em_volume_tools.profiles import zarr3_create_spec


def _pyramid(tmp_path, *, voxel=(8, 8, 8), shape=(16, 16, 16), kind="image"):
    """A small real multiscale zarr, built by the same op the CLI drives."""
    src = str(tmp_path / "src.zarr")
    data = np.random.default_rng(0).integers(0, 5, shape, dtype=np.uint8)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "out.zarr")
    convert(src, dst, voxel_size=voxel, kind=kind, profile="local",
            chunk=(8, 8, 8), min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# argument plumbing
# --------------------------------------------------------------------------- #
def test_every_subcommand_parses():
    for argv, expected in [
        (["info", "/v"], cli.cmd_info),
        (["convert", "--src", "a", "--dst", "b"], cli.cmd_convert),
        (["downsample", "/v", "--start-level", "2"], cli.cmd_downsample),
        (["progress", "/v"], cli.cmd_progress),
        (["create", "/v", "--like", "/r"], cli.cmd_create),
        (["write", "/v", "--src", "a.h5", "--offset", "0,0,0"], cli.cmd_write),
    ]:
        assert cli._parse_args(argv).func is expected


def test_a_local_precomputed_target_has_a_profile_to_use(tmp_path):
    """`convert --format precomputed` to a local path asks for 'local-neuroglancer'.

    That profile was referenced by the CLI but never registered, so the whole path
    failed with "unknown profile" — and it survived because every test that wanted
    precomputed passed 's3-neuroglancer' explicitly, local destination or not.
    """
    from em_volume_tools.profiles import get_profile

    assert get_profile("local-neuroglancer").format == "neuroglancer_precomputed"
    src, dst = _pyramid(tmp_path), str(tmp_path / "pc_out")
    assert cli.cmd_convert(cli._parse_args(
        ["convert", "--src", src, "--dst", dst, "--format", "precomputed",
         "--serial", "--chunk", "8,8,8", "--min-dim", "8"])) == 0
    assert cli.describe(dst)["format"] == "neuroglancer_precomputed"


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_config_defaults_to_bundled_local_and_is_repeatable():
    """--config is repeatable so an overlay carries only what differs."""
    assert cli._configs(cli._parse_args(["convert", "--src", "a", "--dst", "b"])) \
        == ["dask-local"]
    args = cli._parse_args(["convert", "--src", "a", "--dst", "b",
                            "--config", "dask-slurm-example", "--config", "mine.yaml"])
    assert cli._configs(args) == ["dask-slurm-example", "mine.yaml"]


@pytest.mark.parametrize("text,expected", [
    ("1,2,3", (1, 2, 3)),
    ("1x2x3", (1, 2, 3)),
    (None, None),
])
def test_triple_parsing(text, expected):
    assert cli._triple(text, "chunk") == expected


def test_triple_rejects_the_wrong_arity():
    """Silently accepting 2 values would mis-shape every block."""
    with pytest.raises(SystemExit, match="3 comma-separated"):
        cli._triple("1,2", "chunk")


def test_factor_list_parsing():
    assert cli._factor_list("1,2,2;1,2,2") == [(1, 2, 2), (1, 2, 2)]
    assert cli._factor_list(None) is None


# --------------------------------------------------------------------------- #
# per-level voxel sizes — the thing that must not be derived
# --------------------------------------------------------------------------- #
def test_level_voxel_sizes_are_read_not_derived(tmp_path):
    """Shape ratios are inexact, so voxel sizes must come from the metadata.

    Level extents are ceil-divided: a level-0 extent of 13750 over a factor of 4
    stores 3438, and 13750/3438 is 3.9994 — deriving from that reports 31.9953 nm for
    a level that is exactly 32. This fixture reproduces the condition with a
    deliberately indivisible extent.
    """
    dst = _pyramid(tmp_path, shape=(17, 17, 17), voxel=(8.0, 8.0, 8.0))
    sizes = read_level_voxel_sizes({"backend": "zarr3", "path": dst})
    assert sizes and sizes[0] == (8.0, 8.0, 8.0)
    assert sizes[1] == (16.0, 16.0, 16.0), f"derived rather than read: {sizes[1]}"

    # ...and the shape ratio really is inexact here, which is what makes this a trap
    levels = cli.existing_levels(dst, "zarr3")
    ratio = levels[0]["shape"][2] / levels[1]["shape"][2]
    assert ratio != 2.0, "fixture no longer exercises the ceil-division case"


def test_level_voxel_sizes_none_when_unrecorded(tmp_path):
    """A bare zarr array has no OME metadata; report that rather than inventing."""
    bare = str(tmp_path / "bare.zarr")
    data = np.zeros((8, 8, 8), np.uint8)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", bare, data.shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    assert read_level_voxel_sizes({"backend": "zarr3", "path": bare}) is None


# --------------------------------------------------------------------------- #
# inspection shared by info and downsample
# --------------------------------------------------------------------------- #
def test_describe_reports_format_kind_and_levels(tmp_path):
    dst = _pyramid(tmp_path, kind="segmentation")
    d = cli.describe(dst)
    assert d["format"] == "zarr3"
    assert d["meta"]["kind"] == "segmentation"
    assert d["shape"] == (16, 16, 16)
    assert sorted(d["levels"]) == [0, 1]


def test_existing_levels_stops_at_the_first_gap(tmp_path):
    """Probing upward is what makes this work on an in-flight conversion.

    The multiscale group metadata is only written at the very end, so a running
    conversion has levels but no group metadata to enumerate them from.
    """
    dst = _pyramid(tmp_path)
    assert sorted(cli.existing_levels(dst, "zarr3")) == [0, 1]
    assert cli.existing_levels(str(tmp_path / "nope.zarr"), "zarr3") == {}


def test_describe_rejects_a_path_with_no_volume(tmp_path):
    """FileNotFoundError, not SystemExit: `describe` is a library call now.

    The ops use it too (`create --like`, `write`), and a library function that kills
    the interpreter is not usable from a notebook. The CLI turns it back into an exit.
    """
    with pytest.raises(FileNotFoundError, match="no volume found"):
        cli.describe(str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match="no volume found"):
        cli.cmd_info(cli._parse_args(["info", str(tmp_path / "empty")]))


def test_levels_report_their_own_chunking(tmp_path):
    """Chunking is per level, in each level's array metadata — not a volume property.

    A conversion is free to chunk levels differently, so it is read per level rather
    than assumed from level 0.
    """
    dst = _pyramid(tmp_path)
    levels = cli.existing_levels(dst, "zarr3")
    for i, lv in levels.items():
        assert lv["chunks"] == (8, 8, 8), f"level {i}: {lv['chunks']}"
        # unsharded, so the read unit is the write unit
        assert lv["read_chunks"] == lv["chunks"]


def _sharded_pyramid(path):
    """A real sharded volume — built through `convert` so it has group metadata.

    A hand-made level-0 array is not a volume: `detect_backend` finds nothing without
    the OME group, so `info` reports "no volume found".
    """
    path = str(path)
    src = path + ".src"
    data = np.random.default_rng(1).integers(0, 5, (32, 32, 32), dtype=np.uint8)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    convert(src, path, voxel_size=(8, 8, 8), kind="image", profile="local",
            chunk=(8, 8, 8), shard=(16, 16, 16), min_dim=16, delete_existing=True)
    return path


def test_sharded_levels_distinguish_shard_from_chunk(tmp_path):
    """With sharding the write chunk is the SHARD and the read chunk the fetch unit.

    Reporting only one of them would misstate read amplification — the number that
    actually governs how much gets pulled to satisfy a small read.
    """
    lv = cli.existing_levels(_sharded_pyramid(tmp_path / "sh.zarr"), "zarr3")[0]
    assert lv["chunks"] == (16, 16, 16), "write chunk should be the shard"
    assert lv["read_chunks"] == (8, 8, 8), "read chunk should be the inner chunk"


def test_info_runs_against_a_real_pyramid(tmp_path, capsys):
    dst = _pyramid(tmp_path)
    assert cli.cmd_info(cli._parse_args(["info", dst])) == 0
    out = capsys.readouterr().out
    assert "zarr3" in out and "8x8x8" in out and "16x16x16" in out
    assert "chunk" in out, "chunking is what you need to size a read; show it"


def _header(out):
    """The column header line. Asserting on the whole output is a trap: pytest names
    tmp_path after the test, so a test with 'shard' in its name puts that word in the
    printed volume path."""
    return next(l for l in out.splitlines() if l.strip().startswith("level"))


def test_info_only_adds_a_shard_column_when_something_is_sharded(tmp_path, capsys):
    """An unsharded volume should not carry a column of dashes."""
    dst = _pyramid(tmp_path)
    cli.cmd_info(cli._parse_args(["info", dst]))
    unsharded_header = _header(capsys.readouterr().out)
    assert "chunk" in unsharded_header and "shard" not in unsharded_header

    path = _sharded_pyramid(tmp_path / "s.zarr")
    cli.cmd_info(cli._parse_args(["info", path]))
    out = capsys.readouterr().out
    assert "shard" in _header(out)
    assert "16x16x16" in out and "8x8x8" in out


# --------------------------------------------------------------------------- #
# downsample's plan — it must refuse before writing, not after
# --------------------------------------------------------------------------- #
def test_downsample_dry_run_touches_nothing(tmp_path):
    dst = _pyramid(tmp_path)
    before = sorted(os.listdir(dst))
    args = cli._parse_args(["downsample", dst, "--start-level", "0",
                            "--min-dim", "8", "--dry-run"])
    assert cli.cmd_downsample(args) == 0
    assert sorted(os.listdir(dst)) == before


def test_downsample_refuses_a_start_level_beyond_the_schedule(tmp_path):
    """min-dim 8 on a 16^3 volume schedules exactly one level, so 2 is out of range."""
    dst = _pyramid(tmp_path)
    args = cli._parse_args(["downsample", dst, "--start-level", "2",
                            "--min-dim", "8", "--dry-run"])
    with pytest.raises(SystemExit, match="exceeds the deepest level"):
        cli.cmd_downsample(args)


def test_downsample_refuses_to_seed_from_a_level_that_is_not_on_disk(tmp_path):
    """min-dim 4 schedules levels 0..2, but the volume only has 0..1.

    A different branch from the one above: the level is within the schedule, it just
    was never written. Seeding from it would read nothing.
    """
    dst = _pyramid(tmp_path)
    args = cli._parse_args(["downsample", dst, "--start-level", "2",
                            "--min-dim", "4", "--dry-run"])
    with pytest.raises(SystemExit, match="does not exist; nothing to seed from"):
        cli.cmd_downsample(args)


def test_downsample_refuses_when_disk_disagrees_with_the_schedule(tmp_path):
    """--min-dim/--max-levels/--factors differing from the original conversion
    would rebuild a *different* pyramid, leaving it internally inconsistent.

    Caught by comparing the computed schedule against the shapes on disk, before
    anything is written — the failure is otherwise silent and destructive.
    """
    dst = _pyramid(tmp_path)
    args = cli._parse_args(["downsample", dst, "--start-level", "0",
                            "--factors", "1,4,4", "--min-dim", "1", "--dry-run"])
    with pytest.raises(SystemExit, match="disagrees with the computed schedule"):
        cli.cmd_downsample(args)


# --------------------------------------------------------------------------- #
# progress — the manifest counts TASKS, and a task is not always a chunk
# --------------------------------------------------------------------------- #
def _write_manifest(path, lines):
    with open(path, "w") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    return str(path)


def test_manifest_counts_separates_meta_from_tasks(tmp_path):
    p = _write_manifest(tmp_path / "p.jsonl", [
        {"group": 0, "meta": {"total": 4, "task_shape": [8, 32, 32]}},
        {"group": 0, "key": [0, 0, 0], "status": "written"},
        {"group": 0, "key": [0, 1, 1], "status": "empty"},
        {"group": 1, "key": [0, 0, 0], "status": "written"},
    ])
    m = cli._manifest_counts(p)
    assert m[0]["counts"] == {"written": 1, "empty": 1}
    assert m[0]["total"] == 4 and m[0]["task_shape"] == (8, 32, 32)
    assert m[0]["grid"] == (1, 2, 2)          # max index + 1, per axis
    assert m[1]["total"] is None


def test_manifest_counts_is_none_when_there_is_no_manifest(tmp_path):
    assert cli._manifest_counts(str(tmp_path / "nope.jsonl")) is None


def test_task_total_prefers_the_recorded_total_over_the_chunk_grid():
    """The bug: 7,623 tasks divided by 1,680,206 chunks reported 0.17% for a run
    that was 36% done."""
    lv = {"total": 7623, "task_shape": (128, 2048, 2048), "grid": (121, 7, 9)}
    total, note = cli._task_total(lv, 1680206, (128, 128, 128), 0)
    assert total == 7623
    assert "256 destination chunks each" in note


def test_task_total_says_nothing_when_a_task_is_a_chunk():
    lv = {"total": 213378, "task_shape": (128, 128, 128), "grid": (61, 53, 66)}
    total, note = cli._task_total(lv, 213378, (128, 128, 128), 1)
    assert total == 213378 and note == ""


def test_task_total_infers_the_grid_for_a_manifest_written_before_totals():
    """Runs that finished before the total was recorded must still read correctly —
    the recorded block indices pin the task grid once a level is complete."""
    lv = {"total": None, "task_shape": None, "grid": (121, 7, 9)}
    total, note = cli._task_total(lv, 1680206, (128, 128, 128), 0)
    assert total == 7623
    assert "inferred" in note and "upper bound" in note


def test_task_total_falls_back_to_the_chunk_grid():
    lv = {"total": None, "task_shape": None, "grid": (2, 2, 2)}
    assert cli._task_total(lv, 8, (8, 8, 8), 0) == (8, "")


def test_progress_reports_100pct_when_tasks_are_coarser_than_chunks(tmp_path, capsys):
    """End to end: a source chunked coarser than the destination.

    16 destination chunks at level 0, covered by one task. Before the fix this
    printed 1/16 = 6% for a finished conversion.
    """
    vol = np.random.default_rng(3).integers(0, 200, (8, 64, 64), dtype=np.uint8)
    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, vol.shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 64, 64)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)

    dst = str(tmp_path / "out.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="image", profile="local",
            chunk=(8, 16, 16), min_dim=64, delete_existing=True)

    assert cli.cmd_progress(cli._parse_args(["progress", dst])) == 0
    out = capsys.readouterr().out
    level0 = next(l for l in out.splitlines() if l.split()[:1] == ["0"])
    assert "1/1" in level0 and "100.0%" in level0
    assert "destination chunks each" in out, "say what the unit is when it isn't a chunk"


def test_progress_says_so_when_the_manifest_is_missing(tmp_path, capsys):
    """Silently falling back to a storage listing looked identical to --storage."""
    dst = _pyramid(tmp_path)
    os.remove(dst + ".progress.jsonl")
    assert cli.cmd_progress(cli._parse_args(["progress", dst])) == 0
    out = capsys.readouterr().out
    assert "no run manifest at" in out and "storage listing" in out


def test_progress_points_at_a_rebuilds_differently_named_manifest(tmp_path, capsys):
    """A rebuild's manifest is named apart from a conversion's on purpose — reusing
    that name would mark the levels being rebuilt as already done. The cost is that
    the default `progress` looks for is then the wrong file, with nothing to say the
    right one is sitting next to it.
    """
    dst = _pyramid(tmp_path)
    os.rename(dst + ".progress.jsonl", dst + ".regen-from-0.progress.jsonl")
    assert cli.cmd_progress(cli._parse_args(["progress", dst])) == 0
    out = capsys.readouterr().out
    assert "no run manifest at" in out
    assert "--progress-path" in out and "regen-from-0" in out


# --------------------------------------------------------------------------- #
# progress — precomputed, whose scales are not directories
# --------------------------------------------------------------------------- #
def _level_row(text, level=0):
    """The printed row for one level. Shapes contain spaces, so index from the left."""
    return next(l for l in text.splitlines() if l.split()[:1] == [str(level)])


def _done_total(row):
    """``(done, total)`` from the one 'a/b' token in a row."""
    done, total = next(t for t in row.split() if "/" in t).split("/")
    return int(done.replace(",", "")), int(total.replace(",", ""))


def _precomputed(tmp_path, *, labels=True):
    """A real precomputed volume, built by `convert`."""
    from em_volume_tools.profiles import StorageProfile

    shape = (32, 32, 32)
    data = np.zeros(shape, np.uint64)
    data[8:24, 8:24, 8:24] = 5                  # sparse, so `empty` blocks are real
    src = str(tmp_path / "pc.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), data)
    dst = str(tmp_path / "pc.precomputed")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            min_dim=8, delete_existing=True,
            profile=StorageProfile("neuroglancer_precomputed", chunk=(8, 8, 8),
                                   compressor="gzip"))
    return dst


def test_progress_counts_a_precomputed_volume(tmp_path, capsys):
    """It used to print "no levels found yet" for every precomputed volume.

    Levels came from opening `<volume>/<level>` with the zarr v3 driver in a loop;
    precomputed keeps its scales under one `info` with keys like `8_8_8`, so the
    level-0 open failed and the loop ended before it began — for the manifest path
    too, since the loop was shared.
    """
    dst = _precomputed(tmp_path)
    assert cli.cmd_progress(cli._parse_args(["progress", dst])) == 0
    out = capsys.readouterr().out
    assert "no levels found" not in out
    assert "run manifest" in out
    level0 = _level_row(out)
    assert "(32, 32, 32)" in level0 and "100.0%" in level0
    assert _done_total(level0) == (64, 64), "every level-0 block was processed"


def test_progress_storage_counts_precomputed_chunk_objects(tmp_path, capsys):
    """--storage has to look under each scale key, not under a level directory."""
    dst = _precomputed(tmp_path)
    assert cli.cmd_progress(cli._parse_args(["progress", dst, "--storage"])) == 0
    out = capsys.readouterr().out
    assert "storage listing" in out and "no levels found" not in out
    stored, total = _done_total(_level_row(out))
    assert total == 64
    assert 0 < stored < 64, f"expected some but not all of the 64 chunks, got {stored}"


def test_progress_storage_and_manifest_disagree_on_sparse_data_by_design(tmp_path, capsys):
    """The gap is information, not a bug: an all-fill block writes no object.

    So on sparse data the manifest runs ahead of storage — which is why `--storage`
    percentages look alarming on a ground-truth volume that is mostly background.
    """
    dst = _precomputed(tmp_path)
    cli.cmd_progress(cli._parse_args(["progress", dst]))
    from_manifest = capsys.readouterr().out
    cli.cmd_progress(cli._parse_args(["progress", dst, "--storage"]))
    from_storage = capsys.readouterr().out

    done_manifest, total = _done_total(_level_row(from_manifest))
    done_storage, total_storage = _done_total(_level_row(from_storage))
    assert total == total_storage, "both count against the same denominator"
    assert done_manifest == total, "the manifest saw every block"
    assert done_storage < done_manifest, "only the non-empty blocks wrote objects"


def test_progress_reports_nothing_at_a_path_with_no_volume(tmp_path, capsys):
    assert cli.cmd_progress(cli._parse_args(["progress", str(tmp_path / "no")])) == 1
    assert "no volume found" in capsys.readouterr().out
