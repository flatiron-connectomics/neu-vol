"""The em-vol CLI: argument parsing, volume inspection, and the info/downsample plan.

The inspection tests matter most. `info` and `downsample` both have to agree with what
is actually on disk, and both used to be separate scripts that could drift.
"""

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
    ]:
        assert cli._parse_args(argv).func is expected


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
    ratio = levels[0][2] / levels[1][2]
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
    with pytest.raises(SystemExit, match="no volume found"):
        cli.describe(str(tmp_path / "empty"))


def test_info_runs_against_a_real_pyramid(tmp_path, capsys):
    dst = _pyramid(tmp_path)
    assert cli.cmd_info(cli._parse_args(["info", dst])) == 0
    out = capsys.readouterr().out
    assert "zarr3" in out and "8x8x8" in out and "16x16x16" in out


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
