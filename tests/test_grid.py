"""Resolving WHICH grid a volume implies, and `neu-vol align-bbox` over it.

The arithmetic itself lives in `neu_lib.grid` and is tested there. What this file
covers is the part that needs a volume to answer: on a **sharded** level the write unit
is the *shard*, and aligning to the inner read chunk looks like alignment while
protecting against nothing, because the object a partial write rewrites is the shard.
The pyramid's cumulative factor and the source/destination LCM likewise come from real
per-level metadata, never from `2 ** level`.
"""

import numpy as np
import pytest
from neu_lib import grid

from neu_vol import cli, convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec


# --------------------------------------------------------------------------- #
# the volume-end exemption, in one place
# --------------------------------------------------------------------------- #

def test_write_reports_alignment_through_the_shared_predicate():
    """`neu-vol write` and `align-bbox` must not drift apart on the same box."""
    from neu_vol.ops.write import _misaligned_axes

    shape, chunk = (100, 128, 256), (128, 128, 128)
    for start, stop in [((0, 0, 0), (100, 128, 128)),
                        ((0, 0, 8), (100, 128, 136)),
                        ((0, 64, 0), (100, 128, 256)),
                        ((8, 0, 0), (16, 128, 128))]:
        assert _misaligned_axes(start, stop, shape, chunk) == \
            grid.misaligned_axes(start, stop, chunk, extent=shape)


# --------------------------------------------------------------------------- #
# the CLI: resolving WHICH grid
# --------------------------------------------------------------------------- #
def _source(path, shape=(32, 64, 64), chunk=(8, 8, 8)):
    data = np.zeros(shape, dtype=np.uint8)
    data[:4, :8, :8] = 1
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return path


@pytest.fixture
def aniso(tmp_path):
    """A (1,2,2) pyramid: the anisotropy is what makes 2**N wrong."""
    src = _source(str(tmp_path / "src.zarr"))
    dst = str(tmp_path / "aniso.zarr")
    convert(src, dst, voxel_size=(40, 8, 8), profile="local", chunk=(8, 16, 16),
            factors=[(1, 2, 2), (1, 2, 2)], max_levels=3, min_dim=4,
            delete_existing=True)
    return dst


@pytest.fixture
def sharded(tmp_path):
    """A sharded level: write unit 32^3, inner read chunk 8^3."""
    from neu_vol import StorageProfile

    src = _source(str(tmp_path / "s_src.zarr"))
    dst = str(tmp_path / "sharded.zarr")
    convert(src, dst, voxel_size=(8, 8, 8),
            profile=StorageProfile("zarr3", chunk=(8, 8, 8), shard=(32, 32, 32)),
            multiscale=False, delete_existing=True)
    return dst


def _grid_line(capsys):
    return next(ln for ln in capsys.readouterr().err.splitlines() if "grid " in ln)


def test_the_write_unit_of_a_sharded_level_is_the_shard(sharded, capsys):
    """Aligning to the inner chunk here would be alignment that protects nothing."""
    assert cli.main(["align-bbox", "--volume", sharded, "--bbox", "3,3,3,20,20,20"]) == 0
    line = _grid_line(capsys)
    assert "32x32x32" in line and "shard" in line

    assert cli.main(["align-bbox", "--volume", sharded, "--bbox", "3,3,3,20,20,20",
                     "--to", "read-chunk"]) == 0
    assert "8x8x8" in _grid_line(capsys)


def test_the_pyramid_grid_comes_from_real_per_level_voxel_sizes(aniso, capsys):
    """Two (1,2,2) levels means the deepest is 4x in y/x and 1x in z, not 4x in all."""
    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,3,3,20,20,20",
                     "--to", "pyramid"]) == 0
    assert "1x4x4" in _grid_line(capsys)


def test_both_is_the_lcm_of_the_two_grids(aniso, capsys):
    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,3,3,20,20,20",
                     "--to", "both"]) == 0
    line = _grid_line(capsys)
    assert "8x16x16" in line and "LCM" in line          # lcm((8,16,16), (1,4,4))


def test_a_coarse_levels_chunk_is_converted_to_level0_voxels(aniso, capsys):
    """A 16-wide chunk at a level coarsened 4x in y/x is a 64-voxel grid there."""
    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,3,3,20,20,20",
                     "--level", "2"]) == 0
    assert "8x64x64" in _grid_line(capsys)


def test_the_box_may_be_given_at_a_scale_and_is_reported_back_there(aniso, capsys):
    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,1,1,20,8,8",
                     "--scale", "2", "-q"]) == 0
    out = capsys.readouterr().out.strip()
    lo = tuple(int(v) for v in out.split(",")[:3])
    assert lo == (0, 0, 0), "scale-2 y/x of 1 is level-0 4, floored to the 16-grid"

    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,1,1,20,8,8",
                     "--scale", "2"]) == 0
    assert "at scale 2" in capsys.readouterr().err


def test_a_grid_may_be_given_outright_with_no_volume(capsys):
    assert cli.main(["align-bbox", "--block", "64,64,64",
                     "--bbox", "10,10,10,100,100,100", "-q"]) == 0
    assert capsys.readouterr().out.strip() == "0,0,0,128,128,128"


def test_without_a_volume_or_a_block_it_says_what_is_missing():
    with pytest.raises(SystemExit, match="--volume .*or --block"):
        cli.main(["align-bbox", "--bbox", "0,0,0,8,8,8"])


def test_the_box_is_clamped_to_the_volume_and_that_edge_is_fine(aniso, capsys):
    """Growing to align can leave the volume; the trim is reported, not the misalignment.

    A 64-voxel grid over a (32, 64, 64) volume is the case: aligning outward runs past
    the z extent, and the clamped edge then sits mid-block *because the volume's own
    final block is partial there*. Reporting that as misaligned would be wrong.
    """
    assert cli.main(["align-bbox", "--volume", aniso, "--block", "64,64,64",
                     "--bbox", "0,0,0,20,20,20"]) == 0
    err = capsys.readouterr().err
    assert "clamped" in err and "aligned by definition" in err
    assert "NOTE" not in err, "the volume edge must not be reported as misaligned"


def test_quiet_prints_one_line_per_box_and_nothing_else(aniso, capsys):
    assert cli.main(["align-bbox", "--volume", aniso, "-q",
                     "--bbox", "3,3,3,20,20,20", "--bbox", "1,1,1,9,9,9"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["0,0,0,24,32,32", "0,0,0,16,16,16"]
    assert err == ""


def test_origin_mode_says_the_far_edge_is_still_off_the_grid(aniso, capsys):
    assert cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,3,3,20,20,20",
                     "--mode", "origin"]) == 0
    err = capsys.readouterr().err
    assert "NOTE" in err and "preserves the extent" in err


def test_an_inner_box_that_collapses_exits_cleanly(aniso):
    with pytest.raises(SystemExit, match="empty"):
        cli.main(["align-bbox", "--volume", aniso, "--bbox", "3,3,3,5,5,5",
                  "--mode", "inner"])


def test_the_aligned_box_silences_the_crop_warning(aniso, caplog, capsys):
    """The point of the command: it answers the complaint `copy --crop-bbox` makes."""
    assert cli.main(["align-bbox", "--volume", aniso, "-q", "--to", "both",
                     "--bbox", "3,3,3,20,20,20"]) == 0
    box = capsys.readouterr().out.strip()
    args = cli._parse_args(["copy", "--src", aniso, "--dst", "/x",
                            "--crop-bbox", box, "--min-dim", "4",
                            "--factors", "1,2,2;1,2,2"])
    start, stop = cli._crop_bbox(args)
    with caplog.at_level("WARNING"):
        cli._warn_if_crop_unaligned(start, tuple(b - a for a, b in zip(start, stop)),
                                    (40, 8, 8), args)
    assert not caplog.text


