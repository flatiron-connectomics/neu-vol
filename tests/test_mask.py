"""Excluding a region from a copy: `--mask-bbox`, and the bbox conventions it shares.

A mask holds a region *out* of a copy, so every failure here is a leak: the region ends
up in the output and the run reports success. The three ways that can happen are pinned
below — a box in the wrong axis order, a box at the wrong scale, and a coarse level that
did not inherit the hole — plus the one case the code cannot fix and has to shout about,
a destination that already holds the region.
"""

import json

import numpy as np
import pytest

from neu_vol import cli, convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec


def _source(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return path


def _read(volume, level=0):
    be = open_backend({"backend": "zarr3", "path": f"{volume}/{level}"})
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.fixture
def filled(tmp_path):
    """Every voxel nonzero, so anything that reads back as 0 was masked."""
    data = np.full((16, 32, 32), 7, dtype=np.uint16)
    return _source(str(tmp_path / "src.zarr"), data), data


# --------------------------------------------------------------------------- #
# the mask view
# --------------------------------------------------------------------------- #
def test_the_masked_box_is_blank_and_the_rest_is_copied(tmp_path, filled):
    src, data = filled
    dst = str(tmp_path / "out.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
            multiscale=False, mask_boxes=[[(4, 8, 8), (12, 24, 24)]],
            delete_existing=True)

    out = _read(dst)
    assert out.shape == data.shape
    assert not out[4:12, 8:24, 8:24].any(), "the excluded box must be empty"
    kept = out.copy()
    kept[4:12, 8:24, 8:24] = 7                      # put the hole back
    np.testing.assert_array_equal(kept, data), "everything outside is untouched"


def test_a_mask_that_does_not_align_with_the_chunk_grid_still_blanks_exactly(tmp_path,
                                                                            filled):
    """The blanking is per read, not per chunk, so it does not round to the grid."""
    src, _ = filled
    dst = str(tmp_path / "odd.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
            multiscale=False, mask_boxes=[[(3, 5, 7), (5, 9, 11)]], delete_existing=True)
    out = _read(dst)
    assert not out[3:5, 5:9, 7:11].any()
    assert out[2, 5, 7] and out[3, 4, 7] and out[3, 5, 6], "the neighbours survive"
    assert out[5, 9, 11], "the far corner is exclusive"


def test_several_boxes_may_be_excluded(tmp_path, filled):
    src, _ = filled
    dst = str(tmp_path / "two.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
            multiscale=False, delete_existing=True,
            mask_boxes=[[(0, 0, 0), (4, 4, 4)], [(12, 28, 28), (16, 32, 32)]])
    out = _read(dst)
    assert not out[0:4, 0:4, 0:4].any() and not out[12:, 28:, 28:].any()
    assert out[8, 16, 16]


def test_the_mask_value_may_be_something_other_than_zero(tmp_path, filled):
    src, _ = filled
    dst = str(tmp_path / "val.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
            multiscale=False, mask_boxes=[[(0, 0, 0), (8, 8, 8)]], mask_value=99,
            delete_existing=True)
    assert set(np.unique(_read(dst)[0:8, 0:8, 0:8])) == {99}


def test_every_pyramid_level_inherits_the_hole(tmp_path):
    """The level above is derived from the OUTPUT's level 0, so the hole propagates.

    This is the leak that a post-pass over level 0 alone would produce, and it would only
    be visible by zooming out.
    """
    data = np.full((16, 16, 16), 7, dtype=np.uint16)
    src = _source(str(tmp_path / "p.zarr"), data)
    dst = str(tmp_path / "pyr.zarr")
    summary = convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
                      min_dim=4, mask_boxes=[[(0, 0, 0), (8, 8, 8)]],
                      delete_existing=True)
    assert summary["num_levels"] >= 2
    for level in range(summary["num_levels"]):
        out = _read(dst, level)
        half = tuple(slice(0, s // 2) for s in out.shape)
        assert not out[half].any(), f"level {level} kept the excluded region"


def test_masking_composes_with_cropping_in_source_coordinates(tmp_path, filled):
    """A mask means the same box whether or not a crop is also given.

    If the mask were resolved against the crop's frame instead, adding --crop-bbox would
    silently move the hole — and the excluded region would come back.
    """
    src, _ = filled
    dst = str(tmp_path / "both.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(4, 4, 4),
            multiscale=False, crop_start=(4, 8, 8), crop_stop=(12, 24, 24),
            mask_boxes=[[(4, 8, 8), (8, 16, 16)]], delete_existing=True)
    out = _read(dst)
    assert out.shape == (8, 16, 16)
    assert not out[0:4, 0:8, 0:8].any(), "the mask lands where the source says"
    assert out[4:, 8:, 8:].all()


def test_a_mask_outside_the_volume_raises_rather_than_doing_nothing(tmp_path, filled):
    """The caller asked for a region to be excluded; copying everything is not an answer.

    A box in the wrong axis order or the wrong scale looks exactly like this, and the
    output of a silent no-op is indistinguishable from a correct copy.
    """
    src, _ = filled
    with pytest.raises(ValueError, match="does not intersect the volume"):
        convert(src, str(tmp_path / "no.zarr"), voxel_size=(8, 8, 8), profile="local",
                multiscale=False, mask_boxes=[[(100, 100, 100), (104, 104, 104)]],
                delete_existing=True)


def test_an_empty_mask_box_raises(tmp_path, filled):
    src, _ = filled
    with pytest.raises(ValueError, match="empty mask box"):
        convert(src, str(tmp_path / "no.zarr"), voxel_size=(8, 8, 8), profile="local",
                multiscale=False, mask_boxes=[[(4, 4, 4), (4, 8, 8)]],
                delete_existing=True)


def test_the_mask_view_is_read_only():
    from neu_vol.backends.view import MaskBackend

    with pytest.raises(TypeError, match="read-only"):
        MaskBackend.write_region(object(), (), None)


# --------------------------------------------------------------------------- #
# through the CLI, with the shared bbox conventions
# --------------------------------------------------------------------------- #
def test_cli_mask_bbox_excludes_the_region(tmp_path, filled, capsys):
    src, _ = filled
    dst = str(tmp_path / "cli.zarr")
    assert cli.main(["convert", "--src", src, "--dst", dst, "--serial", "--format",
                     "zarr", "--voxel-size", "8,8,8", "--single-level", "--chunk",
                     "8,8,8", "--mask-bbox", "4,8,8,12,24,24"]) == 0
    assert not _read(dst)[4:12, 8:24, 8:24].any()


def test_bbox_order_xyz_reverses_each_corner(tmp_path, filled):
    """Numbers copied out of neuroglancer are xyz; read as zyx they mirror the box."""
    src, _ = filled
    args = cli._parse_args(["convert", "--src", src, "--dst", "/x",
                            "--mask-bbox", "8,8,4,24,24,12", "--bbox-order", "xyz"])
    assert cli._mask_bboxes(args) == [((4, 8, 8), (12, 24, 24))]

    args = cli._parse_args(["convert", "--src", src, "--dst", "/x",
                            "--mask-bbox", "4,8,8,12,24,24"])
    assert cli._mask_bboxes(args) == [((4, 8, 8), (12, 24, 24))]


def test_one_scale_flag_covers_every_box_argument(tmp_path):
    """Two scale flags would let you set one and forget the other.

    The mask would then land 2x away in y/x on this pyramid, excluding the wrong region
    while the crop was right — which is why there is one --bbox-scale, not one per box.
    """
    data = np.full((16, 32, 32), 7, dtype=np.uint16)
    src = _source(str(tmp_path / "a.zarr"), data)
    vol = str(tmp_path / "aniso.zarr")
    convert(src, vol, voxel_size=(40, 8, 8), profile="local", chunk=(8, 8, 8),
            factors=[(1, 2, 2)], max_levels=2, min_dim=8, delete_existing=True)

    args = cli._parse_args(["copy", "--src", vol, "--dst", "/x", "--bbox-scale", "1",
                            "--crop-bbox", "0,0,0,8,8,8",
                            "--mask-bbox", "2,2,2,4,4,4"])
    assert cli._crop_bbox(args) == ((0, 0, 0), (8, 16, 16))
    assert cli._mask_bboxes(args) == [((2, 4, 4), (4, 8, 8))]


def test_copy_reports_what_it_will_exclude(tmp_path, filled, capsys):
    src, _ = filled
    assert cli.main(["copy", "--src", src, "--dst", str(tmp_path / "d"), "--serial",
                     "--voxel-size", "8,8,8", "--kind", "image", "--dry-run",
                     "--mask-bbox", "4,8,8,12,24,24"]) == 0
    out = capsys.readouterr().out
    assert "excluded" in out and "z 4:12" in out
    assert "2,048 voxels" in out, "8 x 16 x 16 voxels held out of the copy"


def test_masking_a_destination_that_exists_warns_about_the_elision(tmp_path, filled,
                                                                   caplog):
    """The one leak the code cannot prevent, so it has to be said loudly.

    An all-fill block is elided rather than written, so a mask cannot erase what an
    earlier unmasked run already stored at that key.
    """
    src, _ = filled
    dst = str(tmp_path / "twice.zarr")
    assert cli.main(["convert", "--src", src, "--dst", dst, "--serial", "--format",
                     "zarr", "--voxel-size", "8,8,8", "--single-level",
                     "--chunk", "8,8,8"]) == 0

    with caplog.at_level("WARNING"):
        cli.main(["convert", "--src", src, "--dst", dst, "--serial", "--format", "zarr",
                  "--voxel-size", "8,8,8", "--single-level", "--chunk", "8,8,8",
                  "--mask-bbox", "0,0,0,8,8,8"])
    assert "--fresh" in caplog.text and "ELIDED" in caplog.text
    # and the warning is telling the truth: the region really did survive
    assert _read(dst)[0:8, 0:8, 0:8].any()

    caplog.clear()
    with caplog.at_level("WARNING"):
        cli.main(["convert", "--src", src, "--dst", dst, "--serial", "--format", "zarr",
                  "--voxel-size", "8,8,8", "--single-level", "--chunk", "8,8,8",
                  "--mask-bbox", "0,0,0,8,8,8", "--fresh"])
    assert "ELIDED" not in caplog.text
    assert not _read(dst)[0:8, 0:8, 0:8].any(), "--fresh is the way out"
