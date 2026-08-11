"""Copying a volume, or a box out of it: ``convert``'s crop and the ``copy`` subcommand.

Two failure modes are worth pinning here, because both are silent and both survive a
green suite otherwise:

* a cropped copy that lands at the origin instead of over the source, which looks fine
  on its own and mismatches the moment it is loaded beside the original;
* ``copy`` losing the source's ``segmentation`` type, which turns the pyramid's reducer
  from a mode into a mean and invents label ids that were never in the data.
"""

import json
import os

import numpy as np
import pytest

from em_volume_tools import cli, convert
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec


def _zarr_source(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return path


def _precomputed_source(path, data, *, voxel=(8, 8, 8), chunk=(8, 8, 8),
                        kind="segmentation"):
    """A real single-scale precomputed volume, built through the op the CLI drives."""
    src = _zarr_source(path + ".src.zarr", data, chunk=chunk)
    convert(src, path, voxel_size=voxel, kind=kind, profile="local-neuroglancer",
            chunk=chunk, multiscale=False, delete_existing=True)
    return path


def _read(volume, level=0, backend="neuroglancer_precomputed"):
    spec = ({"backend": backend, "path": volume, "scale_index": level}
            if backend == "neuroglancer_precomputed"
            else {"backend": backend, "path": os.path.join(volume, str(level))})
    be = open_backend(spec)
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.fixture
def labels():
    """Distinct ids per 4-voxel block, so a mis-cropped read is visible."""
    z, y, x = np.indices((16, 16, 16))
    return (1 + (z // 4) * 16 + (y // 4) * 4 + (x // 4)).astype(np.uint32)


# --------------------------------------------------------------------------- #
# convert's crop
# --------------------------------------------------------------------------- #
def test_crop_copies_the_box_and_keeps_the_source_frame(tmp_path, labels):
    """The copied voxels are the box, and the output sits over it, not at the origin.

    `voxel_offset` is what puts it there. Without the offset shift the crop renders
    16 voxels away from the data it was cut from, which no single-volume check catches.
    """
    src = _precomputed_source(str(tmp_path / "src"), labels)
    dst = str(tmp_path / "crop")
    convert(src, dst, crop_start=(2, 4, 6), crop_stop=(10, 12, 14), kind="segmentation",
            profile="local-neuroglancer", chunk=(8, 8, 8), multiscale=False,
            delete_existing=True)

    np.testing.assert_array_equal(_read(dst), labels[2:10, 4:12, 6:14])
    scale = json.loads((tmp_path / "crop" / "info").read_text())["scales"][0]
    assert scale["voxel_offset"] == [6, 4, 2], \
        "voxel_offset is xyz and must carry the crop origin, or the copy renders " \
        "somewhere the source is not"
    assert scale["size"] == [8, 8, 8]


def test_either_crop_bound_may_be_left_open(tmp_path, labels):
    src = _zarr_source(str(tmp_path / "src.zarr"), labels)
    dst = str(tmp_path / "tail.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), crop_start=(8, 0, 0), profile="local",
            chunk=(8, 8, 8), multiscale=False, delete_existing=True)
    np.testing.assert_array_equal(_read(dst, backend="zarr3"), labels[8:])


def test_a_crop_past_the_edge_is_clipped_not_padded(tmp_path, labels):
    """`convert` trims; only `extract_roi` pads.

    Padding a copy would publish fabricated voxels at the margin, indistinguishable
    from real data once written.
    """
    src = _zarr_source(str(tmp_path / "src.zarr"), labels)
    dst = str(tmp_path / "clip.zarr")
    summary = convert(src, dst, voxel_size=(8, 8, 8), crop_start=(-4, 12, 12),
                      crop_stop=(4, 99, 99), profile="local", chunk=(4, 4, 4),
                      multiscale=False, delete_existing=True)
    assert summary["level_shapes"] == [(4, 4, 4)]
    np.testing.assert_array_equal(_read(dst, backend="zarr3"), labels[0:4, 12:, 12:])


def test_a_crop_outside_the_volume_raises(tmp_path, labels):
    src = _zarr_source(str(tmp_path / "src.zarr"), labels)
    with pytest.raises(ValueError, match="empty crop"):
        convert(src, str(tmp_path / "none.zarr"), voxel_size=(8, 8, 8),
                crop_start=(20, 0, 0), crop_stop=(24, 8, 8), profile="local",
                multiscale=False, delete_existing=True)


def test_extract_roi_still_pads_and_now_reads_the_sources_metadata(tmp_path, labels):
    """The op delegates to `convert`, so it must keep its own contract: it pads.

    It also picks up what it never used to read — a source's voxel size — since it no
    longer resolves the source itself.
    """
    from em_volume_tools import extract_roi

    src = _precomputed_source(str(tmp_path / "src"), labels, voxel=(40, 8, 8))
    dst = str(tmp_path / "roi")
    extract_roi(src, dst, start=(-2, 0, 0), stop=(2, 8, 8), pad_value=7,
                kind="segmentation", profile="local-neuroglancer", chunk=(4, 4, 4),
                delete_existing=True)
    out = _read(dst)
    assert out.shape == (4, 8, 8)
    assert np.all(out[0:2] == 7)                                   # padded margin
    np.testing.assert_array_equal(out[2:4], labels[0:2, 0:8, 0:8])
    scales = json.loads((tmp_path / "roi" / "info").read_text())["scales"]
    assert scales[0]["resolution"] == [8, 8, 40], \
        "voxel size should come from the source now that extract_roi delegates"


def test_convert_crop_bbox_reaches_the_op(tmp_path, labels):
    """`--crop-bbox` on convert, end to end through the parser."""
    src = _zarr_source(str(tmp_path / "src.zarr"), labels)
    dst = str(tmp_path / "conv")
    assert cli.main(["convert", "--src", src, "--dst", dst, "--serial",
                     "--format", "zarr", "--voxel-size", "8,8,8", "--single-level",
                     "--chunk", "4,4,4", "--crop-bbox", "4,4,4,12,12,12"]) == 0
    np.testing.assert_array_equal(_read(dst, backend="zarr3"),
                                  labels[4:12, 4:12, 4:12])


# --------------------------------------------------------------------------- #
# the copy subcommand: the source's parameters are the defaults
# --------------------------------------------------------------------------- #
def test_copy_takes_the_segmentation_type_from_the_source(tmp_path, labels):
    """The reason `copy` exists rather than being a documented use of `convert`.

    `convert`'s --kind defaults to image, so the same command line through it averages
    label ids into ids that were never in the data — while the source's own `info` said
    `"type": "segmentation"` the whole time.
    """
    src = _precomputed_source(str(tmp_path / "src"), labels)
    dst = str(tmp_path / "copy")
    assert cli.main(["copy", "--src", src, "--dst", dst, "--serial",
                     "--min-dim", "8"]) == 0

    info = json.loads((tmp_path / "copy" / "info").read_text())
    assert info["type"] == "segmentation"
    assert info["num_scales" if "num_scales" in info else "scales"], "levels exist"
    # the mode reducer keeps ids; a mean would invent them
    coarse = _read(dst, level=1)
    assert set(np.unique(coarse)).issubset(set(np.unique(labels)))


def test_copy_keeps_the_sources_format_and_chunking(tmp_path, labels):
    src = _zarr_source(str(tmp_path / "src.zarr"), labels, chunk=(4, 8, 8))
    # a bare array records no coordinate metadata, so state what it cannot
    dst = str(tmp_path / "copy_z")
    assert cli.main(["copy", "--src", src, "--dst", dst, "--serial", "--min-dim", "8",
                     "--voxel-size", "8,8,8", "--kind", "segmentation"]) == 0
    meta = json.loads((tmp_path / "copy_z" / "0" / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [4, 8, 8], \
        "the source's own chunking is the default, not convert's 128^3"
    np.testing.assert_array_equal(_read(dst, backend="zarr3"), labels)


def test_copy_refuses_to_guess_the_kind(tmp_path, labels):
    """A bare zarr array records no type. Guessing it is the destructive option."""
    src = _zarr_source(str(tmp_path / "src.zarr"), labels)
    with pytest.raises(SystemExit, match="records no image/segmentation type"):
        cli.main(["copy", "--src", src, "--dst", str(tmp_path / "x"), "--serial",
                  "--voxel-size", "8,8,8"])


def test_copy_of_a_box_is_the_box(tmp_path, labels, capsys):
    src = _precomputed_source(str(tmp_path / "src"), labels)
    dst = str(tmp_path / "boxed")
    assert cli.main(["copy", "--src", src, "--dst", dst, "--serial", "--single-level",
                     "--crop-bbox", "2,4,6,10,12,14"]) == 0
    np.testing.assert_array_equal(_read(dst), labels[2:10, 4:12, 6:14])
    assert "z 2:10" in capsys.readouterr().out


def test_copy_dry_run_writes_nothing(tmp_path, labels, capsys):
    src = _precomputed_source(str(tmp_path / "src"), labels)
    dst = tmp_path / "planned"
    assert cli.main(["copy", "--src", src, "--dst", str(dst), "--serial",
                     "--crop-bbox", "0,0,0,8,8,8", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "segmentation (from the source)" in out and "nothing written" in out
    assert not dst.exists()


def test_copy_needs_a_volume_that_describes_itself(tmp_path):
    (tmp_path / "stack").mkdir()
    with pytest.raises(SystemExit, match="no volume found"):
        cli.main(["copy", "--src", str(tmp_path / "stack"), "--dst",
                  str(tmp_path / "x"), "--serial"])


# --------------------------------------------------------------------------- #
# --crop-scale: real per-level voxel sizes, never 2**N
# --------------------------------------------------------------------------- #
def test_crop_scale_uses_the_real_per_level_voxel_sizes(tmp_path):
    """An anisotropic pyramid is where 2**N is wrong, and it is the common case.

    With factors (1,2,2) a scale-1 voxel is 1 level-0 voxel in z and 2 in y/x, so the
    same six numbers name a box of a different *shape*, not just a different size.
    """
    data = np.arange(16 * 16 * 16, dtype=np.uint16).reshape(16, 16, 16)
    src = str(tmp_path / "aniso")
    _precomputed_source(src + ".s", data, voxel=(40, 8, 8), kind="image")
    convert(src + ".s", src, voxel_size=(40, 8, 8), kind="image",
            profile="local-neuroglancer", chunk=(8, 8, 8), factors=[(1, 2, 2)],
            max_levels=2, min_dim=4, delete_existing=True)

    from em_volume_tools.source_metadata import read_level_voxel_sizes
    assert read_level_voxel_sizes({"backend": "neuroglancer_precomputed", "path": src}) \
        == [(40.0, 8.0, 8.0), (40.0, 16.0, 16.0)]
    assert cli._level0_factor(src, 1) == (1, 2, 2)

    args = cli._parse_args(["copy", "--src", src, "--dst", "/x",
                            "--crop-bbox", "2,1,1,6,4,4", "--crop-scale", "1"])
    assert cli._crop_bbox(args) == ((2, 2, 2), (6, 8, 8))


def test_crop_scale_beyond_the_pyramid_is_an_error(tmp_path, labels):
    src = _precomputed_source(str(tmp_path / "src"), labels)
    args = cli._parse_args(["copy", "--src", src, "--dst", "/x",
                            "--crop-bbox", "0,0,0,4,4,4", "--crop-scale", "5"])
    with pytest.raises(SystemExit, match="records only 1 level"):
        cli._crop_bbox(args)


def test_crop_scale_without_a_box_is_rejected():
    with pytest.raises(SystemExit):
        cli._parse_args(["copy", "--src", "a", "--dst", "b", "--crop-scale", "2"])


def test_an_unaligned_crop_origin_warns_about_the_coarse_levels(tmp_path, caplog):
    """Level 0 is exact; the levels above it are not, and nothing else says so."""
    args = cli._parse_args(["copy", "--src", "a", "--dst", "b",
                            "--crop-bbox", "3,0,0,131,128,128", "--min-dim", "8"])
    with caplog.at_level("WARNING"):
        cli._warn_if_crop_unaligned((3, 0, 0), (128, 128, 128), (8, 8, 8), args)
    assert "not a multiple" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        cli._warn_if_crop_unaligned((16, 0, 0), (128, 128, 128), (8, 8, 8), args)
    assert not caplog.text
