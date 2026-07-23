import os

import numpy as np
import pytest

from em_volume_tools import convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.introspect import read_source_metadata
from em_volume_tools.profiles import zarr3_create_spec


def _make_bare_zarr(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True,
    )
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _scale_trans(summary, level=0):
    ct = summary["attrs"]["ome"]["multiscales"][0]["datasets"][level]["coordinateTransformations"]
    scale = next(t["scale"] for t in ct if t["type"] == "scale")
    trans = next(t["translation"] for t in ct if t["type"] == "translation")
    return scale, trans


def test_convert_reads_voxel_size_and_offset_from_ome_source(tmp_path):
    vol = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    bare = str(tmp_path / "bare.zarr")
    _make_bare_zarr(bare, vol)

    # First produce an OME source with non-trivial voxel_size + offset.
    ome = str(tmp_path / "ome.zarr")
    convert(bare, ome, voxel_size=(8, 8, 8), offset=(0, 16, 32), units="nm",
            multiscale=False, chunk=(8, 8, 8), delete_existing=True)

    # Now convert the OME group WITHOUT specifying metadata -> it must be read back.
    out = str(tmp_path / "out.zarr")
    summary = convert(ome, out, multiscale=False, chunk=(8, 8, 8), delete_existing=True)
    scale, trans = _scale_trans(summary)
    assert scale == [8.0, 8.0, 8.0]
    assert trans == [0.0, 16.0, 32.0]


def test_explicit_args_override_source_metadata(tmp_path):
    vol = np.zeros((8, 8, 8), np.uint16)
    bare = str(tmp_path / "b.zarr")
    _make_bare_zarr(bare, vol)
    ome = str(tmp_path / "o.zarr")
    convert(bare, ome, voxel_size=(8, 8, 8), multiscale=False, chunk=(8, 8, 8), delete_existing=True)

    out = str(tmp_path / "o2.zarr")
    summary = convert(ome, out, voxel_size=(4, 5, 6), multiscale=False,
                      chunk=(8, 8, 8), delete_existing=True)
    scale, _ = _scale_trans(summary)
    assert scale == [4.0, 5.0, 6.0]  # user override wins


def test_convert_reads_metadata_from_precomputed_source(tmp_path):
    vol = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    bare = str(tmp_path / "src.zarr")
    _make_bare_zarr(bare, vol)
    pc = str(tmp_path / "vol.precomputed")
    convert(bare, pc, voxel_size=(8, 8, 8), profile="s3-neuroglancer",
            chunk=(8, 8, 8), multiscale=False, delete_existing=True)

    out = str(tmp_path / "back.zarr")
    summary = convert({"backend": "neuroglancer_precomputed", "path": pc}, out,
                      multiscale=False, chunk=(8, 8, 8), delete_existing=True)
    scale, _ = _scale_trans(summary)
    assert scale == [8.0, 8.0, 8.0]


def test_read_source_metadata_none_for_bare_and_hdf5(tmp_path):
    vol = np.zeros((4, 4, 4), np.uint8)
    bare = str(tmp_path / "bare.zarr")
    _make_bare_zarr(bare, vol)
    assert read_source_metadata({"backend": "zarr3", "path": bare}) is None
    assert read_source_metadata({"backend": "hdf5", "path": "/x", "dataset": "d"}) is None


def test_convert_without_voxel_size_and_no_metadata_raises(tmp_path):
    vol = np.zeros((4, 4, 4), np.uint8)
    bare = str(tmp_path / "bare.zarr")
    _make_bare_zarr(bare, vol)
    with pytest.raises(ValueError, match="voxel_size is required"):
        convert(bare, str(tmp_path / "o.zarr"), multiscale=False, chunk=(4, 4, 4))
