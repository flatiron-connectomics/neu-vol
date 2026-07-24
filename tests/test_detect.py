import os

import numpy as np

from em_volume_tools import convert
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.introspect import detect_backend
from em_volume_tools.profiles import precomputed_create_spec, zarr3_create_spec


def _zarr3(path, data):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _zarr2(path, data):
    import tensorstore as ts
    s = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": path},
                 "metadata": {"shape": list(data.shape), "chunks": [8, 8, 8],
                              "dtype": "<u2", "compressor": None, "order": "C",
                              "fill_value": 0}, "create": True, "delete_existing": True}).result()
    s[...] = data
    return path


def test_detect_zarr3(tmp_path):
    p = str(tmp_path / "a.zarr")
    _zarr3(p, np.zeros((8, 8, 8), np.uint16))
    assert detect_backend(p) == "zarr3"


def test_detect_precomputed(tmp_path):
    p = str(tmp_path / "pc")
    TensorStoreBackend.create(
        precomputed_create_spec("s3-neuroglancer", p, (8, 8, 8), "uint8",
                                resolution_zyx=[8, 8, 8], scale_index=0, chunk=(8, 8, 8)),
        delete_existing=True)
    assert detect_backend(p) == "neuroglancer_precomputed"


def test_detect_zarr2_and_convert(tmp_path):
    vol = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    p = str(tmp_path / "v2.zarr")
    _zarr2(p, vol)
    assert detect_backend(p) == "zarr2"

    # convert autodetects zarr2 and reads it (voxel_size supplied; no OME metadata)
    dst = str(tmp_path / "out.zarr")
    convert(p, dst, voxel_size=(8, 8, 8), chunk=(8, 8, 8), multiscale=False, delete_existing=True)
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(out.read_region((slice(0, 8),) * 3), vol)


def test_detect_none_for_empty_dir(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(d)
    assert detect_backend(d) is None
