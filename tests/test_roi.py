import os

import numpy as np
import pytest

from em_volume_tools import extract_roi
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec
from em_volume_tools.pyramid import mean_downsample


def _make_zarr(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True,
    )
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _full(be):
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.fixture
def src(tmp_path):
    vol = np.arange(16 * 16 * 16, dtype=np.uint16).reshape(16, 16, 16)
    path = str(tmp_path / "src.zarr")
    _make_zarr(path, vol)
    return path, vol


def test_extract_roi_in_bounds(tmp_path, src):
    path, vol = src
    dst = str(tmp_path / "roi.zarr")
    extract_roi(path, dst, start=(2, 4, 6), stop=(10, 12, 14), voxel_size=(8, 8, 8),
                chunk=(4, 4, 4), delete_existing=True)
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    assert out.shape == (8, 8, 8)
    np.testing.assert_array_equal(_full(out), vol[2:10, 4:12, 6:14])


def test_extract_roi_offset_metadata(tmp_path, src):
    path, vol = src
    dst = str(tmp_path / "roi_off.zarr")
    summary = extract_roi(path, dst, start=(2, 0, 0), stop=(10, 8, 8), voxel_size=(8, 8, 8),
                          chunk=(8, 8, 8), delete_existing=True)
    ds0 = summary["attrs"]["ome"]["multiscales"][0]["datasets"][0]
    # translation reflects the crop origin: start * voxel_size on z
    assert ds0["coordinateTransformations"][1]["translation"] == [16.0, 0.0, 0.0]


def test_extract_roi_with_padding(tmp_path, src):
    path, vol = src
    dst = str(tmp_path / "roi_pad.zarr")
    # start negative and stop beyond bounds -> pad with 99
    extract_roi(path, dst, start=(-2, 0, 0), stop=(2, 8, 8), voxel_size=(8, 8, 8),
                pad_value=99, chunk=(4, 4, 4), delete_existing=True)
    out = _full(open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")}))
    assert out.shape == (4, 8, 8)
    assert np.all(out[0:2] == 99)                       # padded region
    np.testing.assert_array_equal(out[2:4], vol[0:2, 0:8, 0:8])  # real data


def test_extract_roi_multiscale(tmp_path, src):
    path, vol = src
    dst = str(tmp_path / "roi_ms.zarr")
    summary = extract_roi(path, dst, start=(0, 0, 0), stop=(16, 16, 16), voxel_size=(8, 8, 8),
                          chunk=(8, 8, 8), multiscale=True, min_dim=8, delete_existing=True)
    assert summary["num_levels"] == 2
    lvl1 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "1")})
    np.testing.assert_array_equal(_full(lvl1), mean_downsample(vol, (2, 2, 2)))
