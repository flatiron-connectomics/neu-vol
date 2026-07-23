import json
import os

import numpy as np
import pytest

from em_volume_tools import convert
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec
from em_volume_tools.pyramid import mean_downsample


def _make_zarr(path, data, *, has_channels=False, num_channels=1, chunk=(8, 8, 8)):
    axes = ("c", "z", "y", "x") if has_channels else ("z", "y", "x")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          has_channels=has_channels, num_channels=num_channels,
                          dimension_names=axes, chunk=chunk),
        delete_existing=True,
    )
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _full(be):
    return be.read_region(tuple(slice(0, s) for s in be.shape))


def test_convert_zarr_to_precomputed_noncubic(tmp_path):
    # distinct z,y,x sizes catch any axis-transpose mistake
    vol = np.random.default_rng(4).integers(0, 256, (16, 12, 8), dtype=np.uint8)  # (z,y,x)
    src = str(tmp_path / "src.zarr")
    _make_zarr(src, vol, chunk=(8, 8, 8))
    dst = str(tmp_path / "out.precomputed")

    summary = convert(src, dst, voxel_size=(8, 8, 8), profile="s3-neuroglancer",
                      chunk=(8, 8, 8), min_dim=8, kind="image", delete_existing=True)
    assert summary["format"] == "neuroglancer_precomputed"
    assert summary["level_shapes"] == [(16, 12, 8), (8, 6, 4)]

    # read each scale back in canonical (z,y,x) order
    s0 = open_backend({"backend": "neuroglancer_precomputed", "path": dst, "scale_index": 0})
    s1 = open_backend({"backend": "neuroglancer_precomputed", "path": dst, "scale_index": 1})
    assert s0.shape == (16, 12, 8)
    np.testing.assert_array_equal(_full(s0), vol)
    np.testing.assert_array_equal(_full(s1), mean_downsample(vol, (2, 2, 2)))

    # info: two scales, native (x,y,z) sizes and resolutions
    info = json.load(open(os.path.join(dst, "info")))
    assert info["num_channels"] == 1
    sizes = sorted(tuple(s["size"]) for s in info["scales"])
    assert sizes == [(4, 6, 8), (8, 12, 16)]  # (x,y,z) = reversed (z,y,x)
    resolutions = sorted(tuple(s["resolution"]) for s in info["scales"])
    assert resolutions == [(8.0, 8.0, 8.0), (16.0, 16.0, 16.0)]


def test_convert_to_precomputed_multichannel(tmp_path):
    data = np.random.default_rng(5).integers(0, 200, (3, 8, 8, 8), dtype=np.uint8)  # (c,z,y,x)
    src = str(tmp_path / "mc.zarr")
    _make_zarr(src, data, has_channels=True, num_channels=3, chunk=(8, 8, 8))
    dst = str(tmp_path / "mc.precomputed")

    summary = convert(src, dst, voxel_size=(8, 8, 8), profile="s3-neuroglancer",
                      chunk=(8, 8, 8), multiscale=False, kind="image", delete_existing=True)
    assert summary["level_shapes"] == [(3, 8, 8, 8)]

    s0 = open_backend({"backend": "neuroglancer_precomputed", "path": dst, "scale_index": 0})
    assert s0.shape == (3, 8, 8, 8)  # canonical (c,z,y,x)
    np.testing.assert_array_equal(_full(s0), data)
    info = json.load(open(os.path.join(dst, "info")))
    assert info["num_channels"] == 3
