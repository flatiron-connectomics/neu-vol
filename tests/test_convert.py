import os

import numpy as np
import pytest

from em_volume_tools import convert
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec
from em_volume_tools.pyramid import mean_downsample, mode_downsample


def _make_zarr(path, data, *, has_channels=False, num_channels=1, chunk=(8, 8, 8)):
    axes = ("c", "z", "y", "x") if has_channels else ("z", "y", "x")
    ch = (num_channels,) + chunk if has_channels else chunk
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


def test_convert_zarr_to_zarr_segmentation_uses_mode(tmp_path):
    seg = np.random.default_rng(2).integers(0, 5, (16, 16, 16), dtype=np.uint64)
    src = str(tmp_path / "seg_src.zarr")
    _make_zarr(src, seg, chunk=(8, 8, 8))
    dst = str(tmp_path / "seg_out.zarr")

    summary = convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation",
                      profile="local", chunk=(8, 8, 8), min_dim=8, delete_existing=True)
    assert summary["level_shapes"] == [(16, 16, 16), (8, 8, 8)]

    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    lvl1 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "1")})
    np.testing.assert_array_equal(_full(lvl0), seg)
    # label-preserving downsample (mode), never averaged
    np.testing.assert_array_equal(_full(lvl1), mode_downsample(seg, (2, 2, 2)))
    assert _full(lvl1).dtype == np.uint64


def test_convert_multichannel_preserves_channels(tmp_path):
    data = np.random.default_rng(3).integers(0, 200, (2, 16, 16, 16), dtype=np.uint8)
    src = str(tmp_path / "ch_src.zarr")
    _make_zarr(src, data, has_channels=True, num_channels=2, chunk=(8, 8, 8))
    dst = str(tmp_path / "ch_out.zarr")

    summary = convert(src, dst, voxel_size=(8, 8, 8), kind="image",
                      axes=("z", "y", "x"), profile="local", chunk=(8, 8, 8),
                      min_dim=8, delete_existing=True)
    # channel axis not downsampled
    assert summary["level_shapes"] == [(2, 16, 16, 16), (2, 8, 8, 8)]

    lvl1 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "1")})
    expected = np.stack([mean_downsample(data[c], (2, 2, 2)) for c in range(2)], axis=0)
    np.testing.assert_array_equal(_full(lvl1), expected)

    # OME metadata: channel axis first, scale 1 / translation 0 on channel
    ms = summary["attrs"]["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["c", "z", "y", "x"]
    assert ms["axes"][0]["type"] == "channel"
    assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [1.0, 16.0, 16.0, 16.0]


def test_convert_infers_channels_from_ndim(tmp_path):
    data = np.zeros((3, 8, 8, 8), dtype=np.uint8)
    src = str(tmp_path / "inf.zarr")
    _make_zarr(src, data, has_channels=True, num_channels=3, chunk=(8, 8, 8))
    dst = str(tmp_path / "inf_out.zarr")
    summary = convert(src, dst, voxel_size=(8, 8, 8), multiscale=False,
                      profile="local", chunk=(8, 8, 8), delete_existing=True)
    assert summary["level_shapes"] == [(3, 8, 8, 8)]
