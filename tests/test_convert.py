import math
import os

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec
from neu_vol.pyramid import mean_downsample, mode_downsample


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


# --------------------------------------------------------------------------- #
# Task shape: reconcile source and destination chunking (level 0 only)
# --------------------------------------------------------------------------- #
def test_plan_task_shape_covers_whole_source_and_destination_chunks():
    from neu_vol.ops._multiscale import plan_task_shape

    # The specimen5 case: a CloudVolume source chunked 128x2048x2048 into 128^3.
    t = plan_task_shape((128, 2048, 2048), (128, 128, 128), (15401, 13544, 16648))
    assert t == (128, 2048, 2048), "should land on exactly one source chunk"
    for a, d in zip(t, (128, 128, 128)):
        assert a % d == 0, "a task must cover WHOLE destination chunks or writes race"


def test_plan_task_shape_respects_a_memory_ceiling():
    """A PNG slice cannot be partially decoded, so the LCM is the whole plane —
    28.8 GB for a 128-slab, which no worker can hold. It must be tiled down while
    staying a destination-chunk multiple."""
    from neu_vol.ops._multiscale import plan_task_shape

    t = plan_task_shape((1, 13544, 16648), (128, 128, 128), (15401, 13544, 16648),
                        itemsize=1, max_bytes=4 * 1024 ** 3)
    assert math.prod(t) <= 4 * 1024 ** 3
    for a, d in zip(t, (128, 128, 128)):
        assert a % d == 0
    assert t[0] == 128, "z must stay a whole destination chunk"


def test_plan_task_shape_is_a_no_op_when_chunkings_agree():
    from neu_vol.ops._multiscale import plan_task_shape

    assert plan_task_shape((64, 64, 64), (64, 64, 64), (512, 512, 512)) == (64, 64, 64)


def test_plan_task_shape_falls_back_when_the_source_has_no_chunking():
    from neu_vol.ops._multiscale import plan_task_shape

    assert plan_task_shape(None, (32, 32, 32), (256, 256, 256)) == (32, 32, 32)


def test_larger_tasks_produce_identical_output_and_fewer_of_them(tmp_path):
    """The whole point: fewer source reads, byte-identical result.

    A source chunked coarser than the destination previously produced one task per
    destination chunk, each re-fetching the same source chunk.
    """
    import json

    vol = np.random.default_rng(7).integers(0, 200, (8, 64, 64), dtype=np.uint8)
    src = str(tmp_path / "src.zarr")
    _make_zarr(src, vol, chunk=(8, 64, 64))          # one coarse source chunk
    dst = str(tmp_path / "out.zarr")
    progress = str(tmp_path / "p.jsonl")

    convert(src, dst, voxel_size=(8, 8, 8), kind="image", profile="local",
            chunk=(8, 16, 16), min_dim=64, delete_existing=True,
            progress_path=progress)

    got = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(_full(got), vol)

    records = [json.loads(l) for l in open(progress) if l.strip()]
    level0 = [r for r in records if r.get("group") == 0 and "status" in r]
    # 4x4 = 16 destination chunks, but they all live in one source chunk.
    assert len(level0) == 1, f"expected 1 task covering the source chunk, got {len(level0)}"

    # ...and the run must SAY so, or a reader counting the 16 chunks reports 6%.
    meta = [r["meta"] for r in records if r.get("group") == 0 and "meta" in r]
    assert meta and meta[-1]["total"] == 1
    assert tuple(meta[-1]["task_shape"]) == (8, 64, 64)
