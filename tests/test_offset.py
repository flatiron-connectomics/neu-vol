"""Nonzero voxel_offset handling for precomputed (translate_to 0-based view)."""

import numpy as np

from em_volume_tools import extract_roi
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import precomputed_create_spec


def _make_precomputed(path, vol):
    be = TensorStoreBackend.create(
        precomputed_create_spec("s3-neuroglancer", path, vol.shape, str(vol.dtype),
                                resolution_zyx=[8, 8, 8], scale_index=0, chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return be


def test_extract_roi_to_precomputed_global_offset(tmp_path):
    vol = np.arange(32 * 32 * 32, dtype=np.uint8).reshape(32, 32, 32)
    src = str(tmp_path / "src.precomputed")
    _make_precomputed(src, vol)
    dst = str(tmp_path / "roi.precomputed")

    # ROI at a nonzero global start -> output gets voxel_offset = start (8 vox)
    summary = extract_roi({"backend": "neuroglancer_precomputed", "path": src, "scale_index": 0},
                          dst, start=(8, 8, 8), stop=(24, 24, 24), voxel_size=(8, 8, 8),
                          kind="image", profile="s3-neuroglancer", chunk=(8, 8, 8),
                          multiscale=False, delete_existing=True)
    assert summary["level_shapes"] == [(16, 16, 16)]

    # the output carries the global voxel_offset in its info
    out = open_backend({"backend": "neuroglancer_precomputed", "path": dst, "scale_index": 0})
    assert out.shape == (16, 16, 16)                       # 0-based canonical view
    np.testing.assert_array_equal(out.read_region((slice(0, 16),) * 3), vol[8:24, 8:24, 8:24])


def test_verify_with_voxel_offset(tmp_path):
    vol = (np.arange(16 * 16 * 16, dtype=np.uint8).reshape(16, 16, 16) + 1)  # nonzero
    src = str(tmp_path / "s.precomputed")
    _make_precomputed(src, vol)
    dst = str(tmp_path / "r.precomputed")
    kw = dict(start=(4, 4, 4), stop=(12, 12, 12), voxel_size=(8, 8, 8), kind="image",
              profile="s3-neuroglancer", chunk=(8, 8, 8), multiscale=False)
    src_spec = {"backend": "neuroglancer_precomputed", "path": src, "scale_index": 0}
    extract_roi(src_spec, dst, delete_existing=True, **kw)
    # verify pass: is_region_stored must build the global (voxel_offset+region) key
    summary = extract_roi(src_spec, dst, verify=True, **kw)
    assert summary["status_counts"].get("skipped", 0) >= 1
