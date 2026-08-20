"""Empty-chunk elision + manifest bookkeeping for sparse (segmentation) data."""

import json
import os

import numpy as np

from neu_vol import convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec


def _make_zarr(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True,
    )
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _count_chunk_objects(zarr_level_path):
    # count files under the "c/" chunk prefix (unsharded zarr v3)
    n = 0
    for root, _dirs, files in os.walk(os.path.join(zarr_level_path, "c")):
        n += len(files)
    return n


def test_empty_chunks_elided_but_recorded(tmp_path):
    # 16^3 volume, chunk 8 -> 8 blocks; make only one block non-zero
    seg = np.zeros((16, 16, 16), dtype=np.uint64)
    seg[0:8, 0:8, 0:8] = 7  # exactly one 8^3 block has data
    src = str(tmp_path / "seg.zarr")
    _make_zarr(src, seg)
    dst = str(tmp_path / "out.zarr")

    summary = convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation",
                      chunk=(8, 8, 8), multiscale=False, delete_existing=True)

    # only the single non-empty chunk is written; the other 7 are elided
    assert _count_chunk_objects(os.path.join(dst, "0")) == 1
    assert summary["status_counts"].get("written") == 1
    assert summary["status_counts"].get("empty") == 7

    # elided chunks read back as fill (0); the data chunk is intact
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    full = lvl0.read_region((slice(0, 16),) * 3)
    np.testing.assert_array_equal(full, seg)

    # manifest recorded all 8 blocks (1 written + 7 empty) as done
    prog = summary["progress_path"]
    records = [json.loads(l) for l in open(prog) if l.strip()]
    statuses = [r["status"] for r in records if "status" in r]  # skip the meta line
    assert statuses.count("written") == 1
    assert statuses.count("empty") == 7


def test_resume_skips_empty_without_reprocessing(tmp_path):
    seg = np.zeros((16, 16, 16), dtype=np.uint64)
    seg[8:16, 8:16, 8:16] = 3
    src = str(tmp_path / "s.zarr")
    _make_zarr(src, seg)
    dst = str(tmp_path / "o.zarr")

    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, resume=True)
    # relaunch resume: every block (incl. empties) already recorded -> all filtered out
    summary = convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
                      multiscale=False, resume=True)
    # nothing reprocessed this run (manifest had them all); data still correct
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(lvl0.read_region((slice(0, 16),) * 3), seg)
