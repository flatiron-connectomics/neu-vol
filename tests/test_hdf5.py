import os

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.base import open_backend


def test_hdf5_source_convert_to_zarr(tmp_path):
    import h5py

    vol = np.random.default_rng(6).integers(0, 500, (16, 16, 16), dtype=np.uint16)
    h5path = str(tmp_path / "vol.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("data", data=vol, chunks=(8, 8, 8))

    dst = str(tmp_path / "from_h5.zarr")
    summary = convert({"backend": "hdf5", "path": h5path, "dataset": "data"}, dst,
                      voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
                      multiscale=False, delete_existing=True)
    assert summary["level_shapes"] == [(16, 16, 16)]
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(out.read_region((slice(0, 16),) * 3), vol)


def test_hdf5_backend_reports_shape_dtype_chunks(tmp_path):
    import h5py

    h5path = str(tmp_path / "x.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("seg", data=np.zeros((4, 5, 6), np.uint64), chunks=(2, 5, 6))
    be = open_backend({"backend": "hdf5", "path": h5path, "dataset": "seg"})
    assert be.shape == (4, 5, 6)
    assert be.dtype == np.dtype("uint64")
    assert be.chunks == (2, 5, 6)


def test_hdf5_missing_dataset_raises(tmp_path):
    import h5py

    h5path = str(tmp_path / "y.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("data", data=np.zeros((2, 2, 2)))
    with pytest.raises(KeyError):
        open_backend({"backend": "hdf5", "path": h5path, "dataset": "nope"})
