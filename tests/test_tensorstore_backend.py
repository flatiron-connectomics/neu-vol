import numpy as np
import pytest

from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec


def test_unsharded_roundtrip(tmp_path):
    path = str(tmp_path / "a.zarr")
    spec = zarr3_create_spec("local", path, (10, 12, 14), "uint16",
                             dimension_names=("z", "y", "x"), chunk=(4, 4, 4))
    be = TensorStoreBackend.create(spec, delete_existing=True)
    assert be.shape == (10, 12, 14)
    assert be.dtype == np.dtype("uint16")
    assert be.chunks == (4, 4, 4)          # unsharded: array-level chunk == read chunk
    assert be.read_chunks == (4, 4, 4)

    data = np.arange(10 * 12 * 14, dtype=np.uint16).reshape(10, 12, 14)
    be.write_region((slice(0, 10), slice(0, 12), slice(0, 14)), data)
    # reopen via spec (as a dask worker would) and verify
    re = open_backend(be.to_spec())
    back = re.read_region((slice(0, 10), slice(0, 12), slice(0, 14)))
    np.testing.assert_array_equal(back, data)


def test_sharded_reports_shard_as_chunks(tmp_path):
    path = str(tmp_path / "s.zarr")
    spec = zarr3_create_spec("ceph", path, (16, 16, 16), "uint8",
                             dimension_names=("z", "y", "x"),
                             chunk=(4, 4, 4), shard=(8, 8, 8))
    be = TensorStoreBackend.create(spec, delete_existing=True)
    # engine tiles by the shard (write granularity); viewer sees the inner chunk
    assert be.chunks == (8, 8, 8)
    assert be.read_chunks == (4, 4, 4)
    be.write_region((slice(0, 8), slice(0, 8), slice(0, 8)), np.full((8, 8, 8), 3, np.uint8))
    assert be.read_region((slice(0, 8), slice(0, 8), slice(0, 8)))[0, 0, 0] == 3


def test_shard_not_multiple_of_chunk_raises(tmp_path):
    with pytest.raises(ValueError):
        zarr3_create_spec("ceph", str(tmp_path / "x.zarr"), (16,), "uint8",
                          chunk=(3,), shard=(8,))
