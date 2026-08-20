"""Exercise the real distributed path (LocalCluster).

This runs the block-map workers through dask.distributed rather than serially,
verifying that the per-block partials, Block objects, and backend specs pickle
and execute on separate worker processes — the same machinery the SLURM path
uses. It is the closest we can get to a SLURM smoke test without submitting jobs.
"""

import os

import numpy as np
import pytest

from neu_vol import convert, start_dask
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec
from neu_vol.pyramid import mean_downsample

# The bundled template from blockrun, resolved by name. This used to be the
# repo-relative "../configs/dask-local.yaml", which broke when that directory became
# untracked site-local scratch — and broke *invisibly*, because a working tree that
# still had the file kept passing while a fresh clone could not.
CONFIG = "dask-local"


def _make_zarr(path, data, chunk=(4, 4, 4)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True,
    )
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


@pytest.mark.slow
def test_convert_over_localcluster(tmp_path):
    vol = np.random.default_rng(7).integers(0, 500, (16, 16, 16), dtype=np.uint16)
    src = str(tmp_path / "src.zarr")
    _make_zarr(src, vol)
    dst = str(tmp_path / "out.zarr")

    with start_dask(num_workers=2, config_path=CONFIG, label="test") as client:
        summary = convert(src, dst, voxel_size=(8, 8, 8), profile="local",
                          chunk=(8, 8, 8), min_dim=8, multiscale=True,
                          client=client, delete_existing=True)

    assert summary["num_levels"] == 2
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    lvl1 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "1")})
    np.testing.assert_array_equal(lvl0.read_region((slice(0, 16),) * 3), vol)
    np.testing.assert_array_equal(lvl1.read_region((slice(0, 8),) * 3),
                                  mean_downsample(vol, (2, 2, 2)))
