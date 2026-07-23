import pytest

from em_volume_tools import open_backend
from em_volume_tools.backends.base import _OPENERS


def test_all_backends_registered():
    assert {"zarr3", "neuroglancer_precomputed", "image_stack", "hdf5"} <= set(_OPENERS)


def test_open_backend_unknown_tag():
    with pytest.raises(ValueError):
        open_backend({"backend": "does-not-exist"})


def test_open_backend_missing_tag():
    with pytest.raises(ValueError):
        open_backend({"path": "/tmp/x"})


def test_open_missing_store_raises(tmp_path):
    with pytest.raises(Exception):
        open_backend({"backend": "zarr3", "path": str(tmp_path / "does-not-exist")})
