"""Transient-failure retry, and the backend cache that reduces exposure to it."""

import pytest

from em_volume_tools.retry import is_transient, with_retry

# Verbatim from the run that died mid-copy: a TLS reset on the destination.
REAL_TRANSIENT = (
    'UNAVAILABLE: Error opening "neuroglancer_precomputed" driver: Error reading '
    '"sample3/seg/info": CURL error SSL connect error: Recv failure: Connection '
    "reset by peer [curl_code='35']"
)
# Verbatim from the run where workers lacked bootstrapped credentials.
REAL_PERMANENT = (
    "PERMISSION_DENIED: AccessDenied: Access Denied "
    "[http_response_code='403'] [x-amzn-requestid='PCTZK4NTYTW4Y8G3']"
)


def test_classifies_the_real_errors_we_have_seen():
    assert is_transient(ValueError(REAL_TRANSIENT))
    assert not is_transient(ValueError(REAL_PERMANENT))


def test_permanent_markers_win_over_transient_ones():
    """Both arrive as ValueError, so text order is the only safeguard.

    A 403 that also mentions a transient-looking phrase must still be permanent —
    otherwise a misconfigured run burns the full backoff budget on every task.
    """
    mixed = "PERMISSION_DENIED: Access Denied; CURL error Connection reset by peer"
    assert not is_transient(ValueError(mixed))


def test_unrecognized_errors_are_permanent():
    """A new failure mode should surface promptly, not be retried into a timeout."""
    assert not is_transient(ValueError("something entirely new"))
    assert not is_transient(KeyError("shape"))


def test_retries_transient_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError(REAL_TRANSIENT)
        return "ok"

    assert with_retry(flaky, attempts=5, base_delay=0.001) == "ok"
    assert len(calls) == 3


def test_permanent_error_is_not_retried():
    calls = []

    def denied():
        calls.append(1)
        raise ValueError(REAL_PERMANENT)

    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        with_retry(denied, attempts=5, base_delay=0.001)
    assert len(calls) == 1, "a 403 must fail on the first attempt"


def test_raises_the_last_error_when_attempts_run_out():
    calls = []

    def always_flaky():
        calls.append(1)
        raise ValueError(REAL_TRANSIENT)

    with pytest.raises(ValueError, match="UNAVAILABLE"):
        with_retry(always_flaky, attempts=3, base_delay=0.001)
    assert len(calls) == 3


def test_backoff_grows_and_is_bounded(monkeypatch):
    waits = []
    monkeypatch.setattr("em_volume_tools.retry.time.sleep", waits.append)

    def always_flaky():
        raise ValueError(REAL_TRANSIENT)

    with pytest.raises(ValueError):
        with_retry(always_flaky, attempts=6, base_delay=1.0, max_delay=4.0, jitter=False)
    assert waits == [1.0, 2.0, 4.0, 4.0, 4.0], waits      # doubles, then clamps


# --------------------------------------------------------------------------- #
# Backend cache — an uncached open costs a TLS handshake per block
# --------------------------------------------------------------------------- #
def test_backends_are_cached_per_spec(tmp_path):
    import numpy as np

    from em_volume_tools.backends import base
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    path = str(tmp_path / "vol.zarr")
    TensorStoreBackend.create(
        zarr3_create_spec("local", path, (16, 16, 16), "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)

    base.clear_backend_cache()
    spec = {"backend": "zarr3", "path": path}
    first = base.open_backend(spec)
    assert base.open_backend(spec) is first, "reopening the same spec must reuse the handle"
    assert base.open_backend(dict(spec)) is first, "an equal spec must hit the cache"

    # a different spec must NOT collide
    other = str(tmp_path / "other.zarr")
    TensorStoreBackend.create(
        zarr3_create_spec("local", other, (16, 16, 16), "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    assert base.open_backend({"backend": "zarr3", "path": other}) is not first

    # the shared handle still works for I/O
    first.write_region((slice(0, 8),) * 3, np.full((8, 8, 8), 5, np.uint64))
    assert int(base.open_backend(spec).read_region((slice(0, 8),) * 3)[0, 0, 0]) == 5


def test_recreating_a_volume_invalidates_the_cache(tmp_path):
    """A cached handle must not keep serving a destroyed volume's metadata."""
    from em_volume_tools.backends import base
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    path = str(tmp_path / "vol.zarr")
    spec16 = zarr3_create_spec("local", path, (16, 16, 16), "uint64",
                               dimension_names=("z", "y", "x"), chunk=(8, 8, 8))
    TensorStoreBackend.create(spec16, delete_existing=True)
    cached = base.open_backend({"backend": "zarr3", "path": path})
    assert tuple(int(s) for s in cached.shape) == (16, 16, 16)

    # recreate at the same location with a different shape
    spec32 = zarr3_create_spec("local", path, (32, 32, 32), "uint64",
                               dimension_names=("z", "y", "x"), chunk=(8, 8, 8))
    TensorStoreBackend.create(spec32, delete_existing=True)
    fresh = base.open_backend({"backend": "zarr3", "path": path})
    assert tuple(int(s) for s in fresh.shape) == (32, 32, 32), \
        "stale cached handle survived a delete_existing recreate"
