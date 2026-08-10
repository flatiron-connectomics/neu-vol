"""Transient-failure retry, and the backend cache that reduces exposure to it."""

import pytest

from em_volume_tools.retry import is_transient, with_retry

# Verbatim from the run that died mid-copy: a TLS reset on the destination.
REAL_TRANSIENT = (
    'UNAVAILABLE: Error opening "neuroglancer_precomputed" driver: Error reading '
    '"sample3/seg/info": CURL error SSL connect error: Recv failure: Connection '
    "reset by peer [curl_code='35']"
)
# Verbatim from a downsample that died two seconds into an 85,536-task level: one
# worker could not resolve the S3 hostname.
REAL_DNS = (
    'UNAVAILABLE: Error opening "neuroglancer_precomputed" driver: Error reading '
    '"sample3/gt_v1/info": CURL error Could not resolve hostname: Could not resolve '
    "host: my-bucket.s3.amazonaws.com [curl_code='6']"
)
# Verbatim from the run where workers lacked bootstrapped credentials.
REAL_PERMANENT = (
    "PERMISSION_DENIED: AccessDenied: Access Denied "
    "[http_response_code='403'] [x-amzn-requestid='PCTZK4NTYTW4Y8G3']"
)


def test_classifies_the_real_errors_we_have_seen():
    assert is_transient(ValueError(REAL_TRANSIENT))
    assert is_transient(ValueError(REAL_DNS))
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
# --------------------------------------------------------------------------- #
# the block workers, which are where this actually has to be wired in
# --------------------------------------------------------------------------- #
def _flaky_open(monkeypatch, error, fail_times):
    """Make the next ``fail_times`` open_backend calls raise ``error``."""
    from em_volume_tools.ops import _multiscale

    real = _multiscale.open_backend
    state = {"left": fail_times, "calls": 0}

    def fake(spec):
        state["calls"] += 1
        if state["left"] > 0:
            state["left"] -= 1
            raise ValueError(error)
        return real(spec)

    monkeypatch.setattr(_multiscale, "open_backend", fake)
    return state


def _one_block_volume(tmp_path, name):
    """A source and a destination array, and the single block that spans them."""
    import numpy as np
    from em_blockrun import iter_blocks
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    shape = (8, 8, 8)
    src = TensorStoreBackend.create(
        zarr3_create_spec("local", str(tmp_path / f"{name}.src"), shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=shape),
        delete_existing=True)
    src.write_region(tuple(slice(0, s) for s in shape),
                     np.ones(shape, np.uint8))
    dst = TensorStoreBackend.create(
        zarr3_create_spec("local", str(tmp_path / f"{name}.dst"), shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=shape),
        delete_existing=True)
    block = next(iter(iter_blocks(shape, shape)))
    return src.to_spec(), dst.to_spec(), block


@pytest.mark.parametrize("worker", ["copy", "downsample"])
def test_a_block_survives_a_transient_failure(tmp_path, monkeypatch, worker):
    """The gap this closes: `with_retry` existed but the block workers never used it.

    A run is tens of thousands of tasks; at that scale one bad connection is close to
    certain, and fail-fast turned it into a lost run twice — a TLS reset 11 TB into a
    copy, and a DNS lookup two seconds into a downsample.
    """
    import numpy as np
    from em_volume_tools.backends.base import open_backend
    from em_volume_tools.ops._multiscale import _copy_block, _downsample_block

    monkeypatch.setattr("time.sleep", lambda *_: None)      # no real backoff
    src_spec, dst_spec, block = _one_block_volume(tmp_path, worker)
    state = _flaky_open(monkeypatch, REAL_DNS, fail_times=2)

    if worker == "copy":
        index, status = _copy_block(block, src_spec=src_spec, dst_spec=dst_spec,
                                    out_dtype="uint8")
    else:
        index, status = _downsample_block(block, src_spec=src_spec, dst_spec=dst_spec,
                                          factor=(1, 1, 1), kind="image")
    assert status == "written" and state["left"] == 0
    out = open_backend(dst_spec).read_region(block.region)
    assert np.array_equal(out, np.ones((8, 8, 8), np.uint8)), "the retry wrote nothing"


def test_a_block_does_not_retry_a_permanent_failure(tmp_path, monkeypatch):
    """A 403 must fail on the first attempt — retrying it burns the backoff budget on
    every task of a misconfigured run before failing anyway."""
    from em_volume_tools.ops._multiscale import _copy_block

    monkeypatch.setattr("time.sleep", lambda *_: None)
    src_spec, dst_spec, block = _one_block_volume(tmp_path, "perm")
    state = _flaky_open(monkeypatch, REAL_PERMANENT, fail_times=99)

    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        _copy_block(block, src_spec=src_spec, dst_spec=dst_spec, out_dtype="uint8")
    assert state["calls"] == 1, f"retried a permanent error {state['calls']} times"


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
