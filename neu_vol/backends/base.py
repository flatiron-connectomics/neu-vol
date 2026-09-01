"""Backend protocol and spec-based opener.

The block-map engine is backend-agnostic: it only needs to read and write
rectangular regions and know a store's shape/dtype/chunks. Concrete backends
(TensorStore for zarr v3 + precomputed, tifffile/imageio for image stacks, h5py
for HDF5 shards) implement :class:`ArrayBackend`.

Backends are opened from a JSON-serializable **spec** (a plain dict). This is the
key to dask: a per-block task ships the spec (paths + params), not an open
handle, and reopens the backend on the worker.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

# A region is a per-axis tuple of slices in the backend's own (canonical) order.
Region = tuple[slice, ...]


@runtime_checkable
class ArrayBackend(Protocol):
    """Minimal chunked-array interface the engine depends on."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Full array shape in canonical order ((c,) z, y, x)."""
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    @property
    def chunks(self) -> tuple[int, ...]:
        """Storage chunk shape (the write granularity), same axis order as shape."""
        ...

    def read_region(self, region: Region) -> np.ndarray:
        """Read a rectangular region; returns an array of shape matching ``region``."""
        ...

    def write_region(self, region: Region, data: np.ndarray) -> None:
        """Write ``data`` into ``region``. ``data.shape`` must match the region."""
        ...

    def to_spec(self) -> dict[str, Any]:
        """Return a JSON-serializable spec that :func:`open_backend` can reopen."""
        ...


# Registry of spec "backend" tags -> a callable that opens the backend from a spec.
# Concrete backends register themselves on import (see backends/__init__.py).
_OPENERS: dict[str, Any] = {}


def register_backend(tag: str, opener: Any) -> None:
    """Register ``opener(spec) -> ArrayBackend`` under a spec tag."""
    _OPENERS[tag] = opener


# Opened backends, keyed by their spec. Opening is NOT cheap: for
# neuroglancer_precomputed it reads `info` from the store, so a remote destination
# costs an HTTPS round trip and a TLS handshake *per open*. Workers call
# open_backend once per block, so an uncached open turns one copy stage into tens
# of thousands of handshakes — and one of those eventually gets reset by the peer,
# which is fatal to a fail-fast stage. Measured: a whole-volume scale-0 copy is
# 10,692 blocks x 2 opens.
#
# Keyed on the full spec, so a different scale_index / path / kvstore is a
# different entry. Per process, so dask workers each build their own.
_BACKENDS: dict[str, ArrayBackend] = {}


def _spec_key(spec: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(spec), sort_keys=True, default=str)


def clear_backend_cache() -> None:
    """Drop all cached backends.

    Call after anything that invalidates an open handle — deleting and recreating
    a volume at the same location, most importantly, where a cached handle would
    otherwise keep serving the old metadata.
    """
    _BACKENDS.clear()


def spec_names_path(spec: Any, path: str) -> bool:
    """Whether ``spec`` reaches ``path``, nested views included.

    A view backend (``crop``, ``mask``, ``remap``) carries its source spec inside itself,
    so a match has to be looked for at any depth: a cached crop over an HDF5 file holds
    that file's handle just as directly as the plain backend does.
    """
    if isinstance(spec, Mapping):
        if str(spec.get("path", "")) == path:
            return True
        return any(spec_names_path(v, path) for v in spec.values())
    if isinstance(spec, (list, tuple)):
        return any(spec_names_path(v, path) for v in spec)
    return False


def release_backends(path: str) -> int:
    """Drop and **close** every cached backend that reads ``path``. Returns how many.

    Call before rewriting a file in place. This exists because of a failure that says
    nothing about caching: an ``HDF5Backend`` holds an open ``h5py.File`` for the life of
    the cache entry, and HDF5 refuses to open a file for writing while any handle has it
    open read-only — ``OSError: Unable to synchronously open file (file is already open for
    read-only)``. So reading a file and then writing to it *in the same process* — exactly
    what a notebook cleaning pass does, and what a re-run of one does — failed on the
    write with an error pointing at the file rather than at the reader still holding it.

    The other half is staleness: an entry that survived a rewrite would keep serving the
    array and the attributes from before it, which is the same hazard
    :func:`clear_backend_cache` was written for and is silent instead of loud.

    Closing, not merely dropping, is the load-bearing part — the lock belongs to the open
    handle, and waiting for garbage collection to release it makes the failure depend on
    when a reference happens to die. A backend with no ``close`` is simply dropped.
    """
    dropped = 0
    for key, backend in list(_BACKENDS.items()):
        try:
            spec = backend.to_spec()
        except Exception:                                    # pragma: no cover - defensive
            continue
        if not spec_names_path(spec, str(path)):
            continue
        _BACKENDS.pop(key, None)
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()
        dropped += 1
    return dropped


def open_backend(spec: Mapping[str, Any]) -> ArrayBackend:
    """Open a backend from a serializable spec (cached per spec, per process).

    ``spec["backend"]`` selects the implementation (e.g. ``"zarr3"``,
    ``"neuroglancer_precomputed"``, ``"image_stack"``, ``"hdf5"``).

    The returned backend is **shared**: callers must treat it as a handle for
    region I/O and not mutate it. Use :func:`clear_backend_cache` if a volume is
    recreated underneath.
    """
    try:
        tag = spec["backend"]
    except KeyError as e:
        raise ValueError("spec is missing required key 'backend'") from e
    if tag not in _OPENERS:
        raise ValueError(
            f"unknown backend {tag!r}; registered: {sorted(_OPENERS)}"
        )
    key = _spec_key(spec)
    backend = _BACKENDS.get(key)
    if backend is None:
        backend = _OPENERS[tag](spec)
        _BACKENDS[key] = backend
    return backend
