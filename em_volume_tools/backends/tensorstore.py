"""TensorStore-backed reader/writer for zarr v3 and neuroglancer-precomputed.

Primary read/write engine (see docs/DESIGN.md §2). Handles array chunk data and
codecs (incl. the zarr v3 sharding codec). ``.chunks`` reports the *array-level*
chunk shape — the shard when sharding is on — which is exactly the write-block
granularity the engine should tile by (docs/DESIGN.md §6b).

OME-NGFF group ``multiscales`` metadata is written separately (see ngff.py);
precomputed multiscale is intrinsic to the format and handled by this driver.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

#: spec tags this backend serves.
TAGS = ("zarr3", "neuroglancer_precomputed")

_TAG_TO_DRIVER = {
    "zarr3": "zarr3",
    "neuroglancer_precomputed": "neuroglancer_precomputed",
}


def _kvstore_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a TensorStore kvstore dict from ``spec['kvstore']`` or ``spec['path']``."""
    if "kvstore" in spec:
        return dict(spec["kvstore"])
    if "path" in spec:
        return {"driver": "file", "path": str(spec["path"])}
    raise ValueError("spec needs either 'kvstore' or 'path'")


def _compressor_codec(name: str | None, level: int | None) -> list[dict[str, Any]]:
    if name is None:
        return []
    if name == "gzip":
        return [{"name": "gzip", "configuration": {"level": level if level is not None else 5}}]
    if name == "zstd":
        return [{"name": "zstd", "configuration": {"level": level if level is not None else 5}}]
    if name == "blosc":
        return [{"name": "blosc", "configuration": {"cname": "zstd", "clevel": level if level is not None else 5, "shuffle": "shuffle"}}]
    raise ValueError(f"unsupported compressor {name!r}")


def _zarr3_metadata(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build zarr v3 metadata from a normalized create spec.

    Keys: shape, dtype, chunk_shape (array-level = shard if sharded),
    optional inner_chunk_shape (enables sharding), compressor, compressor_level,
    dimension_names.
    """
    shape = [int(s) for s in spec["shape"]]
    chunk_shape = [int(c) for c in spec["chunk_shape"]]
    inner = spec.get("inner_chunk_shape")
    base_codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    base_codecs += _compressor_codec(spec.get("compressor"), spec.get("compressor_level"))

    if inner is not None:
        codecs = [{
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [int(c) for c in inner],
                "codecs": base_codecs,
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ],
            },
        }]
    else:
        codecs = base_codecs

    metadata: dict[str, Any] = {
        "shape": shape,
        "data_type": str(spec["dtype"]),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "codecs": codecs,
    }
    if spec.get("dimension_names") is not None:
        metadata["dimension_names"] = list(spec["dimension_names"])
    return metadata


class TensorStoreBackend:
    """Wraps an open ``tensorstore.TensorStore`` behind the ArrayBackend protocol."""

    def __init__(self, store: Any, tag: str, kvstore: Mapping[str, Any]):
        self._store = store
        self._tag = tag
        self._kvstore = dict(kvstore)

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "TensorStoreBackend":
        """Open an existing store (read/write). Metadata is read from storage."""
        import tensorstore as ts

        tag = spec["backend"]
        kv = _kvstore_from_spec(spec)
        ts_spec: dict[str, Any] = {"driver": _TAG_TO_DRIVER[tag], "kvstore": kv}
        store = ts.open(ts_spec).result()
        return cls(store, tag, kv)

    @classmethod
    def create(cls, spec: Mapping[str, Any], *, delete_existing: bool = False) -> "TensorStoreBackend":
        """Create a new store from a normalized create spec (zarr3 only for now)."""
        import tensorstore as ts

        tag = spec["backend"]
        if tag != "zarr3":
            raise NotImplementedError(f"create for {tag!r} not implemented yet (zarr3 only)")
        kv = _kvstore_from_spec(spec)
        ts_spec = {
            "driver": "zarr3",
            "kvstore": kv,
            "metadata": _zarr3_metadata(spec),
            "create": True,
            "delete_existing": delete_existing,
        }
        store = ts.open(ts_spec).result()
        return cls(store, tag, kv)

    # -- ArrayBackend ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._store.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._store.dtype.numpy_dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        # Array-level (write) chunk = the shard when sharding is enabled.
        return tuple(int(s) for s in self._store.chunk_layout.write_chunk.shape)

    @property
    def read_chunks(self) -> tuple[int, ...]:
        """Inner read chunk (viewer-facing granularity); == chunks when unsharded."""
        return tuple(int(s) for s in self._store.chunk_layout.read_chunk.shape)

    def read_region(self, region: Region) -> np.ndarray:
        return self._store[region].read().result()

    def write_region(self, region: Region, data: np.ndarray) -> None:
        self._store[region].write(data).result()

    def to_spec(self) -> dict[str, Any]:
        # Minimal, reopenable, picklable spec (metadata lives in storage).
        return {"backend": self._tag, "kvstore": dict(self._kvstore)}


for _tag in TAGS:
    register_backend(_tag, TensorStoreBackend.open)
