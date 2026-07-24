"""TensorStore-backed reader/writer for zarr v3 and neuroglancer-precomputed.

Primary read/write engine (docs/DESIGN.md §2). Handles array chunk data and
codecs (incl. the zarr v3 sharding codec). ``.chunks`` reports the *array-level*
chunk shape — the shard when sharding is on — which is the write-block
granularity the engine tiles by (docs/DESIGN.md §6b).

Axis order: everything the engine sees is canonical ``(z, y, x)`` /
``(c, z, y, x)``. zarr v3 arrays are created in that order directly. Precomputed
is natively ``(x, y, z, channel)``; this backend presents a **transposed view**
and reads with ``order="C"`` so canonical, C-contiguous arrays reach the engine
(see the precomputed-axis-order note).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAGS = ("zarr3", "neuroglancer_precomputed")

_TAG_TO_DRIVER = {
    "zarr3": "zarr3",
    "neuroglancer_precomputed": "neuroglancer_precomputed",
}


def _kvstore_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if "kvstore" in spec:
        return dict(spec["kvstore"])
    if "path" in spec:
        from ..location import to_kvstore  # local path or s3://... URL
        return to_kvstore(spec["path"])
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


def _rev(seq):
    """Reverse a canonical (z, y, x) triple to precomputed (x, y, z)."""
    return [int(v) for v in seq[::-1]]


class TensorStoreBackend:
    """Wraps a ``tensorstore.TensorStore`` (canonical view) behind ArrayBackend."""

    def __init__(self, store: Any, tag: str, kvstore: Mapping[str, Any],
                 scale_index: int | None = None):
        self._store = store
        self._tag = tag
        self._kvstore = dict(kvstore)
        self._scale_index = scale_index
        self._scale_key_cache: str | None = None
        self._view = self._canonical_view(store, tag)

    @staticmethod
    def _canonical_view(store: Any, tag: str) -> Any:
        """Return a view in canonical (z,y,x)/(c,z,y,x) order."""
        if tag != "neuroglancer_precomputed":
            return store  # zarr3 arrays are created canonically
        import tensorstore as ts

        num_channels = int(store.shape[-1])  # native (x, y, z, channel)
        if num_channels == 1:
            return store[ts.d["channel"][0]].transpose([2, 1, 0])       # -> (z, y, x)
        return store.transpose([3, 2, 1, 0])                            # -> (c, z, y, x)

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "TensorStoreBackend":
        import tensorstore as ts

        tag = spec["backend"]
        kv = _kvstore_from_spec(spec)
        ts_spec: dict[str, Any] = {"driver": _TAG_TO_DRIVER[tag], "kvstore": kv}
        scale_index = spec.get("scale_index")
        if tag == "neuroglancer_precomputed" and scale_index is not None:
            ts_spec["scale_index"] = int(scale_index)
        store = ts.open(ts_spec).result()
        return cls(store, tag, kv, scale_index)

    @classmethod
    def open_or_create(cls, spec: Mapping[str, Any], *, resume: bool,
                       delete_existing: bool = False) -> "TensorStoreBackend":
        """Create the store, or (when ``resume``) open it if it already exists.

        On resume, an existing store with a matching shape is opened (so already
        written blocks survive); otherwise it is created fresh.
        """
        if not resume:
            return cls.create(spec, delete_existing=delete_existing)
        reopen: dict[str, Any] = {"backend": spec["backend"], "kvstore": _kvstore_from_spec(spec)}
        if spec["backend"] == "neuroglancer_precomputed" and "scale_index" in spec:
            reopen["scale_index"] = spec["scale_index"]
        try:
            be = cls.open(reopen)
            if tuple(int(s) for s in be.shape) == tuple(int(s) for s in spec["shape"]):
                return be
        except Exception:
            pass
        return cls.create(spec, delete_existing=False)

    @classmethod
    def create(cls, spec: Mapping[str, Any], *, delete_existing: bool = False) -> "TensorStoreBackend":
        import tensorstore as ts

        tag = spec["backend"]
        kv = _kvstore_from_spec(spec)
        if tag == "zarr3":
            ts_spec = {"driver": "zarr3", "kvstore": kv, "metadata": _zarr3_metadata(spec),
                       "create": True, "delete_existing": delete_existing}
            store = ts.open(ts_spec).result()
            return cls(store, tag, kv)
        if tag == "neuroglancer_precomputed":
            return cls._create_precomputed(spec, kv, delete_existing=delete_existing)
        raise ValueError(f"unknown backend {tag!r}")

    @classmethod
    def _create_precomputed(cls, spec, kv, *, delete_existing):
        import tensorstore as ts

        num_channels = int(spec.get("num_channels", 1))
        canonical_shape = [int(s) for s in spec["shape"]]
        spatial = canonical_shape[-3:]  # (z, y, x)
        encoding = spec.get("encoding", "raw")
        scale_metadata = {
            "size": _rev(spatial),
            "resolution": _rev(spec["resolution_zyx"]),
            "encoding": encoding,
            "chunk_size": _rev(spec["chunk_zyx"]),
            "voxel_offset": _rev(spec.get("voxel_offset_zyx", [0, 0, 0])),
        }
        if encoding == "compressed_segmentation":
            block = spec.get("compressed_segmentation_block_size_zyx", [8, 8, 8])
            scale_metadata["compressed_segmentation_block_size"] = _rev(block)
        ts_spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": kv,
            "multiscale_metadata": {
                "type": spec.get("type", "image"),
                "data_type": str(spec["dtype"]),
                "num_channels": num_channels,
            },
            "scale_metadata": scale_metadata,
            "create": True,
            "delete_existing": delete_existing,
        }
        store = ts.open(ts_spec).result()
        return cls(store, "neuroglancer_precomputed", kv, spec.get("scale_index"))

    # -- ArrayBackend ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._view.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._view.dtype.numpy_dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._view.chunk_layout.write_chunk.shape)

    @property
    def read_chunks(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._view.chunk_layout.read_chunk.shape)

    def read_region(self, region: Region) -> np.ndarray:
        # order="C" so the canonical view returns a C-contiguous array (the
        # reducers' reshape logic and cross-store writes depend on it).
        return np.ascontiguousarray(self._view[region].read(order="C").result())

    def write_region(self, region: Region, data: np.ndarray) -> None:
        self._view[region].write(data).result()

    def is_region_stored(self, region: Region) -> bool:
        """True if ``region`` is already fully written (authoritative, for verify).

        zarr3 uses TensorStore's ``storage_statistics``. precomputed's
        storage_statistics is broken in this TensorStore version, so we check the
        chunk object directly via the kvstore (valid for unsharded precomputed
        where one engine block == one chunk, which is our S3 case).
        """
        if self._tag == "zarr3":
            stats = self._view[region].storage_statistics(query_fully_stored=True).result()
            return bool(stats.fully_stored)
        # neuroglancer_precomputed: <scale_key>/<x0>-<x1>_<y0>-<y1>_<z0>-<z1>
        (z0, z1), (y0, y1), (x0, x1) = [(s.start, s.stop) for s in region[-3:]]
        key = f"{self._precomputed_scale_key()}/{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
        return self._store.kvstore.read(key).result().state == "value"

    def _precomputed_scale_key(self) -> str:
        if self._scale_key_cache is None:
            self._scale_key_cache = self._store.spec().to_json()["scale_metadata"]["key"]
        return self._scale_key_cache

    def to_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"backend": self._tag, "kvstore": dict(self._kvstore)}
        if self._tag == "neuroglancer_precomputed" and self._scale_index is not None:
            spec["scale_index"] = int(self._scale_index)
        return spec


for _tag in TAGS:
    register_backend(_tag, TensorStoreBackend.open)
