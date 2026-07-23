"""Read coordinate metadata + the level-0 data location from a source volume.

``convert`` uses this so you don't have to re-specify ``voxel_size``/``offset``
when the source already carries them (OME-NGFF zarr groups, precomputed ``info``).
Any field the caller passes explicitly overrides what's read here. Sources
without reliable metadata (image stacks, bare arrays, HDF5) return ``None``.

Returned dict keys: ``data_spec`` (the array/scale spec to actually read),
``voxel_size``/``offset`` (canonical z,y,x), ``units``, ``spatial_axes``,
``has_channels``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping


def _kvstore_of(spec: Mapping[str, Any], *, trailing_slash: bool = True) -> dict[str, Any]:
    if "kvstore" in spec:
        kv = dict(spec["kvstore"])
    elif "path" in spec:
        kv = {"driver": "file", "path": str(spec["path"])}
    else:
        raise ValueError("spec needs 'kvstore' or 'path'")
    if trailing_slash and "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    return kv


def _read_key(kvstore: Mapping[str, Any], key: str) -> bytes | None:
    import tensorstore as ts

    kv = ts.KvStore.open(dict(kvstore)).result()
    res = kv.read(key).result()
    if res.state != "value":
        return None
    return bytes(res.value)


def read_source_metadata(spec: Mapping[str, Any]) -> dict | None:
    backend = spec.get("backend")
    if backend == "zarr3":
        return _read_zarr_ome(spec)
    if backend == "neuroglancer_precomputed":
        return _read_precomputed(spec)
    return None


def _base_path(spec: Mapping[str, Any]) -> str:
    if "path" in spec:
        return str(spec["path"])
    return str(spec["kvstore"]["path"])


def _read_zarr_ome(spec: Mapping[str, Any]) -> dict | None:
    raw = _read_key(_kvstore_of(spec), "zarr.json")
    if raw is None:
        return None
    meta = json.loads(raw)
    attrs = meta.get("attributes", {})
    ome = attrs.get("ome")
    if meta.get("node_type") != "group" or not ome:
        return None  # bare array (or non-OME group) -> no coordinate metadata
    ms = ome["multiscales"][0]
    axes = ms["axes"]
    ds0 = ms["datasets"][0]
    ct = ds0["coordinateTransformations"]
    scale = next(t["scale"] for t in ct if t["type"] == "scale")
    translation = next((t["translation"] for t in ct if t["type"] == "translation"),
                       [0.0] * len(scale))

    spatial = [i for i, a in enumerate(axes) if a.get("type") == "space"]
    has_channels = any(a.get("type") == "channel" for a in axes)
    return {
        "data_spec": {"backend": "zarr3", "path": os.path.join(_base_path(spec), ds0["path"])},
        "voxel_size": tuple(scale[i] for i in spatial),
        "offset": tuple(translation[i] for i in spatial),
        "units": axes[spatial[0]].get("unit"),
        "spatial_axes": tuple(axes[i]["name"] for i in spatial),
        "has_channels": has_channels,
    }


def _read_precomputed(spec: Mapping[str, Any]) -> dict | None:
    raw = _read_key(_kvstore_of(spec), "info")
    if raw is None:
        return None
    info = json.loads(raw)
    scales = info["scales"]
    # level 0 = finest scale (smallest resolution).
    idx, sc0 = min(enumerate(scales), key=lambda kv: tuple(kv[1]["resolution"]))
    res_xyz = sc0["resolution"]
    voxel_size = tuple(float(v) for v in res_xyz[::-1])          # -> (z, y, x)
    voxel_offset_xyz = sc0.get("voxel_offset", [0, 0, 0])
    offset = tuple(float(o) * v for o, v in zip(voxel_offset_xyz[::-1], voxel_size))
    nch = int(info.get("num_channels", 1))
    return {
        "data_spec": {"backend": "neuroglancer_precomputed",
                      "path": _base_path(spec), "scale_index": idx},
        "voxel_size": voxel_size,
        "offset": offset,
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": nch > 1,
    }
