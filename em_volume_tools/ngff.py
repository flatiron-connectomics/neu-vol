"""OME-NGFF 0.5 multiscale group metadata.

TensorStore writes the per-level array data; this module writes the *group-level*
``multiscales`` metadata that ties those arrays into an OME-Zarr pyramid
(docs/DESIGN.md §6a). The attribute schema (``{"ome": {"version": "0.5",
"multiscales": [...]}}``) matches what ngff-zarr emits, and can be validated with
:func:`validate_attrs` (needs the jsonschema dep).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# Map short unit strings to OME/UDUNITS names accepted by the 0.5 spec.
_UNIT_MAP = {
    "nm": "nanometer",
    "nanometer": "nanometer",
    "um": "micrometer",
    "µm": "micrometer",
    "micron": "micrometer",
    "micrometer": "micrometer",
    "mm": "millimeter",
    "millimeter": "millimeter",
    "m": "meter",
    "meter": "meter",
    "angstrom": "angstrom",
    "A": "angstrom",
}


def ome_unit(units: str | None) -> str | None:
    if units is None:
        return None
    return _UNIT_MAP.get(units, units)


def build_dataset(path: str, scale: Sequence[float], translation: Sequence[float]) -> dict[str, Any]:
    """One ``multiscales.datasets`` entry (scale then translation transforms)."""
    return {
        "path": str(path),
        "coordinateTransformations": [
            {"type": "scale", "scale": [float(s) for s in scale]},
            {"type": "translation", "translation": [float(t) for t in translation]},
        ],
    }


def build_multiscales_attrs(
    *,
    axis_names: Sequence[str],
    axis_types: Sequence[str],
    axis_units: Sequence[str | None],
    datasets: Sequence[Mapping[str, Any]],
    name: str = "image",
    method_type: str | None = None,
) -> dict[str, Any]:
    """Assemble the OME-NGFF 0.5 group attributes dict.

    ``axis_*`` and each dataset's transforms must span *all* axes in order
    (channel axis included, with scale 1 / translation 0).
    """
    axes = []
    for nm, ty, un in zip(axis_names, axis_types, axis_units):
        ax: dict[str, Any] = {"name": nm, "type": ty}
        if un is not None:
            ax["unit"] = un
        axes.append(ax)
    ms: dict[str, Any] = {"axes": axes, "datasets": list(datasets), "name": name}
    if method_type is not None:
        ms["type"] = method_type
    return {"ome": {"version": "0.5", "multiscales": [ms]}}


def validate_attrs(attrs: Mapping[str, Any], *, strict: bool = False) -> None:
    """Validate against the OME-NGFF 0.5 image schema (raises on failure)."""
    import ngff_zarr as nz

    nz.validate(dict(attrs), version="0.5", model="image", strict=strict)


def write_group_metadata(kvstore: Mapping[str, Any], attrs: Mapping[str, Any]) -> None:
    """Write the group ``zarr.json`` (node_type=group) with ``attrs`` to ``kvstore``.

    ``kvstore`` is a TensorStore kvstore spec for the *group* path (e.g.
    ``{"driver": "file", "path": "/.../vol.zarr"}``), so this works uniformly for
    file / s3 / gcs stores.
    """
    import tensorstore as ts

    # kvstore 'path' is a raw key prefix; ensure a trailing separator so the
    # "zarr.json" key isn't concatenated onto the group name.
    spec = dict(kvstore)
    if "path" in spec and not str(spec["path"]).endswith("/"):
        spec["path"] = str(spec["path"]) + "/"
    kv = ts.KvStore.open(spec).result()
    group_meta = {"zarr_format": 3, "node_type": "group", "attributes": dict(attrs)}
    kv.write("zarr.json", json.dumps(group_meta, indent=2).encode()).result()
