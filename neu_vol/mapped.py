"""A supervoxel volume read through a lookup table, addressed by a small descriptor file.

What this is for: a segmentation delivered as supervoxels plus several agglomeration
*views* (webknossos' ``agglomerate_view_NN``) is one volume and N tables, not N volumes.
Each view is ``table[supervoxels]``, so it can be read on the fly — for evaluation,
inspection, a crop — and converted into a real volume only once a view is chosen.

A view is addressed by a **descriptor**, a JSON file naming the volume and the table::

    {"neu_vol": "mapped-volume", "version": 1,
     "source": "/path/to/supervoxels.precomputed",
     "table": "agglomerate_view_80.npy", "mapping": "agglomerate_view_80"}

Being a file is what lets every location-taking entry point use it unchanged:
``neu-vol info``, ``read_piece`` (so ``neu-eval compare --segmentation``), and
``neu-vol convert --src`` for the conversion. Relative paths in it resolve against the
descriptor's own directory, so a directory of tables and descriptors can be moved whole.

**Coarse levels are approximate, level 0 is exact.** A coarse level of the source is a
*mode* of supervoxels, and ``table[mode(sv)]`` is not ``mode(table[sv])``: a coarse voxel
gets the view's id for its most common supervoxel, not its most common agglomerate. Fine
for looking; for anything measured, read level 0. ``convert`` reads level 0 and rebuilds
the pyramid from the mapped labels, so a converted view is exact at every level.

Descriptors and tables may be local or on an object store (``s3://…``), through
``neu_vol.location`` like every other read here. A table is a few MB per view and is
loaded whole, once per process.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Mapping

import numpy as np

MARKER = "mapped-volume"
TAG = "mapped"


def _is_remote(location: str) -> bool:
    return "://" in location


def absolute(location: str) -> str:
    """A local path made absolute; a URL unchanged. Descriptors record locations this way,
    so they mean the same thing from whatever directory they are read."""
    return location if _is_remote(location) else os.path.abspath(location)


def _parent(location: str) -> str:
    return location.rstrip("/").rpartition("/")[0] if _is_remote(location) \
        else os.path.dirname(os.path.abspath(location))


def _resolve(value: str, base: str) -> str:
    if _is_remote(value) or os.path.isabs(value):
        return value
    if _is_remote(base):
        return f"{base.rstrip('/')}/{value.removeprefix('./')}"
    return os.path.normpath(os.path.join(base, value))


def _read(location: str) -> bytes | None:
    from .location import read_bytes

    return read_bytes(location)


def is_descriptor(location: Any) -> bool:
    """True if ``location`` is a JSON object (local or remote) carrying this marker.

    Only ``.json`` names are read, so detection costs nothing for any other location.
    """
    if not isinstance(location, str) or not location.endswith(".json"):
        return False
    if not _is_remote(location) and not os.path.isfile(location):
        return False
    try:
        raw = _read(location)
        return raw is not None and json.loads(raw).get("neu_vol") == MARKER
    except Exception:                                          # noqa: BLE001
        return False


def read_descriptor(path: str) -> dict:
    """The descriptor at ``path``, with ``source`` and ``table`` resolved."""
    raw = _read(path)
    if raw is None:
        raise FileNotFoundError(f"no descriptor at {path}")
    d = json.loads(raw)
    if d.get("neu_vol") != MARKER:
        raise ValueError(f"{path} is not a mapped-volume descriptor")
    base = _parent(path)
    return {**d, "source": _resolve(d["source"], base), "table": _resolve(d["table"], base),
            "descriptor": absolute(path)}


def write_descriptor(path: str, *, source: str, table: str, mapping: str,
                     origin: Mapping[str, Any] | None = None) -> None:
    """Write a descriptor at ``path`` (local or remote).

    ``table`` is stored as given, so pass a bare file name for a table beside the
    descriptor — that keeps a directory of views movable as a whole. ``source`` is made
    absolute if it is a relative local path.
    """
    from .location import write_bytes

    doc = {"neu_vol": MARKER, "version": 1, "source": absolute(source),
           "table": table, "mapping": mapping}
    if origin:
        doc["origin"] = dict(origin)
    write_bytes(path, (json.dumps(doc, indent=2) + "\n").encode())


def table_bytes(table: np.ndarray) -> bytes:
    """A table as ``.npy`` bytes, for writing through ``location``."""
    buf = io.BytesIO()
    np.save(buf, table)
    return buf.getvalue()


#: Tables, per path, per process. A view opened at eight levels loads its table once.
_TABLES: dict[str, np.ndarray] = {}


def load_table(path: str) -> np.ndarray:
    if path not in _TABLES:
        raw = _read(path)
        if raw is None:
            raise FileNotFoundError(f"no lookup table at {path}")
        table = np.load(io.BytesIO(raw))
        if table.ndim != 1 or table.dtype.kind != "u":
            raise ValueError(f"{path} is not a 1-D unsigned lookup table "
                             f"({table.dtype}, shape {table.shape})")
        _TABLES[path] = table
    return _TABLES[path]


def clear_cache() -> None:
    _TABLES.clear()


def location_spec(path: str) -> dict:
    """The ``mapped`` spec for a descriptor: the source's own spec, wrapped."""
    from .source_metadata import detect_backend
    from .source_metadata import location_spec as _location_spec

    d = read_descriptor(path)
    fmt = detect_backend(d["source"])
    if fmt is None:
        raise FileNotFoundError(f"{path} maps {d['source']}, which is not a volume")
    if fmt == TAG:
        raise ValueError(f"{path} maps another mapped volume; chain the tables instead")
    return {"backend": TAG, "descriptor": d["descriptor"], "table": d["table"],
            "mapping": d.get("mapping"), "source_location": d["source"],
            "source_format": fmt, "source": _location_spec(d["source"], fmt)}
