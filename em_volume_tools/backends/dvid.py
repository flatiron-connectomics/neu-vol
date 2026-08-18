"""Read-only backend over a DVID ``labelmap`` instance.

DVID is not a kvstore, so this backend is addressed by ``server``/``uuid``/``instance``
rather than by a path — see ``em_volume_tools.dvid.parse_url`` for the URL form and
``source_metadata.detect_backend`` for where it is recognised.

**Addressing and version resolution live in** ``em_volume_tools/dvid.py``, not here: they
are the same for every instance type, and consumers above this package need them without
wanting an array. This module is only the labelmap part — geometry and the array view.
The shared names are re-exported below so existing callers keep working.

**Blocks that hold no labels are omitted from DVID's response**, which is why there is
no occupancy prefilter anywhere in this package for DVID sources: an empty region costs
a round trip and no bandwidth (measured at the 512^3 task size: 0.03 s and zero bytes,
against 0.4-0.6 s and ~12 MB for a full one). Filtering them out would save the cheap
tasks and leave the expensive ones untouched. See NOTES-TODO for the measurements.

Reads go through ``fetch_labelmap_voxels``, which requests ``?compression=blocks`` —
DVID's *stored* form, so the server never decompresses. Inflation happens here, in this
process, via libdvid; under dask that is the worker, so it is already spread across the
fleet. It is also the dominant per-task cost (~1.4 s of a 1.9 s 512^3 read), which means
throughput is bounded by DVID's serving rate rather than by our CPU. DVID is typically a
**shared** service and answers overload with 503 — see ``retry.TRANSIENT_MARKERS``.

Read-only: this package converts a labelmap into a neuroglancer volume; it never writes
segmentation back to DVID, which is the proofreading tool's job.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .. import dvid as _dvid
# Re-exported: `em_volume_tools.dvid` is their home — addressing and version resolution
# are the same for every DVID instance type, so they cannot live in a module whose
# subject is the label array. These aliases exist because this is where callers have
# always imported them from.
from ..dvid import (MISSING, check_instance_type, clear_node_cache,  # noqa: F401
                    instance_info, is_url, node_provenance, node_summary, parse_url,
                    resolve_node, spec_url)
from .base import Region, register_backend

TAG = "dvid"

# Every call below goes through `_dvid.` rather than the aliases above, deliberately: the
# aliases are separate bindings, so patching one module would leave the other pointing at
# the real function. One module attribute is one patch point — and a test that stubs the
# server has to actually stub it, or it silently starts needing the network.
_MISSING = MISSING


def provenance(spec: Mapping[str, Any], node: Mapping[str, Any]) -> dict:
    """What this labelmap pull came from — the shared node record plus label facts.

    ``maxlabel`` advances as proofreading creates bodies, so together with the node's DAG
    metadata it characterises the snapshot. It is per-*instance*, unlike DVID's mutation
    id and lastmod endpoints, which are both per-body and therefore useless here.
    """
    out = _dvid.node_provenance(spec, node)
    out["supervoxels"] = bool(spec.get("supervoxels", False))
    # Nice to have, not worth failing a completed run over.
    try:
        from neuclease.dvid import fetch_maxlabel

        out["maxlabel"] = int(fetch_maxlabel(out["server"], node["uuid"],
                                             out["instance"]))
    except Exception as exc:                                   # noqa: BLE001
        out["maxlabel_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def geometry(info: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    """Shape / chunks / voxel size / max scale, all zyx, from DVID's ``/info``.

    The volume is presented **origin-anchored**: shape is ``MaxPoint + 1``, so a voxel
    index means the same thing here as it does in DVID and in any other view of the
    same dataset. The margin below ``MinPoint`` is empty, which costs one 30 ms round
    trip per task and nothing on disk — the alternative, translating by ``MinPoint``,
    would buy a slightly smaller volume at the price of every coordinate disagreeing
    with DVID's own.
    """
    ext = info.get("Extended", {})
    _dvid.check_instance_type(info, spec, "labelmap")

    lo_xyz, hi_xyz = ext.get("MinPoint"), ext.get("MaxPoint")
    if lo_xyz is None or hi_xyz is None:
        raise ValueError(
            f"{_dvid.spec_url(spec)} reports no extents (MinPoint/MaxPoint are null), "
            f"which DVID does for an instance that has been created but never written "
            f"to. There is nothing to read.")
    lo = tuple(int(v) for v in lo_xyz[::-1])
    if any(v < 0 for v in lo):
        raise ValueError(
            f"{_dvid.spec_url(spec)} has a negative MinPoint {lo_xyz!r}. This backend "
            f"presents DVID's own coordinates origin-anchored, which would clip data "
            f"at negative indices. Support for a translated frame is not implemented.")

    block = tuple(int(v) for v in ext["BlockSize"][::-1])
    voxel = tuple(float(v) for v in ext["VoxelSize"][::-1])
    return {
        "shape0": tuple(int(v) + 1 for v in hi_xyz[::-1]),
        "min_point": lo,
        "chunks": block,
        "voxel_size": voxel,
        "max_scale": int(ext.get("MaxDownresLevel", 0)),
        "dtype": np.dtype("uint64"),
    }


def level_shape(shape0, scale: int) -> tuple[int, ...]:
    """Shape at a DVID downres level.

    DVID halves every axis per level, so unlike a general pyramid this factor really is
    ``2**scale`` — it is DVID's documented model rather than the assumption CLAUDE.md
    invariant 1 warns against, and it must not be copied to any other backend.
    """
    return tuple(-(-int(s) // 2 ** int(scale)) for s in shape0)


class DVIDBackend:
    """Read-only ``(z, y, x)`` view over one scale of a DVID labelmap instance."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._server, self._uuid, self._instance = _dvid.address(spec)
        self._scale = int(spec.get("scale_index", 0))
        self._supervoxels = bool(spec.get("supervoxels", False))

        geom = geometry(_dvid.instance_info(spec), spec)
        if self._scale > geom["max_scale"]:
            raise ValueError(
                f"{_dvid.spec_url(spec)} has scales 0..{geom['max_scale']} "
                f"(MaxDownresLevel), but scale {self._scale} was requested")
        self._shape = level_shape(geom["shape0"], self._scale)
        self._chunks = geom["chunks"]
        self._dtype = geom["dtype"]
        self.voxel_size = tuple(v * 2 ** self._scale for v in geom["voxel_size"])

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "DVIDBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    def read_region(self, region: Region) -> np.ndarray:
        try:
            from neuclease.dvid import fetch_labelmap_voxels
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        box = np.array([[int(s.start) for s in region],
                        [int(s.stop) for s in region]], dtype=np.int64)
        # fetch_labelmap_voxels rounds the request out to DVID's 64-voxel block grid and
        # truncates the result back, so an unaligned region is correct but wasteful. Our
        # task shapes are whole multiples of `chunks`, so in practice no rounding occurs.
        out = fetch_labelmap_voxels(
            self._server, self._uuid, self._instance, box,
            scale=self._scale, supervoxels=self._supervoxels)
        return np.ascontiguousarray(out)

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError(
            "DVID labelmap instances are read-only here. Proofreading edits belong in "
            "the proofreading tool; this package only pulls snapshots out.")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, DVIDBackend.open)
