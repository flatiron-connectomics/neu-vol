"""What a volume was made from, written beside it as ``provenance.json``.

Written for **every** source, not just DVID — "where did this come from" is not a
DVID-specific question, and a record that only sometimes exists is one nobody learns to
look for. DVID is what forced the issue: a branch ref like ``93fdbc:main`` names a
different node tomorrow, so without recording the resolved uuid there is no way to say
afterwards which proofreading snapshot a volume holds.

**A sidecar file, not a field inside the format's own metadata.** It would be tempting
to put this in precomputed's ``info`` or zarr's OME attributes, but those documents are
read by neuroglancer and by ngff-zarr's validator, and an unrecognised key is at best
ignored and at worst a validation failure. A sidecar is uniform across precomputed,
zarr and anything added later, and it cannot break a viewer.

Written through ``location.write_json``, so it lands the same way on a local path or an
object store, and it is written **after** the data — a provenance record for a run that
died half way through would be a lie.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The sidecar's name at the destination root.
FILENAME = "provenance.json"


def source_provenance(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Per-backend source facts, or a generic description of the spec.

    Backends that can say something specific do; everything else is described by its
    spec, which already names the location it read.
    """
    backend = spec.get("backend")
    if backend == "dvid":
        from ..backends.dvid import provenance, resolve_node

        # Already resolved by `read_source_metadata`, so this re-reads a concrete node
        # rather than a ref and cannot pick a different one. `requested_ref` carries
        # what was typed, which resolution has by now replaced.
        node = resolve_node(spec)
        node = {**node, "ref": spec.get("requested_ref", node["ref"]),
                "walked": spec.get("ancestors_walked", node["walked"])}
        return provenance(spec, node)
    return {"source": backend, **{k: v for k, v in spec.items() if k != "backend"}}


def build_record(*, src_spec: Mapping[str, Any], dst: str, **run: Any) -> dict[str, Any]:
    """The record for one conversion. ``run`` carries whatever the caller thinks matters."""
    import datetime

    return {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "em-volume-tools",
        "destination": str(dst),
        "source": source_provenance(src_spec),
        "run": {k: v for k, v in run.items() if v is not None},
    }


def write(dst: str, record: Mapping[str, Any]) -> None:
    """Write ``provenance.json`` at ``dst``. Never fails the run.

    A conversion that succeeded and then could not write its sidecar has still produced
    the data, and losing it to a permissions problem on one small file would be a poor
    trade. The failure is logged loudly instead.
    """
    from ..location import write_json

    try:
        write_json(dst, dict(record), FILENAME)
        logger.info("provenance written to %s/%s", str(dst).rstrip("/"), FILENAME)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("could not write %s at %s (%s: %s). The data is fine; only the "
                       "record of where it came from is missing.",
                       FILENAME, dst, type(exc).__name__, exc)
