"""Fetch a webknossos layer's agglomerate views as lookup tables, and describe each view.

One supervoxel volume plus one table per view replaces one volume per view: a table is
``table[supervoxel] = agglomerate`` for every supervoxel id, a few MB, fetched in about
half a minute, and the view is ``table[volume]`` — read on the fly through a descriptor
(``neu_vol/mapped.py``) or converted once chosen. The tables are also data in their own
right: which supervoxels a view merges is exactly what separates an agglomeration error
from a supervoxel error.

Writes, into ``out`` (a local directory or an object-store prefix):

- ``<view>.npy`` — the dense uint64 table, index = supervoxel id;
- ``<view>.json`` — a descriptor over ``volume``, when a supervoxel volume is given;
- ``mappings.json`` — what was fetched, from which dataset, and per-view counts.

Resumable: a view whose table exists is skipped unless ``overwrite``.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

INDEX = "mappings.json"


def fetch_mapping_tables(src: str, out: str, *, volume: str | None = None,
                         mappings: Sequence[str] | None = None, overwrite: bool = False,
                         max_id: int | None = None, threads: int = 4,
                         progress=None) -> dict:
    """Fetch tables for ``mappings`` (default: every agglomerate view) of ``src``.

    ``src`` is ``wk://DATASET/LAYER``. ``volume`` is the supervoxel volume the descriptors
    should read — normally a ``convert`` of the same ``src`` with no mapping. ``max_id``
    overrides the layer's recorded ``largestSegmentId`` as the highest supervoxel id.
    ``threads`` views are fetched at once. ``progress(name, seconds,
    n_agglomerates)`` is called after each view, in the calling thread.
    """
    from .. import mapped, webknossos as wk
    from ..backends.webknossos import geometry
    from ..source_metadata import location_spec, read_source_metadata

    spec = read_source_metadata(location_spec(src, "webknossos"))["provenance_spec"]
    geom = geometry(wk.dataset_info(spec["dataset_id"]), spec["layer"])
    if geom["kind"] != "segmentation":
        raise ValueError(f"{src} is not a segmentation layer")
    largest = max_id if max_id is not None else geom["largest_segment_id"]
    if largest is None:
        raise ValueError(f"{src} records no largestSegmentId; pass the highest supervoxel "
                         f"id explicitly (max_id / --max-id)")
    n_ids = int(largest) + 1

    available = wk.mappings(spec)["agglomerate"]
    wanted = list(mappings) if mappings else available
    unknown = [m for m in wanted if m not in available]
    if unknown:
        raise ValueError(f"{src} has no agglomerate view {', '.join(unknown)}. "
                         f"Available: {', '.join(available) or 'none'}")
    if volume is not None:
        from ..source_metadata import detect_backend

        if detect_backend(volume) is None:
            raise FileNotFoundError(f"--volume {volume} is not a volume")

    # Everything below goes through `location`, so `out` may be a local directory or an
    # object-store prefix. (It used to be os-only, and `--out s3://…` silently became a
    # local directory named `s3:` under the working directory.)
    from ..location import read_json, read_range, write_bytes, write_json

    def at(name: str) -> str:
        return f"{out.rstrip('/')}/{name}"

    index = read_json(at(INDEX)) or {}
    if index.get("dataset_id") not in (None, spec["dataset_id"]) or \
            index.get("layer") not in (None, spec["layer"]):
        raise ValueError(
            f"{out} holds tables for {index.get('dataset_id')}/{index.get('layer')}, "
            f"not {spec['dataset_id']}/{spec['layer']}; use another location")
    origin = {"url": spec["url"], "dataset": spec["dataset"],
              "dataset_id": spec["dataset_id"], "organization": spec.get("organization"),
              "layer": spec["layer"]}
    index.update({**origin, "n_ids": n_ids, "default_mapping": geom["default_mapping"]})
    if volume is not None:
        index["volume"] = mapped.absolute(volume)
    views = index.setdefault("views", {})

    def fetch(name: str) -> tuple[str, dict, float]:
        """One view's table. Runs in a worker thread: no shared state.

        A single object write is atomic on both drivers (tensorstore's file driver writes
        a temporary and renames), so a half-written table never looks done."""
        t = time.time()
        table = wk.agglomerate_table(spec, name, n_ids)
        write_bytes(at(f"{name}.npy"), mapped.table_bytes(table))
        agglomerates = np.unique(table[1:])
        return name, {
            "table": f"{name}.npy",
            "n_agglomerates": int(agglomerates.size - (agglomerates[:1] == 0).sum()),
            "max_agglomerate_id": int(table.max()) if table.size else 0,
            "fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, time.time() - t

    def record(name: str) -> None:
        """Descriptor and index for a finished view — main thread only."""
        if volume is not None:
            mapped.write_descriptor(at(f"{name}.json"), source=volume,
                                    table=f"{name}.npy", mapping=name,
                                    origin={**origin, "n_ids": n_ids})
        # Rewritten per view, so an interrupted fetch still records what it has.
        write_json(at(INDEX), index)

    todo = []
    for name in wanted:
        # One byte, not `exists`: that reads the whole object, and twenty tables is
        # ~360 MB of download just to learn they are already there.
        if not overwrite and read_range(at(f"{name}.npy"), 0, 1) is not None:
            logger.info("%s: table exists, skipped (overwrite to refetch)", name)
            record(name)
        else:
            todo.append(name)

    # Threads, not processes or dask: a view is ~two dozen HTTP requests and the time is
    # the server's, so concurrency is all that helps. Bounded, because the server is
    # usually a shared public service.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        futures = [pool.submit(fetch, name) for name in todo]
        for fut in as_completed(futures):
            name, entry, seconds = fut.result()
            views[name] = entry
            record(name)
            if progress:
                progress(name, seconds, entry["n_agglomerates"])
    return index
