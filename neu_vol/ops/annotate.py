"""Where a sparse volume's data actually is: occupied chunks, covered by maximal boxes.

**This is analysis, not presentation.** It answers "which regions of this volume hold
anything" and returns boxes in level-0 voxels. Turning those into a neuroglancer layer is
`neu-glance bboxes`, in a package that sits above this one — neu-vol knows nothing about
viewers. `ops.relabel`, `ops.maskvalue` and `ops._multiscale` (the ``--sparse`` path) are the
other callers, and they want the boxes rather than a layer.

The boxes come from the volume itself rather than from whatever was written
into it, so they cannot drift from the data. Three steps, cheapest first:

1. **Which chunk objects exist.** TensorStore never persists an all-fill chunk, so the
   set of present keys *is* the occupied footprint. No voxel reads at all.
2. **Cover the occupied cells with maximal boxes.** Connected components would be
   wrong: two regions written face to face merge into one component whenever their
   chunk-aligned footprints touch, even with a real gap in the data between them.
3. **Tighten each box to its nonzero voxels at a coarse level.** A 384-voxel level-0
   box is 96 voxels at 32 nm, so this is nearly free, at the cost of quantizing the
   answer to one coarse voxel — which is why extents come back as 252 or 256 rather
   than a uniform 256. Raise ``tighten_level`` for cheaper, ``0`` for exact.

Coordinates are **zyx** throughout, as everywhere in this package. The flip to xyz belongs to
whatever presents them, which is `neu-glance`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as np

from ..location import list_keys, read_json
from ..source_metadata import PRECOMPUTED_GZ, detect_backend, existing_levels, level_spec

#: Documents that sit beside a zarr array's chunks and are not chunks. v3 writes
#: ``zarr.json``; the v2 names appear on volumes read through the same driver.
ZARR_METADATA = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}


class NoOccupancy(RuntimeError):
    """No chunk key under the level's prefix could be read as a chunk index."""


# --------------------------------------------------------------------------- #
# occupancy: which chunks exist
# --------------------------------------------------------------------------- #
def _precomputed_cell(key: str, cell: tuple[int, ...]) -> tuple[int, ...] | None:
    """``x0-x1_y0-y1_z0-z1`` -> zyx cell index.

    A CloudVolume-written volume appends ``.gz`` to every chunk key, which is exactly
    the kind of volume most likely to be sparse — strip it rather than failing to
    parse the whole level.
    """
    name = key.rsplit("/", 1)[-1]
    if name.endswith(".gz"):
        name = name[:-3]
    parts = name.split("_")
    if len(parts) != 3:
        return None
    try:
        starts_xyz = [int(p.split("-")[0]) for p in parts]
    except ValueError:
        return None
    # keys are xyz, cell shape is zyx
    return tuple(starts_xyz[a] // cell[2 - a] for a in (2, 1, 0))


def _zarr_cell(key: str, ndim: int) -> tuple[int, ...] | None:
    """``c/0/1/2`` (or ``c.0.1.2``) -> zyx cell index.

    zarr v3 permits either separator. A leading channel axis shows up as one extra
    index, which is dropped: this reports spatial occupancy.
    """
    if not (key.startswith("c/") or key.startswith("c.")):
        return None
    parts = key[2:].replace(".", "/").split("/")
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        return None
    if len(idx) == ndim + 1:
        idx = idx[1:]
    return tuple(idx) if len(idx) == ndim else None


def occupied_cells(volume: str, fmt: str, level: int,
                   cell: tuple[int, ...]) -> set[tuple[int, ...]]:
    """The zyx chunk-grid cells that have a stored object at ``level``.

    This is the only listing the whole operation does, and it is what makes the result
    trustworthy on a sparse volume: an all-fill chunk is never written, so presence and
    occupancy are the same question.
    """
    if fmt == "zarr3":
        # The array's own metadata document is not a chunk and must not count towards the
        # "objects exist but none are chunk keys" test below — an EMPTY level holds
        # exactly `zarr.json`, and letting that through reported a freshly created volume
        # as sharded.
        keys = [k for k in list_keys(volume, str(level))
                if k.rsplit("/", 1)[-1] not in ZARR_METADATA]
        cells = {c for c in (_zarr_cell(k, len(cell)) for k in keys) if c}
    else:
        info = read_json(volume.rstrip("/") + "/info") or {}
        scales = sorted(info.get("scales", []), key=lambda s: tuple(s["resolution"]))
        if level >= len(scales):
            raise NoOccupancy(f"{volume} has no scale {level}")
        keys = list_keys(volume, scales[level]["key"])
        cells = {c for c in (_precomputed_cell(k, cell) for k in keys) if c}
    if keys and not cells:
        raise NoOccupancy(
            f"{len(keys)} objects under level {level} of {volume}, none of them a "
            f"chunk key. A SHARDED level looks like this — occupancy comes from chunk "
            f"presence, and a shard hides which of its chunks exist. Try a coarser "
            f"--level, which is often unsharded.")
    return cells


def maximal_boxes(cells: set[tuple[int, ...]]) -> list[tuple[tuple, tuple]]:
    """Cover a set of grid cells with axis-aligned boxes, as ``(lo, hi)`` cell ranges.

    Each pass takes the lexicographically smallest remaining cell and grows it along
    each axis in turn while every cell of the candidate box is still present, then
    removes it. For a footprint that is a union of a few written blocks this recovers
    the blocks; in the worst case it degrades to one box per cell. It can never emit a
    box containing an absent cell, which is the property that matters — a box is a
    claim that there is data there.

    Connected components would be simpler and wrong: two blocks written face to face
    share a chunk boundary and merge into a single region spanning both, plus the empty
    corner between them.
    """
    left, out = set(cells), []
    while left:
        lo = min(left)
        hi = list(lo)
        for axis in range(len(lo)):
            while True:
                nxt = list(hi)
                nxt[axis] += 1
                spans = [range(lo[a], nxt[a] + 1) for a in range(len(lo))]
                if all(c in left for c in _grid(spans)):
                    hi = nxt
                else:
                    break
        box = list(_grid([range(lo[a], hi[a] + 1) for a in range(len(lo))]))
        left -= set(box)
        out.append((tuple(lo), tuple(h + 1 for h in hi)))
    return sorted(out)


def _grid(spans):
    return itertools.product(*spans)


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #
def _factor(voxel_sizes, level: int) -> tuple[int, ...]:
    """Level-0 voxels per ``level`` voxel, per axis.

    Read from the recorded per-level voxel sizes rather than assumed to be
    ``2**level``: real pyramids are anisotropic, and shape ratios are ceil-divided and
    do not divide exactly.
    """
    if voxel_sizes is None:
        if level:
            raise ValueError(
                f"level {level} needs the per-level voxel sizes to convert back to "
                f"level-0 coordinates, and this volume records none; use level 0")
        return (1, 1, 1)
    fine, coarse = voxel_sizes[0], voxel_sizes[level]
    out = []
    for f, c in zip(fine, coarse):
        r = c / f
        if abs(r - round(r)) > 1e-6:
            raise ValueError(f"level {level} is {r:g}x level 0 on one axis, "
                             f"which is not a whole number of voxels")
        out.append(int(round(r)))
    return tuple(out)


def labeled_regions(volume: str, *, level: int = 0, tighten_level: int | None = 2,
                    fmt: str | None = None) -> tuple[list[dict], dict]:
    """Occupied boxes of ``volume``, in **level-0 voxels, zyx**, plus a context dict.

    Each region is ``{"lo", "hi", "cells", "n_labels"}``; ``n_labels`` is the number of
    distinct nonzero values seen at ``tighten_level`` and is ``None`` when tightening is
    off. It is a floor, not a count — a label too small to survive to that level is
    invisible there.
    """
    from ..backends.base import open_backend
    from ..source_metadata import read_level_voxel_sizes

    volume = volume.rstrip("/")
    fmt = fmt or detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")
    # Only the chunk KEYS differ between the two precomputed flavours, and level
    # geometry comes from `info`, which is the same document either way.
    open_fmt = "neuroglancer_precomputed" if fmt == PRECOMPUTED_GZ else fmt

    levels = existing_levels(volume, open_fmt)
    if level not in levels:
        # Strict, unlike tightening below: occupancy at another level is an answer to
        # a different question, and silently substituting one would be misreported.
        raise ValueError(f"{volume} has no level {level} "
                         f"(present: {sorted(levels)})")
    # Tightening is an optimisation — a coarse read standing in for an exact one — so a
    # missing level is not an error, it just means the shortcut isn't available. A
    # single-level volume (what `create` makes, and what `write` fills) has only level
    # 0, and refusing to annotate it because level 2 is absent would be absurd.
    # Clamping goes FINER, so the result is more exact and only slower, bounded by the
    # occupied footprint. Recorded in the context so the caller can say it happened.
    tighten_asked, deepest = tighten_level, max(levels)
    if tighten_level is not None:
        tighten_level = min(tighten_level, deepest)
    cell = levels[level]["chunks"]
    if not cell:
        raise ValueError(f"level {level} of {volume} does not report a chunk shape")

    voxel_sizes = read_level_voxel_sizes({"backend": open_fmt, "path": volume})
    step = _factor(voxel_sizes, level)

    cells = occupied_cells(volume, fmt, level, cell)
    boxes = maximal_boxes(cells)
    context = {"format": fmt, "levels": levels, "cell": cell, "level": level,
               "tighten_level": tighten_level, "voxel_sizes": voxel_sizes,
               "n_chunks": len(cells),
               "tighten_clamped_from": (tighten_asked
                                        if tighten_asked != tighten_level else None)}
    if not boxes:
        return [], context

    tight_step = (_factor(voxel_sizes, tighten_level)
                  if tighten_level is not None else None)
    coarse = (open_backend(level_spec(volume, open_fmt, tighten_level))
              if tighten_level is not None else None)

    regions = []
    for lo_cell, hi_cell in boxes:
        lo = tuple(lo_cell[a] * cell[a] * step[a] for a in range(3))
        hi = tuple(hi_cell[a] * cell[a] * step[a] for a in range(3))
        n_cells = math.prod(hi_cell[a] - lo_cell[a] for a in range(3))
        if coarse is None:
            regions.append({"lo": lo, "hi": hi, "cells": n_cells, "n_labels": None})
            continue
        c_lo = [lo[a] // tight_step[a] for a in range(3)]
        c_hi = [-(-hi[a] // tight_step[a]) for a in range(3)]
        data = coarse.read_region(tuple(slice(c_lo[a], c_hi[a]) for a in range(3)))
        nz = data.nonzero()
        if not len(nz[0]):
            # Chunks exist but hold nothing at this level: either genuinely all-fill
            # objects, or a region too thin to survive the coarsening. Keep the box —
            # dropping it would hide data — but say so.
            regions.append({"lo": lo, "hi": hi, "cells": n_cells, "n_labels": 0})
            continue
        tlo = tuple((c_lo[a] + int(nz[a].min())) * tight_step[a] for a in range(3))
        thi = tuple((c_lo[a] + int(nz[a].max()) + 1) * tight_step[a] for a in range(3))
        regions.append({"lo": tlo, "hi": thi, "cells": n_cells,
                        "n_labels": int(np.unique(data[data != 0]).size)})
    return regions, context
