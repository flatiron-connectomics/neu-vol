"""Neuroglancer annotation layers: derived from a volume's occupancy, or authored from
coordinates you supply.

Two entry points, one output. `bboxes-json` asks the volume *where its data is* and gets
one box per written region — for a volume holding a handful of labeled boxes inside a
large empty frame, finding them is the hard part. `annotate-json` takes points, boxes,
lines and ellipsoids you already have (a synapse table, an ROI list) and puts them in the
same kind of layer. Both end at :func:`local_layer`.

Everything here writes **local** annotations, carried inline in the viewer state. For a
set large enough that that stops working — synapses over a whole volume — the answer is
the `neuroglancer_annotations_v1` precomputed format, which is a different piece of
software with a different trade-off; see the em-annotate note in NOTES-TODO.

**The annotations are local, not a precomputed annotation layer, and that is the whole
point.** Neuroglancer builds its annotation list by iterating the layer's source, and
``MultiscaleAnnotationSource`` — the class behind every *precomputed* annotation source
— defines ``[Symbol.iterator]`` as an empty generator. A precomputed annotation layer
therefore renders in the viewport but contributes no rows to the Annotations tab: no
list to click through, and ``[`` / ``]`` do not step. Local annotations, carried inline
in the state, list and navigate. Nothing here writes to the store.

The occupancy boxes come from the volume itself rather than from whatever was written
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

Coordinates are **zyx in memory** throughout, converted to xyz in exactly one place —
:func:`build_annotation` — because that is the conversion no test notices: a mirrored
annotation is a perfectly valid annotation somewhere else in the volume.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any, Sequence

import numpy as np

from ..location import list_keys, read_json
from ..source_metadata import PRECOMPUTED_GZ, detect_backend, existing_levels, level_spec

# Neuroglancer's annotation type strings, used to keep each annotation on one line
# when rendering. From `AnnotationType` in src/annotation/index.ts.
ANNOTATION_TYPES = ("point", "line", "axis_aligned_bounding_box", "ellipsoid",
                    "polyline")

#: Documents that sit beside a zarr array's chunks and are not chunks. v3 writes
#: ``zarr.json``; the v2 names appear on volumes read through the same driver.
ZARR_METADATA = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}

# OME-NGFF spells units out; precomputed always means nm. Anything not here leaves the
# layer unitless, which the caller is warned about — a unitless annotation layer does
# not align with a layer that has physical units.
_UNIT_METRES = {
    "nm": 1e-9, "nanometer": 1e-9, "nanometre": 1e-9,
    "um": 1e-6, "µm": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "m": 1.0, "meter": 1.0, "metre": 1.0,
}

# How a viewer addresses each format. `zarr://` is neuroglancer's zarr driver; the
# `_gz` variant is still precomputed as far as a viewer is concerned.
_LAYER_SCHEME = {"neuroglancer_precomputed": "precomputed", PRECOMPUTED_GZ: "precomputed",
                 "zarr3": "zarr"}


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


# --------------------------------------------------------------------------- #
# neuroglancer JSON
# --------------------------------------------------------------------------- #
def output_dimensions(voxel_size_zyx, units: str | None) -> tuple[dict, str | None]:
    """``{"x": [scale, "m"], ...}`` for the layer, and a warning if units are unknown.

    Declared on the layer itself so its coordinates are interpreted in its own frame
    rather than whatever the viewer happens to be displaying in — which is what lets
    the layer be pasted into any state of the same volume.
    """
    if voxel_size_zyx is None:
        return ({d: [1, ""] for d in "xyz"},
                "no voxel size recorded: the layer is unitless, so it will only line "
                "up if the viewer's own dimensions are voxels. Pass --voxel-size.")
    metres = _UNIT_METRES.get(str(units or "nm").lower())
    if metres is None:
        return ({d: [1, ""] for d in "xyz"},
                f"unrecognised unit {units!r}: the layer is unitless and may not line "
                f"up. Pass --voxel-size to state the size in nm.")
    xyz = tuple(voxel_size_zyx)[::-1]
    return {d: [float(v) * metres, "m"] for d, v in zip("xyz", xyz)}, None


#: The annotation kinds authored here: neuroglancer's own ``type`` string, and the
#: geometry fields each one carries. Both come from ``annotationTypeHandlers`` /
#: ``annotationToJson`` in neuroglancer's ``src/annotation/index.ts``, which is the
#: authority — a bbox is stored as two *corners*, not an origin and a size.
#:
#: ``polyline`` is deliberately absent. It takes an arbitrary number of points per
#: annotation, which one flat CSV row cannot carry, and inventing a grouping column for
#: it would be a format of our own rather than a way of writing theirs.
KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "point": ("point", ("point",)),
    "box": ("axis_aligned_bounding_box", ("pointA", "pointB")),
    "line": ("line", ("pointA", "pointB")),
    "ellipsoid": ("ellipsoid", ("center", "radii")),
}

#: The zyx column names each kind needs, one tuple per geometry field above.
CSV_COLUMNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}

#: Columns any kind may carry. ``segments`` is what links an annotation to bodies, so
#: selecting one in the viewer selects them; neuroglancer writes those ids as *strings*.
OPTIONAL_COLUMNS = ("id", "description", "segments")

#: How many of a kind's leading coordinate groups are *positions*. An ellipsoid's second
#: group is radii — an extent, not a place — so a bounds check must not treat it as one.
POSITION_GROUPS = {"point": 1, "box": 2, "line": 2, "ellipsoid": 1}


def positions(record: dict):
    """The position tuples of a record, skipping any that are extents (zyx)."""
    return record["coords"][:POSITION_GROUPS[record["kind"]]]


def read_annotation_csv(text: str, kind: str, *, source: str = "<csv>") -> list[dict]:
    """Rows of a CSV as annotation records: ``{"kind", "coords", id?, ...}``.

    ``coords`` is one zyx tuple per geometry field of ``kind`` — so a point has one and
    a box has two — in whatever units the file is written in; converting them is
    :func:`rescale`'s job, not this one's.

    Columns are addressed **by name**, never by position: a synapse table has its own
    column order and often extra columns, and silently reading the wrong three numbers
    is the failure this avoids. Unknown columns are ignored.
    """
    import csv
    import io

    if kind not in CSV_COLUMNS:
        raise ValueError(f"unknown annotation kind {kind!r}; known: {sorted(KINDS)}")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{source}: no data rows (a header naming the columns is "
                         f"required)")
    needed = [c for group in CSV_COLUMNS[kind] for c in group]
    present = {(k or "").strip() for k in rows[0]}
    missing = [c for c in needed if c not in present]
    if missing:
        raise ValueError(
            f"{source}: {kind} needs column(s) {', '.join(missing)} — expected "
            f"{', '.join(needed)}{' plus any of ' + ', '.join(OPTIONAL_COLUMNS)}. "
            f"Found: {', '.join(sorted(present)) or '(no header)'}")

    records = []
    for n, row in enumerate(rows, start=2):        # row 1 is the header
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        try:
            coords = tuple(tuple(float(clean[c]) for c in group)
                           for group in CSV_COLUMNS[kind])
        except ValueError as e:
            raise ValueError(f"{source} line {n}: {e}") from None
        rec: dict[str, Any] = {"kind": kind, "coords": coords}
        if clean.get("id"):
            rec["id"] = clean["id"]
        if clean.get("description"):
            rec["description"] = clean["description"]
        if clean.get("segments"):
            rec["segments"] = _parse_segments(clean["segments"], f"{source} line {n}")
        records.append(rec)
    return records


def _parse_segments(value: str, where: str) -> list[str]:
    """Segment ids, however they were separated. Kept as strings, as the state wants."""
    ids = [p for p in value.replace("|", " ").replace(",", " ").split() if p]
    for i in ids:
        # A float here means a spreadsheet turned a 19-digit body id into 1.23e+18, and
        # the annotation would then link to a body that does not exist.
        if not i.isdigit():
            raise ValueError(f"{where}: segment id {i!r} is not a whole number")
    return ids


def rescale(records: list[dict], factor_zyx: Sequence[float]) -> list[dict]:
    """Multiply every coordinate of every record by ``factor_zyx`` (per axis).

    One operation covers both conversions the CLI offers, because both are per-axis
    scalings: voxels at scale N to level-0 voxels (the real per-level ratio), and
    physical nm to level-0 voxels (the reciprocal of the voxel size). Radii scale
    exactly as positions do.
    """
    out = []
    for r in records:
        out.append({**r, "coords": tuple(tuple(c * f for c, f in zip(group, factor_zyx))
                                         for group in r["coords"])})
    return out


def build_annotation(record: dict, ident: str) -> dict:
    """One annotation object, zyx in and **xyz out** — the only place that flips.

    A ``box`` gets its corners sorted per axis: neuroglancer stores two corners with no
    requirement about order, and a reversed pair renders as nothing at all. A ``line``
    is left alone, because for a line the order *is* the direction.
    """
    kind = record["kind"]
    type_name, fields = KINDS[kind]
    coords = list(record["coords"])
    if kind == "box":
        lo, hi = coords
        coords = [tuple(min(a, b) for a, b in zip(lo, hi)),
                  tuple(max(a, b) for a, b in zip(lo, hi))]
    ann: dict[str, Any] = {"type": type_name, "id": ident}
    if record.get("description"):
        ann["description"] = record["description"]
    for field, group in zip(fields, coords):
        ann[field] = [float(v) for v in tuple(group)[::-1]]          # zyx -> xyz
    if record.get("segments"):
        # Related segments: an array per relationship, and a local layer has exactly
        # one. Ids are strings — a uint64 body id does not survive a JSON number.
        ann["segments"] = [[str(s) for s in record["segments"]]]
    return ann


def local_layer(annotations: list[dict], dims: dict, *, name: str = "annotations",
                color: str = "#ffee00") -> dict:
    """The layer envelope for inline (``local://annotations``) annotations.

    Local rather than a precomputed annotation source, for the reason in the module
    docstring: only local annotations appear in the Annotations tab, which is what makes
    them clickable and steppable.
    """
    return {
        "type": "annotation",
        "name": name,
        # opens the layer panel straight onto the clickable list
        "tab": "annotations",
        "source": {"url": "local://annotations",
                   "transform": {"outputDimensions": dims}},
        "annotationColor": color,
        "annotations": annotations,
    }


def annotation_layer(regions: list[dict], dims: dict, *, name: str = "regions",
                     color: str = "#ffee00", kind: str = "box",
                     label: str = "r") -> dict:
    """The occupancy layer: one annotation per region of :func:`labeled_regions`.

    ``regions`` are zyx ``{"lo", "hi"}``; the flip to xyz happens in
    :func:`build_annotation`, as it does for every annotation this module writes.
    """
    annotations = []
    for i, r in enumerate(regions):
        lo, hi = tuple(r["lo"]), tuple(r["hi"])
        ident = f"{label}{i:02d}"
        extent = "x".join(str(hi[a] - lo[a]) for a in range(3))
        note = f"{ident}  {extent} vox"
        if r.get("n_labels") is not None:
            note += f"  {r['n_labels']} labels"
        record = ({"kind": "point",
                   "coords": (tuple((lo[a] + hi[a]) / 2 for a in range(3)),)}
                  if kind == "point" else {"kind": "box", "coords": (lo, hi)})
        annotations.append(build_annotation({**record, "description": note}, ident))
    return local_layer(annotations, dims, name=name, color=color)


def viewer_state(volume: str, fmt: str, layer: dict, regions: list[dict],
                 dims: dict, kind: str | None = None) -> dict:
    """A complete, loadable state: the volume plus the annotation layer."""
    scheme = _LAYER_SCHEME.get(fmt, "precomputed")
    name = volume.rstrip("/").rsplit("/", 1)[-1]
    first = regions[0] if regions else {"lo": (0, 0, 0), "hi": (0, 0, 0)}
    centre = [(first["lo"][a] + first["hi"][a]) / 2 for a in (2, 1, 0)]   # -> xyz
    return {
        "dimensions": dims,
        "position": centre,
        "crossSectionScale": 2,
        "projectionScale": 4096,
        "layers": [
            {"type": "segmentation" if kind == "segmentation" else "image",
             "name": name, "source": f"{scheme}://{volume.rstrip('/')}"},
            layer,
        ],
        "selectedLayer": {"visible": True, "layer": layer["name"]},
        "layout": "4panel",
    }


def render(obj: Any) -> str:
    """``json.dumps`` with each annotation kept on a single line.

    The output of this command is meant to be *pasted* into neuroglancer's JSON editor,
    so it is read by a person: one line per annotation is the difference between
    twelve rows and two hundred.
    """
    holes: dict[str, str] = {}

    def fold(o):
        if isinstance(o, dict):
            if o.get("type") in ANNOTATION_TYPES:
                # Plain ASCII on purpose. json.dumps escapes control characters, so a
                # NUL sentinel is written out as a six-character escape sequence, and
                # the substitution below then misses every one of them silently.
                key = f"__em_vol_annotation_{len(holes)}__"
                holes[key] = json.dumps(o, separators=(", ", ": "))
                return key
            return {k: fold(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fold(v) for v in o]
        return o

    text = json.dumps(fold(obj), indent=1)
    for key, line in holes.items():
        text = text.replace(f'"{key}"', line)
    return text
