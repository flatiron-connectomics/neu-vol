"""Replace label values with background, in a volume that already has them.

Manual segmentation does not always call background 0. A tool that numbers labels from 0
makes it 1, and then a "background" voxel is a perfectly ordinary label as far as anything
here is concerned. Two consequences, and the second is the one that costs:

1. Meshed or skeletonised, background becomes a body — an enormous one.
2. **An all-background block of 1s is not all-fill, so it is stored.** The volume ends up
   with a chunk object everywhere data was written, whether or not it holds anything, and
   "which chunk objects exist" stops answering "where is the data". That question is what
   `bboxes-json`, `relabel`, `downsample --sparse` and neu-morpho's occupancy filter all
   ask, so they all quietly start saying "everywhere".

Which is why the better fix is at ingest — `neu-vol write`/`to-hdf5`/`convert` take
``--background``, and correcting it there happens *before* the storage decision. This op is
for data that has already landed.

**Both destinations restore the sparsity**, which is worth stating because the opposite is
easy to assume: writing zeros over a stored chunk *deletes the object* — measured on both
zarr v3 and precomputed, 8 chunk objects down to 7 after zeroing one block — so in place
fixes the storage as well as the values. (The `--mask-bbox` warning on `convert`/`copy` is
about a different situation: there the block worker returns ``"empty"`` and **never issues
the write**, so nothing removes what was already at that key.)

``--out`` is still the preferred destination, for :mod:`neu_vol.ops.relabel`'s
reason rather than a storage one: a sparse copy is cheap and the original stays as the
record of what was actually annotated.

Only the level given is touched, so the levels above it keep the old values. Single-scale
like ``write`` and ``relabel``, and for the same reason: run
``neu-vol downsample --start-level <level>`` afterwards.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

from blockrun import iter_blocks

from ..backends.base import open_backend
from ..source_metadata import detect_backend, existing_levels, level_spec
from .annotate import labeled_regions

logger = logging.getLogger(__name__)

#: Ceiling on one read. Blocks are whole multiples of the level's chunk, so a write is
#: never a partial-chunk read-modify-write however this is tuned.
DEFAULT_BLOCK_BYTES = 2 * 1024 ** 3


def _block_shape(chunk: Sequence[int], region: Sequence[int], itemsize: int,
                 max_bytes: int) -> tuple[int, ...]:
    """A whole number of chunks per axis, within ``max_bytes`` and inside the region."""
    shape = tuple(int(c) for c in chunk)
    while math.prod(shape) * itemsize > max_bytes and max(shape) > 1:
        axis = int(np.argmax(shape))
        if shape[axis] <= int(chunk[axis]):
            break
        shape = shape[:axis] + (shape[axis] // 2,) + shape[axis + 1:]
    grown = list(shape)
    for axis in range(len(grown)):
        while (grown[axis] + int(chunk[axis]) <= int(region[axis])
               and math.prod(grown) * itemsize * 2 <= max_bytes):
            grown[axis] += int(chunk[axis])
    return tuple(grown)


def plan_mask_values(volume: str, values: Sequence[int], *, out: str | None = None,
                     in_place: bool = False, level: int = 0,
                     to: int = 0) -> dict:
    """What :func:`apply_mask_values` would do. Reads metadata and occupancy only.

    Exactly one of ``out`` / ``in_place``, as in ``relabel``: overwriting a volume is a
    thing to ask for, never a default.
    """
    if bool(out) == bool(in_place):
        raise ValueError("give either out= (a new volume) or in_place=True, not both. "
                         "A new volume is preferred: a sparse copy is cheap and the "
                         "original stays as the record of what was annotated")
    values = [int(v) for v in np.asarray(values).ravel()]
    if not values:
        raise ValueError("no values to replace")
    if to in values:
        raise ValueError(f"{to} is both a value to replace and the replacement")

    volume = volume.rstrip("/")
    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")
    levels = existing_levels(volume, fmt)
    if level not in levels:
        raise ValueError(f"{volume} has no level {level}; present: {sorted(levels)}")

    src = open_backend(level_spec(volume, fmt, level))
    regions, _ctx = labeled_regions(volume, level=level, tighten_level=None, fmt=fmt)
    chunk = tuple(int(c) for c in src.chunks)
    dtype = np.dtype(src.dtype)
    boxes = [{"lo": tuple(int(v) for v in r["lo"]), "hi": tuple(int(v) for v in r["hi"])}
             for r in regions]
    return {
        "volume": volume, "format": fmt, "level": level,
        "destination": volume if in_place else str(out), "in_place": bool(in_place),
        "values": values, "to": int(to), "dtype": str(dtype), "chunk": chunk,
        "shape": tuple(int(s) for s in src.shape), "regions": boxes,
        # Levels above this one keep the old values, exactly as after `write`/`relabel`.
        "stale_levels": sorted(i for i in levels if i > level),
        "n_voxels": sum(math.prod(b - a for a, b in zip(r["lo"], r["hi"]))
                        for r in boxes),
    }


def apply_mask_values(plan: dict, *, dry_run: bool = False, overwrite: bool = False,
                      max_bytes: int = DEFAULT_BLOCK_BYTES) -> dict:
    """Carry out a :func:`plan_mask_values`, returning what it changed.

    Counts are exact rather than sampled, because they are the only evidence that the
    values you named were the values that were there: ``replaced`` of 0 means nothing
    matched, and ``already_background`` above 0 means two labels have just become one.
    """
    volume, fmt, level = plan["volume"], plan["format"], plan["level"]
    values = np.asarray(plan["values"], dtype=np.dtype(plan["dtype"]))
    to = plan["to"]
    src = open_backend(level_spec(volume, fmt, level))
    itemsize = np.dtype(plan["dtype"]).itemsize

    dst = None
    if not dry_run:
        if not plan["in_place"]:
            from .create import create_volume

            create_volume(plan["destination"], like=volume, overwrite=overwrite)
        dst = open_backend(level_spec(plan["destination"], fmt, level))

    replaced = already = written = skipped = 0
    for r in plan["regions"]:
        extent = tuple(b - a for a, b in zip(r["lo"], r["hi"]))
        block = _block_shape(plan["chunk"], extent, itemsize, max_bytes)
        for b in iter_blocks(extent, block):
            region = tuple(slice(r["lo"][a] + s.start, r["lo"][a] + s.stop)
                           for a, s in enumerate(b.region))
            data = src.read_region(region)
            hit = np.isin(data, values)
            n_hit = int(hit.sum())
            already += int((data == to).sum())
            replaced += n_hit
            if n_hit:
                data = data.copy() if not data.flags.writeable else data
                data[hit] = to
            elif plan["in_place"]:
                # Nothing to change, and in place the block is already correct. Writing it
                # back would only risk touching a chunk this op has no business in.
                skipped += 1
                continue
            if not dry_run:
                dst.write_region(region, data)
            written += 1

    result = {**{k: plan[k] for k in ("volume", "destination", "in_place", "level",
                                     "values", "to", "dtype", "stale_levels")},
              "n_regions": len(plan["regions"]), "voxels_replaced": replaced,
              "voxels_already_background": already, "blocks_written": written,
              "blocks_unchanged": skipped, "dry_run": bool(dry_run)}
    if not replaced:
        logger.warning("none of the values %s occur in level %d of %s — nothing to do, "
                       "which usually means the background value is not what was expected",
                       plan["values"], level, volume)
    if already:
        logger.warning("%d voxel(s) already held %d before this ran, so those and the "
                       "%d replaced are now one value. If %d was a real label rather than "
                       "a second background, this merged two labels.",
                       already, to, replaced, to)
    if plan["stale_levels"]:
        logger.warning("levels %s still hold the old values; run `neu-vol downsample "
                       "--start-level %d`", plan["stale_levels"], level)
    return result


def mask_values(volume: str, values: Sequence[int], *, out: str | None = None,
                in_place: bool = False, level: int = 0, to: int = 0,
                dry_run: bool = False, overwrite: bool = False) -> dict:
    """Plan and apply in one call. See :func:`plan_mask_values`."""
    plan = plan_mask_values(volume, values, out=out, in_place=in_place, level=level, to=to)
    return apply_mask_values(plan, dry_run=dry_run, overwrite=overwrite)
