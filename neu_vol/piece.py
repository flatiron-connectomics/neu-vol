"""Read a box out of any source, as a :class:`neu_lib.Piece`.

One function, and it is the answer to "give me these voxels and everything known about
where they are". Every consumer wanted it: neu-glance to host a crop, `write` to place
one, `to-hdf5` to pack one, and a notebook to look at one — and each was assembling the
array, the voxel size and the origin separately, which is three chances to drop the origin
and land the data at nm zero (invariant 1).

    piece = read_piece("gt.h5:/vol_03700")               # frame from the file
    piece = read_piece("s3://bucket/em", level=2, crop=piece)   # the SAME physical box

**The crop may be given three ways, and the physical one is the point.** A voxel box means
nothing outside its own frame — two levels of one volume have different voxel sizes, and a
crop and its parent have different origins — so nanometres are the only thing that
transfers between them. Handing this function another ``Piece`` and getting the same
physical region out of a different source is what makes "show me the image under this
ground-truth crop" a single call.

This lives in neu-vol rather than neu-lib because it opens stores, which is exactly the
line neu-lib draws.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def _crop_request(crop: Any, what: str = "crop"):
    """``(voxel_box, nm_bounds)`` — at most one of them.

    Voxels are resolved here; nanometres cannot be, because converting them needs the
    target level's own voxel size *and* origin, neither known until it is opened.
    """
    if crop is None:
        return None, None
    bounds = getattr(crop, "bounds_nm", None)
    if bounds is not None:
        return None, tuple(bounds)
    if isinstance(crop, Mapping):
        if "nm" not in crop:
            raise ValueError(f"{what} mapping must carry 'nm': (lo, hi); got {crop!r}")
        return None, tuple(crop["nm"])
    flat = tuple(crop)
    if len(flat) == 6 and all(isinstance(v, (int, float)) for v in flat):
        lo, hi = tuple(int(v) for v in flat[:3]), tuple(int(v) for v in flat[3:])
    elif len(flat) == 2:
        lo, hi = tuple(int(v) for v in flat[0]), tuple(int(v) for v in flat[1])
    else:
        raise ValueError(
            f"{what} must be ((z0,y0,x0), (z1,y1,x1)) or (z0,y0,x0,z1,y1,x1) in whole "
            f"VOXELS of the level being read, a Piece to take the same physical box as, "
            f"or {{'nm': (lo, hi)}} — got {crop!r}")
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError(f"{what} corners are zyx, so 3 values each; got {lo} / {hi}")
    return (lo, hi), None


def read_piece(src: str | Mapping[str, Any], kind: str | None = None, *,
               level: int = 0, crop: Any = None, dataset: str | None = None,
               src_format: str | None = None,
               voxel_size: Sequence[float] | None = None,
               backend: Any = None):
    """The voxels at ``src``, with the frame that says where they are.

    ``src`` is ``PATH`` or ``PATH:/DATASET`` — the trailing form selects an array inside an
    HDF5 container, and **only a leading slash** makes it one, so a scheme's own colon
    (``s3://…``) is left alone. A spec mapping or an open ``backend=`` is accepted too.

    ``crop`` is a box in whole voxels of the level being read (``((lo), (hi))`` or the flat
    six), **or** anything carrying ``bounds_nm`` — another :class:`~neu_lib.Piece` — meaning
    *the same physical box as that*, **or** ``{"nm": (lo, hi)}``. A physical box is
    converted using this level's own voxel size and origin and grown outward, so the read
    contains what was asked for rather than dropping a face where the levels do not divide
    evenly.

    ``kind`` overrides what the source records. Where the source records nothing it stays
    ``None`` and is **not** guessed: a uint8 label array is indistinguishable from an image
    by dtype, and getting it wrong averages label ids into ids that were never in the data.

    ``voxel_size`` overrides the recorded scale, and is required for a source that records
    none — a slice stack, or a plain HDF5 file.

    Store logging is filtered for the duration (``logs.quiet_reads``): this is a notebook
    entry point, and an S3 open emits two benign ``AuthCredentialsProvider`` lines per
    prefix.
    """
    from neu_lib import Frame, Piece

    from .backends.base import open_backend
    from .logs import quiet_reads
    from .source_metadata import (describe, level_spec, location_spec,
                                  read_level_voxel_sizes, read_source_metadata,
                                  require_one_array)

    box, box_nm = _crop_request(crop)
    if isinstance(src, Mapping):
        spec, path, fmt = dict(src), str(src.get("path") or src.get("source") or src), None
        fmt = src.get("backend")
        meta_from_spec = True
    else:
        # The LAST colon, and a tail of `//…` is a scheme's own, not a dataset. Splitting
        # on the first colon made `s3://bucket/x` into path `s3` + dataset `//bucket/x`,
        # which starts with a slash and so passed a naive leading-slash test.
        path, _, tail = str(src).rpartition(":")
        # `len(path) > 1` also rules out a Windows drive letter, cheaply.
        if not tail.startswith("/") or tail.startswith("//") or len(path) < 2:
            path, tail = str(src), ""
        dataset = dataset or (tail or None)
        spec, fmt, meta_from_spec = None, src_format, False

    with quiet_reads():
        if backend is not None:
            spec = dict(backend.to_spec())
            fmt = fmt or spec.get("backend")
            dataset = spec.get("dataset") or dataset
            meta = read_source_metadata(spec) or {}
        elif meta_from_spec:
            meta = read_source_metadata(spec) or {}
        elif fmt is None:
            described = describe(path, dataset=dataset or None)
            if described["shape"] is None:
                require_one_array(described, path, "read_piece")
            fmt, spec = described["format"], described["spec"]
            meta = described["meta"] or {}
            dataset = described.get("dataset") or dataset
        else:
            # A format named outright, for a file whose name detection would not recognise.
            spec = location_spec(path, fmt, dataset=dataset or None)
            meta = read_source_metadata(spec) or {}
            dataset = spec.get("dataset") or dataset

        single = fmt in ("hdf5", "image_stack")
        if level and single:
            raise ValueError(
                f"level {level} needs a multiscale volume; {path} is {fmt}, a single array")

        # **A zarr OME group is not an array**, so level 0 goes through the metadata's own
        # `data_spec` (which names the level's subdirectory) rather than the group path.
        # Addressing the path directly fails to open at all — the trap `ops/pack.py`
        # documents — and it hides, because precomputed selects a scale with `scale_index`
        # on one path and zarr level 1 does get a subdirectory. Only zarr level 0 breaks.
        if backend is not None or single or meta_from_spec:
            read_spec = spec
        elif level:
            read_spec = level_spec(path, fmt, level, dataset=dataset or None)
        elif meta.get("data_spec"):
            read_spec = meta["data_spec"]
        else:
            read_spec = spec          # a bare array: it has no levels to descend into
        source = backend or open_backend(read_spec)
        shape = tuple(int(s) for s in source.shape)

        per_level = read_level_voxel_sizes(spec) or []
        voxel = (tuple(float(v) for v in voxel_size) if voxel_size
                 else (tuple(per_level[level]) if level < len(per_level) else None)
                 or (tuple(meta["voxel_size"]) if meta.get("voxel_size") else None))
        if voxel is None:
            raise ValueError(
                f"{path} records no voxel size, so nothing here knows its physical scale; "
                f"pass voxel_size=(z, y, x)")

        channel_axis = len(shape) == 4
        spatial = shape[1:] if channel_axis else shape
        recorded = (tuple(float(o) for o in meta["offset"])
                    if meta.get("offset") else (0.0, 0.0, 0.0))

        if box_nm is not None:
            box = _physical_to_voxels(box_nm, Frame(voxel_size_nm=voxel,
                                                   origin_nm=recorded),
                                      spatial, path=path, level=level)

        lo = (0, 0, 0)
        if box:
            lo, hi = box
            for axis, (start, stop, extent) in enumerate(zip(lo, hi, spatial)):
                if not 0 <= start < stop <= extent:
                    hint = (" — a crop in NANOMETRES looks like this"
                            if any(v > e * 4 for v, e in zip(hi, spatial)) else "")
                    raise ValueError(
                        f"crop {lo}:{hi} does not fit {path}'s level-{level} extent "
                        f"{spatial} on axis {axis}. The box is in whole VOXELS of that "
                        f"level, zyx, half-open{hint}")
            region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        else:
            region = tuple(slice(0, int(s)) for s in spatial)
        if channel_axis:
            region = (slice(0, shape[0]),) + region

        # A crop out of something that already knows where it belongs lands at the SUM, the
        # same rule `neu-vol to-hdf5 --crop-bbox` follows.
        origin = tuple(o + i * v for o, i, v in zip(recorded, lo, voxel))
        return Piece(array=source.read_region(region),
                     frame=Frame(voxel_size_nm=voxel, origin_nm=origin),
                     kind=kind or meta.get("kind"))


def _physical_to_voxels(bounds_nm, frame, spatial, *, path: str, level: int):
    """A physical box as a voxel box in ``frame``, clamped to ``spatial``.

    **A clamp that removes most of the box means the wrong volume**, not an edge case.
    Taking a physical box off one dataset's crop and reading it out of another's image is
    the easy mistake: the numbers are plausible, the read succeeds, and what comes back is
    a thin slab nobody asked for. So the fraction is reported.
    """
    box = frame.voxel_box(bounds_nm)
    clamped = (tuple(max(0, v) for v in box.lo),
               tuple(min(e, v) for v, e in zip(box.hi, spatial)))
    logger.info("crop %s nm -> level-%d voxels %s:%s", bounds_nm, level, *clamped)
    if any(b <= a for a, b in zip(*clamped)):
        raise ValueError(
            f"the physical box {bounds_nm} nm does not overlap {path}'s level-{level} "
            f"extent {spatial} at {frame.voxel_size_nm} nm/voxel starting "
            f"{frame.origin_nm} nm. If the box came from another dataset's crop, the two "
            f"are not the same volume")
    asked = math.prod(b - a for a, b in zip(box.lo, box.hi))
    got = math.prod(b - a for a, b in zip(*clamped))
    if got < asked:
        where = [f"{'zyx'[a]} {box.lo[a]}:{box.hi[a]} -> {clamped[0][a]}:{clamped[1][a]}"
                 for a in range(3)
                 if (box.lo[a], box.hi[a]) != (clamped[0][a], clamped[1][a])]
        mostly = got * 2 < asked
        (logger.warning if mostly else logger.info)(
            "the physical box does not fit %s's level-%d extent %s and was clipped to "
            "%.0f%% of it (%s)%s", path, level, spatial, 100.0 * got / asked,
            "; ".join(where),
            ". Losing most of a box usually means it came from a different dataset than "
            "this volume" if mostly else "")
    return clamped
