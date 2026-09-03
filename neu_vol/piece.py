"""Read a box out of any source as a :class:`neu_lib.Piece`, and write one back.

Two functions, and between them they are the answer to "give me these voxels and everything
known about where they are" and "put them back". Every consumer wanted the first:
neu-glance to host a crop, `write` to place one, `to-hdf5` to pack one, and a notebook to
look at one — and each was assembling the array, the voxel size and the origin separately,
which is three chances to drop the origin and land the data at nm zero (invariant NM-SPACE).

    piece = read_piece("gt.h5:/vol_03700")               # frame from the file
    piece = read_piece("s3://bucket/em", level=2, crop=piece)   # the SAME physical box
    write_piece(cleaned, "gt_cleaned.h5")                # ...and back out again

:func:`write_piece` closes that loop for an array **already in memory**, which is the shape
of every in-process workflow: read a crop, transform it (``piece.apply``, neu-proc), write
the result. Before it, the only way out was :func:`~neu_vol.pack_hdf5`, which reads from a
*location* — so an in-memory result had to be written to a temporary file first in order to
be written to a file. It shares `ops/pack.py`'s layout helpers rather than reimplementing
them, so the two produce the same file and `neu-vol write` can place either with no
arguments.

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

#: How much a single read may pull into memory before it has to be asked for explicitly.
#: **This exists because the alternative is a hang, not an error.** A whole level 0 of a
#: production EM volume is terabytes (measured on one: 11260x9000x13750 = 1.27 TiB), and
#: `read_piece(volume)` with no crop would sit there reading it — which in a notebook looks
#: exactly like a wedged kernel, with every later cell pending behind it. Nothing about the
#: call says "this is 1.27 TiB", so the function has to.
DEFAULT_MAX_BYTES = 4 * 1024 ** 3


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
               src_format: str | None = None, dtype: str | type | None = None,
               voxel_size: Sequence[float] | None = None,
               name: str | None = None,
               max_bytes: int | None = DEFAULT_MAX_BYTES,
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

    ``dtype`` casts the voxels **after** the read, so a piece can arrive in the dtype it is
    going to be used in — the destination volume's, usually, since a set of crops exported
    by different tools arrives as uint8/uint16/uint32/uint64 and a consumer that has to
    handle all four is the thing this avoids. A **narrowing** cast is warned about rather
    than refused (``--cast`` on `neu-vol write` sets the same precedent): widening among
    unsigned ints loses nothing, while narrowing wraps label ids into other label ids, which
    is silent and unrecoverable. ``max_bytes`` governs the *read*, which is what can hang,
    so a cast to a wider dtype costs memory the cap did not count.

    The piece is **named after the source** — ``stem`` or ``stem/dataset``
    (:func:`piece_name`) — so a consumer three calls away can still say what it is looking
    at. ``name`` overrides that, for when the derived one is longer than it needs to be;
    ``Piece.with_name`` renames one afterwards.

    ``max_bytes`` caps what one call will pull into memory, and **refusing is the whole
    point**: a whole production level 0 is terabytes, so reading it does not fail, it
    *hangs* — which in a notebook is indistinguishable from a wedged kernel. The error names
    the size and which level would fit. ``None`` lifts the cap for a caller who means it.

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
        if max_bytes is not None:
            _refuse_if_too_big(region, source.dtype, max_bytes, path=path, level=level,
                               per_level=per_level, spatial=spatial, cropped=bool(box))

        # A crop out of something that already knows where it belongs lands at the SUM, the
        # same rule `neu-vol to-hdf5 --crop-bbox` follows.
        origin = tuple(o + i * v for o, i, v in zip(recorded, lo, voxel))
        array = source.read_region(region)
        if dtype is not None:
            array = _cast(array, dtype, path=path, dataset=dataset)
        return Piece(array=array,
                     frame=Frame(voxel_size_nm=voxel, origin_nm=origin),
                     kind=kind or meta.get("kind"),
                     name=name or piece_name(path, dataset))


def _cast(array, dtype, *, path: str, dataset: str | None):
    """``array`` as ``dtype``, saying so when the cast can change the values.

    Warned about, not refused, for the reason `neu-vol write --cast` exists: reading a
    piece deliberately narrower is legitimate, and only the caller knows the range their
    labels actually use. What is not legitimate is *not being told* — a uint64 id narrowed
    to uint32 comes back as a different, perfectly plausible id, which no later check can
    catch because a segmentation has no invalid values.

    ``np.can_cast``'s ``"safe"`` rule, so it costs nothing: it compares the two dtypes, not
    the voxels, and a full-volume value check on every read is not a trade worth making.
    """
    import numpy as np

    want = np.dtype(dtype)
    if array.dtype == want:
        return array
    if not np.can_cast(array.dtype, want, "safe"):
        logger.warning(
            "reading %s%s as %s from %s is not a safe cast: a value that does not fit "
            "wraps to a different one, and for label ids that is silent — the result is "
            "another plausible id. Widening (uint32 -> uint64) is always safe",
            path, f":{dataset}" if dataset else "", want, array.dtype)
    return array.astype(want)


def _host_block(block):
    """One block of a piece's array as a host numpy array.

    **Not** ``piece.to_numpy()`` once up front: a ``Piece`` may hold a *lazy* array — a dask
    or zarr array, an open h5py dataset — and converting the whole thing would materialise
    exactly what a blocked write exists to avoid. **Not** ``np.asarray`` alone either: cupy
    makes ``__array__`` raise on purpose, so a device array needs its own ``.get()``. Same
    rule as ``neu_lib``'s own conversion, and checked by module rather than by catching
    numpy's ``TypeError``, which also emits a spurious ``__array__`` deprecation warning
    that would reach the caller looking like a bug in their code.
    """
    import numpy as np

    if type(block).__module__.split(".")[0] == "cupy":
        return np.asarray(block.get())
    return np.asanyarray(block)


def write_piece(piece, out: str, *, dataset: str | None = None,
                dtype: Any = None, chunk: Sequence[int] | None = None,
                compression: str | None = "gzip", overwrite: bool = False,
                dry_run: bool = False,
                voxel_size_field: str | None = None,
                offset_field: str | None = None,
                block_bytes: int | None = None) -> dict:
    """Write a :class:`neu_lib.Piece` to an HDF5 file, frame and position included.

    The inverse of :func:`read_piece`, and the round trip is the point::

        piece = read_piece("gt_v1_eval.h5:/z07901")
        cleaned = piece.apply(dust).apply(dilate)
        write_piece(cleaned, "gt_v1_eval_cleaned.h5")    # -> /z07901, same frame

    **Four arguments `pack_hdf5` needs are not here, because the piece answers them.**
    ``voxel_size`` is ``piece.frame``'s; ``voxel_offset`` is ``piece.origin_voxel``, which
    *raises* rather than rounding when the origin is not a whole number of voxels, since
    rounding would shift the piece against whatever it is meant to line up with; ``axes`` is
    always ``zyx`` and ``units`` always ``nm``, both by construction of the type (invariants
    ZYX-XYZ and NM-SPACE). So there is no axis-order question on this path at all — the file
    records ``axes="zyx"`` and ``neu-vol write`` reads it rather than assuming.

    ``kind`` travels too, when the piece has one, and that is what makes the round trip
    lossless: a cleaned segmentation read back is still a segmentation, where before it came
    back as ``None`` and the next thing to coarsen it would have been free to average label
    ids. Use ``piece.with_kind(...)`` to change it.

    ``dataset`` defaults to **the last component of ``piece.name``** when that name has one
    (``"gt_eval/z07901"`` -> ``/z07901``), which is what makes a bag of crops
    round-trip through a whole cleaning pass with no arguments — :func:`read_piece` names a
    piece after its source, so the names are already there. A piece named after a volume
    rather than an array inside one has no such component and gets
    :data:`~neu_vol.ops.pack.DEFAULT_DATASET`. Pass ``dataset`` for anything else, including
    a nested path, which the derived name does not preserve.

    An existing file is **added to** when its recorded frame matches, which is how a set of
    cleaned crops accumulates into one file; a frame that disagrees, and a dataset name
    already in use without ``overwrite``, are refused rather than guessed
    (:func:`~neu_vol.ops.pack.check_hdf5_target`).

    Writing is **blocked**, so a piece holding a lazy array streams rather than being
    materialised whole. ``dry_run`` returns the same plan dict without touching the file.

    ``out`` must be a local path with an HDF5 extension — h5py has no object-store driver,
    and detection goes by name, so a file written under any other name would be unreadable
    by everything here that has to *recognise* it. To place a piece into an existing volume
    instead, write it here and then ``neu-vol write <volume> --src <file>``, which needs no
    offset because this recorded one.
    """
    import numpy as np

    from blockrun import iter_blocks

    from .backends.hdf5 import HDF5_EXTENSIONS, require_local_path
    from .logs import quiet_reads
    from .ops.pack import (DEFAULT_BLOCK_BYTES, DEFAULT_OFFSET_FIELD,
                           DEFAULT_VOXEL_SIZE_FIELD, _block_shape, check_hdf5_target,
                           hdf5_dataset_name, write_hdf5_array)

    voxel_size_field = voxel_size_field or DEFAULT_VOXEL_SIZE_FIELD
    offset_field = offset_field or DEFAULT_OFFSET_FIELD

    out = require_local_path(out, "the HDF5 file to write")
    if not out.lower().endswith(HDF5_EXTENSIONS):
        raise ValueError(
            f"{out!r} does not end in one of {', '.join(HDF5_EXTENSIONS)}, and this writes "
            f"an HDF5 file. The extension is not cosmetic: HDF5 has no marker object, so "
            f"`describe`, `neu-vol info` and `neu-vol write` recognise a container BY NAME "
            f"and would not recognise this one. To write into a volume instead, write the "
            f"piece to an .h5 and place it with `neu-vol write <volume> --src <file>`")

    # A piece is zyx nanometres by construction; those are the two invariants that make the
    # arguments `pack_hdf5` has to ask for unnecessary here.
    axes = ("z", "y", "x")
    units = "nm"
    voxel_size = tuple(float(v) for v in piece.voxel_size_nm)
    voxel_offset = tuple(int(v) for v in piece.origin_voxel)

    name = hdf5_dataset_name(dataset if dataset is not None
                             else _dataset_from_piece_name(piece.name))
    shape = piece.shape
    out_dtype = np.dtype(dtype or piece.dtype)
    frame = {"voxel_size": voxel_size, "units": units, "axes": axes}

    _existing, existing_datasets, replacing = check_hdf5_target(
        out, name, frame, overwrite=overwrite, voxel_size_field=voxel_size_field)
    others = [d for d in existing_datasets if d != name]
    block = _block_shape(shape, out_dtype.itemsize, block_bytes or DEFAULT_BLOCK_BYTES)
    plan = {
        "out": out, "dataset": name, "piece": piece.name, "shape": shape,
        "dtype": str(out_dtype), "has_channels": piece.channel_axis,
        "voxel_size": voxel_size, "voxel_offset": voxel_offset, "units": units,
        "axes": axes, "kind": piece.kind, "chunk": tuple(chunk) if chunk else None,
        "compression": compression, "nbytes": float(out_dtype.itemsize) * math.prod(shape),
        "blocks": len(list(iter_blocks(shape, block))), "block_shape": block,
        "replacing": replacing, "appending": bool(existing_datasets) and not replacing,
        "other_datasets": others,
    }
    if dry_run:
        return plan

    if others:
        # INFO, where `pack_hdf5` warns: a file of many datasets is what this path is FOR —
        # a bag of crops written back one at a time — so warning would fire once per crop on
        # a normal cleaning pass, twelve times over for a thirteen-crop set. The fact stays
        # on the record, and a reader that later needs a name gets `sole_dataset`'s error,
        # which lists them.
        logger.info("%s now holds %d volumetric datasets (%s); readers must name one "
                    "— `neu-vol write --dataset %s`, or HDF5Backend's `dataset` key",
                    out, len(others) + 1, ", ".join(others + [name]), name)

    array = piece.array
    # `quiet_reads` for the same reason `read_piece` uses it: this is a notebook entry
    # point, and a lazy array may reach across a store to fill each block.
    with quiet_reads():
        write_hdf5_array(out, name, lambda region: _host_block(array[region]),
                         shape=shape, dtype=out_dtype, voxel_size=voxel_size,
                         voxel_offset=voxel_offset, units=units, axes=axes,
                         kind=piece.kind, chunk=chunk, compression=compression,
                         replacing=replacing, block=block,
                         has_channels=piece.channel_axis,
                         voxel_size_field=voxel_size_field, offset_field=offset_field)
    logger.info("wrote %s -> %s%s (%s %s, %d block(s))", piece.name or "piece", out, name,
                shape, out_dtype, plan["blocks"])
    return plan


def _dataset_from_piece_name(name: str | None) -> str | None:
    """The array half of a piece's name, or ``None`` if it names no array.

    :func:`piece_name` builds ``stem/inner`` for an array inside a container and a bare
    ``stem`` for a volume, so the slash is exactly the signal that the source had a dataset
    name of its own worth reusing. Without one, the caller gets the default dataset rather
    than a file whose single array is named after a volume directory.
    """
    import posixpath

    if not name or "/" not in str(name):
        return None
    return posixpath.basename(str(name).strip("/")) or None


#: Suffixes stripped when naming a piece after its source. A volume's directory name is
#: the useful part; the format marker is not.
_NAME_SUFFIXES = (".h5", ".hdf5", ".hdf", ".he5", ".zarr", ".precomputed", ".n5")


def piece_name(path: str, dataset: str | None = None) -> str:
    """What to call a piece read from ``path`` — ``stem`` or ``stem/dataset``.

    Both halves, because either alone is ambiguous in a way that bites: two crops from
    different files share a dataset name (``/data`` is what `to-hdf5` writes by default),
    and one file's nine crops share a stem. Serving two of either is ordinary, and
    neuroglancer keys a layer by name, so a collision is a collision rather than a
    duplicate.
    """
    import os
    import posixpath

    stem = os.path.basename(str(path).rstrip("/")) or str(path)
    for suffix in _NAME_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    inner = posixpath.basename(str(dataset).strip("/")) if dataset else ""
    return f"{stem}/{inner}" if inner else (stem or "piece")


def _size(nbytes: float) -> str:
    """Bytes as the unit a human would use, so a small cap does not read as "0.0 GiB"."""
    for unit, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if nbytes >= scale:
            return f"{nbytes / scale:,.1f} {unit}"
    return f"{nbytes:.0f} B"


def _refuse_if_too_big(region, dtype, max_bytes: int, *, path: str, level: int,
                       per_level, spatial, cropped: bool) -> None:
    """Raise before reading, naming the size and a level that would fit.

    Before, not after: the read is the thing that takes forever, and a message that arrives
    once it finishes is no message at all.
    """
    import numpy as np

    extents = [s.stop - s.start for s in region]
    nbytes = math.prod(extents) * np.dtype(dtype).itemsize
    if nbytes <= max_bytes:
        return

    # Which coarser level WOULD fit, predicted from the recorded per-level voxel sizes
    # rather than by opening anything — the ratio to level 0 gives each level's extent.
    fits = ""
    if per_level and level < len(per_level):
        here = per_level[level]
        for i in range(level + 1, len(per_level)):
            factor = [c / f for c, f in zip(per_level[i], here)]
            shrunk = math.prod(max(1, e / f) for e, f in zip(extents, factor + [1.0]))
            if shrunk * np.dtype(dtype).itemsize <= max_bytes:
                fits = (f" level={i} would be about "
                        f"{_size(shrunk * np.dtype(dtype).itemsize)}.")
                break

    raise ValueError(
        f"reading {tuple(extents)} from {path} at level {level} is "
        f"{_size(nbytes)}, over the {_size(max_bytes)} cap — and "
        f"the failure this prevents is a HANG, not an error: a read that size does not stop, "
        f"it just never finishes, which in a notebook looks like a wedged kernel."
        + ("" if cropped else " Pass crop= to take a box out of it.")
        + fits
        + " Raise max_bytes= (or pass None) if you mean it.")


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
