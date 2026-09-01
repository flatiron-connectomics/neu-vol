"""Pack a small volume into one HDF5 file, described well enough to be written back.

The inverse of ``ops/write.py``: that places a piece into a large volume, this produces
the piece. An image stack straight off a microscope or an annotation tool is a directory
of PNGs with no coordinates attached; packing it records **where it belongs and at what
scale**, so `neu-vol write` can later place it without anyone re-typing an offset.

An HDF5 source that **already describes itself** is repacked without losing what it says:
`voxel_offset`, `voxel_size` and any recorded `axes` are read back (as attributes or as
top-level datasets — files written elsewhere do both) and become the defaults. That makes
this the way to bring a foreign file into the canonical layout: point it at the file and
pass nothing.

The source is anything readable — an image stack, an HDF5 file, or **a volume**, in which
case ``level`` and ``crop_start``/``crop_stop`` take a box out of one level of it. That
direction closes a loop: extract a region, work on it, write it straight back. Both
defaults exist to make the round trip argument-free — the level's own recorded voxel size
becomes the frame, and the crop origin becomes the recorded ``voxel_offset``.

What gets recorded, and why those names:

* ``voxel_offset`` — integer voxels, on the dataset. This is the field
  ``ops/write.resolve_offset`` already looks for, and the name is precomputed's.
* ``voxel_size`` / ``offset`` / ``units`` / ``axes`` — the frame, in this package's own
  vocabulary (:class:`~neu_vol.meta.VoxelMeta`), on the root *and* the dataset so
  that either the file or the array alone is self-describing. ``offset`` here is the
  physical position in ``units``; ``voxel_offset`` is the same place counted in voxels,
  and both are written because the two consumers ask different questions.
* ``axes`` is the one that pays for itself immediately: ``voxel_offset``'s axis order was
  previously **unknowable from the file** — precomputed means xyz, this package means zyx,
  and a wrong guess mirrors the piece through the z=x diagonal with nothing downstream
  able to tell. A file written here says which it is, and
  :meth:`~neu_vol.backends.hdf5.HDF5Backend.stored_axes` is how the read side
  finds out.

Reads are blocked, so a "small" volume that turns out not to be still packs: the whole
array is never held in memory at once. Serial and in-process, like ``create``/``write`` —
one file is not a cluster's worth of work.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

from blockrun import iter_blocks

from ..backends.base import open_backend
from ..source_metadata import PRECOMPUTED_GZ
from .write import source_spec

logger = logging.getLogger(__name__)

#: The dataset a file gets when the caller names none — the same default the reader
#: assumes (``HDF5Backend`` opens ``spec.get("dataset", "/data")``), so a file packed
#: without arguments is readable without arguments.
DEFAULT_DATASET = "/data"

#: Frame attributes that must agree when adding a dataset to an existing file. Not
#: ``voxel_offset``: every dataset has its own, which is what makes several pieces in one
#: file useful rather than contradictory. ``voxel_size`` is the one whose *name* is a
#: parameter, since other tools have their own word for it — see ``voxel_size_field``.
FRAME_ATTRS = ("voxel_size", "units", "axes")

#: This package's names for the two fields other writers are most likely to spell
#: differently. Both are settable per call, and both are used for reading *and* writing so
#: a file stays readable by whatever wrote its siblings.
DEFAULT_VOXEL_SIZE_FIELD = "voxel_size"
DEFAULT_OFFSET_FIELD = "voxel_offset"

#: Ceiling on one read, so packing a volume that is larger than advertised streams instead
#: of dying. Blocks cover whole z-slabs, the natural read unit of an image stack.
DEFAULT_BLOCK_BYTES = 1 * 1024 ** 3

#: Attribute recording what the voxels MEAN (:data:`neu_lib.KINDS`), on the dataset.
#:
#: HDF5 has no agreed-on place for this, which is why the *reader* refuses to guess one —
#: a uint8 label array is indistinguishable from an image by dtype, and guessing wrong
#: authorises averaging label ids into ids that were never in the data. Recording it when
#: it is **known** is the other half of that: a :class:`neu_lib.Piece` carries its kind, so
#: writing one out and reading it back should not lose it. Read back only when the value is
#: one this package would have written, the same rule the zarr reader applies to the
#: multiscales ``type``.
#:
#: On the dataset only, not the root: it describes an array, where ``voxel_size`` / ``units``
#: / ``axes`` describe the file's coordinate system and several arrays share it.
KIND_ATTR = "kind"


def resolve_source(src: str | dict, src_format: str | None = None, level: int = 0,
                   dataset: str | None = None,
                   voxel_size_field: str = "voxel_size",
                   offset_field: str = "voxel_offset") -> tuple[dict, dict]:
    """The spec for the level to read, and what the source says about its own frame.

    Two kinds of source, and they need different handling:

    * a **multiscale volume** — resolved through
      :func:`~neu_vol.source_metadata.level_spec`, which is the only thing that
      addresses a level correctly for both formats (zarr v3 puts each level in its own
      subdirectory, precomputed selects with ``scale_index``). Going straight at the path
      is what made a zarr *group* fail to open at all: a group is not an array. Its
      recorded per-level voxel size and units come back too, so nobody retypes what the
      volume already knows.
    * **anything else** — an image stack, a bare array: no levels, and nothing to say
      about scale. ``level`` other than 0 is then an error rather than silently ignored.
      An **HDF5** source is in this group but does have somewhere to record a scale, so it
      is asked: repacking a file that already carries a ``voxel_size`` should not make
      anyone retype it.
    """
    from ..source_metadata import (detect_backend, existing_levels, level_spec,
                                   read_level_voxel_sizes, read_source_metadata)

    fmt = None if isinstance(src, dict) else (src_format or detect_backend(str(src)))
    if fmt not in ("zarr3", "neuroglancer_precomputed", PRECOMPUTED_GZ):
        if level:
            raise ValueError(
                f"--level {level} needs a multiscale volume; {src!r} is "
                f"{fmt or src_format or 'not one'}, which has no levels")
        spec = source_spec(src, src_format, dataset)
        frame = {}
        if spec.get("backend") == "hdf5":
            # An HDF5 file may already describe itself — files written elsewhere routinely
            # carry `voxel_offset` and `voxel_size`, as attributes or as top-level datasets
            # (`stored_*` searches both). Repacking one must not quietly drop what it says:
            # a piece that arrives knowing where it belongs and leaves claiming the origin
            # is the kind of loss nothing downstream can detect.
            backend = open_backend(spec)
            size = backend.stored_voxel_size(voxel_size_field)
            if size:
                frame["voxel_size"] = tuple(size[0])
                logger.info("voxel_size taken from the source, %s", size[1])
            order = backend.stored_axes()
            if order:
                frame["axes"] = tuple(order[0])
                logger.info("axes taken from the source, %s", order[1])
            meaning = backend.stored_kind()
            if meaning:
                frame["kind"] = meaning[0]
                logger.info("kind taken from the source, %s", meaning[1])
            found = backend.stored_offset(offset_field)
            if found:
                # Through `resolve_offset` so the axis-order rule lives in one place: a
                # recorded `axes` decides it, else zyx, and the provenance says which.
                from .write import resolve_offset

                value, prov = resolve_offset(backend, None, field=offset_field,
                                             order=None, ndim=len(found[0]))
                frame["voxel_offset"] = tuple(value)
                logger.info("voxel_offset taken %s", prov)
        return spec, frame

    volume = str(src).rstrip("/")
    per_level = read_level_voxel_sizes({"backend": fmt, "path": volume}) or []
    levels = existing_levels(volume, "neuroglancer_precomputed"
                             if fmt == PRECOMPUTED_GZ else fmt)
    if not levels:
        # A bare zarr *array* carries a `zarr.json` too, so detection calls it zarr3 while
        # it has no levels at all — addressing it as `<path>/0` finds nothing. Same
        # distinction `copy` has to make.
        if level:
            raise ValueError(f"{volume} has no levels (it looks like a bare array, not a "
                             f"multiscale volume), so --level {level} means nothing")
        return source_spec(src, src_format, dataset), {}
    if level not in levels:
        raise ValueError(f"{volume} has no level {level} (present: {sorted(levels)})")
    meta = read_source_metadata({"backend": fmt, "path": volume}) or {}
    frame = {}
    if level < len(per_level):
        frame["voxel_size"] = tuple(per_level[level])
    if meta.get("units"):
        frame["units"] = meta["units"]
    if meta.get("kind"):
        # A volume states what its voxels mean; the packed piece should not have to be
        # told again. This is the one direction that is safe to carry automatically —
        # inferring it from a dtype is the mistake, reading a recorded value is not.
        frame["kind"] = meta["kind"]
    return level_spec(volume, fmt, level), frame


def _block_shape(shape: Sequence[int], itemsize: int,
                 max_bytes: int = DEFAULT_BLOCK_BYTES) -> tuple[int, ...]:
    """Whole slabs along the slowest axis, sized to ``max_bytes``. At least one plane."""
    shape = tuple(int(s) for s in shape)
    plane = max(1, math.prod(shape[1:]) * itemsize)
    n = max(1, min(shape[0], max_bytes // plane))
    return (n,) + shape[1:]


def _read_frame(h5file, voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD) -> dict:
    """The frame recorded at a file's root, in comparable form. Empty if it records none.

    Keyed by this package's names whatever the file calls them, so the comparison below
    does not have to care which spelling was chosen.
    """
    out = {}
    for key in FRAME_ATTRS:
        attr = voxel_size_field if key == "voxel_size" else key
        if attr in h5file.attrs:
            value = h5file.attrs[attr]
            if key == "units":
                out[key] = value.decode() if isinstance(value, bytes) else str(value)
            elif key == "axes":
                out[key] = _axes_tuple(value)
            else:
                out[key] = tuple(float(v) for v in np.asarray(value).ravel())
    return out


def _axes_tuple(value: Any) -> tuple[str, ...]:
    """An axis-order attribute as ``("z","y","x")``, through the reader's own parser."""
    from ..backends.hdf5 import axes_string

    return tuple(axes_string(value))


def _frame_mismatch(existing: dict, wanted: dict) -> list[str]:
    """Human-readable differences between two frames, in the order of FRAME_ATTRS."""
    bad = []
    for key in FRAME_ATTRS:
        if key not in existing:
            continue
        old, new = existing[key], wanted[key]
        same = (np.allclose(old, new) if key == "voxel_size"
                else tuple(old) == tuple(new))
        if not same:
            bad.append(f"{key}: file has {old}, this would write {new}")
    return bad


# --------------------------------------------------------------------------- #
# The file layout, in one place
#
# Two callers write an HDF5 piece and they must produce the *same* file: `pack_hdf5`
# streams one out of a source location, and `neu_vol.write_piece` writes an array that is
# already in memory. The three functions below are everything either of them knows about
# the layout — the dataset name, what an existing file allows, and which attributes land
# where — so a change to the convention cannot reach one writer and miss the other. That
# had a real cost available: a file whose `axes` said one thing and whose sibling dataset
# said another would place the two pieces mirrored against each other.
# --------------------------------------------------------------------------- #
def hdf5_dataset_name(dataset: str | None) -> str:
    """A dataset name as a rooted path, defaulting to :data:`DEFAULT_DATASET`.

    Rooted because that is the form ``h5py`` reports back from ``visititems`` and the form
    every caller compares against; ``"gt/piece0"`` and ``"/gt/piece0"`` naming the same
    array while failing an equality test is how a second write silently became an append.
    """
    name = dataset or DEFAULT_DATASET
    return name if name.startswith("/") else "/" + name


def check_hdf5_target(out: str, name: str, frame: dict, *, overwrite: bool = False,
                      voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD
                      ) -> tuple[dict, list[str], bool]:
    """What an existing ``out`` allows: ``(its frame, its datasets, replacing)``.

    An existing file is **added to** rather than replaced — several pieces of one volume
    in one file is a legitimate arrangement, and each dataset keeps its own
    ``voxel_offset``. Two things are refused instead of guessed:

    * a frame that disagrees with the file's, since one file describing two coordinate
      systems has no correct reading;
    * a dataset name already in use, unless ``overwrite``.

    A missing file is not an error — it is the ordinary case — so this answers with empties.
    """
    import os

    import h5py

    if not os.path.exists(out):
        return {}, [], False

    with h5py.File(out, "r") as f:
        existing_frame = _read_frame(f, voxel_size_field)
        found: list[str] = []
        f.visititems(lambda n, o: found.append("/" + n)
                     if isinstance(o, h5py.Dataset) and o.ndim >= 3 else None)
        existing_datasets = sorted(found)

    mismatch = _frame_mismatch(existing_frame, frame)
    if mismatch:
        raise ValueError(
            f"{out} already records a different frame, so adding to it would make one "
            f"file describe two coordinate systems: " + "; ".join(mismatch))
    replacing = name in existing_datasets
    if replacing and not overwrite:
        raise FileExistsError(
            f"{out} already has a dataset at {name}. Give another name, or pass "
            f"overwrite=True to replace it.")
    return existing_frame, existing_datasets, replacing


def write_hdf5_array(out: str, name: str, read_region, *, shape: Sequence[int],
                     dtype: Any, voxel_size: Sequence[float],
                     voxel_offset: Sequence[int], units: str = "nm",
                     axes: Sequence[str] = ("z", "y", "x"), kind: str | None = None,
                     chunk: Sequence[int] | None = None,
                     compression: str | None = "gzip", replacing: bool = False,
                     block: Sequence[int] | None = None,
                     has_channels: bool = False,
                     voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD,
                     offset_field: str = DEFAULT_OFFSET_FIELD) -> None:
    """Create ``out``'s ``name`` from ``read_region``, and record the frame beside it.

    ``read_region(region) -> ndarray`` is called once per block, so nothing here holds the
    whole array: a "small" piece that turns out not to be still writes. ``voxel_offset``
    describes the **spatial** axes; a leading channel axis, if there is one, is prepended
    as 0, since a channel is not a position.

    What is recorded, and why in two places: ``voxel_size`` / ``units`` / ``axes`` go on the
    root *and* the dataset, so either the file or the array alone is self-describing;
    ``voxel_offset`` (whole voxels), ``offset`` (the same place in ``units``) and ``kind``
    go on the dataset alone, being that piece's own. ``axes`` is the one that pays for
    itself immediately — ``voxel_offset``'s axis order is otherwise **unknowable from the
    file**, since the name is precomputed's and means xyz there while everything in this
    package is zyx, and a wrong guess mirrors the piece through the z=x diagonal with
    nothing downstream able to tell.
    """
    import h5py

    from ..backends.base import release_backends

    # Before opening for writing: a cached reader holds an open handle, and HDF5 refuses to
    # open a file for writing while one does — so reading a file and then writing to it in
    # the same process failed on the write, with an error naming the file rather than the
    # reader still holding it. `release_backends` closes those and says why.
    release_backends(out)

    shape = tuple(int(s) for s in shape)
    out_dtype = np.dtype(dtype)
    axes = tuple(axes)
    voxel_size = tuple(float(v) for v in voxel_size)
    voxel_offset = tuple(int(v) for v in voxel_offset)

    chunk_full = tuple(chunk) if chunk else tuple(min(64, s) for s in shape)
    if len(chunk_full) != len(shape):
        raise ValueError(f"chunk {chunk_full} does not match the source's {len(shape)} axes")
    chunk_full = tuple(min(int(c), int(s)) for c, s in zip(chunk_full, shape))
    block = tuple(block) if block else _block_shape(shape, out_dtype.itemsize)

    with h5py.File(out, "a") as f:
        if replacing:
            del f[name]
        dset = f.create_dataset(name, shape=shape, dtype=out_dtype, chunks=chunk_full,
                                **({"compression": compression} if compression else {}))
        for b in iter_blocks(shape, block):
            data = read_region(b.region)
            if str(data.dtype) != str(out_dtype):
                data = data.astype(out_dtype)
            dset[b.region] = data

        for target in (f, dset):
            target.attrs[voxel_size_field] = np.asarray(voxel_size, dtype="float64")
            target.attrs["units"] = units
            target.attrs["axes"] = "".join(axes)
        full_offset = ((0,) + voxel_offset) if has_channels else voxel_offset
        dset.attrs[offset_field] = np.asarray(full_offset, dtype="int64")
        dset.attrs["offset"] = np.asarray(
            [o * v for o, v in zip(voxel_offset, voxel_size)], dtype="float64")
        if kind is not None:
            dset.attrs[KIND_ATTR] = str(kind)


def pack_hdf5(
    src: str | dict,
    out: str,
    *,
    voxel_size: Sequence[float] | None = None,
    voxel_offset: Sequence[int] | None = None,
    units: str | None = None,
    axes: Sequence[str] = ("z", "y", "x"),
    level: int = 0,
    crop_start: Sequence[int] | None = None,
    crop_stop: Sequence[int] | None = None,
    dataset: str | None = None,
    src_dataset: str | None = None,
    src_format: str | None = None,
    voxel_size_field: str = DEFAULT_VOXEL_SIZE_FIELD,
    offset_field: str = DEFAULT_OFFSET_FIELD,
    background: Sequence[int] | None = None,
    kind: str | None = None,
    dtype: str | None = None,
    chunk: Sequence[int] | None = None,
    compression: str | None = "gzip",
    overwrite: bool = False,
    dry_run: bool = False,
    max_bytes: int = DEFAULT_BLOCK_BYTES,
) -> dict:
    """Write ``src`` into ``out`` as an HDF5 dataset carrying its frame and position.

    ``dataset`` defaults to :data:`DEFAULT_DATASET`. An existing file is **added to**
    rather than replaced, provided its recorded frame matches — several pieces of one
    volume in one file is a legitimate arrangement, and each dataset keeps its own
    ``voxel_offset``. Two things are refused instead of guessed: a frame that disagrees
    with the file's, and a dataset name already in use (pass another ``dataset`` or
    ``overwrite=True``).

    ``voxel_offset`` is in the same axis order as ``axes``, whole voxels, and names the
    position of the piece's ``(0, ...)`` corner in the volume it belongs to.

    ``kind`` records what the voxels mean (:data:`KIND_ATTR`) and defaults to what the
    source recorded — a precomputed or zarr volume says so in its own metadata, an image
    stack or a bare array cannot. Pass it explicitly for those, or leave it unrecorded:
    guessing from the dtype is the mistake that later averages label ids.

    To write an array that is **already in memory**, use
    :func:`neu_vol.write_piece` — same file layout, no round trip through a store.
    """
    from ..backends.hdf5 import require_local_path

    # Up front, before any of the source is read: h5py has no object-store driver, and
    # discovering that after streaming a volume through it would be a poor trade.
    out = require_local_path(out, "the HDF5 file to write")
    name = hdf5_dataset_name(dataset)
    axes = tuple(axes)

    # `write.source_spec` for a file someone is pointing at — the same situation `write`
    # is in, so guessing image_stack from a directory or glob is worth doing rather than
    # demanding --src-format — and `level_spec` for a level of a volume, which also reports
    # the frame that volume already records.
    spec, frame = resolve_source(src, src_format, level, src_dataset, voxel_size_field,
                                 offset_field)
    from ..backends.base import spec_names_path

    if spec_names_path(spec, out):
        # The source is read block by block *while* the destination is open for writing, and
        # HDF5 permits only one of those at a time on one file — so this has never worked; it
        # failed on `h5py.File(out, "a")` with an error about the file being open read-only.
        # Said plainly here instead, because the intent (add a dataset to this same file) is
        # reasonable and the way to do it is a second file, or `neu_vol.write_piece` on a
        # piece already read out.
        raise ValueError(
            f"src and out are the same file ({out}), which HDF5 cannot do: the source is "
            f"read while the destination is open for writing, and one file cannot be both. "
            f"Pack into a new file, or `read_piece` the array and `write_piece` it back.")
    if voxel_size is None:
        voxel_size = frame.get("voxel_size")
        if voxel_size is None:
            raise ValueError(
                "voxel_size is required: this source records no physical scale — an image "
                "stack, an HDF5 file or a bare array records none — and attaching one is "
                "the point of packing")
    if units is None:
        units = frame.get("units") or "nm"
    voxel_size = tuple(float(v) for v in voxel_size)

    backend = open_backend(spec)
    shape = tuple(int(s) for s in backend.shape)
    out_dtype = np.dtype(dtype or backend.dtype)
    n_spatial = len(axes)
    if len(shape) not in (n_spatial, n_spatial + 1):
        raise ValueError(f"source is {len(shape)}-D, which is neither {n_spatial} spatial "
                         f"axes {axes} nor those plus a leading channel axis")
    has_channels = len(shape) == n_spatial + 1

    # A crop is the same read-only view `convert` lays over a source, so a box out of a
    # volume costs no new machinery. Its origin becomes the piece's voxel_offset unless the
    # caller said otherwise: a region extracted from level N belongs back at that spot, and
    # having to restate it would be the one number nobody should have to type twice.
    # Under the crop: it corrects what the source means by background, which is true of the
    # data and not of any box over it. Doing it now also means the packed file is already
    # correct, so nothing downstream has to know the source had this quirk.
    if background:
        spec = {"backend": "remap", "source": dict(spec),
                "values": [int(v) for v in background], "to": 0}
        backend = open_backend(spec)
        logger.info("treating %s as background: replaced with 0 while packing",
                    [int(v) for v in background])

    crop_origin = None
    if crop_start is not None or crop_stop is not None:
        from .convert import _resolve_crop

        spec, shape, crop_origin = _resolve_crop(
            spec, shape, start=crop_start, stop=crop_stop, n_spatial=n_spatial,
            has_channels=has_channels, pad_value=0, clip=True)
        backend = open_backend(spec)
    if voxel_offset is None:
        # Where the source says it sits, plus where in it the crop started: a box taken out
        # of a piece that knows its own position belongs at the sum, not at the box's
        # offset within the piece. Zeros only when the source records nothing.
        base = frame.get("voxel_offset") or (0,) * n_spatial
        voxel_offset = (tuple(b + c for b, c in zip(base, crop_origin))
                        if crop_origin is not None else tuple(base))
    voxel_offset = tuple(int(v) for v in voxel_offset)
    if not (len(voxel_size) == len(voxel_offset) == len(axes)):
        raise ValueError(f"voxel_size {voxel_size}, voxel_offset {voxel_offset} and axes "
                         f"{axes} must describe the same number of axes")

    if kind is None:
        kind = frame.get("kind")

    # The frame describes the spatial axes; a channel axis is not a physical dimension.
    frame = {"voxel_size": voxel_size, "units": units, "axes": axes}

    _existing_frame, existing_datasets, replacing = check_hdf5_target(
        out, name, frame, overwrite=overwrite, voxel_size_field=voxel_size_field)

    others = [d for d in existing_datasets if d != name]
    block = _block_shape(shape, out_dtype.itemsize, max_bytes)
    plan = {
        "out": out, "dataset": name, "src_spec": spec, "shape": shape,
        "dtype": str(out_dtype), "has_channels": has_channels,
        "voxel_size": voxel_size, "voxel_offset": voxel_offset, "units": units,
        "axes": axes, "kind": kind, "chunk": tuple(chunk) if chunk else None,
        "compression": compression, "nbytes": float(out_dtype.itemsize) * math.prod(shape),
        "blocks": len(list(iter_blocks(shape, block))), "block_shape": block,
        "replacing": replacing, "appending": bool(existing_datasets) and not replacing,
        "other_datasets": others, "level": level, "crop_origin": crop_origin,
    }
    if dry_run:
        return plan

    # A second volumetric dataset makes `sole_dataset` ambiguous, so every later reader
    # has to name one. Cheap to say now, confusing to discover later.
    if others:
        logger.warning("%s will hold %d volumetric datasets (%s); readers must name one "
                       "— `neu-vol write --dataset %s`, or HDF5Backend's `dataset` key",
                       out, len(others) + 1, ", ".join(others + [name]), name)

    write_hdf5_array(out, name, backend.read_region, shape=shape, dtype=out_dtype,
                     voxel_size=voxel_size, voxel_offset=voxel_offset, units=units,
                     axes=axes, kind=kind, chunk=chunk, compression=compression,
                     replacing=replacing, block=block, has_channels=has_channels,
                     voxel_size_field=voxel_size_field, offset_field=offset_field)
    logger.info("packed %s -> %s%s (%s %s, %d block(s))", spec, out, name, shape,
                out_dtype, plan["blocks"])
    return plan
