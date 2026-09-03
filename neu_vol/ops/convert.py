"""Convert an existing volume (any source backend) into a multiscale zarr v3 or
neuroglancer-precomputed volume, whole or cropped to a box.

Generalizes ``ingest``: the source is any registered backend. Coordinate metadata
(``voxel_size``/``offset``/``units``/``axes``) is read from the source when it
carries it — OME-NGFF zarr groups, precomputed ``info``, a DVID instance, and an HDF5
file that records its own frame beside the array; anything the caller passes explicitly
overrides the read value. Sources that record nothing (bare arrays, image stacks, a
plain HDF5 file) require the caller to supply at least ``voxel_size``.

``crop_start``/``crop_stop`` restrict the copy to a box **without changing the model
space**: the physical offset shifts by the crop origin, so the output lands on top of
the source in a viewer rather than at the origin (CLAUDE.md invariant 1). This is
where cropping belongs, rather than beside the metadata resolution above — the crop
view wraps the resolved ``data_spec``, so a cropped copy reads through exactly the
reader detection chose (invariant 9). ``ops/roi.py`` is a thin wrapper over it.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ..source_metadata import (detect_backend, location_spec, read_source_metadata,
                               require_populated_scale)
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def _resolve_crop(
    data_spec: dict,
    src_shape: Sequence[int],
    *,
    start: Sequence[int] | None,
    stop: Sequence[int] | None,
    n_spatial: int,
    has_channels: bool,
    pad_value: float,
    clip: bool,
) -> tuple[dict, tuple[int, ...], tuple[int, ...]]:
    """A crop view over ``data_spec``: ``(spec, output shape, spatial start)``.

    ``start``/``stop`` are voxel indices into ``data_spec``'s own (level-0) grid, in
    spatial ``(z, y, x)``; a leading channel axis is always taken in full. Either may
    be ``None``, meaning the volume's own bound, so a half-open box needs only the end
    that matters.

    ``clip`` trims the box to the source extent — the default, because the alternative
    is inventing voxels: :class:`~neu_vol.backends.view.CropBackend` happily
    reads outside the volume and fills with ``pad_value``, which for a copy would
    publish fabricated data at the margin with nothing to show it was fabricated.
    ``clip=False`` is the crop-*and*-pad contract of :func:`~neu_vol.extract_roi`.
    """
    spatial = tuple(int(s) for s in (src_shape[1:] if has_channels else src_shape))
    start = (0,) * n_spatial if start is None else tuple(int(v) for v in start)
    stop = spatial if stop is None else tuple(int(v) for v in stop)
    if len(start) != n_spatial or len(stop) != n_spatial:
        raise ValueError(
            f"crop start/stop must have {n_spatial} spatial entries, got "
            f"{start} / {stop}"
        )
    if clip:
        c_start = tuple(max(0, a) for a in start)
        c_stop = tuple(min(b, d) for b, d in zip(stop, spatial))
        if (c_start, c_stop) != (start, stop):
            logger.info("crop clipped to the source extent: [%s:%s] -> [%s:%s]",
                        start, stop, c_start, c_stop)
            start, stop = c_start, c_stop
    out_spatial = tuple(b - a for a, b in zip(start, stop))
    if any(s <= 0 for s in out_spatial):
        raise ValueError(
            f"empty crop: start={start} stop={stop} (source spatial shape {spatial})"
        )
    origin = ((0,) + start) if has_channels else start
    out_shape = ((int(src_shape[0]),) + out_spatial) if has_channels else out_spatial
    spec = {"backend": "crop", "source": dict(data_spec), "origin": list(origin),
            "shape": list(out_shape), "pad_value": pad_value}
    return spec, out_shape, start


def _resolve_masks(
    data_spec: dict,
    src_shape: Sequence[int],
    boxes: Sequence[Sequence[Sequence[int]]],
    *,
    value: float,
    n_spatial: int,
    has_channels: bool,
) -> dict:
    """A mask view over ``data_spec``, blanking each spatial ``(lo, hi)`` box.

    Boxes are spatial ``(z, y, x)`` in the source's own level-0 voxels; a channel axis is
    always masked in full, since "exclude this region" is not a per-channel statement.

    **A box that misses the volume raises.** The caller asked for a region to be excluded;
    copying everything instead is the one outcome that cannot be allowed to pass quietly,
    and a box in the wrong axis order or the wrong scale looks exactly like this.
    """
    spatial = tuple(int(s) for s in (src_shape[1:] if has_channels else src_shape))
    full = []
    for lo, hi in boxes:
        if len(lo) != n_spatial or len(hi) != n_spatial:
            raise ValueError(f"mask box must have {n_spatial} spatial entries per corner, "
                             f"got {tuple(lo)} / {tuple(hi)}")
        lo = tuple(int(v) for v in lo)
        hi = tuple(int(v) for v in hi)
        if any(b <= a for a, b in zip(lo, hi)):
            raise ValueError(f"empty mask box {lo}:{hi}")
        if any(b <= 0 or a >= d for a, b, d in zip(lo, hi, spatial)):
            raise ValueError(
                f"mask box {lo}:{hi} does not intersect the volume (spatial shape "
                f"{spatial}), so nothing would be excluded. Check the axis order (these "
                f"are z,y,x) and the scale the box is in.")
        if has_channels:
            lo, hi = (0,) + lo, (int(src_shape[0]),) + hi
        full.append([list(lo), list(hi)])
    logger.info("masking %d region(s) with fill value %r", len(full), value)
    return {"backend": "mask", "source": dict(data_spec), "boxes": full, "value": value}


def convert(
    src: str | dict,
    dst: str,
    *,
    voxel_size: Sequence[float] | None = None,
    src_format: str | None = None,
    src_level: int | None = None,
    units: str | None = None,
    axes: Sequence[str] | None = None,
    offset: Sequence[float] | None = None,
    has_channels: bool | None = None,
    crop_start: Sequence[int] | None = None,
    crop_stop: Sequence[int] | None = None,
    pad_value: float = 0,
    clip_crop: bool = True,
    mask_boxes: Sequence[Sequence[Sequence[int]]] | None = None,
    mask_value: float = 0,
    background: Sequence[int] | None = None,
    supervoxels: bool = False,
    prefer_locked: bool = False,
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str | None = None,
    encoding: str | None = None,
    compressed_segmentation_block_size: Sequence[int] = (8, 8, 8),
    multiscale: bool = True,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    resume: bool = False,
    verify: bool = False,
    sparse: bool = False,
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Convert ``src`` into a multiscale volume at ``dst``, whole or cropped.

    ``src`` is a path or a full backend spec dict. For a path, ``src_format`` is
    auto-detected unless given: ``info``->precomputed, ``zarr.json``->zarr3,
    ``.zarray``/``.zgroup``->zarr2, a ``dvid://`` URL->dvid, and — from the name, since
    neither has a marker object — an HDF5 file or a glob, file or directory of 2D slices.
    Metadata not passed explicitly is taken from the source where available;
    ``voxel_size`` is required if the source carries none, which an image stack always
    does and an HDF5 file does unless it records its own frame.

    With ``resume=True`` an interrupted run continues, skipping already-done blocks
    instead of recreating them.

    ``background`` names the values the source uses for background — manual segmentation
    numbered from 0 makes it 1 — and replaces them with 0 as the source is read. Do this
    rather than fixing it afterwards: an all-background block of 1s is not all-fill, so
    every one of them would be *stored*, and the volume would no longer answer "where is
    the data" by which chunks exist.

    ``sparse=True`` skips **pyramid** tasks whose input holds no stored chunk, which on a
    sparse volume is nearly all of them. It cannot skip any of the level-0 copy: the
    source there is foreign, and whether it has stored a chunk is not a question this
    package can ask of it.

    ``mask_boxes`` is a list of spatial ``(lo, hi)`` pairs to *exclude*: everything else
    is copied and those regions are written as ``mask_value``. Because the pyramid is
    derived from the output's own level 0, the hole propagates to every level. A box that
    does not intersect the volume raises rather than quietly copying everything.

    ``crop_start``/``crop_stop`` copy only ``src[crop_start:crop_stop]`` (spatial
    ``(z, y, x)`` voxels of the source's level 0, either bound optional). The output's
    physical offset shifts by the crop origin, so it stays in the source's coordinate
    frame. The pyramid is then built **from the cropped level 0**, so its coarse levels
    are the crop's own reductions rather than slices of the source's coarse levels —
    for a crop whose origin is not a multiple of the cumulative factor, a coarse voxel
    of the output straddles the source's differently.
    """
    if isinstance(src, dict):
        src_spec = dict(src)
    else:
        fmt = src_format or detect_backend(src)
        if fmt is None:
            raise ValueError(
                f"could not detect source format at {src!r}: no info / zarr.json / "
                f".zarray marker, and the name is neither an HDF5 file nor a glob, "
                f"file or directory of 2D slices. Pass src_format= explicitly"
            )
        # Not every source is a store: an image stack is addressed by `source`, an HDF5
        # file by path plus the dataset inside it, and a DVID instance by
        # server/uuid/instance. `location_spec` owns all four forms.
        src_spec = location_spec(src, fmt)

    if src_level is not None:
        # Which scale of a multiscale SOURCE becomes the output's level 0. Left unset,
        # `read_source_metadata` picks the finest one that actually stores chunks; this
        # names a scale instead. The output is a new volume either way — its own level 0
        # is whatever was read, at that scale's real voxel size (invariant NM-SPACE),
        # never at an assumed 2**level of the source's finest.
        src_spec["scale_index"] = int(src_level)

    if prefer_locked:
        if src_spec.get("backend") != "dvid":
            raise ValueError(
                f"prefer_locked=True applies to DVID sources only, but the source is "
                f"{src_spec.get('backend')!r}")
        src_spec["prefer_locked"] = True

    if supervoxels:
        # Only set when asked, so a spec dict that already carries it is not clobbered
        # by the default. Rejected rather than ignored for a non-DVID source: silently
        # dropping it would copy agglomerated bodies while the caller believed they had
        # asked for supervoxels, and nothing downstream could tell the difference.
        if src_spec.get("backend") != "dvid":
            raise ValueError(
                f"supervoxels=True applies to DVID sources only, but the source is "
                f"{src_spec.get('backend')!r}")
        src_spec["supervoxels"] = True

    meta = read_source_metadata(src_spec)

    # The array/scale to actually read (level 0 of an OME group / the finest precomputed
    # scale that stores data, or `src_level` if it was named).
    data_spec = meta["data_spec"] if meta else src_spec

    # Refuse a source scale that was declared but never written, BEFORE any work: it
    # opens, reports the extent its `info` claims and reads as the fill value at every
    # block, so the run would succeed and write a volume of zeros. Only fires when some
    # other scale stores data and this one does not — see `require_populated_scale`.
    require_populated_scale(data_spec, op="convert")

    # Warned here rather than in the metadata reader, which is also on the read-only
    # inspection path: this is where an export is about to happen, which is what makes
    # an open node a problem worth interrupting for.
    if data_spec.get("backend") == "dvid" and meta:
        from ..dvid import resolve_node, spec_url

        if not resolve_node(data_spec)["locked"]:
            logger.warning(
                "dvid node %s is OPEN (not locked), so its data can change while this "
                "runs and this export is not reproducible. provenance.json will name "
                "the node. For an immutable snapshot re-run with --dvid-locked, which "
                "takes the newest locked ancestor. Source: %s",
                data_spec.get("uuid"), spec_url(data_spec))

    # Resolve coordinate metadata: explicit arg > source metadata > default.
    if voxel_size is None:
        if not meta:
            raise ValueError(
                "voxel_size is required: source has no readable coordinate metadata"
            )
        voxel_size = meta["voxel_size"]
    if axes is None:
        axes = meta["spatial_axes"] if meta else ("z", "y", "x")
    if units is None:
        units = (meta.get("units") if meta else None) or "nm"
    if offset is None and meta:
        offset = meta["offset"]
    if has_channels is None and meta is not None:
        has_channels = meta["has_channels"]
    if kind is None:
        # Explicit > what the source records > image. Falling back to `image` when the
        # source SAYS `segmentation` was the old behaviour and it is indefensible: the
        # pyramid would average label ids into ids that were never in the data, silently.
        # That footgun is the reason `copy` exists; inheriting here narrows it to sources
        # that genuinely record nothing (image stacks, HDF5, bare arrays).
        kind = (meta or {}).get("kind") or "image"
        if meta and meta.get("kind"):
            logger.info("kind=%s, from the source's own metadata", kind)

    src_backend = open_backend(data_spec)
    src_shape = src_backend.shape
    n_spatial = len(axes)

    if has_channels is None:
        has_channels = len(src_shape) == n_spatial + 1
    if len(src_shape) != n_spatial + (1 if has_channels else 0):
        raise ValueError(
            f"source ndim {len(src_shape)} incompatible with {n_spatial} spatial axes "
            f"{'+ channel' if has_channels else ''}"
        )
    num_channels = int(src_shape[0]) if has_channels else 1

    # The views wrap the *resolved* data_spec, so the read goes through the reader
    # detection chose rather than a hand-built one (invariant 9), and the crop's offset
    # shift keeps the output in the source's frame (invariant 1).
    #
    # Mask first, crop over it: the mask's boxes are then in the source's own coordinates
    # rather than the crop's, which is what a caller means by "exclude this region of the
    # volume" and keeps the two arguments independent.
    read_spec, read_shape = data_spec, src_shape
    out_offset = tuple(offset) if offset else (0.0,) * n_spatial
    if background:
        # Innermost, under mask and crop: it corrects what the source *means* by background,
        # which is true of the data itself and not of any box laid over it. It has to happen
        # here rather than afterwards because an all-background block of 1s is not all-fill,
        # so without it every such block is stored and the volume stops being sparse.
        read_spec = {"backend": "remap", "source": dict(read_spec),
                     "values": [int(v) for v in background], "to": 0}
        logger.info("treating %s as background: replaced with 0 as the source is read",
                    [int(v) for v in background])
    if mask_boxes:
        read_spec = _resolve_masks(read_spec, src_shape, mask_boxes, value=mask_value,
                                   n_spatial=n_spatial, has_channels=has_channels)
    if crop_start is not None or crop_stop is not None:
        read_spec, read_shape, start = _resolve_crop(
            read_spec, src_shape, start=crop_start, stop=crop_stop,
            n_spatial=n_spatial, has_channels=has_channels, pad_value=pad_value,
            clip=clip_crop)
        out_offset = tuple(o + a * v for o, a, v in zip(out_offset, start, voxel_size))
        logger.info("crop [%s:%s] of %s -> shape=%s offset=%s nm",
                    start, tuple(a + b for a, b in zip(start, read_shape[-n_spatial:])),
                    src_shape, read_shape, out_offset)
    logger.info("convert %s -> %s | shape=%s dtype=%s channels=%s voxel_size=%s",
                data_spec, dst, read_shape, src_backend.dtype,
                num_channels if has_channels else 1, tuple(voxel_size))

    summary = materialize_multiscale(
        src_spec=read_spec,
        src_shape=read_shape,
        src_dtype=str(src_backend.dtype),
        dst=dst,
        profile=profile,
        voxel_size=tuple(voxel_size),
        offset=out_offset,
        units=units,
        spatial_axes=tuple(axes),
        has_channels=has_channels,
        num_channels=num_channels,
        dtype=dtype,
        kind=kind,
        encoding=encoding,
        compressed_segmentation_block_size=tuple(compressed_segmentation_block_size),
        multiscale=multiscale,
        factors=factors,
        max_levels=max_levels,
        min_dim=min_dim,
        name=name,
        chunk=chunk,
        shard=shard,
        client=client,
        npartitions=npartitions,
        delete_existing=delete_existing,
        resume=resume,
        verify=verify,
        sparse=sparse,
        progress_path=progress_path,
        validate=validate,
    )

    # After the data, never before: a provenance record for a run that died half way
    # through would claim a volume exists that does not. The spec recorded is the
    # *resolved* one — for DVID that is the concrete node, not the branch ref, which is
    # the entire point of writing this at all.
    from . import provenance as _provenance

    _provenance.write(dst, _provenance.build_record(
        src_spec=(meta or {}).get("provenance_spec") or src_spec,
        dst=dst,
        kind=kind,
        voxel_size=list(voxel_size),
        crop_start=list(crop_start) if crop_start is not None else None,
        crop_stop=list(crop_stop) if crop_stop is not None else None,
        mask_boxes=[[list(lo), list(hi)] for lo, hi in mask_boxes] if mask_boxes else None,
        background=list(background) if background else None,
        num_levels=summary.get("num_levels"),
        status_counts=summary.get("status_counts"),
    ))
    return summary
