"""Convert an existing volume (any source backend) into a multiscale zarr v3 or
neuroglancer-precomputed volume, whole or cropped to a box.

Generalizes ``ingest``: the source is any registered backend. Coordinate metadata
(``voxel_size``/``offset``/``units``/``axes``) is read from the source when it
carries it (OME-NGFF zarr groups, precomputed ``info``); anything the caller
passes explicitly overrides the read value. Sources without metadata (bare
arrays, HDF5) require the caller to supply at least ``voxel_size``.

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
from ..source_metadata import detect_backend, read_source_metadata
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
    is inventing voxels: :class:`~em_volume_tools.backends.view.CropBackend` happily
    reads outside the volume and fills with ``pad_value``, which for a copy would
    publish fabricated data at the margin with nothing to show it was fabricated.
    ``clip=False`` is the crop-*and*-pad contract of :func:`~em_volume_tools.extract_roi`.
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
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str = "image",
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
    auto-detected (``info``->precomputed, ``zarr.json``->zarr3,
    ``.zarray``/``.zgroup``->zarr2) unless given. Metadata not passed explicitly is
    taken from the source where available; ``voxel_size`` is required if the source
    carries none. With ``resume=True`` an interrupted run continues, skipping
    already-done blocks instead of recreating them.

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
                f"could not detect source format at {src!r} (no info/zarr.json/.zarray); "
                "pass src_format= explicitly (use 'image_stack' for a directory or "
                "glob of ordered 2D slices)"
            )
        # The image-stack backend addresses its input as `source` (a directory or a
        # glob), not `path`. It is also never auto-detected — a directory of PNGs is
        # not distinguishable from any other directory of PNGs — so it only arrives
        # here when asked for by name.
        src_spec = ({"backend": fmt, "source": src} if fmt == "image_stack"
                    else {"backend": fmt, "path": src})
    meta = read_source_metadata(src_spec)

    # The array/scale to actually read (level 0 of an OME group / finest precomputed scale).
    data_spec = meta["data_spec"] if meta else src_spec

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

    return materialize_multiscale(
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
