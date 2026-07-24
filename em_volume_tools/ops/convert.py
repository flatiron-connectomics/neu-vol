"""Convert an existing volume (any source backend) into a multiscale zarr v3 or
neuroglancer-precomputed volume.

Generalizes ``ingest``: the source is any registered backend. Coordinate metadata
(``voxel_size``/``offset``/``units``/``axes``) is read from the source when it
carries it (OME-NGFF zarr groups, precomputed ``info``); anything the caller
passes explicitly overrides the read value. Sources without metadata (bare
arrays, HDF5) require the caller to supply at least ``voxel_size``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ..introspect import detect_backend, read_source_metadata
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


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
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Convert ``src`` into a multiscale volume at ``dst``.

    ``src`` is a path or a full backend spec dict. For a path, ``src_format`` is
    auto-detected (``info``->precomputed, ``zarr.json``->zarr3,
    ``.zarray``/``.zgroup``->zarr2) unless given. Metadata not passed explicitly is
    taken from the source where available; ``voxel_size`` is required if the source
    carries none. With ``resume=True`` an interrupted run continues, skipping
    already-done blocks instead of recreating them.
    """
    if isinstance(src, dict):
        src_spec = dict(src)
    else:
        fmt = src_format or detect_backend(src)
        if fmt is None:
            raise ValueError(
                f"could not detect source format at {src!r} (no info/zarr.json/.zarray); "
                "pass src_format= explicitly"
            )
        src_spec = {"backend": fmt, "path": src}
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
    logger.info("convert %s -> %s | shape=%s dtype=%s channels=%s voxel_size=%s",
                data_spec, dst, src_shape, src_backend.dtype,
                num_channels if has_channels else 1, tuple(voxel_size))

    return materialize_multiscale(
        src_spec=data_spec,
        src_shape=src_shape,
        src_dtype=str(src_backend.dtype),
        dst=dst,
        profile=profile,
        voxel_size=tuple(voxel_size),
        offset=tuple(offset) if offset else (0.0,) * n_spatial,
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
        progress_path=progress_path,
        validate=validate,
    )
