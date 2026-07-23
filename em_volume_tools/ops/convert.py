"""Convert an existing volume (any source backend) into a multiscale zarr v3.

Generalizes ``ingest``: the source is any registered backend (another zarr v3,
neuroglancer-precomputed, HDF5, or an image stack) rather than specifically an
image stack. Shares the copy + pyramid + OME-metadata core (_multiscale.py).

Coordinate metadata (``voxel_size``/``offset``/``units``/``axes``) is supplied by
the caller — reading it from source OME/precomputed metadata is a later
enhancement.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def convert(
    src: str | dict,
    dst: str,
    voxel_size: Sequence[float],
    *,
    src_format: str = "zarr3",
    units: str = "nm",
    axes: Sequence[str] = ("z", "y", "x"),
    offset: Sequence[float] | None = None,
    has_channels: bool | None = None,
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str = "image",
    multiscale: bool = True,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    validate: bool = True,
) -> dict:
    """Convert ``src`` into a multiscale zarr v3 group at ``dst``.

    ``src`` is either a path (opened with ``src_format``) or a full backend spec
    dict (e.g. ``{"backend": "hdf5", "path": ..., "dataset": ...}``). When
    ``has_channels`` is ``None`` it is inferred as ``ndim == len(axes) + 1``.
    """
    src_spec = dict(src) if isinstance(src, dict) else {"backend": src_format, "path": src}
    src_backend = open_backend(src_spec)
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
    logger.info("convert %s -> %s | shape=%s dtype=%s channels=%s",
                src_spec, dst, src_shape, src_backend.dtype, num_channels if has_channels else 1)

    return materialize_multiscale(
        src_spec=src_spec,
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
        validate=validate,
    )
