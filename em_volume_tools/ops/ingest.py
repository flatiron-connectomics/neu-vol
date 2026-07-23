"""Ingest an image stack into a multiscale OME-NGFF 0.5 zarr v3 volume.

The first vertical slice (docs/DESIGN.md §7): image stack (TIFF/PNG glob or
multipage TIFF) -> level-0 zarr array -> strict level-by-level pyramid -> OME
group metadata. The heavy lifting is shared with ``convert`` in _multiscale.py.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def ingest_image_stack(
    src: str,
    dst: str,
    voxel_size: Sequence[float],
    *,
    units: str = "nm",
    axes: Sequence[str] = ("z", "y", "x"),
    offset: Sequence[float] | None = None,
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
    """Ingest ``src`` image stack into a multiscale zarr v3 group at ``dst``.

    ``voxel_size``/``offset``/``axes`` are in canonical ``(z, y, x)`` order.
    Returns a summary dict (per-level paths, shapes, scales).
    """
    src_spec = {"backend": "image_stack", "source": src}
    src_backend = open_backend(src_spec)
    logger.info("ingest %s -> %s | shape=%s dtype=%s", src, dst, src_backend.shape, src_backend.dtype)
    return materialize_multiscale(
        src_spec=src_spec,
        src_shape=src_backend.shape,
        src_dtype=str(src_backend.dtype),
        dst=dst,
        profile=profile,
        voxel_size=tuple(voxel_size),
        offset=tuple(offset) if offset else (0.0,) * len(voxel_size),
        units=units,
        spatial_axes=tuple(axes),
        has_channels=False,     # v1 image stacks are single-channel 2D slices
        num_channels=1,
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
