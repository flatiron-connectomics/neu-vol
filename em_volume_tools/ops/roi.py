"""Extract / crop / pad a region of interest into a new multiscale volume.

Wraps the source in a read-only crop view (backends/view.py) and runs the shared
materialize engine, so the ROI output is chunked, optionally multiscale, and in
either zarr v3 or precomputed — just like ``convert``. The crop origin shifts the
physical offset accordingly. ``start``/``stop`` are spatial ``(z, y, x)``; a
leading channel axis is preserved in full.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def extract_roi(
    src: str | dict,
    dst: str,
    start: Sequence[int],
    stop: Sequence[int],
    voxel_size: Sequence[float],
    *,
    src_format: str = "zarr3",
    pad_value: float = 0,
    units: str = "nm",
    axes: Sequence[str] = ("z", "y", "x"),
    offset: Sequence[float] | None = None,
    has_channels: bool | None = None,
    profile: str = "local",
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    dtype: str | None = None,
    kind: str = "image",
    encoding: str | None = None,
    compressed_segmentation_block_size: Sequence[int] = (8, 8, 8),
    multiscale: bool = False,
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
    """Extract ``src[start:stop]`` (spatial) into a new volume at ``dst``.

    ``start`` may be negative and ``stop`` may exceed the source extent; the
    out-of-bounds margin is filled with ``pad_value`` (crop *and* pad in one).
    """
    src_spec = dict(src) if isinstance(src, dict) else {"backend": src_format, "path": src}
    src_backend = open_backend(src_spec)
    src_shape = src_backend.shape
    n_spatial = len(axes)

    if has_channels is None:
        has_channels = len(src_shape) == n_spatial + 1
    if len(start) != n_spatial or len(stop) != n_spatial:
        raise ValueError(f"start/stop must have {n_spatial} spatial entries")

    # Full-ndim origin/shape for the crop view (channel axis taken in full).
    if has_channels:
        num_channels = int(src_shape[0])
        origin = (0,) + tuple(int(s) for s in start)
        out_shape = (num_channels,) + tuple(int(b - a) for a, b in zip(start, stop))
    else:
        num_channels = 1
        origin = tuple(int(s) for s in start)
        out_shape = tuple(int(b - a) for a, b in zip(start, stop))
    if any(s <= 0 for s in out_shape):
        raise ValueError(f"empty ROI: start={tuple(start)} stop={tuple(stop)}")

    crop_spec = {"backend": "crop", "source": src_spec, "origin": list(origin),
                 "shape": list(out_shape), "pad_value": pad_value}

    # Physical offset of the ROI origin = base offset + start * voxel_size.
    base_offset = tuple(offset) if offset else (0.0,) * n_spatial
    roi_offset = tuple(o + a * v for o, a, v in zip(base_offset, start, voxel_size))
    logger.info("extract_roi %s [%s:%s] -> %s (shape=%s)", src_spec, tuple(start), tuple(stop), dst, out_shape)

    return materialize_multiscale(
        src_spec=crop_spec, src_shape=out_shape, src_dtype=str(src_backend.dtype),
        dst=dst, profile=profile, voxel_size=tuple(voxel_size), offset=roi_offset,
        units=units, spatial_axes=tuple(axes), has_channels=has_channels,
        num_channels=num_channels, dtype=dtype, kind=kind, encoding=encoding,
        compressed_segmentation_block_size=tuple(compressed_segmentation_block_size),
        multiscale=multiscale,
        factors=factors, max_levels=max_levels, min_dim=min_dim, name=name,
        chunk=chunk, shard=shard, client=client, npartitions=npartitions,
        delete_existing=delete_existing, resume=resume, verify=verify,
        progress_path=progress_path, validate=validate,
    )
