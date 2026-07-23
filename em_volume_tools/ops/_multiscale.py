"""Shared engine for materializing a multiscale volume from any source.

``ingest`` (image-stack source) and ``convert`` (any backend source) reduce to:
create level 0, block-copy the source in, build a strict level-by-level pyramid,
then finalize metadata. The copy+pyramid loop is target-agnostic (``_run_multiscale``
with a ``create_level`` callback); zarr v3 adds OME-NGFF group metadata after,
while precomputed's multiscale ``info`` is written incrementally at scale create.
See docs/DESIGN.md §6-7. A leading channel axis ``(c, z, y, x)`` is supported and
never downsampled.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Sequence

from ..backends.base import Region, open_backend
from ..backends.tensorstore import TensorStoreBackend
from ..engine import Block, block_map, iter_blocks
from ..ngff import build_dataset, build_multiscales_attrs, ome_unit, validate_attrs, write_group_metadata
from ..profiles import StorageProfile, get_profile, precomputed_create_spec, zarr3_create_spec
from ..pyramid import cumulative_factors, downsample_schedule, get_reducer, level_scale_translation

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable per-block workers
# --------------------------------------------------------------------------- #
def _input_region(out_region: Region, factor: Sequence[int], src_shape: Sequence[int]) -> Region:
    return tuple(
        slice(s.start * f, min(s.stop * f, dim))
        for s, f, dim in zip(out_region, factor, src_shape)
    )


def _copy_block(block: Block, *, src_spec: dict, dst_spec: dict, out_dtype: str) -> tuple:
    src = open_backend(src_spec)
    dst = open_backend(dst_spec)
    data = src.read_region(block.region)
    if str(data.dtype) != out_dtype:
        data = data.astype(out_dtype)
    dst.write_region(block.region, data)
    return ("ok", block.index)


def _downsample_block(block: Block, *, src_spec: dict, dst_spec: dict,
                      factor: tuple, kind: str) -> tuple:
    src = open_backend(src_spec)
    dst = open_backend(dst_spec)
    data = src.read_region(_input_region(block.region, factor, src.shape))
    out = get_reducer(kind)(data, factor)
    dst.write_region(block.region, out)
    return ("ok", block.index)


# --------------------------------------------------------------------------- #
# Shared copy + pyramid loop
# --------------------------------------------------------------------------- #
def _full_factor(spatial_factor: Sequence[int], has_channels: bool) -> tuple[int, ...]:
    return ((1,) + tuple(spatial_factor)) if has_channels else tuple(spatial_factor)


def _downsampled(shape: Sequence[int], factor: Sequence[int]) -> tuple[int, ...]:
    return tuple(-(-s // f) for s, f in zip(shape, factor))


def _run_multiscale(
    *,
    src_spec: dict,
    src_shape: Sequence[int],
    out_dtype: str,
    has_channels: bool,
    n_spatial: int,
    voxel_size: Sequence[float],
    kind: str,
    multiscale: bool,
    factors: Sequence[Sequence[int]] | None,
    max_levels: int,
    min_dim: int,
    create_level: Callable[[int, Sequence[int], Sequence[int]], TensorStoreBackend],
    client: Any | None,
    npartitions: int | None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Create + fill each level. Returns (level_shapes, cumulative_factors)."""
    src_shape = tuple(int(s) for s in src_shape)
    identity = tuple([1] * n_spatial)

    # level 0
    lvl0 = create_level(0, src_shape, identity)
    prev_spec = lvl0.to_spec()
    blocks = list(iter_blocks(lvl0.shape, lvl0.chunks))
    block_map(blocks, functools.partial(_copy_block, src_spec=src_spec,
                                        dst_spec=prev_spec, out_dtype=out_dtype),
              client=client, npartitions=npartitions)
    logger.info("level 0: shape=%s chunks=%s (%d blocks)", lvl0.shape, lvl0.chunks, len(blocks))

    level_shapes = [src_shape]
    cum = [identity]
    prev_shape = src_shape
    cur_cum = identity

    spatial_shape = src_shape[1:] if has_channels else src_shape
    schedule = (
        downsample_schedule(spatial_shape, voxel_size, factors=factors,
                            max_levels=max_levels, min_dim=min_dim)
        if multiscale else []
    )
    for i, f in enumerate(schedule, start=1):
        ff = _full_factor(f, has_channels)
        lvl_shape = _downsampled(prev_shape, ff)
        cur_cum = tuple(c * x for c, x in zip(cur_cum, f))
        lvl = create_level(i, lvl_shape, cur_cum)
        lvl_spec = lvl.to_spec()
        blocks = list(iter_blocks(lvl.shape, lvl.chunks))
        block_map(blocks, functools.partial(_downsample_block, src_spec=prev_spec,
                                            dst_spec=lvl_spec, factor=ff, kind=kind),
                  client=client, npartitions=npartitions)
        logger.info("level %d: shape=%s factor=%s (%d blocks)", i, lvl_shape, ff, len(blocks))
        level_shapes.append(lvl_shape)
        cum.append(cur_cum)
        prev_shape = lvl_shape
        prev_spec = lvl_spec

    return level_shapes, cum


# --------------------------------------------------------------------------- #
# Target: zarr v3 (+ OME-NGFF 0.5 metadata)
# --------------------------------------------------------------------------- #
def materialize_zarr_multiscale(
    *, src_spec, src_shape, src_dtype, dst, profile, voxel_size, offset, units,
    spatial_axes, has_channels, num_channels, dtype, kind, multiscale, factors,
    max_levels, min_dim, name, chunk, shard, client, npartitions, delete_existing, validate,
    encoding=None, compressed_segmentation_block_size=(8, 8, 8),  # precomputed-only; ignored here
) -> dict:
    prof = get_profile(profile)
    out_dtype = dtype or str(src_dtype)
    dst = dst.rstrip("/")
    dim_names = (["c"] + list(spatial_axes)) if has_channels else list(spatial_axes)

    def create_level(i, shape, cum):
        return TensorStoreBackend.create(
            zarr3_create_spec(prof, os.path.join(dst, str(i)), shape, out_dtype,
                              has_channels=has_channels, num_channels=num_channels,
                              dimension_names=dim_names, chunk=chunk, shard=shard),
            delete_existing=delete_existing,
        )

    level_shapes, cum = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
    )

    datasets, scales = [], []
    for i, F in enumerate(cum):
        s_scale, s_trans = level_scale_translation(voxel_size, offset, F)
        if has_channels:
            s_scale, s_trans = [1.0] + s_scale, [0.0] + s_trans
        datasets.append(build_dataset(str(i), s_scale, s_trans))
        scales.append(s_scale)

    axis_types = (["channel"] if has_channels else []) + ["space"] * len(spatial_axes)
    axis_units = ([None] if has_channels else []) + [ome_unit(units)] * len(spatial_axes)
    attrs = build_multiscales_attrs(axis_names=dim_names, axis_types=axis_types,
                                    axis_units=axis_units, datasets=datasets,
                                    name=name, method_type=kind)
    if validate:
        try:
            validate_attrs(attrs)
        except ImportError:
            logger.warning("jsonschema not available; skipping OME-NGFF validation")
    write_group_metadata({"driver": "file", "path": dst}, attrs)
    logger.info("wrote OME-NGFF 0.5 metadata: %d levels", len(datasets))

    return {"dst": dst, "format": "zarr3", "num_levels": len(level_shapes),
            "level_shapes": level_shapes, "level_scales": scales,
            "dtype": out_dtype, "attrs": attrs}


# --------------------------------------------------------------------------- #
# Target: neuroglancer-precomputed (intrinsic multiscale info)
# --------------------------------------------------------------------------- #
def materialize_precomputed_multiscale(
    *, src_spec, src_shape, src_dtype, dst, profile, voxel_size, offset, units,
    spatial_axes, has_channels, num_channels, dtype, kind, multiscale, factors,
    max_levels, min_dim, name, chunk, shard, client, npartitions, delete_existing, validate,
    encoding=None, compressed_segmentation_block_size=(8, 8, 8),
) -> dict:
    prof = get_profile(profile)
    out_dtype = dtype or str(src_dtype)
    dst = dst.rstrip("/")
    pc_type = "segmentation" if kind == "segmentation" else "image"

    # Default encoding: compressed_segmentation for label data, raw otherwise.
    if encoding is None:
        encoding = "compressed_segmentation" if kind == "segmentation" else "raw"
    if encoding == "compressed_segmentation" and out_dtype not in ("uint32", "uint64"):
        raise ValueError(
            f"compressed_segmentation requires uint32/uint64, got {out_dtype!r}; "
            "pass encoding='raw' or a suitable dtype"
        )

    def create_level(i, shape, cum):
        resolution = [v * c for v, c in zip(voxel_size, cum)]      # (z, y, x) nm
        voxel_offset = [int(round(o / r)) for o, r in zip(offset, resolution)] if any(offset) else [0, 0, 0]
        spec = precomputed_create_spec(
            prof, dst, shape, out_dtype, resolution_zyx=resolution, scale_index=i,
            num_channels=num_channels, chunk=chunk, encoding=encoding, type_=pc_type,
            voxel_offset_zyx=voxel_offset,
            compressed_segmentation_block_size=compressed_segmentation_block_size,
        )
        return TensorStoreBackend.create(spec, delete_existing=(delete_existing and i == 0))

    level_shapes, cum = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
    )
    scales = [[float(v * c) for v, c in zip(voxel_size, F)] for F in cum]
    logger.info("wrote precomputed multiscale info: %d scales (encoding=%s)", len(level_shapes), encoding)
    return {"dst": dst, "format": "neuroglancer_precomputed", "num_levels": len(level_shapes),
            "level_shapes": level_shapes, "level_scales": scales, "dtype": out_dtype,
            "encoding": encoding}


def materialize_multiscale(**kw) -> dict:
    """Dispatch to the zarr3 or precomputed materializer by profile format."""
    prof = get_profile(kw["profile"])
    if prof.format == "zarr3":
        return materialize_zarr_multiscale(**kw)
    if prof.format == "neuroglancer_precomputed":
        return materialize_precomputed_multiscale(**kw)
    raise NotImplementedError(f"target format {prof.format!r} not supported")
