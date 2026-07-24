"""Shared engine for materializing a multiscale volume from any source.

``ingest`` (image-stack source) and ``convert`` (any backend source) reduce to:
create level 0, block-copy the source in, build a strict level-by-level pyramid,
then finalize metadata. The copy+pyramid loop is target-agnostic (``_run_multiscale``
with a ``create_level`` callback); zarr v3 adds OME-NGFF group metadata after,
while precomputed's multiscale ``info`` is written incrementally at scale create.
See docs/DESIGN.md §6-7. A leading channel axis ``(c, z, y, x)`` is supported and
never downsampled.

Resume + sparsity: each block worker returns ``(index, status)`` where status is
``written`` / ``empty`` (equalled fill value, elided) / ``skipped`` (verify found
it present). The driver records these to a single-writer Manifest; on resume the
already-done blocks (written *or* empty) are filtered out before dispatch, so
empty chunks aren't reprocessed. ``verify=True`` ignores the manifest and instead
checks storage authoritatively per block.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Sequence

from ..backends.base import Region, open_backend
from ..backends.tensorstore import TensorStoreBackend
from ..engine import Block, block_map, iter_blocks
from ..location import default_progress_path, join, to_kvstore
from ..manifest import Manifest
from ..ngff import build_dataset, build_multiscales_attrs, ome_unit, validate_attrs, write_group_metadata
from ..profiles import get_profile, precomputed_create_spec, zarr3_create_spec
from ..pyramid import cumulative_factors, downsample_schedule, get_reducer, level_scale_translation

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable per-block workers  -> return (block.index, status)
# --------------------------------------------------------------------------- #
def _input_region(out_region: Region, factor: Sequence[int], src_shape: Sequence[int]) -> Region:
    return tuple(
        slice(s.start * f, min(s.stop * f, dim))
        for s, f, dim in zip(out_region, factor, src_shape)
    )


def _copy_block(block: Block, *, src_spec: dict, dst_spec: dict, out_dtype: str,
                verify: bool = False) -> tuple:
    dst = open_backend(dst_spec)
    if verify and dst.is_region_stored(block.region):
        return (block.index, "skipped")
    src = open_backend(src_spec)
    data = src.read_region(block.region)
    if str(data.dtype) != out_dtype:
        data = data.astype(out_dtype)
    if not data.any():                      # all fill value -> elide (sparse-friendly)
        return (block.index, "empty")
    dst.write_region(block.region, data)
    return (block.index, "written")


def _downsample_block(block: Block, *, src_spec: dict, dst_spec: dict,
                      factor: tuple, kind: str, verify: bool = False) -> tuple:
    dst = open_backend(dst_spec)
    if verify and dst.is_region_stored(block.region):
        return (block.index, "skipped")
    src = open_backend(src_spec)
    data = src.read_region(_input_region(block.region, factor, src.shape))
    out = get_reducer(kind)(data, factor)
    if not out.any():
        return (block.index, "empty")
    dst.write_region(block.region, out)
    return (block.index, "written")


# --------------------------------------------------------------------------- #
# Shared copy + pyramid loop
# --------------------------------------------------------------------------- #
def _full_factor(spatial_factor: Sequence[int], has_channels: bool) -> tuple[int, ...]:
    return ((1,) + tuple(spatial_factor)) if has_channels else tuple(spatial_factor)


def _downsampled(shape: Sequence[int], factor: Sequence[int]) -> tuple[int, ...]:
    return tuple(-(-s // f) for s, f in zip(shape, factor))


def _run_level(manifest, level, backend, worker_factory, *, resume, verify, client, npartitions):
    """Dispatch one level's blocks, filtering already-done ones and recording results."""
    blocks = list(iter_blocks(backend.shape, backend.chunks))
    total = len(blocks)
    if resume and not verify:
        done = manifest.done_indices(level)
        blocks = [b for b in blocks if b.index not in done]
    on_result = lambda res, lvl=level: manifest.record(lvl, res)  # noqa: E731
    block_map(blocks, worker_factory(verify=verify), client=client,
              npartitions=npartitions, on_result=on_result)
    logger.info("level %d: %d blocks (%d already done, shape=%s chunks=%s)",
                level, total, total - len(blocks), backend.shape, backend.chunks)


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
    resume: bool,
    verify: bool,
    progress_path: str | None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict[str, int]]:
    """Create + fill each level. Returns (level_shapes, cumulative_factors, status_counts)."""
    src_shape = tuple(int(s) for s in src_shape)
    identity = tuple([1] * n_spatial)

    manifest = Manifest(progress_path)
    if resume:
        manifest.load()
    else:
        manifest.reset()

    try:
        # level 0 (copy)
        lvl0 = create_level(0, src_shape, identity)
        prev_spec = lvl0.to_spec()
        _run_level(manifest, 0, lvl0,
                   lambda *, verify: functools.partial(_copy_block, src_spec=src_spec,
                                                        dst_spec=prev_spec, out_dtype=out_dtype,
                                                        verify=verify),
                   resume=resume, verify=verify, client=client, npartitions=npartitions)

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
            src_for_lvl = prev_spec
            _run_level(manifest, i, lvl,
                       lambda *, verify, s=src_for_lvl, d=lvl_spec, fac=ff: functools.partial(
                           _downsample_block, src_spec=s, dst_spec=d, factor=fac, kind=kind,
                           verify=verify),
                       resume=resume, verify=verify, client=client, npartitions=npartitions)
            level_shapes.append(lvl_shape)
            cum.append(cur_cum)
            prev_shape = lvl_shape
            prev_spec = lvl_spec

        return level_shapes, cum, manifest.counts()
    finally:
        manifest.close()


# --------------------------------------------------------------------------- #
# Target: zarr v3 (+ OME-NGFF 0.5 metadata)
# --------------------------------------------------------------------------- #
def materialize_zarr_multiscale(
    *, src_spec, src_shape, src_dtype, dst, profile, voxel_size, offset, units,
    spatial_axes, has_channels, num_channels, dtype, kind, multiscale, factors,
    max_levels, min_dim, name, chunk, shard, client, npartitions, delete_existing, validate,
    resume=False, verify=False, progress_path=None,
    encoding=None, compressed_segmentation_block_size=(8, 8, 8),  # precomputed-only; ignored here
) -> dict:
    prof = get_profile(profile)
    out_dtype = dtype or str(src_dtype)
    base_kv = to_kvstore(dst)
    progress_path = progress_path or default_progress_path(dst)
    dim_names = (["c"] + list(spatial_axes)) if has_channels else list(spatial_axes)

    def create_level(i, shape, cum):
        return TensorStoreBackend.open_or_create(
            zarr3_create_spec(prof, join(base_kv, str(i)), shape, out_dtype,
                              has_channels=has_channels, num_channels=num_channels,
                              dimension_names=dim_names, chunk=chunk, shard=shard),
            resume=resume or verify, delete_existing=delete_existing,
        )

    level_shapes, cum, counts = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
        resume=resume, verify=verify, progress_path=progress_path,
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
    write_group_metadata(base_kv, attrs)
    logger.info("wrote OME-NGFF 0.5 metadata: %d levels", len(datasets))

    return {"dst": dst, "format": "zarr3", "num_levels": len(level_shapes),
            "level_shapes": level_shapes, "level_scales": scales,
            "dtype": out_dtype, "attrs": attrs, "status_counts": counts,
            "progress_path": progress_path}


# --------------------------------------------------------------------------- #
# Target: neuroglancer-precomputed (intrinsic multiscale info)
# --------------------------------------------------------------------------- #
def materialize_precomputed_multiscale(
    *, src_spec, src_shape, src_dtype, dst, profile, voxel_size, offset, units,
    spatial_axes, has_channels, num_channels, dtype, kind, multiscale, factors,
    max_levels, min_dim, name, chunk, shard, client, npartitions, delete_existing, validate,
    resume=False, verify=False, progress_path=None,
    encoding=None, compressed_segmentation_block_size=(8, 8, 8),
) -> dict:
    prof = get_profile(profile)
    out_dtype = dtype or str(src_dtype)
    base_kv = to_kvstore(dst)
    progress_path = progress_path or default_progress_path(dst)
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
            prof, base_kv, shape, out_dtype, resolution_zyx=resolution, scale_index=i,
            num_channels=num_channels, chunk=chunk, encoding=encoding, type_=pc_type,
            voxel_offset_zyx=voxel_offset,
            compressed_segmentation_block_size=compressed_segmentation_block_size,
        )
        # For precomputed, delete_existing must apply only to scale 0 (shared volume).
        return TensorStoreBackend.open_or_create(
            spec, resume=resume or verify, delete_existing=(delete_existing and i == 0))

    level_shapes, cum, counts = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
        resume=resume, verify=verify, progress_path=progress_path,
    )
    scales = [[float(v * c) for v, c in zip(voxel_size, F)] for F in cum]
    logger.info("wrote precomputed multiscale info: %d scales (encoding=%s)", len(level_shapes), encoding)
    return {"dst": dst, "format": "neuroglancer_precomputed", "num_levels": len(level_shapes),
            "level_shapes": level_shapes, "level_scales": scales, "dtype": out_dtype,
            "encoding": encoding, "status_counts": counts, "progress_path": progress_path}


def materialize_multiscale(**kw) -> dict:
    """Dispatch to the zarr3 or precomputed materializer by profile format."""
    prof = get_profile(kw["profile"])
    if prof.format == "zarr3":
        return materialize_zarr_multiscale(**kw)
    if prof.format == "neuroglancer_precomputed":
        return materialize_precomputed_multiscale(**kw)
    raise NotImplementedError(f"target format {prof.format!r} not supported")
