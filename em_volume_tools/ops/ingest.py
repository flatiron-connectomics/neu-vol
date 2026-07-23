"""Ingest an image stack into a multiscale OME-NGFF 0.5 zarr v3 volume.

The first vertical slice (docs/DESIGN.md §7): image stack (TIFF/PNG glob or
multipage TIFF) -> level-0 zarr array -> strict level-by-level pyramid -> OME
group metadata. Every stage is a block-map over output chunks; the same code
runs serially (``client=None``) or across dask workers.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Sequence

from ..backends.base import Region, open_backend
from ..backends.tensorstore import TensorStoreBackend
from ..engine import Block, block_map, iter_blocks
from ..meta import VoxelMeta
from ..ngff import build_dataset, build_multiscales_attrs, ome_unit, validate_attrs, write_group_metadata
from ..profiles import get_profile, zarr3_create_spec
from ..pyramid import cumulative_factors, downsample_schedule, get_reducer, level_scale_translation

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable per-block workers (bound with functools.partial at call sites)
# --------------------------------------------------------------------------- #
def _input_region(out_region: Region, factor: Sequence[int], src_shape: Sequence[int]) -> Region:
    """Source (finer-level) region feeding an output block region."""
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
# Ingest
# --------------------------------------------------------------------------- #
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
    prof = get_profile(profile)
    if prof.format != "zarr3":
        raise NotImplementedError(f"ingest target format {prof.format!r} not supported yet (zarr3 only)")

    # --- source ---
    src_spec = {"backend": "image_stack", "source": src}
    src_backend = open_backend(src_spec)
    src_shape = src_backend.shape
    out_dtype = dtype or str(src_backend.dtype)
    base_meta = VoxelMeta(tuple(voxel_size), tuple(offset) if offset else (), units, tuple(axes))
    logger.info("ingest %s -> %s | shape=%s dtype=%s", src, dst, src_shape, out_dtype)

    dst = dst.rstrip("/")

    def _level_path(i: int) -> str:
        return os.path.join(dst, str(i))

    # --- level 0: create + copy ---
    lvl0 = TensorStoreBackend.create(
        zarr3_create_spec(prof, _level_path(0), src_shape, out_dtype,
                          dimension_names=axes, chunk=chunk, shard=shard),
        delete_existing=delete_existing,
    )
    lvl0_spec = lvl0.to_spec()
    blocks = list(iter_blocks(lvl0.shape, lvl0.chunks))
    block_map(
        blocks,
        functools.partial(_copy_block, src_spec=src_spec, dst_spec=lvl0_spec, out_dtype=out_dtype),
        client=client, npartitions=npartitions,
    )
    logger.info("level 0 written: shape=%s chunks=%s (%d blocks)", lvl0.shape, lvl0.chunks, len(blocks))

    level_shapes = [src_shape]
    prev_spec = lvl0_spec

    # --- pyramid: strict level-by-level ---
    schedule = downsample_schedule(src_shape, voxel_size, factors=factors,
                                   max_levels=max_levels, min_dim=min_dim) if multiscale else []
    prev_shape = src_shape
    for i, f in enumerate(schedule, start=1):
        lvl_shape = tuple(-(-s // ff) for s, ff in zip(prev_shape, f))
        lvl = TensorStoreBackend.create(
            zarr3_create_spec(prof, _level_path(i), lvl_shape, out_dtype,
                              dimension_names=axes, chunk=chunk, shard=shard),
            delete_existing=delete_existing,
        )
        lvl_spec = lvl.to_spec()
        blocks = list(iter_blocks(lvl.shape, lvl.chunks))
        block_map(
            blocks,
            functools.partial(_downsample_block, src_spec=prev_spec, dst_spec=lvl_spec,
                              factor=tuple(f), kind=kind),
            client=client, npartitions=npartitions,
        )
        logger.info("level %d written: shape=%s factor=%s (%d blocks)", i, lvl_shape, f, len(blocks))
        level_shapes.append(lvl_shape)
        prev_shape = lvl_shape
        prev_spec = lvl_spec

    # --- OME-NGFF 0.5 group metadata ---
    cum = cumulative_factors(schedule, len(axes))  # levels 0..L
    datasets = []
    scales = []
    for i, F in enumerate(cum):
        scale, translation = level_scale_translation(base_meta.voxel_size, base_meta.offset, F)
        datasets.append(build_dataset(str(i), scale, translation))
        scales.append(scale)
    attrs = build_multiscales_attrs(
        axis_names=axes,
        axis_types=["space"] * len(axes),
        axis_units=[ome_unit(units)] * len(axes),
        datasets=datasets,
        name=name,
        method_type=kind,
    )
    if validate:
        try:
            validate_attrs(attrs)
        except ImportError:
            logger.warning("jsonschema not available; skipping OME-NGFF validation")
    write_group_metadata({"driver": "file", "path": dst}, attrs)
    logger.info("wrote OME-NGFF 0.5 metadata: %d levels", len(datasets))

    return {
        "dst": dst,
        "num_levels": len(level_shapes),
        "level_shapes": level_shapes,
        "level_scales": scales,
        "dtype": out_dtype,
        "attrs": attrs,
    }
