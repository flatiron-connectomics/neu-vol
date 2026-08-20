"""Shared engine for materializing a multiscale volume from any source.

``ingest`` (image-stack source) and ``convert`` (any backend source) reduce to:
create level 0, block-copy the source in, build a strict level-by-level pyramid,
then finalize metadata. The copy+pyramid loop is target-agnostic (``_run_multiscale``
with a ``create_level`` callback); zarr v3 adds OME-NGFF group metadata after,
while precomputed's multiscale ``info`` is written incrementally at scale create.
See docs/DESIGN.md §6-7. A leading channel axis ``(c, z, y, x)`` is supported and
never downsampled.

Every per-block worker is wrapped in :func:`neu_vol.retry.with_retry`. At this
scale transient object-store failures are not hypothetical — one bad connection or DNS
lookup among tens of thousands of tasks would otherwise end a run that is succeeding.

Resume + sparsity: each block worker returns ``(index, status)`` where status is
``written`` / ``empty`` (equalled fill value, elided) / ``skipped`` (verify found
it present). The driver records these to a single-writer Manifest; on resume the
already-done blocks (written *or* empty) are filtered out before dispatch, so
empty chunks aren't reprocessed. ``verify=True`` ignores the manifest and instead
checks storage authoritatively per block.
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

from blockrun import Block, Manifest, block_map, iter_blocks

from ..backends.base import Region, open_backend
from ..backends.tensorstore import TensorStoreBackend
from ..location import default_progress_path, join, to_kvstore
from ..ngff import build_dataset, build_multiscales_attrs, ome_unit, validate_attrs, write_group_metadata
from ..profiles import get_profile, precomputed_create_spec, zarr3_create_spec
from ..pyramid import cumulative_factors, downsample_schedule, get_reducer, level_scale_translation
from ..retry import with_retry

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable per-block workers  -> return (block.index, status)
# --------------------------------------------------------------------------- #
def _input_region(out_region: Region, factor: Sequence[int], src_shape: Sequence[int]) -> Region:
    return tuple(
        slice(s.start * f, min(s.stop * f, dim))
        for s, f, dim in zip(out_region, factor, src_shape)
    )


# Per-block work is wrapped in `with_retry`, and the whole block is the retry unit:
# open, read and write are all idempotent for a fixed region, so repeating them
# produces the same result. Blocks fail fast (CLAUDE.md invariant 5) and that stays
# true — retry only removes the failures that would have fixed themselves. Without it
# a single transient network error anywhere across tens of thousands of tasks kills a
# run that is otherwise succeeding: observed as a connection reset on an 11 TB copy,
# and again as one worker failing to resolve the S3 hostname (curl_code=6) two seconds
# into an 85,536-task level. A permanent error (403, malformed spec) is re-raised
# immediately — see retry.is_transient, which checks permanent markers first.
def _copy_block(block: Block, *, src_spec: dict, dst_spec: dict, out_dtype: str,
                verify: bool = False) -> tuple:
    def once():
        dst = open_backend(dst_spec)
        if verify and dst.is_region_stored(block.region):
            return (block.index, "skipped")
        src = open_backend(src_spec)
        data = src.read_region(block.region)
        if str(data.dtype) != out_dtype:
            data = data.astype(out_dtype)
        if not data.any():                  # all fill value -> elide (sparse-friendly)
            return (block.index, "empty")
        dst.write_region(block.region, data)
        return (block.index, "written")

    return with_retry(once, label=f"copy block {block.index}")


def _downsample_block(block: Block, *, src_spec: dict, dst_spec: dict,
                      factor: tuple, kind: str, verify: bool = False) -> tuple:
    def once():
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

    return with_retry(once, label=f"downsample block {block.index}")


# --------------------------------------------------------------------------- #
# Shared copy + pyramid loop
# --------------------------------------------------------------------------- #
def _full_factor(spatial_factor: Sequence[int], has_channels: bool) -> tuple[int, ...]:
    return ((1,) + tuple(spatial_factor)) if has_channels else tuple(spatial_factor)


def _downsampled(shape: Sequence[int], factor: Sequence[int]) -> tuple[int, ...]:
    return tuple(-(-s // f) for s, f in zip(shape, factor))


#: Ceiling on one task's in-memory source array. Tasks are sized from the source and
#: destination chunkings, and that product can be enormous — a full-width z-slab of a
#: PNG stack is 28.8 GB — so it is capped and the shortfall paid as bounded re-reads.
DEFAULT_TASK_BYTES = 4 * 1024 ** 3


def plan_task_shape(src_chunks, dst_chunks, dst_shape, itemsize=1,
                    max_bytes=DEFAULT_TASK_BYTES):
    """A task shape that covers whole source AND whole destination chunks.

    Tasks used to be one destination chunk each, ignoring the source entirely. When
    the source's unit is larger, every task re-fetches a source unit its neighbours
    also need — measured on a CloudVolume source with 128x2048x2048 chunks written to
    128^3: **256 destination chunks per source chunk, so 256x the reads** (~331 TB for
    1.5 TB of data). For a PNG stack, where a slice cannot be partially decoded, the
    same effect was 13,886x.

    The natural unit is the per-axis LCM of the two chunkings, clamped to the volume.
    Where that exceeds ``max_bytes`` it is halved along its largest axis until it
    fits, **always staying a multiple of the destination chunk** — that multiple is
    not negotiable, because two tasks sharing a destination chunk race on a partial
    write (see CLAUDE.md). Falling below a whole source unit only costs re-reads.
    """
    if not src_chunks or len(src_chunks) != len(dst_chunks):
        return tuple(dst_chunks)                       # nothing to align to

    shape = []
    for src, dst, extent in zip(src_chunks, dst_chunks, dst_shape):
        src, dst = max(1, int(src)), max(1, int(dst))
        unit = src * dst // math.gcd(src, dst)         # lcm
        # No point exceeding the volume; keep it a dst multiple on the way down.
        if unit > extent:
            unit = max(dst, (extent // dst) * dst) if extent >= dst else dst
        shape.append(unit)

    def nbytes(s):
        return math.prod(s) * itemsize

    # Halve the largest axis until it fits, never below one destination chunk.
    while nbytes(shape) > max_bytes:
        i = max(range(len(shape)),
                key=lambda k: (shape[k] // dst_chunks[k], shape[k]))
        dst = max(1, int(dst_chunks[i]))
        multiples = shape[i] // dst
        if multiples <= 1:
            break                                      # cannot shrink further
        shape[i] = max(dst, (multiples // 2) * dst)
    return tuple(shape)


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


class NothingStored(RuntimeError):
    """A level to be downsampled from holds no stored chunks at all."""


def _stored_cells_fn(dst: str, fmt: str):
    """``(level, cell_shape) -> {stored chunk cells}`` for this destination.

    Deferred to :func:`neu_vol.ops.annotate.occupied_cells`, which already knows
    how each format spells a chunk key — precomputed's ``x0-x1_y0-y1_z0-z1`` in xyz under
    a per-scale prefix, zarr v3's ``c/z/y/x``. It raises ``NoOccupancy`` on a **sharded**
    level, which is the right answer rather than a wrong one: a shard hides which of its
    chunks exist, so presence stops being the same question as occupancy.
    """
    def stored_cells(level: int, cell: tuple[int, ...]) -> set:
        from .annotate import occupied_cells

        return occupied_cells(dst, fmt, level, cell)

    return stored_cells


def _sparse_skip(stored_cells, src_level, src_shape, src_chunks, factor, has_channels,
                 *, strict: bool):
    """A predicate: does this output block's input contain no stored chunk at all?

    **This is exact, not a heuristic, and that is what separates it from the occupancy
    prefilters elsewhere** (CLAUDE.md invariant 6, where coarse-scale *label* occupancy
    misses blocks and needs dilating). Here the question is not "does this region
    probably hold labels" but "does the source object exist": TensorStore never persists
    an all-fill chunk, so a chunk with no object *is* all fill. If none of the source
    chunks a block reads exists, its input is entirely fill, its output is entirely fill,
    and :func:`_downsample_block` would return ``"empty"`` and write nothing. Skipping
    changes not one byte of output — it only skips the read that discovers the zeros.

    The error direction is safe too: a writer that *did* store all-fill chunks (anything
    not TensorStore) only makes this keep blocks it needn't, costing a read.

    One listing per level, not one probe per block — an existence check per task is the
    round trip we are trying to avoid.
    """
    spatial = slice(1, None) if has_channels else slice(None)
    cell = tuple(int(c) for c in src_chunks[spatial])
    cells = stored_cells(src_level, cell)
    if not cells:
        # An empty SEED means the run would write nothing and report success — invariant
        # 4's failure mode, and what a wrong --start-level looks like. An empty level
        # part-way up is ordinary: mode-reducing a handful of labelled voxels among
        # background yields background, so sparse data legitimately dies out below the
        # top of the pyramid. Nothing above an empty level can hold data either.
        if strict:
            raise NothingStored(
                f"level {src_level} has no stored chunks, so every task above it would "
                f"be skipped and the run would write nothing while reporting success. "
                f"If the level really is empty there is nothing to downsample; "
                f"otherwise check --start-level and the volume, and re-run without "
                f"--sparse to be certain.")
        logger.info("level %d holds no stored chunks: the data has reduced to nothing, "
                    "so this level and everything above it stay empty", src_level)
        return lambda block: True
    logger.info("level %d holds %d stored chunks of %s", src_level, len(cells), cell)

    def skip(block) -> bool:
        region = _input_region(block.region, factor, src_shape)[spatial]
        spans = [range(s.start // c, (s.stop - 1) // c + 1) for s, c in zip(region, cell)]
        return not any(idx in cells for idx in itertools.product(*spans))

    return skip


def _run_level(manifest, level, backend, worker_factory, *, resume, verify, client,
               npartitions, task_shape=None, skip=None):
    """Dispatch one level's blocks, filtering already-done ones and recording results.

    The task total goes into the manifest **before** dispatch. It is the only place
    it is known — a task may span many destination chunks (see ``plan_task_shape``),
    so a reader counting the chunk grid gets a denominator up to 256x too large, and
    `neu-vol progress` did exactly that. Logging is likewise before *and* after: the
    single after-the-fact line this replaced arrived only once the level finished,
    reporting "0 already done" about work that had by then all been done.

    ``skip`` drops blocks before anything else, and therefore before the total is
    recorded: the denominator has to be the work actually dispatched, or `neu-vol
    progress` reports against a grid the run never intended to cover.
    """
    unit = tuple(int(c) for c in (task_shape or backend.chunks))
    blocks = list(iter_blocks(backend.shape, unit))
    grid_total = len(blocks)
    skipped = 0
    if skip is not None:
        blocks = [b for b in blocks if not skip(b)]
        skipped = grid_total - len(blocks)
        logger.info("level %d: %d of %d grid tasks have input data; skipping %d whose "
                    "input holds no stored chunk (their output is all fill, which is "
                    "written nowhere)", level, len(blocks), grid_total, skipped)
    total = len(blocks)
    manifest.record_meta(level, total=total, task_shape=list(unit),
                         shape=[int(s) for s in backend.shape],
                         chunks=[int(c) for c in backend.chunks],
                         # Recorded whenever the filter ran, even at zero, so a reader can
                         # tell "nothing was skipped" from "nothing was filtered".
                         **({"grid_total": grid_total, "skipped_empty": skipped}
                            if skip is not None else {}))
    if resume and not verify:
        done = manifest.done_keys(level)
        blocks = [b for b in blocks if b.index not in done]
    logger.info("level %d: %d of %d tasks of %s to run (%d already done, "
                "shape=%s chunks=%s)",
                level, len(blocks), total, unit, total - len(blocks),
                tuple(int(s) for s in backend.shape), tuple(int(c) for c in backend.chunks))
    on_result = lambda res, lvl=level: manifest.record(lvl, res)  # noqa: E731
    t0 = time.monotonic()
    block_map(blocks, worker_factory(verify=verify), client=client,
              npartitions=npartitions, on_result=on_result)
    logger.info("level %d finished in %s: %s", level, _elapsed(time.monotonic() - t0),
                manifest.counts(level) or "nothing to do")


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
    seed_level: int | None = None,
    open_level: Callable[[int, Sequence[int], Sequence[int]], TensorStoreBackend] | None = None,
    sparse: bool = False,
    stored_cells: Callable[[int, tuple[int, ...]], set] | None = None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict[str, int]]:
    """Create + fill each level. Returns (level_shapes, cumulative_factors, status_counts).

    ``seed_level=None`` is a conversion: level 0 is **created** and ``src_spec`` is
    copied into it, then the pyramid cascades from there.

    An integer ``seed_level`` rebuilds an **existing** pyramid in place: that level is
    read as the input, levels below it are left alone, and everything above is
    regenerated. **Zero is a legitimate seed** — rebuilding the whole pyramid from
    level 0 is the ordinary repair — which is why this is a separate parameter rather
    than ``start_level == 0`` doing double duty. It did once, and a rebuild from level
    0 took the conversion path and died trying to create a level that already existed.

    The schedule is still computed from level 0, so regenerated levels keep the factors
    the original run used — ``downsample_schedule`` is iterative, so its tail from the
    seed is what a schedule rooted there would produce anyway, and computing the whole
    thing also yields the shapes needed to rewrite complete metadata.

    The seed is obtained through ``open_level``, never ``create_level``: opening
    for creation with ``resume=False`` would recreate the very level being
    regenerated from, destroying the input.

    ``sparse`` skips pyramid tasks whose input holds no stored chunk — exact, not a
    guess, see :func:`_sparse_skip`. It applies to levels 1..L only: **level 0 is never
    filtered**, because its source is foreign and its emptiness is not ours to know. So
    it makes a `downsample`/`rebuild` of a sparse volume nearly free while leaving a
    `convert`'s level-0 copy exactly as expensive as it was.
    """
    src_shape = tuple(int(s) for s in src_shape)
    identity = tuple([1] * n_spatial)

    spatial_shape = src_shape[1:] if has_channels else src_shape
    schedule = (
        downsample_schedule(spatial_shape, voxel_size, factors=factors,
                            max_levels=max_levels, min_dim=min_dim)
        if multiscale else []
    )
    if seed_level is not None and not 0 <= seed_level <= len(schedule):
        raise ValueError(
            f"start_level {seed_level} out of range: this volume's schedule has "
            f"levels 0..{len(schedule)}")
    if seed_level is not None and open_level is None:
        raise ValueError("seeding from an existing level requires open_level")

    # Shapes/cumulative factors for EVERY level, including the ones left alone —
    # the caller needs them to write metadata describing the whole pyramid.
    level_shapes = [src_shape]
    cum = [identity]
    for f in schedule:
        ff = _full_factor(f, has_channels)
        level_shapes.append(_downsampled(level_shapes[-1], ff))
        cum.append(tuple(c * x for c, x in zip(cum[-1], f)))

    manifest = Manifest(progress_path)
    if resume:
        manifest.load()
    else:
        manifest.reset()

    try:
        if seed_level is None:
            lvl0 = create_level(0, src_shape, identity)
            prev_spec = lvl0.to_spec()
            # Level 0 is the ONLY level that reads a foreign source, so it is the only
            # one whose task shape has to reconcile two chunkings. Levels above read
            # the level below, where the two already agree.
            task_shape = None
            try:
                src_chunks = open_backend(src_spec).chunks
                task_shape = plan_task_shape(
                    src_chunks, lvl0.chunks, lvl0.shape,
                    itemsize=np.dtype(out_dtype).itemsize)
            except Exception:                 # a source need not expose chunking
                src_chunks = None
            if task_shape and tuple(task_shape) != tuple(lvl0.chunks):
                per_task = math.prod(t // c for t, c in zip(task_shape, lvl0.chunks))
                logger.info(
                    "level 0 task shape %s (source chunks %s, destination chunks %s): "
                    "%d destination chunks per task, so each source chunk is read once "
                    "rather than %d times",
                    task_shape, tuple(src_chunks), tuple(lvl0.chunks), per_task, per_task)
            _run_level(manifest, 0, lvl0,
                       lambda *, verify: functools.partial(_copy_block, src_spec=src_spec,
                                                            dst_spec=prev_spec, out_dtype=out_dtype,
                                                            verify=verify),
                       resume=resume, verify=verify, client=client, npartitions=npartitions,
                       task_shape=task_shape)
            prev_shape, prev_chunks = lvl0.shape, lvl0.chunks
        else:
            seed = open_level(seed_level, level_shapes[seed_level], cum[seed_level])
            got = tuple(int(s) for s in seed.shape)
            want = tuple(int(s) for s in level_shapes[seed_level])
            if got != want:
                raise ValueError(
                    f"level {seed_level} has shape {got}, but the pyramid schedule "
                    f"predicts {want}. Regenerating on top of it would produce a "
                    f"volume whose levels disagree; check voxel_size/factors, or "
                    f"rebuild from a level that matches.")
            prev_spec = seed.to_spec()
            prev_shape, prev_chunks = seed.shape, seed.chunks
            logger.info("rebuilding from existing level %d %s (level%s below it "
                        "untouched)", seed_level, got,
                        f"s 0-{seed_level - 1}" if seed_level else " 0 is the seed; none")

        if sparse and stored_cells is None:
            raise ValueError("sparse=True needs stored_cells, which reports which chunk "
                             "objects a level actually has")
        for i in range((0 if seed_level is None else seed_level) + 1, len(schedule) + 1):
            ff = _full_factor(schedule[i - 1], has_channels)
            lvl = create_level(i, level_shapes[i], cum[i])
            lvl_spec = lvl.to_spec()
            src_for_lvl = prev_spec
            # The filter reads the level BELOW, which by now is written — for a rebuild
            # that is the seed, for a conversion the level 0 just copied.
            first = i - 1 == (0 if seed_level is None else seed_level)
            skip = (_sparse_skip(stored_cells, i - 1, prev_shape, prev_chunks, ff,
                                 has_channels, strict=first) if sparse else None)
            _run_level(manifest, i, lvl,
                       lambda *, verify, s=src_for_lvl, d=lvl_spec, fac=ff: functools.partial(
                           _downsample_block, src_spec=s, dst_spec=d, factor=fac, kind=kind,
                           verify=verify),
                       resume=resume, verify=verify, client=client, npartitions=npartitions,
                       skip=skip)
            prev_spec, prev_shape, prev_chunks = lvl_spec, lvl.shape, lvl.chunks

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
    resume=False, verify=False, progress_path=None, seed_level=None, sparse=False,
    encoding=None, compressed_segmentation_block_size=(8, 8, 8),  # precomputed-only; ignored here
) -> dict:
    prof = get_profile(profile)
    out_dtype = dtype or str(src_dtype)
    base_kv = to_kvstore(dst)
    progress_path = progress_path or default_progress_path(dst)
    dim_names = (["c"] + list(spatial_axes)) if has_channels else list(spatial_axes)

    def _level_spec(i, shape):
        return zarr3_create_spec(prof, join(base_kv, str(i)), shape, out_dtype,
                                 has_channels=has_channels, num_channels=num_channels,
                                 dimension_names=dim_names, chunk=chunk, shard=shard)

    def create_level(i, shape, cum):
        # A rebuild targets levels that already exist, so it must reopen them
        # rather than create: creating over an existing level is an error.
        return TensorStoreBackend.open_or_create(
            _level_spec(i, shape), resume=resume or verify or seed_level is not None,
            delete_existing=delete_existing)

    def open_level(i, shape, cum):
        """Open an existing level. Never creates — this is the rebuild seed."""
        return TensorStoreBackend.open({"backend": "zarr3",
                                        "kvstore": join(base_kv, str(i))})

    level_shapes, cum, counts = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
        resume=resume, verify=verify, progress_path=progress_path,
        seed_level=seed_level, open_level=open_level, sparse=sparse,
        stored_cells=_stored_cells_fn(dst, "zarr3"),
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
    resume=False, verify=False, progress_path=None, seed_level=None, sparse=False,
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
        # A rebuild reopens existing scales; creating over one is an error.
        return TensorStoreBackend.open_or_create(
            spec, resume=resume or verify or seed_level is not None,
            delete_existing=(delete_existing and i == 0))

    def open_level(i, shape, cum):
        """Open an existing scale. Never creates — this is the rebuild seed."""
        return TensorStoreBackend.open({"backend": "neuroglancer_precomputed",
                                        "kvstore": dict(base_kv), "scale_index": i})

    level_shapes, cum, counts = _run_multiscale(
        src_spec=src_spec, src_shape=src_shape, out_dtype=out_dtype,
        has_channels=has_channels, n_spatial=len(spatial_axes), voxel_size=voxel_size,
        kind=kind, multiscale=multiscale, factors=factors, max_levels=max_levels,
        min_dim=min_dim, create_level=create_level, client=client, npartitions=npartitions,
        resume=resume, verify=verify, progress_path=progress_path,
        seed_level=seed_level, open_level=open_level, sparse=sparse,
        stored_cells=_stored_cells_fn(dst, "neuroglancer_precomputed"),
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
