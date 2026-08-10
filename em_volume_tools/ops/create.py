"""Create an **empty** multiscale volume, from explicit specs or a reference.

``convert``/``ingest`` create a volume and fill it in one pass, from one source.
This is the other shape of the problem: you have several small subvolumes — image
stacks, HDF5 files — that belong at known positions inside one larger frame, and
you want the frame to exist first so each piece can be written into it
independently (see :mod:`em_volume_tools.ops.write`).

Creating a level costs one ``zarr.json``: a zarr array with no chunks written reads
back as the fill value everywhere, so an empty pyramid is a few hundred bytes, not a
volume-sized allocation.

``like`` is the point of the module. Passing a reference volume copies its geometry —
level shapes, per-level voxel sizes and chunking, dtype, origin, axes — so the new
volume shares the reference's coordinate frame and a voxel index means the same thing
in both. That is what makes "write this chunk at (z, y, x) of level 2" well defined.
Any explicit argument overrides what the reference says.

**The reference's own level shapes are mirrored verbatim**, rather than recomputed
from a downsample schedule, whenever ``shape`` and ``factors`` are not overridden.
Recomputing is how two volumes that should be aligned end up one voxel apart: level
shapes are ceil-divided, so a schedule that differs from the reference's by a single
``min_dim``/``max_levels`` choice produces a different pyramid.

Either target format: zarr v3 (+ OME-NGFF group metadata) or neuroglancer-precomputed
(scales in one intrinsic ``info``), chosen by the storage profile exactly as
``convert`` chooses it. With no profile given it follows the reference's own format,
which is what "like this volume" ought to mean.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.tensorstore import TensorStoreBackend
from ..location import exists, is_local, join, to_kvstore
from ..ngff import (build_dataset, build_multiscales_attrs, ome_unit,
                    validate_attrs, write_group_metadata)
from ..profiles import (StorageProfile, get_profile, precomputed_create_spec,
                        zarr3_create_spec)
from ..pyramid import downsample_schedule, level_scale_translation
from ..source_metadata import describe, existing_levels, other_format_markers

logger = logging.getLogger(__name__)


def _spatial(shape: Sequence[int], has_channels: bool) -> tuple[int, ...]:
    return tuple(int(s) for s in (shape[1:] if has_channels else shape))


def _cum_factor(voxel: Sequence[float], voxel0: Sequence[float]) -> tuple[int, ...]:
    """Cumulative downsample factor of a level, from its voxel size relative to level 0."""
    return tuple(max(1, int(round(v / v0))) for v, v0 in zip(voxel, voxel0))


def _mirror_levels(ref: dict, has_channels: bool, voxel_size: Sequence[float],
                   limit: int | None) -> tuple[list[tuple[int, ...]], list[tuple[float, ...]]]:
    """Level shapes and voxel sizes copied from a reference volume."""
    indices = sorted(ref["levels"])[: limit or None]
    per_level = ref["level_voxel_sizes"]
    shapes = [_spatial(ref["levels"][i]["shape"], has_channels) for i in indices]
    voxels = []
    for n, i in enumerate(indices):
        if per_level and i < len(per_level):
            voxels.append(tuple(float(v) for v in per_level[i]))
        else:
            # No recorded per-level scale (a bare array, or metadata we don't trust).
            # Shape ratios are ceil-divided so they are inexact — 13750/3438 is 3.9994
            # — but rounding an integer factor out of them is safe.
            factor = tuple(max(1, int(round(a / b))) for a, b in zip(shapes[0], shapes[n]))
            voxels.append(tuple(float(v * f) for v, f in zip(voxel_size, factor)))
    return shapes, voxels


def _scheduled_levels(spatial0, voxel_size, *, factors, max_levels, min_dim,
                      limit) -> tuple[list[tuple[int, ...]], list[tuple[float, ...]]]:
    """Level shapes and voxel sizes from a computed downsample schedule."""
    schedule = downsample_schedule(spatial0, voxel_size, factors=factors,
                                   max_levels=max_levels, min_dim=min_dim)
    if limit is not None:
        schedule = schedule[: max(0, limit - 1)]
    shapes = [tuple(int(s) for s in spatial0)]
    voxels = [tuple(float(v) for v in voxel_size)]
    for f in schedule:
        shapes.append(tuple(-(-s // x) for s, x in zip(shapes[-1], f)))
        voxels.append(tuple(v * x for v, x in zip(voxels[-1], f)))
    return shapes, voxels


def _level_chunking(ref: dict | None, index: int, has_channels: bool,
                    chunk, shard) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """This level's (chunk, shard), preferring explicit arguments over the reference.

    A sharded reference level reports the shard as its *write* chunk and the inner
    chunk as its *read* chunk; reproducing it means splitting them back apart.
    """
    if chunk is not None:
        return tuple(chunk), (tuple(shard) if shard is not None else None)
    lv = (ref or {}).get("levels", {}).get(index) if ref else None
    if not lv or not lv.get("chunks"):
        return None, (tuple(shard) if shard is not None else None)
    write = _spatial(lv["chunks"], has_channels)
    read = _spatial(lv["read_chunks"], has_channels) if lv.get("read_chunks") else write
    if shard is not None:
        return read, tuple(shard)
    return (read, write) if read != write else (write, None)


def profile_for(format: str | None, ref: dict | None, dst: str) -> str:
    """The storage profile implied by a target format, a reference, and a destination.

    With no format given it follows the **reference's own format**: "like this volume"
    most usefully includes *what kind of volume it is*, and asking to mirror a
    precomputed frame and silently getting zarr is the sort of thing you only discover
    when the viewer cannot open the result. With neither, zarr v3.

    Shared with the CLI so `--format` cannot come to mean something different there.
    """
    fmt = str(format or (ref["format"] if ref else "zarr"))
    if fmt.startswith("neuroglancer_precomputed") or fmt == "precomputed":
        return "local-neuroglancer" if is_local(dst) else "s3-neuroglancer"
    if fmt.startswith("zarr"):
        return "local"
    raise ValueError(f"unknown format {format!r}; use 'zarr' or 'precomputed'")


def _precomputed_encoding(encoding: str | None, kind: str, dtype: str) -> str:
    """Chunk encoding for a precomputed scale, defaulted from the data kind.

    Same rule ``convert`` uses: ``compressed_segmentation`` for labels, ``raw``
    otherwise. It is checked against the dtype here rather than at create time
    because tensorstore's failure for the wrong pairing is not obviously about this.
    """
    if encoding is None:
        encoding = "compressed_segmentation" if kind == "segmentation" else "raw"
    if encoding == "compressed_segmentation" and dtype not in ("uint32", "uint64"):
        raise ValueError(
            f"compressed_segmentation requires uint32/uint64, got {dtype!r}; "
            f"pass encoding='raw' or a suitable dtype")
    return encoding


def plan_volume(
    dst: str,
    *,
    like: str | None = None,
    shape: Sequence[int] | None = None,
    dtype: str | None = None,
    voxel_size: Sequence[float] | None = None,
    offset: Sequence[float] | None = None,
    units: str | None = None,
    axes: Sequence[str] | None = None,
    has_channels: bool | None = None,
    format: str | None = None,
    profile: str | StorageProfile | None = None,
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    levels: int | None = None,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    kind: str | None = None,
    name: str = "image",
    encoding: str | None = None,
    compressed_segmentation_block_size: Sequence[int] = (8, 8, 8),
) -> dict:
    """Resolve the geometry of the volume ``create_volume`` would build. Writes nothing.

    Split out so ``--dry-run`` reports exactly what a real run would create, rather
    than a second implementation of the same arithmetic.

    ``format`` is ``"zarr"`` or ``"precomputed"``, defaulting to the reference's own
    (see :func:`profile_for`); ``profile`` overrides it with a named storage profile
    and its chunk/compressor defaults. ``encoding`` and
    ``compressed_segmentation_block_size`` apply to precomputed only.
    """
    ref = describe(like) if like else None
    prof = get_profile(profile if profile is not None
                       else profile_for(format, ref, dst))
    precomputed = prof.format == "neuroglancer_precomputed"

    if has_channels is None:
        has_channels = bool(ref["has_channels"]) if ref else False
    ref_meta = (ref["meta"] or {}) if ref else {}
    num_channels = int(ref["shape"][0]) if (ref and has_channels) else 1

    if voxel_size is None:
        voxel_size = ref_meta.get("voxel_size")
    if voxel_size is None:
        raise ValueError(
            "no voxel size: pass voxel_size=, or like= a volume that records one "
            "(image stacks and bare arrays do not)")
    voxel_size = tuple(float(v) for v in voxel_size)

    dtype = dtype or (ref["dtype"] if ref else None)
    if not dtype or dtype == "?":
        raise ValueError("no dtype: pass dtype=, or like= a volume to copy it from")
    axes = tuple(axes or ref_meta.get("spatial_axes") or ("z", "y", "x"))
    units = units or ref_meta.get("units") or "nm"
    kind = kind or ref_meta.get("kind") or "image"
    offset = tuple(float(o) for o in (offset if offset is not None
                                      else ref_meta.get("offset") or (0.0,) * len(axes)))

    if shape is not None:
        spatial0 = tuple(int(s) for s in shape)
    elif ref:
        spatial0 = _spatial(ref["shape"], has_channels)
    else:
        raise ValueError("no shape: pass shape=, or like= a volume to copy it from")
    for n, v in (("axes", axes), ("voxel_size", voxel_size), ("offset", offset)):
        if len(v) != len(spatial0):
            raise ValueError(f"{n} has {len(v)} entries but the volume has "
                             f"{len(spatial0)} spatial axes {spatial0}")

    if precomputed:
        encoding = _precomputed_encoding(encoding, kind, dtype)
        if shard is not None:
            raise ValueError(
                "precomputed sharding is not implemented (precomputed_create_spec "
                "takes no shard); drop --shard, or write zarr, which does shard")
        if ome_unit(units) != "nanometer":
            # precomputed's `resolution` has no unit field — it is nm by convention,
            # and a viewer will read whatever numbers are there as nm.
            logger.warning("precomputed records resolution in nanometres and has "
                           "nowhere to put units=%r; the numbers %s will be read as "
                           "nm", units, voxel_size)

    # Mirror the reference's pyramid verbatim unless something that changes it was
    # overridden. Recomputing a schedule that "should" match is how two volumes meant
    # to share a frame end up disagreeing by a voxel at level 3.
    mirrored = bool(ref) and shape is None and factors is None and bool(ref["levels"])
    if mirrored:
        level_shapes, level_voxels = _mirror_levels(ref, has_channels, voxel_size, levels)
    else:
        level_shapes, level_voxels = _scheduled_levels(
            spatial0, voxel_size, factors=factors, max_levels=max_levels,
            min_dim=min_dim, limit=levels)

    plan_levels = []
    for i, (lshape, lvoxel) in enumerate(zip(level_shapes, level_voxels)):
        c, s = _level_chunking(ref if mirrored else None, i, has_channels, chunk, shard)
        cum = _cum_factor(lvoxel, voxel_size)
        scale, translation = level_scale_translation(voxel_size, offset, cum)
        if has_channels:
            scale, translation = [1.0] + scale, [0.0] + translation
        plan_levels.append({
            "level": i,
            "shape": ((num_channels,) + tuple(lshape)) if has_channels else tuple(lshape),
            "voxel_size": tuple(lvoxel),
            "chunk": c, "shard": s,
            "scale": scale, "translation": translation,
            # precomputed indexes each scale from its own voxel_offset rather than
            # from 0, and carries the origin there instead of in a transform.
            "voxel_offset": [int(round(o / v)) for o, v in zip(offset, lvoxel)],
        })

    dim_names = (["c"] + list(axes)) if has_channels else list(axes)
    axis_types = (["channel"] if has_channels else []) + ["space"] * len(axes)
    axis_units = ([None] if has_channels else []) + [ome_unit(units)] * len(axes)
    # precomputed's multiscale metadata is intrinsic — tensorstore writes each scale
    # into one `info` as it is created — so there is no group document to build here.
    attrs = None if precomputed else build_multiscales_attrs(
        axis_names=dim_names, axis_types=axis_types, axis_units=axis_units,
        datasets=[build_dataset(str(lv["level"]), lv["scale"], lv["translation"])
                  for lv in plan_levels],
        name=name, method_type=kind)

    return {"dst": dst, "format": prof.format, "dtype": dtype, "kind": kind,
            "units": units, "axes": tuple(axes), "voxel_size": voxel_size,
            "offset": offset, "has_channels": has_channels,
            "num_channels": num_channels, "profile": prof,
            "mirrored": mirrored, "like": like,
            "levels": plan_levels, "num_levels": len(plan_levels),
            "dimension_names": dim_names, "attrs": attrs,
            "encoding": encoding if precomputed else None,
            "compressed_segmentation_block_size":
                tuple(compressed_segmentation_block_size) if precomputed else None,
            "name": name}


def _record_chunk(lv: dict, backend, has_channels: bool) -> None:
    """Fill in the level's actual chunk shape, which only the opened store knows.

    ``chunk`` is ``None`` in the plan whenever it was left to the profile, so the
    returned summary would otherwise be less informative than what was created.
    """
    if lv["chunk"] is None:
        lv["chunk"] = _spatial(backend.chunks, has_channels)
    logger.info("level %d: %s chunk %s%s", lv["level"], tuple(lv["shape"]),
                tuple(lv["chunk"]),
                f" shard {tuple(lv['shard'])}" if lv["shard"] else "")


def _create_zarr_levels(dst, plan: dict, *, overwrite: bool, validate: bool) -> None:
    base_kv = to_kvstore(dst)
    for lv in plan["levels"]:
        spec = zarr3_create_spec(
            plan["profile"], join(base_kv, str(lv["level"])), lv["shape"], plan["dtype"],
            has_channels=plan["has_channels"], num_channels=plan["num_channels"],
            dimension_names=plan["dimension_names"], chunk=lv["chunk"], shard=lv["shard"])
        _record_chunk(lv, TensorStoreBackend.create(spec, delete_existing=overwrite),
                      plan["has_channels"])
    if validate:
        try:
            validate_attrs(plan["attrs"])
        except ImportError:
            logger.warning("jsonschema not available; skipping OME-NGFF validation")
    write_group_metadata(base_kv, plan["attrs"])


def _create_precomputed_scales(dst, plan: dict, *, overwrite: bool) -> None:
    """Create every scale of a precomputed volume, which share one ``info``.

    That sharing is the whole difference from zarr, and it has one sharp edge:
    ``delete_existing`` must apply to **scale 0 only**. Every scale lives under the
    same prefix, so deleting on scale 1 would take the scale 0 just created with it.
    Creating scale *i* appends to the ``info`` the earlier scales wrote.
    """
    base_kv = to_kvstore(dst)
    pc_type = "segmentation" if plan["kind"] == "segmentation" else "image"
    for lv in plan["levels"]:
        spec = precomputed_create_spec(
            plan["profile"], base_kv, lv["shape"], plan["dtype"],
            resolution_zyx=lv["voxel_size"], scale_index=lv["level"],
            num_channels=plan["num_channels"], chunk=lv["chunk"],
            encoding=plan["encoding"], type_=pc_type,
            voxel_offset_zyx=lv["voxel_offset"],
            compressed_segmentation_block_size=plan["compressed_segmentation_block_size"])
        _record_chunk(lv,
                      TensorStoreBackend.create(spec,
                                                delete_existing=(overwrite and
                                                                 lv["level"] == 0)),
                      plan["has_channels"])


def create_volume(dst: str, *, overwrite: bool = False, validate: bool = True,
                  **kw: Any) -> dict:
    """Create an empty multiscale volume at ``dst``. See :func:`plan_volume`.

    Every level is created and none holds data — no chunk objects, so the whole volume
    is a handful of JSON documents until something writes into it.
    """
    plan = plan_volume(dst, **kw)
    base_kv = to_kvstore(dst)
    precomputed = plan["format"] == "neuroglancer_precomputed"
    marker = "info" if precomputed else "zarr.json"
    before = existing_levels(dst, plan["format"])

    # A volume of the OTHER format here is not something --overwrite can fix, and it is
    # the one case where proceeding is silently destructive: each format writes only
    # its own marker, so the two would sit in one directory with `detect_backend`
    # reading `info` first — the zarr becomes unreachable while its chunks still take
    # up the store. Neither --overwrite path cleans this up either: precomputed's
    # delete_existing wipes the prefix (taking a zarr with it, unannounced), and
    # zarr's deletes only its own level arrays, leaving a stale `info` in charge.
    foreign = other_format_markers(dst, plan["format"])
    if foreign:
        raise FileExistsError(
            f"{dst} already holds a different kind of volume ({', '.join(foreign)}); "
            f"creating a {plan['format']} volume here would leave both in one "
            f"directory and the older one would become unreachable — `info` is "
            f"checked before `zarr.json`. --overwrite will not resolve this. Delete "
            f"{dst} and create it again, or pick another destination.")

    if (before or exists(base_kv, marker)) and not overwrite:
        raise FileExistsError(
            f"{dst} already exists ({len(before)} level(s)); pass overwrite=True to "
            f"replace it. To add data to it instead, use `em-vol write`.")
    stale = [i for i in before if i >= plan["num_levels"]]
    if stale and not precomputed:
        # Left in place rather than deleted: dropping a level means deleting a whole
        # key prefix, which is not something to do implicitly. But `existing_levels`
        # probes upward, so these WILL be found by other commands and disagree with
        # the group metadata written below. (Precomputed cannot get here: recreating
        # it starts by deleting scale 0's prefix, which is the whole volume.)
        logger.warning("levels %s already exist above the %d being created and are "
                       "NOT removed; they will still be found on disk but are absent "
                       "from the group metadata. Delete them by hand if unwanted.",
                       stale, plan["num_levels"])

    if precomputed:
        _create_precomputed_scales(dst, plan, overwrite=overwrite)
    else:
        _create_zarr_levels(dst, plan, overwrite=overwrite, validate=validate)
    logger.info("created %s: %d empty %s, %s", dst, plan["num_levels"],
                "scales" if precomputed else "levels",
                f"encoding={plan['encoding']}" if precomputed
                else "OME-NGFF 0.5 metadata written")
    return {**plan, "created": True}
