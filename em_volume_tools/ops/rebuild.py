"""Regenerate a multiscale pyramid in place, from a level you trust.

When the coarse end of a pyramid is wrong — blank, aliased, or built from a bad
intermediate — the fix is to re-derive it, not to reconvert. Downsampling is
**cascaded** (level N is built from level N-1, not from level 0), so a single bad
level poisons everything above it and re-deriving must start at or below the
lowest good one.

    rebuild_pyramid("s3://bucket/prefix/volume", start_level=2, kind="image")

Levels at or below ``start_level`` are never opened for writing; the level itself
is the input. Everything above it is rebuilt with the same factors the original
run used, because the schedule is recomputed from level 0 rather than from the
seed (``downsample_schedule`` is iterative, so its tail matches, and the full
computation also yields the shapes needed to rewrite complete metadata).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..backends.base import open_backend
from ..source_metadata import detect_backend, read_source_metadata
from ..location import default_progress_path
from ..profiles import PROFILES
from ._multiscale import materialize_multiscale

logger = logging.getLogger(__name__)


def _profile_for(fmt: str) -> str:
    for name, prof in PROFILES.items():
        if prof.format == fmt:
            return name
    raise ValueError(f"no storage profile for format {fmt!r}")


def rebuild_pyramid(
    dst: str,
    *,
    start_level: int,
    kind: str | None = None,
    profile: str | None = None,
    voxel_size: Sequence[float] | None = None,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    chunk: Sequence[int] | None = None,
    shard: Sequence[int] | None = None,
    encoding: str | None = None,
    name: str = "image",
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = False,
    verify: bool = False,
    progress_path: str | None = None,
    validate: bool = True,
) -> dict:
    """Rebuild levels ``start_level+1 …`` of the volume at ``dst``, in place.

    ``start_level`` is the level to derive from — it is read, never written. **0 is
    valid and is the ordinary case**: rebuild the whole pyramid from full resolution.

    ``kind`` picks the reducer: ``"image"`` is a mask-weighted mean (anti-aliased),
    ``"segmentation"`` a label-preserving mode. It defaults to whatever the volume
    already records — precomputed stores it as ``info["type"]``, OME as the
    multiscales ``type`` — because guessing wrong is silent and destructive:
    averaging label ids invents ids that were never in the data. If the volume
    records nothing usable, it must be passed.

    Metadata (shape, voxel size, offset, axes) comes from the existing volume.
    ``voxel_size`` overrides it, for a volume whose recorded scale is itself the
    thing that was wrong.

    The progress manifest defaults to a name distinct from a conversion's, because
    reusing that one would mark the levels you are rebuilding as already done and
    the run would silently do nothing.
    """
    if start_level < 0:
        raise ValueError(f"start_level must be >= 0, got {start_level}")

    fmt = detect_backend(dst)
    if fmt is None:
        raise ValueError(
            f"could not detect a volume at {dst!r} (no info/zarr.json/.zarray)")
    spec = {"backend": fmt, "path": dst}
    meta = read_source_metadata(spec)
    if not meta and voxel_size is None:
        raise ValueError(
            f"{dst!r} has no readable coordinate metadata; pass voxel_size=")

    # Level 0 of the existing volume defines the pyramid the schedule must match.
    level0_spec = meta["data_spec"] if meta else spec
    level0 = open_backend(level0_spec)
    axes = tuple(meta["spatial_axes"]) if meta else ("z", "y", "x")
    has_channels = meta["has_channels"] if meta else len(level0.shape) == len(axes) + 1
    voxel_size = tuple(voxel_size if voxel_size is not None else meta["voxel_size"])
    offset = tuple(meta["offset"]) if (meta and meta.get("offset")) else (0.0,) * len(axes)

    if kind is None:
        kind = (meta or {}).get("kind")
        if kind is None:
            raise ValueError(
                f"{dst!r} does not record whether it is image or segmentation data, "
                "so the downsampling reducer cannot be inferred; pass "
                "kind='image' (mask-weighted mean) or kind='segmentation' "
                "(label-preserving mode). Averaging label ids would invent ids "
                "that are not in the data.")
        logger.info("kind=%r taken from the volume's metadata", kind)
    if kind not in ("image", "segmentation"):
        raise ValueError(f"kind must be 'image' or 'segmentation', got {kind!r}")

    # Keep the existing chunking by default: silently rechunking only the
    # regenerated levels would leave the pyramid internally inconsistent.
    if chunk is None:
        try:
            existing = tuple(int(c) for c in level0.chunks)
            chunk = existing[1:] if has_channels else existing
        except Exception:                       # backend without chunk info
            chunk = None

    if progress_path is None:
        progress_path = default_progress_path(dst).replace(
            ".progress.jsonl", f".regen-from-{start_level}.progress.jsonl")

    logger.info("rebuild %s from level %d | level0 shape=%s voxel_size=%s kind=%s",
                dst, start_level, tuple(level0.shape), voxel_size, kind)

    return materialize_multiscale(
        # Never read: a rebuild seeds from a level that already exists, so there is no
        # level-0 copy. It is passed because the signature is shared with `convert`.
        src_spec=level0_spec,
        src_shape=level0.shape,
        src_dtype=str(level0.dtype),
        dst=dst,
        profile=profile or _profile_for(fmt),
        voxel_size=voxel_size,
        offset=offset,
        units=(meta.get("units") if meta else None) or "nm",
        spatial_axes=axes,
        has_channels=has_channels,
        num_channels=int(level0.shape[0]) if has_channels else 1,
        dtype=None,
        kind=kind,
        encoding=encoding,
        multiscale=True,
        factors=factors,
        max_levels=max_levels,
        min_dim=min_dim,
        name=name,
        chunk=chunk,
        shard=shard,
        client=client,
        npartitions=npartitions,
        delete_existing=False,         # never destroy a volume we are repairing
        resume=resume,
        verify=verify,
        progress_path=progress_path,
        validate=validate,
        # An int, never None: every level of a rebuild is seeded from one that exists,
        # including level 0. None would mean "convert" — create level 0 and copy a
        # source into it — which is how `--start-level 0` used to die with
        # ALREADY_EXISTS on a volume it was supposed to be repairing.
        seed_level=start_level,
    )
