"""em-vol: convert, downsample and inspect large 3D volumes, locally or on SLURM.

The command-line entry point for this package. Every write is block-mapped over a dask
cluster (local or SLURM, chosen by ``--config``) and resumable, so an interrupted run
continues where it stopped rather than starting over.

    em-vol info    <volume>                      # what is it, what levels exist
    em-vol convert --src ... --dst ...           # build a multiscale volume
    em-vol downsample <volume> --start-level 2   # rebuild levels above a trusted one
    em-vol progress <volume>                     # chunks written, per level
    em-vol create  <dst> --like <reference>      # an EMPTY volume in a known frame
    em-vol write   <volume> --src ... --offset   # put one subvolume into it
    em-vol annotations <volume>                  # a viewer layer marking where data is

``create`` + ``write`` are the small-pieces path, and they are not block-mapped:
``create`` lays out an empty volume (optionally copying a reference's frame exactly),
then each ``write`` places one image stack / HDF5 / array into one level of it at a
voxel offset. Single-scale by design — run ``downsample`` afterwards if the result
needs a pyramid.

``python -m em_volume_tools`` is equivalent. Run ``em-vol <subcommand> --help`` for the
arguments of each.

**--config takes a bundled template name or a path, and is repeatable**, deep-merged
left to right — so a site config is the few keys that differ from a template rather
than a fork of it. Templates and validation live in :mod:`em_blockrun.dask_config`;
site-specific configs do not belong in this repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

from em_blockrun import bundled_configs, start_dask

log = logging.getLogger("em-vol")


# --------------------------------------------------------------------------- #
# shared argument groups
# --------------------------------------------------------------------------- #
def _add_cluster_args(p, *, default_workers=4, serial=True):
    p.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH",
                   help=f"dask config: a bundled template ({', '.join(bundled_configs())}) "
                        f"or a path to your own YAML. Repeatable, deep-merged left to "
                        f"right, so an overlay need only carry what differs. "
                        f"Default: dask-local")
    p.add_argument("--workers", type=int, default=default_workers)
    if serial:
        p.add_argument("--serial", action="store_true",
                       help="no dask at all — run in this process (smallest smoke test)")
    p.add_argument("--store-logs", action="store_true",
                   help="keep TensorStore's benign S3 credential-chain logging "
                        "(suppressed by default; real errors are never suppressed)")


def _triple(value, name):
    """'z,y,x' (or 'zxyxx') -> a 3-tuple of ints, or None."""
    if value is None:
        return None
    parts = tuple(int(v) for v in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise SystemExit(f"--{name} needs 3 comma-separated values, got {value!r}")
    return parts


def _ftriple(value, name):
    """'z,y,x' -> a 3-tuple of floats, or None. Voxel sizes are not always integral."""
    if value is None:
        return None
    parts = tuple(float(v) for v in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise SystemExit(f"--{name} needs 3 comma-separated values, got {value!r}")
    return parts


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _factor_list(value):
    """'1,2,2;1,2,2' -> [(1,2,2), (1,2,2)], or None for auto."""
    if value is None:
        return None
    return [tuple(int(v) for v in level.split(",")) for level in value.split(";") if level]


def _configs(args):
    return args.config or ["dask-local"]


# Volume inspection lives in source_metadata (it composes that module's readers, and
# the ops need it too — `create --like` and `write` both do). Re-exported here because
# `cli.describe` is where callers and tests already look for it.
from em_volume_tools.source_metadata import (describe, existing_levels,  # noqa: E402
                                             level_spec)


def _describe(volume: str) -> dict:
    """``describe`` with a missing volume turned into a clean CLI exit."""
    try:
        return describe(volume)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    d = _describe(args.volume)
    meta = d["meta"] or {}
    print(f"{args.volume}")
    print(f"  format      {d['format']}")
    print(f"  dtype       {d['dtype']}")
    print(f"  kind        {meta.get('kind') or '(not recorded)'}")
    if meta.get("voxel_size"):
        print(f"  voxel size  {'x'.join(f'{v:g}' for v in meta['voxel_size'])} nm")
    else:
        print("  voxel size  (not recorded — --voxel-size required for convert/downsample)")
    if meta.get("offset"):
        print(f"  offset      {tuple(meta['offset'])}")
    if d["has_channels"]:
        print("  channels    yes (leading axis)")
    if d["other_markers"]:
        # Two volumes in one directory. Whichever loses the detection order is
        # unreachable through every path in this package, but still occupies the store.
        print(f"\n  WARNING: this directory also contains "
              f"{', '.join(d['other_markers'])} — a second volume of another format "
              f"is shadowed\n  here and cannot be opened while {d['format']} wins "
              f"detection. Move or delete one of them.\n")

    if not d["levels"]:
        print("  levels      none found")
        return 0
    # Each level's OWN recorded voxel size — never derived from shape ratios, which
    # ceil-division makes inexact (32 nm reads back as 31.9953), and never 2**level,
    # which is wrong on the anisotropic pyramids that are common.
    per_level = d["level_voxel_sizes"]
    sharded = any(lv["read_chunks"] and lv["chunks"] != lv["read_chunks"]
                  for lv in d["levels"].values())
    header = f"  {'level':>5}  {'shape':>24}  {'voxel nm':>20}  {'chunk':>17}"
    print(header + (f"  {'shard':>17}" if sharded else ""))
    for i, lv in sorted(d["levels"].items()):
        vox = ("x".join(f"{v:g}" for v in per_level[i])
               if per_level and i < len(per_level) else "(not recorded)")
        # With sharding the *read* chunk is the unit actually fetched, so that is
        # what belongs in the "chunk" column; the write chunk is the shard.
        chunk = lv["read_chunks"] or lv["chunks"]
        chunk_s = "x".join(str(c) for c in chunk) if chunk else "?"
        row = f"  {i:>5}  {str(lv['shape']):>24}  {vox:>20}  {chunk_s:>17}"
        if sharded:
            shard = lv["chunks"] if lv["chunks"] != lv["read_chunks"] else None
            row += f"  {('x'.join(str(c) for c in shard) if shard else '—'):>17}"
        print(row)
    return 0


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #
def cmd_convert(args) -> int:
    import contextlib
    import os

    from em_volume_tools import StorageProfile, convert

    dst = args.dst.rstrip("/")
    if dst.startswith("s3://") and not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.path.exists(os.path.expanduser("~/.aws/credentials"))):
        log.warning("no AWS credentials in the environment or ~/.aws/credentials; "
                    "S3 writes will fail with AccessDenied")

    chunk = _triple(args.chunk, "chunk")
    voxel = _triple(args.voxel_size, "voxel-size")

    # `both` writes two volumes from one read-side setup: precomputed for viewing,
    # zarr for downstream compute. Suffixes keep them distinguishable at one --dst.
    targets: list[tuple[object, str]] = []
    if args.format in ("precomputed", "both"):
        profile = args.profile or ("s3-neuroglancer" if dst.startswith("s3://")
                                   else "local-neuroglancer")
        targets.append((profile, dst + (".precomputed" if args.format == "both" else "")))
    if args.format in ("zarr", "both"):
        profile = args.profile or StorageProfile("zarr3", chunk=chunk, shard=args.shard
                                                 and _triple(args.shard, "shard"))
        targets.append((profile, dst + (".zarr" if args.format == "both" else "")))

    with _maybe_cluster(args) as client:
        for profile, target in targets:
            log.info("convert %s -> %s (%s)", args.src, target, args.kind)
            summary = convert(
                args.src, target, voxel_size=voxel, src_format=args.src_format,
                profile=profile, kind=args.kind, chunk=chunk,
                factors=_factor_list(args.factors), max_levels=args.max_levels,
                min_dim=args.min_dim, multiscale=not args.single_level,
                client=client, resume=not args.fresh, delete_existing=args.fresh,
                validate=not args.no_validate,
            )
            log.info("done %s: %d levels, status=%s", target,
                     summary["num_levels"], summary.get("status_counts"))
    return 0


# --------------------------------------------------------------------------- #
# downsample (rebuild a pyramid in place)
# --------------------------------------------------------------------------- #
def _downsample_plan(args):
    """The schedule this would produce, beside what is on disk.

    ``--min-dim`` / ``--max-levels`` / ``--factors`` decide how many levels a pyramid
    has, so values differing from the original conversion rebuild a *different*
    pyramid. Comparing against the shapes on disk is how that gets caught before
    anything is written.
    """
    from em_volume_tools.pyramid import downsample_schedule

    d = _describe(args.volume)
    meta = d["meta"] or {}
    if not meta.get("voxel_size") and args.voxel_size is None:
        raise SystemExit(f"{args.volume} has no coordinate metadata; pass --voxel-size")
    voxel = _triple(args.voxel_size, "voxel-size") or tuple(meta["voxel_size"])
    spatial = d["shape"][1:] if d["has_channels"] else d["shape"]

    schedule = downsample_schedule(spatial, voxel, factors=_factor_list(args.factors),
                                   max_levels=args.max_levels, min_dim=args.min_dim)
    shapes, voxels = [tuple(spatial)], [voxel]
    for factor in schedule:
        shapes.append(tuple(-(-s // f) for s, f in zip(shapes[-1], factor)))
        voxels.append(tuple(v * f for v, f in zip(voxels[-1], factor)))
    return d, shapes, voxels, (args.kind or meta.get("kind")), len(schedule)


def cmd_downsample(args) -> int:
    from em_volume_tools import rebuild_pyramid

    d, shapes, voxels, kind, n_sched = _downsample_plan(args)
    existing = d["levels"]

    log.info("volume  %s", args.volume)
    log.info("format  %s   kind %s%s", d["format"], kind,
             "" if args.kind else " (from the volume's metadata)")
    if kind is None:
        raise SystemExit("this volume records no image/segmentation type; pass --kind. "
                         "Guessing wrong is silent and destructive — averaging label "
                         "ids invents ids that were never in the data.")

    mismatch = False
    for i, (shape, voxel) in enumerate(zip(shapes, voxels)):
        level = existing.get(i)
        on_disk = level["shape"] if level else None
        if on_disk is not None and tuple(on_disk[-len(shape):]) != tuple(shape):
            state, mismatch = f"MISMATCH on disk {on_disk}", True
        elif i < args.start_level:
            state = "kept"
        elif i == args.start_level:
            state = "SEED (read only)" if on_disk else "MISSING — cannot seed from it"
        else:
            state = "rebuild" + ("" if on_disk else " (creates level)")
        log.info("  level %d  %-22s %-16s %s", i, str(tuple(shape)),
                 "x".join(f"{v:g}" for v in voxel) + " nm", state)

    if args.start_level > n_sched:
        raise SystemExit(f"--start-level {args.start_level} exceeds the deepest level "
                         f"({n_sched}) this schedule produces")
    if args.start_level not in existing:
        raise SystemExit(f"level {args.start_level} does not exist; nothing to seed from")
    if mismatch:
        raise SystemExit(
            "a level on disk disagrees with the computed schedule — --min-dim / "
            "--max-levels / --factors probably differ from the original conversion. "
            "Rebuilding now would leave the pyramid inconsistent.")
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    kw = dict(start_level=args.start_level, kind=kind, profile=args.profile,
              voxel_size=_triple(args.voxel_size, "voxel-size"),
              factors=_factor_list(args.factors), max_levels=args.max_levels,
              min_dim=args.min_dim, chunk=_triple(args.chunk, "chunk"),
              encoding=args.encoding, resume=args.resume,
              progress_path=args.progress_path)
    with _maybe_cluster(args) as client:
        summary = rebuild_pyramid(args.volume, client=client, **kw)

    log.info("rebuilt levels %d-%d of %s", args.start_level + 1,
             summary["num_levels"] - 1, args.volume)
    log.info("status counts: %s", summary["status_counts"])
    log.info("progress: %s", summary["progress_path"])
    return 0


# --------------------------------------------------------------------------- #
# create (an empty volume) + write (one subvolume into it)
# --------------------------------------------------------------------------- #
def cmd_create(args) -> int:
    from em_volume_tools.ops.create import create_volume, plan_volume

    dst = args.dst.rstrip("/")
    kw = dict(like=args.like, shape=_triple(args.shape, "shape"), dtype=args.dtype,
              voxel_size=_ftriple(args.voxel_size, "voxel-size"),
              offset=_ftriple(args.offset_nm, "offset-nm"), units=args.units,
              format=args.format, profile=args.profile,
              chunk=_triple(args.chunk, "chunk"),
              shard=_triple(args.shard, "shard"), levels=args.levels,
              factors=_factor_list(args.factors), max_levels=args.max_levels,
              min_dim=args.min_dim, kind=args.kind, name=args.name,
              encoding=args.encoding)
    try:
        plan = (plan_volume(dst, **kw) if args.dry_run else
                create_volume(dst, overwrite=args.overwrite,
                              validate=not args.no_validate, **kw))
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from None

    precomputed = plan["format"] == "neuroglancer_precomputed"
    print(f"{dst}")
    print(f"  format      {plan['format']}"
          f"{'' if precomputed else ' (OME-NGFF 0.5)'}")
    print(f"  dtype       {plan['dtype']}")
    print(f"  kind        {plan['kind']}")
    if precomputed:
        print(f"  encoding    {plan['encoding']}")
    print(f"  voxel size  {'x'.join(f'{v:g}' for v in plan['voxel_size'])} {plan['units']}")
    print(f"  origin      {tuple(plan['offset'])}")
    if plan["like"]:
        print(f"  geometry    {'copied from' if plan['mirrored'] else 'partly from'} "
              f"{plan['like']}")
    sharded = any(lv["shard"] for lv in plan["levels"])
    header = f"  {'level':>5}  {'shape':>24}  {'voxel nm':>20}  {'chunk':>17}"
    print(header + (f"  {'shard':>17}" if sharded else ""))
    for lv in plan["levels"]:
        chunk = "x".join(str(c) for c in lv["chunk"]) if lv["chunk"] else "(profile default)"
        row = (f"  {lv['level']:>5}  {str(tuple(lv['shape'])):>24}  "
               f"{'x'.join(f'{v:g}' for v in lv['voxel_size']):>20}  {chunk:>17}")
        if sharded:
            row += f"  {('x'.join(str(c) for c in lv['shard']) if lv['shard'] else '—'):>17}"
        print(row)
    print("--dry-run: nothing created" if args.dry_run else
          f"created {plan['num_levels']} empty level(s); fill them with `em-vol write`")
    return 0


def _source_label(spec: dict) -> str:
    """How a resolved source spec should be shown: where it is, and what was read."""
    where = spec.get("path", spec.get("source", "?"))
    return f"{where}  [{spec['backend']}" + \
           (f":{spec['dataset']}]" if spec.get("dataset") else "]")


def _print_one_write(r: dict) -> None:
    print(f"  source   {_source_label(r['src_spec'])}  "
          f"{r['src_shape']} {r['src_dtype']}")
    print(f"  offset   {r['offset']} ({r['offset_from']})")
    print("  region   " + "  ".join(
        f"{ax} {a}:{b}" for ax, a, b in zip("zyx", r["start"], r["stop"])))
    print(f"  tiles    {r['num_tiles']} of {r['task_shape']}  "
          f"({_human_bytes(r['nbytes'])}, level-{r['level']} chunk {r['dst_chunk']})")


def _print_write_table(results: list[dict]) -> None:
    """One row per source. A per-source block would be a screenful for a batch."""
    print(f"  {'source':<34} {'shape':>16} {'offset':>18}  {'tiles':>5}  {'size':>9}")
    for r in results:
        name = os.path.basename(str(r["src_spec"].get("path")
                                    or r["src_spec"].get("source", "?")).rstrip("/"))
        print(f"  {name[:34]:<34} {str(r['src_shape']):>16} "
              f"{str(tuple(r['start'])):>18}  {r['num_tiles']:>5}  "
              f"{_human_bytes(r['nbytes']):>9}")


def cmd_write(args) -> int:
    from em_volume_tools.ops.write import write_subvolumes

    offsets = ([_triple(o, "offset") for o in args.offset] if args.offset else None)
    if offsets is not None and len(offsets) != len(args.src):
        raise SystemExit(
            f"{len(args.src)} --src but {len(offsets)} --offset: give one --offset per "
            f"--src, or none and let each source supply its own (see --offset-field)")
    try:
        results = write_subvolumes(
            args.volume, args.src, offsets, level=args.level,
            offset_level=args.offset_level, offset_field=args.offset_field,
            offset_order=args.offset_order, src_format=args.src_format,
            dataset=args.dataset, cast=args.cast, dry_run=args.dry_run)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        raise SystemExit(str(e).strip("'")) from None

    first = results[0]
    print(f"{first['volume']}  level {first['level']}  "
          f"{str(first['dst_shape'])} {first['dst_dtype']}  "
          f"chunk {first['dst_chunk']}")
    if len(results) == 1:
        _print_one_write(first)
    else:
        _print_write_table(results)

    unaligned = [i for i, r in enumerate(results) if r["misaligned_axes"]]
    if unaligned:
        which = "the region" if len(results) == 1 else f"{len(unaligned)} of the regions"
        print(f"  NOTE     {which} end inside a chunk, so those chunks are "
              f"read-modify-written —\n           the data already in them is kept. "
              f"Never run two such writes at once into\n           chunks they share; "
              f"one update would be lost silently.")
    for i, j in first.get("overlaps") or []:
        print(f"  WARNING  sources {i} and {j} overlap; the later one wins where "
              f"they meet")

    tiles = sum(r["num_tiles"] for r in results)
    if args.dry_run:
        print(f"--dry-run: nothing written ({len(results)} source(s), {tiles} tile(s), "
              f"{_human_bytes(sum(r['nbytes'] for r in results))})")
    else:
        print(f"wrote {len(results)} source(s), {tiles} tile(s) in "
              f"{sum(r['seconds'] for r in results):.1f}s")
    return 0


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def _manifest_counts(path: str) -> dict[int, dict] | None:
    """Per-level progress from a run's manifest, or None if absent.

    ``{level: {"counts": {status: n}, "total": int|None, "task_shape": [...]|None,
    "grid": (max index + 1 per axis)}}``.

    **The unit is the task, which is not always the chunk.** ``plan_task_shape``
    sizes a task to cover whole source *and* destination chunks, so one manifest
    record can stand for 256 destination chunks; the driver records the level's task
    ``total`` for exactly this reason. ``grid`` is the fallback for manifests written
    before it did — the largest block index seen per axis, which pins the task grid
    once a level is complete and lower-bounds it before that.

    **Statuses are kept apart, not summed into a "done" number.** ``empty`` means the
    block was read and found to be entirely the fill value, so tensorstore wrote no
    object — a legitimate outcome for sparse data, and an alarm when it is *every*
    block: that is what a source the reader cannot actually read looks like. Reporting
    only a total let a conversion of 1.9 M blocks read as all-zeros and still print
    100%, which is how a whole run was lost.

    **Much cheaper than listing the store.** A LIST over an object store enumerates
    every chunk — 674k keys for one level of a full-resolution volume, paginated a
    thousand at a time — while the manifest is a local append-only JSONL the run
    already wrote. Measured on that volume: 1.5 s for 772k records, against tens of
    seconds of S3 listing.

    It answers a subtly different question, though, and that is why ``--storage``
    exists: the manifest is what the *run recorded*, storage is what is *there*. They
    diverge if objects were deleted underneath, or if a different run wrote the
    volume. For "how far along is my conversion" the manifest is both faster and more
    directly the thing you asked.

    A torn final line is normal — the writer may be appending as this reads — so a
    line that does not parse is skipped rather than fatal.
    """
    if not path or not os.path.exists(path):
        return None
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            group = rec.get("group", rec.get("level"))
            if not isinstance(group, int):
                continue
            lv = out.setdefault(group, {"counts": {}, "total": None,
                                        "task_shape": None, "grid": None})
            if "status" not in rec:                      # a meta line: the denominator
                meta = rec.get("meta") or {}
                if isinstance(meta.get("total"), int):
                    lv["total"] = meta["total"]
                if meta.get("task_shape"):
                    lv["task_shape"] = tuple(meta["task_shape"])
                continue
            status = str(rec["status"])
            lv["counts"][status] = lv["counts"].get(status, 0) + 1
            key = rec.get("key", rec.get("index"))
            if isinstance(key, list) and all(isinstance(i, int) for i in key):
                lv["grid"] = (tuple(key) if lv["grid"] is None else
                              tuple(max(a, b) for a, b in zip(lv["grid"], key)))
    for lv in out.values():
        if lv["grid"] is not None:
            lv["grid"] = tuple(i + 1 for i in lv["grid"])
    return out


def _task_total(lv: dict, chunk_total: int, chunk: tuple, level: int) -> tuple[int, str]:
    """The denominator for a level's manifest count, and a note if it isn't the chunk grid.

    The manifest counts **tasks**; ``chunk_total`` is the destination *chunk* grid.
    They are equal for every level a pyramid derives from the level below, and
    differ at level 0 whenever the source's natural read unit is coarser than the
    destination chunk — specimen5's is 128x2048x2048 against 128^3, so 7,623 tasks
    against 1,680,206 chunks, and dividing one by the other reported 0.17% for a
    run that was 36% done.

    Order of preference:
      1. the ``total`` the driver recorded — exact, and known before dispatch;
      2. the grid implied by the largest recorded block index, for manifests from
         before that was recorded. It is exact for a level that finished and a
         lower bound for one still running, so the percentage is an upper bound;
      3. the chunk grid, which is right whenever a task *is* a chunk.
    """
    if lv.get("total"):
        total = lv["total"]
        task = lv.get("task_shape")
        if total != chunk_total and task:
            per = math.prod(max(1, t // c) for t, c in zip(task, chunk))
            return total, (f"level {level} is counted in tasks of "
                           f"{tuple(task)}, {per} destination chunks each "
                           f"({chunk_total:,} chunks in {total:,} tasks)")
        return total, ""
    grid = lv.get("grid")
    if grid and math.prod(grid) < chunk_total:
        implied = math.prod(grid)
        return implied, (
            f"level {level}: this manifest predates the recorded task total, so "
            f"{implied:,} is inferred from the recorded block indices (grid {grid}, "
            f"against {chunk_total:,} chunks). Exact if the level finished, a lower "
            f"bound if it is still running — so the % is an upper bound")
    return chunk_total, ""


# Statuses meaning the block will not be retried. "failed" is excluded on purpose:
# em-blockrun's is_done tests key presence, so a resumed run retries failures, and
# counting them as done would overstate progress.
_DONE_STATUSES = ("written", "empty", "skipped")


def _other_manifests(manifest_path: str) -> list[str]:
    """Manifests sitting beside the one we looked for, for the message when it is absent.

    A rebuild names its manifest ``<name>.regen-from-<n>.progress.jsonl`` — deliberately,
    so it cannot be mistaken for a conversion's and mark the levels being rebuilt as
    already done. The cost is that the default `progress` looks for is then the wrong
    file, and "no run manifest" alone gives no hint that the right one is right there.
    """
    import glob as _glob

    suffix = ".progress.jsonl"
    stem = manifest_path[: -len(suffix)] if manifest_path.endswith(suffix) else manifest_path
    return sorted(p for p in _glob.glob(f"{stem}*{suffix}") if p != manifest_path)


def _stored_chunks(volume: str, fmt: str, level: int, scale_key: str | None) -> int:
    """Count the chunk objects actually stored for one level.

    The two formats put them in different places, which is the whole reason this used
    to report "no levels found" for precomputed: zarr v3 gives each level its own
    subdirectory with data under ``c/`` (one object per chunk, or per *shard* when
    sharded), while precomputed keys every scale's chunks under its own scale key
    (``8_8_8/…``) beside one shared ``info``.
    """
    from em_volume_tools.location import list_keys

    if fmt == "zarr3":
        keys = list_keys(volume, str(level))
        return sum(1 for k in keys if k.startswith("c/"))
    keys = list_keys(volume, scale_key or "")
    # A precomputed chunk key looks like `0-128_0-128_0-128`; anything else under the
    # prefix (a stray file) is not a chunk.
    return sum(1 for k in keys if "_" in k.rsplit("/", 1)[-1] and "-" in k.rsplit("/", 1)[-1])


def _precomputed_scale_keys(volume: str) -> dict[int, str]:
    """``{level: scale key}`` finest-first, matching how levels are numbered elsewhere."""
    from em_volume_tools.location import read_json

    info = read_json(volume.rstrip("/") + "/info") or {}
    ordered = sorted(info.get("scales", []), key=lambda s: tuple(s["resolution"]))
    return {i: s["key"] for i, s in enumerate(ordered)}


def cmd_progress(args) -> int:
    """Blocks done per level, against the number the level has.

    The unit differs by source: from the manifest it is the **task** the run
    dispatched (see ``_task_total``), from ``--storage`` it is the stored chunk
    object. They coincide except at level 0 of a conversion whose source reads in
    units coarser than the destination chunk.

    Counted through the kvstore rather than a filesystem walk, so it is one listing
    per level and works against an object store — the previous version hardcoded the
    ``file`` driver and could not inspect the s3 volumes ``convert`` writes.

    **Levels come from the shared inspection, so every format `info` understands works
    here too.** They used to come from opening ``<volume>/<level>`` with the zarr v3
    driver in a loop, which meant precomputed — whose scales live under one ``info`` —
    broke out at level 0 and reported "no levels found yet", and said so for the
    manifest path as well, since the loop was shared.

    **Every store opened here still goes through ``ensure_credentials``**, now by
    construction rather than by remembering: reads go via ``location`` and
    ``open_backend``, which both bootstrap. tensorstore's S3 *profile* provider cannot
    read ``~/.aws/credentials``, so that call copies the profile into the ``AWS_*`` env
    vars its *environment* provider does read. Skipping it does not fail outright — the
    credential chain simply falls through to the EC2 instance-metadata service, which
    does not exist off EC2, and each probe waits out a socket timeout while logging
    ``AWS_AUTH_CREDENTIALS_PROVIDER_IMDS_SOURCE_FAILURE``. That is slow and noisy
    rather than broken, which is exactly why it survives review; see the S3 bootstrap
    note in CLAUDE.md.
    """
    from em_volume_tools.location import default_progress_path
    from em_volume_tools.source_metadata import PRECOMPUTED_GZ, detect_backend

    volume = args.volume.rstrip("/")
    manifest_path = args.progress_path or default_progress_path(volume)
    manifest = None if args.storage else _manifest_counts(manifest_path)

    fmt = detect_backend(volume)
    if fmt == PRECOMPUTED_GZ:
        # Only the chunk KEYS differ, and nothing here reads a chunk — shapes and
        # scale keys come from `info`, which is the same document either way.
        fmt = "neuroglancer_precomputed"
    if fmt is None:
        print(f"no volume found at {volume}")
        return 1
    levels = existing_levels(volume, fmt)
    scale_keys = (_precomputed_scale_keys(volume)
                  if fmt == "neuroglancer_precomputed" else {})

    # Timestamped because this is a point-in-time count, and the way you get a rate
    # out of it is to diff two runs — which needs to know how far apart they were.
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    source = "run manifest" if manifest is not None else "storage listing"
    header = (f"{'level':>5} {'shape':>24} {'chunk':>18} {'done/total':>20} {'%':>7}")
    if manifest is not None:
        header += f" {'written':>10} {'empty':>10}"
    print(f"{volume}\n{stamp}  (counted from the {source})")
    # Falling back to storage because no manifest was found used to be indistinguishable
    # from --storage: same header, no other tell. Say it, and say where it looked.
    if manifest is None and not args.storage:
        print(f"  no run manifest at {manifest_path}")
        nearby = _other_manifests(manifest_path)
        if nearby:
            print(f"  but these are beside it — a rebuild names its own:\n"
                  + "".join(f"      --progress-path {p}\n" for p in nearby), end="")
        else:
            print(f"  (pass --progress-path if the run wrote it elsewhere; a remote "
                  f"--dst defaults it to the launching directory)")
    print(header)

    done_total = expected_total = written_total = empty_total = 0
    notes: list[str] = []
    for level, lv_info in sorted(levels.items()):
        shape = lv_info["shape"]
        chunk = lv_info["chunks"]
        if not chunk:
            print(f"{level:>5} {str(shape):>24} {'(chunking unknown)':>18}")
            continue
        expected = math.prod(math.ceil(s / c) for s, c in zip(shape, chunk))
        extra = ""
        if manifest is not None:
            lv = manifest.get(level, {})
            by_status = lv.get("counts", {})
            done = sum(n for s, n in by_status.items() if s in _DONE_STATUSES)
            written, empty = by_status.get("written", 0), by_status.get("empty", 0)
            written_total += written
            empty_total += empty
            extra = f" {written:>10,} {empty:>10,}"
            expected, note = _task_total(lv, expected, chunk, level)
            if note:
                notes.append(note)
        else:
            done = _stored_chunks(volume, fmt, level, scale_keys.get(level))
        done_total += done
        expected_total += expected
        pct = 100.0 * done / expected if expected else 100.0
        print(f"{level:>5} {str(shape):>24} {str(chunk):>18} "
              f"{f'{done}/{expected}':>20} {pct:>6.1f}%{extra}")

    if not levels:
        print("no levels found yet — level 0 has not been created")
        return 1
    pct = 100.0 * done_total / expected_total if expected_total else 100.0
    total_extra = (f" {written_total:>10,} {empty_total:>10,}"
                   if manifest is not None else "")
    print(f"{'TOTAL':>5} {'':>24} {'':>18} "
          f"{f'{done_total}/{expected_total}':>20} {pct:>6.1f}%{total_extra}")
    for note in notes:
        print(f"\n  {note}")

    # An all-empty run is the signature of a source the reader cannot actually read:
    # every block came back as fill value, so tensorstore wrote no object and the
    # totals still say 100%. Loud, because it costs a whole conversion to discover.
    if manifest is not None and done_total and not written_total:
        print(f"\nWARNING: {empty_total:,} blocks processed, NONE written.\n"
              f"  Every block read as entirely the fill value, so no chunk objects\n"
              f"  exist — the destination holds only metadata. Usually this means the\n"
              f"  SOURCE could not be read: a precomputed volume written by\n"
              f"  CloudVolume stores '.gz'-suffixed chunks that tensorstore requests\n"
              f"  without the suffix and reads as zeros. Check with:\n"
              f"      em-vol info <src>   and   ls <src>/<scale-key> | head")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def cmd_annotations(args) -> int:
    """A neuroglancer annotation layer marking where a sparse volume's data is.

    The JSON goes to **stdout** and the human summary to **stderr**, so
    ``em-vol annotations vol > layer.json`` works and so does reading the table while
    it runs. ``--out`` writes the JSON to a file instead.

    Reads only, and barely: the boxes come from listing which chunk objects exist,
    which on a sparse volume is the occupancy question exactly, plus one coarse read
    per region to tighten it. See :mod:`em_volume_tools.ops.annotate` for why the
    annotations are local rather than a precomputed annotation layer.
    """
    from em_volume_tools.ops.annotate import (NoOccupancy, annotation_layer,
                                              labeled_regions, output_dimensions,
                                              render, viewer_state)

    volume = args.volume.rstrip("/")
    tighten = None if args.no_tighten else args.tighten_level
    try:
        regions, ctx = labeled_regions(volume, level=args.level, tighten_level=tighten)
    except (FileNotFoundError, ValueError, NoOccupancy) as e:
        raise SystemExit(str(e)) from None

    d = _describe(volume)
    meta = d["meta"] or {}
    voxel = _ftriple(args.voxel_size, "voxel-size") or meta.get("voxel_size")
    units = "nm" if args.voxel_size else meta.get("units")
    dims, warning = output_dimensions(voxel, units)

    err = sys.stderr
    print(f"{volume}", file=err)
    print(f"  {ctx['n_chunks']} chunk objects at level {ctx['level']} "
          f"(cell {ctx['cell']}) -> {len(regions)} region(s)", file=err)
    tighten = ctx["tighten_level"]
    if tighten is not None:
        nm = (f" ({ctx['voxel_sizes'][tighten][0]:g} nm)"
              if ctx["voxel_sizes"] else "")
        clamped = ("" if ctx["tighten_clamped_from"] is None else
                   f", the deepest there is — level {ctx['tighten_clamped_from']} was "
                   f"asked for but does not exist, so this is exact and slower")
        print(f"  tightened at level {tighten}{nm}{clamped}", file=err)
    if not regions:
        print("  nothing is stored — no annotations to make", file=err)
        return 1
    print(f"\n{'#':>3}  {'z':>15} {'y':>15} {'x':>15}  {'extent zyx':>18} "
          f"{'labels':>7}", file=err)
    for i, r in enumerate(regions):
        span = " ".join(f"{r['lo'][a]:6d}-{r['hi'][a]:6d}" for a in range(3))
        ext = "x".join(str(r["hi"][a] - r["lo"][a]) for a in range(3))
        n = "-" if r["n_labels"] is None else f"{r['n_labels']:,}"
        print(f"{i:>3}  {span}  {ext:>18} {n:>7}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = annotation_layer(regions, dims, name=args.name or f"{volume.rsplit('/', 1)[-1]}-regions",
                             color=args.color, kind=args.kind,
                             label=args.label or "r")
    obj = (viewer_state(volume, ctx["format"], layer, regions, dims, meta.get("kind"))
           if args.state else layer)
    text = render(obj)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.out} — paste it into the `layers` array of a "
              f"neuroglancer state" if not args.state else
              f"\nwrote {args.out} — load it with neuroglancer's {{}} JSON editor",
              file=err)
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def _maybe_cluster(args):
    """A dask client, or None for --serial / --workers 0."""
    import contextlib

    if getattr(args, "serial", False) or not args.workers:
        log.info("serial mode: no dask client")
        return contextlib.nullcontext(None)
    return start_dask(args.workers, _configs(args), label="em-vol")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="em-vol", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # --- info ---------------------------------------------------------------
    q = sub.add_parser("info", help="what a volume is, and which levels exist",
                       description="Report a volume's format, coordinate metadata "
                                   "and the levels present. Reads only.")
    q.add_argument("volume")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_info)

    # --- convert ------------------------------------------------------------
    q = sub.add_parser("convert", help="build a multiscale volume from a source",
                       description=cmd_convert.__doc__ or
                       "Convert a source volume into a multiscale zarr and/or "
                       "neuroglancer-precomputed volume. Resumable.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, help="path, s3://..., precomputed or .zarr")
    q.add_argument("--dst", required=True,
                   help="destination base (path or s3://bucket/prefix). With "
                        "--format both, '.precomputed' and '.zarr' are appended")
    q.add_argument("--src-format", default=None,
                   help="backend for --src (default: auto-detect). Use 'image_stack' "
                        "for a directory or glob of ordered 2D slices (PNG/TIFF) — "
                        "that one is never auto-detected, and needs --voxel-size "
                        "since image files carry no physical scale")
    q.add_argument("--format", choices=("precomputed", "zarr", "both"),
                   default="precomputed")
    q.add_argument("--kind", choices=("image", "probability", "segmentation"),
                   default="image",
                   help="reducer for the pyramid: mean for image/probability, "
                        "label-preserving mode for segmentation. Getting this wrong "
                        "is silent — averaging label ids invents ids")
    q.add_argument("--voxel-size", default=None, help="z,y,x nm (default: from source)")
    q.add_argument("--chunk", default="128,128,128", help="z,y,x chunk")
    q.add_argument("--shard", default=None, help="z,y,x shard (zarr only)")
    q.add_argument("--profile", default=None,
                   help="storage profile name (default: chosen from --format and "
                        "whether --dst is remote)")
    q.add_argument("--factors", default=None,
                   help="explicit per-level factors, e.g. '1,2,2;1,2,2' (default auto)")
    q.add_argument("--max-levels", type=int, default=8)
    q.add_argument("--min-dim", type=int, default=128,
                   help="stop when the largest spatial dim is <= this")
    q.add_argument("--single-level", action="store_true",
                   help="write level 0 only, no pyramid")
    q.add_argument("--fresh", action="store_true",
                   help="delete and restart instead of resuming")
    q.add_argument("--no-validate", action="store_true",
                   help="skip OME metadata validation")
    _add_cluster_args(q)
    q.set_defaults(func=cmd_convert)

    # --- downsample ---------------------------------------------------------
    q = sub.add_parser(
        "downsample", help="rebuild pyramid levels above a trusted one, in place",
        description="Regenerate a multiscale pyramid IN PLACE, from a level you "
                    "trust.\n\nDownsampling is cascaded (level N derives from N-1), "
                    "so a bad level poisons every level above it. --start-level is "
                    "read, never written; everything above it is rebuilt. This does "
                    "not create a new volume — use `convert` for that.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the multiscale volume to repair (path or s3://...)")
    q.add_argument("--start-level", type=int, default=0,
                   help="level to derive from; it is read, never written (default: 0)")
    q.add_argument("--kind", choices=("image", "segmentation"), default=None,
                   help="reducer (default: read from the volume)")
    q.add_argument("--min-dim", type=int, default=128)
    q.add_argument("--max-levels", type=int, default=8)
    q.add_argument("--factors", default=None,
                   help="explicit per-level factors, e.g. '1,2,2;1,2,2'. MUST match "
                        "the original conversion, as must --min-dim/--max-levels")
    q.add_argument("--chunk", default=None, help="z,y,x (default: level 0's chunking)")
    q.add_argument("--voxel-size", default=None,
                   help="z,y,x nm; overrides the volume's own metadata")
    q.add_argument("--encoding", default=None, help="precomputed encoding (e.g. raw)")
    q.add_argument("--profile", default=None, help="storage profile")
    q.add_argument("--resume", action="store_true",
                   help="continue an interrupted rebuild")
    q.add_argument("--progress-path", default=None)
    q.add_argument("--dry-run", action="store_true",
                   help="report the plan and exit, touching nothing")
    _add_cluster_args(q)
    q.set_defaults(func=cmd_downsample)

    # --- create -------------------------------------------------------------
    q = sub.add_parser(
        "create", help="lay out an EMPTY multiscale volume",
        description="Create an empty zarr v3 or neuroglancer-precomputed volume — "
                    "every level exists, no chunk data.\n\nUse it when several small "
                    "pieces (image stacks, HDF5 files) belong at known positions "
                    "inside one frame: create the frame, then `em-vol write` each "
                    "piece into it.\n\n--like copies "
                    "a reference volume's geometry (level shapes, per-level voxel "
                    "sizes and chunking, dtype, origin), so a voxel index means the "
                    "same thing in both. Its level shapes are copied verbatim rather "
                    "than recomputed, unless --shape or --factors overrides them. "
                    "Anything you pass explicitly wins over the reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("dst", help="where to create it (path or s3://bucket/prefix)")
    q.add_argument("--like", default=None, metavar="VOLUME",
                   help="reference volume to copy the frame from (zarr or "
                        "precomputed, local or remote)")
    q.add_argument("--shape", default=None,
                   help="z,y,x level-0 shape (default: --like's). Overriding it means "
                        "the pyramid is recomputed rather than copied")
    q.add_argument("--dtype", default=None,
                   help="e.g. uint8, uint64 (default: --like's). Worth setting when "
                        "the frame comes from an image volume but the pieces are labels")
    q.add_argument("--voxel-size", default=None, help="z,y,x nm (default: --like's)")
    q.add_argument("--offset-nm", default=None,
                   help="z,y,x physical origin of the volume (default: --like's, else 0)")
    q.add_argument("--units", default=None, help="default: --like's, else nm")
    q.add_argument("--kind", choices=("image", "segmentation"), default=None,
                   help="recorded in the metadata and read back by `downsample` to "
                        "pick its reducer (default: --like's, else image). For "
                        "precomputed it also picks the default encoding")
    q.add_argument("--format", choices=("zarr", "precomputed"), default=None,
                   help="target format (default: --like's format, else zarr). "
                        "precomputed keeps its scales in one intrinsic `info`; zarr "
                        "gets OME-NGFF 0.5 group metadata")
    q.add_argument("--encoding", default=None,
                   help="precomputed chunk encoding: raw, jpeg, "
                        "compressed_segmentation (default: compressed_segmentation "
                        "for --kind segmentation, raw otherwise)")
    q.add_argument("--chunk", default=None,
                   help="z,y,x, all levels (default: --like's per-level chunking)")
    q.add_argument("--shard", default=None, help="z,y,x shard")
    q.add_argument("--levels", type=int, default=None,
                   help="how many levels to create, counting level 0 "
                        "(default: as many as --like has, else the auto schedule). "
                        "--levels 1 for a single-scale volume")
    q.add_argument("--factors", default=None,
                   help="explicit per-level factors, e.g. '1,2,2;1,2,2'; forces the "
                        "pyramid to be computed rather than copied")
    q.add_argument("--max-levels", type=int, default=8)
    q.add_argument("--min-dim", type=int, default=128)
    q.add_argument("--profile", default=None,
                   help="storage profile name, overriding --format and its chunk/"
                        "compressor defaults (local, ceph, local-neuroglancer, "
                        "s3-neuroglancer)")
    q.add_argument("--name", default="image", help="OME multiscales name (zarr only)")
    q.add_argument("--overwrite", action="store_true",
                   help="replace an existing volume at --dst (DESTROYS its data)")
    q.add_argument("--no-validate", action="store_true",
                   help="skip OME metadata validation")
    q.add_argument("--dry-run", action="store_true",
                   help="print the layout and exit, creating nothing")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_create)

    # --- write --------------------------------------------------------------
    q = sub.add_parser(
        "write", help="write one subvolume into an existing volume",
        description="Place a subvolume — an image stack, an HDF5 dataset, a region "
                    "of another volume — into an existing volume at a voxel offset.\n\n"
                    "SINGLE-SCALE on purpose: it writes --level and touches no other, "
                    "because how a patch should look when coarsened is a separate "
                    "decision (averaging label ids invents ids). Run `em-vol "
                    "downsample` afterwards if the result needs a pyramid.\n\n"
                    "Runs in this process — no dask. For anything big enough to need "
                    "a cluster, use `em-vol convert`.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the destination volume (path or s3://...)")
    q.add_argument("--src", required=True, action="append",
                   help="an .h5/.hdf5 file, a directory or glob of ordered 2D slices, "
                        "or another volume. REPEATABLE: pass several to write a batch "
                        "in one go. All of them are checked — offsets, bounds, dtype "
                        "— before any is written, so a mistake in the last one does "
                        "not leave the earlier ones half-applied")
    q.add_argument("--offset", default=None, action="append",
                   help="z,y,x voxel offset of the subvolume's origin, in --level's "
                        "voxels unless --offset-level says otherwise. OPTIONAL: with "
                        "no --offset the source is asked for one, which an HDF5 file "
                        "often records (see --offset-field). With several --src, give "
                        "either no --offset at all or one per --src, in order")
    q.add_argument("--offset-field", default="voxel_offset", metavar="NAME",
                   help="what the source calls its stored offset, used when --offset "
                        "is omitted. In an HDF5 file this is looked for in the "
                        "dataset's attributes, the root group's attributes, and a "
                        "top-level dataset of that name (default: voxel_offset)")
    q.add_argument("--offset-order", choices=("zyx", "xyz"), default="zyx",
                   help="axis order of the offset, whether typed or read from the "
                        "source. Worth checking on a stored one: 'voxel_offset' is "
                        "precomputed's field name and precomputed means XYZ, while "
                        "everything in this package is zyx — reversed, the piece "
                        "lands mirrored through the z=x diagonal (default: zyx)")
    q.add_argument("--level", type=int, default=0, help="which level to write into")
    q.add_argument("--offset-level", type=int, default=None,
                   help="the level whose voxels --offset is expressed in, when that "
                        "is not --level (e.g. coordinates read off level 0 while "
                        "writing level 2). Converted with the recorded per-level "
                        "voxel sizes; a non-integral result is an error")
    q.add_argument("--src-format", default=None,
                   help="backend for --src (default: guessed from the path — .h5 is "
                        "hdf5, a directory/glob/image file is image_stack, anything "
                        "with an info or zarr.json is detected properly)")
    q.add_argument("--dataset", default=None,
                   help="HDF5 dataset path (default: the file's only 3D+ dataset)")
    q.add_argument("--cast", action="store_true",
                   help="allow a lossy dtype conversion into the destination")
    q.add_argument("--dry-run", action="store_true",
                   help="report the region, tiles and alignment; write nothing")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_write)

    # --- progress -----------------------------------------------------------
    q = sub.add_parser("progress", help="blocks done per level, for a live run",
                       description="Point-in-time counts for a multiscale volume being "
                                   "written, zarr v3 or precomputed. Reads only; "
                                   "re-run to refresh.\n\nOn sparse data the manifest "
                                   "runs ahead of --storage and that gap is real: an "
                                   "all-fill block is elided, so it writes no object. "
                                   "Use the manifest to answer 'how far along is my "
                                   "run'.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume (path or s3://...)")
    q.add_argument("--progress-path", default=None,
                   help="the run's progress manifest (default: the same path the "
                        "conversion would have used — note a REBUILD names its own "
                        "'<name>.regen-from-<n>.progress.jsonl'). Counting from it is "
                        "far cheaper than listing an object store")
    q.add_argument("--storage", action="store_true",
                   help="count what is actually stored instead of what the run "
                        "recorded — authoritative, but lists every chunk")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_progress)

    # --- annotations --------------------------------------------------------
    q = sub.add_parser(
        "annotations", help="a neuroglancer layer marking where the data is",
        description="Emit a neuroglancer annotation layer with one bounding box per "
                    "written region of a SPARSE volume — a ground-truth volume, an "
                    "ROI export — so the viewer gets a clickable list that jumps "
                    "between them.\n\n"
                    "The annotations are LOCAL (inline in the state), not a "
                    "precomputed annotation layer, because neuroglancer does not list "
                    "precomputed annotations: it builds the list by iterating the "
                    "source, and the class behind every precomputed annotation source "
                    "defines that iterator as empty. Those render but cannot be "
                    "clicked through.\n\n"
                    "JSON to stdout, summary to stderr, so `> layer.json` works. "
                    "Reads only; writes nothing to the volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume (path or s3://...)")
    q.add_argument("--out", default=None, metavar="PATH",
                   help="write the JSON here instead of to stdout")
    q.add_argument("--state", action="store_true",
                   help="emit a complete loadable viewer state (volume layer + "
                        "annotations) rather than just the layer to paste")
    q.add_argument("--level", type=int, default=0,
                   help="the level whose chunk objects define the footprint. Coarser "
                        "is a cheaper listing and a blockier box; coordinates are "
                        "reported in level-0 voxels either way (default: 0)")
    q.add_argument("--tighten-level", type=int, default=2, metavar="N",
                   help="shrink each box to its nonzero voxels by reading it at this "
                        "level. Cheap — a 384-voxel box is 96 voxels at 32 nm — and "
                        "exact to one voxel there, so a coarser level gives a looser "
                        "box and 0 gives the exact one (default: 2)")
    q.add_argument("--no-tighten", action="store_true",
                   help="skip the reads entirely: boxes stay chunk-aligned and no "
                        "label counts are reported")
    q.add_argument("--kind", choices=("box", "point"), default="box",
                   help="bounding boxes, or one point at each region's centre — "
                        "points stay clickable when zoomed out (default: box)")
    q.add_argument("--name", default=None,
                   help="layer name (default: '<volume>-regions')")
    q.add_argument("--label", default=None, metavar="PREFIX",
                   help="prefix for annotation ids and descriptions, numbered from 00 "
                        "(default: r)")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, when the volume records none or "
                        "records it in units this does not recognise. Without a "
                        "usable one the layer is unitless and will not line up")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_annotations)

    args = p.parse_args(argv)

    # Image files carry no physical scale, and the op would otherwise fail deep inside
    # the conversion rather than here. Everything else can read it from the source.
    if (getattr(args, "src_format", None) == "image_stack"
            and not getattr(args, "voxel_size", None)):
        p.error("--src-format image_stack requires --voxel-size: image files record "
                "no physical scale, so it cannot be read from the source "
                "(e.g. --voxel-size 8,8,8 for 8 nm isotropic)")
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from em_volume_tools.logs import quiet_store_logs

    with quiet_store_logs(not getattr(args, "store_logs", False)):
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
