"""em-vol: convert, downsample and inspect large 3D volumes, locally or on SLURM.

The command-line entry point for this package. Every write is block-mapped over a dask
cluster (local or SLURM, chosen by ``--config``) and resumable, so an interrupted run
continues where it stopped rather than starting over.

    em-vol info    <volume>                      # what is it, what levels exist
    em-vol convert --src ... --dst ...           # build a multiscale volume
    em-vol downsample <volume> --start-level 2   # rebuild levels above a trusted one
    em-vol progress <volume>                     # chunks written, per level

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


def _factor_list(value):
    """'1,2,2;1,2,2' -> [(1,2,2), (1,2,2)], or None for auto."""
    if value is None:
        return None
    return [tuple(int(v) for v in level.split(",")) for level in value.split(";") if level]


def _configs(args):
    return args.config or ["dask-local"]


# --------------------------------------------------------------------------- #
# shared inspection: what is this volume, and which levels exist
# --------------------------------------------------------------------------- #
def describe(volume: str) -> dict:
    """Detected format, coordinate metadata and the levels actually present.

    Shared by ``info`` and ``downsample``'s plan, so the two cannot disagree about
    what is on disk.
    """
    from em_volume_tools.backends.base import open_backend
    from em_volume_tools.source_metadata import (detect_backend,
                                                 read_level_voxel_sizes,
                                                 read_source_metadata)

    fmt = detect_backend(volume)
    if fmt is None:
        raise SystemExit(f"no volume found at {volume}")
    spec = {"backend": fmt, "path": volume}
    meta = read_source_metadata(spec)
    level0 = open_backend(meta["data_spec"] if meta else spec)
    shape = tuple(int(s) for s in level0.shape)
    return {"format": fmt, "meta": meta, "shape": shape,
            "dtype": str(getattr(level0, "dtype", "?")),
            "has_channels": bool(meta["has_channels"]) if meta else False,
            "level_voxel_sizes": read_level_voxel_sizes(spec),
            "levels": existing_levels(volume, fmt)}


def existing_levels(volume: str, fmt: str, probe: int = 12) -> dict[int, dict]:
    """``{level: {"shape", "chunks", "read_chunks"}}`` for the levels that open.

    Probes upward until one misses. The multiscale group metadata is written at the
    very end of a conversion, so an in-flight volume has levels but no group metadata
    — probing is what makes this work on a run that is still going.

    Chunking is **per level**, not a property of the volume: it lives in each level's
    own array metadata (zarr's ``zarr.json``, precomputed's per-scale ``chunk_sizes``),
    and a conversion is free to chunk levels differently. So it is read here, where
    each level is opened anyway, rather than assumed from level 0.

    ``chunks`` is the *write* chunk and ``read_chunks`` the *read* chunk. They differ
    only when the level is sharded, where the write chunk is the shard and the read
    chunk is the unit actually fetched — which is the number that governs read
    amplification, so both are worth having.
    """
    from em_volume_tools.backends.base import open_backend

    out: dict[int, dict] = {}
    for i in range(probe):
        spec = ({"backend": fmt, "path": f"{volume.rstrip('/')}/{i}"} if fmt == "zarr3"
                else {"backend": fmt, "path": volume, "scale_index": i})
        try:
            be = open_backend(spec)
            shape = tuple(int(s) for s in be.shape)
        except Exception:
            break
        # chunks come from the same open, but a backend need not expose them
        def _maybe(attr):
            try:
                return tuple(int(c) for c in getattr(be, attr))
            except Exception:
                return None
        out[i] = {"shape": shape, "chunks": _maybe("chunks"),
                  "read_chunks": _maybe("read_chunks")}
    return out


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    d = describe(args.volume)
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

    d = describe(args.volume)
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


def cmd_progress(args) -> int:
    """Blocks done per level, against the number the level has.

    The unit differs by source: from the manifest it is the **task** the run
    dispatched (see ``_task_total``), from ``--storage`` it is the stored chunk
    object. They coincide except at level 0 of a conversion whose source reads in
    units coarser than the destination chunk.

    Counted through the kvstore rather than a filesystem walk, so it is one listing
    per level and works against an object store — the previous version hardcoded the
    ``file`` driver and could not inspect the s3 volumes ``convert`` writes.

    **Every store opened here goes through ``ensure_credentials``.** tensorstore's S3
    *profile* provider cannot read ``~/.aws/credentials``, so that call copies the
    profile into the ``AWS_*`` env vars its *environment* provider does read. Skipping
    it does not fail outright — the credential chain simply falls through to the EC2
    instance-metadata service, which does not exist off EC2, and each probe waits out
    a socket timeout while logging
    ``AWS_AUTH_CREDENTIALS_PROVIDER_IMDS_SOURCE_FAILURE``. That is slow and noisy
    rather than broken, which is exactly why it survives review; see the S3 bootstrap
    note in CLAUDE.md.
    """
    import tensorstore as ts

    from em_volume_tools.location import (default_progress_path, ensure_credentials,
                                          to_kvstore)

    volume = args.volume.rstrip("/")
    manifest_path = args.progress_path or default_progress_path(volume)
    manifest = None if args.storage else _manifest_counts(manifest_path)

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
        print(f"  no run manifest at {manifest_path}\n"
              f"  (pass --progress-path if the run wrote it elsewhere; a remote --dst "
              f"defaults it to the launching directory)")
    print(header)

    done_total = expected_total = written_total = empty_total = 0
    notes: list[str] = []
    level = 0
    while True:
        kv = ensure_credentials(to_kvstore(f"{volume}/{level}"))
        try:
            store = ts.open({"driver": "zarr3", "kvstore": kv}).result()
        except Exception:
            break
        shape = tuple(int(s) for s in store.shape)
        chunk = tuple(int(c) for c in store.chunk_layout.write_chunk.shape)
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
            # Unsharded zarr v3 puts chunk objects under "c/"; the metadata key is
            # zarr.json, which must not be counted as data.
            listing = ts.KvStore.open(ensure_credentials(
                {**kv, "path": kv.get("path", "").rstrip("/") + "/"})
            ).result().list().result()
            done = sum(1 for k in listing
                       if (k.decode() if isinstance(k, bytes) else str(k)).startswith("c/"))
        done_total += done
        expected_total += expected
        pct = 100.0 * done / expected if expected else 100.0
        print(f"{level:>5} {str(shape):>24} {str(chunk):>18} "
              f"{f'{done}/{expected}':>20} {pct:>6.1f}%{extra}")
        level += 1

    if level == 0:
        print("no levels found yet (level 0 not created), or this is not "
              "unsharded zarr v3 — precomputed and sharded volumes are not counted")
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
    q.add_argument("--start-level", type=int, required=True,
                   help="level to derive from; it is read, never written")
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

    # --- progress -----------------------------------------------------------
    q = sub.add_parser("progress", help="chunks written per level, for a live run",
                       description="Point-in-time chunk counts for a multiscale zarr "
                                   "v3 volume being written. Reads only; re-run to "
                                   "refresh.")
    q.add_argument("volume", help="the .zarr group (path or s3://...)")
    q.add_argument("--progress-path", default=None,
                   help="the run's progress manifest (default: the same path the "
                        "conversion would have used). Counting from it is far "
                        "cheaper than listing an object store")
    q.add_argument("--storage", action="store_true",
                   help="count what is actually stored instead of what the run "
                        "recorded — authoritative, but lists every chunk")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_progress)

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
