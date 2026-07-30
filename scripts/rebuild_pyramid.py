"""Regenerate a multiscale pyramid in place, from a level you trust.

Downsampling is cascaded (level N is derived from level N-1), so a bad level
poisons every level above it. Point --start-level at the deepest level you still
trust; it is read, never written, and everything above it is rebuilt.

    # see what would be rebuilt, and check the schedule matches what is on disk
    python scripts/rebuild_pyramid.py s3://bucket/prefix/volume --start-level 2 --dry-run

    # locally, in-process (small volumes)
    python scripts/rebuild_pyramid.py /mnt/ceph/.../vol.zarr --start-level 2 --serial

    # on SLURM, surviving logout
    nohup python -u scripts/rebuild_pyramid.py s3://bucket/prefix/volume \
        --start-level 2 --config configs/dask-slurm-gen.yaml --workers 48 \
        > rebuild.log 2>&1 &
    squeue -u "$USER"      # read-only; don't poll in a tight loop

**--min-dim / --max-levels / --factors must match the original conversion.** They
decide how many levels the pyramid has, so different values here rebuild a
different pyramid. --dry-run prints the computed schedule beside the levels that
actually exist, which is how you check.

--kind defaults to what the volume records (precomputed ``info["type"]``, OME
multiscales ``type``). It selects the reducer -- mean for images, mode for labels
-- and getting it wrong is silent: averaging label ids invents ids that were
never in the data.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild-pyramid")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume", help="the multiscale volume to repair (path or s3://...)")
    p.add_argument("--start-level", type=int, required=True,
                   help="level to derive from; it is read, never written")
    p.add_argument("--kind", choices=("image", "segmentation"), default=None,
                   help="reducer: mean vs mode (default: read from the volume)")

    p.add_argument("--min-dim", type=int, default=128,
                   help="stop when the largest spatial dim is <= this (default 128)")
    p.add_argument("--max-levels", type=int, default=8)
    p.add_argument("--factors", default=None,
                   help="explicit per-level factors, e.g. '1,2,2;1,2,2' (default: auto)")
    p.add_argument("--chunk", default=None, help="z,y,x (default: level 0's chunking)")
    p.add_argument("--voxel-size", default=None,
                   help="z,y,x nm; overrides the volume's own metadata")
    p.add_argument("--encoding", default=None, help="precomputed encoding (e.g. raw)")
    p.add_argument("--profile", default=None, help="storage profile (default: by format)")

    p.add_argument("--config", default="configs/dask-local.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--serial", action="store_true", help="no dask; run in this process")
    p.add_argument("--resume", action="store_true", help="continue an interrupted rebuild")
    p.add_argument("--progress-path", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan and exit, touching nothing")
    return p.parse_args(argv)


def _triple(s, name):
    if s is None:
        return None
    parts = tuple(int(v) for v in s.replace("x", ",").split(","))
    if len(parts) != 3:
        raise SystemExit(f"--{name} needs 3 comma-separated values, got {s!r}")
    return parts


def _factors(s):
    if s is None:
        return None
    return [tuple(int(v) for v in level.split(",")) for level in s.split(";") if level]


def _existing_levels(volume, fmt, n):
    """Which levels are actually present, and their shapes."""
    from em_volume_tools.backends.base import open_backend

    out = {}
    for i in range(n):
        spec = ({"backend": fmt, "path": f"{volume.rstrip('/')}/{i}"} if fmt == "zarr3"
                else {"backend": fmt, "path": volume, "scale_index": i})
        try:
            out[i] = tuple(int(s) for s in open_backend(spec).shape)
        except Exception:
            pass
    return out


def _plan(args):
    """Compute the schedule and compare it with what is on disk."""
    from em_volume_tools.backends.base import open_backend
    from em_volume_tools.introspect import detect_backend, read_source_metadata
    from em_volume_tools.pyramid import downsample_schedule

    fmt = detect_backend(args.volume)
    if fmt is None:
        raise SystemExit(f"no volume found at {args.volume}")
    meta = read_source_metadata({"backend": fmt, "path": args.volume})
    if not meta and args.voxel_size is None:
        raise SystemExit(f"{args.volume} has no coordinate metadata; pass --voxel-size")

    voxel = _triple(args.voxel_size, "voxel-size") or tuple(meta["voxel_size"])
    level0 = open_backend(meta["data_spec"] if meta else {"backend": fmt, "path": args.volume})
    shape = tuple(int(s) for s in level0.shape)
    has_ch = meta["has_channels"] if meta else False
    spatial = shape[1:] if has_ch else shape

    schedule = downsample_schedule(spatial, voxel, factors=_factors(args.factors),
                                   max_levels=args.max_levels, min_dim=args.min_dim)
    shapes, vox = [spatial], [voxel]
    for f in schedule:
        shapes.append(tuple(-(-s // x) for s, x in zip(shapes[-1], f)))
        vox.append(tuple(v * x for v, x in zip(vox[-1], f)))

    kind = args.kind or (meta or {}).get("kind")
    return fmt, meta, shapes, vox, kind, len(schedule)


def main(argv=None) -> int:
    args = _parse_args(argv)
    fmt, meta, shapes, vox, kind, n_sched = _plan(args)
    existing = _existing_levels(args.volume, fmt, n_sched + 1)

    log.info("volume  %s", args.volume)
    log.info("format  %s   kind %s%s", fmt, kind,
             "" if args.kind else " (from the volume's metadata)")
    if kind is None:
        raise SystemExit("this volume records no image/segmentation type; pass --kind")

    mismatch = False
    for i, (shp, v) in enumerate(zip(shapes, vox)):
        on_disk = existing.get(i)
        if on_disk is not None and tuple(on_disk[-len(shp):]) != tuple(shp):
            state, mismatch = f"MISMATCH on disk {on_disk}", True
        elif i < args.start_level:
            state = "kept"
        elif i == args.start_level:
            state = "SEED (read only)" if on_disk else "MISSING -- cannot seed from it"
        else:
            state = "rebuild" + ("" if on_disk else " (creates level)")
        log.info("  level %d  %-22s %-16s %s", i, str(tuple(shp)),
                 "x".join(f"{x:g}" for x in v) + " nm", state)

    if args.start_level > n_sched:
        raise SystemExit(f"--start-level {args.start_level} exceeds the deepest level "
                         f"({n_sched}) this schedule produces")
    if args.start_level not in existing:
        raise SystemExit(f"level {args.start_level} does not exist; nothing to seed from")
    if mismatch:
        raise SystemExit(
            "a level on disk disagrees with the computed schedule -- --min-dim / "
            "--max-levels / --factors probably differ from the original conversion. "
            "Rebuilding now would leave the pyramid inconsistent.")
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    from em_volume_tools import rebuild_pyramid, start_dask

    kw = dict(start_level=args.start_level, kind=kind, profile=args.profile,
              voxel_size=_triple(args.voxel_size, "voxel-size"),
              factors=_factors(args.factors), max_levels=args.max_levels,
              min_dim=args.min_dim, chunk=_triple(args.chunk, "chunk"),
              encoding=args.encoding, resume=args.resume,
              progress_path=args.progress_path)

    if args.serial:
        log.info("serial mode: no dask client")
        summary = rebuild_pyramid(args.volume, client=None, **kw)
    else:
        with start_dask(args.workers, config_path=args.config, label="em-rebuild") as client:
            summary = rebuild_pyramid(args.volume, client=client, **kw)

    log.info("rebuilt levels %d-%d of %s", args.start_level + 1, summary["num_levels"] - 1,
             args.volume)
    log.info("status counts: %s", summary["status_counts"])
    log.info("progress: %s", summary["progress_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
