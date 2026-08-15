"""em-vol: convert, downsample and inspect large 3D volumes, locally or on SLURM.

The command-line entry point for this package. Every write is block-mapped over a dask
cluster (local or SLURM, chosen by ``--config``) and resumable, so an interrupted run
continues where it stopped rather than starting over.

    em-vol info    <volume>                      # what is it, what levels exist
    em-vol convert --src ... --dst ...           # build a multiscale volume
    em-vol copy    --src ... --dst ...           # copy one as it is, whole or a box
    em-vol downsample <volume> --start-level 2   # rebuild levels above a trusted one
    em-vol progress <volume>                     # chunks written, per level
    em-vol create  <dst> --like <reference>      # an EMPTY volume in a known frame
    em-vol write   <volume> --src ... --offset   # put one subvolume into it
    em-vol align-bbox --volume ... --bbox ...     # move a box onto the block grid
    em-vol bboxes-json <volume>                  # a viewer layer of boxes over the data
    em-vol annotate-json --points syn.csv        # a viewer layer of your own coordinates
    em-vol relabel <volume> --out ...            # one id range per occupied region
    em-vol ng-url-gen --image ... --seg ...      # a neuroglancer link with a full state

``convert`` and ``copy`` are the same operation under two defaulting policies: convert
states the output it wants, copy takes the source's own format, chunking, voxel size and
image/segmentation type. Either copies the whole volume or one ``--crop-bbox``.

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

from em_blockrun import bundled_configs

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


def _add_convert_args(p, *, source_defaults: bool):
    """The arguments shared by ``convert`` and ``copy``.

    ``source_defaults`` is the only difference between the two subcommands: with it,
    ``--format`` / ``--kind`` / ``--voxel-size`` / ``--chunk`` default to whatever the
    source records instead of to fixed values, and a source that records none of them
    is an error rather than a silent guess. Everything else — the pyramid schedule, the
    crop, resume, the cluster — is identical, so it lives here and cannot drift.
    """
    from_source = " (default: from the source)" if source_defaults else ""
    p.add_argument("--src", required=True,
                   help="path, s3://..., precomputed, .zarr, or a DVID labelmap as "
                        "dvid://SERVER/UUID/INSTANCE — three segments, always: SERVER "
                        "may carry a port (emdata3:8900) and defaults to http, UUID may "
                        "be abbreviated or carry a branch (93fdbc:main), INSTANCE is the "
                        "labelmap name. Use dvid+https:// for a TLS server. e.g. "
                        "dvid://dvid.example.org/93fdbc:main/labels")
    p.add_argument("--dvid-supervoxels", action="store_true",
                   help="read SUPERVOXELS rather than agglomerated bodies (DVID sources "
                        "only). Default is bodies, i.e. the proofread segmentation")
    p.add_argument("--dvid-locked", action="store_true",
                   help="pull the newest LOCKED node at or before the given ref, rather "
                        "than the ref itself (DVID sources only). A branch ref such as "
                        "93fdbc:main resolves to the branch HEAD, which in a "
                        "lock-and-spawn repo is the open, still-mutable node — legal to "
                        "read, but its data can change while the run is going and the "
                        "pull is not reproducible. Needs the repo:branch form, since "
                        "walking ancestors uses DVID's ref~N syntax. Either way the "
                        "resolved node id is recorded in provenance.json")
    p.add_argument("--dst", required=True,
                   help="destination base (path or s3://bucket/prefix). With "
                        "--format both, '.precomputed' and '.zarr' are appended. "
                        "For a DVID source it may contain placeholders resolved from "
                        "the node actually exported: {uuid} (8 hex chars; {uuid:N} for "
                        "N, {uuid:full} for all 32), {branch}, {instance} — so "
                        "'seg_{uuid}' names the export after the version it came from, "
                        "which a branch ref alone does not")
    p.add_argument("--format", choices=("precomputed", "zarr", "both"),
                   default=None if source_defaults else "precomputed",
                   help="output format" + (from_source or " (default: precomputed)"))
    # Default None for BOTH commands now: `convert` resolves explicit > source > image,
    # so a source that records `segmentation` is no longer downgraded to `image` and
    # averaged into label ids that were never in the data. Only a source recording
    # nothing (image stack, HDF5, bare array) still falls back to image.
    p.add_argument("--kind", choices=("image", "probability", "segmentation"),
                   default=None,
                   help="reducer for the pyramid: mean for image/probability, "
                        "label-preserving mode for segmentation. Getting this wrong "
                        "is silent — averaging label ids invents ids. Default: from the "
                        "source where it records one, else image")
    p.add_argument("--voxel-size", default=None,
                   help="z,y,x nm (default: from source)")
    p.add_argument("--chunk", default=None if source_defaults else "128,128,128",
                   help="z,y,x chunk" + (from_source or " (default: 128,128,128)"))
    p.add_argument("--shard", default=None,
                   help="z,y,x shard (zarr only)" + from_source)
    p.add_argument("--crop-bbox", default=None, metavar="Z0,Y0,X0,Z1,Y1,X1",
                   help="copy only this box instead of the whole volume, in voxels at "
                        "--bbox-scale. Half-open, clipped to the volume. The output "
                        "keeps the source's frame — its physical offset shifts by the "
                        "crop origin — so the two overlay in a viewer")
    p.add_argument("--mask-bbox", action="append", default=None,
                   metavar="Z0,Y0,X0,Z1,Y1,X1",
                   help="EXCLUDE this box: copy everything else, and write --mask-value "
                        "inside it. Repeatable. The hole is inherited by every pyramid "
                        "level, since each is derived from the one below. Coordinates "
                        "are the SOURCE's, so a mask means the same box whether or not "
                        "--crop-bbox is also given. A box that misses the volume is an "
                        "error, not a no-op")
    p.add_argument("--mask-value", type=float, default=0,
                   help="what to write inside --mask-bbox (default: 0, which is "
                        "background for a segmentation and is elided rather than stored)")
    p.add_argument("--background", default=None, metavar="V[,V...]",
                   help="value(s) the SOURCE uses for background, replaced with 0 as it "
                        "is read. Manual segmentation numbered from 0 makes background 1, "
                        "and an all-background block of 1s is not all-fill, so without "
                        "this every such block is stored and the volume stops answering "
                        "'where is the data' by which chunks exist")
    p.add_argument("--bbox-scale", type=int, default=0, metavar="N",
                   help="the scale EVERY bbox argument is given in (default: 0, level-0 "
                        "voxels). Converted using the source's own per-level voxel "
                        "sizes, never an assumed 2**N, since real pyramids are "
                        "anisotropic. The same six values name a different box at "
                        "every scale, so check the resolved level-0 box that is logged")
    p.add_argument("--bbox-order", choices=("zyx", "xyz"), default="zyx",
                   help="the axis order of EVERY bbox argument (default: zyx, as "
                        "everywhere else here). Pass xyz for numbers copied straight out "
                        "of neuroglancer, which displays xyz; each corner is reversed, "
                        "and the resolved zyx box is logged")
    p.add_argument("--profile", default=None,
                   help="storage profile name (default: chosen from --format and "
                        "whether --dst is remote)")
    p.add_argument("--factors", default=None,
                   help="explicit per-level factors, e.g. '1,2,2;1,2,2' (default auto). "
                        "An explicit schedule is used verbatim: --max-levels and "
                        "--min-dim do not apply to it")
    p.add_argument("--max-levels", type=int, default=8,
                   help="at most this many levels, COUNTING LEVEL 0 (default: 8, i.e. "
                        "levels 0-7). Usually --min-dim stops the pyramid first")
    p.add_argument("--min-dim", type=int, default=128,
                   help="stop when the largest spatial dim is <= this")
    p.add_argument("--single-level", action="store_true",
                   help="write level 0 only, no pyramid")
    p.add_argument("--sparse", action="store_true",
                   help="skip PYRAMID tasks whose input holds no stored chunk. On a "
                        "sparse volume that is nearly all of them, and it is exact "
                        "rather than a guess: an all-fill chunk is never stored, so a "
                        "task with no stored input would write nothing anyway. One "
                        "listing per level replaces the reads. It cannot skip any of "
                        "the level-0 copy, whose source is foreign")
    p.add_argument("--fresh", action="store_true",
                   help="delete and restart instead of resuming")
    p.add_argument("--no-validate", action="store_true",
                   help="skip OME metadata validation")
    _add_cluster_args(p)


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


def _sextuple(value, name, order="zyx"):
    """'z0,y0,x0,z1,y1,x1' -> two 3-tuples of ints (start, stop), always zyx.

    ``order="xyz"`` reverses **each corner**, not the six values: the input is then
    ``x0,y0,z0,x1,y1,z1``, which is the order neuroglancer displays and therefore the
    order a box gets copied out of a viewer in.
    """
    parts = tuple(int(v) for v in value.split(","))
    if len(parts) != 6:
        raise SystemExit(f"--{name} needs 6 comma-separated values "
                         f"({','.join(order[i] + d for d in '01' for i in range(3))}), "
                         f"got {value!r}")
    start, stop = parts[:3], parts[3:]
    if order == "xyz":
        start, stop = start[::-1], stop[::-1]
    if any(b <= a for a, b in zip(start, stop)):
        raise SystemExit(f"--{name} is empty or inverted: every start must be below its "
                         f"stop, got start={start} stop={stop}")
    return start, stop


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
    """``describe`` with a missing or unaddressable volume turned into a clean CLI exit.

    ``ValueError`` is caught alongside ``FileNotFoundError`` because a location can now
    be malformed rather than merely absent: a ``dvid://`` URL with the wrong number of
    segments is a user typo, and its message already says exactly what is wrong, so it
    should read as an error rather than as a traceback.
    """
    try:
        return describe(volume)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
def _print_dvid_nodes(volume: str) -> None:
    """Which node a ref points at now, and the newest locked one.

    Both, because they are different answers to "which version would I get" and the
    difference is exactly the reproducibility question: HEAD of a branch is the open,
    still-mutable node in a lock-and-spawn repo. The uuids are printed in full so a
    destination name can be built from one — that is the usual reason to run this.
    """
    from em_volume_tools.backends.dvid import node_summary
    from em_volume_tools.source_metadata import location_spec

    try:
        summary = node_summary(location_spec(volume, "dvid"))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  nodes       (could not resolve: {type(exc).__name__}: {exc})")
        return

    head, locked = summary["head"], summary["locked"]
    print(f"  requested   {head['ref']}")
    print(f"  latest      {head['uuid']}  {'locked' if head['locked'] else 'OPEN'}")
    if locked is None:
        print(f"  latest lock (none reachable: {summary['locked_error']})")
    elif locked["uuid"] == head["uuid"]:
        print("  latest lock (same as above)")
    else:
        print(f"  latest lock {locked['uuid']}  locked, "
              f"{locked['walked']} ancestor(s) back  [--dvid-locked]")
    if not head["locked"]:
        print("              the latest node is OPEN: its data can still change, so an "
              "export from it\n              is not reproducible. --dvid-locked takes "
              "the locked one instead.")


def _print_provenance(volume: str, full: bool) -> None:
    """Report ``provenance.json`` if the volume has one.

    A one-line summary by default and the whole document under ``--provenance``: the
    record is what answers "which proofreading snapshot is this", which is worth
    surfacing without being asked, but it is too long to print in full every time.
    """
    import json

    from em_volume_tools.location import read_json
    from em_volume_tools.ops.provenance import FILENAME

    try:
        rec = read_json(volume, FILENAME)
    except Exception:                                          # noqa: BLE001
        return                                                 # not a store, or no access
    if not rec:
        if full:
            print(f"\n  no {FILENAME} here — it is written by `em-vol convert`, so a "
                  f"volume made\n  another way (or before provenance existed) has none.")
        return

    src = rec.get("source") or {}
    if full:
        print(f"\n  {FILENAME}:")
        for line in json.dumps(rec, indent=2).splitlines():
            print(f"    {line}")
        return
    written, origin = rec.get("written", "?"), src.get("url") or src.get("source") or "?"
    print(f"\n  provenance  from {origin}")
    if src.get("uuid"):
        print(f"              node {src['uuid']} "
              f"({'locked' if src.get('locked') else 'OPEN'})")
    print(f"              written {written}   (--provenance for the full record)")


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
    if d["format"] == "dvid":
        _print_dvid_nodes(args.volume)
    else:
        _print_provenance(args.volume, getattr(args, "provenance", False))
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
# convert  (+ copy, which is convert with the source's own parameters as defaults)
# --------------------------------------------------------------------------- #
def _warn_if_no_aws(dst: str) -> None:
    if dst.startswith("s3://") and not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.path.exists(os.path.expanduser("~/.aws/credentials"))):
        log.warning("no AWS credentials in the environment or ~/.aws/credentials; "
                    "S3 writes will fail with AccessDenied")


def _src_spec(args, src: str | None = None):
    """The source spec for a peek at metadata, carrying this command's source options.

    The crop helpers below read the source's voxel sizes *before* `convert` runs, so
    they build their own spec — and if that spec omits `--dvid-locked`, the peek
    resolves the branch ref to a different node than the run does, and warns that the
    node is open when the run is about to use a locked one. One builder, so the peek and
    the run always ask about the same thing.

    Returns ``(spec, fmt)``; ``fmt`` is None when nothing at the location is a volume.
    """
    from em_volume_tools.source_metadata import detect_backend, location_spec

    src = args.src if src is None else src
    fmt = detect_backend(src)
    if fmt is None:
        return None, None
    spec = location_spec(src, fmt)
    if fmt == "dvid":
        if getattr(args, "dvid_locked", False):
            spec["prefer_locked"] = True
        if getattr(args, "dvid_supervoxels", False):
            spec["supervoxels"] = True
    return spec, fmt


def _level0_factor(src: str, scale: int, per_level=None, args=None) -> tuple[int, ...]:
    """How many level-0 voxels one scale-``scale`` voxel spans, per axis.

    Read from each level's OWN recorded voxel size, never ``2**scale``: real pyramids
    are anisotropic and ``(1, 2, 2)`` — halve x/y, leave z — is common (CLAUDE.md
    invariant 1). Shape ratios are not used either; ceil-division makes them inexact.

    ``args`` is optional so the volume-argument commands (``align-bbox``, ``relabel``)
    can keep calling with a bare path; where it is given the source options travel with
    it, so a DVID peek resolves the same node the run will use.
    """
    from em_volume_tools.source_metadata import (detect_backend, location_spec,
                                                 read_level_voxel_sizes)

    if per_level is None:
        if args is not None:
            spec, fmt = _src_spec(args, src)
        else:
            fmt = detect_backend(src)
            spec = location_spec(src, fmt) if fmt else None
        if fmt is None:
            raise SystemExit(f"--bbox-scale {scale} needs the source's per-level voxel "
                             f"sizes and nothing at {src} looks like a volume")
        per_level = read_level_voxel_sizes(spec)
    if not per_level:
        raise SystemExit(f"--bbox-scale {scale} needs the source's per-level voxel "
                         f"sizes, and {src} records none. Give --crop-bbox in level-0 "
                         f"voxels instead (--bbox-scale 0).")
    if scale >= len(per_level):
        raise SystemExit(f"--bbox-scale {scale}: the source records only "
                         f"{len(per_level)} level(s) (0-{len(per_level) - 1})")
    factor = tuple(s / b for s, b in zip(per_level[scale], per_level[0]))
    if any(abs(f - round(f)) > 1e-6 for f in factor):
        raise SystemExit(f"--bbox-scale {scale}: its voxel size {per_level[scale]} is "
                         f"not an integer multiple of level 0's {per_level[0]}, so a "
                         f"box there does not land on level-0 voxels. Use "
                         f"--bbox-scale 0.")
    return tuple(int(round(f)) for f in factor)


def _src_voxel_size(args):
    """The source's own level-0 voxel size, or ``None``. Metadata reads only."""
    from em_volume_tools.source_metadata import read_source_metadata

    spec, fmt = _src_spec(args)
    meta = read_source_metadata(spec) if fmt else None
    return tuple(meta["voxel_size"]) if meta else None


def _resolve_bbox(value, name, args, per_level=None):
    """One ``--*-bbox`` value as ``(lo, hi)`` in level-0 zyx voxels.

    Applies the command's two shared bbox conventions in order: ``--bbox-order``, then
    ``--bbox-scale``. One flag each per command rather than per box argument, because a
    crop at one scale and a mask at another is a mistake nobody makes on purpose — while
    setting only one of two scale flags is a mistake anyone could make once, and it puts
    the mask in the wrong place with nothing to show for it.
    """
    lo, hi = _sextuple(value, name, order=getattr(args, "bbox_order", "zyx"))
    scale = getattr(args, "bbox_scale", 0) or 0
    if scale:
        factor = _level0_factor(args.src, scale, per_level, args)
        lo = tuple(a * f for a, f in zip(lo, factor))
        hi = tuple(b * f for b, f in zip(hi, factor))
        log.info("--%s is in scale-%d voxels (%s level-0 voxels each): level-0 box "
                 "%s:%s", name, scale, factor, lo, hi)
    elif getattr(args, "bbox_order", "zyx") == "xyz":
        log.info("--%s read as xyz: level-0 zyx box %s:%s", name, lo, hi)
    return lo, hi


def _crop_bbox(args, per_level=None):
    """``(start, stop)`` in level-0 voxels from ``--crop-bbox``, or ``(None, None)``."""
    if not getattr(args, "crop_bbox", None):
        return None, None
    return _resolve_bbox(args.crop_bbox, "crop-bbox", args, per_level)


def _mask_bboxes(args, per_level=None):
    """The ``--mask-bbox`` boxes as ``[(lo, hi), ...]`` in level-0 voxels."""
    return [_resolve_bbox(v, "mask-bbox", args, per_level)
            for v in (getattr(args, "mask_bbox", None) or [])]


def _pyramid_levels(shape, voxel, args):
    """``[(shape, voxel_size)]`` per level, for the schedule these arguments imply."""
    from em_volume_tools.pyramid import downsample_schedule

    if args.single_level:
        return [(tuple(shape), tuple(voxel))]
    schedule = downsample_schedule(shape, voxel, factors=_factor_list(args.factors),
                                   max_levels=args.max_levels, min_dim=args.min_dim)
    levels = [(tuple(shape), tuple(voxel))]
    for factor in schedule:
        levels.append((tuple(-(-s // f) for s, f in zip(levels[-1][0], factor)),
                       tuple(v * f for v, f in zip(levels[-1][1], factor))))
    return levels


def _warn_if_crop_unaligned(start, shape, voxel, args) -> None:
    """A crop origin off the pyramid's own grid shifts every coarse level.

    The output's pyramid is built from the *cropped* level 0, so its reduction windows
    start at the crop origin. When that origin is not a multiple of a level's
    cumulative factor, that level's voxels straddle the source's coarse voxels
    differently and its ``voxel_offset`` rounds to the nearest coarse voxel — the
    output still overlays the source at level 0 while drifting by up to half a voxel at
    the top. Aligning the origin avoids it; nothing else can, since the crop defines
    the grid.
    """
    from em_volume_tools.pyramid import cumulative_factors, downsample_schedule

    if args.single_level or not voxel:
        return  # no pyramid, or no voxel size to derive the schedule's factors from
    schedule = downsample_schedule(shape, voxel, factors=_factor_list(args.factors),
                                   max_levels=args.max_levels, min_dim=args.min_dim)
    coarsest = cumulative_factors(schedule, len(shape))[-1]
    if any(a % f for a, f in zip(start, coarsest)):
        aligned = tuple(a - a % f for a, f in zip(start, coarsest))
        log.warning("crop origin %s is not a multiple of the coarsest cumulative "
                    "factor %s: the pyramid is built from the cropped level 0, so its "
                    "coarse levels sit on a different grid than the source's and their "
                    "voxel_offset rounds. Level 0 is exact either way. Use origin %s "
                    "to keep both grids in step.", tuple(start), coarsest, aligned)


def _level0_chunking(d, src):
    """``(read chunk, write chunk)`` of a source's level 0, either possibly ``None``.

    ``describe`` reports chunking per level, but only for levels it can *find*: a bare
    zarr array has no level subdirectories, so it reports none and its chunking lives
    on the array itself. They differ when the level is sharded — the write chunk is the
    shard — and a copy wants to carry both across.
    """
    from em_volume_tools.backends.base import open_backend

    lvl0 = d["levels"].get(0) or {}
    if lvl0.get("read_chunks") or lvl0.get("chunks"):
        return lvl0.get("read_chunks"), lvl0.get("chunks")
    meta = d["meta"] or {}
    try:
        be = open_backend(meta.get("data_spec") or {"backend": d["format"], "path": src})
    except Exception as e:
        # A backend that cannot be opened here can still be read by the workers — the
        # cloudvolume one needs a package only the em-vol-cv environment carries. Fall
        # back to the profile's default chunking rather than fail before starting.
        log.info("could not read the source's chunking (%s); using the profile default", e)
        return None, None

    def _maybe(attr):
        try:
            return tuple(int(c) for c in getattr(be, attr))
        except Exception:
            return None

    return _maybe("read_chunks"), _maybe("chunks")


def _convert_targets(fmt, dst, profile_arg, chunk, shard):
    """``(profile, destination)`` per requested output format.

    ``both`` writes two volumes from one read-side setup: precomputed for viewing, zarr
    for downstream compute. Suffixes keep them distinguishable at one ``--dst``.
    """
    from em_volume_tools import StorageProfile

    targets: list[tuple[object, str]] = []
    if fmt in ("precomputed", "both"):
        profile = profile_arg or ("s3-neuroglancer" if dst.startswith("s3://")
                                  else "local-neuroglancer")
        targets.append((profile, dst + (".precomputed" if fmt == "both" else "")))
    if fmt in ("zarr", "both"):
        # `chunk` is None only when nothing asked for one and the source records none;
        # a StorageProfile built with chunk=None fails deep inside the create spec, so
        # fall back to the named profile's own default.
        profile = profile_arg or (StorageProfile("zarr3", chunk=chunk, shard=shard)
                                  if chunk else "local")
        targets.append((profile, dst + (".zarr" if fmt == "both" else "")))
    return targets


def _warn_if_masking_a_destination_that_exists(args, dst, masks) -> None:
    """The one way a mask leaks: a destination that already holds the region.

    A block wholly inside the mask reads as all fill, and ``_copy_block`` returns "empty"
    *without writing* — that elision is what keeps a sparse copy cheap (8,615 of 10,692
    blocks on this dataset). So nothing clears an object already sitting at that key from
    an earlier, unmasked run: the excluded region would survive in the new volume, and
    every level above it, with the run reporting success.
    """
    from em_volume_tools.location import exists

    if not masks or getattr(args, "fresh", False):
        return
    if any(exists(dst, marker) for marker in ("info", "zarr.json")):
        log.warning(
            "%s already exists and --mask-bbox was given without --fresh. Blocks wholly "
            "inside a mask are written as fill, and an all-fill block is ELIDED rather "
            "than written — so anything already stored in the masked region stays there, "
            "in every level, and the run still reports success. Use --fresh (or delete "
            "the destination) unless you are resuming a run that had the same mask.", dst)


def _expand_dst(args, dst: str) -> str:
    """Resolve ``{uuid}``-style placeholders in a destination, and say what they became.

    Reported rather than done quietly: the resolved path is where the data lands and
    what the progress manifest is named after, so it is the thing to copy out of the log.
    """
    from em_volume_tools.ops.naming import expand, has_placeholder

    if not has_placeholder(dst):
        return dst
    spec, fmt = _src_spec(args)
    if fmt is None:
        raise SystemExit(f"--dst {dst!r} contains a placeholder, which is resolved from "
                         f"the source, and nothing at {args.src} looks like a volume")
    # Resolution has to match what the run will read, so go through the same spec
    # builder — with --dvid-locked that is a different node than the ref points at.
    from em_volume_tools.source_metadata import read_source_metadata

    meta = read_source_metadata(spec) or {}
    try:
        out = expand(dst, meta.get("provenance_spec") or spec)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    log.info("--dst %s -> %s", dst, out)
    return out


def _run_convert(args, *, fmt, voxel, chunk, shard, kind, crop, masks=()) -> int:
    """The shared body of ``convert`` and ``copy``: block-map the copy per target."""
    from em_volume_tools import convert

    # Before anything derives from it: the format-suffixed targets, the progress
    # manifest name and the resume check all take the destination as given, so a path
    # still holding `{uuid}` would reach every one of them.
    dst = _expand_dst(args, args.dst.rstrip("/"))
    _warn_if_no_aws(dst)
    crop_start, crop_stop = crop
    with _maybe_cluster(args) as client:
        for profile, target in _convert_targets(fmt, dst, args.profile, chunk, shard):
            _warn_if_masking_a_destination_that_exists(args, target, masks)
            log.info("%s %s -> %s (%s)", args.command, args.src, target, kind)
            # Said before the run, not after: it is what you watch progress with, and the
            # digest in a remote destination's name is not something to reconstruct by eye.
            from em_volume_tools.location import default_progress_path

            log.info("progress: %s", default_progress_path(target))
            summary = convert(
                args.src, target, voxel_size=voxel,
                src_format=getattr(args, "src_format", None),
                profile=profile, kind=kind, chunk=chunk,
                crop_start=crop_start, crop_stop=crop_stop,
                mask_boxes=[[lo, hi] for lo, hi in masks],
                mask_value=args.mask_value,
                background=(_int_list(args.background, "background")
                            if args.background else None),
                supervoxels=getattr(args, "dvid_supervoxels", False),
                prefer_locked=getattr(args, "dvid_locked", False),
                factors=_factor_list(args.factors), max_levels=args.max_levels,
                min_dim=args.min_dim, multiscale=not args.single_level,
                sparse=args.sparse,
                client=client, resume=not args.fresh, delete_existing=args.fresh,
                validate=not args.no_validate,
            )
            log.info("done %s: %d levels, status=%s", target,
                     summary["num_levels"], summary.get("status_counts"))
    return 0


def cmd_convert(args) -> int:
    chunk = _triple(args.chunk, "chunk")
    voxel = _ftriple(args.voxel_size, "voxel-size")
    crop = _crop_bbox(args)
    masks = _mask_bboxes(args)
    if crop[0] is not None:
        _warn_if_crop_unaligned(crop[0], tuple(b - a for a, b in zip(*crop)),
                                voxel or _src_voxel_size(args), args)
    return _run_convert(args, fmt=args.format, voxel=voxel, chunk=chunk,
                        shard=_triple(args.shard, "shard"), kind=args.kind, crop=crop,
                        masks=masks)


def cmd_copy(args) -> int:
    """Copy a volume, or a box out of it, keeping the source's own parameters."""
    # DVID is not a storage format we can reproduce, so there is nothing here to
    # "copy": every one of `copy`'s source-derived defaults either does not exist
    # (format) or is the wrong choice for an output (DVID's 64^3 blocks, which give 8x
    # the object count of the usual 128^3 — and ceph enforces inode quotas). Calling it
    # `copy` also suggests the result is a duplicate of a thing that is, in the open-node
    # case, still changing. `convert` now inherits kind and voxel size from the source
    # anyway, so it does everything `copy` would have here and says what it does.
    from em_volume_tools.backends.dvid import is_url as _is_dvid_url

    if _is_dvid_url(args.src):
        raise SystemExit(
            f"`copy` does not take a DVID source. Nothing is being duplicated — DVID is "
            f"a server, not a storage format, so this is a conversion into precomputed "
            f"or zarr. Use `convert`, which reads the kind and voxel size from the "
            f"instance just as `copy` would:\n\n"
            f"    em-vol convert --src {args.src} --dst {args.dst}\n\n"
            f"Add --dvid-locked for an immutable node, and --crop-bbox to take a box.")

    try:
        d = describe(args.src)
    except ValueError as e:
        raise SystemExit(str(e)) from None       # malformed location, not a missing one
    except FileNotFoundError:
        raise SystemExit(
            f"no volume found at {args.src}. `copy` takes its format, chunking, voxel "
            f"size and image/segmentation type from the source, so it needs a volume "
            f"that records them: precomputed (`info`), an OME-NGFF zarr group, or a "
            f"DVID labelmap (dvid://server/uuid/instance). For an image stack, HDF5 or "
            f"a bare array, use `em-vol convert` and state them.") \
            from None
    meta = d["meta"] or {}

    # A `.gz`-chunked source (CloudVolume) copies out as PLAIN precomputed, which is
    # much of the point: the copy is addressable by tensorstore, the original is not.
    # DVID is the same case for a different reason: it is not a storage format we can
    # write at all, so precomputed here is a DEFAULT rather than the source's own —
    # which is why `from_source` below excludes it from the "(from the source)" note.
    fmt = args.format or ("zarr" if d["format"] == "zarr3" else "precomputed")
    fmt_from_source = not args.format and d["format"] in ("zarr3",
                                                          "neuroglancer_precomputed")
    kind = args.kind or meta.get("kind")
    if kind is None:
        raise SystemExit(
            f"{args.src} records no image/segmentation type; pass --kind. Guessing "
            f"wrong is silent and destructive — averaging label ids invents ids that "
            f"were never in the data.")
    voxel = _ftriple(args.voxel_size, "voxel-size") or (meta.get("voxel_size") and
                                                        tuple(meta["voxel_size"]))
    if not voxel:
        raise SystemExit(f"{args.src} records no voxel size; pass --voxel-size")
    # Sharded levels report the inner read chunk and the shard separately; carry both.
    read_chunk, write_chunk = _level0_chunking(d, args.src)
    chunk = _triple(args.chunk, "chunk") or read_chunk or write_chunk
    shard = _triple(args.shard, "shard")
    if shard is None and read_chunk and write_chunk and read_chunk != write_chunk:
        shard = write_chunk

    # The box reported here is the one that will be copied: `convert` clips a crop to
    # the source extent rather than padding, so clip before reporting or the printed
    # shape and byte count are not what runs.
    spatial = tuple(d["shape"][1:] if d["has_channels"] else d["shape"])
    crop_start, crop_stop = _crop_bbox(args, d["level_voxel_sizes"])
    masks = _mask_bboxes(args, d["level_voxel_sizes"])
    start = tuple(max(0, a) for a in crop_start) if crop_start else (0,) * len(spatial)
    stop = (tuple(min(b, s) for b, s in zip(crop_stop, spatial)) if crop_stop
            else spatial)
    out_shape = tuple(b - a for a, b in zip(start, stop))
    if any(s <= 0 for s in out_shape):
        raise SystemExit(f"--crop-bbox {start}:{stop} does not intersect the volume "
                         f"(spatial shape {tuple(spatial)})")

    import numpy as np

    levels = _pyramid_levels(out_shape, voxel, args)
    nbytes = float(np.dtype(d["dtype"]).itemsize) * math.prod(out_shape)
    dst = args.dst.rstrip("/")
    print(f"{args.src}  ->  {dst}")
    print(f"  format      {fmt}{' (from the source)' if fmt_from_source else ''}")
    print(f"  dtype       {d['dtype']}")
    print(f"  kind        {kind}{'' if args.kind else ' (from the source)'}")
    print(f"  voxel size  {'x'.join(f'{v:g}' for v in voxel)} nm"
          f"{'' if args.voxel_size else ' (from the source)'}")
    print(f"  chunk       {'x'.join(str(c) for c in chunk) if chunk else '(profile default)'}"
          f"{'' if args.chunk else ' (from the source)'}"
          + (f"   shard {'x'.join(str(c) for c in shard)}" if shard else ""))
    if crop_start or crop_stop:
        print("  crop        " + "  ".join(f"{ax} {a}:{b}"
                                           for ax, a, b in zip("zyx", start, stop)))
    for i, (m_lo, m_hi) in enumerate(masks):
        excluded = math.prod(b - a for a, b in zip(m_lo, m_hi))
        print(f"  excluded    " + "  ".join(f"{ax} {a}:{b}" for ax, a, b
                                            in zip("zyx", m_lo, m_hi))
              + f"   = {excluded:,} voxels -> {args.mask_value:g}"
              + (f"  [{i + 1} of {len(masks)}]" if len(masks) > 1 else ""))
    print(f"  copying     {tuple(out_shape)} = {_human_bytes(nbytes)} at level 0"
          f"{'' if (crop_start or crop_stop) else ' (the whole volume)'}")
    print(f"  levels      {len(levels)}"
          + ("" if len(levels) == 1 else
             ", each downsampled from the one below — the source's own coarse levels "
             "are not copied"))
    for i, (shape, vox) in enumerate(levels):
        print(f"  {i:>5}  {str(shape):>24}  {'x'.join(f'{v:g}' for v in vox):>20}")
    if crop_start:
        _warn_if_crop_unaligned(start, out_shape, voxel, args)
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0
    return _run_convert(args, fmt=fmt, voxel=voxel, chunk=chunk, shard=shard,
                        kind=kind, crop=(crop_start, crop_stop), masks=masks)


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

    # The default comes from the volume, not from `convert`'s 8: how many levels this
    # pyramid has is something it already records. Through the op's own resolver, so the
    # --dry-run table cannot state a plan different from the one that runs.
    max_levels = args.max_levels
    if max_levels is None:
        from em_volume_tools.ops.rebuild import resolve_max_levels

        max_levels, why = resolve_max_levels(d["level_voxel_sizes"])
        log.info("levels  at most %d (%s)", max_levels, why)
    schedule = downsample_schedule(spatial, voxel, factors=_factor_list(args.factors),
                                   max_levels=max_levels, min_dim=args.min_dim)
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
    # Levels the schedule does not reach are left exactly as they are — whatever that is.
    # Empty (never built) means the volume vanishes at the coarsest zooms; populated from
    # before the data changed means it serves the OLD data there. Every shape matches
    # either way, so the mismatch check above cannot see this; only the count differs.
    extra = sorted(i for i in existing if i >= len(shapes))
    if extra:
        log.warning("levels %s exist on disk but are ABOVE this schedule, so they are "
                    "left untouched: whatever they hold now — nothing, or data from "
                    "before the change you are rebuilding for — is what they keep "
                    "serving at the coarsest zooms. Raise --max-levels to %d to rebuild "
                    "them (it counts level 0), or delete them.", extra, max(extra) + 1)
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    kw = dict(start_level=args.start_level, kind=kind, profile=args.profile,
              voxel_size=_triple(args.voxel_size, "voxel-size"),
              factors=_factor_list(args.factors), max_levels=args.max_levels,
              min_dim=args.min_dim, chunk=_triple(args.chunk, "chunk"),
              encoding=args.encoding, resume=args.resume, sparse=args.sparse,
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
    """How a resolved source spec should be shown: where it is, and what was read.

    A view backend (crop, mask) carries the real source nested under ``source``, so unwrap
    to it — printing the wrapper's dict tells the reader nothing about which file was read.
    The views it passed through are named after the location.
    """
    views = []
    while isinstance(spec.get("source"), dict):
        views.append(spec["backend"])
        spec = spec["source"]
    where = spec.get("path", spec.get("source", "?"))
    read_as = spec["backend"] + (f":{spec['dataset']}" if spec.get("dataset") else "")
    return f"{where}  [{', '.join([read_as] + views)}]"


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
            voxel_size_field=args.voxel_size_field,
            offset_order=args.offset_order, src_format=args.src_format,
            dataset=args.dataset, cast=args.cast, dry_run=args.dry_run,
            background=(_int_list(args.background, "background")
                        if args.background else None))
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
# bboxes-json
# --------------------------------------------------------------------------- #
def _write_text(path: str, text: str) -> None:
    """Write ``text`` to a local path or an object store, uniformly.

    Through ``location`` rather than ``open()`` so ``--out s3://...`` works: the file
    driver creates parent directories and the s3 driver bootstraps credentials, and
    neither needs a branch here.
    """
    from em_volume_tools.location import write_bytes

    write_bytes(path, text.encode())


def cmd_bboxes_json(args) -> int:
    """A neuroglancer annotation layer of bounding boxes over a sparse volume's data.

    The JSON goes to **stdout** and the human summary to **stderr**, so
    ``em-vol bboxes-json vol > layer.json`` works and so does reading the table while
    it runs. ``--out`` writes the JSON somewhere instead, local or remote.

    Reads only: the boxes come from listing which chunk objects exist, which on a
    sparse volume is the occupancy question exactly, plus one read per region to
    tighten it. See :mod:`em_volume_tools.ops.annotate` for why the annotations are
    local rather than a precomputed annotation layer.
    """
    from em_volume_tools.ops.annotate import (NoOccupancy, annotation_layer,
                                              labeled_regions, output_dimensions,
                                              render, viewer_state)

    volume = args.volume.rstrip("/")
    # Tightening defaults to the level the footprint came from, so the cost of the
    # reads scales with the level the caller already chose, and at the default
    # --level 0 the boxes are exact rather than quantized to a coarser voxel.
    tighten = None if args.no_tighten else (
        args.level if args.tighten_level is None else args.tighten_level)
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
        print("  nothing is stored — no boxes to make", file=err)
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
        _write_text(args.out, text + "\n")
        print(f"\nwrote {args.out} — " + ("load it with neuroglancer's {} JSON editor"
                                          if args.state else
                                          "paste it into the `layers` array of a "
                                          "neuroglancer state, or pass it to "
                                          "`em-vol ng-url-gen --layer`"), file=err)
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# mask-by-value
# --------------------------------------------------------------------------- #
def _int_list(value, name):
    """'1' or '1,2,3' -> a list of ints."""
    try:
        return [int(v) for v in str(value).replace(" ", "").split(",") if v]
    except ValueError:
        raise SystemExit(f"--{name} takes whole numbers, got {value!r}") from None


def cmd_mask_by_value(args) -> int:
    """Replace label values with background in a volume that already holds them.

    For data already written. Correcting it at ingest is better — `write`, `to-hdf5` and
    `convert` all take --background — because that happens before the storage decision.
    """
    from em_volume_tools.ops.maskvalue import apply_mask_values, plan_mask_values

    try:
        plan = plan_mask_values(args.volume, _int_list(args.values, "values"),
                                out=args.out, in_place=args.in_place, level=args.level,
                                to=args.to)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None

    print(f"{plan['volume']}  ->  "
          f"{'itself (in place)' if plan['in_place'] else plan['destination']}")
    print(f"  level       {plan['level']}  {plan['shape']} {plan['dtype']}  "
          f"chunk {plan['chunk']}")
    print(f"  replacing   {plan['values']} -> {plan['to']}")
    print(f"  regions     {len(plan['regions'])} stored, {plan['n_voxels']:,} voxels to scan")
    if plan["in_place"]:
        print("  NOTE        in place: the original values are not recoverable afterwards. "
              "Chunks that\n              become all-background are removed from the store, "
              "so the sparsity is\n              restored either way.")
    try:
        result = apply_mask_values(plan, dry_run=args.dry_run, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from None

    print(f"  replaced    {result['voxels_replaced']:,} voxel(s)")
    if result["voxels_already_background"]:
        print(f"  WARNING     {result['voxels_already_background']:,} voxel(s) already "
              f"held {plan['to']}; those and the replaced ones are now one value")
    print(f"  blocks      {result['blocks_written']} written"
          + (f", {result['blocks_unchanged']} unchanged and left alone"
             if result["blocks_unchanged"] else ""))
    if result["stale_levels"]:
        print(f"  stale       levels {result['stale_levels']} still hold the old values — "
              f"run `em-vol downsample --start-level {plan['level']}`")
    print("--dry-run: nothing written" if args.dry_run else "done")
    return 0


# --------------------------------------------------------------------------- #
# to-hdf5
# --------------------------------------------------------------------------- #
def cmd_to_hdf5(args) -> int:
    """Pack a small volume into an HDF5 file that `em-vol write` can place.

    The inverse of `write`: this makes the piece, that puts it somewhere. The frame and
    the position travel with the data, so nobody re-types an offset — and because the
    axis order is recorded, `write` no longer has to be told it.
    """
    from em_volume_tools.ops.pack import pack_hdf5

    axes = tuple(args.axes)
    if len(axes) != 3 or set(axes) != {"z", "y", "x"}:
        raise SystemExit(f"--axes must be a permutation of zyx, got {args.axes!r}")
    crop = _sextuple(args.crop_bbox, "crop-bbox") if args.crop_bbox else (None, None)
    try:
        plan = pack_hdf5(
            args.src, args.out, voxel_size=_ftriple(args.voxel_size, "voxel-size"),
            voxel_offset=_triple(args.offset, "offset"), units=args.units, axes=axes,
            level=args.level, crop_start=crop[0], crop_stop=crop[1],
            dataset=args.dataset, src_dataset=args.src_dataset,
            src_format=args.src_format, dtype=args.dtype,
            voxel_size_field=args.voxel_size_field, offset_field=args.offset_field,
            background=(_int_list(args.background, "background")
                        if args.background else None),
            chunk=_triple(args.chunk, "chunk"),
            compression=None if args.compression == "none" else args.compression,
            overwrite=args.overwrite, dry_run=args.dry_run)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        raise SystemExit(str(e).strip("'")) from None

    print(f"{plan['out']}{plan['dataset']}")
    print(f"  source      {_source_label(plan['src_spec'])}"
          + (f"  level {plan['level']}" if plan["level"] else ""))
    if plan["crop_origin"] is not None:
        print("  cropped     " + "  ".join(
            f"{ax} {a}:{a + n}" for ax, a, n in zip(
                plan["axes"], plan["crop_origin"], plan["shape"][-3:])))
    print(f"  shape       {plan['shape']} {plan['dtype']}"
          f"{'  (channel-leading)' if plan['has_channels'] else ''}"
          f"   {_human_bytes(plan['nbytes'])}")
    print(f"  voxel size  {'x'.join(f'{v:g}' for v in plan['voxel_size'])} {plan['units']}")
    print("  voxel_offset " + "  ".join(f"{ax} {o}" for ax, o
                                        in zip(plan['axes'], plan['voxel_offset'])))
    print(f"  axes        {''.join(plan['axes'])}  (recorded, so `em-vol write` need not "
          f"be told the order)")
    print(f"  chunk       {plan['chunk'] or 'up to 64 per axis'}"
          f"   compression {plan['compression'] or 'none'}")
    if plan["other_datasets"]:
        print(f"  also holds  {', '.join(plan['other_datasets'])}"
              f"  — readers must name a dataset")
    if plan["replacing"]:
        print("  replacing   the existing dataset of that name")
    print("--dry-run: nothing written" if args.dry_run else
          f"wrote {plan['blocks']} block(s); place it with `em-vol write <volume> --src "
          f"{plan['out']}" + (f" --dataset {plan['dataset']}"
                              if plan["other_datasets"] else "") + "`")
    return 0


# --------------------------------------------------------------------------- #
# align-bbox
# --------------------------------------------------------------------------- #
def _cumulative_factor(per_level, level: int, volume: str) -> tuple[int, ...]:
    """How many level-0 voxels one voxel of ``level`` spans, per axis.

    From the recorded per-level voxel sizes, never ``2**level`` (invariant 1). Shared by
    both grid choices below, because both are questions about the same ratio: a chunk at
    level N covers ``chunk * factor(N)`` level-0 voxels, and the pyramid's own grid *is*
    the factor of its deepest level.
    """
    if not per_level:
        raise SystemExit(f"{volume} records no per-level voxel sizes, so a level-{level} "
                         f"grid cannot be expressed in level-0 voxels. Give the grid "
                         f"directly with --block.")
    if level >= len(per_level):
        raise SystemExit(f"{volume} records {len(per_level)} level(s) "
                         f"(0-{len(per_level) - 1}), so there is no level {level}")
    factor = tuple(s / b for s, b in zip(per_level[level], per_level[0]))
    if any(abs(f - round(f)) > 1e-6 for f in factor):
        raise SystemExit(f"level {level} of {volume} has voxel size {per_level[level]}, "
                         f"not an integer multiple of level 0's {per_level[0]}: its grid "
                         f"does not land on level-0 voxels")
    return tuple(int(round(f)) for f in factor)


def _align_grid(args, d):
    """``(block in level-0 voxels, where it came from)`` for ``--to`` / ``--block``."""
    from em_volume_tools.grid import lcm_grid

    if args.block:
        return _triple(args.block, "block"), "--block"
    per_level, levels = d["level_voxel_sizes"], d["levels"]

    if args.level == 0 and not levels:
        # A bare array reports no levels at all; its chunking is on the array itself.
        read_chunk, write_unit = _level0_chunking(d, args.volume)
        at = (1, 1, 1)
    else:
        if args.level not in levels:
            raise SystemExit(f"{args.volume} has no level {args.level} "
                             f"(present: {sorted(levels) or 'none'}). Give the grid "
                             f"directly with --block.")
        lvl = levels[args.level]
        write_unit, read_chunk = lvl.get("chunks"), lvl.get("read_chunks")
        # A level's own chunk is in ITS voxels; the box is in level-0 ones.
        at = ((1, 1, 1) if args.level == 0
              else _cumulative_factor(per_level, args.level, args.volume))

    # The write unit is the SHARD where a level is sharded — that is the object a partial
    # write rewrites. `read_chunks` is the inner chunk, which governs read amplification
    # and offers no protection at all against a partial-shard update.
    sharded = bool(read_chunk and write_unit and read_chunk != write_unit)

    def scaled(block, what):
        if not block:
            raise SystemExit(f"level {args.level} of {args.volume} reports no {what}; "
                             f"give the grid directly with --block")
        return tuple(int(c) * f for c, f in zip(block, at)), (
            f"level-{args.level} {what}"
            + (f", x{at} to level-0 voxels" if any(f != 1 for f in at) else ""))

    def pyramid_grid():
        """The deepest level's cumulative factor — the grid every level shares."""
        if not per_level or len(per_level) == 1:
            return (1, 1, 1), "no pyramid: a single level constrains nothing"
        deepest = len(per_level) - 1
        return (_cumulative_factor(per_level, deepest, args.volume),
                f"cumulative factor of level {deepest}, the deepest")

    if args.to == "write-unit":
        return scaled(write_unit, "shard" if sharded else "chunk")
    if args.to == "read-chunk":
        return scaled(read_chunk or write_unit, "read chunk")
    if args.to == "pyramid":
        return pyramid_grid()
    unit, unit_from = scaled(write_unit, "shard" if sharded else "chunk")
    deepest, _ = pyramid_grid()
    return lcm_grid(unit, deepest), (f"LCM of the {unit_from} and the pyramid factor "
                                     f"{deepest}")


def _box_line(label: str, lo, hi) -> str:
    """``label  z a:b  y a:b  x a:b   (extent) = N voxels``.

    The voxel count is the number that says whether aligning outward matters: growing a
    box by one block per axis is a small-looking change to three numbers and a large one
    to the product.
    """
    extent = tuple(b - a for a, b in zip(lo, hi))
    return (f"  {label:<12}" + "  ".join(f"{ax} {a}:{b}"
                                        for ax, a, b in zip("zyx", lo, hi))
            + f"   {extent} = {math.prod(extent):,} voxels")


def cmd_align_bbox(args) -> int:
    """Move a box onto a block grid, so the run that uses it does not straddle one.

    Read-only and instant: it reads a volume's metadata for the grid and nothing else.
    ``--quiet`` prints only the aligned box, which is what makes it composable —
    ``--crop-bbox $(em-vol align-bbox ... -q)``.
    """
    from em_volume_tools.grid import align_box, clamp_box, misaligned_axes

    if not (args.volume or args.block):
        raise SystemExit("give --volume (to take the grid from it) or --block z,y,x")
    boxes = [_sextuple(b, "bbox") for b in args.bbox]
    d = _describe(args.volume) if args.volume else None
    block, source = _align_grid(args, d)

    extent = None
    if d:
        extent = tuple(d["shape"][1:] if d["has_channels"] else d["shape"])
    factor = (_level0_factor(args.volume, args.scale) if args.scale else None)

    out = sys.stdout if args.quiet else sys.stderr
    if not args.quiet:
        print(f"{args.volume or '(no volume)'}", file=out)
        print(f"  grid        {'x'.join(str(b) for b in block)}  ({source})", file=out)
        if extent:
            print(f"  extent      {extent} (zyx, level-0 voxels)", file=out)
        if factor:
            print(f"  input scale {args.scale}: x{factor} to level-0 voxels", file=out)

    for lo, hi in boxes:
        if factor:
            lo = tuple(a * f for a, f in zip(lo, factor))
            hi = tuple(b * f for b, f in zip(hi, factor))
        try:
            a_lo, a_hi = align_box(lo, hi, block, args.mode)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        clamped = False
        if extent:
            c_lo, c_hi = clamp_box(a_lo, a_hi, extent)
            clamped = (c_lo, c_hi) != (a_lo, a_hi)
            a_lo, a_hi = c_lo, c_hi
            if any(b <= a for a, b in zip(a_lo, a_hi)):
                raise SystemExit(f"the aligned box {a_lo}:{a_hi} does not intersect the "
                                 f"volume (extent {extent})")
        pasteable = ",".join(str(v) for v in (*a_lo, *a_hi))
        if args.quiet:
            print(pasteable)
            continue

        print("\n" + _box_line("requested", lo, hi), file=out)
        print(_box_line("aligned", a_lo, a_hi), file=out)
        grew = tuple((b - a) - (d_ - c) for a, b, c, d_ in zip(a_lo, a_hi, lo, hi))
        moved = tuple(a - c for a, c in zip(a_lo, lo))
        print(f"  change      mode {args.mode}; "
              + ("already aligned" if not any(grew) and not any(moved) else
                 f"extent {'+' if sum(grew) >= 0 else ''}{grew}, origin moved {moved}"),
              file=out)
        if clamped:
            # Clamping can put the far edge back off the grid, and that edge is fine:
            # the volume's own final block is partial there.
            print(f"  clamped     to the volume's extent — an edge landing on it is "
                  f"aligned by definition, since that block is partial in the volume "
                  f"too", file=out)
        left = misaligned_axes(a_lo, a_hi, block, extent=extent)
        if left:
            print(f"  NOTE        axis/axes {left} still off the grid"
                  + (" — mode 'origin' preserves the extent, so only the origin is "
                     "aligned" if args.mode == "origin" else ""), file=out)
        if args.scale:
            back = tuple(v / f for v, f in zip((*a_lo, *a_hi), (*factor, *factor)))
            exact = all(abs(v - round(v)) < 1e-9 for v in back)
            print(f"  at scale {args.scale}   "
                  + (",".join(str(int(round(v))) for v in back) if exact else
                     "not representable in whole scale-"
                     f"{args.scale} voxels (the grid is finer than that scale)"),
                  file=out)
        print(f"  --crop-bbox {pasteable}", file=out)
    return 0


# --------------------------------------------------------------------------- #
# annotate-json
# --------------------------------------------------------------------------- #
#: ``--points FILE`` / ``--point z,y,x`` per kind. One flag per kind rather than
#: ``--csv`` plus ``--kind``, so a file's kind is visible in the command line and one
#: layer can mix kinds — which local annotations allow and a precomputed source does not.
_ANN_FLAGS = (("points", "point"), ("boxes", "box"), ("lines", "line"),
              ("ellipsoids", "ellipsoid"))

#: The same column spec as ``ops.annotate.CSV_COLUMNS``, repeated here because the
#: parser needs it at build time and importing ``ops`` at module scope would pull
#: tensorstore into every `em-vol --help` (see test_cli_contract). A test asserts the
#: two agree, which is the only thing keeping the duplication honest.
_ANN_CSV_COLUMNS = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}


def _read_ann_file(path: str) -> str:
    """CSV text from a path, an object-store URL, or ``-`` for stdin."""
    if path == "-":
        return sys.stdin.read()
    from em_volume_tools.location import read_bytes

    data = read_bytes(path)
    if data is None:
        raise SystemExit(f"could not read {path}")
    return data.decode()


def _inline_records(args) -> list[dict]:
    """Records from the repeatable inline flags, in the order they were given."""
    records = []
    for _plural, kind in _ANN_FLAGS:
        for value in getattr(args, kind) or []:
            n = 3 * len(_ANN_CSV_COLUMNS[kind])
            parts = [p for p in value.replace(" ", "").split(",") if p]
            if len(parts) != n:
                raise SystemExit(
                    f"--{kind} needs {n} comma-separated numbers "
                    f"({', '.join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)}), "
                    f"got {value!r}")
            try:
                flat = [float(p) for p in parts]
            except ValueError:
                raise SystemExit(f"--{kind} {value!r}: not all numbers") from None
            records.append({"kind": kind,
                            "coords": tuple(tuple(flat[i:i + 3])
                                            for i in range(0, n, 3))})
    return records


def _ann_frame(args):
    """``(voxel_size_zyx, units, level-0 spatial shape or None)`` for the layer's frame.

    The frame is what makes the coordinates mean anything: an annotation layer declares
    its own ``outputDimensions``, and one that disagrees with the volume puts every
    annotation in the wrong place while still loading.
    """
    voxel = _ftriple(args.voxel_size, "voxel-size")
    if not args.volume:
        return voxel, ("nm" if voxel else None), None
    d = _describe(args.volume)
    meta = d["meta"] or {}
    shape = tuple(d["shape"][1:] if d["has_channels"] else d["shape"])
    return (voxel or (meta.get("voxel_size") and tuple(meta["voxel_size"])),
            "nm" if voxel else meta.get("units"), shape)


def cmd_annotate_json(args) -> int:
    """A neuroglancer annotation layer built from coordinates you supply.

    The JSON goes to **stdout** and the summary to **stderr**, as with ``bboxes-json``,
    so ``em-vol annotate-json ... > layer.json`` works. Pass the result to
    ``em-vol ng-url-gen --layer`` for a link.

    Reads nothing but the volume's metadata — and only to get the frame right.
    """
    from em_volume_tools.ops.annotate import (build_annotation, local_layer,
                                              output_dimensions,
                                              read_annotation_csv, render, rescale)

    records = []
    for plural, kind in _ANN_FLAGS:
        for path in getattr(args, plural) or []:
            try:
                records += read_annotation_csv(_read_ann_file(path), kind, source=path)
            except ValueError as e:
                raise SystemExit(str(e)) from None
    records += _inline_records(args)
    if not records:
        raise SystemExit(
            "nothing to annotate: give at least one of --points/--boxes/--lines/"
            "--ellipsoids (a CSV path, a URL, or - for stdin) or an inline "
            "--point/--box/--line/--ellipsoid")

    voxel, units, shape = _ann_frame(args)
    err = sys.stderr
    print(f"{args.volume or '(no volume: frame from --voxel-size)'}", file=err)
    if voxel:
        print(f"  frame       {'x'.join(f'{v:g}' for v in voxel)} {units or '?'} "
              f"per level-0 voxel", file=err)

    # Coordinates land in the layer as level-0 voxels, because that is the frame
    # `outputDimensions` states. Both other input units are one per-axis scaling.
    if args.scale:
        if not args.volume:
            raise SystemExit("--scale needs --volume: the conversion uses the volume's "
                             "own per-level voxel sizes, which cannot be guessed")
        factor = _level0_factor(args.volume, args.scale)
        records = rescale(records, factor)
        print(f"  scale {args.scale}     x{factor} (zyx) -> level-0 voxels", file=err)
    elif args.nm:
        if not voxel:
            raise SystemExit("--nm needs the level-0 voxel size: pass --voxel-size or "
                             "a --volume that records one")
        records = rescale(records, tuple(1.0 / v for v in voxel))
        print(f"  nm          /{tuple(voxel)} (zyx) -> level-0 voxels", file=err)

    ids, annotations = set(), []
    for i, r in enumerate(records):
        ident = str(r.get("id") or f"{args.label}{i:03d}")
        if ident in ids:
            raise SystemExit(f"duplicate annotation id {ident!r}: neuroglancer keys its "
                             f"annotations by id, so duplicates collide. Fix the `id` "
                             f"column, or drop it and let them be numbered.")
        ids.add(ident)
        annotations.append(build_annotation(r, ident))

    dims, warning = output_dimensions(voxel, units)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print(f"  annotations {len(annotations)}: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), file=err)
    n_seg = sum(1 for r in records if r.get("segments"))
    if n_seg:
        print(f"  segments    {n_seg} annotation(s) linked to body ids", file=err)

    # A wrong unit is the failure mode here, and it does not look like one: coordinates
    # 8x off are still valid annotations, just somewhere else. The volume's extent is
    # the only check available, so make it loudly.
    if shape:
        from em_volume_tools.ops.annotate import positions

        outside = [i for i, r in enumerate(records)
                   if any(not (0 <= c <= shape[a])
                          for g in positions(r) for a, c in enumerate(g))]
        where = f"the level-0 extent {shape} (zyx)"
        if outside:
            print(f"\n  WARNING: {len(outside)} annotation(s) fall outside {where} — "
                  f"first at index {outside[0]}. Check whether the coordinates are in "
                  f"another scale's voxels (--scale N) or in nm (--nm).", file=err)
        else:
            print(f"  bounds      all inside {where}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = local_layer(annotations, dims, name=args.name, color=args.color)
    text = render(layer)
    if args.out:
        _write_text(args.out, text + "\n")
        print(f"\nwrote {args.out} — paste it into the `layers` array of a neuroglancer "
              f"state, or pass it to `em-vol ng-url-gen --layer`", file=err)
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# relabel
# --------------------------------------------------------------------------- #
def cmd_relabel(args) -> int:
    """Give each occupied region of a sparse volume its own range of label ids.

    Runs in this process, no dask: the regions are renumbered in order because each
    range starts where the last ended, so there is nothing to parallelise.
    """
    from em_volume_tools.ops.relabel import (apply_relabel, default_map_path,
                                             plan_relabel)

    try:
        plan = plan_relabel(args.volume, out=args.out, in_place=args.in_place,
                            level=args.level, block_size=args.block_size)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None

    where = ("in place" if plan["in_place"] else f"-> {plan['destination']}")
    print(f"{plan['volume']}  {where}")
    print(f"  level {plan['level']}, dtype {plan['dtype']}, chunk {plan['chunk']}: "
          f"{plan['n_chunks']} stored chunks in {len(plan['regions'])} region(s)")
    if plan["block_size"]:
        print(f"  region k numbered from {plan['block_size']}*k+1, so the region is "
              f"readable off the id")
    else:
        print(f"  numbered consecutively from 1, each region continuing the last")

    map_path = args.map or default_map_path(plan["destination"], plan["level"])
    try:
        result = apply_relabel(plan, dry_run=args.dry_run, overwrite=args.overwrite,
                               map_path=map_path)
    except (FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from None

    print(f"\n{'#':>3} {'chunks':>7} {'labels':>7} {'new ids':>17}  box zyx")
    for e in result["regions"]:
        rng = "-" if not e["new_id_range"] else \
            f"{e['new_id_range'][0]}-{e['new_id_range'][1]}"
        box = " ".join(f"{lo}-{hi}" for lo, hi in zip(e["lo_zyx"], e["hi_zyx"]))
        print(f"{e['index']:>3} {e['chunks']:>7} {e['n_labels']:>7} {rng:>17}  {box}")

    print(f"\n{result['n_labels_in']} label-instances, "
          f"{result['n_distinct_in']} distinct before, "
          f"{result['n_labels_out']} after")
    if result["collisions_resolved"]:
        # The whole point of the operation, so it is stated as a number rather than
        # left for the reader to infer from the two totals.
        print(f"  {result['collisions_resolved']} id(s) were shared by more than one "
              f"region and are now distinct")
    else:
        print(f"  no id was shared between regions — this volume did not need it")
    if result["stale_levels"]:
        print(f"\n  WARNING: level(s) {result['stale_levels']} still hold the OLD ids "
              f"and now disagree with level {plan['level']}.\n"
              f"      em-vol downsample {plan['destination']} "
              f"--start-level {plan['level']}")

    if args.dry_run:
        print(f"\nDRY RUN: nothing written. The regions were read — the mapping is only "
              f"knowable from the voxels — so a real run repeats those reads.")
    else:
        print(f"\nwrote the old->new mapping to {result['map_path']}")
    return 0


# --------------------------------------------------------------------------- #
# ng-url-gen
# --------------------------------------------------------------------------- #
def cmd_ng_url_gen(args) -> int:
    """A neuroglancer URL carrying a whole viewer state.

    The URL goes to **stdout** and the summary to **stderr**, so it can be piped or
    assigned. Reads the volumes' metadata to get the source scheme and the coordinate
    space right, which is the part that fails silently when a state is written by hand.
    """
    from em_volume_tools.ops.ngurl import (DEFAULT_VIEWER, LAYOUTS, LONG_URL,
                                           VolumeProblem, annotation_extent,
                                           build_state, load_layer, state_url,
                                           volume_extent, volume_layer)

    err = sys.stderr
    layers, frame, extents = [], None, []
    try:
        # --image before --seg so the segmentation draws over the image, which is the
        # order anyone wants and the opposite of alphabetical.
        for volume in args.image or []:
            layer, info = volume_layer(volume, kind="image", opacity=args.image_opacity)
            layers.append(layer)
            frame = frame or info
            extents.append(volume_extent(volume, info["format"]))
        for i, volume in enumerate(args.seg or []):
            # --segments applies to the seg layers in order, so one list for one
            # segmentation is the common case and several stay unambiguous.
            picked = args.segments[i] if i < len(args.segments or []) else None
            layer, info = volume_layer(
                volume, kind="segmentation", name=None,
                segments=[int(s) for s in picked.replace(",", " ").split()] if picked
                else None)
            layers.append(layer)
            frame = frame or info
            extents.append(volume_extent(volume, info["format"]))
        for path in args.layer or []:
            layers.extend(load_layer(path))
    except (VolumeProblem, ValueError, json.JSONDecodeError) as e:
        raise SystemExit(str(e)) from None

    if not layers:
        raise SystemExit("nothing to show: pass at least one of --image, --seg or --layer")

    voxel = _ftriple(args.voxel_size, "voxel-size")
    units = "nm" if voxel else (frame or {}).get("units")
    if voxel is None:
        voxel = (frame or {}).get("voxel_size")
    if voxel is None:
        # A state whose dimensions disagree with its layers loads and puts everything
        # in the wrong place, so this is worth refusing rather than guessing.
        raise SystemExit(
            "no voxel size available: --layer files carry their own frame but do not "
            "establish the viewer's, and no --image/--seg volume recorded one. Pass "
            "--voxel-size Z,Y,X (nm).")

    position = _ftriple(args.position, "position")
    if position is not None and args.position_order == "xyz":
        position = tuple(reversed(position))

    # The whole frame of the largest volume, so an unspecified view opens centred and
    # zoomed out rather than on the origin corner. Falling back to the annotations means
    # a --layer-only link still frames its boxes instead of the origin.
    known = [e for e in extents if e]
    fit = max(known, key=lambda e: max(e[0])) if known else annotation_extent(layers)

    state, warning = build_state(
        layers, voxel_size_zyx=voxel, units=units, position_zyx=position,
        layout=args.layout, cross_section_scale=args.cross_section_scale,
        projection_scale=args.projection_scale,
        selected=args.select or (layers[-1]["name"] if args.select_last else None),
        show_slices=False if args.hide_slices else None,
        frame=fit)
    url = state_url(state, args.viewer)

    print(f"{len(layers)} layer(s): "
          + ", ".join(f"{lyr['name']} ({lyr['type']})" for lyr in layers), file=err)
    print(f"  voxel size {tuple(voxel)} zyx, units {units or '?'}", file=err)
    if position is not None:
        print(f"  position {tuple(position)} zyx "
              f"(given as {args.position_order})", file=err)
    elif fit:
        print(f"  view centred on the {tuple(int(v) for v in fit[0])} zyx frame, "
              f"zoomed to fit it", file=err)
    else:
        print(f"  no volume or annotation established a frame — neuroglancer will open "
              f"at the origin, zoomed in. Pass --position to place the view.", file=err)
    print(f"  layout {args.layout}, viewer {args.viewer}", file=err)
    if warning:
        print(f"  WARNING: {warning}", file=err)
    if len(url) > LONG_URL:
        print(f"  note: the URL is {len(url):,} characters. Everything after '#!' is a "
              f"fragment and never reaches a server, but some mail and chat clients "
              f"wrap or truncate at less than this — send it as a file if it matters.",
              file=err)

    if args.state_out:
        _write_text(args.state_out, json.dumps(state, indent=1) + "\n")
        print(f"  wrote the state to {args.state_out}", file=err)
    if args.out:
        _write_text(args.out, url + "\n")
        print(f"  wrote the URL to {args.out}", file=err)
    else:
        print(url)
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
    # Imported here, not at module scope: it pulls in dask.distributed (~1 s), and the
    # read-only subcommands that never reach this line should not pay for it.
    from em_blockrun import start_dask

    return start_dask(args.workers, _configs(args), label="em-vol")


def build_parser() -> argparse.ArgumentParser:
    """The full ``em-vol`` parser, built but not run.

    Separate from :func:`_parse_args` so the documentation can render it: the CLI
    reference is generated from *this* object by ``sphinx-argparse``, which is what
    stops the published usage from drifting away from ``--help``.
    """
    p = argparse.ArgumentParser(
        prog="em-vol", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # --- info ---------------------------------------------------------------
    q = sub.add_parser("info", help="what a volume is, and which levels exist",
                       description="Report a volume's format, coordinate metadata "
                                   "and the levels present. Reads only.")
    q.add_argument("volume")
    q.add_argument("--provenance", action="store_true",
                   help="print the whole provenance.json record if the volume has one "
                        "(a one-line summary is shown without this). Written by "
                        "`convert`; it names the source a volume was made from — for a "
                        "DVID export, the resolved node, which is the only way to say "
                        "afterwards which proofreading snapshot this is")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_info)

    # --- convert / copy -----------------------------------------------------
    # One argument set, two defaulting policies: `convert` states the output it wants,
    # `copy` takes the source's. See _add_convert_args.
    q = sub.add_parser("convert", help="build a multiscale volume from a source",
                       description=cmd_convert.__doc__ or
                       "Convert a source volume into a multiscale zarr and/or "
                       "neuroglancer-precomputed volume, whole or cropped to a box. "
                       "Resumable.\n\nTo copy a volume as it is — same format, "
                       "chunking, voxel size and image/segmentation type — use "
                       "`em-vol copy`, which reads those from the source instead of "
                       "defaulting them.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src-format", default=None,
                   help="backend for --src (default: auto-detect; a dvid:// URL is "
                        "detected by its scheme). Use 'image_stack' for a directory or "
                        "glob of ordered 2D slices (PNG/TIFF) — that one is never "
                        "auto-detected, and needs --voxel-size since image files carry "
                        "no physical scale")
    _add_convert_args(q, source_defaults=False)
    q.set_defaults(func=cmd_convert)

    q = sub.add_parser(
        "copy", help="copy a volume, or a box out of it, as it is",
        description="Copy a volume — or a box out of it — keeping the source's own "
                    "format, chunking, voxel size and image/segmentation type.\n\n"
                    "It is `convert` with a different defaulting policy, and that is "
                    "the whole point: `convert` defaults to precomputed, 128^3 chunks "
                    "and --kind image, so copying a segmentation with it and forgetting "
                    "--kind segmentation averages label ids into ids that were never in "
                    "the data — silently, while the source's own metadata said "
                    "`segmentation` all along. Here every one of those comes from the "
                    "source, and anything it does not record is an error rather than a "
                    "guess. Pass any of them to override.\n\n"
                    "--crop-bbox copies one box instead of the whole volume, and the "
                    "output keeps the source's coordinate frame (its physical offset "
                    "shifts by the crop origin), so the two overlay in a viewer. "
                    "--mask-bbox is the complement: copy everything EXCEPT a box, which "
                    "is how you hold a region out of a copy. The two compose, and mask "
                    "coordinates are always the source's.\n\n"
                    "The pyramid is REBUILT from the copied level 0, not copied: the "
                    "source's coarse levels are never read. For a crop that is what you "
                    "want — a slice of the source's coarse level is not the reduction of "
                    "the crop — but a whole-volume copy pays to recompute what already "
                    "exists.\n\nResumable, like `convert`; re-run to continue.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_convert_args(q, source_defaults=True)
    q.add_argument("--dry-run", action="store_true",
                   help="report what would be copied — resolved parameters, the box, "
                        "the level shapes — and write nothing")
    q.set_defaults(func=cmd_copy)

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
    q.add_argument("--max-levels", type=int, default=None,
                   help="at most this many levels, COUNTING LEVEL 0. Defaults to as many "
                        "as the volume records, since that is a property of the pyramid "
                        "being repaired rather than a preference — pass it only to "
                        "EXTEND a pyramid. --dry-run compares the schedule against the "
                        "levels on disk and refuses if they disagree")
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
    q.add_argument("--sparse", action="store_true",
                   help="skip tasks whose input holds no stored chunk. On a sparse "
                        "volume — ground truth, an ROI export — that is nearly every "
                        "task, and it is EXACT rather than a guess: an all-fill chunk is "
                        "never stored, so a task whose input objects are all absent "
                        "would write nothing anyway. Each level costs one listing "
                        "instead of a read per task. Refuses if the level it seeds from "
                        "has no stored chunks at all, rather than writing nothing and "
                        "reporting success. Not usable on a SHARDED level, which hides "
                        "which of its chunks exist")
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
    q.add_argument("--max-levels", type=int, default=None,
                   help="at most this many levels, COUNTING LEVEL 0. Truncates a "
                        "mirrored --like pyramid too (each kept level stays verbatim). "
                        "Default: no cap when mirroring a reference, 8 when the pyramid "
                        "is computed")
    q.add_argument("--min-dim", type=int, default=128,
                   help="stop when the largest spatial dim is <= this. Applies only when "
                        "the pyramid is COMPUTED — a mirrored --like pyramid keeps every "
                        "level the reference has, since dropping its small levels by "
                        "default would silently break the shared frame")
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
    q.add_argument("--background", default=None, metavar="V[,V...]",
                   help="value(s) the source uses for background, replaced with 0 as it is "
                        "read. Manually segmented pieces numbered from 0 have background 1, "
                        "and an all-background block of 1s is not all-fill, so without this "
                        "it is stored as data")
    q.add_argument("--voxel-size-field", default="voxel_size", metavar="NAME",
                   help="what the source calls its recorded voxel size, if any. Used only "
                        "to CHECK it against the level being written into — a piece "
                        "extracted from another level fits and places cleanly while being "
                        "at the wrong resolution, which nothing else here would notice "
                        "(default: voxel_size)")
    q.add_argument("--offset-order", choices=("zyx", "xyz"), default=None,
                   help="axis order of the offset, whether typed or read from the "
                        "source. Default: whatever the source RECORDS (an `axes` "
                        "attribute, as `em-vol to-hdf5` writes), else zyx. Worth "
                        "checking on a stored offset from elsewhere: 'voxel_offset' is "
                        "precomputed's field name and precomputed means XYZ, while "
                        "everything in this package is zyx — reversed, the piece lands "
                        "mirrored through the z=x diagonal. Whichever applied is echoed")
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

    # --- bboxes-json --------------------------------------------------------
    q = sub.add_parser(
        "bboxes-json", help="a neuroglancer layer of boxes marking where the data is",
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
    q.add_argument("--out", default=None, metavar="PATH_OR_URL",
                   help="write the JSON here instead of to stdout. May be a local path "
                        "or an object store location (s3://...)")
    q.add_argument("--state", action="store_true",
                   help="emit a complete loadable viewer state (volume layer + "
                        "annotations) rather than just the layer to paste. For a state "
                        "with an image layer and a starting view, use `ng-url-gen`")
    q.add_argument("--level", type=int, default=0,
                   help="the level whose chunk objects define the footprint. Coarser "
                        "is a cheaper listing and a blockier box; coordinates are "
                        "reported in level-0 voxels either way (default: 0)")
    q.add_argument("--tighten-level", type=int, default=None, metavar="N",
                   help="shrink each box to its nonzero voxels by reading it at this "
                        "level, instead of leaving it on the chunk grid. DEFAULTS TO "
                        "--level, so the reads cost what the level you picked costs and "
                        "the boxes are exact in the level-0 voxels they are reported in. "
                        "Raise it on a volume whose occupied footprint is large: each "
                        "level is a factor smaller to read, at the price of quantizing "
                        "each bound to one voxel there")
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
    q.set_defaults(func=cmd_bboxes_json)

    # --- mask-by-value ------------------------------------------------------
    q = sub.add_parser(
        "mask-by-value", help="replace label values with background",
        description="Replace one or more label values with 0 in a volume that already "
                    "holds them.\n\n"
                    "Manual segmentation does not always call background 0 — a tool "
                    "numbering labels from 0 makes it 1 — and that one fact breaks more "
                    "than it looks. Background becomes a body when meshed, and, worse, an "
                    "all-background block of 1s is NOT all-fill, so it gets stored: the "
                    "volume ends up with a chunk object everywhere data was written, and "
                    "'which chunks exist' stops answering 'where is the data'. That is the "
                    "question `bboxes-json`, `relabel`, `downsample --sparse` and "
                    "em-seg-morpho's occupancy filter all ask.\n\n"
                    "PREFER FIXING IT AT INGEST: `em-vol write`, `to-hdf5` and `convert` "
                    "all take --background, and there the correction happens before the "
                    "storage decision. This command is for data that has already landed.\n\n"
                    "Either destination restores the sparsity — writing zeros over a "
                    "stored chunk removes the object, on both formats — so --out is "
                    "preferred for the ordinary reason instead: a sparse copy is cheap, "
                    "and the original stays as the record of what was annotated.\n\n"
                    "SINGLE-SCALE, like `write` and `relabel`: run `em-vol downsample "
                    "--start-level <level>` afterwards.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume to read (path or s3://...)")
    q.add_argument("--values", required=True, metavar="V[,V...]",
                   help="the value(s) to replace, e.g. 1")
    q.add_argument("--to", type=int, default=0,
                   help="what to replace them with (default: 0, the only value the "
                        "storage layer treats as empty)")
    dest = q.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", default=None, metavar="VOLUME",
                      help="write the corrected volume here instead, created with "
                           "`--like <volume>` so a voxel index means the same thing in "
                           "both. Preferred: cheap on sparse data, and the original stays "
                           "as the record of what was annotated")
    dest.add_argument("--in-place", action="store_true",
                      help="overwrite the values in the volume itself. Restores the "
                           "sparsity too, but the original values are then unrecoverable")
    q.add_argument("--level", type=int, default=0,
                   help="which level to correct (default: 0)")
    q.add_argument("--overwrite", action="store_true",
                   help="allow --out to replace an existing volume")
    q.add_argument("--dry-run", action="store_true",
                   help="report and count what would change, writing nothing. Still "
                        "reads, since the counts come from the voxels")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_mask_by_value)

    # --- to-hdf5 ------------------------------------------------------------
    from em_volume_tools.ops.pack import (DEFAULT_DATASET, DEFAULT_OFFSET_FIELD,
                                          DEFAULT_VOXEL_SIZE_FIELD)

    q = sub.add_parser(
        "to-hdf5", help="pack a small volume into one HDF5 file, with its frame",
        description="Pack an image stack (or any readable source) into a single HDF5 "
                    "file, recording where it belongs and at what scale.\n\n"
                    "The inverse of `em-vol write`: that places a piece into a large "
                    "volume, this produces the piece. An image stack off a microscope or "
                    "an annotation tool has no coordinates attached, so this attaches "
                    "them — and then `em-vol write <volume> --src piece.h5` needs no "
                    "--offset at all.\n\n"
                    "What it records: `voxel_offset` (whole voxels, on the dataset — the "
                    "field `write` already looks for), and `voxel_size` / `offset` / "
                    "`units` / `axes` in this package's own vocabulary, on the root and "
                    "the dataset both. `axes` is the one that earns its keep: the axis "
                    "order of a stored voxel_offset was previously unknowable from the "
                    "file — precomputed means xyz, this package means zyx, and reversed, "
                    "a piece lands mirrored through the z=x diagonal. A file written here "
                    "says which.\n\n"
                    f"The dataset defaults to {DEFAULT_DATASET}, which is also what the "
                    "reader assumes, so a file packed with no arguments reads with none. "
                    "An existing file is ADDED TO when its recorded frame matches — "
                    "several pieces of one volume in one file is a legitimate "
                    "arrangement, each with its own voxel_offset — and refused when it "
                    "does not. A name already in use needs --dataset or --overwrite.\n\n"
                    "Reads are blocked, so a volume larger than advertised streams rather "
                    "than filling memory. Serial and in-process, like `create`/`write`.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True,
                   help="image stack (directory or glob of ordered 2D slices), or any "
                        "readable volume")
    q.add_argument("--out", required=True, metavar="FILE.h5",
                   help="the HDF5 file to write. Created, or added to if it already "
                        "records the same frame")
    q.add_argument("--src-format", default=None,
                   help="backend for --src (default: auto-detect, falling back to "
                        "image_stack for a directory or glob of images)")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="physical size of one voxel, e.g. 8,8,8. Required for a source "
                        "that records none (an image stack, an HDF5 file, a bare array); "
                        "read from the volume, at --level, when the source is one")
    q.add_argument("--offset", default=None, metavar="Z,Y,X",
                   help="whole-voxel position of this piece's (0,0,0) corner in the "
                        "volume it belongs to, in --axes order. Default: what an HDF5 "
                        "--src already records (--offset-field, searched in the dataset's "
                        "attributes, the root's, and a top-level dataset of that name), "
                        "plus the --crop-bbox origin if one is given; else 0,0,0")
    q.add_argument("--level", type=int, default=0,
                   help="which level of a multiscale --src to read (default: 0). The "
                        "level's own recorded voxel size becomes the default frame, and "
                        "--crop-bbox is in that level's voxels, so `em-vol write --level "
                        "N` puts the piece back where it came from")
    q.add_argument("--crop-bbox", default=None, metavar="Z0,Y0,X0,Z1,Y1,X1",
                   help="pack only this box, in --level's voxels. Half-open, clipped to "
                        "the source. Its origin becomes the recorded voxel_offset unless "
                        "--offset says otherwise")
    q.add_argument("--src-dataset", default=None, metavar="NAME",
                   help="dataset to read, when --src is itself an HDF5 file with more "
                        "than one")
    q.add_argument("--voxel-size-field", default=DEFAULT_VOXEL_SIZE_FIELD, metavar="NAME",
                   help=f"attribute to record the voxel size under, and to READ it from "
                        f"when --src is an HDF5 file that carries one (default: "
                        f"{DEFAULT_VOXEL_SIZE_FIELD}, this package's own name). Set it to "
                        f"match files another tool wrote, so a file stays readable by "
                        f"whatever wrote its siblings")
    q.add_argument("--offset-field", default=DEFAULT_OFFSET_FIELD, metavar="NAME",
                   help=f"attribute to record the voxel offset under (default: "
                        f"{DEFAULT_OFFSET_FIELD}, which is what `em-vol write` looks for; "
                        f"change both together or write will not find it)")
    q.add_argument("--units", default="nm",
                   help="unit for --voxel-size (default: nm)")
    q.add_argument("--axes", default="zyx",
                   help="axis order of the array AND of --voxel-size/--offset, recorded "
                        "in the file so no reader has to guess (default: zyx)")
    q.add_argument("--dataset", default=None, metavar="NAME",
                   help=f"dataset to write (default: {DEFAULT_DATASET}, which is what a "
                        f"reader assumes when it is not told)")
    q.add_argument("--dtype", default=None,
                   help="cast to this dtype (default: the source's)")
    q.add_argument("--chunk", default=None, metavar="Z,Y,X",
                   help="HDF5 storage chunk (default: up to 64 per axis). This is what "
                        "governs partial reads when the piece is written back")
    q.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip",
                   help="dataset compression (default: gzip)")
    q.add_argument("--background", default=None, metavar="V[,V...]",
                   help="value(s) the source uses for background, replaced with 0 while "
                        "packing, so the packed file is already correct and nothing "
                        "downstream needs to know the source had this quirk")
    q.add_argument("--overwrite", action="store_true",
                   help="replace the dataset if that name is already used")
    q.add_argument("--dry-run", action="store_true",
                   help="report what would be written, and write nothing")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_to_hdf5)

    # --- align-bbox ---------------------------------------------------------
    from em_volume_tools.grid import MODES

    q = sub.add_parser(
        "align-bbox", help="move a box onto a volume's block grid",
        description="Align a bounding box to a block grid and print it back, ready to "
                    "paste into --crop-bbox or --roi.\n\n"
                    "WHICH GRID is the real question, and there are three, each with a "
                    "different cost when a box straddles it:\n\n"
                    "  write-unit  the chunk, or the SHARD where the level is sharded. "
                    "A partial write is a read-modify-write: it keeps the object's "
                    "existing data, but two concurrent partial writes into one object "
                    "lose one of them, silently. Aligning to the inner chunk of a "
                    "sharded level protects against nothing.\n"
                    "  pyramid     the cumulative factor of the deepest level. A crop "
                    "that misses it has coarse levels on their own grid, each level's "
                    "voxel_offset rounding to it — level 0 stays exact, so nothing "
                    "looks wrong.\n"
                    "  both        the per-axis LCM of those two. What a cropped copy "
                    "of a multiscale volume wants.\n"
                    "  read-chunk  the inner chunk. Read amplification, not write "
                    "safety.\n\n"
                    "Boxes are half-open, so a bound already on a boundary stays put. "
                    "Everything is per axis: real grids are anisotropic, and a level's "
                    "chunk is converted to level-0 voxels through its own recorded "
                    "voxel size, never an assumed 2**N.\n\n"
                    "Reads a volume's metadata and nothing else. Writes nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--bbox", action="append", required=True,
                   metavar="Z0,Y0,X0,Z1,Y1,X1",
                   help="the box to align, half-open, in voxels at --scale. Repeatable")
    q.add_argument("--volume", default=None, metavar="PATH_OR_URL",
                   help="take the grid and the extent from this volume. Required unless "
                        "--block gives the grid outright")
    q.add_argument("--block", default=None, metavar="Z,Y,X",
                   help="align to this grid instead of one read from a volume, in "
                        "level-0 voxels. Works with no --volume at all")
    q.add_argument("--to", choices=("write-unit", "pyramid", "both", "read-chunk"),
                   default="write-unit",
                   help="which of the volume's grids to align to (default: write-unit). "
                        "See above — they answer different questions")
    q.add_argument("--level", type=int, default=0,
                   help="take the chunk/shard from this level rather than 0. Its blocks "
                        "are converted to level-0 voxels, so a 128-chunk at a level "
                        "coarsened 4x is a 512-voxel grid (default: 0)")
    q.add_argument("--mode", choices=MODES, default="outer",
                   help="outer: grow to cover the box (never loses a voxel, cannot "
                        "fail). inner: shrink to fit inside it. nearest: round both "
                        "ends. origin: align the origin and keep the EXTENT exactly, "
                        "for a fixed-size crop — its far edge then stays off the grid "
                        "(default: outer)")
    q.add_argument("--scale", type=int, default=0, metavar="N",
                   help="the box is in scale-N voxels; convert to level 0 with the "
                        "volume's real per-level voxel sizes. The aligned box is also "
                        "reported back at scale N when it is exactly representable there")
    q.add_argument("-q", "--quiet", action="store_true",
                   help="print only the aligned box, one line per --bbox, so it can be "
                        "substituted: --crop-bbox $(em-vol align-bbox ... -q)")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_align_bbox)

    # --- annotate-json ------------------------------------------------------
    q = sub.add_parser(
        "annotate-json", help="a neuroglancer annotation layer from coordinates you give",
        description="Emit a neuroglancer annotation layer from coordinates you supply — "
                    "points, boxes, lines, ellipsoids — as CSV files or inline flags. "
                    "The counterpart to `bboxes-json`, which derives its boxes from "
                    "where a volume's data is; this one annotates what you already "
                    "know, such as a synapse table.\n\n"
                    "CSV columns are addressed BY NAME, so a table with its own column "
                    "order and extra columns works untouched: z,y,x for points, "
                    "z0,y0,x0,z1,y1,x1 for boxes and lines, z,y,x,rz,ry,rx for "
                    "ellipsoids, plus optional id, description and segments (whitespace-"
                    " or comma-separated body ids, which make the annotation select "
                    "those bodies when you click it).\n\n"
                    "Coordinates are LEVEL-0 VOXELS unless you say otherwise: --scale N "
                    "converts from scale-N voxels using the volume's own per-level "
                    "voxel sizes, --nm from physical nanometres. Getting this wrong is "
                    "the failure mode of the whole command and does not look like an "
                    "error — coordinates 8x off are still valid annotations, just "
                    "somewhere else — so annotations landing outside the volume are "
                    "reported.\n\n"
                    "The annotations are LOCAL (inline in the state), which is what "
                    "makes them appear in the Annotations tab and step with [ and ]. "
                    "That also bounds how many are practical: a state carries them all. "
                    "For a whole volume's worth of synapses the answer is the "
                    "precomputed annotation format, which is not this command.\n\n"
                    "JSON to stdout, summary to stderr, so `> layer.json` works.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for plural, kind in _ANN_FLAGS:
        cols = ",".join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)
        q.add_argument(f"--{plural}", action="append", default=None,
                       metavar="CSV",
                       help=f"CSV of {plural} (columns {cols}). A path, an s3:// URL, "
                            f"or - for stdin. Repeatable")
        q.add_argument(f"--{kind}", action="append", default=None,
                       metavar=cols.upper(),
                       help=f"one {kind} inline, as {cols}. Repeatable")
    q.add_argument("--volume", default=None, metavar="PATH_OR_URL",
                   help="the volume these annotate. Read for its voxel size, units and "
                        "extent only — nothing else, and never its voxels. Without it, "
                        "pass --voxel-size, or the layer is unitless and will not line up")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding the volume's own")
    units = q.add_mutually_exclusive_group()
    units.add_argument("--scale", type=int, default=0, metavar="N",
                       help="the coordinates are in scale-N voxels; convert them to "
                            "level 0 using the volume's real per-level voxel sizes "
                            "(never an assumed 2**N). Needs --volume")
    units.add_argument("--nm", action="store_true",
                       help="the coordinates are physical nanometres; divide by the "
                            "level-0 voxel size")
    q.add_argument("--name", default="annotations", help="layer name")
    q.add_argument("--label", default="a", metavar="PREFIX",
                   help="prefix for generated annotation ids, numbered from 000. Only "
                        "used for rows with no `id` column (default: a)")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--out", default=None, metavar="PATH_OR_URL",
                   help="write the JSON here instead of to stdout (local or s3://...)")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_annotate_json)

    # --- relabel ------------------------------------------------------------
    q = sub.add_parser(
        "relabel", help="give each occupied region its own range of label ids",
        description="Renumber a SPARSE segmentation volume so each occupied region gets "
                    "a disjoint range of ids.\n\n"
                    "Ground truth annotated chunk by chunk usually numbers every chunk "
                    "from 1, so one integer means a different cell in each — meshed, "
                    "that becomes a single body with components scattered across the "
                    "volume. This walks the regions in order and gives each its own "
                    "range, so an id identifies one cell in one region.\n\n"
                    "Regions come from stored-chunk occupancy (as `em-vol "
                    "bboxes-json`), so they are disjoint and chunk-aligned: no write is "
                    "a partial-chunk update. Serial by construction — each range starts "
                    "where the last ended — so it runs in this process, no dask.\n\n"
                    "SINGLE-SCALE, like `em-vol write`: run `em-vol downsample "
                    "--start-level <level>` afterwards, then re-mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume to read (path or s3://...)")
    dest = q.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", default=None, metavar="VOLUME",
                      help="write the renumbered volume here instead, created empty "
                           "with `--like <volume>` so a voxel index means the same "
                           "thing in both. Preferred: a sparse copy is nearly free and "
                           "the original stays as the record of the raw annotation")
    dest.add_argument("--in-place", action="store_true",
                      help="overwrite the ids in the volume itself. Recoverable only "
                           "through the mapping file")
    q.add_argument("--level", type=int, default=0,
                   help="which level to renumber (default: 0)")
    q.add_argument("--block-size", type=int, default=None, metavar="N",
                   help="number region k from N*k+1 instead of consecutively, so the "
                        "region a label came from is readable off the id. Refuses if a "
                        "region holds more than N labels")
    q.add_argument("--map", default=None, metavar="PATH_OR_URL",
                   help="where to write the old->new mapping — a local path or an object "
                        "store location (s3://...), so it can sit beside a remote volume "
                        "(default: '<destination>.relabel-<level>.json' here). It is the "
                        "only way back from a new id to the region and label it came from")
    q.add_argument("--overwrite", action="store_true",
                   help="allow --out to replace an existing volume")
    q.add_argument("--dry-run", action="store_true",
                   help="report the regions and the ranges they would get; write "
                        "nothing. Still reads them, since the ids come from the voxels")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_relabel)

    # --- ng-url-gen ---------------------------------------------------------
    from em_volume_tools.ops.ngurl import DEFAULT_VIEWER, LAYOUTS

    q = sub.add_parser(
        "ng-url-gen", help="a neuroglancer URL carrying a whole viewer state",
        description="Build a neuroglancer link from volumes and layer files.\n\n"
                    "Neuroglancer keeps its whole state in the URL fragment, so a link "
                    "IS the state — which volumes are loaded, where the view sits, which "
                    "segments are selected. This reads the volumes to get the source "
                    "scheme and the coordinate space right, which is the part that fails "
                    "silently by hand: a `dimensions` block that disagrees with the data "
                    "loads fine and puts every layer in the wrong place.\n\n"
                    "Everything after '#!' is a fragment and never reaches a server, so "
                    "a link carries no data anywhere — but the whole state travels in it, "
                    "and a large inline annotation layer makes for a long URL.\n\n"
                    "Composes with `bboxes-json`: that writes a layer, --layer inlines "
                    "it here.\n\n"
                    "  em-vol bboxes-json s3://.../gt_v2 --label gt --out gt.json\n"
                    "  em-vol ng-url-gen --image s3://.../em --seg s3://.../gt_v2 \\\n"
                    "      --layer gt.json --segments 1,2,3 --layout xy-3d\n\n"
                    "URL to stdout, summary to stderr. Reads only.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--image", action="append", metavar="VOLUME",
                   help="an image volume (repeatable). Drawn beneath the segmentations")
    q.add_argument("--seg", action="append", metavar="VOLUME",
                   help="a segmentation volume (repeatable)")
    q.add_argument("--segments", action="append", metavar="IDS",
                   help="comma-separated segment ids to select, applied to the --seg "
                        "volumes in order. Repeat for a second segmentation")
    q.add_argument("--layer", action="append", metavar="PATH_OR_URL",
                   help="a JSON file holding a layer, or a state whose layers are taken "
                        "— e.g. the output of `em-vol bboxes-json` (repeatable)")
    q.add_argument("--position", default=None, metavar="Z,Y,X",
                   help="where to put the crosshair, in level-0 voxels")
    q.add_argument("--position-order", choices=("zyx", "xyz"), default="zyx",
                   help="axis order of --position. Everything in this package is zyx, "
                        "but neuroglancer DISPLAYS xyz — so pass --position-order xyz to "
                        "use numbers copied straight out of the viewer (default: zyx)")
    q.add_argument("--layout", default="4panel", choices=LAYOUTS,
                   help="neuroglancer panel layout (default: 4panel)")
    q.add_argument("--hide-slices", action="store_true",
                   help="set showSlices false, hiding the cross-section planes inside the "
                        "3D panel — the usual thing to want when the link is about meshes "
                        "or skeletons. The 2D panels are unaffected; use --layout 3d for "
                        "those. Omitted from the state unless passed, so a link without it "
                        "opens the way the viewer normally would")
    q.add_argument("--cross-section-scale", type=float, default=None, metavar="S",
                   help="zoom of the 2D panels: nm per screen pixel, smaller is closer")
    q.add_argument("--projection-scale", type=float, default=None, metavar="S",
                   help="zoom of the 3D panel")
    q.add_argument("--image-opacity", type=float, default=None, metavar="F",
                   help="opacity for the --image layers")
    q.add_argument("--select", default=None, metavar="LAYER_NAME",
                   help="open the side panel on this layer, i.e. set the state's "
                        "`selectedLayer` to {visible: true, layer: LAYER_NAME}. The name "
                        "is the layer's, which for a volume is the last component of its "
                        "path unless --name gave it another")
    q.add_argument("--select-last", action="store_true",
                   help="the same, on whichever layer was added last — with a bboxes "
                        "layer last, that is its clickable list of regions")
    q.add_argument("--viewer", default=DEFAULT_VIEWER,
                   help=f"viewer base URL (default: {DEFAULT_VIEWER})")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding what the volumes record. "
                        "Required when every layer comes from --layer, since a layer "
                        "file carries its own frame but does not establish the viewer's")
    q.add_argument("--out", default=None, metavar="PATH_OR_URL",
                   help="write the URL here instead of stdout (local or s3://...)")
    q.add_argument("--state-out", default=None, metavar="PATH_OR_URL",
                   help="also write the state as JSON, for pasting into the {} editor")
    q.add_argument("--store-logs", action="store_true")
    q.set_defaults(func=cmd_ng_url_gen)

    return p


def _parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    # Image files carry no physical scale, and the op would otherwise fail deep inside
    # the conversion rather than here. Everything else can read it from the source.
    if (getattr(args, "src_format", None) == "image_stack"
            and not getattr(args, "voxel_size", None)):
        p.error("--src-format image_stack requires --voxel-size: image files record "
                "no physical scale, so it cannot be read from the source "
                "(e.g. --voxel-size 8,8,8 for 8 nm isotropic)")
    # A bbox convention with no box to apply it to is the sign of a half-finished command
    # line; saying so beats copying the whole volume when a crop or a mask was meant.
    boxes = bool(getattr(args, "crop_bbox", None) or getattr(args, "mask_bbox", None))
    if getattr(args, "bbox_scale", 0) and not boxes:
        p.error("--bbox-scale has no effect without --crop-bbox or --mask-bbox")
    if getattr(args, "bbox_order", "zyx") != "zyx" and not boxes:
        p.error("--bbox-order has no effect without --crop-bbox or --mask-bbox")
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
