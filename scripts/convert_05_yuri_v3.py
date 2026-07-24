"""Convert sample3/05_yuri_v3 (neuroglancer-precomputed image) -> multiscale zarr v3.

Source : /path/to/data/jwu/sample3/05_yuri_v3   (uint8, ~11260x9000x13750, 8nm)
Dest   : /path/to/data/eschomburg/sample3/05_yuri_v3.zarr  (512^3 chunks)

Reads the finest precomputed scale and builds a new mean-downsampled pyramid.
voxel_size/offset are read from the source `info` automatically.

Usage
-----
  # 1) local subvolume smoke test (centered box) + integrity check vs source:
  pixi run -e dev python scripts/convert_05_yuri_v3.py test --workers 4 --subvol 1024

  # 2) full run on Rusty/SLURM (edit configs/dask-slurm-gen.yaml: account, etc.),
  #    launched from a workstation so it survives logout:
  nohup pixi run -e dev python -u scripts/convert_05_yuri_v3.py full \
      --config configs/dask-slurm-gen.yaml --workers 48 > yuri_v3.log 2>&1 &
  squeue -u "$USER"          # monitor jobs (read-only)

  # progress (written chunks per level), from any terminal:
  pixi run -e dev python scripts/zarr_progress.py
  # ...and the dask dashboard URL is printed in the log at startup.

Resume: the full run is resume-safe (resume=True) -- if it dies (walltime, node
failure), just relaunch the same command and it skips already-written blocks.
Use --fresh to wipe and restart from scratch instead.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os

import numpy as np

from em_volume_tools import StorageProfile, convert, extract_roi, start_dask
from em_volume_tools.backends.base import open_backend
from em_volume_tools.introspect import read_source_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("convert_05_yuri_v3")

SRC = "/path/to/data/jwu/sample3/05_yuri_v3"
DST = "/path/to/data/eschomburg/sample3/05_yuri_v3.zarr"
CHUNK = (512, 512, 512)      # zarr read/write chunk (unsharded)
KIND = "image"               # uint8 image -> mean downsample

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_spec() -> dict:
    # scale_index 0 = finest scale (8nm).
    return {"backend": "neuroglancer_precomputed", "path": SRC, "scale_index": 0}


def out_profile() -> StorageProfile:
    return StorageProfile("zarr3", chunk=CHUNK, shard=None, compressor="zstd")


def middle_box(shape, side):
    """Centered (start, stop) cube of edge `side`, clipped to the volume."""
    start = tuple(max(0, s // 2 - side // 2) for s in shape)
    stop = tuple(min(sh, st + side) for st, sh in zip(start, shape))
    return start, stop


@contextlib.contextmanager
def maybe_cluster(workers, config):
    """Yield a dask client (SLURM/local) or None for in-process serial execution."""
    if workers and workers > 0:
        with start_dask(workers, config_path=config, label="yuri-v3") as client:
            yield client
    else:
        yield None


def run_test(args):
    src_be = open_backend(source_spec())
    shape = src_be.shape
    meta = read_source_metadata(source_spec())
    start, stop = middle_box(shape, args.subvol)
    test_dst = DST.replace(".zarr", f".test-{args.subvol}.zarr")
    os.makedirs(os.path.dirname(test_dst), exist_ok=True)
    log.info("source shape=%s dtype=%s voxel_size=%s", shape, src_be.dtype, meta["voxel_size"])
    log.info("subvolume start=%s stop=%s -> %s", start, stop, test_dst)

    with maybe_cluster(args.workers, args.config) as client:
        summary = extract_roi(
            source_spec(), test_dst, start=start, stop=stop,
            voxel_size=meta["voxel_size"], units=meta["units"], kind=KIND,
            profile=out_profile(), chunk=CHUNK, multiscale=True, min_dim=args.min_dim,
            client=client, npartitions=args.npartitions, delete_existing=True,
        )
    log.info("wrote %d levels: shapes=%s", summary["num_levels"], summary["level_shapes"])

    # integrity check: level-0 of the output must equal the source subvolume
    out0 = open_backend({"backend": "zarr3", "path": os.path.join(test_dst, "0")})
    got = out0.read_region(tuple(slice(0, s) for s in out0.shape))
    want = src_be.read_region(tuple(slice(a, b) for a, b in zip(start, stop)))
    if got.shape == want.shape and np.array_equal(got, want):
        log.info("VERIFIED: output level-0 matches source subvolume (%s voxels)", got.size)
    else:
        raise SystemExit(f"MISMATCH: got {got.shape} vs want {want.shape}")


def run_full(args):
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    src_be = open_backend(source_spec())
    log.info("FULL conversion: source shape=%s dtype=%s -> %s (chunk=%s)",
             src_be.shape, src_be.dtype, DST, CHUNK)
    with maybe_cluster(args.workers, args.config) as client:
        summary = convert(
            source_spec(), DST, profile=out_profile(), kind=KIND, chunk=CHUNK,
            multiscale=True, min_dim=args.min_dim, client=client,
            npartitions=args.npartitions,
            resume=not args.fresh,          # relaunch continues; skips written blocks
            delete_existing=args.fresh,     # --fresh wipes and restarts
            validate=False,  # schema already proven in tests; avoids driver-side net dependency
        )
    log.info("done: %d levels, shapes=%s", summary["num_levels"], summary["level_shapes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test", "full"])
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "dask-local.yaml"),
                    help="dask YAML config (use configs/dask-slurm-gen.yaml for SLURM)")
    ap.add_argument("--workers", type=int, default=4,
                    help="dask workers; 0 = run serially in-process (no cluster)")
    ap.add_argument("--subvol", type=int, default=1024, help="test-mode cube edge (voxels)")
    ap.add_argument("--min-dim", type=int, default=128,
                    help="stop adding pyramid levels once max spatial dim <= this")
    ap.add_argument("--npartitions", type=int, default=None,
                    help="dask partitions per block-map (default: one per block)")
    ap.add_argument("--fresh", action="store_true",
                    help="full mode: wipe and restart instead of resuming an interrupted run")
    args = ap.parse_args()
    (run_test if args.mode == "test" else run_full)(args)


if __name__ == "__main__":
    main()
