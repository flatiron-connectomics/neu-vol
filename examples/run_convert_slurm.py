"""Example driver: convert a volume to multiscale zarr/precomputed on Rusty/SLURM.

Run this ON A WORKSTATION, in a session that outlives your terminal. It starts a
dask cluster whose workers are SLURM jobs (dask submits the sbatch itself via
`scale()` — see docs/dask-slurm-rusty.md), then runs the conversion with every
block dispatched to those workers.

    # smoke test first (local, no SLURM):
    python examples/run_convert_slurm.py --config configs/dask-local.yaml --workers 2

    # full run on SLURM, surviving logout:
    nohup python -u examples/run_convert_slurm.py \
        --config configs/dask-slurm-gen.yaml --workers 48 > run.log 2>&1 &
    squeue -u "$USER"        # watch your jobs (read-only; don't poll in tight loops)

Edit SRC/DST/VOXEL_SIZE/etc. below (or wire up argparse) for your data. Because
tasks are idempotent, re-running the same command resumes an interrupted run.
"""

from __future__ import annotations

import argparse
import logging

from em_volume_tools import convert, start_dask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- edit these for your dataset -------------------------------------------
SRC = "/path/to/scratch/data/in.zarr"       # or a spec dict / precomputed / hdf5
DST = "/path/to/scratch/data/out.precomputed"
VOXEL_SIZE = (8, 8, 8)          # (z, y, x) nm
PROFILE = "s3-neuroglancer"     # "ceph" for sharded zarr intermediates, etc.
KIND = "image"                  # "image"/"probability" (mean) | "segmentation" (mode)
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dask-slurm-gen.yaml")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    # Keep num_workers so jobs x cores <= your QOS CPU cap (see cookbook Gotcha #4).
    with start_dask(args.workers, config_path=args.config, label="em-convert") as client:
        summary = convert(SRC, DST, voxel_size=VOXEL_SIZE, profile=PROFILE, kind=KIND,
                          client=client, delete_existing=False)
    logging.info("done: %s levels, shapes=%s", summary["num_levels"], summary["level_shapes"])


if __name__ == "__main__":
    main()
