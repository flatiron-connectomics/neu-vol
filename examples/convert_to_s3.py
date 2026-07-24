"""Example: write a segmentation to S3 as multiscale precomputed and/or zarr.

TensorStore's `s3` kvstore does the transport; pass the destination as an
`s3://bucket/prefix` URL (or a kvstore dict for region/endpoint control). Resume
is manifest-based and works for both formats.

Credentials (this example uses environment variables):
  export AWS_ACCESS_KEY_ID=...        # in the DRIVER's environment
  export AWS_SECRET_ACCESS_KEY=...
  # (optional) export AWS_SESSION_TOKEN=... ; AWS_DEFAULT_REGION=us-east-1

  On SLURM, workers need these too. Slurm's default `--export=ALL` propagates the
  submitting environment to the job, so launching the driver with the vars set is
  usually enough. If your site disables that, either use a shared ~/.aws
  credentials file (readable on the worker nodes) or add the exports to the dask
  job_script_prologue. Do NOT bake secrets into a committed config.

Usage:
  # local smoke test against a real bucket (few workers), precomputed:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
    pixi run -e dev python examples/convert_to_s3.py \
      --src /mnt/ceph/.../seg.zarr --format precomputed \
      --dst s3://my-bucket/segmentations/sample3 --workers 4

  # full run on SLURM (resumable -- relaunch to continue):
  nohup pixi run -e dev python -u examples/convert_to_s3.py \
      --src ... --dst s3://... --format both \
      --config configs/dask-slurm-any.yaml --workers 48 > s3.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os

from em_volume_tools import StorageProfile, convert, start_dask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("convert_to_s3")


@contextlib.contextmanager
def maybe_cluster(workers, config):
    if workers and workers > 0:
        with start_dask(workers, config_path=config, label="to-s3") as client:
            yield client
    else:
        yield None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source (path, s3://, precomputed, or .zarr)")
    ap.add_argument("--dst", required=True, help="destination s3://bucket/prefix (base name)")
    ap.add_argument("--src-format", default="zarr3", help="backend for --src if a bare path")
    ap.add_argument("--format", choices=["precomputed", "zarr", "both"], default="precomputed")
    ap.add_argument("--kind", default="segmentation", help="image | probability | segmentation")
    ap.add_argument("--voxel-size", default=None, help="z,y,x nm (omit to read from source)")
    ap.add_argument("--chunk", default="128,128,128", help="z,y,x chunk (viewer-facing)")
    ap.add_argument("--config", default="configs/dask-slurm-any.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="restart instead of resuming")
    args = ap.parse_args()

    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials"))):
        log.warning("No AWS credentials found in env or ~/.aws/credentials; S3 writes will fail.")

    voxel_size = tuple(float(x) for x in args.voxel_size.split(",")) if args.voxel_size else None
    chunk = tuple(int(x) for x in args.chunk.split(","))

    # precomputed: unsharded small chunks (web viewing); zarr: sharded on ceph-style,
    # here unsharded to S3 as well. Adjust per your viewer/storage needs.
    targets = []
    if args.format in ("precomputed", "both"):
        targets.append(("s3-neuroglancer", args.dst.rstrip("/") + ".precomputed"))
    if args.format in ("zarr", "both"):
        targets.append((StorageProfile("zarr3", chunk=chunk, shard=None), args.dst.rstrip("/") + ".zarr"))

    with maybe_cluster(args.workers, args.config) as client:
        for profile, dst in targets:
            log.info("writing %s -> %s", args.kind, dst)
            summary = convert(
                args.src, dst, voxel_size=voxel_size, src_format=args.src_format,
                profile=profile, kind=args.kind, chunk=chunk, multiscale=True,
                client=client, resume=not args.fresh, delete_existing=args.fresh,
                validate=False,
            )
            log.info("done %s: %d levels, status=%s", dst, summary["num_levels"],
                     summary.get("status_counts"))


if __name__ == "__main__":
    main()
