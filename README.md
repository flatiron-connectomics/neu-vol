# em-volume-tools

Chunked I/O, conversion, and multiscale generation for large 3D EM volumes
(images, probabilities/affinities, segmentations), orchestrated with dask —
locally on a workstation or on the Flatiron Rusty cluster via SLURM.

- Architecture and design decisions: [`docs/DESIGN.md`](docs/DESIGN.md)
- Dask + SLURM orchestration cookbook: [`docs/dask-slurm-rusty.md`](docs/dask-slurm-rusty.md)

## Environment

One conda environment covers this repo, its substrate (`em-blockrun`) and its
consumers, each installed editable. Runtime deps come from conda-forge, so
`--no-deps` keeps pip from re-resolving them.

```bash
conda activate em-lib
pip install --no-deps -e ../em-blockrun -e .
python -m pytest -q
```

`em-blockrun` must be a **sibling directory** — the layering depends on it. The
combined environment spec lives one level up, at `em-libraries/environment.yml`.

Previously managed with pixi, with envs detached to ceph to dodge the GPFS home
inode quota; see `docs/DESIGN.md` §8 for that history and why it changed.

## Status

Working first vertical slice: **image stack → multiscale OME-NGFF 0.5 zarr v3**.

```python
from em_volume_tools import ingest_image_stack

ingest_image_stack(
    "/path/to/slices/*.tif",     # glob, directory, or a single multipage TIFF
    "/path/to/out.zarr",
    voxel_size=(8, 8, 8), units="nm",
    profile="local",             # "local" | "ceph" (sharded) | ...
    kind="image",                # "image"/"probability" (mean) | "segmentation" (mode)
    # client=start_dask(...) to run blocks across a cluster
)
```

`convert()` does the same from any source backend (zarr v3, precomputed, HDF5)
to **zarr v3 or neuroglancer-precomputed**, single- or multi-channel. It reads
`voxel_size`/`offset`/`units` from the source when present (OME-NGFF groups,
precomputed `info`); explicit args override. Segmentations default to
`compressed_segmentation` encoding on precomputed.

```python
from em_volume_tools import convert
convert("in.zarr", "out.precomputed",           # voxel_size read from in.zarr's OME metadata
        profile="s3-neuroglancer", kind="segmentation")   # mode pyramid, compressed_segmentation
```

`extract_roi()` crops/pads a region (and can pyramid it) into either format.

Implemented: `VoxelMeta`, `Volume`, block-map engine, `start_dask`, TensorStore
zarr v3 (sharded/unsharded) **and** precomputed (canonical-axis view + multiscale
`info`), image-stack / HDF5 / crop-view sources, type-aware pyramids, storage
profiles, OME-NGFF 0.5 metadata, and the `ingest` / `convert` / `extract_roi` ops.
Outputs verified via ngff-zarr's reader (zarr) and TensorStore round-trip
(precomputed); the distributed path is exercised over a real `LocalCluster`.

## S3 / object stores

Destinations can be local paths, `s3://bucket/prefix` URLs, or kvstore dicts (for
region/endpoint control) — TensorStore handles the transport. Credentials come
from the environment (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) or `~/.aws`.

```python
convert("seg.zarr", "s3://my-bucket/seg/sample3.precomputed",
        profile="s3-neuroglancer", kind="segmentation", resume=True)
```

See `examples/convert_to_s3.py` (incl. notes on propagating creds to SLURM workers).

## Resume & sparsity

`resume=True` makes a run continue after an interruption via a **single-writer
manifest** (the driver records each completed block as results stream back), so no
per-object existence scan is needed and it works for **both** zarr and precomputed.
All-fill (e.g. all-zero) chunks are **elided** (not written) and recorded as
`empty`, so sparse segmentations stay small and resume never reprocesses them.
`verify=True` instead checks storage authoritatively per block.

## Running on the cluster

The same ops run across SLURM workers by passing a `client` from `start_dask`.
See `examples/run_convert_slurm.py` and `docs/dask-slurm-rusty.md`:

```bash
# smoke test locally, then launch on Rusty surviving logout:
nohup python -u examples/run_convert_slurm.py \
    --config configs/dask-slurm-gen.yaml --workers 48 > run.log 2>&1 &
squeue -u "$USER"
```

Next: brightness/normalization + morphological transforms, a CLI. See `docs/DESIGN.md`.
