# em-volume-tools

Chunked I/O, conversion, and multiscale generation for large 3D EM volumes
(images, probabilities/affinities, segmentations), orchestrated with dask —
locally on a workstation or on the Flatiron Rusty cluster via SLURM.

- Architecture and design decisions: [`docs/DESIGN.md`](docs/DESIGN.md)
- Dask + SLURM orchestration cookbook: [`docs/dask-slurm-rusty.md`](docs/dask-slurm-rusty.md)

## Environment

Managed with [pixi](https://pixi.sh). The environment is **detached to ceph**
(`/mnt/ceph/users/<user>/pixi-envs`) to avoid the GPFS home inode quota and so
that SLURM workers on Rusty can import it directly. The pixi package cache stays
on local `/home` (no quota).

```bash
pixi install          # solve + create the detached env on ceph
pixi run test         # run the test suite
pixi shell            # drop into the environment
```

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
to **zarr v3 or neuroglancer-precomputed**, single- or multi-channel:

```python
from em_volume_tools import convert
convert("in.zarr", "out.precomputed", voxel_size=(8, 8, 8),
        profile="s3-neuroglancer", kind="segmentation")   # mode-downsampled pyramid
```

`extract_roi()` crops/pads a region (and can pyramid it) into either format.

Implemented: `VoxelMeta`, `Volume`, block-map engine, `start_dask`, TensorStore
zarr v3 (sharded/unsharded) **and** precomputed (canonical-axis view + multiscale
`info`), image-stack / HDF5 / crop-view sources, type-aware pyramids, storage
profiles, OME-NGFF 0.5 metadata, and the `ingest` / `convert` / `extract_roi` ops.
Outputs verified via ngff-zarr's reader (zarr) and TensorStore round-trip
(precomputed); the distributed path is exercised over a real `LocalCluster`.

## Running on the cluster

The same ops run across SLURM workers by passing a `client` from `start_dask`.
See `examples/run_convert_slurm.py` and `docs/dask-slurm-rusty.md`:

```bash
# smoke test locally, then launch on Rusty surviving logout:
nohup python -u examples/run_convert_slurm.py \
    --config configs/dask-slurm-gen.yaml --workers 48 > run.log 2>&1 &
squeue -u "$USER"
```

Next: read source metadata on `convert`, brightness/normalization + morphological
transforms, a CLI. See `docs/DESIGN.md`.
