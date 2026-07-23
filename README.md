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

Output is read back cleanly by ngff-zarr's own reader (verified). Implemented:
`VoxelMeta`, `Volume`, block-map engine, `start_dask`, TensorStore zarr v3 backend
(sharded/unsharded), image-stack source, type-aware pyramids, storage profiles,
OME-NGFF 0.5 metadata. Next: precomputed writer, `convert`/`roi` ops, HDF5 source,
SLURM smoke test. See `docs/DESIGN.md` for the roadmap.
