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

`create_volume()` + `write_subvolume()` are the other shape of the problem: several
small pieces that belong at known positions inside one frame, rather than one source
converted wholesale. Create the (empty) frame — optionally copying a reference
volume's geometry exactly — then place each piece into one level of it.

```python
from em_volume_tools import create_volume, write_subvolume, write_subvolumes

create_volume("annotations.precomputed", like="s3://.../image.zarr",  # same frame, so
              dtype="uint64", kind="segmentation")                    # a voxel index
write_subvolume("annotations.precomputed", "piece.h5", (1024, 4096, 4096))  # matches
write_subvolumes("annotations.precomputed", glob("pieces/*.h5"))  # offsets from the files
```

An offset may be **omitted** when the source records one — an HDF5 file's
`voxel_offset`, looked for in the dataset's attributes, the root group's attributes,
and a top-level dataset of that name. Any backend may supply one by implementing
`stored_offset`. Check `offset_order=` if the value came from a precomputed-flavoured
writer: that field name means *xyz* there, and everything here is zyx.

`write_subvolumes` plans **every** source — offsets, bounds, dtype — before writing
any of them, so a mistyped offset in the last file is caught while the volume is still
untouched. Pieces that overlap each other are reported, not refused.

Either format: `format="zarr"` or `"precomputed"`, defaulting to the reference's own —
"like this volume" includes what kind of volume it is. Creating a level costs one
`zarr.json` (an unwritten zarr array reads back as the fill value), and an empty
precomputed volume is a single `info` listing every scale — so the frame is a few
hundred bytes either way. The writes are **single-scale by design**: how a patch
should look when coarsened is a separate decision (averaging label ids invents ids),
so run `downsample` afterwards if the result needs a pyramid.

Implemented: `VoxelMeta`, `Volume`, block-map engine, `start_dask`, TensorStore
zarr v3 (sharded/unsharded) **and** precomputed (canonical-axis view + multiscale
`info`), image-stack / HDF5 / crop-view sources, type-aware pyramids, storage
profiles, OME-NGFF 0.5 metadata, and the `ingest` / `convert` / `extract_roi` /
`create_volume` / `write_subvolume` ops.
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

On SLURM the **workers** need credentials too. Slurm's default `--export=ALL`
propagates the submitting environment, so launching with the vars set is usually
enough; if your site disables that, use a shared `~/.aws/credentials` readable on the
worker nodes, or add the exports to the dask `job-script-prologue`. Never bake secrets
into a config file.

## Resume & sparsity

`resume=True` makes a run continue after an interruption via a **single-writer
manifest** (the driver records each completed block as results stream back), so no
per-object existence scan is needed and it works for **both** zarr and precomputed.
All-fill (e.g. all-zero) chunks are **elided** (not written) and recorded as
`empty`, so sparse segmentations stay small and resume never reprocesses them.
`verify=True` instead checks storage authoritatively per block.

## The `em-vol` command

Installing the package provides **`em-vol`** (equivalently `python -m em_volume_tools`):

```bash
em-vol info    <volume>                      # format, voxel sizes, chunking, levels
em-vol convert --src ... --dst ...           # build a multiscale volume
em-vol downsample <volume> --start-level 2   # rebuild levels above a trusted one
em-vol progress <volume>                     # chunks written, per level
em-vol create  <dst> --like <reference>      # an EMPTY volume in a known frame
em-vol write   <volume> --src ... --offset   # put one subvolume into it
```

`info` and `progress` read only. `downsample` rebuilds a pyramid **in place** from a
level you trust — cascaded downsampling means a bad level poisons everything above it
— and `--dry-run` prints the schedule beside what is on disk, refusing if they
disagree rather than leaving the pyramid inconsistent. Use `convert` to build a *new*
volume.

`create` and `write` are the small-pieces path and are **not** block-mapped — they run
in the calling process, no dask. Both take `--dry-run`.

```bash
em-vol create /abs/annotations.precomputed --like s3://.../image.precomputed \
    --dtype uint64 --kind segmentation            # empty; same frame as the image
em-vol write /abs/annotations.precomputed --src piece.h5 --offset 1024,4096,4096
em-vol write /abs/annotations.precomputed --src slices/ --offset 4096,16384,16384 \
    --level 2 --offset-level 0                    # coords read off level 0
em-vol write /abs/annotations.precomputed \
    --src a.h5 --src b.h5 --src c.h5              # each file's own voxel_offset
```

`--src` is repeatable and `--offset` is optional: with none given, each source is asked
for its own (`--offset-field`, default `voxel_offset`). Every source in a batch is
checked before any is written. `--offset-order xyz` if the stored numbers are xyz — the
field name is precomputed's, where it is, and reversed the piece lands mirrored through
the z=x diagonal.

`--format zarr|precomputed` picks the target (default: the reference's own).
Precomputed keeps its scales in one intrinsic `info` and takes `--encoding`
(`compressed_segmentation` for `--kind segmentation`, else `raw`); it cannot shard,
and says so rather than dropping `--shard` on the floor.

`--like` copies the reference's level shapes verbatim rather than recomputing a
schedule, so the two volumes cannot drift a voxel apart partway up the pyramid.
`write` reports whether the region is aligned to the destination's chunk grid: an
unaligned edge means those chunks are read-modify-written, which is correct on its
own but loses one of two updates if overlapping writes ever run at once.

```bash
# smoke test locally, then launch on SLURM surviving logout:
em-vol convert --src ... --dst ... --serial --single-level
nohup env PYTHONUNBUFFERED=1 em-vol convert --src ... --dst s3://... \
    --config dask-slurm-example --config ~/my-site.yaml --workers 48 > run.log 2>&1 &
squeue -u "$USER"
```

`PYTHONUNBUFFERED=1` is the console-script equivalent of `python -u`; without it the
log lags a long run in 8 KB blocks.

### Cluster config

`--config` takes a **bundled template name or a path**, and is **repeatable**,
deep-merged left to right. The templates (`dask-local`, `dask-slurm-example`) ship
with **em-blockrun**, next to `start_dask`, so all its consumers share one set.
An overlay carries only the keys that differ:

```yaml
# my-site.yaml
jobqueue:
  slurm:
    account: my-account
    queue: my-partition
    log-directory: /path/with/room
```

Unrecognised keys raise rather than merging silently. Site-specific configs are
deliberately not in this repo; the top-level `configs/` is gitignored.

Next: brightness/normalization + morphological transforms. See `docs/DESIGN.md`.

## License

©2026 The Simons Foundation, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE). See
[CONTRIBUTING.md](CONTRIBUTING.md) for how contributions are licensed.
