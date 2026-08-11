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
em-vol bboxes-json <volume>                  # a viewer layer of boxes over the data
em-vol relabel <volume> --out ...            # one id range per occupied region
em-vol ng-url-gen --image ... --seg ...      # a neuroglancer link with a full state
```

`info`, `progress`, `bboxes-json` and `ng-url-gen` read only. `downsample` rebuilds a pyramid **in place** from a
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

### Finding the data in a sparse volume

A volume holding a few labeled boxes inside a large empty frame is hard to *view* —
the boxes are needles in the frame. `bboxes-json` emits a neuroglancer annotation
layer with one bounding box per written region, giving a clickable list that jumps
between them:

```bash
em-vol bboxes-json s3://.../gt_v1 --label gt        # layer JSON to stdout, table to stderr
em-vol bboxes-json s3://.../gt_v1 --out layer.json  # or s3://... — both work
em-vol bboxes-json s3://.../gt_v1 --state --out state.json   # a whole loadable state
```

The boxes come from the volume itself, so they cannot drift from the data: an all-fill
chunk is never stored, so *which chunk objects exist* is the occupancy question exactly
— no voxel reads. Those cells are then covered with maximal boxes (not connected
components: two regions written face to face merge into one, plus the empty corner
between them), and each box is then tightened to its nonzero voxels.

`--tighten-level` defaults to `--level`, so at the default level 0 the boxes are exact
in the voxels they are reported in, and the reads cost what the level you picked costs.
Raise it when the occupied footprint is large — each level is a factor smaller to read,
at the price of quantizing every bound to one voxel there. `--no-tighten` leaves the
boxes on the chunk grid and reads nothing.

The annotations are **local** — inline in the state — rather than a precomputed
annotation layer, because neuroglancer does not *list* precomputed annotations: it
builds the annotation panel by iterating the layer's source, and the class behind every
precomputed annotation source defines that iterator as empty. Those annotations render
in the viewport but cannot be clicked through, which defeats the purpose here.

### Sharing a view

`ng-url-gen` builds a neuroglancer link carrying a whole viewer state — which volumes
are loaded, where the view sits, which segments are selected. It reads the volumes to
get the source scheme and the coordinate space right, which is the part that fails
silently by hand: a `dimensions` block that disagrees with the data loads fine and puts
every layer in the wrong place.

```bash
em-vol bboxes-json s3://.../gt_v2 --label gt --out gt.json
em-vol ng-url-gen --image s3://.../em --seg s3://.../gt_v2 \
    --layer gt.json --segments 1,2,3 --layout xy-3d --select-last
```

The opening view is centred on the largest volume and zoomed out to fit it, because
neuroglancer with no `position` opens at the origin **corner** at one voxel per pixel —
which on a large volume is a view of its empty edge. `--position`,
`--cross-section-scale` and `--projection-scale` each override that part independently.
With only `--layer` files, the annotations' own bounding box sets the view instead.

It composes with `bboxes-json` through `--layer`, which takes either a bare layer or a
whole state and uses its layers, so it does not matter which the other command emitted.
`--position` is zyx like every other coordinate here; pass `--position-order xyz` to use
numbers copied straight out of the viewer, since that is what neuroglancer displays.

Everything after `#!` is a URL fragment and never reaches a server, so the link carries
no data anywhere — but the whole state travels in it, and a large inline annotation layer
makes for a long URL. `--state-out` writes the JSON alongside for pasting into the `{}`
editor instead.

### Ground truth annotated chunk by chunk

Annotation tools usually number each chunk from 1, so the same integer means a different
cell in every chunk. Meshed, that becomes one body with components scattered across the
volume — correct for the label, useless as ground truth. `relabel` gives each occupied
region its own range:

```bash
em-vol relabel s3://.../gt_v1 --out s3://.../gt_v2 --dry-run   # reads, writes nothing
em-vol relabel s3://.../gt_v1 --out s3://.../gt_v2             # then --start-level 0
em-vol downsample s3://.../gt_v2 --start-level 0
```

Regions are the same stored-chunk footprints `bboxes-json` uses, so they are pairwise
disjoint and chunk-aligned — no write is a partial-chunk update. It is serial by
construction, since each range begins where the last ended, so it runs in the calling
process with no dask. `--block-size N` numbers region *k* from `N*k+1` instead, making
the source region readable off the id.

The old→new mapping is written to `<destination>.relabel-<level>.json` (`--map` to
place it) — the only way back from a new id to the region and original label it came
from. `--out` is preferred over `--in-place`: a sparse copy is nearly free and the
original stays as the record of the raw annotation. Single-scale, like `write`, so the
levels above are stale until `downsample` re-runs; it says so.

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
