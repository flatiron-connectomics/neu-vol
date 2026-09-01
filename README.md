# neu-vol

Chunked I/O, conversion, and multiscale generation for large 3D EM volumes
(images, probabilities/affinities, segmentations), orchestrated with dask —
locally on a workstation or across a SLURM cluster.

Dask + SLURM orchestration background: [`blockrun/docs/dask-slurm.md`](https://github.com/flatiron-connectomics/blockrun/blob/main/docs/dask-slurm.md)

## Environment

One conda environment covers this repo, its substrate (`blockrun`) and its
consumers, each installed editable. Runtime deps come from conda-forge, so
`--no-deps` keeps pip from re-resolving them.

```bash
conda activate neu-env
pip install --no-deps -e ../blockrun -e .
python -m pytest -q
```

`blockrun` must be a **sibling directory** — the layering depends on it. The
[neu-suite](https://github.com/flatiron-connectomics/neu-suite) README describes how to
build the shared environment.

The environment must live somewhere the cluster's workers can also read: an env on
a node's local disk is invisible to them, and the failure looks like a missing
import rather than a path problem.

## Status

Working first vertical slice: **image stack → multiscale OME-NGFF 0.5 zarr v3**.

```python
from neu_vol import ingest_image_stack

ingest_image_stack(
    "/path/to/slices/*.tif",     # glob, directory, or a single multipage TIFF
    "/path/to/out.zarr",
    voxel_size=(8, 8, 8), units="nm",
    profile="local",             # "local" | "ceph" (sharded) | ...
    kind="image",                # "image"/"probability" (mean) | "segmentation" (mode)
    # client=start_dask(...) to run blocks across a cluster
)
```

`convert()` does the same from any source backend (zarr v3, precomputed, DVID, HDF5,
image stack) to **zarr v3 or neuroglancer-precomputed**, single- or multi-channel. It
reads `voxel_size`/`offset`/`units` from the source when present (OME-NGFF groups,
precomputed `info`, a DVID instance, an HDF5 file that records its own frame); explicit
args override. Segmentations default to `compressed_segmentation` encoding on
precomputed.

The source format is detected, so `--src-format` is only for overriding it. Volumes are
recognised by their metadata document (`info`, `zarr.json`, `.zarray`/`.zgroup`) and DVID
by its URL scheme; **an HDF5 file and a stack of 2D slices are recognised by name**,
since neither has a marker object — an `.h5`/`.hdf5` file, a glob, a multipage TIFF, or a
directory that actually contains `.tif`/`.png` slices. A marker always wins, so a stray
image beside an `info` changes nothing.

```python
from neu_vol import convert
convert("in.zarr", "out.precomputed",           # voxel_size read from in.zarr's OME metadata
        profile="s3-neuroglancer", kind="segmentation")   # mode pyramid, compressed_segmentation
```

`crop_start`/`crop_stop` copy **one box** instead of the whole volume, clipped to the
source extent. The output's physical offset shifts by the crop origin, so it stays in
the source's coordinate frame and overlays the original in a viewer; its pyramid is then
built from the cropped level 0, never from slices of the source's coarse levels.

```python
convert("in.precomputed", "crop.precomputed", kind="segmentation",
        crop_start=(5632, 4480, 6784), crop_stop=(5760, 4736, 7040))   # zyx voxels
```

`extract_roi()` is the same crop with the opposite out-of-bounds policy: `start` may be
negative and `stop` may pass the extent, and the margin is filled with `pad_value`
instead of trimmed. It delegates to `convert()`.

`mask_boxes` is the complement — copy everything *except* those boxes, writing
`mask_value` inside them. It masks on the read side, so every pyramid level inherits the
hole; a box that misses the volume raises rather than quietly copying everything. Note
that an all-fill block is **elided rather than written**, so a mask cannot erase a region
the destination already holds from an earlier run.

On the command line, `neu-vol convert --crop-bbox z0,y0,x0,z1,y1,x1` and
`--mask-bbox` (repeatable) do this, with `--bbox-scale N` and `--bbox-order xyz` applying
to every box the command takes, and
**`neu-vol copy` is the same command with the source's own parameters as the defaults** —
format, chunking, voxel size and image/segmentation type all read from the source, with
anything it does not record an error rather than a guess. That last part is why it exists:
`convert --kind` defaults to `image`, so copying a segmentation and forgetting the flag
averages label ids into ids that were never in the data, silently, while the source's
`info` said `segmentation` all along.

`create_volume()` + `write_subvolume()` are the other shape of the problem: several
small pieces that belong at known positions inside one frame, rather than one source
converted wholesale. Create the (empty) frame — optionally copying a reference
volume's geometry exactly — then place each piece into one level of it.

```python
from neu_vol import create_volume, write_subvolume, write_subvolumes

create_volume("annotations.precomputed", like="s3://.../image.zarr",  # same frame, so
              dtype="uint64", kind="segmentation")                    # a voxel index
write_subvolume("annotations.precomputed", "piece.h5", (1024, 4096, 4096))  # matches
write_subvolumes("annotations.precomputed", glob("pieces/*.h5"))  # offsets from the files
```

An offset may be **omitted** when the source records one — an HDF5 file's
`voxel_offset`, looked for in the dataset's attributes, the root group's attributes,
and a top-level dataset of that name. Any backend may supply one by implementing
`stored_offset`. The axis order is likewise read from the file when it states one (an
`axes` attribute, via `stored_axes`), falling back to zyx; `offset_order=` overrides both.
Check it if the value came from a precomputed-flavoured writer that records no `axes`:
that field name means *xyz* there, and everything here is zyx.

`pack_hdf5()` produces such a file — the inverse of `write_subvolume`. It packs an image
stack (or any readable source) into one HDF5 dataset carrying `voxel_offset`, `voxel_size`,
`offset`, `units` and `axes`, so the piece can be placed later with no arguments:

```python
from neu_vol import pack_hdf5
pack_hdf5("slices/", "piece.h5", voxel_size=(40, 8, 8), voxel_offset=(24, 128, 256))
```

The dataset defaults to `/data`, which is what the reader assumes when it is not told —
though a reader given only a path now *resolves* the name instead, and only falls back on
the default when nothing else says. An existing file is added to if its recorded frame
matches — several pieces of one volume in one file, each with its own `voxel_offset` — and
refused if it does not. `neu-vol info <file.h5>` lists what such a file ended up holding.

`read_piece()` is the general read: a box out of any source, as a `neu_lib.Piece` that
carries the frame, the kind and a name derived from the source. It is what the neu-glance
viewer constructors call, so a notebook and a viewer see the same thing:

```python
from neu_vol import read_piece
gt = read_piece("gt.h5:/vol_03700", "segmentation")   # frame from the file's attributes
em = read_piece("s3://my-bucket/em", crop=gt)         # the SAME physical box
gt.bbox, gt.bounds_nm                                 # where it is, in voxels and in nm
```

`crop=` takes a voxel box, another `Piece` (meaning *the same physical box as that*), or
`{"nm": (lo, hi)}` — nanometres being the only space that transfers between frames, since
two levels have different voxel sizes and a crop and its parent different origins. A single
read is capped at 4 GiB, because a whole production level 0 is terabytes and a read that
size does not fail, it **hangs**; the error names a coarser level that would fit.

`dtype=` casts after the read, so a piece arrives in the dtype it will be used in — crops
exported by different tools come as uint8/16/32/64, and a consumer that has to handle all
four is what this avoids. A narrowing cast is warned about rather than refused: widening
among unsigned ints loses nothing, while narrowing wraps a label id into another plausible
label id, which nothing downstream can detect.

`write_piece()` is the inverse, and the door for an array **already in memory** — where
`pack_hdf5` reads from a location. Read a crop, transform it, write the result:

```python
from neu_vol import read_piece, write_piece
piece = read_piece("gt_v1_eval.h5:/vol_03700", "segmentation")
write_piece(piece.apply(clean), "gt_v1_eval_cleaned.h5")     # -> /vol_03700, same frame
```

Four of `pack_hdf5`'s arguments are absent because the piece answers them: `voxel_size` is
the frame's, `voxel_offset` is `piece.origin_voxel` (which raises rather than rounding a
fractional origin), `axes` is always zyx and `units` always nm. The dataset name comes from
the piece's own name, so a whole bag of crops round-trips through a cleaning pass with no
arguments at all, accumulating into one output file. `kind` travels too — a cleaned
segmentation reads back as one. Both writers go through the same layout code in `ops/pack`,
so `write_subvolume` places either file the same way.

`open_hdf5()` is the read side — one dataset of a file, ready to read regions from, with
no spec to write out:

```python
from neu_vol import open_hdf5
be = open_hdf5("piece.h5")                     # one dataset: found
be = open_hdf5("gt_v1_eval.h5", "/z07901")     # a container: name it
be.read_region((slice(0, 64), slice(0, 64), slice(0, 64)))
```

The name is required when the file holds several arrays, since choosing one of thirteen
crops on your behalf would be wrong twelve times out of thirteen; the error lists them.
`open_backend` stays spec-only on purpose — it is the primitive a per-block dask task
reopens on a worker, so its reader is named rather than inferred. To find out what
something *is*, use `describe`.

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

### What each pyramid level is

`read_scales(src)` returns a `neu_lib.ScaleInfo` per level, finest first, read from the
source's own metadata — precomputed's `info` or an OME group's `zarr.json`:

```python
from neu_vol import read_scales, scale_spec, describe_scales

levels = read_scales("s3://bucket/seg.precomputed")
levels[2].voxel_size          # (32.0, 32.0, 32.0) — this level's OWN resolution
levels[2].shape               # voxels, zyx
levels[2].frame.to_nm([0, 0, 0])   # where its voxel (0,0,0) sits, offset included
print(describe_scales(src))   # the pyramid as a table
```

**Never derive a factor from `2 ** level`.** Real pyramids are anisotropic — halving x
and y while leaving z alone is ordinary — so `factor_from` computes it from the real
voxel sizes. Each level also carries its **origin**, from precomputed's `voxel_offset`
(that level's voxels, so the nm origin is the product) or OME's `translation` (already
physical). A cropped volume has a non-zero one, and reconstructing a transform from just
the voxel size silently places its data as though it started at nm zero.

`scale_spec(src, n)` builds the read spec for one level. **Always use it rather than
writing a spec by hand**: the key is `scale_index`, an unrecognised key is *silently
ignored*, and `{"scale": 2}` therefore opens at full resolution while reporting the
scale-0 shape — so coordinates meant for scale 2 read the wrong place and come back
empty rather than raising.

Implemented: `VoxelMeta`, `Volume`, block-map engine, `start_dask`, TensorStore
zarr v3 (sharded/unsharded) **and** precomputed (canonical-axis view + multiscale
`info`), image-stack / HDF5 / crop-view sources, type-aware pyramids, storage
profiles, OME-NGFF 0.5 metadata, per-level scale metadata, and the `ingest` /
`convert` / `extract_roi` / `create_volume` / `write_subvolume` ops.
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

## The `neu-vol` command

Installing the package provides **`neu-vol`** (equivalently `python -m neu_vol`):

```bash
neu-vol info    <volume>                      # format, voxel sizes, chunking, levels
                                              #   (also an HDF5 file or a slice stack)
neu-vol convert --src ... --dst ...           # build a multiscale volume
neu-vol downsample <volume> --start-level 2   # rebuild levels above a trusted one
neu-vol progress <volume>                     # chunks written, per level
neu-vol create  <dst> --like <reference>      # an EMPTY volume in a known frame
neu-vol write   <volume> --src ... --offset   # put one subvolume into it
neu-vol to-hdf5 --src slices/ --out piece.h5  # pack a piece, frame and position included
neu-vol align-bbox --volume ... --bbox ...    # move a box onto the block grid
neu-vol relabel <volume> --out ...            # one id range per occupied region
neu-vol mask-by-value <volume> --values 1     # background that is not 0, made 0
```

Anything a **viewer** consumes lives in [neu-glance](../neu-glance): `neu-glance gen` for a link or a
state, `neu-glance bboxes` for a layer of boxes over a sparse volume's data, `neu-glance annotate`
for a layer of your own coordinates. This package supplies the occupancy analysis those
build on (`ops.annotate.labeled_regions`) and otherwise knows nothing about neuroglancer.
The three used to be `neu-vol bboxes-json`, `annotate-json` and `ng-url-gen`; the rename was
a clean break with no aliases.

`info`, `progress` and `align-bbox` read
only. `downsample` rebuilds a pyramid **in place** from a
level you trust — cascaded downsampling means a bad level poisons everything above it
— and `--dry-run` prints the schedule beside what is on disk, refusing if they
disagree rather than leaving the pyramid inconsistent. Use `convert` to build a *new*
volume.

`create` and `write` are the small-pieces path and are **not** block-mapped — they run
in the calling process, no dask. Both take `--dry-run`.

```bash
neu-vol create /abs/annotations.precomputed --like s3://.../image.precomputed \
    --dtype uint64 --kind segmentation            # empty; same frame as the image
neu-vol write /abs/annotations.precomputed --src piece.h5 --offset 1024,4096,4096
neu-vol write /abs/annotations.precomputed --src slices/ --offset 4096,16384,16384 \
    --level 2 --offset-level 0                    # coords read off level 0
neu-vol write /abs/annotations.precomputed \
    --src a.h5 --src b.h5 --src c.h5              # each file's own voxel_offset
```

`--src` is repeatable and `--offset` is optional: with none given, each source is asked
for its own (`--offset-field`, default `voxel_offset`). Every source in a batch is
checked before any is written. `--offset-order xyz` if the stored numbers are xyz — the
field name is precomputed's, where it is, and reversed the piece lands mirrored through
the z=x diagonal.

`--all-datasets` writes **every** volumetric dataset of an HDF5 `--src`, each at its own
recorded offset — a bag of ground-truth crops in one file goes back into the volume in one
command, rather than one `--dataset` at a time:

```bash
neu-vol write <volume> --src gt_v1_eval_cleaned.h5 --all-datasets
neu-vol write <volume> --src gt_v1_eval_cleaned.h5 --all-datasets 'z*'   # a subset
```

The optional glob is matched against the dataset path *and* its basename, so `'z*'` and
`'/edge_*'` both mean what they look like, and a pattern matching nothing **raises** rather
than writing nothing and reporting success. The expansion feeds the same batch, so all of
them are still planned before any is written. It cannot be combined with `--dataset` (which
names one) or `--offset` (which belongs to a source you typed). `ops.write.container_sources`
is the same thing from Python.

`--format zarr|precomputed` picks the target (default: the reference's own).
Precomputed keeps its scales in one intrinsic `info` and takes `--encoding`
(`compressed_segmentation` for `--kind segmentation`, else `raw`); it cannot shard,
and says so rather than dropping `--shard` on the floor.

`--like` copies the reference's level shapes verbatim rather than recomputing a
schedule, so the two volumes cannot drift a voxel apart partway up the pyramid.
`write` reports whether the region is aligned to the destination's chunk grid: an
unaligned edge means those chunks are read-modify-written, which is correct on its
own but loses one of two updates if overlapping writes ever run at once.

### Inspecting anything readable, not just a volume

`neu-vol info` works on every format the package can read — a zarr or precomputed
volume, a `dvid://` instance, an HDF5 file, or a directory/glob of 2D slices:

```bash
neu-vol info piece.h5                        # shape, dtype, and the frame it records
neu-vol info slices/                          # a stack: shape and dtype; no scale to read
neu-vol info gt_v1_eval.h5                    # SEVERAL datasets: lists them all
neu-vol info gt_v1_eval.h5 --dataset /z07901  # ...then the full report on one
```

The last two are the case worth knowing. An HDF5 file is a **container**, and here it is
usually a bag of crops rather than one volume — several annotated chunks in one file, each
keeping its own `voxel_offset` so `write` can place it back. A path names the container,
not an array, so `info` lists every 3D+ dataset with its shape, chunking and offset, and
`--dataset` selects one. With exactly one dataset it is found and nothing has to be said.

`describe()` is the library form of all of this and returns the same picture — for a
container, `datasets` listing every array with its own frame and `shape`/`meta` left
`None`, since thirteen differently-shaped crops have no single answer. It is a **dict
that shows itself**: `print(describe(f))` gives the table above, a bare `describe(f)` in
a notebook cell renders it, and `describe(f).frame()` is a DataFrame — one row per
dataset for a container, one per level for a volume, holding the numbers rather than the
formatted strings. Subscripting is unchanged, and pandas is only needed for `.frame()`.

Everything else that resolves an array (`create --like`, `align-bbox`, `convert`,
`to-hdf5 --src`) needs exactly one, and each takes the dataset name where it can. If a
file records no frame, `info` says which attribute names it looked for — `voxel_size`,
`voxel_offset`, `offset`, `units`, `axes` — because "records no scale" and "spells it
differently" look identical otherwise, and those names are parameters
(`--voxel-size-field` / `--offset-field`) for exactly that reason.

The ops that **rewrite a volume in place** — `downsample`, `relabel`, `mask-by-value`,
`write`, `progress` — refuse a container and say so: one array in one container has no
pyramid to rebuild, and no chunk objects whose presence answers "where is the data".
Convert it into a volume first, or use `write` to place it *into* one.

### Putting a box on the grid

`align-bbox` moves a bounding box onto a block grid and prints it back, ready to
substitute into `--crop-bbox` or `--roi`:

```bash
neu-vol align-bbox --volume V --bbox 5600,4470,6790,5770,4740,7050 --to both
neu-vol copy --src V --dst D --crop-bbox $(neu-vol align-bbox --volume V --bbox ... -q)
```

**Which grid is the question**, and there are three: the **write unit** (the chunk, or the
**shard** where the level is sharded — a partial write is a read-modify-write, and two
concurrent ones into a single object lose an update silently); the **pyramid's cumulative
factor** (a crop that misses it has coarse levels on their own grid, level 0 still exact);
and the **per-axis LCM** of the two, which is what a cropped copy wants. `--to read-chunk`
asks the fourth question — read amplification — which is not a safety one.

Modes: `outer` (cover; cannot fail), `inner` (be covered), `nearest`, and `origin`, which
aligns the origin and keeps the extent exactly for a fixed-size crop. `--block z,y,x`
needs no volume; `--scale N` takes the box in another level's voxels via the real per-level
voxel sizes. Boxes are half-open, so a bound already on a boundary stays put, and a bound
at the volume's own extent counts as aligned — that final block is partial in the volume
too. `neu-vol write` reports alignment through the same predicate, so the two agree.

### Finding the data in a sparse volume

A volume holding a few labeled boxes inside a large empty frame is hard to *view* — the
boxes are needles in the frame. Finding them is `ops.annotate.labeled_regions`, and the
boxes come from the volume itself so they cannot drift from the data: an all-fill chunk is
never stored, so *which chunk objects exist* is the occupancy question exactly — no voxel
reads. Those cells are then covered with maximal boxes (not connected components: two
regions written face to face merge into one, plus the empty corner between them), and each
box is tightened to its nonzero voxels at a coarse level.

`relabel`, `mask-by-value` and `downsample --sparse` all ask that same question. To *see*
the answer, `neu-glance bboxes` turns it into a viewer layer:

```bash
neu-glance bboxes s3://.../gt_v1 --label gt                     # layer JSON to stdout
neu-glance bboxes s3://.../gt_v1 --format url --out link.txt    # or a whole link
```

### Ground truth annotated chunk by chunk

Annotation tools usually number each chunk from 1, so the same integer means a different
cell in every chunk. Meshed, that becomes one body with components scattered across the
volume — correct for the label, useless as ground truth. `relabel` gives each occupied
region its own range:

```bash
neu-vol relabel s3://.../gt_v1 --out s3://.../gt_v2 --dry-run   # reads, writes nothing
neu-vol relabel s3://.../gt_v1 --out s3://.../gt_v2             # then --start-level 0
neu-vol downsample s3://.../gt_v2 --start-level 0
```

Regions are the same stored-chunk footprints `labeled_regions` reports, so they are pairwise
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
neu-vol convert --src ... --dst ... --serial --single-level
nohup env PYTHONUNBUFFERED=1 neu-vol convert --src ... --dst s3://... \
    --config dask-slurm-example --config ~/my-site.yaml --workers 48 > run.log 2>&1 &
squeue -u "$USER"
```

`PYTHONUNBUFFERED=1` is the console-script equivalent of `python -u`; without it the
log lags a long run in 8 KB blocks.

### Cluster config

`--config` takes a **bundled template name or a path**, and is **repeatable**,
deep-merged left to right. The templates (`dask-local`, `dask-slurm-example`) ship
with **blockrun**, next to `start_dask`, so all its consumers share one set.
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

Next: brightness/normalization + morphological transforms.

## License

©2026 The Simons Foundation, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE). See
[CONTRIBUTING.md](CONTRIBUTING.md) for how contributions are licensed.
