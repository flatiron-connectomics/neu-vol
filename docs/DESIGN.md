# em-volume-tools — Design

A library for access, conversion, transformation, and storage of large 3D
volumetric arrays (EM images, probabilities/affinities, segmentations), with
chunked I/O and dask orchestration (local workstation or Rusty/SLURM). Replaces
ad-hoc chunkflow pipelines with a smaller, explicit toolkit.

Orchestration foundation: see `docs/dask-slurm-rusty.md` (the `start_dask`
cookbook). This library provides the *volume* layer that sits on top of it.

---

## 1. Scope

**v1 (build first):**
- Ingest image stacks (multi-file 2D TIFF/PNG, or single-file multipage TIFF) →
  chunked volume.
- Format conversion among: zarr v3, neuroglancer-precomputed, HDF5 (chunk
  shards, read/assemble), image stacks (read).
- Multiscale pyramid generation (type-aware downsampling).
- ROI extraction / cropping / padding.
- Dtype conversion (cast, with optional clip/scale range).

**First vertical slice:** image stack → multiscale zarr v3.

**Later (not v1):** brightness scaling / normalization, morphological ops
(dilation/erosion) on segmentations, relabeling, channel algebra, CLI.

**Planned — DVID source backend (~week of 2026-07-31):** fetch segmentation
label arrays in chunks from a DVID server into precomputed/zarr. Implemented as a
read-only `backends/dvid.py` (tag `"dvid"`) that slots into the existing source
pattern, with the DVID client as an **optional dependency** (a PyPI extra +
lazy import inside the backend, so it's never required). Details in the
`dvid-source-todo` memory.

**Explicitly out of scope:** meshing, skeletonization (separate library; this
lib owns the multiscale *image/label arrays* that those attach to).

---

## 2. Backends (wrap, don't reinvent)

| Backend | Library | Role |
|---|---|---|
| zarr v3 (array data) | TensorStore (primary), zarr-python | read + write array chunks, incl. sharding codec |
| OME-Zarr multiscale metadata | ngff-zarr | write/validate the group-level `multiscales` block (see §6a) |
| neuroglancer-precomputed | TensorStore | read + write, incl. intrinsic multiscale `info` (avoid CloudVolume for writing) |
| image stacks | tifffile, imageio | read-only source (TIFF/PNG, multipage TIFF) |
| HDF5 | h5py | read/assemble chunk shards |

- **TensorStore is the primary chunked read/write engine** for both zarr v3 and
  precomputed. CloudVolume is, at most, an optional *read* fallback — never the
  default writer for segmentations (observed precomputed corruption that
  TensorStore/neuroglancer mis-read).
- Precomputed is the target when meshes/skeletons will later be attached
  (separate lib). This lib produces the image/label array + multiscale metadata.

---

## 3. Core abstraction

**`Volume`** — a *lazy handle* over a backend store. Holds no data. Knows:
`shape`, `dtype`, `chunks`, channel count, and a `VoxelMeta`. Exposes region
I/O: `vol[z0:z1, y0:y1, x0:x1] -> ndarray`, `vol.write_region(sl, arr)`.

**Backend protocol** (`backends/base.py`): `create`, `open`, `read_region`,
`write_region`, `info`. The engine is backend-agnostic — a conversion is "read
from backend A, write to backend B, block by block."

**`VoxelMeta`** (`meta.py`) — first-class coordinate metadata: `voxel_size`,
`offset`, `units`, axis order. Read from source where present (zarr/precomputed/
OME); for image stacks (no reliable metadata) the caller **must** supply
`voxel_size`. Translate between precomputed (nm, XYZ) and zarr/OME axis
conventions on conversion.

**Locations** (`location.py`) — everything downstream speaks TensorStore *kvstore
specs*, so a local path and an `s3://` / `gs://` URL take the same code path.
`to_kvstore` normalizes, `join` appends path segments, and the byte/JSON layer —
`read_bytes` / `write_bytes` / `read_json` / `write_json` / `exists` — carries
sidecar metadata (a precomputed `info`, a mesh fragment) to either kind of
destination. Two consequences worth knowing:

- The **file driver creates parent directories on write**, so these replace
  `os.makedirs` + `open` rather than supplementing them, and a single kvstore key
  write is atomic — no tmp-plus-rename needed.
- A **missing object reads as `None`** (`state == "missing"`), so existence is one
  request with no separate stat and no exception handling.

`is_local` / `local_path` are the escape hatch for things that genuinely need
POSIX — sqlite, appended logs, `os.replace`. Callers use them to reject a remote
location up front with a clear message instead of failing deep inside a write;
em-seg-morpho's `--work-dir` is exactly this case.

---

## 4. Block-map engine + dask

`engine.py` iterates over **output chunks**, computes the input region each
needs, and emits one task per output block. Tasks are the `items` fed to the
cookbook's `db.from_sequence(items).map(fn)` under `start_dask` (local or SLURM
via one YAML config).

Rules (from the cookbook gotchas):
- **Idempotent tasks** — skip an output block already written; enables
  resume-by-relaunch and `walltime` growth across restarts. Resume is
  **manifest-based** (`resume=True`): the driver is the single writer, and as
  per-block results stream back (`client.map` + `as_completed`) it appends each
  block's status to a JSONL manifest (`location.default_progress_path`, on ceph /
  next to a local dst). On resume the ops filter out already-done blocks before
  dispatch — no per-object scan, works for **both** zarr and precomputed, and
  records intent so empty chunks aren't reprocessed. `verify=True` instead checks
  storage authoritatively per block (zarr via `storage_statistics`; precomputed
  via a kvstore chunk-key existence check, since its `storage_statistics` is
  broken in tensorstore 0.1.84).
- **Empty-chunk elision** — a block equal to the fill value (0) is not written
  (read back as fill) and recorded as `empty`. Keeps sparse segmentations small.
- **Destinations are locations** — `location.to_kvstore` maps local paths /
  `s3://` / `gs://` / kvstore dicts to TensorStore kvstores; ops join subpaths and
  write group metadata through them, so S3 works with no code change (creds via
  `AWS_*` env or `~/.aws`).
- **No big arrays through the scheduler** — workers write to the store, return
  small status tuples.
- **Thread-pinned single-thread workers**; disable dask memory manager with
  `false` (not `0.0`).
- Bound per-task memory by the output-block size × input footprint.

---

## 5. Storage target profiles (destination-aware defaults)

Key idea: **decouple read-chunk size from file/object size**, choose per
destination. All knobs overridable.

| Profile | Format | Read chunk | File/object strategy | Rationale |
|---|---|---|---|---|
| `s3-neuroglancer` | precomputed | 128³ (or 256³) | **unsharded** small objects | cheap HTTP range reads; S3 has no inode quota |
| `ceph` | zarr v3 | 128³/256³ | **sharding codec**, shard ≈ 512³–1024³ | web-like granularity, few inodes (quota-safe) |
| `local` | zarr v3 | 128³ | unsharded | simplest for dev/smoke tests |

Context: full datasets ≈ 10,000³ voxels. At 128³ that is ~475k chunks *per
level* — fine as S3 objects, a quota problem as individual ceph files, hence the
zarr sharding codec for ceph intermediates. Precomputed sharding is *supported*
but not default (S3 viewing prefers small unsharded chunks).

---

### 6a. Multiscale metadata: array-level vs group-level

TensorStore is **array-level** — it writes each scale as an individual zarr array
(chunks, sharding, dtype, zarr-v3 `dimension_units`/labels) but does **not**
author OME-Zarr's group-level `multiscales` metadata (dataset list,
`coordinateTransformations`, `axes`). That layer is a convention on top of plain
zarr arrays.

Decision: **strict OME-Zarr (OME-NGFF) 0.5 compliance**, written via
**`ngff-zarr`**. NGFF 0.5 is the zarr-v3-based revision (0.4 is zarr-v2). Flow
for zarr output: TensorStore writes each level array → `ngff-zarr` writes the
`multiscales` group metadata (axes, per-level scale/translation from
`VoxelMeta`). Output must validate against the 0.5 spec (napari/validators).

Contrast: **precomputed multiscale is intrinsic to the format** (scales in the
single `info` file), so TensorStore handles it natively — no separate metadata
step needed there.

## 6. Pyramids

Built **level-by-level**: level 0 = ingest/convert; each subsequent level is its
own resumable block-map reading a downsample-factor region from the level above.

- **Schedule:** derived from `VoxelMeta.voxel_size` toward (near-)isotropy.
  Typical data here is isotropic 8×8×8 nm → plain `2×2×2` every level.
  Anisotropic input → in-plane `2×2×1` until Z ≈ XY, then `2×2×2`. Caller can
  override with explicit per-level factors.
- **Type-aware reducers:**
  - images / probabilities → mean (anti-aliased) downsample.
  - segmentations (uint64) → **mode / label-preserving** (never interpolate IDs).
  - caller-overridable.

### 6a. Rebuilding a pyramid in place

Because each level is derived from the one below it, a bad level poisons every
level above it — and re-converting to fix the coarse end means recopying level 0.
`rebuild_pyramid(dst, start_level=k)` (`ops/rebuild.py`, CLI at
`em-vol downsample`) regenerates levels `k+1…` in place. Levels at or
below `k` are never opened for writing.

- The schedule is recomputed **from level 0**, not from the seed. Its tail is the
  same either way (`downsample_schedule` is iterative), but computing all of it
  also yields the shapes needed to rewrite metadata describing the whole pyramid,
  and lets the seed's shape be checked against what the schedule predicts.
- **`min_dim` / `max_levels` / `factors` must match the original conversion**, or
  the two disagree on level count. `--dry-run` prints the computed schedule beside
  the levels on disk, which is how you check. Note `max_levels` counts levels
  **including level 0** — it bounds the schedule at `max_levels - 1`, since the schedule
  covers levels 1..L. It formerly bounded the list itself and so allowed one level more
  than it named; a volume built before that fix needs one higher a value here.
- The seed goes through a separate `open_level` callback that only ever opens.
  Routing it through `create_level` would let `resume=False` recreate the very
  level being rebuilt from, destroying the input.
- `create_level` must *reopen* the regenerated levels rather than create them —
  they already exist, and creating over an existing precomputed scale is an error.
- `kind` defaults to what the volume records (precomputed `info["type"]`, OME
  multiscales `type`). It picks the reducer, so a wrong value is silent and
  destructive: averaging label ids invents ids that were never in the data.

### 6b. Schedule & memory

Strict **level-by-level materialization**: each level is its own `block_map`
phase reading from the *already-written* finer level on disk (a barrier between
levels; full parallelism within a level; free checkpointing). Crucially we
**never recurse multiple levels in-memory** — that would need `(∏f)^L` voxels
per output block.

Per-task memory is depth-independent:
`mem ≈ (∏ factor_i)·block_voxels·dtype + block_voxels·dtype + reducer_temp`.
For 2×2×2 into a 128³ block at uint64: input region 256³×8B ≈ 134 MB. Tune via
block size. Read cost is optimal — each level is read exactly once, total ≈
`1.14·V` for isotropic 2×2×2.

The reducers must accumulate at *output* size, not upcast the whole input
region: `mean_downsample` sums via `reduce(dtype=...)` and only builds a
validity mask for ragged blocks, keeping a 512³-block downsample at ~3 GB. An
earlier version upcast the 1024³ input to float64 (data+mask ≈ 8.6 GB each,
~30 GB/task) and OOM-killed SLURM workers.

**Task granularity vs. sharding:** a zarr v3 shard is one file of many inner
chunks; two workers writing different inner chunks of the *same* shard race on a
read-modify-write with no cross-process lock. So for **sharded** outputs a task
owns a **whole shard** (read `factor×shard`, write all its inner chunks) — no
races, write-efficient, at the cost of shard-sized per-task memory. Unsharded
(S3) tasks stay at chunk granularity. This falls out of the backend reporting
the *array-level* chunk (= shard when sharding) as `.chunks`, so engine
block = shard automatically.

**Ragged edges:** when a level's dimension isn't divisible by the factor, the
last block reads a clipped region; reducers operate over the real voxels only
(mask-weighted mean; mode over valid entries), not assuming full windows.

---

## 7. Data types

| Kind | Raw | Working / stored |
|---|---|---|
| images | float32 | uint8 after scale+normalize (viewing); higher bit-depth kept for affinity prediction |
| probabilities / affinities | — | uint8, 1- or 3-channel (higher granularity possible) |
| segmentations | — | uint64 |

v1 dtype handling = **cast with optional clip/scale range**. Full
brightness/normalization pipelines come later.

---

## 8. Environment / packaging (one shared conda env)

**Current:** a single conda environment covering all three repos, each installed
editable (`pip install --no-deps -e .`). Superseded pixi; see the history below
for what pixi was solving and which of those constraints still bind.

- Env name `em-lib`, under `~/miniforge3/envs` (GPFS home → visible to Rusty
  workers; see the shared-visibility constraint below).
- Python is pinned to **3.12**, and this is not a preference: `vol2mesh` and
  `dvidutils` are built for py312 only on flyem-forge, with no py313 build.
- Those two are also **conda-only** — no PyPI equivalent exists, so pip can
  never resolve them and the conda env has to provide them. This is the reason
  the stack can't simply be a set of pip requirements.
- Runtime deps come from conda-forge, so editable installs use `--no-deps`;
  otherwise pip re-resolves conda-provided binaries (tensorstore, h5py) from
  PyPI and invites an ABI mismatch.
- Combined spec: `em-libraries/environment.yml` + `pypi_requirements.txt`.
  conda cannot jointly solve multiple env files (`conda env create` takes a
  single `-f`), and layering them via `conda env update` gives sequential solves
  where `flyem-forge` priority lands last — hence one combined file.

**Constraint that survives from the pixi design:** the env *must* live on shared
storage. `/home` is local XFS on the workstation, so SLURM workers on Rusty
can't see it; GPFS `/mnt/home` and ceph both work.

**Constraint that returned:** pixi detached its envs to
`/path/to/scratch/pixi-envs` specifically to dodge the GPFS home
**inode** quota. A conda env in `~/miniforge3/envs` puts that inode load back on
home. If the quota becomes tight, relocate with `conda create -p
/mnt/ceph/users/<user>/conda-envs/em-lib` rather than reaching back for pixi —
ceph does the same double duty (inode relief *and* worker visibility, no
conda-pack / rsync).

*Historical (pixi 0.62.2):* `detached-environments` → ceph, set project-local in
`.pixi/config.toml`, with the repo in home keeping only source + `pixi.toml` +
lockfile. Cache deliberately stayed on local `/home` (pixi's default
`~/.cache/rattler`, no inode quota, touched only at solve time on the driver).

---

## 9. Proposed module layout

The dask/SLURM orchestration substrate lives in a **separate shared package,
`em-blockrun`** (`../em-blockrun`, an editable sibling install): `start_dask`,
`block_map`, `Manifest`, `iter_blocks`/`Block`, `idempotent`, and the dask config
templates. It has no EM/volume deps and is reused by other projects (e.g. the
planned meshing package). `em_volume_tools` re-exports the common names for
convenience.

```
em_volume_tools/                # volume/EM-specific
├── volume.py        # Volume handle
├── meta.py          # VoxelMeta + coordinate/axis conversion
├── location.py      # local/s3/gs/kvstore normalization + subpath join
├── aws.py           # S3 credential bootstrap (tensorstore profile-provider workaround)
├── source_metadata.py  # read a source's coordinates + detect_backend (autodetect);
│                    #   describe/existing_levels compose those into a whole-volume
│                    #   picture (opens every level — the expensive tier)
├── backends/
│   ├── base.py      # ArrayBackend protocol + spec opener registry
│   ├── tensorstore.py  # zarr v3 (sharded/unsharded) + precomputed (canonical view)
│   ├── imagestack.py   # tifffile/imageio (read-only source)
│   ├── hdf5.py         # h5py (read)
│   ├── view.py         # CropBackend: read-only crop/pad view over a source
│   └── (zarr2 read via the tensorstore driver)
├── grid.py          # align a box to a block grid (outer/inner/nearest/origin), and
│                    #   the one alignment predicate `write` and `align-bbox` share
├── pyramid.py       # downsample schedule + type-aware reducers + OME transforms
├── profiles.py      # storage target profiles (§5) + create-spec builders
├── ngff.py          # OME-NGFF 0.5 group metadata (build/validate/write)
├── ops/
│   ├── _multiscale.py  # shared copy+pyramid loop; zarr3 & precomputed targets
│   ├── ingest.py    # image stack -> multiscale volume
│   ├── convert.py   # any source backend -> multiscale volume, whole or cropped
│   │                #   (crop view + mask view, both from backends/view.py)
│   ├── roi.py       # extract_roi: convert + the crop-AND-PAD policy
│   ├── create.py    # create_volume: an EMPTY volume (zarr3 or precomputed), specced
│   │                #   by hand or `like=` a reference (§10) — no source, no data
│   ├── write.py     # write_subvolume: one piece into one level, at a voxel offset
│   ├── rebuild.py   # rebuild_pyramid: levels above a trusted one, in place
│   ├── relabel.py   # one disjoint id range per occupied region
│   └── annotate.py  # WHERE a sparse volume's data is: occupied chunks -> maximal
│                    #   boxes. Analysis, not presentation — em-ngl turns these into
│                    #   a viewer layer; relabel/maskvalue/--sparse want the boxes
└── cli.py           # the `em-vol` command (below)

../em-blockrun/em_blockrun/      # shared dask/SLURM substrate (no EM deps)
├── dask_runner.py   # start_dask (LocalCluster / SLURMCluster from one config)
├── engine.py        # block_map (client.map + as_completed), iter_blocks, Block, idempotent
├── manifest.py      # Manifest (single-writer resume/intent log)
└── configs/         # dask config templates (local / slurm-gen / slurm-any)

em_volume_tools/cli.py          # the `em-vol` command:
                                #   info / convert / copy / downsample / progress /
                                #   create / write / to-hdf5 / align-bbox / relabel /
                                #   mask-by-value

../em-ngl/em_ngl/               # everything a VIEWER consumes (a separate package):
                                #   em-ngl gen / annotate / bboxes / parse / shaders
```

**There are three block grids, and they answer different questions** — `grid.py` does the
arithmetic, the CLI resolves which grid: the **write unit** (chunk, or shard when the
level is sharded) governs whether a write is a read-modify-write; the **pyramid's
cumulative factor** governs whether a crop's coarse levels land on the source's grid; and
the **per-axis LCM of a source and destination chunking** governs how many times a source
chunk is re-fetched (§ invariant 10, `plan_task_shape`). Aligning to the wrong one is
alignment that buys nothing — most sharply on a sharded level, where the inner read chunk
offers no protection against a partial-shard update.

`convert` and `copy` are one implementation under two defaulting policies (`cli.
_add_convert_args`): convert states the output it wants, copy takes the source's own
format, chunking, voxel size and image/segmentation type and errors where the source
records none. The policy is the whole subcommand — convert's `--kind image` default
turns a segmentation copy into averaged label ids with nothing raised, which is not a
documentation problem.

Cropping lives in `convert`, not beside it, because it has to wrap the **resolved**
`data_spec` — the one detection chose, `.gz` reader included (§ invariant 9) — and
because the offset shift that keeps the output in the source's frame needs the source's
own offset, which only the metadata resolution has. `roi.py` is what is left of
`extract_roi` once that moves: the crop-and-pad policy, and nothing else.

---

## 10. Create-then-write: several small pieces into one frame

`ingest`/`convert`/`extract_roi` all have the same shape — *one* source, materialized
wholesale (or to one box) and block-mapped over dask. The other real workflow does not fit that: a
handful of small subvolumes (image stacks, HDF5 files) that belong at known positions
inside one larger volume, arriving at different times. Forcing it through `convert`
would mean one volume per piece, or a source backend that stitches them, and neither
is what the coordinates actually say.

So it splits into two ops, and the split is the design:

- **`create_volume` lays out an empty volume.** Every level exists; no chunk data does.
  This is nearly free because an unwritten chunk reads back as the fill value — a whole
  empty pyramid is a few JSON documents regardless of its nominal size (for
  precomputed, literally one `info`).
- **`write_subvolume` places one piece into one level**, at a voxel offset.

Both formats are targets. `create` dispatches on the profile's format exactly as
`materialize_multiscale` does, and the difference is where multiscale metadata lives:
zarr gets an OME-NGFF group document written after the levels, while precomputed's
scales accumulate in one intrinsic `info` as each is created. That sharing has one
sharp edge — `delete_existing` may apply to **scale 0 only**, since every scale lives
under the same prefix and deleting on scale 1 would take scale 0 with it. Precomputed
also carries the origin as a per-scale integer `voxel_offset` rather than a
translation transform, and cannot shard (`precomputed_create_spec` has no shard
support; §"open items"), which is refused rather than ignored.

Decisions worth recording:

**With no format given, `like=` decides it too.** Mirroring a precomputed frame and
silently getting zarr is the sort of thing you discover when the viewer cannot open the
result. `profile_for` is shared with the CLI so `--format` cannot come to mean
something different there.

**`like=` copies the reference's level shapes verbatim, not its schedule.** The point
of a reference is that a voxel index means the same thing in both volumes. Recomputing
the pyramid from level 0 would usually agree — and when it did not (shapes are
ceil-divided, so one differing `min_dim`/`max_levels`/`factors` is enough) the two
volumes would be a voxel apart partway up, silently. Per-level voxel sizes are read
from the reference's own metadata for the same reason `read_level_voxel_sizes` exists:
never `2**level`, never a shape ratio.

**Creating over a volume of the *other* format is refused.** Each format writes only
its own marker, so making a precomputed volume where a zarr one lives leaves `info` and
`zarr.json` in one directory — and `detect_backend` checks `info` first, so the zarr
becomes unreachable through every path in this package while its chunks still occupy
the store. `--overwrite` cannot fix it either, and asymmetrically: precomputed's
`delete_existing` wipes the prefix (taking the zarr with it, unannounced), zarr's
deletes only its own level arrays and leaves a stale `info` in charge. So it refuses and
says to delete the destination. `describe` reports the condition too, for directories
already in it.

**The offset may come from the source.** Writers record where a subvolume came from
beside the array; HDF5 files routinely carry a `voxel_offset`. Re-typing that is tedious
and a chance to mistype a coordinate, so a backend may expose an optional
`stored_offset(name)` and `write` uses it when no offset is passed — an optional backend
capability, not an HDF5 branch in the op, even though HDF5 is the only backend that
answers today. **The axis order is asked for, not guessed**: `voxel_offset` is
precomputed's field name and precomputed means xyz, while a canonical `(z, y, x)` array
suggests zyx, and nothing in the file distinguishes them. Reversed, the piece lands
mirrored through the z=x diagonal (CLAUDE.md invariant 2) with nothing downstream able
to tell — so the default is zyx and the provenance and any reversal are printed.

**A batch is planned in full before any of it is written.** Planning resolves offsets
and checks bounds, dtype and level, and touches nothing; doing all of it up front means
a mistyped offset in the last file is caught while the volume is still clean rather than
after three pieces have landed. Writing then fails fast, because a failure there is a
storage problem and continuing would bury it. Pieces that overlap each other are
reported rather than refused — doing it deliberately is legitimate, but the result looks
identical either way, so silence would hide a mistyped offset that buries an earlier
piece.

**Writes are single-scale, deliberately.** Coarsening a patch is a separate decision
from placing it — averaging label ids invents ids, and a patch's correct coarse
representation may depend on what surrounds it, which the patch does not know. Run
`downsample` afterwards if the result needs a pyramid.

**Tiles are cut on the destination's global chunk grid.** A write that does not start
and end on that grid makes TensorStore read-modify-write the boundary chunks: it
fetches what is stored, overlays the new region, and writes the whole chunk back, so a
partially-covered chunk **keeps** the data already in it — including data a previous
`write` invocation put there. Nothing here implements that, but everything here depends
on it, and if it ever stopped being true the loss would be invisible (the piece just
written looks perfect; its neighbours are gone), so it is pinned by tests across a
plain chunk, a shard's inner chunk and a compressed_segmentation block — three
different merge paths inside TensorStore. Serially that is all fine; it is a lost
update the moment two pieces sharing a boundary chunk are written concurrently, with
nothing left behind to detect it. So
tiling anchors to the global grid (never to the region's own start, which would put
interior boundaries mid-chunk), and the alignment of the caller's own region is
reported rather than silently accepted. These ops run in the calling process, no dask,
which is also what keeps that hazard confined to what the user does across invocations.


## 11. Finding the data again: occupancy boxes

Create-then-write produces a volume that is mostly empty on purpose, and that makes it
hard to *look at*: twelve labeled boxes in an 11260×9000×13750 frame are needles.
`ops.annotate.labeled_regions` closes that loop by reporting one box per written region.

**This is analysis, and its callers are mostly not viewers.** `relabel` needs the regions
to give each one a disjoint id range, `mask-by-value` to know what to read, and
`downsample --sparse` to skip tasks whose input has no stored chunk. Turning the boxes into
something clickable is `em-ngl bboxes`, in a package above this one — which is why the
JSON half of this module moved out and only the analysis stayed.

**Occupancy comes from chunk presence, not from voxels.** TensorStore never persists a
chunk that is entirely the fill value, so on a sparse volume the set of stored objects
*is* the footprint. One listing per level answers the question with no reads at all,
and it cannot disagree with the data the way a record of what was written could. The
two formats key their chunks differently — precomputed's `x0-x1_y0-y1_z0-z1` (xyz, and
`.gz`-suffixed when CloudVolume wrote it) against zarr v3's `c/z/y/x` — so parsing is
per-format and the result is normalised to zyx cell indices immediately.

**Maximal boxes, not connected components.** This is the one decision with a wrong
answer that looks right. Two regions written face to face have chunk-aligned footprints
that touch, so components merge them into a single region spanning both *and the empty
corner between them* — measured on sample3's gt_v1, where it turned 12 GT chunks into
11. Growing a box greedily from the lexicographically smallest free cell, axis by axis,
recovers the blocks, and it can never emit a box containing an absent cell — which is
the property that matters, since a box is a claim that there is data there. Regions that
are genuinely contiguous *do* merge, correctly: nothing in the stored chunks
distinguishes that from one write of twice the size.

**Tightening defaults to the footprint's own level, and is an optimisation either way.**
Chunk-aligned boxes are blocky (a 256³ region rounds out to 384³ at 128³ chunks), so each
box is shrunk to its nonzero voxels by one read at `--tighten-level`. That defaulted to a
fixed level 2 at first, which made the common invocation quantize every bound to a coarse
voxel — the reason real extents came out 252 instead of 256, a surprise that had to be
explained rather than read off. Defaulting to `--level` instead means the boxes are exact
in the level-0 voxels they are reported in whenever the footprint came from level 0, and
the reads never cost more than the level the caller already chose. Raising it trades
exactness for a factor per level, which is the right knob when the occupied footprint is
large. When the level does not exist the level clamps *finer* rather than raising: a
single-level volume is the normal state of one `create` made and `write` filled, and
refusing to annotate exactly those would be absurd. Finer means more exact and only
slower, bounded by the occupied footprint — but it is reported, or an unexplained slow
run looks like a hang.

**Nothing here is written to the store** — this op is read-only, unlike every other write
path in the package.

**Relabelling reuses the region finder, and that is the point of §12.**

The reasoning about *presenting* these boxes — why the annotations must be local rather than
a precomputed source, and why the layer declares its own `outputDimensions` — moved to
em-ngl with the code. It is in that package's README and `em_ngl/layers.py`.


## 12. `relabel`: one id range per occupied region

Ground truth annotated chunk by chunk comes back with every chunk numbered from 1, so an
integer names a different cell in each. Measured on sample3's gt_v1 at scale 0: 3,832
label-instances across 12 regions but only **1,901 distinct ids, 508 of them used by more
than one region**. Nothing downstream can tell — it meshes into one body whose components
are scattered across the volume, correct for the label and useless as ground truth.

The operation is a walk over the occupied regions giving each its own range. Two
properties from §11 make it almost free of design:

- Regions come from **stored-chunk occupancy**, so they are pairwise disjoint by
  construction. No voxel belongs to two, so no id assignment can conflict.
- It is **serial by construction** — the next range begins where the last ended — so
  there is nothing to parallelise and nothing to coordinate. That is why this is a
  calling-process op like `create`/`write` rather than a block-mapped one.

**Boxes are deliberately not tightened**, which is the one place §11's default is wrong
here. A tightened box is the bounding box of nonzero voxels seen at a *coarse* level, and
mode downsampling can drop a stray voxel — so it can exclude scale-0 data that really is
there. Renumbering inside it would then leave that data holding an old id, silently
mixing two numbering schemes in one volume, and the volume would look fine. The
chunk-aligned box provably covers every stored chunk. It also sits on the destination's
chunk grid, so no write is a partial-chunk read-modify-write (§10's hazard never arises).

**The mapping is an output, not a side effect**, and is written by default to a path
derived from the destination. Once ids are overwritten it is the only route from a new id
back to the region and original label, so losing it by forgetting a flag is not a failure
mode worth having. It also carries each region's box, which makes it a complete record
without the volume.

**No default destination.** `--out` and `--in-place` are both reasonable and neither is
safe to assume: one publishes a second volume, the other overwrites the ids it is derived
from. `--out` is documented as preferred because a sparse copy is nearly free (324 chunk
objects for gt_v1) and it leaves the raw annotation intact, but the choice is stated
explicitly at the call site.

**Single-scale, for §10's reason.** Renumbering level 0 leaves the levels above holding
the old ids. Coarsening relabelled ids is a mode downsample like any other, but it is a
separate decision and a separate run, so this reports the stale levels and names the
`downsample` command rather than doing it. `--block-size N` numbers region *k* from
`N*k+1`, trading id density for being able to read the source region off the id, and
refuses rather than letting a region overflow into the next range — which would recreate
the exact collision the operation exists to remove.


## 13. Why the viewer side is a separate package

Everything that produces something a *viewer* consumes now lives in **em-ngl**: states,
links, annotation layers, shaders. This package writes volumes and does not know
neuroglancer exists.

The split was not about file size. It was forced by a layering smell that appeared as soon
as the viewer side grew: `ops/ngurl.py` had acquired a shader that reads `prop_conf_pre`,
plus the `body_pre`/`body_post` relationship semantics of a synapse source — concepts owned
by em-annotation, which sits *above* em-volume-tools. No import violated the layering, but
the knowledge was in the wrong repository, and every further viewer feature would have
dragged more of a higher layer's vocabulary downward. A package above both consumers is
where that belongs.

What made it cheap is that the seam already existed. `ops/annotate.py` was two modules
sharing a file, divided by a section banner: occupancy analysis (which `relabel`,
`mask-by-value` and `downsample --sparse` all need) and neuroglancer JSON emission (which
nothing here needs). The analysis stayed, the emission left, and `ops/ngurl.py` — a leaf
nothing but the CLI imported — moved whole.

`em-vol bboxes-json`, `annotate-json` and `ng-url-gen` became `em-ngl bboxes`, `annotate`
and `gen`. A clean break with no aliases, as with the earlier `em-seg-morpho` →
`em-morpho run` rename: an old invocation fails loudly rather than quietly doing something
slightly different.
