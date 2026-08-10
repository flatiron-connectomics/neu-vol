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
  the levels on disk, which is how you check.
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
├── pyramid.py       # downsample schedule + type-aware reducers + OME transforms
├── profiles.py      # storage target profiles (§5) + create-spec builders
├── ngff.py          # OME-NGFF 0.5 group metadata (build/validate/write)
├── ops/
│   ├── _multiscale.py  # shared copy+pyramid loop; zarr3 & precomputed targets
│   ├── ingest.py    # image stack -> multiscale volume
│   ├── convert.py   # any source backend -> multiscale volume
│   ├── roi.py       # extract_roi: crop / pad (-> multiscale)
│   ├── create.py    # create_volume: an EMPTY volume (zarr3 or precomputed), specced
│   │                #   by hand or `like=` a reference (§10) — no source, no data
│   └── write.py     # write_subvolume: one piece into one level, at a voxel offset
└── cli.py           # (later)

../em-blockrun/em_blockrun/      # shared dask/SLURM substrate (no EM deps)
├── dask_runner.py   # start_dask (LocalCluster / SLURMCluster from one config)
├── engine.py        # block_map (client.map + as_completed), iter_blocks, Block, idempotent
├── manifest.py      # Manifest (single-writer resume/intent log)
└── configs/         # dask config templates (local / slurm-gen / slurm-any)

em_volume_tools/cli.py          # the `em-vol` command:
                                #   info / convert / downsample / progress / create / write
```

---

## 10. Create-then-write: several small pieces into one frame

`ingest`/`convert`/`extract_roi` all have the same shape — *one* source, materialized
wholesale, block-mapped over dask. The other real workflow does not fit that: a
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


## 11. Finding the data again: `annotations`

Create-then-write produces a volume that is mostly empty on purpose, and that makes it
hard to *look at*: twelve labeled boxes in an 11260×9000×13750 frame are needles.
`em-vol annotations` closes that loop by emitting a neuroglancer annotation layer with
one box per written region, so the viewer gets a list to click through.

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

**Tightening is an optimisation, so a missing level is not an error.** Chunk-aligned
boxes are blocky (a 256³ region rounds out to 384³ at 128³ chunks), so each box is
shrunk to its nonzero voxels by one read at `--tighten-level` — nearly free, because a
384-voxel box is 96 voxels at 32 nm, at the cost of quantizing the bound to one coarse
voxel. When that level does not exist the level clamps *finer* rather than raising: a
single-level volume is the normal state of one `create` made and `write` filled, and
refusing to annotate exactly those would be absurd. Finer means more exact and only
slower, bounded by the occupied footprint — but it is reported, or an unexplained slow
run looks like a hang.

**The annotations are local, and that is forced by the viewer, not chosen.**
Neuroglancer builds its annotation list by iterating the layer's source, and
`MultiscaleAnnotationSource` — the class behind every *precomputed* annotation source —
defines `[Symbol.iterator]` as an empty generator. So a precomputed annotation layer
renders in the viewport and contributes **no rows** to the Annotations tab: nothing to
click, and `[`/`]` do not step. Since the whole purpose here is navigation, the layer is
emitted as inline `local://annotations` in the state instead. Two consequences worth
recording: nothing is written to the store (this op is read-only, unlike every other
write path in the package), and a precomputed annotation layer would not have helped
anyway, because unlike meshes and skeletons annotations cannot be named from a volume's
own `info` — a viewer must add them as a separate source regardless, so a state file is
the distribution unit either way.

**The layer declares its own `outputDimensions`.** Annotation coordinates are read in
the layer's frame rather than the viewer's, which is what lets one layer be pasted into
any state of the same volume. That needs a real unit: precomputed always records nm and
OME-NGFF spells its units out, but an unrecognised one leaves the layer unitless and
misaligned, so it warns and points at `--voxel-size` rather than inventing a scale — a
wrong scale would place every box somewhere plausible and wrong.
