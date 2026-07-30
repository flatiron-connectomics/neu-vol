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
├── introspect.py    # read source metadata + detect_backend (autodetect format)
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
│   └── roi.py       # extract_roi: crop / pad (-> multiscale)
└── cli.py           # (later)

../em-blockrun/em_blockrun/      # shared dask/SLURM substrate (no EM deps)
├── dask_runner.py   # start_dask (LocalCluster / SLURMCluster from one config)
├── engine.py        # block_map (client.map + as_completed), iter_blocks, Block, idempotent
├── manifest.py      # Manifest (single-writer resume/intent log)
└── configs/         # dask config templates (local / slurm-gen / slurm-any)

examples/run_convert_slurm.py   # Rusty/SLURM driver
```
