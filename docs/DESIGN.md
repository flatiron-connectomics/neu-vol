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

---

## 4. Block-map engine + dask

`engine.py` iterates over **output chunks**, computes the input region each
needs, and emits one task per output block. Tasks are the `items` fed to the
cookbook's `db.from_sequence(items).map(fn)` under `start_dask` (local or SLURM
via one YAML config).

Rules (from the cookbook gotchas):
- **Idempotent tasks** — skip an output block already written; enables
  resume-by-relaunch and `walltime` growth across restarts.
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

## 8. Environment / packaging (pixi + ceph relocation)

Use **pixi**, but relocate its heavy, high-inode storage off the GPFS home
(inode-quota constrained):

- `detached-environments` → `/path/to/scratch/pixi-envs` (set
  project-local in `.pixi/config.toml`; envs live on ceph, repo in home keeps
  only source + `pixi.toml` + lockfile).
- **Cache stays on local `/home`** (pixi's default `~/.cache/rattler` is already
  `/home` = local XFS, *no* inode quota). No reason to move it to ceph — that
  would only consume ceph inodes. It's only touched at solve/install time on the
  driver.
- The env *must* be on shared ceph anyway: `/home` is local to the workstation,
  so SLURM workers on Rusty can't see it. ceph does double duty — dodging the
  GPFS-home inode quota *and* making the env importable by workers (no
  conda-pack / rsync).
- Verified with pixi 0.62.2: `pixi config set --local detached-environments <path>`.

---

## 9. Proposed module layout

```
em_volume_tools/                # v1 IMPLEMENTED (45 tests)
├── volume.py        # Volume handle
├── meta.py          # VoxelMeta + coordinate/axis conversion
├── backends/
│   ├── base.py      # ArrayBackend protocol + spec opener registry
│   ├── tensorstore.py  # zarr v3 (sharded/unsharded) + precomputed (canonical view)
│   ├── imagestack.py   # tifffile/imageio (read-only source)
│   ├── hdf5.py         # h5py (read)
│   └── view.py         # CropBackend: read-only crop/pad view over a source
├── engine.py        # block-map engine (output-block iteration, idempotent tasks)
├── dask_runner.py   # start_dask (from cookbook)
├── pyramid.py       # downsample schedule + type-aware reducers + OME transforms
├── profiles.py      # storage target profiles (§5) + create-spec builders
├── ngff.py          # OME-NGFF 0.5 group metadata (build/validate/write)
├── ops/
│   ├── _multiscale.py  # shared copy+pyramid loop; zarr3 & precomputed targets
│   ├── ingest.py    # image stack -> multiscale volume
│   ├── convert.py   # any source backend -> multiscale volume
│   └── roi.py       # extract_roi: crop / pad (-> multiscale)
└── cli.py           # (later)
examples/run_convert_slurm.py   # Rusty/SLURM driver
```
