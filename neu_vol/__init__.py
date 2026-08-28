"""neu-vol: chunked I/O, conversion, and multiscale generation for
large 3D EM volumes, orchestrated with dask (local workstation or SLURM cluster).

See the README for what it does, and blockrun's docs/dask-slurm.md for the
orchestration background.
"""

__version__ = "0.1.0"

from .meta import VoxelMeta
from .volume import Volume
# NOTE: `BBox` and the grid arithmetic moved to the shared `neu_lib` package and are
# deliberately NOT re-exported here. Import them from where they live — a second
# binding for the same name is the patch-point hazard the dvid modules already
# document, and naming `neu_lib` is what keeps the layering legible in a diff.
# Orchestration substrate now lives in the shared blockrun package; re-export
# the common names here for convenience / backward compatibility.
from blockrun import Block, iter_blocks, block_map, idempotent
from .backends.base import (ArrayBackend, clear_backend_cache, open_backend,
                            register_backend)
# The one format-specific opener, and the reason it earns a top-level name: an HDF5 file
# is a container, so "open this path" is ambiguous in a way it is not for a volume, and
# `open_backend` deliberately stays spec-only (see `open_hdf5`'s own docstring). Costs
# nothing at import: `backends/__init__` already imports this module, and h5py is loaded
# lazily inside it.
from .backends.hdf5 import open_hdf5
from .retry import is_transient, with_retry
from .profiles import StorageProfile, PROFILES, get_profile
from .ops import (convert, create_volume, extract_roi, ingest_image_stack,
                  mask_values, pack_hdf5, plan_subvolume_write, plan_volume,
                  rebuild_pyramid, write_subvolume, write_subvolumes)
from .piece import read_piece
from .source_metadata import describe, existing_levels, level_spec
# Per-level shape, voxel size and origin, read from the source's own metadata. Came
# from neu-morpho, which had meant neu-draw importing a meshing package to learn a
# volume's voxel size.
from .scales import describe_scales, read_scales, scale_spec
# Location handling + byte/JSON I/O that works the same for local paths and
# object stores, so consumers never branch on the destination.
from .location import (exists, is_local, local_path, read_bytes, read_json,
                       to_kvstore, write_bytes, write_json)

__all__ = [
    "__version__",
    "VoxelMeta",
    "Volume",
    "Block",
    "iter_blocks",
    "block_map",
    "idempotent",
    "start_dask",
    "ArrayBackend",
    "open_backend",
    "clear_backend_cache",
    "register_backend",
    "is_transient",
    "with_retry",
    "StorageProfile",
    "PROFILES",
    "get_profile",
    "ingest_image_stack",
    "convert",
    "extract_roi",
    "rebuild_pyramid",
    "create_volume",
    "plan_volume",
    "write_subvolume",
    "write_subvolumes",
    "plan_subvolume_write",
    "pack_hdf5",
    "open_hdf5",
    "read_piece",
    "mask_values",
    "describe",
    "describe_scales",
    "read_scales",
    "scale_spec",
    "existing_levels",
    "level_spec",
    "to_kvstore",
    "is_local",
    "local_path",
    "read_bytes",
    "write_bytes",
    "read_json",
    "write_json",
    "exists",
]


# Re-exported lazily, for the reason blockrun defers it: resolving `start_dask` at
# import time pulls in `dask.distributed`, which is ~1 s and which no read-only op needs.
# `from neu_vol import start_dask` still works; it just resolves on first use.
def __getattr__(name: str):
    if name == "start_dask":
        from blockrun import start_dask

        return start_dask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
