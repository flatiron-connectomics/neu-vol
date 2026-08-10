"""em-volume-tools: chunked I/O, conversion, and multiscale generation for
large 3D EM volumes, orchestrated with dask (local workstation or Rusty/SLURM).

See docs/DESIGN.md for the architecture and docs/dask-slurm-rusty.md for the
orchestration cookbook.
"""

__version__ = "0.1.0"

from .meta import VoxelMeta
from .volume import Volume
# Orchestration substrate now lives in the shared em-blockrun package; re-export
# the common names here for convenience / backward compatibility.
from em_blockrun import Block, iter_blocks, block_map, idempotent, start_dask
from .backends.base import (ArrayBackend, clear_backend_cache, open_backend,
                            register_backend)
from .retry import is_transient, with_retry
from .profiles import StorageProfile, PROFILES, get_profile
from .ops import (convert, create_volume, extract_roi, ingest_image_stack,
                  plan_subvolume_write, plan_volume, rebuild_pyramid,
                  write_subvolume, write_subvolumes)
from .source_metadata import describe, existing_levels, level_spec
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
    "describe",
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
