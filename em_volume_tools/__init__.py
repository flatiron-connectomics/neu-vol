"""em-volume-tools: chunked I/O, conversion, and multiscale generation for
large 3D EM volumes, orchestrated with dask (local workstation or Rusty/SLURM).

See docs/DESIGN.md for the architecture and docs/dask-slurm-rusty.md for the
orchestration cookbook.
"""

__version__ = "0.1.0"

from .meta import VoxelMeta
from .volume import Volume
from .engine import Block, iter_blocks, block_map, idempotent
from .dask_runner import start_dask
from .backends.base import ArrayBackend, open_backend, register_backend
from .profiles import StorageProfile, PROFILES, get_profile
from .ops import ingest_image_stack, convert, extract_roi

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
    "register_backend",
    "StorageProfile",
    "PROFILES",
    "get_profile",
    "ingest_image_stack",
    "convert",
    "extract_roi",
]
