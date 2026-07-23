"""Backend registry.

Importing this package registers all concrete backends with the spec opener, so
``open_backend(spec)`` works for any supported format.
"""

from __future__ import annotations

from .base import ArrayBackend, Region, open_backend, register_backend

# Import concrete backends for their self-registration side effects.
from . import tensorstore, imagestack, hdf5, view  # noqa: E402,F401

__all__ = [
    "ArrayBackend",
    "Region",
    "open_backend",
    "register_backend",
]
