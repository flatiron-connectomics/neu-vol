"""Backend registry.

Importing this package registers all concrete backends with the spec opener, so
``open_backend(spec)`` works for any supported format.
"""

from __future__ import annotations

from .base import ArrayBackend, Region, open_backend, register_backend

# Import concrete backends for their self-registration side effects. cloudvolume is
# included even though `cloud-volume` may be absent: the module imports fine without
# it and raises an explanatory ImportError only when a volume actually needs it, so
# the failure lands at open time with guidance rather than at import time for
# everyone.
from . import tensorstore, imagestack, hdf5, view, cloudvolume  # noqa: E402,F401

__all__ = [
    "ArrayBackend",
    "Region",
    "open_backend",
    "register_backend",
]
