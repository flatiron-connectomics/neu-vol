"""Backend registry.

Importing this package registers all concrete backends with the spec opener, so
``open_backend(spec)`` works for any supported format.
"""

from __future__ import annotations

from .base import ArrayBackend, Region, open_backend, register_backend

# Import concrete backends for their self-registration side effects. cloudvolume and
# dvid are included even though `cloud-volume` / `neuclease` may be absent: both
# modules import fine without them and raise an explanatory ImportError only when a
# volume actually needs one, so the failure lands at open time with guidance rather
# than at import time for everyone. For dvid that is also a startup-latency contract —
# `import neuclease.dvid` costs ~9 s, which nothing that merely lists backends should
# pay (see tests/test_cli_contract.py).
from . import tensorstore, imagestack, hdf5, view, cloudvolume, dvid  # noqa: E402,F401

__all__ = [
    "ArrayBackend",
    "Region",
    "open_backend",
    "register_backend",
]
