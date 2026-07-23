"""HDF5-backed reader for chunk shards (h5py).

Reads/assembles volumes stored as chunks across HDF5 files (see docs/DESIGN.md
§2). Read-focused for v1.

STUB: interface fixed here; implementation lands in step 3.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "hdf5"


class HDF5Backend:
    """View over a dataset in an HDF5 file (single file for v1)."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "HDF5Backend":
        raise NotImplementedError("HDF5Backend.open — step 3")

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    def chunks(self) -> tuple[int, ...]:
        raise NotImplementedError

    def read_region(self, region: Region) -> np.ndarray:
        raise NotImplementedError

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise NotImplementedError("HDF5 writing — later")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, HDF5Backend.open)
