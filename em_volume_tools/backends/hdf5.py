"""HDF5-backed reader (h5py).

Reads a dataset from an HDF5 file as a volume (docs/DESIGN.md §2). Read-focused
for v1; the dataset is assumed to be in canonical ``(z, y, x)`` / ``(c, z, y, x)``
order. Each backend instance holds an open file handle, and is reopened from its
spec on each dask worker (specs are picklable; open handles are not).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "hdf5"


class HDF5Backend:
    """View over a single dataset in an HDF5 file."""

    def __init__(self, spec: Mapping[str, Any]):
        import h5py

        self._spec = dict(spec)
        self._path = str(spec["path"])
        self._dataset = str(spec.get("dataset", "/data"))
        self._file = h5py.File(self._path, "r")
        if self._dataset not in self._file:
            self._file.close()
            raise KeyError(f"dataset {self._dataset!r} not found in {self._path}")
        self._dset = self._file[self._dataset]

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "HDF5Backend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._dset.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dset.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        # HDF5 storage chunks if chunked, else treat the whole array as one chunk.
        return tuple(self._dset.chunks) if self._dset.chunks else self.shape

    def read_region(self, region: Region) -> np.ndarray:
        return np.asarray(self._dset[region])

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise NotImplementedError("HDF5 writing not supported yet")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, HDF5Backend.open)
