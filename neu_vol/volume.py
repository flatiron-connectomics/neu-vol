"""The :class:`Volume` handle: a lazy view over a backend store plus metadata.

A ``Volume`` holds no data. It pairs an :class:`~.backends.base.ArrayBackend`
(region I/O, shape/dtype/chunks) with a :class:`~.meta.VoxelMeta` (physical
coordinates) and channel semantics.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .backends.base import ArrayBackend, Region, open_backend
from .meta import VoxelMeta


class Volume:
    """A chunked 3D (optionally multi-channel) volume with physical metadata.

    Canonical axis order is ``(z, y, x)`` or ``(c, z, y, x)``. ``meta`` describes
    the spatial axes only.
    """

    def __init__(
        self,
        backend: ArrayBackend,
        meta: VoxelMeta,
        *,
        has_channels: bool = False,
    ):
        self.backend = backend
        self.meta = meta
        self.has_channels = has_channels
        expected = meta.ndim + (1 if has_channels else 0)
        if len(backend.shape) != expected:
            raise ValueError(
                f"backend shape {backend.shape} has {len(backend.shape)} dims; "
                f"expected {expected} for {meta.ndim} spatial axes"
                f"{' + channel' if has_channels else ''}"
            )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], meta: VoxelMeta,
                  *, has_channels: bool = False) -> "Volume":
        """Open a volume from a serializable backend spec (dask-friendly)."""
        return cls(open_backend(spec), meta, has_channels=has_channels)

    def to_spec(self) -> dict[str, Any]:
        """Serializable description: backend spec + metadata, for shipping to workers."""
        return {
            "spec": self.backend.to_spec(),
            "meta": {
                "voxel_size": self.meta.voxel_size,
                "offset": self.meta.offset,
                "units": self.meta.units,
                "axes": self.meta.axes,
            },
            "has_channels": self.has_channels,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Volume":
        """Inverse of :meth:`to_spec`."""
        m = d["meta"]
        meta = VoxelMeta(
            voxel_size=tuple(m["voxel_size"]),
            offset=tuple(m["offset"]),
            units=m["units"],
            axes=tuple(m["axes"]),
        )
        return cls.from_spec(d["spec"], meta, has_channels=d["has_channels"])

    # -- shape / dtype -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.backend.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.backend.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(self.backend.chunks)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial axes only (drops a leading channel axis)."""
        return self.shape[1:] if self.has_channels else self.shape

    @property
    def num_channels(self) -> int:
        return self.shape[0] if self.has_channels else 1

    # -- region I/O --------------------------------------------------------
    def read(self, region: Region) -> np.ndarray:
        return self.backend.read_region(region)

    def write(self, region: Region, data: np.ndarray) -> None:
        self.backend.write_region(region, data)

    def __getitem__(self, index: Region) -> np.ndarray:
        return self.backend.read_region(_as_region(index))

    def __repr__(self) -> str:
        return (
            f"Volume(shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}, "
            f"voxel_size={self.meta.voxel_size} {self.meta.units})"
        )


def _as_region(index: Any) -> Region:
    """Normalize a ``__getitem__`` index into a tuple of slices."""
    if not isinstance(index, tuple):
        index = (index,)
    if not all(isinstance(s, slice) for s in index):
        raise TypeError("Volume indexing supports slices only (e.g. vol[z0:z1, y0:y1, x0:x1])")
    return index
