"""Read-only view backends that wrap another backend.

``CropBackend`` presents a shifted/resized window over a source backend, padding
out-of-bounds regions with a fill value. Because it satisfies the ArrayBackend
protocol and opens from a spec, the materialize engine can crop + pad + pyramid a
region exactly as it would any other source (see ops/roi.py). Read-only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, open_backend, register_backend

TAG = "crop"


class CropBackend:
    """A window ``[origin, origin+shape)`` over a source backend (canonical order).

    ``origin`` may be negative and ``shape`` may extend past the source; regions
    outside the source are filled with ``pad_value``.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._source_spec = dict(spec["source"])
        self._origin = tuple(int(o) for o in spec["origin"])
        self._shape = tuple(int(s) for s in spec["shape"])
        self._pad_value = spec.get("pad_value", 0)
        self._source = open_backend(self._source_spec)
        if not (len(self._origin) == len(self._shape) == len(self._source.shape)):
            raise ValueError("origin/shape/source rank mismatch")

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "CropBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._source.chunks

    def read_region(self, region: Region) -> np.ndarray:
        out_shape = tuple(s.stop - s.start for s in region)
        out = np.full(out_shape, self._pad_value, dtype=self.dtype)
        src_slices, dst_slices = [], []
        for a, s in enumerate(region):
            src_start = s.start + self._origin[a]
            src_stop = s.stop + self._origin[a]
            cs = max(src_start, 0)
            ce = min(src_stop, self._source.shape[a])
            if ce <= cs:
                return out  # this axis is fully out of bounds -> all padding
            d0 = cs - src_start
            src_slices.append(slice(cs, ce))
            dst_slices.append(slice(d0, d0 + (ce - cs)))
        out[tuple(dst_slices)] = self._source.read_region(tuple(src_slices))
        return out

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("crop views are read-only")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, CropBackend.open)
