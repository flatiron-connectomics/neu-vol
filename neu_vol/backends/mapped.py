"""A source read through a lookup table: ``table[source]``. See ``neu_vol/mapped.py``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .. import mapped as _mapped
from .base import Region, open_backend, register_backend


class MappedBackend:
    """Read-only view applying a dense ``table[id]`` to every voxel of a source.

    The output dtype is the source's when every table value fits in it — a 2M-supervoxel
    uint32 volume maps to agglomerate ids that fit in uint32, and widening it would double
    every downstream byte for nothing — and uint64 otherwise.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._source = open_backend(dict(spec["source"]))
        table = _mapped.load_table(str(spec["table"]))
        src = np.dtype(self._source.dtype)
        if src.kind != "u":
            # Signed ids would index from the END of the table — a negative label reading
            # as some real agglomerate, silently.
            raise ValueError(f"cannot map a {src} volume: a lookup table indexes "
                             f"unsigned labels")
        top = int(table.max()) if table.size else 0
        self._dtype = src if top <= np.iinfo(src).max else np.dtype("uint64")
        self._table = table.astype(self._dtype, copy=False)

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "MappedBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._source.chunks

    @property
    def read_chunks(self) -> tuple[int, ...]:
        # The unit actually fetched from a sharded source; a lookup does not change it.
        return self._source.read_chunks

    @property
    def voxel_size(self):
        return getattr(self._source, "voxel_size", None)

    def read_region(self, region: Region) -> np.ndarray:
        data = self._source.read_region(region)
        if data.size and int(data.max()) >= self._table.size:
            # Not clipped or passed through: an id past the table means the table and the
            # volume are not the same segmentation, and every answer would be wrong.
            raise ValueError(
                f"{self._spec.get('descriptor') or self._spec['table']}: the volume holds "
                f"id {int(data.max())}, but the table covers ids 0..{self._table.size - 1}. "
                f"The table was fetched for a different segmentation than this volume.")
        return self._table[data]

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("mapped views are read-only; convert one to write a volume")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(_mapped.TAG, MappedBackend.open)
