"""Coordinate metadata for volumes.

Canonical internal axis order is C-order spatial ``(z, y, x)``, optionally with a
leading ``c`` (channel) axis -> ``(c, z, y, x)``. ``VoxelMeta`` describes only the
*spatial* axes; the channel axis carries no physical size. Backends translate
to/from their native conventions (e.g. neuroglancer-precomputed's ``(x, y, z, c)``
with resolution listed in ``(x, y, z)``).

See docs/DESIGN.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

# Default spatial axis names, canonical (C-order) spatial ordering.
DEFAULT_AXES: tuple[str, ...] = ("z", "y", "x")


@dataclass(frozen=True)
class VoxelMeta:
    """Physical coordinate metadata for the spatial axes of a volume.

    Attributes
    ----------
    voxel_size:
        Physical size of one voxel along each spatial axis, in ``units``,
        in ``axes`` order (default ``(z, y, x)``).
    offset:
        Physical position of the ``(0, ...)`` voxel's origin (the OME-NGFF
        "translation"), in ``units``, same axis order. Defaults to zeros.
    units:
        Physical unit string for ``voxel_size``/``offset`` (e.g. ``"nm"``).
    axes:
        Names of the spatial axes, in order. Length must match ``voxel_size``.
    """

    voxel_size: tuple[float, ...]
    offset: tuple[float, ...] = field(default=())
    units: str = "nm"
    axes: tuple[str, ...] = DEFAULT_AXES

    def __post_init__(self) -> None:
        object.__setattr__(self, "voxel_size", tuple(float(v) for v in self.voxel_size))
        if not self.offset:
            object.__setattr__(self, "offset", (0.0,) * len(self.voxel_size))
        else:
            object.__setattr__(self, "offset", tuple(float(v) for v in self.offset))
        object.__setattr__(self, "axes", tuple(self.axes))
        n = len(self.voxel_size)
        if len(self.offset) != n:
            raise ValueError(f"offset has {len(self.offset)} entries, expected {n}")
        if len(self.axes) != n:
            raise ValueError(f"axes has {len(self.axes)} entries, expected {n}")

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.voxel_size)

    def downsampled(self, factors: Sequence[int]) -> "VoxelMeta":
        """Return metadata for a level downsampled by integer ``factors``.

        Uses corner alignment: physical ``offset`` is unchanged and
        ``voxel_size`` is multiplied by ``factors``. (Center-aligned variants,
        which shift the translation by half a source voxel, are a pyramid-policy
        concern handled in pyramid.py.)
        """
        if len(factors) != self.ndim:
            raise ValueError(f"factors has {len(factors)} entries, expected {self.ndim}")
        new_size = tuple(vs * f for vs, f in zip(self.voxel_size, factors))
        return replace(self, voxel_size=new_size)

    def with_axes(self, order: Sequence[str]) -> "VoxelMeta":
        """Reorder the spatial axes (and their sizes/offsets) to ``order``.

        Useful when translating to a backend with a different axis convention
        (e.g. precomputed ``(x, y, z)``).
        """
        order = tuple(order)
        if sorted(order) != sorted(self.axes):
            raise ValueError(f"cannot reorder {self.axes} to {order}")
        idx = [self.axes.index(a) for a in order]
        return replace(
            self,
            voxel_size=tuple(self.voxel_size[i] for i in idx),
            offset=tuple(self.offset[i] for i in idx),
            axes=order,
        )
