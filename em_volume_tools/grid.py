"""Aligning a box to a block grid: the arithmetic, per axis, and nothing else.

A box that does not sit on a grid costs something different depending on *which* grid it
misses, and there are three in play in this package:

* the **write unit** — the chunk, or the shard where a level is sharded. A partial write
  is a read-modify-write: it keeps the chunk's existing data, but two concurrent partial
  writes into one object lose one of them, silently.
* the **pyramid's cumulative factor**. A crop whose origin misses it has coarse levels
  sitting on their own grid, each level's ``voxel_offset`` rounding to it. Level 0 stays
  exact, so nothing looks wrong.
* the **per-axis LCM of a source and destination chunking**, which is what decides how
  many times a source chunk is re-fetched (see ``ops/_multiscale.plan_task_shape``).

This module knows about none of them. It takes a block shape and does integer arithmetic
on each axis independently, because real grids are anisotropic — ``(1, 2, 2)`` pyramids
and ``(128, 2048, 2048)`` chunkings are both ordinary. Resolving *which* grid a volume
implies is the CLI's job.

**Boxes are half-open**, ``[lo, hi)``, as everywhere else here: a ``hi`` already on a
boundary must stay where it is, which is the off-by-one that makes ``outer`` grow a
correctly-sized box by a whole block.
"""

from __future__ import annotations

import math
from typing import Sequence

#: How each end of the box moves. Rounding is per axis and independent.
MODES = ("outer", "inner", "nearest", "origin")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _nearest(value: int, block: int) -> int:
    """``value`` rounded to the closest multiple of ``block``, halves upward.

    Not :func:`round`, which rounds halves to even — so 192 with a 128 block would go to
    128 while 320 went to 384, and the same box would move differently depending on where
    it sat. Predictability matters more than symmetry here.
    """
    return math.floor(value / block + 0.5) * block


def align_box(lo: Sequence[int], hi: Sequence[int], block: Sequence[int],
              mode: str = "outer") -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Align the half-open box ``[lo, hi)`` to the ``block`` grid, per axis.

    ``outer`` floors ``lo`` and ceils ``hi``: never loses a requested voxel, and is the
    only mode that cannot fail. ``inner`` does the reverse, so the result is contained in
    the request. ``nearest`` rounds both ends. ``origin`` aligns ``lo`` (to the nearest
    multiple) and keeps the **extent** exactly, which is what a fixed-size crop needs —
    its ``hi`` is then generally not on the grid, and that is the trade it makes.

    Raises ``ValueError`` if the result would be empty on some axis: ``inner`` on a box
    spanning no whole block, or ``nearest`` on a box whose ends round together. Both mean
    the request is small relative to the grid, which the caller has to know about rather
    than receive as an empty box.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {', '.join(MODES)}")
    if not (len(lo) == len(hi) == len(block)):
        raise ValueError(f"rank mismatch: lo={tuple(lo)} hi={tuple(hi)} "
                         f"block={tuple(block)}")
    if any(b <= 0 for b in block):
        raise ValueError(f"block must be positive on every axis, got {tuple(block)}")

    out_lo, out_hi = [], []
    for axis, (a, b, k) in enumerate(zip(lo, hi, block)):
        a, b, k = int(a), int(b), int(k)
        if mode == "outer":
            na, nb = (a // k) * k, _ceil_div(b, k) * k
        elif mode == "inner":
            na, nb = _ceil_div(a, k) * k, (b // k) * k
        elif mode == "nearest":
            na, nb = _nearest(a, k), _nearest(b, k)
        else:                                            # origin: keep the extent
            na = _nearest(a, k)
            nb = na + (b - a)
        if nb <= na:
            raise ValueError(
                f"aligning [{a}, {b}) to a {k}-voxel grid with mode {mode!r} leaves "
                f"axis {axis} empty ({na}:{nb}): the box spans no whole block there. "
                f"Use mode 'outer', which only ever grows a box.")
        out_lo.append(na)
        out_hi.append(nb)
    return tuple(out_lo), tuple(out_hi)


def clamp_box(lo: Sequence[int], hi: Sequence[int],
              extent: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Trim the box to ``[0, extent)`` per axis. Separate from aligning on purpose.

    Clamping can put an edge back off the grid, and that edge is nonetheless fine — see
    :func:`misaligned_axes`. Keeping the two steps apart is what lets a caller report
    "grew to align, then trimmed to the volume" instead of one unexplained box.
    """
    return (tuple(max(0, int(a)) for a in lo),
            tuple(min(int(b), int(e)) for b, e in zip(hi, extent)))


def misaligned_axes(lo: Sequence[int], hi: Sequence[int], block: Sequence[int],
                    extent: Sequence[int] | None = None) -> list[int]:
    """Axes where an edge of ``[lo, hi)`` falls inside a block.

    **An edge that coincides with the end of the volume is aligned by definition**: the
    final block is partial in the volume itself, so there is no neighbouring data in it
    to read-modify-write against. Without ``extent`` that exemption cannot apply, and a
    box ending at a partial final block reads as misaligned.
    """
    bad = []
    for axis, (a, b, k) in enumerate(zip(lo, hi, block)):
        at_end = extent is not None and int(b) == int(extent[axis])
        if int(a) % int(k) or (int(b) % int(k) and not at_end):
            bad.append(axis)
    return bad


def lcm_grid(*grids: Sequence[int]) -> tuple[int, ...]:
    """Per-axis least common multiple of several block shapes.

    A box on this grid is on all of them at once — the write unit *and* the pyramid
    factor, say. Per axis, never a single number: the LCM of ``(128, 128, 128)`` and
    ``(1, 2, 2)`` is ``(128, 128, 128)``, and collapsing that to a scalar would align z
    to 128 for no reason.
    """
    if not grids:
        raise ValueError("lcm_grid needs at least one grid")
    if len({len(g) for g in grids}) != 1:
        raise ValueError(f"rank mismatch between grids: {[tuple(g) for g in grids]}")
    return tuple(math.lcm(*(int(g[a]) for g in grids)) for a in range(len(grids[0])))
