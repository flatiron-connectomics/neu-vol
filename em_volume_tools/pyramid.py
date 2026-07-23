"""Multiscale pyramid: downsample schedule, type-aware reducers, OME transforms.

Strict level-by-level coarsening (docs/DESIGN.md §6-6b): each level is produced
from the previous (finer) level on disk. This module supplies the pure pieces —
the *schedule* (per-level integer factors), the *reducers* (mean for
image/probability, mode for label-preserving segmentation), and the
*coordinate transforms* (center-aligned scale/translation) for OME-NGFF metadata.
The orchestration lives in ops/ (block-map per level).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def _downsampled_shape(shape: Sequence[int], factor: Sequence[int]) -> tuple[int, ...]:
    """Ceil-division shape after downsampling (ragged edges kept, docs §6b)."""
    return tuple(-(-s // f) for s, f in zip(shape, factor))


def auto_factor(voxel_size: Sequence[float], *, isotropy_threshold: float = 2.0) -> tuple[int, ...]:
    """Per-axis factor for one level, coarsening the finest axes toward isotropy.

    Downsample an axis by 2 iff its voxel size is within ``isotropy_threshold`` of
    the finest axis; otherwise leave it (factor 1). Isotropic input -> all 2s.
    """
    m = min(voxel_size)
    return tuple(2 if vs < isotropy_threshold * m else 1 for vs in voxel_size)


def downsample_schedule(
    shape: Sequence[int],
    voxel_size: Sequence[float],
    *,
    factors: Sequence[Sequence[int]] | None = None,
    max_levels: int = 8,
    min_dim: int = 128,
    isotropy_threshold: float = 2.0,
) -> list[tuple[int, ...]]:
    """Return per-level relative downsample factors for levels 1..L (level 0 excluded).

    If ``factors`` is given it is used verbatim. Otherwise factors are derived
    with :func:`auto_factor`, stopping when the volume's largest spatial
    dimension is <= ``min_dim``, when no axis can be coarsened, or at
    ``max_levels``.
    """
    if factors is not None:
        return [tuple(int(x) for x in f) for f in factors]

    schedule: list[tuple[int, ...]] = []
    cur_shape = tuple(int(s) for s in shape)
    cur_vox = tuple(float(v) for v in voxel_size)
    while len(schedule) < max_levels:
        if max(cur_shape) <= min_dim:
            break
        f = auto_factor(cur_vox, isotropy_threshold=isotropy_threshold)
        if all(x == 1 for x in f):
            break
        schedule.append(f)
        cur_shape = _downsampled_shape(cur_shape, f)
        cur_vox = tuple(v * x for v, x in zip(cur_vox, f))
    return schedule


def cumulative_factors(schedule: Sequence[Sequence[int]], ndim: int) -> list[tuple[int, ...]]:
    """Cumulative factors relative to level 0, for levels 0..L (level 0 = all 1s).

    ``ndim`` is required so the level-0 identity is correct even when
    ``schedule`` is empty (single-scale output).
    """
    cum = [tuple([1] * ndim)]
    for f in schedule:
        cum.append(tuple(c * x for c, x in zip(cum[-1], f)))
    return cum


# --------------------------------------------------------------------------- #
# Reducers  (operate on one already-loaded input region -> one output block)
# --------------------------------------------------------------------------- #
def _split_windows(a: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Reshape (C-contiguous) so each axis splits into (out, factor)."""
    new_shape: list[int] = []
    for s, f in zip(a.shape, factors):
        new_shape += [s // f, f]
    return a.reshape(new_shape)


def mean_downsample(arr: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Mask-weighted mean downsample (anti-aliasing); ragged edges use real voxels only.

    Integer dtypes are rounded; float dtypes preserved. For images/probabilities.
    """
    factors = tuple(int(f) for f in factors)
    pad = [(0, (-s) % f) for s, f in zip(arr.shape, factors)]
    data = np.pad(arr.astype(np.float64), pad, mode="constant", constant_values=0.0)
    mask = np.pad(np.ones(arr.shape, np.float64), pad, mode="constant", constant_values=0.0)
    win = tuple(range(1, 2 * arr.ndim, 2))
    total = _split_windows(data, factors).sum(axis=win)
    count = _split_windows(mask, factors).sum(axis=win)
    res = total / count
    if np.issubdtype(arr.dtype, np.integer):
        return np.rint(res).astype(arr.dtype)
    return res.astype(arr.dtype)


def mode_downsample(arr: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Label-preserving mode downsample; ties broken to the smallest label.

    For segmentations (never interpolate IDs). Vectorized for small windows
    (prod(factors) small, e.g. 8 for 2x2x2); chunk the input for very large
    windows if memory-bound.
    """
    factors = tuple(int(f) for f in factors)
    ndim = arr.ndim
    out_shape = _downsampled_shape(arr.shape, factors)
    pad = [(0, (-s) % f) for s, f in zip(arr.shape, factors)]
    data = np.pad(arr, pad, mode="constant", constant_values=0)
    valid = np.pad(np.ones(arr.shape, bool), pad, mode="constant", constant_values=False)

    out_axes = tuple(range(0, 2 * ndim, 2))
    win_axes = tuple(range(1, 2 * ndim, 2))
    W = int(np.prod(factors))
    vals = _split_windows(data, factors).transpose(out_axes + win_axes).reshape(-1, W)
    val = _split_windows(valid, factors).transpose(out_axes + win_axes).reshape(-1, W)

    # Sort each window ascending so tie-break favors the smallest label.
    order = np.argsort(vals, axis=1, kind="stable")
    vals_s = np.take_along_axis(vals, order, axis=1)
    val_s = np.take_along_axis(val, order, axis=1)
    eq = vals_s[:, :, None] == vals_s[:, None, :]
    counts = (eq & val_s[:, None, :]).sum(axis=2)
    counts[~val_s] = -1
    best = counts.argmax(axis=1)
    mode = vals_s[np.arange(vals_s.shape[0]), best]
    return mode.reshape(out_shape).astype(arr.dtype)


#: Reducer registry by data kind.
REDUCERS: dict[str, Callable[[np.ndarray, Sequence[int]], np.ndarray]] = {
    "image": mean_downsample,
    "probability": mean_downsample,
    "segmentation": mode_downsample,
}


def get_reducer(kind: str) -> Callable[[np.ndarray, Sequence[int]], np.ndarray]:
    try:
        return REDUCERS[kind]
    except KeyError as e:
        raise ValueError(f"unknown data kind {kind!r}; known: {sorted(REDUCERS)}") from e


# --------------------------------------------------------------------------- #
# OME-NGFF coordinate transforms (center-aligned, matching ngff-zarr)
# --------------------------------------------------------------------------- #
def level_scale_translation(
    voxel_size: Sequence[float],
    offset: Sequence[float],
    cum_factor: Sequence[int],
) -> tuple[list[float], list[float]]:
    """Center-aligned (scale, translation) for a level with cumulative ``cum_factor``.

    ``scale = voxel_size * F``; ``translation = offset + 0.5 * (F - 1) * voxel_size``
    (a downsampled voxel's center sits at the centroid of the source voxels it
    covers). Matches ngff-zarr's convention.
    """
    scale = [float(v * f) for v, f in zip(voxel_size, cum_factor)]
    translation = [
        float(o + 0.5 * (f - 1) * v)
        for o, v, f in zip(offset, voxel_size, cum_factor)
    ]
    return scale, translation
