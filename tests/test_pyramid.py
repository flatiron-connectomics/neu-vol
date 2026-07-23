import numpy as np
import pytest

from em_volume_tools.pyramid import (
    auto_factor,
    cumulative_factors,
    downsample_schedule,
    level_scale_translation,
    mean_downsample,
    mode_downsample,
)


def test_mean_downsample_exact():
    a = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    out = mean_downsample(a, (2, 2, 2))
    assert out.shape == (2, 2, 2)
    assert out.dtype == np.uint16
    # first output voxel = mean of the 2x2x2 corner
    corner = a[0:2, 0:2, 0:2].mean()
    assert out[0, 0, 0] == np.rint(corner)


def test_mean_downsample_ragged_edge_uses_real_voxels():
    a = np.ones((3, 2, 2), dtype=np.float32)
    out = mean_downsample(a, (2, 2, 2))
    assert out.shape == (2, 1, 1)  # ceil(3/2)=2
    # both output voxels average only real (all-ones) voxels -> 1.0, no zero-padding bias
    assert np.allclose(out, 1.0)


def test_mode_downsample_label_preserving():
    a = np.full((2, 2, 2), 5, dtype=np.uint64)
    a[1, 1, 1] = 9
    out = mode_downsample(a, (2, 2, 2))
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == 5  # 5 appears 7x, 9 once -> mode 5 (a real label, not averaged)


def test_mode_downsample_majority_and_tiebreak():
    # window of 4: values [0,0,7,7] -> tie between 0 and 7 -> smallest (0)
    a = np.array([[[0, 0]], [[7, 7]]], dtype=np.uint64).reshape(2, 2, 1)
    out = mode_downsample(a, (2, 2, 1))
    assert out[0, 0, 0] == 0
    # only labels present, no interpolation to intermediate values
    a2 = np.array([1, 1, 1, 8], dtype=np.uint64).reshape(2, 2, 1)
    out2 = mode_downsample(a2, (2, 2, 1))
    assert out2[0, 0, 0] == 1


def test_auto_factor_isotropic_vs_anisotropic():
    assert auto_factor((8, 8, 8)) == (2, 2, 2)
    assert auto_factor((40, 8, 8)) == (1, 2, 2)  # z coarse -> in-plane only


def test_downsample_schedule_isotropic():
    sched = downsample_schedule((64, 64, 64), (8, 8, 8), min_dim=8)
    assert all(f == (2, 2, 2) for f in sched)
    # 64 -> 32 -> 16 -> 8 : stop when max dim <= 8
    assert len(sched) == 3


def test_downsample_schedule_explicit_factors_passthrough():
    sched = downsample_schedule((64, 64, 64), (8, 8, 8), factors=[(2, 2, 2), (1, 2, 2)])
    assert sched == [(2, 2, 2), (1, 2, 2)]


def test_cumulative_factors():
    cum = cumulative_factors([(2, 2, 2), (2, 2, 1)], ndim=3)
    assert cum == [(1, 1, 1), (2, 2, 2), (4, 4, 2)]
    # empty schedule (single-scale) still yields correct level-0 identity
    assert cumulative_factors([], ndim=3) == [(1, 1, 1)]


def test_level_scale_translation_center_aligned():
    # matches ngff-zarr: factor 2 on 8nm -> scale 16, translation 4
    scale, translation = level_scale_translation((8, 8, 8), (0, 0, 0), (2, 2, 2))
    assert scale == [16.0, 16.0, 16.0]
    assert translation == [4.0, 4.0, 4.0]
