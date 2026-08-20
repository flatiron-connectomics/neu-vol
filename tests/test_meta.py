import pytest

from neu_vol import VoxelMeta


def test_defaults_and_validation():
    m = VoxelMeta(voxel_size=(8, 8, 8))
    assert m.ndim == 3
    assert m.offset == (0.0, 0.0, 0.0)
    assert m.units == "nm"
    assert m.axes == ("z", "y", "x")
    assert m.voxel_size == (8.0, 8.0, 8.0)  # coerced to float


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        VoxelMeta(voxel_size=(8, 8, 8), offset=(0, 0))
    with pytest.raises(ValueError):
        VoxelMeta(voxel_size=(8, 8, 8), axes=("z", "y"))


def test_downsampled_scales_voxel_size_corner_aligned():
    m = VoxelMeta(voxel_size=(8, 8, 8), offset=(0, 100, 200))
    d = m.downsampled((2, 2, 2))
    assert d.voxel_size == (16.0, 16.0, 16.0)
    assert d.offset == (0.0, 100.0, 200.0)  # corner alignment: offset unchanged
    # anisotropic in-plane-only downsample
    d2 = m.downsampled((1, 2, 2))
    assert d2.voxel_size == (8.0, 16.0, 16.0)


def test_with_axes_reorders_consistently():
    m = VoxelMeta(voxel_size=(40, 8, 8), offset=(1, 2, 3), axes=("z", "y", "x"))
    xyz = m.with_axes(("x", "y", "z"))
    assert xyz.axes == ("x", "y", "z")
    assert xyz.voxel_size == (8.0, 8.0, 40.0)
    assert xyz.offset == (3.0, 2.0, 1.0)


def test_with_axes_rejects_unknown_order():
    m = VoxelMeta(voxel_size=(8, 8, 8))
    with pytest.raises(ValueError):
        m.with_axes(("z", "y", "c"))
