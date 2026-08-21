"""Per-level shape, voxel size and origin, read from a source's own metadata.

Two things here are silent when wrong. A voxel size taken as ``2 ** index`` is correct
for an isotropic pyramid and wrong for the common one that halves x/y and leaves z
alone. And a level's **origin** was dropped entirely for a long time — every level came
back claiming to start at nm zero, which is true of a volume written from the origin and
false for anything cropped, with nothing to say so either way.
"""

import json

import pytest

from neu_vol import read_scales, scale_spec

# --------------------------------------------------------------------------- #
# precomputed
# --------------------------------------------------------------------------- #
def _precomputed(root, scales):
    root.mkdir()
    (root / "info").write_text(json.dumps({
        "@type": "neuroglancer_multiscale_volume", "type": "segmentation",
        "data_type": "uint64", "num_channels": 1, "scales": scales,
    }))
    return str(root)


def test_read_scales_from_precomputed_info(tmp_path):
    """Voxel size must come from each scale's own metadata, not 2**index.

    The pyramid here downsamples 2x in x/y but never in z — the exact shape that a
    2**scale assumption gets wrong.
    """
    src = _precomputed(tmp_path / "seg.precomputed", [
        {"key": "8_8_40", "resolution": [8, 8, 40], "size": [1024, 1024, 100],
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
        {"key": "16_16_40", "resolution": [16, 16, 40], "size": [512, 512, 100],
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
    ])

    scales = read_scales(src)
    assert len(scales) == 2
    # resolution/size are xyz in the file, zyx in ScaleInfo
    assert scales[0].voxel_size == (40.0, 8.0, 8.0) and scales[0].shape == (100, 1024, 1024)
    assert scales[1].voxel_size == (40.0, 16.0, 16.0) and scales[1].shape == (100, 512, 512)

    # the real factor is (1, 2, 2) — NOT (2, 2, 2) as 2**index would give
    assert scales[1].factor_from(scales[0]) == (1.0, 2.0, 2.0)

    spec = scale_spec(src, 1)
    assert spec["backend"] == "neuroglancer_precomputed" and spec["scale_index"] == 1


def test_read_scales_rejects_sources_without_metadata(tmp_path):
    bare = tmp_path / "nothing"
    bare.mkdir()
    with pytest.raises(ValueError):
        read_scales(str(bare))


# --------------------------------------------------------------------------- #
# the origin, which used to be dropped
# --------------------------------------------------------------------------- #
def test_voxel_offset_becomes_the_levels_nm_origin(tmp_path):
    """`voxel_offset` is in THIS level's voxels, so the nm origin is the product.

    A volume written from the origin has no offset and passes whether this is read or
    not, which is why dropping it survived: every fixture started at zero.
    """
    src = _precomputed(tmp_path / "crop.precomputed", [
        {"key": "8_8_40", "resolution": [8, 8, 40], "size": [64, 64, 16],
         "voxel_offset": [100, 200, 3],                      # xyz voxels
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
        {"key": "16_16_40", "resolution": [16, 16, 40], "size": [32, 32, 16],
         "voxel_offset": [50, 100, 3],                       # same nm, coarser voxels
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
    ])

    fine, coarse = read_scales(src)
    # zyx: offset (3, 200, 100) voxels x (40, 8, 8) nm
    assert fine.origin_nm == (120.0, 1600.0, 800.0)
    # the coarser level states half the voxels at twice the size — the SAME nm origin,
    # which is the property that makes an offset expressible per level at all
    assert coarse.origin_nm == (120.0, 1600.0, 800.0)


def test_a_level_with_no_offset_starts_at_zero(tmp_path):
    src = _precomputed(tmp_path / "plain.precomputed", [
        {"key": "8_8_8", "resolution": [8, 8, 8], "size": [64, 64, 64],
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
    ])
    assert read_scales(src)[0].origin_nm == (0.0, 0.0, 0.0)


def test_the_level_carries_its_frame_and_it_converts(tmp_path):
    """The point of carrying a Frame rather than a bare voxel size: the level can place
    its own voxels in nm, and callers stop rebuilding a transform from one field."""
    src = _precomputed(tmp_path / "f.precomputed", [
        {"key": "8_8_40", "resolution": [8, 8, 40], "size": [64, 64, 16],
         "voxel_offset": [10, 0, 0],
         "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
    ])
    level = read_scales(src)[0]

    assert level.frame.voxel_size_nm == (40.0, 8.0, 8.0)
    # voxel (0,0,0) of this level sits at its origin, not at nm zero
    assert tuple(level.frame.to_nm([0, 0, 0])) == (0.0, 0.0, 80.0)
    assert tuple(level.frame.to_nm([1, 0, 0])) == (40.0, 0.0, 80.0)
    # and back
    assert tuple(level.frame.to_voxel([40.0, 0.0, 80.0])) == (1.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# zarr / OME-NGFF
# --------------------------------------------------------------------------- #
def _ome(root, datasets):
    root.mkdir()
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": "z", "type": "space", "unit": "nanometer"},
                     {"name": "y", "type": "space", "unit": "nanometer"},
                     {"name": "x", "type": "space", "unit": "nanometer"}],
            "datasets": datasets,
        }]}},
    }))
    for ds in datasets:
        sub = root / ds["path"]
        sub.mkdir()
        (sub / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "array", "shape": [16, 64, 64],
            "data_type": "uint8", "chunk_grid": {"name": "regular",
            "configuration": {"chunk_shape": [16, 64, 64]}},
        }))
    return str(root)


def test_ome_translation_is_already_physical(tmp_path):
    """The zarr spelling of an offset is `translation`, and unlike precomputed's
    `voxel_offset` it is in PHYSICAL units — multiplying it by the voxel size would
    scale the origin by the resolution and put the data somewhere plausible."""
    src = _ome(tmp_path / "v.zarr", [
        {"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [40, 8, 8]},
            {"type": "translation", "translation": [120, 1600, 800]}]},
    ])

    level = read_scales(src)[0]
    assert level.voxel_size == (40.0, 8.0, 8.0)
    assert level.origin_nm == (120.0, 1600.0, 800.0)     # NOT 120*40, 1600*8, 800*8


def test_ome_without_a_translation_starts_at_zero(tmp_path):
    src = _ome(tmp_path / "w.zarr", [
        {"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [40, 8, 8]}]},
    ])
    assert read_scales(src)[0].origin_nm == (0.0, 0.0, 0.0)
