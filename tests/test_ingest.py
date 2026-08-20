import json
import os

import numpy as np
import pytest

from neu_vol import ingest_image_stack
from neu_vol.backends.base import open_backend
from neu_vol.pyramid import mean_downsample


def _write_multipage_tiff(path, vol):
    import tifffile

    tifffile.imwrite(path, vol)  # (z, y, x) -> multipage


def _write_slice_pngs(dirpath, vol):
    import imageio.v3 as iio

    os.makedirs(dirpath, exist_ok=True)
    for z in range(vol.shape[0]):
        iio.imwrite(os.path.join(dirpath, f"slice_{z:03d}.png"), vol[z])


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 1000, size=(16, 16, 16), dtype=np.uint16)


def _full(be):
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.mark.parametrize("profile,chunk,shard", [
    ("local", (8, 8, 8), None),
    ("ceph", (4, 4, 4), (8, 8, 8)),   # sharded: block == shard granularity
])
def test_ingest_multipage_tiff_multiscale(tmp_path, volume, profile, chunk, shard):
    src = str(tmp_path / "stack.tif")
    _write_multipage_tiff(src, volume)
    dst = str(tmp_path / "out.zarr")

    summary = ingest_image_stack(
        src, dst, voxel_size=(8, 8, 8), profile=profile,
        chunk=chunk, shard=shard, min_dim=8, delete_existing=True,
    )

    # 16 -> 8 : exactly one downsample level
    assert summary["num_levels"] == 2
    assert summary["level_shapes"] == [(16, 16, 16), (8, 8, 8)]

    # level 0 is a faithful copy
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(_full(lvl0), volume)

    # level 1 is the mean-downsampled volume
    lvl1 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "1")})
    np.testing.assert_array_equal(_full(lvl1), mean_downsample(volume, (2, 2, 2)))

    # OME-NGFF 0.5 group metadata written + well-formed
    grp = json.load(open(os.path.join(dst, "zarr.json")))
    assert grp["node_type"] == "group"
    ms = grp["attributes"]["ome"]["multiscales"][0]
    assert grp["attributes"]["ome"]["version"] == "0.5"
    assert [d["path"] for d in ms["datasets"]] == ["0", "1"]
    # center-aligned transforms
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [8.0, 8.0, 8.0]
    assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [16.0, 16.0, 16.0]
    assert ms["datasets"][1]["coordinateTransformations"][1]["translation"] == [4.0, 4.0, 4.0]


def test_ingest_slice_pngs_single_scale(tmp_path):
    vol = (np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8) % 256).astype(np.uint8)
    src_dir = str(tmp_path / "slices")
    _write_slice_pngs(src_dir, vol)
    dst = str(tmp_path / "png.zarr")

    summary = ingest_image_stack(
        os.path.join(src_dir, "*.png"), dst, voxel_size=(8, 8, 8),
        profile="local", chunk=(8, 8, 8), multiscale=False, delete_existing=True,
    )
    assert summary["num_levels"] == 1
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(_full(lvl0), vol)


def test_ingest_roundtrips_through_ngff_zarr_reader(tmp_path, volume):
    """Independent proof of OME-NGFF 0.5 compliance: ngff-zarr reads what we wrote."""
    import ngff_zarr as nz

    src = str(tmp_path / "stack.tif")
    _write_multipage_tiff(src, volume)
    dst = str(tmp_path / "rt.zarr")
    ingest_image_stack(src, dst, voxel_size=(8, 8, 8), profile="local",
                       chunk=(8, 8, 8), min_dim=8, delete_existing=True)

    ms = nz.from_ngff_zarr(dst)
    assert len(ms.images) == 2
    im0 = ms.images[0]
    assert im0.dims == ("z", "y", "x")
    assert im0.scale == {"z": 8.0, "y": 8.0, "x": 8.0}
    assert ms.images[1].scale == {"z": 16.0, "y": 16.0, "x": 16.0}
    assert ms.images[1].translation == {"z": 4.0, "y": 4.0, "x": 4.0}
    np.testing.assert_array_equal(np.asarray(im0.data), volume)


def test_ingest_validates_ome_metadata(tmp_path, volume):
    # validation runs by default; corrupting the schema would raise. Here we just
    # confirm the produced attrs validate.
    from neu_vol.ngff import validate_attrs

    src = str(tmp_path / "s.tif")
    _write_multipage_tiff(src, volume)
    summary = ingest_image_stack(src, str(tmp_path / "o.zarr"), voxel_size=(8, 8, 8),
                                 profile="local", chunk=(8, 8, 8), min_dim=8,
                                 delete_existing=True, validate=False)
    validate_attrs(summary["attrs"])  # raises if non-compliant
