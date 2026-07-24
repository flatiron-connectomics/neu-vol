"""S3 destination plumbing (no network): specs carry s3 kvstores, subpaths join."""

from em_volume_tools.location import join, to_kvstore
from em_volume_tools.profiles import precomputed_create_spec, zarr3_create_spec


def test_zarr3_create_spec_s3_kvstore_and_level_join():
    base = to_kvstore("s3://my-bucket/seg/sample3.zarr")
    level0 = join(base, "0")
    spec = zarr3_create_spec("local", level0, (16, 16, 16), "uint64",
                             dimension_names=("z", "y", "x"), chunk=(8, 8, 8))
    assert spec["kvstore"] == {"driver": "s3", "bucket": "my-bucket",
                               "path": "seg/sample3.zarr/0"}
    assert "path" not in spec  # location lives entirely in the kvstore


def test_precomputed_create_spec_s3_kvstore():
    spec = precomputed_create_spec("s3-neuroglancer", "s3://bkt/vol", (8, 8, 8), "uint64",
                                   resolution_zyx=[8, 8, 8], scale_index=0,
                                   encoding="compressed_segmentation")
    assert spec["kvstore"] == {"driver": "s3", "bucket": "bkt", "path": "vol"}
    assert spec["compressed_segmentation_block_size_zyx"] == [8, 8, 8]
