import pytest

from em_volume_tools.location import default_progress_path, join, spec_kvstore, to_kvstore


def test_to_kvstore_local_path():
    assert to_kvstore("/mnt/x/vol.zarr") == {"driver": "file", "path": "/mnt/x/vol.zarr"}


def test_to_kvstore_s3_and_gs():
    assert to_kvstore("s3://my-bucket/a/b") == {"driver": "s3", "bucket": "my-bucket", "path": "a/b"}
    assert to_kvstore("gs://bkt/p") == {"driver": "gcs", "bucket": "bkt", "path": "p"}
    assert to_kvstore("s3://bucket-only") == {"driver": "s3", "bucket": "bucket-only", "path": ""}


def test_to_kvstore_passthrough_dict():
    kv = {"driver": "s3", "bucket": "b", "path": "p", "endpoint": "https://x"}
    assert to_kvstore(kv) == kv


def test_join_local_absolute_keeps_leading_slash():
    kv = to_kvstore("/mnt/x/vol.zarr")
    assert join(kv, "0") == {"driver": "file", "path": "/mnt/x/vol.zarr/0"}
    assert join(kv, "scale0", "image")["path"] == "/mnt/x/vol.zarr/scale0/image"


def test_join_s3_prefix():
    kv = to_kvstore("s3://b/pre/fix")
    assert join(kv, "1")["path"] == "pre/fix/1"
    # bucket root (empty prefix)
    assert join(to_kvstore("s3://b"), "0")["path"] == "0"


def test_spec_kvstore_forms():
    assert spec_kvstore({"backend": "zarr3", "path": "/a/b"}) == {"driver": "file", "path": "/a/b"}
    assert spec_kvstore({"backend": "zarr3", "path": "s3://bk/p"})["driver"] == "s3"
    kv = {"driver": "s3", "bucket": "b", "path": "p"}
    assert spec_kvstore({"backend": "zarr3", "kvstore": kv}) == kv
    with pytest.raises(ValueError):
        spec_kvstore({"backend": "zarr3"})


def test_default_progress_path():
    assert default_progress_path("/a/vol.zarr") == "/a/vol.zarr.progress.jsonl"
    assert default_progress_path("s3://b/data/seg").endswith("seg.progress.jsonl")
