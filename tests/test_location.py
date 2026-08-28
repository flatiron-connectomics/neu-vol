import os

import pytest

from neu_vol.location import default_progress_path, join, spec_kvstore, to_kvstore


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
    remote = default_progress_path("s3://b/data/seg")
    assert remote.startswith("seg.") and remote.endswith(".progress.jsonl")


def test_remote_manifests_do_not_collide_on_the_last_path_component():
    """Two runs sharing a manifest skip each other's blocks on resume, silently.

    `…/specimen3/gt_v1_eval` and `…/specimen5/gt_v1_eval` is an ordinary pair of
    destinations, and the basename alone made them one file.
    """
    a = default_progress_path("s3://bucket/wasp-seg-gt/specimen3/gt_v1_eval")
    b = default_progress_path("s3://bucket/wasp-seg-gt/specimen5/gt_v1_eval")
    c = default_progress_path("s3://other-bucket/wasp-seg-gt/specimen3/gt_v1_eval")
    assert a != b != c and a != c
    assert all(p.startswith("gt_v1_eval.") for p in (a, b, c)), "still readable"


def test_one_destination_is_one_manifest_however_it_is_spelled():
    """Resume and `neu-vol progress` both re-derive this name, so it has to be stable.

    A trailing slash resolving to a second manifest would make a resumed run start over
    and `neu-vol progress` report on an empty file.
    """
    plain = default_progress_path("s3://b/data/seg")
    assert default_progress_path("s3://b/data/seg/") == plain
    assert default_progress_path({"driver": "s3", "bucket": "b", "path": "data/seg"}) == plain


# --------------------------------------------------------------------------- #
# Byte / JSON I/O — one code path for local files and object stores
# --------------------------------------------------------------------------- #
def test_is_local_and_local_path():
    from neu_vol.location import is_local, local_path

    assert is_local("/mnt/x/vol") and is_local({"driver": "file", "path": "/x"})
    assert not is_local("s3://b/p") and not is_local("gs://b/p")
    assert local_path("/mnt/x/vol") == "/mnt/x/vol"
    with pytest.raises(ValueError, match="not a local path"):
        local_path("s3://b/p")


def test_write_read_bytes_creates_parent_dirs(tmp_path):
    """The file driver makes directories on write, so callers need no makedirs."""
    import os

    from neu_vol.location import exists, read_bytes, write_bytes

    base = str(tmp_path / "vol")
    write_bytes(base, b"\x00\x01", "mesh", "deep", "42")
    assert os.path.exists(f"{base}/mesh/deep/42")
    assert read_bytes(base, "mesh", "deep", "42") == b"\x00\x01"
    assert exists(base, "mesh", "deep", "42")


def test_missing_object_reads_as_none_not_an_error(tmp_path):
    """Existence checks must not need a separate stat, nor raise."""
    from neu_vol.location import exists, read_bytes, read_json

    base = str(tmp_path / "vol")
    assert read_bytes(base, "nope") is None
    assert read_json(base, "nope") is None
    assert not exists(base, "nope")


def test_json_round_trip_and_overwrite(tmp_path):
    from neu_vol.location import read_json, write_json

    base = str(tmp_path / "vol")
    write_json(base, {"type": "segmentation", "scales": [1]}, "info")
    assert read_json(base, "info")["type"] == "segmentation"
    write_json(base, {"type": "segmentation", "mesh": "mesh"}, "info")
    assert read_json(base, "info")["mesh"] == "mesh"     # overwrite, not append


def test_full_path_with_no_parts(tmp_path):
    """A location may already name the object itself."""
    from neu_vol.location import read_json, write_json

    p = str(tmp_path / "vol" / "info")
    write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}


def test_opening_an_s3_store_bootstraps_credentials(tmp_path, monkeypatch):
    """Every store-opening path must bootstrap, or it 403s on un-bootstrapped workers.

    tensorstore's profile provider cannot read ~/.aws/credentials, so credentials
    reach it only as AWS_* env vars, set per process. A path that skips the
    bootstrap fails *only* on workers that happened not to open an S3 store
    earlier in the run — invisible locally and intermittent on dask.
    """
    from neu_vol import location as L

    creds = tmp_path / "credentials"
    creds.write_text("[default]\naws_access_key_id = AKIATEST\n"
                     "aws_secret_access_key = secret\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    L._kv("s3://bucket/prefix", "info")
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIATEST"


def test_local_stores_need_no_credentials(tmp_path, monkeypatch):
    """The bootstrap must not fire for the file driver."""
    from neu_vol import location as L

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    L.write_bytes(str(tmp_path / "vol"), b"x", "info")
    assert "AWS_ACCESS_KEY_ID" not in os.environ


def test_stores_are_reused_per_prefix(tmp_path, monkeypatch):
    """Reopening per object is what floods the logs and re-resolves credentials.

    Each S3 open reconstructs the credential provider chain and emits two absl
    ERROR lines. Stage 2 writes ~3 objects per body, so at 20k bodies a
    per-object open means ~60k opens and ~120k spurious error lines — enough to
    bury a real failure.
    """
    import tensorstore as ts

    from neu_vol import location as L

    monkeypatch.setattr(L, "_STORES", {})
    opens = []
    real_open = ts.KvStore.open
    monkeypatch.setattr(ts.KvStore, "open",
                        lambda *a, **k: (opens.append(1), real_open(*a, **k))[1])

    base = str(tmp_path / "vol")
    for body in range(50):
        L.write_bytes(f"{base}/mesh", b"d", str(body))
        L.write_bytes(f"{base}/mesh", b"i", f"{body}.index")
    assert len(opens) == 1, f"{len(opens)} opens for one prefix; the store is not reused"

    L.write_bytes(f"{base}/skeleton", b"s", "1")      # a second prefix
    assert len(opens) == 2
    # reuse must not blur prefixes together
    assert L.read_bytes(f"{base}/mesh", "7") == b"d"
    assert L.read_bytes(f"{base}/skeleton", "1") == b"s"
    assert L.read_bytes(f"{base}/skeleton", "7") is None


def test_both_store_opening_paths_share_one_bootstrap():
    """The backend and the byte helpers must not drift apart again."""
    import inspect

    from neu_vol.backends import tensorstore as ts_backend

    src = inspect.getsource(ts_backend._kvstore_from_spec)
    assert "ensure_credentials" in src, \
        "the backend must use location.ensure_credentials, not its own copy"


def test_remote_locations_split_into_prefix_and_key():
    """s3 keys must not be concatenated onto the prefix (no network needed)."""
    from neu_vol.location import _kv

    store, key = _kv("s3://bucket/sample3/segmentation", "mesh", "info")
    assert key == "info"
    assert store.spec().to_json()["path"] == "sample3/segmentation/mesh/"

    store, key = _kv("s3://bucket/info")          # bucket root
    assert key == "info"
    assert store.spec().to_json().get("path", "") == ""


# --------------------------------------------------------------------------- #
# a local path is resolved, because tensorstore refuses a relative one
# --------------------------------------------------------------------------- #
def test_a_relative_local_path_is_made_absolute(tmp_path, monkeypatch):
    """**tensorstore's file driver rejects a relative path outright** — a `..` segment gives
    `Invalid file path` from inside its JSON binding, naming neither the caller nor the
    reason. So `neu_vol.describe("../data/piece.h5")` from a notebook one directory down
    failed, which is the most ordinary way anyone refers to a file.

    Resolved in `to_kvstore` because that is the one place every local location passes
    through — and an absolute path also still means the same thing on a dask worker whose
    working directory is elsewhere, which is the harder version of the same bug.
    """
    import os

    from neu_vol.location import to_kvstore

    (tmp_path / "data").mkdir()
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")

    kv = to_kvstore("../data/piece.h5")
    assert kv["path"] == str(tmp_path / "data" / "piece.h5")
    assert os.path.isabs(kv["path"]) and ".." not in kv["path"]
    # relative-to-cwd works the same way
    assert to_kvstore("piece.h5")["path"] == str(tmp_path / "notebooks" / "piece.h5")


def test_a_tilde_is_expanded(monkeypatch, tmp_path):
    """h5py takes a relative path but does not expand `~`, so it reached the reader verbatim
    and failed with errno 2 on a path that plainly exists."""
    from neu_vol.location import to_kvstore

    monkeypatch.setenv("HOME", str(tmp_path))
    assert to_kvstore("~/vol.zarr")["path"] == str(tmp_path / "vol.zarr")


def test_a_hand_written_file_spec_is_resolved_too(tmp_path, monkeypatch):
    """A spec someone typed is as likely to carry a relative path as a string is."""
    from neu_vol.location import to_kvstore

    monkeypatch.chdir(tmp_path)
    assert to_kvstore({"driver": "file", "path": "vol.zarr"})["path"] \
        == str(tmp_path / "vol.zarr")


def test_a_remote_location_is_untouched():
    """Only the file driver. An s3 path is a key prefix, not a filesystem path, and
    absolutising one would both corrupt it and rename every progress manifest, since a
    remote manifest's name carries a digest of the location."""
    from neu_vol.location import default_progress_path, to_kvstore

    assert to_kvstore("s3://my-bucket/a/b") == {"driver": "s3", "bucket": "my-bucket",
                                                "path": "a/b"}
    # the remote manifest name carries a digest of the location, so it must not move
    assert default_progress_path("s3://my-bucket/a/gt_v1").startswith("gt_v1.")
    assert to_kvstore({"driver": "s3", "bucket": "b", "path": "../x"})["path"] == "../x"


def test_an_empty_location_is_left_alone():
    """Degenerate, and turning it into the cwd would make `detect_file_backend("")` stat the
    working directory instead of answering None."""
    from neu_vol.location import to_kvstore

    assert to_kvstore("")["path"] == ""
