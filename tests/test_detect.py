import os

import numpy as np

from em_volume_tools import convert
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.source_metadata import detect_backend
from em_volume_tools.profiles import precomputed_create_spec, zarr3_create_spec


def _zarr3(path, data):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return be


def _zarr2(path, data):
    import tensorstore as ts
    s = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": path},
                 "metadata": {"shape": list(data.shape), "chunks": [8, 8, 8],
                              "dtype": "<u2", "compressor": None, "order": "C",
                              "fill_value": 0}, "create": True, "delete_existing": True}).result()
    s[...] = data
    return path


def test_detect_zarr3(tmp_path):
    p = str(tmp_path / "a.zarr")
    _zarr3(p, np.zeros((8, 8, 8), np.uint16))
    assert detect_backend(p) == "zarr3"


def test_detect_precomputed(tmp_path):
    p = str(tmp_path / "pc")
    TensorStoreBackend.create(
        precomputed_create_spec("s3-neuroglancer", p, (8, 8, 8), "uint8",
                                resolution_zyx=[8, 8, 8], scale_index=0, chunk=(8, 8, 8)),
        delete_existing=True)
    assert detect_backend(p) == "neuroglancer_precomputed"


def test_detect_zarr2_and_convert(tmp_path):
    vol = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    p = str(tmp_path / "v2.zarr")
    _zarr2(p, vol)
    assert detect_backend(p) == "zarr2"

    # convert autodetects zarr2 and reads it (voxel_size supplied; no OME metadata)
    dst = str(tmp_path / "out.zarr")
    convert(p, dst, voxel_size=(8, 8, 8), chunk=(8, 8, 8), multiscale=False, delete_existing=True)
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(out.read_region((slice(0, 8),) * 3), vol)


def test_detect_none_for_empty_dir(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(d)
    assert detect_backend(d) is None


# --------------------------------------------------------------------------- #
# CloudVolume-written precomputed: chunks carry a .gz suffix
# --------------------------------------------------------------------------- #
def _precomputed_info(tmp_path, key="8_8_8"):
    """A minimal precomputed volume; caller decides how chunks are named."""
    import json
    root = tmp_path / "vol"
    (root / key).mkdir(parents=True)
    (root / "info").write_text(json.dumps({
        "type": "image", "data_type": "uint8", "num_channels": 1,
        "scales": [{"key": key, "size": [256, 256, 256], "resolution": [8, 8, 8],
                    "chunk_sizes": [[64, 64, 64]], "encoding": "raw",
                    "voxel_offset": [0, 0, 0]}]}))
    return root


def test_gzipped_chunks_are_detected_as_a_distinct_backend(tmp_path):
    """The whole point: tensorstore reads these as ZEROS and raises nothing.

    A conversion of 1.9M blocks was lost to exactly this, so detection must not
    report plain `neuroglancer_precomputed` and let the run proceed.
    """
    from em_volume_tools.source_metadata import PRECOMPUTED_GZ, detect_backend

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64.gz").write_bytes(b"\x1f\x8b junk")
    assert detect_backend(str(root)) == PRECOMPUTED_GZ


def test_normal_precomputed_is_unaffected(tmp_path):
    from em_volume_tools.source_metadata import detect_backend

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64").write_bytes(b"\x00" * 8)
    assert detect_backend(str(root)) == "neuroglancer_precomputed"


def test_a_scale_with_no_chunks_yet_is_not_flagged(tmp_path):
    """An in-flight or empty volume has no chunks to judge; do not guess `.gz`."""
    from em_volume_tools.source_metadata import detect_backend

    assert detect_backend(str(_precomputed_info(tmp_path))) == "neuroglancer_precomputed"


def test_opening_a_gz_volume_without_cloudvolume_fails_with_guidance(tmp_path):
    """Fail fast and say what to do — not a silent zero-filled read."""
    import pytest

    from em_volume_tools.backends.base import open_backend
    from em_volume_tools.source_metadata import PRECOMPUTED_GZ

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64.gz").write_bytes(b"\x1f\x8b junk")
    try:
        import cloudvolume  # noqa: F401
        pytest.skip("cloud-volume is installed here; the missing-dep path cannot run")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="cloud-volume|em-vol-cv"):
        open_backend({"backend": PRECOMPUTED_GZ, "path": str(root)})


def test_data_spec_keeps_the_detected_backend(tmp_path):
    """`data_spec` selects the reader, so it must not contradict detection.

    This exact bug cost a full conversion run: detection correctly returned
    PRECOMPUTED_GZ, but `_read_precomputed` hardcoded "neuroglancer_precomputed" into
    the spec the workers actually read through — so every block went to tensorstore,
    requested unsuffixed keys, and came back as zeros with nothing raised.
    """
    from em_volume_tools.source_metadata import PRECOMPUTED_GZ, read_source_metadata

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64.gz").write_bytes(b"\x1f\x8b junk")
    meta = read_source_metadata({"backend": PRECOMPUTED_GZ, "path": str(root)})
    assert meta is not None, "the .gz variant reads the SAME info document"
    assert meta["data_spec"]["backend"] == PRECOMPUTED_GZ
    # ...and the metadata itself still comes through, which is why it dispatches here
    assert meta["voxel_size"] == (8.0, 8.0, 8.0)
    assert meta["kind"] == "image"


def test_plain_precomputed_data_spec_is_unchanged(tmp_path):
    from em_volume_tools.source_metadata import read_source_metadata

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64").write_bytes(b"\x00" * 8)
    meta = read_source_metadata({"backend": "neuroglancer_precomputed", "path": str(root)})
    assert meta["data_spec"]["backend"] == "neuroglancer_precomputed"


def test_cloudvolume_url_handles_every_spec_form(tmp_path):
    """The URL must build from a `kvstore` spec, not just a `path` one.

    `convert` passes `read_source_metadata`'s `data_spec`, which carries `kvstore`
    and no `path`. Reading `spec["path"]` yielded `precomputed://s3:///` — empty
    bucket and path — and every test that built a spec by hand passed anyway, because
    hand-built specs use `path`. Cover all three forms.
    """
    from em_volume_tools.backends.cloudvolume import _url

    root = _precomputed_info(tmp_path)
    (root / "8_8_8" / "0-64_0-64_0-64.gz").write_bytes(b"\x1f\x8b junk")

    from em_volume_tools.source_metadata import PRECOMPUTED_GZ, read_source_metadata

    by_path = _url({"backend": PRECOMPUTED_GZ, "path": str(root)})
    assert by_path == f"precomputed://file://{root}"

    meta = read_source_metadata({"backend": PRECOMPUTED_GZ, "path": str(root)})
    assert _url(meta["data_spec"]) == by_path, "kvstore form must agree with path form"

    assert _url({"backend": PRECOMPUTED_GZ, "path": "s3://bkt/a/b"}) \
        == "precomputed://s3://bkt/a/b"


def test_cloudvolume_url_rejects_an_unsupported_driver():
    import pytest

    from em_volume_tools.backends.cloudvolume import _url

    with pytest.raises(ValueError, match="local paths and s3"):
        _url({"kvstore": {"driver": "gcs", "bucket": "b", "path": "p"}})
