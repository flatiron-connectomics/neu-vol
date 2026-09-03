"""What `detect_backend` is allowed to cost.

It is called by nearly every op before anything else happens, so a cheap-looking probe
inside it is paid everywhere. The gzip check used to list the *finest* scale's prefix
with `limit=4` — which reads as bounded and is not, because `list_keys` awaits the whole
listing before Python truncates it. Measured on a dense 8-level EM volume on S3: **51 s
in `detect_backend`**, on every command that touched it.

These tests pin the shape of the fix rather than a duration: the probe must ask about a
named object, and if it ever lists, it must list the scale with the fewest chunks.
"""

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec
from neu_vol.source_metadata import (PRECOMPUTED_GZ, _first_chunk_key,
                                            detect_backend,
                                            precomputed_chunks_are_gzipped)


def _precomputed(tmp_path, name="vol", shape=(32, 32, 32)):
    data = np.ones(shape, "uint8")
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), data)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), profile="local-neuroglancer",
            chunk=(8, 8, 8), factors=[(2, 2, 2)], min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# the predicted key
# --------------------------------------------------------------------------- #
def test_the_origin_chunk_key_is_clipped_to_the_scale_size():
    """A scale smaller than one chunk stores `0-88_0-71_0-108`, not `0-128_...`."""
    assert _first_chunk_key(
        {"size": [88, 71, 108], "chunk_sizes": [[128, 128, 128]]}) == "0-88_0-71_0-108"
    assert _first_chunk_key(
        {"size": [512, 512, 512], "chunk_sizes": [[128, 128, 128]]}) \
        == "0-128_0-128_0-128"


def test_a_voxel_offset_shifts_the_key():
    assert _first_chunk_key({"size": [256, 256, 256], "voxel_offset": [64, 0, 32],
                             "chunk_sizes": [[128, 128, 128]]}) \
        == "64-192_0-128_32-160"


def test_unusable_metadata_gives_no_key_rather_than_a_wrong_one():
    assert _first_chunk_key({}) is None
    assert _first_chunk_key({"size": [1, 2, 3]}) is None


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
def test_a_plain_volume_is_detected_without_listing_anything(tmp_path, monkeypatch):
    """The point of the fix: no enumeration on the happy path.

    A listing here is what cost 51 s, and it scales with the volume rather than with the
    question being asked, so it must not happen at all when the predicted key resolves.
    """
    dst = _precomputed(tmp_path)
    from neu_vol import location

    def refuse(*a, **k):
        raise AssertionError("detect_backend listed a prefix instead of probing a key")

    monkeypatch.setattr(location, "list_keys", refuse)
    monkeypatch.setattr("neu_vol.source_metadata.list_keys", refuse,
                        raising=False)
    assert detect_backend(dst) == "neuroglancer_precomputed"


def test_gz_suffixed_chunks_are_detected(tmp_path):
    """The failure this exists to prevent: tensorstore asks for the unsuffixed key,
    reads the fill value, and a whole conversion comes back as zeros with nothing
    raised."""
    import os
    import shutil

    dst = _precomputed(tmp_path, "gz")
    # rename every chunk object the way CloudVolume writes them
    for root, _dirs, files in os.walk(dst):
        for f in files:
            if "_" in f and "-" in f:
                shutil.move(os.path.join(root, f), os.path.join(root, f + ".gz"))
    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_the_coarsest_scale_is_probed(tmp_path):
    """Any scale answers the question — gzipping is how the volume was written — so the
    one with the fewest chunks is the one to ask about, in case it has to list."""
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "many")
    scales = read_json(dst, "info")["scales"]
    coarsest = max(scales, key=lambda s: tuple(s["resolution"]))
    finest = min(scales, key=lambda s: tuple(s["resolution"]))
    assert coarsest["key"] != finest["key"], "test volume needs >1 scale"
    # asking about either scale gives the same answer, which is what makes the
    # cheap one a legitimate substitute
    assert precomputed_chunks_are_gzipped(dst, coarsest) is False
    assert precomputed_chunks_are_gzipped(dst, finest) is False


def test_an_absent_origin_chunk_falls_back_to_a_listing(tmp_path):
    """Legitimate on a sparse volume: the predicted key may hold nothing at all."""
    import os

    dst = _precomputed(tmp_path, "sparse")
    from neu_vol.location import read_json
    coarsest = max(read_json(dst, "info")["scales"],
                   key=lambda s: tuple(s["resolution"]))
    key = _first_chunk_key(coarsest)
    origin = os.path.join(dst, coarsest["key"], key)
    assert os.path.exists(origin)
    os.remove(origin)                       # now only non-origin chunks exist
    # still answerable, via the listing fallback
    assert precomputed_chunks_are_gzipped(dst, coarsest) is False


def test_a_volume_whose_declared_scales_are_mostly_empty_is_still_detected(tmp_path):
    """An `info` may declare scales that were never written, and the coarsest is not
    guaranteed to be one of the real ones.

    A sample3 neuropil mask declares seven scales and stores exactly one. Asking only
    about the coarsest found no origin chunk, listed an empty prefix and answered "not
    gzipped" — so every block read as zeros, the failure this probe exists to prevent,
    reached by a different road than the one it was written for.
    """
    import os
    import shutil

    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "mostly_empty")
    scales = read_json(dst, "info")["scales"]
    assert len(scales) > 1, "test volume needs >1 scale"
    finest = min(scales, key=lambda s: tuple(s["resolution"]))

    # Keep the FINEST scale's chunks only, `.gz`-suffixed the way CloudVolume writes
    # them. Every other scale keeps its `info` entry and loses its objects, so the
    # coarsest — what detection reaches for first — is one of the empty ones.
    for sc in scales:
        d = os.path.join(dst, sc["key"])
        if sc["key"] != finest["key"]:
            shutil.rmtree(d)
            continue
        for f in os.listdir(d):
            shutil.move(os.path.join(d, f), os.path.join(d, f + ".gz"))

    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_gz_is_detected_when_no_scale_has_an_origin_chunk(tmp_path):
    """Pass 1 misses entirely on a volume sparse at every origin, and the one scale
    holding data may not be the one the listing fallback reaches first.

    A registered neuropil mask declared five scales, wrote only `64_64_64`, and had
    none of its 26 chunks at the origin. Pass 1 found nothing anywhere; pass 2 listed
    only the coarsest declared scale, which was one of the unwritten ones, found no
    chunk name and answered "not gzipped". tensorstore then asked for unsuffixed keys
    and all 80 level-0 tasks recorded "empty" — a full run that wrote nothing and
    raised nothing.
    """
    import os
    import shutil

    from neu_vol.location import read_json
    from neu_vol.source_metadata import _first_chunk_key

    dst = _precomputed(tmp_path, "no_origin_gz")
    scales = read_json(dst, "info")["scales"]
    finest = min(scales, key=lambda s: tuple(s["resolution"]))

    for sc in scales:
        d = os.path.join(dst, sc["key"])
        if sc["key"] != finest["key"]:
            shutil.rmtree(d)                    # declared, never written
            continue
        os.remove(os.path.join(d, _first_chunk_key(sc)))    # and no ORIGIN chunk
        for f in os.listdir(d):
            shutil.move(os.path.join(d, f), os.path.join(d, f + ".gz"))
    assert os.listdir(os.path.join(dst, finest["key"])), "test needs surviving chunks"

    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_the_listing_fallback_still_goes_to_the_coarsest_scale(tmp_path, monkeypatch):
    """Sweeping every scale's origin key must not put the listing back on a dense one.

    Pass 2 lists the FIRST scale offered, and `detect_backend` orders them
    coarsest-first; listing the finest is what cost 51 s.
    """
    import os

    from neu_vol import location
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "no_origins")
    scales = read_json(dst, "info")["scales"]
    coarsest = max(scales, key=lambda s: tuple(s["resolution"]))
    for sc in scales:                       # no origin chunk anywhere -> pass 1 misses
        os.remove(os.path.join(dst, sc["key"], _first_chunk_key(sc)))

    listed = []
    real = location.list_keys
    monkeypatch.setattr(location, "list_keys",
                        lambda loc, prefix="", *a, **k: (listed.append(prefix),
                                                         real(loc, prefix, *a, **k))[1])
    assert detect_backend(dst) == "neuroglancer_precomputed"
    assert listed == [coarsest["key"]]


def test_an_empty_scale_is_not_gzipped(tmp_path):
    """Nothing stored is not evidence either way; False keeps the ordinary reader."""
    assert precomputed_chunks_are_gzipped(
        str(tmp_path), {"key": "8_8_8", "size": [8, 8, 8],
                        "chunk_sizes": [[8, 8, 8]]}) is False


# The companion to these — that building a viewer layer must not open every level — moved
# with `volume_layer` to neu-glance, where it is pinned against that package's own caller.


def test_limit_is_documented_as_not_bounding_the_request():
    """The docstring promised "one small request" and delivered a full enumeration.

    Pinned as a test because the next person to reach for `limit` to make a listing
    cheap will read the docstring, not the tensorstore source.
    """
    from neu_vol.location import list_keys

    doc = list_keys.__doc__
    assert "not the request" in doc.lower() or "does not bound" in doc.lower()
