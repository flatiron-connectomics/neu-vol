"""Which scale of a multiscale source becomes level 0.

An `info` is a CLAIM about a volume's layout; the chunk objects are the volume. A
sample3 neuropil mask declares seven scales (8 nm through 512 nm) and stores exactly
one — `64_64_64`, 18 chunk objects. Reading "level 0" there meant reading a scale that
was never written: it opened, reported the full 11260x9000x13750 extent its metadata
claims, and returned the fill value at every block. A 10.1 TiB copy of zeros, with
nothing raised anywhere.

So level 0 is the finest scale that actually STORES data, `--src-level` names one
explicitly, and a scale that stores nothing while another does is refused outright.
"""

import json
import os
import shutil

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec
from neu_vol.source_metadata import (describe, finest_populated_scale,
                                     read_source_metadata,
                                     require_populated_scale, scale_stores_chunks)


def _precomputed(tmp_path, name="vol", shape=(32, 32, 32)):
    """A real two-scale precomputed volume, every scale written."""
    data = np.arange(int(np.prod(shape)), dtype="uint64").reshape(shape) % 7 + 1
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), data)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), profile="local-neuroglancer",
            chunk=(8, 8, 8), factors=[(2, 2, 2)], min_dim=8, delete_existing=True,
            kind="segmentation")
    return dst


def _strip_finest(dst):
    """Delete the finest scale's chunks, keeping its `info` entry. Returns (fine, coarse).

    This is the shape of the mask exactly: metadata declaring a scale that the store
    does not have.
    """
    with open(os.path.join(dst, "info")) as fh:
        scales = json.load(fh)["scales"]
    fine = min(scales, key=lambda s: tuple(s["resolution"]))
    coarse = max(scales, key=lambda s: tuple(s["resolution"]))
    assert fine["key"] != coarse["key"], "test volume needs >1 scale"
    shutil.rmtree(os.path.join(dst, fine["key"]))
    os.makedirs(os.path.join(dst, fine["key"]))      # the prefix, with nothing in it
    return fine, coarse


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
def test_a_written_scale_is_distinguished_from_a_merely_declared_one(tmp_path):
    dst = _precomputed(tmp_path, "declared")
    fine, coarse = _strip_finest(dst)
    assert scale_stores_chunks(dst, coarse) is True
    assert scale_stores_chunks(dst, fine) is False


def test_the_finest_populated_scale_is_found(tmp_path):
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "finest")
    scales = read_json(dst, "info")["scales"]
    assert finest_populated_scale(dst, scales) == 0      # everything written

    _strip_finest(dst)
    scales = read_json(dst, "info")["scales"]
    idx = finest_populated_scale(dst, scales)
    assert idx is not None and idx != 0
    assert scales[idx]["key"] == "16_16_16"


def test_an_unwritten_scale_is_distinguished_from_a_sparse_one(tmp_path):
    """The pair that makes the heuristic sound. Both scales lack an origin chunk; only
    one of them is actually empty."""
    from neu_vol.location import read_json
    from neu_vol.source_metadata import _first_chunk_key, scale_is_empty

    dst = _precomputed(tmp_path, "pair")
    scales = read_json(dst, "info")["scales"]
    os.remove(os.path.join(dst, scales[0]["key"], _first_chunk_key(scales[0])))
    assert scale_stores_chunks(dst, scales[0]) is False    # cheap probe: no evidence
    assert scale_is_empty(dst, scales[0]) is False         # conclusive: it has chunks

    _strip_finest(dst)
    assert scale_is_empty(dst, scales[0]) is True


def test_a_volume_sparse_at_the_origin_of_every_scale_is_still_resolved(tmp_path):
    """The registered neuropil mask: 26 chunks at 64 nm, none of them the origin one,
    and no origin chunk at any other scale either.

    Tier 1 sees nothing anywhere. Giving up there fell back to the finest DECLARED
    scale — precisely the failure this is meant to prevent — so `copy` refused a
    perfectly good volume rather than reading it.
    """
    from neu_vol.location import read_json
    from neu_vol.source_metadata import _first_chunk_key

    dst = _precomputed(tmp_path, "no_origin_anywhere")
    fine, _coarse = _strip_finest(dst)                    # 8_8_8 declared but unwritten
    scales = read_json(dst, "info")["scales"]
    for sc in scales:                                     # and nothing at any origin
        origin = os.path.join(dst, sc["key"], _first_chunk_key(sc))
        if os.path.exists(origin):
            os.remove(origin)
    assert not any(scale_stores_chunks(dst, sc) for sc in scales), "tier 1 must miss"

    assert finest_populated_scale(dst, scales) == 1       # not 0, which stores nothing
    assert describe(dst)["source_level"] == 1


def test_a_volume_that_stores_nothing_anywhere_gives_no_answer(tmp_path):
    """None, not 0. The caller has to decide what to do about it; `require_populated_
    scale` warns rather than raising, since an empty volume is legal."""
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "nothing")
    scales = read_json(dst, "info")["scales"]
    for sc in scales:
        shutil.rmtree(os.path.join(dst, sc["key"]))
    assert finest_populated_scale(dst, scales) is None


# --------------------------------------------------------------------------- #
# what level 0 resolves to
# --------------------------------------------------------------------------- #
def test_level_0_is_the_finest_scale_that_stores_data(tmp_path):
    """The bug: the finest DECLARED scale opened and read as the fill value."""
    dst = _precomputed(tmp_path, "resolve")
    _strip_finest(dst)

    meta = read_source_metadata({"backend": "neuroglancer_precomputed", "path": dst})
    assert meta["data_spec"]["scale_index"] == 1
    # and its voxel size is that scale's OWN, not 2**level of the finest (invariant
    # NM-SPACE) — the two agree on this pyramid, but the value must come from the scale.
    assert tuple(meta["voxel_size"]) == (16.0, 16.0, 16.0)


def test_an_explicit_scale_index_wins_over_the_probe(tmp_path):
    """`--src-level` must be able to name a scale the probe would not have chosen —
    including an empty one, which `neu-vol write` legitimately does when filling a
    level that does not exist yet."""
    dst = _precomputed(tmp_path, "explicit")
    _strip_finest(dst)

    meta = read_source_metadata({"backend": "neuroglancer_precomputed", "path": dst,
                                 "scale_index": 0})
    assert meta["data_spec"]["scale_index"] == 0
    assert tuple(meta["voxel_size"]) == (8.0, 8.0, 8.0)


def test_an_out_of_range_level_is_an_error_not_a_wrap(tmp_path):
    dst = _precomputed(tmp_path, "range")
    with pytest.raises(ValueError, match="does not exist"):
        read_source_metadata({"backend": "neuroglancer_precomputed", "path": dst,
                              "scale_index": 9})


def test_an_ordinary_volume_is_unaffected(tmp_path):
    """Every scale written -> level 0 is still scale 0, and the probe costs one check."""
    dst = _precomputed(tmp_path, "ordinary")
    meta = read_source_metadata({"backend": "neuroglancer_precomputed", "path": dst})
    assert meta["data_spec"]["scale_index"] == 0
    assert tuple(meta["voxel_size"]) == (8.0, 8.0, 8.0)


def test_describe_reports_which_source_level_it_described(tmp_path):
    """`copy` derives voxel size, chunking and extent from this, so it has to say
    which scale they came from."""
    dst = _precomputed(tmp_path, "described")
    assert describe(dst)["source_level"] == 0

    _strip_finest(dst)
    d = describe(dst)
    assert d["source_level"] == 1
    assert tuple(d["meta"]["voxel_size"]) == (16.0, 16.0, 16.0)
    assert d["shape"] == (16, 16, 16)
    # `levels` still maps the whole DECLARED pyramid — it answers a different question
    assert 0 in d["levels"]

    assert describe(dst, level=0)["source_level"] == 0


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
def test_reading_a_declared_but_unwritten_scale_is_refused(tmp_path):
    dst = _precomputed(tmp_path, "refused")
    _strip_finest(dst)
    with pytest.raises(ValueError, match="stores no data"):
        require_populated_scale({"backend": "neuroglancer_precomputed", "path": dst,
                                 "scale_index": 0})
    # the message has to name a level that WOULD work, or it just says "no"
    with pytest.raises(ValueError, match=r"levels that DO store data: 1"):
        require_populated_scale({"backend": "neuroglancer_precomputed", "path": dst,
                                 "scale_index": 0})


def test_convert_refuses_before_it_writes_anything(tmp_path):
    dst = _precomputed(tmp_path, "guarded")
    _strip_finest(dst)
    out = str(tmp_path / "out")
    with pytest.raises(ValueError, match="stores no data"):
        convert(dst, out, src_level=0, profile="local-neuroglancer",
                delete_existing=True)
    assert not os.path.exists(os.path.join(out, "info"))


def test_a_sparse_scale_with_no_origin_chunk_is_allowed(tmp_path):
    """An absent ORIGIN chunk is not an unwritten scale.

    A sparse volume elides all-background chunks, so a ground-truth crop away from the
    origin stores nothing there while being perfectly written. Refusing one of those
    would be a worse failure than the one this guard prevents.
    """
    from neu_vol.source_metadata import _first_chunk_key

    dst = _precomputed(tmp_path, "sparse")
    with open(os.path.join(dst, "info")) as fh:
        scales = json.load(fh)["scales"]
    for sc in scales:
        os.remove(os.path.join(dst, sc["key"], _first_chunk_key(sc)))

    require_populated_scale({"backend": "neuroglancer_precomputed", "path": dst,
                             "scale_index": 0})           # does not raise


def test_a_sparse_volume_is_not_read_at_the_wrong_scale(tmp_path):
    """The regression this heuristic nearly shipped.

    A sparse volume's finest scale can have no origin chunk while a coarser one does —
    the coarse level covers more ground per chunk, so the reduction of a nearby crop
    lands in its origin chunk. Choosing by origin alone silently promoted level 1 to
    level 0, halving the resolution of every rebuilt pyramid.
    """
    from neu_vol.location import read_json

    labels = np.zeros((64, 64, 64), "uint64")
    labels[16:32, 16:32, 16:32] = 7                        # nothing at the origin
    src = str(tmp_path / "gt.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, labels.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in labels.shape), labels)
    dst = str(tmp_path / "gt.precomputed")
    convert(src, dst, voxel_size=(8, 8, 8), profile="local-neuroglancer",
            chunk=(16, 16, 16), factors=[(2, 2, 2)], min_dim=8, kind="segmentation",
            delete_existing=True)

    scales = read_json(dst, "info")["scales"]
    assert not scale_stores_chunks(dst, scales[0]), "test needs an elided origin chunk"
    assert scale_stores_chunks(dst, scales[1]), "test needs a populated coarse origin"
    # ...and the answer is still scale 0, because scale 0 is not empty
    assert finest_populated_scale(dst, scales) == 0
    assert describe(dst)["source_level"] == 0


def test_a_non_precomputed_source_is_not_probed(tmp_path):
    """zarr has no `info` and no scale keyspace to ask about; the guard is a no-op."""
    require_populated_scale({"backend": "zarr3", "path": str(tmp_path / "nope")})


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_a_copy_of_a_mostly_unwritten_volume_carries_the_real_data(tmp_path):
    """The whole point: the output holds the labels, not a plausible field of zeros."""
    dst = _precomputed(tmp_path, "e2e")
    _strip_finest(dst)
    out = str(tmp_path / "e2e_copy")

    convert(dst, out, profile="local-neuroglancer", delete_existing=True)

    d = describe(out)
    assert d["shape"] == (16, 16, 16)                       # the 16 nm scale, not 32^3
    assert tuple(d["meta"]["voxel_size"]) == (16.0, 16.0, 16.0)
    from neu_vol.backends.base import open_backend
    got = open_backend(d["meta"]["data_spec"]).read_region(
        tuple(slice(0, s) for s in d["shape"]))
    assert np.count_nonzero(got) > 0
