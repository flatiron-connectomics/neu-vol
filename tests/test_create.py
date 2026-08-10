"""`create`: an empty volume, laid out either by hand or from a reference's frame.

The property that matters is that ``--like`` produces a volume in the *same
coordinate frame* as its reference — same level shapes, same per-level scale and
translation. That is what makes "write this piece at (z, y, x) of level 2" mean the
same thing in both, and it is why the reference's level shapes are copied rather than
recomputed from a schedule.
"""

import json
import os

import numpy as np
import pytest

from em_volume_tools import convert, create_volume, describe, plan_volume
from em_volume_tools.backends.base import open_backend
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.profiles import zarr3_create_spec


def _reference(tmp_path, *, shape=(32, 32, 32), voxel=(8, 8, 8), chunk=(8, 8, 8),
               kind="image", shard=None, name="ref"):
    """A real multiscale volume, built by `convert` so it has genuine metadata."""
    src = str(tmp_path / f"{name}.src.zarr")
    data = np.random.default_rng(0).integers(0, 200, shape, dtype=np.uint8)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), data)
    dst = str(tmp_path / f"{name}.zarr")
    convert(src, dst, voxel_size=voxel, kind=kind, profile="local", chunk=chunk,
            shard=shard, min_dim=8, delete_existing=True)
    return dst


def _precomputed_reference(tmp_path, *, shape=(32, 32, 32), voxel=(40, 8, 8),
                           chunk=(8, 8, 8), name="ref"):
    """A real precomputed volume. Anisotropic by default, because its schedule then
    coarsens x/y before z and nothing may assume the levels are 2**n apart."""
    from em_volume_tools.profiles import StorageProfile

    src = str(tmp_path / f"{name}.pc.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), np.zeros(shape, np.uint8))
    dst = str(tmp_path / f"{name}.precomputed")
    convert(src, dst, voxel_size=voxel, kind="image", min_dim=8, chunk=chunk,
            profile=StorageProfile("neuroglancer_precomputed", chunk=chunk,
                                   compressor="gzip"),
            delete_existing=True)
    return dst


def _attrs(volume):
    return json.load(open(os.path.join(volume, "zarr.json")))["attributes"]["ome"]


def _info(volume):
    return json.load(open(os.path.join(volume, "info")))


# --------------------------------------------------------------------------- #
# --like: the same frame, not a similar one
# --------------------------------------------------------------------------- #
def test_like_reproduces_the_references_frame_exactly(tmp_path):
    """Level shapes and coordinate transforms must be identical, not merely close.

    A schedule recomputed from level 0 can differ from the one the reference actually
    used — shapes are ceil-divided, so one differing --min-dim gives a pyramid that is
    a voxel off partway up. Then an offset that is right in one volume is wrong in the
    other, and nothing reports it.
    """
    ref = _reference(tmp_path, shape=(33, 33, 33))          # deliberately indivisible
    out = str(tmp_path / "out.zarr")
    create_volume(out, like=ref)

    a, b = describe(ref), describe(out)
    assert {i: lv["shape"] for i, lv in b["levels"].items()} == \
           {i: lv["shape"] for i, lv in a["levels"].items()}
    assert b["level_voxel_sizes"] == a["level_voxel_sizes"]
    assert _attrs(out)["multiscales"][0]["datasets"] == \
           _attrs(ref)["multiscales"][0]["datasets"], "coordinate transforms differ"
    assert _attrs(out)["multiscales"][0]["axes"] == _attrs(ref)["multiscales"][0]["axes"]


def test_created_levels_hold_no_data_at_all(tmp_path):
    """An empty pyramid is a few JSON documents; that is what makes this cheap.

    A zarr array with no chunks written reads back as the fill value, so there is
    nothing to allocate up front even for a volume of billions of voxels.
    """
    out = str(tmp_path / "empty.zarr")
    create_volume(out, like=_reference(tmp_path))
    files = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs]
    assert files and all(f.endswith("zarr.json") for f in files), files
    level0 = open_backend({"backend": "zarr3", "path": os.path.join(out, "0")})
    assert not level0.read_region((slice(0, 32),) * 3).any()


def test_dtype_and_kind_can_differ_from_the_reference(tmp_path):
    """The common case: an image volume gives the frame, labels go into it."""
    out = str(tmp_path / "labels.zarr")
    create_volume(out, like=_reference(tmp_path), dtype="uint64", kind="segmentation")
    d = describe(out)
    assert d["dtype"] == "uint64"
    assert d["meta"]["kind"] == "segmentation", \
        "kind is what `downsample` reads to pick mean vs mode; it must survive"
    assert d["levels"][0]["shape"] == (32, 32, 32), "the frame should be unchanged"


def test_levels_limits_how_much_of_the_pyramid_is_created(tmp_path):
    ref = _reference(tmp_path)
    assert len(describe(ref)["levels"]) == 3
    out = str(tmp_path / "one.zarr")
    plan = create_volume(out, like=ref, levels=1)
    assert plan["num_levels"] == 1
    assert sorted(describe(out)["levels"]) == [0]


def test_a_sharded_reference_keeps_its_shard_and_chunk_apart(tmp_path):
    """A sharded level reports the shard as its write chunk and the inner as its read
    chunk; copying the geometry means splitting them back apart, not collapsing them."""
    ref = _reference(tmp_path, shape=(32, 32, 32), chunk=(8, 8, 8), shard=(16, 16, 16))
    out = str(tmp_path / "sh_out.zarr")
    create_volume(out, like=ref)
    lv = describe(out)["levels"][0]
    assert lv["chunks"] == (16, 16, 16) and lv["read_chunks"] == (8, 8, 8)


def test_a_precomputed_reference_can_give_a_zarr_volume_in_its_frame(tmp_path):
    """The reference is usually the precomputed volume already being viewed, and the
    new volume need not be the same format — but it must be the same frame."""
    ref = _precomputed_reference(tmp_path)
    out = str(tmp_path / "from_pc.zarr")
    create_volume(out, like=ref, dtype="uint64", kind="segmentation", profile="local")
    a, b = describe(ref), describe(out)
    assert b["format"] == "zarr3" and a["format"] == "neuroglancer_precomputed"
    assert b["level_voxel_sizes"] == a["level_voxel_sizes"] is not None
    assert {i: lv["shape"] for i, lv in b["levels"].items()} == \
           {i: lv["shape"] for i, lv in a["levels"].items()}


def test_explicit_chunk_overrides_the_reference(tmp_path):
    out = str(tmp_path / "chunked.zarr")
    create_volume(out, like=_reference(tmp_path, chunk=(8, 8, 8)), chunk=(16, 16, 16))
    assert describe(out)["levels"][0]["chunks"] == (16, 16, 16)


# --------------------------------------------------------------------------- #
# precomputed targets
#
# The format difference that matters: precomputed keeps every scale in ONE `info`
# under one prefix, where zarr gives each level its own array and directory.
# --------------------------------------------------------------------------- #
def test_a_precomputed_reference_gives_a_precomputed_volume_by_default(tmp_path):
    """"Like this volume" should include *what kind of volume it is*.

    Silently getting zarr from `--like <precomputed>` is the sort of thing you only
    discover when the viewer cannot open the result.
    """
    ref = _precomputed_reference(tmp_path)
    out = str(tmp_path / "out.precomputed")
    plan = create_volume(out, like=ref, dtype="uint64", kind="segmentation")
    assert plan["format"] == "neuroglancer_precomputed"
    a, b = describe(ref), describe(out)
    assert b["format"] == "neuroglancer_precomputed"
    assert b["level_voxel_sizes"] == a["level_voxel_sizes"]
    assert {i: lv["shape"] for i, lv in b["levels"].items()} == \
           {i: lv["shape"] for i, lv in a["levels"].items()}
    assert b["dtype"] == "uint64" and b["meta"]["kind"] == "segmentation"


def test_an_empty_precomputed_volume_is_one_info_document(tmp_path):
    out = str(tmp_path / "empty.precomputed")
    create_volume(out, like=_reference(tmp_path), format="precomputed")
    files = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs]
    assert files == [os.path.join(out, "info")], files
    assert len(_info(out)["scales"]) == len(describe(out)["levels"]) == 3


def test_every_scale_lands_in_the_one_info_with_its_own_geometry(tmp_path):
    ref = _precomputed_reference(tmp_path)
    out = str(tmp_path / "scales.precomputed")
    create_volume(out, like=ref)
    got = [(s["size"], s["resolution"]) for s in _info(out)["scales"]]
    want = [(s["size"], s["resolution"]) for s in _info(ref)["scales"]]
    assert got == want


def test_the_encoding_follows_the_kind_and_can_be_overridden(tmp_path):
    ref = _reference(tmp_path)
    labels = str(tmp_path / "seg.precomputed")
    create_volume(labels, like=ref, format="precomputed", dtype="uint64",
                  kind="segmentation")
    assert all(s["encoding"] == "compressed_segmentation"
               for s in _info(labels)["scales"])

    image = str(tmp_path / "img.precomputed")
    create_volume(image, like=ref, format="precomputed")
    assert all(s["encoding"] == "raw" for s in _info(image)["scales"])

    raw_labels = str(tmp_path / "raw.precomputed")
    create_volume(raw_labels, like=ref, format="precomputed", dtype="uint64",
                  kind="segmentation", encoding="raw")
    assert all(s["encoding"] == "raw" for s in _info(raw_labels)["scales"])


def test_compressed_segmentation_refuses_a_dtype_it_cannot_encode(tmp_path):
    """Caught here rather than inside tensorstore, whose error does not name this."""
    with pytest.raises(ValueError, match="compressed_segmentation requires"):
        plan_volume(str(tmp_path / "x.precomputed"), shape=(8, 8, 8), dtype="uint8",
                    voxel_size=(8, 8, 8), profile="local-neuroglancer",
                    kind="segmentation")


def test_precomputed_says_it_cannot_shard_rather_than_ignoring_it(tmp_path):
    """`precomputed_create_spec` takes no shard, so a shard here would vanish —
    and the file count it was meant to control is an inode-quota problem on ceph."""
    with pytest.raises(ValueError, match="sharding is not implemented"):
        plan_volume(str(tmp_path / "x.precomputed"), shape=(64, 64, 64), dtype="uint8",
                    voxel_size=(8, 8, 8), profile="local-neuroglancer",
                    chunk=(16, 16, 16), shard=(32, 32, 32))


def test_a_physical_origin_becomes_a_per_scale_voxel_offset(tmp_path):
    """precomputed has no translation transform — the origin lives in each scale's
    integer `voxel_offset`, which means it is expressed in *that scale's* voxels."""
    out = str(tmp_path / "off.precomputed")
    create_volume(out, shape=(64, 64, 64), dtype="uint8", voxel_size=(8, 8, 8),
                  offset=(80, 160, 240), profile="local-neuroglancer",
                  chunk=(16, 16, 16), min_dim=8)
    scales = {s["key"]: s["voxel_offset"] for s in _info(out)["scales"]}
    assert scales["8_8_8"] == [30, 20, 10], "xyz on disk, from (z,y,x) nm /8"
    assert scales["16_16_16"] == [15, 10, 5]
    assert describe(out)["meta"]["offset"] == (80.0, 160.0, 240.0), "reads back"


def test_overwriting_precomputed_does_not_delete_the_scales_it_just_created(tmp_path):
    """Every scale shares one prefix, so `delete_existing` may only apply to scale 0.

    Applying it per scale would take the volume down to whichever scale was written
    last — with the `info` still claiming all of them.
    """
    ref = _precomputed_reference(tmp_path)
    out = str(tmp_path / "twice.precomputed")
    create_volume(out, like=ref)
    with pytest.raises(FileExistsError, match="already exists"):
        create_volume(out, like=ref)
    create_volume(out, like=ref, overwrite=True)
    assert len(_info(out)["scales"]) == 5
    assert sorted(describe(out)["levels"]) == [0, 1, 2, 3, 4]


def test_the_target_profile_follows_format_then_reference_then_destination():
    """One resolver, shared with the CLI, so `--format` cannot drift between them."""
    from em_volume_tools.ops.create import profile_for

    pc = {"format": "neuroglancer_precomputed"}
    assert profile_for(None, pc, "/local/path") == "local-neuroglancer"
    assert profile_for(None, pc, "s3://bucket/prefix") == "s3-neuroglancer"
    assert profile_for(None, {"format": "zarr3"}, "/local/path") == "local"
    assert profile_for(None, None, "/local/path") == "local"
    assert profile_for("precomputed", {"format": "zarr3"}, "/p") == "local-neuroglancer"
    assert profile_for("zarr", pc, "/p") == "local"
    with pytest.raises(ValueError, match="unknown format"):
        profile_for("hdf5", None, "/p")


def test_the_format_can_be_chosen_against_the_references_own(tmp_path):
    zarr_from_pc = str(tmp_path / "z.zarr")
    create_volume(zarr_from_pc, like=_precomputed_reference(tmp_path), format="zarr")
    assert describe(zarr_from_pc)["format"] == "zarr3"

    pc_from_zarr = str(tmp_path / "p.precomputed")
    create_volume(pc_from_zarr, like=_reference(tmp_path), format="precomputed")
    assert describe(pc_from_zarr)["format"] == "neuroglancer_precomputed"


# --------------------------------------------------------------------------- #
# by hand, and what has to be said
# --------------------------------------------------------------------------- #
def test_a_volume_can_be_specified_without_a_reference(tmp_path):
    out = str(tmp_path / "manual.zarr")
    plan = create_volume(out, shape=(64, 64, 64), dtype="uint16",
                         voxel_size=(40.0, 8.0, 8.0), chunk=(16, 16, 16), min_dim=16)
    d = describe(out)
    assert d["dtype"] == "uint16"
    assert d["meta"]["voxel_size"] == (40.0, 8.0, 8.0)
    # anisotropic voxels: the auto schedule coarsens x/y first and leaves z alone
    assert plan["levels"][1]["voxel_size"] == (40.0, 16.0, 16.0)
    assert d["levels"][1]["shape"] == (64, 32, 32)


def test_overriding_the_shape_recomputes_the_pyramid_rather_than_copying_it(tmp_path):
    ref = _reference(tmp_path, shape=(32, 32, 32))
    plan = plan_volume(str(tmp_path / "resized.zarr"), like=ref, shape=(16, 16, 16),
                       min_dim=8)
    assert not plan["mirrored"]
    assert plan["levels"][0]["shape"] == (16, 16, 16)


@pytest.mark.parametrize("missing, kw", [
    ("voxel size", dict(shape=(8, 8, 8), dtype="uint8")),
    ("dtype", dict(shape=(8, 8, 8), voxel_size=(8, 8, 8))),
    ("shape", dict(dtype="uint8", voxel_size=(8, 8, 8))),
])
def test_missing_geometry_says_what_is_missing(tmp_path, missing, kw):
    with pytest.raises(ValueError, match=missing):
        plan_volume(str(tmp_path / "x.zarr"), **kw)


def test_axes_and_voxel_size_must_agree_on_how_many_axes_there_are(tmp_path):
    with pytest.raises(ValueError, match="voxel_size has 2 entries"):
        plan_volume(str(tmp_path / "x.zarr"), shape=(8, 8, 8), dtype="uint8",
                    voxel_size=(8, 8))


# --------------------------------------------------------------------------- #
# not clobbering things
# --------------------------------------------------------------------------- #
def test_an_existing_volume_is_not_replaced_by_accident(tmp_path):
    """Creating over a volume destroys it, so it takes saying so."""
    ref = _reference(tmp_path)
    out = str(tmp_path / "twice.zarr")
    create_volume(out, like=ref)
    with pytest.raises(FileExistsError, match="already exists"):
        create_volume(out, like=ref)
    create_volume(out, like=ref, overwrite=True)          # explicit is fine


def test_overwriting_with_fewer_levels_says_the_extra_ones_remain(tmp_path, caplog):
    """Deleting a level means deleting a key prefix — not something to do implicitly.

    But `existing_levels` probes upward, so a leftover level IS still found, and it
    would then disagree with the group metadata. Say so rather than leave it silent.
    """
    ref = _reference(tmp_path)
    out = str(tmp_path / "shrink.zarr")
    create_volume(out, like=ref)
    with caplog.at_level("WARNING"):
        create_volume(out, like=ref, levels=1, overwrite=True)
    assert "NOT removed" in caplog.text
    assert sorted(describe(out)["levels"]) == [0, 1, 2], "they really are still there"
    assert len(_attrs(out)["multiscales"][0]["datasets"]) == 1


def test_a_volume_of_the_other_format_here_is_refused(tmp_path):
    """The one case where proceeding is silently destructive.

    Each format writes only its own marker, so creating precomputed over a zarr leaves
    `info` and `zarr.json` in one directory — and `detect_backend` checks `info` first,
    so from then on the zarr cannot be opened by anything in this package while its
    chunks still occupy the store. It used to just do it.
    """
    zarr_vol = _reference(tmp_path)
    with pytest.raises(FileExistsError, match="different kind of volume"):
        create_volume(zarr_vol, like=zarr_vol, format="precomputed")

    pc_vol = _precomputed_reference(tmp_path, name="pc")
    with pytest.raises(FileExistsError, match="different kind of volume"):
        create_volume(pc_vol, like=pc_vol, format="zarr")


def test_overwrite_does_not_bypass_the_other_format_guard(tmp_path):
    """--overwrite cannot fix it, so it must not pretend to.

    Neither deletion path cleans up: precomputed's wipes the whole prefix (taking a
    zarr with it, unannounced), zarr's deletes only its own level arrays and leaves a
    stale `info` in charge of detection.
    """
    zarr_vol = _reference(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite will not resolve this"):
        create_volume(zarr_vol, like=zarr_vol, format="precomputed", overwrite=True)
    assert describe(zarr_vol)["format"] == "zarr3", "the zarr must be untouched"


def test_describe_reports_a_second_volume_shadowed_in_the_same_directory(tmp_path):
    """So a directory already in this state can be found, not just prevented."""
    vol = _reference(tmp_path)
    assert describe(vol)["other_markers"] == []
    with open(os.path.join(vol, "info"), "w") as f:
        json.dump({"@type": "neuroglancer_multiscale_volume", "type": "image",
                   "data_type": "uint8", "num_channels": 1,
                   "scales": [{"key": "8_8_8", "size": [32, 32, 32],
                               "resolution": [8, 8, 8], "encoding": "raw",
                               "chunk_sizes": [[8, 8, 8]], "voxel_offset": [0, 0, 0]}]},
                  f)
    d = describe(vol)
    assert d["format"] == "neuroglancer_precomputed", "info wins detection"
    assert d["other_markers"] == ["zarr.json"], "and the zarr underneath is reported"


def test_plan_writes_nothing(tmp_path):
    out = str(tmp_path / "planned.zarr")
    plan = plan_volume(out, like=_reference(tmp_path))
    assert plan["num_levels"] == 3
    assert not os.path.exists(out)
