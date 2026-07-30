"""Regenerating a pyramid in place from a chosen level."""

import numpy as np
import pytest

from em_volume_tools import convert, rebuild_pyramid


# min_dim must match between the conversion and the rebuild, or the two compute
# different schedules and disagree on how many levels the pyramid has.
MIN_DIM = 16


def _volume(shape=(128, 128, 128), seed=0):
    """Structured data, so a wrong reducer or a skipped level is visible."""
    rng = np.random.default_rng(seed)
    z = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    x = np.arange(shape[2])[None, None, :]
    ramp = ((z + y + x) % 251).astype(np.uint8)
    return (ramp // 2 + rng.integers(0, 40, shape, dtype=np.uint8)).astype(np.uint8)


def _src(tmp_path, vol, name="src.zarr"):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    path = str(tmp_path / name)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(32, 32, 32)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _levels(dst, fmt="zarr3"):
    """Read every level back as an array."""
    from em_volume_tools.backends.base import open_backend

    out = []
    i = 0
    while True:
        try:
            be = open_backend({"backend": fmt, "path": f"{dst}/{i}"})
        except Exception:
            break
        out.append(be.read_region(tuple(slice(0, s) for s in be.shape)))
        i += 1
    return out


def test_rebuild_reproduces_a_full_conversion(tmp_path):
    """The whole point: rebuilt levels must equal freshly converted ones."""
    vol = _volume()
    src = _src(tmp_path, vol)
    ref = str(tmp_path / "ref.zarr")
    tgt = str(tmp_path / "tgt.zarr")
    for d in (ref, tgt):
        convert(src, d, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    ref_levels = _levels(ref)
    assert len(ref_levels) >= 3, "need a few levels for this test to mean anything"

    # Corrupt every level above 1, then rebuild from 1.
    from em_volume_tools.backends.base import clear_backend_cache, open_backend
    for i in range(2, len(ref_levels)):
        be = open_backend({"backend": "zarr3", "path": f"{tgt}/{i}"})
        be.write_region(tuple(slice(0, s) for s in be.shape),
                        np.zeros(be.shape, np.uint8))
    clear_backend_cache()
    assert not _levels(tgt)[2].any(), "corruption did not take"

    rebuild_pyramid(tgt, start_level=1, kind="image", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    clear_backend_cache()
    got = _levels(tgt)
    assert len(got) == len(ref_levels)
    for i, (a, b) in enumerate(zip(ref_levels, got)):
        np.testing.assert_array_equal(a, b, err_msg=f"level {i} differs after rebuild")


def test_levels_at_or_below_start_are_untouched(tmp_path):
    """The seed level is input, never output — writing it would destroy the source."""
    from em_volume_tools.backends.base import clear_backend_cache, open_backend

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    # Mark levels 0 and 1 with a value the real data cannot produce, and check it
    # survives: if either were recreated or rewritten, the mark would be gone.
    marks = {}
    for i in (0, 1):
        be = open_backend({"backend": "zarr3", "path": f"{dst}/{i}"})
        be.write_region((slice(0, 1), slice(0, 1), slice(0, 1)),
                        np.array([[[255]]], np.uint8))
        marks[i] = 255
    clear_backend_cache()

    rebuild_pyramid(dst, start_level=1, kind="image", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    clear_backend_cache()
    for i, want in marks.items():
        be = open_backend({"backend": "zarr3", "path": f"{dst}/{i}"})
        got = int(be.read_region((slice(0, 1), slice(0, 1), slice(0, 1)))[0, 0, 0])
        assert got == want, f"level {i} was modified by a rebuild that starts at 1"


def test_seed_is_opened_not_created_under_resume_false(tmp_path):
    """resume=False must not recreate the seed — that would erase the input."""
    from em_volume_tools.backends.base import clear_backend_cache, open_backend

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    before = open_backend({"backend": "zarr3", "path": f"{dst}/1"})
    seed_before = before.read_region(tuple(slice(0, s) for s in before.shape)).copy()
    clear_backend_cache()

    rebuild_pyramid(dst, start_level=1, kind="image", chunk=(32, 32, 32),
                    min_dim=MIN_DIM, client=None, resume=False)
    clear_backend_cache()
    after = open_backend({"backend": "zarr3", "path": f"{dst}/1"})
    np.testing.assert_array_equal(
        seed_before, after.read_region(tuple(slice(0, s) for s in after.shape)),
        err_msg="the seed level was rewritten")


def test_start_level_beyond_the_schedule_is_rejected(tmp_path):
    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    res = convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    with pytest.raises(ValueError, match="out of range"):
        rebuild_pyramid(dst, start_level=res["num_levels"] + 3, kind="image",
                        min_dim=MIN_DIM, client=None)
    with pytest.raises(ValueError, match="must be >= 0"):
        rebuild_pyramid(dst, start_level=-1, kind="image", min_dim=MIN_DIM, client=None)


def test_a_seed_whose_shape_contradicts_the_schedule_is_refused(tmp_path):
    """Rebuilding onto a mismatched pyramid would leave levels disagreeing."""
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec
    from em_volume_tools.backends.base import clear_backend_cache

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    # Replace level 1 with an array of the wrong shape.
    TensorStoreBackend.create(
        zarr3_create_spec("local", f"{dst}/1", (99, 99, 99), "uint8",
                          dimension_names=("z", "y", "x"), chunk=(32, 32, 32)),
        delete_existing=True)
    clear_backend_cache()

    with pytest.raises(ValueError, match="schedule predicts"):
        rebuild_pyramid(dst, start_level=1, kind="image", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)


def test_metadata_still_describes_every_level(tmp_path):
    """Skipped levels must remain in the group metadata, not just on disk."""
    import json

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    ref = convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    out = rebuild_pyramid(dst, start_level=1, kind="image", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    assert out["num_levels"] == ref["num_levels"]
    assert out["level_shapes"] == ref["level_shapes"]
    assert out["level_scales"] == ref["level_scales"]

    attrs = json.load(open(f"{dst}/zarr.json"))["attributes"]
    datasets = attrs["ome"]["multiscales"][0]["datasets"]
    assert [d["path"] for d in datasets] == [str(i) for i in range(ref["num_levels"])]


def test_rebuild_uses_a_distinct_progress_file(tmp_path):
    """Reusing a conversion's manifest would mark the rebuilt levels done."""
    import os

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    conv = convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    out = rebuild_pyramid(dst, start_level=1, kind="image", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)

    assert out["progress_path"] != conv["progress_path"]
    assert "regen-from-1" in out["progress_path"]
    assert os.path.exists(out["progress_path"])


def test_precomputed_rebuild(tmp_path):
    """The s3 case, exercised locally: precomputed keeps info per scale."""
    from em_volume_tools.backends.base import clear_backend_cache, open_backend

    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "pc")
    ref = convert(src, dst, profile="s3-neuroglancer", kind="image",
                  voxel_size=(8.0, 8.0, 8.0), encoding="raw",
                  chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    assert ref["num_levels"] >= 3

    def read(i):
        be = open_backend({"backend": "neuroglancer_precomputed", "path": dst,
                           "scale_index": i})
        return be.read_region(tuple(slice(0, s) for s in be.shape))

    want = [read(i) for i in range(ref["num_levels"])]
    clear_backend_cache()

    out = rebuild_pyramid(dst, start_level=1, kind="image", encoding="raw",
                          chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    clear_backend_cache()
    assert out["num_levels"] == ref["num_levels"]
    for i in range(ref["num_levels"]):
        np.testing.assert_array_equal(want[i], read(i), err_msg=f"scale {i} differs")


# --------------------------------------------------------------------------- #
# kind: the reducer, not the dtype. Wrong choice corrupts silently.
# --------------------------------------------------------------------------- #
def _labels(tmp_path, shape=(64, 64, 64)):
    vol = np.zeros(shape, np.uint64)
    vol[:32] = 100
    vol[32:] = 200                     # two labels; a mean would invent 150
    return vol


def test_kind_is_inferred_from_the_volume(tmp_path):
    """Neither reducer is a safe default, so it comes from what the volume records."""
    from em_volume_tools.introspect import detect_backend, read_source_metadata

    src = _src(tmp_path, _labels(tmp_path).astype(np.uint8), name="lab.zarr")
    for prof, sub in (("local", "z"), ("s3-neuroglancer", "pc")):
        dst = str(tmp_path / sub)
        convert(src, dst, profile=prof, kind="segmentation", voxel_size=(8.0, 8.0, 8.0),
                encoding="raw", chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
        meta = read_source_metadata({"backend": detect_backend(dst), "path": dst})
        assert meta["kind"] == "segmentation", f"{prof} lost the volume type"


def test_segmentation_rebuild_does_not_average_labels(tmp_path):
    """A mean reducer on labels invents ids that were never in the data."""
    from em_volume_tools.backends.base import clear_backend_cache, open_backend

    vol = _labels(tmp_path).astype(np.uint8)
    src = _src(tmp_path, vol, name="lab.zarr")
    dst = str(tmp_path / "seg.zarr")
    convert(src, dst, profile="local", kind="segmentation", voxel_size=(8.0, 8.0, 8.0),
            chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    clear_backend_cache()

    # kind not passed -> inferred as segmentation from the volume
    rebuild_pyramid(dst, start_level=1, chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    clear_backend_cache()

    for lvl in _levels(dst)[1:]:
        stray = set(np.unique(lvl)) - {0, 100, 200}
        assert not stray, f"downsampling invented labels {stray}"


def test_kind_must_be_given_when_the_volume_records_none(tmp_path):
    """A bare array carries no type; guessing would be silent corruption."""
    src = _src(tmp_path, _volume(), name="bare.zarr")
    with pytest.raises(ValueError, match="does not record whether"):
        rebuild_pyramid(src["path"], start_level=0, voxel_size=(8.0, 8.0, 8.0),
                        min_dim=MIN_DIM, client=None)


def test_kind_is_validated(tmp_path):
    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
            chunk=(32, 32, 32), min_dim=MIN_DIM, client=None)
    with pytest.raises(ValueError, match="must be 'image' or 'segmentation'"):
        rebuild_pyramid(dst, start_level=1, kind="mean", min_dim=MIN_DIM, client=None)


# --------------------------------------------------------------------------- #
# CLI (scripts/rebuild_pyramid.py)
# --------------------------------------------------------------------------- #
def _cli():
    """Load the script by path; scripts/ is not a package."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rebuild_pyramid.py"
    spec = importlib.util.spec_from_file_location("rebuild_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _converted(tmp_path, **kw):
    src = _src(tmp_path, _volume())
    dst = str(tmp_path / "v.zarr")
    convert(src, dst, profile="local", kind="image", voxel_size=(8.0, 8.0, 8.0),
            chunk=(32, 32, 32), min_dim=MIN_DIM, client=None, **kw)
    return dst


def test_cli_dry_run_touches_nothing(tmp_path):
    from em_volume_tools.backends.base import clear_backend_cache

    dst = _converted(tmp_path)
    before = [a.copy() for a in _levels(dst)]
    clear_backend_cache()

    assert _cli().main([dst, "--start-level", "1", "--min-dim", str(MIN_DIM),
                        "--dry-run"]) == 0
    clear_backend_cache()
    for i, (a, b) in enumerate(zip(before, _levels(dst))):
        np.testing.assert_array_equal(a, b, err_msg=f"--dry-run modified level {i}")


def test_cli_rebuilds(tmp_path):
    from em_volume_tools.backends.base import clear_backend_cache, open_backend

    dst = _converted(tmp_path)
    want = [a.copy() for a in _levels(dst)]
    be = open_backend({"backend": "zarr3", "path": f"{dst}/2"})
    be.write_region(tuple(slice(0, s) for s in be.shape), np.zeros(be.shape, np.uint8))
    clear_backend_cache()

    assert _cli().main([dst, "--start-level", "1", "--min-dim", str(MIN_DIM),
                        "--serial"]) == 0
    clear_backend_cache()
    for i, (a, b) in enumerate(zip(want, _levels(dst))):
        np.testing.assert_array_equal(a, b, err_msg=f"level {i} wrong after CLI rebuild")


def test_cli_refuses_a_schedule_that_disagrees_with_disk(tmp_path):
    """A --min-dim differing from the original conversion must not rebuild."""
    dst = _converted(tmp_path)
    with pytest.raises(SystemExit) as e:
        _cli().main([dst, "--start-level", "1", "--min-dim", "128", "--serial"])
    assert "schedule" in str(e.value) or "exceeds" in str(e.value)


def test_cli_refuses_a_seed_level_beyond_the_schedule(tmp_path):
    dst = _converted(tmp_path)
    with pytest.raises(SystemExit, match="exceeds the deepest level"):
        _cli().main([dst, "--start-level", "99", "--min-dim", str(MIN_DIM), "--serial"])


def test_cli_refuses_a_seed_level_that_is_absent_on_disk(tmp_path):
    """In the schedule but not written — there is nothing to derive from."""
    import shutil

    from em_volume_tools.backends.base import clear_backend_cache

    dst = _converted(tmp_path)
    shutil.rmtree(f"{dst}/2")
    clear_backend_cache()
    with pytest.raises(SystemExit, match="does not exist"):
        _cli().main([dst, "--start-level", "2", "--min-dim", str(MIN_DIM), "--serial"])
