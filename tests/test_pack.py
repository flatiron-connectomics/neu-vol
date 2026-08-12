"""`em-vol to-hdf5`: pack a small volume into an HDF5 file `em-vol write` can place.

The test that matters is the **round trip** — pack a stack, then write it into a volume
with no `--offset` and no `--offset-order` — because that is the whole claim: the frame and
the position travel with the data. Everything else here guards a way of getting that
silently wrong, chiefly the axis order, which reversed puts the piece mirrored through the
z=x diagonal with nothing downstream able to tell.
"""

import numpy as np
import pytest

from em_volume_tools import cli, create_volume, pack_hdf5
from em_volume_tools.backends.base import open_backend
from em_volume_tools.ops.pack import DEFAULT_DATASET


def _stack(tmp_path, name="stack", *, shape=(4, 6, 8), start=1):
    """A directory of ordered PNG slices, each plane distinct."""
    from PIL import Image

    d = tmp_path / name
    d.mkdir()
    data = np.arange(start, start + np.prod(shape), dtype=np.uint8).reshape(shape)
    for z in range(shape[0]):
        Image.fromarray(data[z]).save(d / f"s{z:03d}.png")
    return str(d), data


def _h5(path, dataset=DEFAULT_DATASET):
    import h5py

    with h5py.File(path, "r") as f:
        dset = f[dataset]
        return (np.asarray(dset[()]),
                {k: dset.attrs[k] for k in dset.attrs},
                {k: f.attrs[k] for k in f.attrs})


# --------------------------------------------------------------------------- #
# what lands in the file
# --------------------------------------------------------------------------- #
def test_the_stack_and_its_frame_are_written(tmp_path):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "piece.h5")
    plan = pack_hdf5(src, out, voxel_size=(40, 8, 8), voxel_offset=(16, 32, 64),
                     src_format="image_stack")

    arr, dattrs, rattrs = _h5(out)
    np.testing.assert_array_equal(arr, data)
    assert plan["dataset"] == DEFAULT_DATASET, "the default the reader also assumes"
    assert list(dattrs["voxel_offset"]) == [16, 32, 64]
    assert list(dattrs["voxel_size"]) == [40, 8, 8]
    assert list(dattrs["offset"]) == [16 * 40, 32 * 8, 64 * 8], "physical, from the voxels"
    assert dattrs["axes"] == "zyx" and rattrs["axes"] == "zyx"
    assert rattrs["units"] == "nm"
    assert list(rattrs["voxel_size"]) == [40, 8, 8], "the file's frame, not just the array's"


def test_the_default_dataset_is_what_the_reader_looks_for(tmp_path):
    """Packed with no arguments, read with no arguments."""
    src, data = _stack(tmp_path)
    out = str(tmp_path / "p.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")

    from em_volume_tools.backends.hdf5 import sole_dataset

    assert sole_dataset(out) == DEFAULT_DATASET
    be = open_backend({"backend": "hdf5", "path": out})       # no `dataset` key
    np.testing.assert_array_equal(be.read_region((slice(0, 4), slice(0, 6), slice(0, 8))),
                                  data)


def test_a_named_dataset_and_chunking_are_honoured(tmp_path):
    import h5py

    src, _ = _stack(tmp_path)
    out = str(tmp_path / "named.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dataset="gt/piece0", chunk=(2, 3, 4),
              compression=None, src_format="image_stack")
    with h5py.File(out, "r") as f:
        assert f["/gt/piece0"].chunks == (2, 3, 4)
        assert f["/gt/piece0"].compression is None


def test_the_dtype_can_be_cast(tmp_path):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "cast.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dtype="uint32", src_format="image_stack")
    arr, _, _ = _h5(out)
    assert arr.dtype == np.uint32
    np.testing.assert_array_equal(arr, data)


def test_a_big_source_is_read_in_blocks(tmp_path):
    """A "small volume" that is not stays packable: the array is never held whole."""
    src, data = _stack(tmp_path, shape=(8, 6, 8))
    out = str(tmp_path / "blocked.h5")
    plan = pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack",
                     max_bytes=6 * 8)                       # one plane at a time
    assert plan["blocks"] == 8 and plan["block_shape"] == (1, 6, 8)
    np.testing.assert_array_equal(_h5(out)[0], data)


# --------------------------------------------------------------------------- #
# adding to an existing file
# --------------------------------------------------------------------------- #
def test_a_second_piece_can_join_a_file_with_the_same_frame(tmp_path, caplog):
    """Several pieces of one volume in one file, each with its own voxel_offset."""
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "two.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(0, 0, 0), dataset="a",
              src_format="image_stack")
    with caplog.at_level("WARNING"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(4, 0, 0), dataset="b",
                  src_format="image_stack")

    import h5py

    with h5py.File(out, "r") as f:
        assert list(f["/a"].attrs["voxel_offset"]) == [0, 0, 0]
        assert list(f["/b"].attrs["voxel_offset"]) == [4, 0, 0]
    assert "readers must name one" in caplog.text, \
        "sole_dataset stops working, so say so before it does"


def test_a_frame_that_disagrees_is_refused(tmp_path):
    """One file describing two coordinate systems is not a thing worth allowing."""
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "clash.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dataset="a", src_format="image_stack")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(4, 4, 4), dataset="b", src_format="image_stack")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), units="um", dataset="c",
                  src_format="image_stack")


def test_reusing_a_name_needs_overwrite(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "same.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")
    with pytest.raises(FileExistsError, match="already has a dataset"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")

    src2, data2 = _stack(tmp_path, "other", start=100)
    plan = pack_hdf5(src2, out, voxel_size=(8, 8, 8), src_format="image_stack",
                     overwrite=True)
    assert plan["replacing"]
    np.testing.assert_array_equal(_h5(out)[0], data2)


def test_a_mismatched_axes_count_is_refused(tmp_path):
    src, _ = _stack(tmp_path)
    with pytest.raises(ValueError, match="same number of axes"):
        pack_hdf5(src, str(tmp_path / "x.h5"), voxel_size=(8, 8),
                  src_format="image_stack")


# --------------------------------------------------------------------------- #
# the round trip, which is the point
# --------------------------------------------------------------------------- #
def test_the_packed_piece_writes_back_with_no_offset_and_no_order(tmp_path):
    """Pack, then place — the frame and position travel with the data.

    No --offset, and no --offset-order either: the recorded `axes` answers the question
    that previously had to be asked.
    """
    from em_volume_tools import write_subvolume

    src, data = _stack(tmp_path)
    piece = str(tmp_path / "piece.h5")
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(4, 6, 8),
              src_format="image_stack")

    vol = str(tmp_path / "vol.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint8",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, piece)

    assert result["offset"] == (4, 6, 8)
    assert "voxel_offset" in result["offset_from"]
    assert "recorded in the source" in result["offset_from"], \
        "the order came from the file, not from a default"
    be = open_backend({"backend": "zarr3", "path": f"{vol}/0"})
    np.testing.assert_array_equal(
        be.read_region((slice(4, 8), slice(6, 12), slice(8, 16))), data)


def test_an_xyz_file_is_read_as_xyz_without_being_told(tmp_path):
    """The mirroring bug, prevented by the file saying which order it used.

    Packing with axes="xyz" records that, so `write` reverses the offset itself — where
    before, the numbers were indistinguishable from zyx and a wrong guess put the piece
    somewhere else entirely.
    """
    from em_volume_tools import write_subvolume

    src, data = _stack(tmp_path, shape=(4, 4, 4))
    piece = str(tmp_path / "xyz.h5")
    # the same place, written the other way round: xyz (x=8, y=6, z=4)
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(8, 6, 4), axes=("x", "y", "z"),
              src_format="image_stack")

    vol = str(tmp_path / "v.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint8",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, piece)
    assert result["offset"] == (4, 6, 8), "reversed to zyx on the file's own authority"
    assert "read as xyz" in result["offset_from"]
    assert "recorded in the source" in result["offset_from"]


def test_an_explicit_order_still_wins_over_the_file(tmp_path):
    from em_volume_tools.ops.write import resolve_offset

    src, _ = _stack(tmp_path, shape=(4, 4, 4))
    piece = str(tmp_path / "o.h5")
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(1, 2, 3),
              src_format="image_stack")
    be = open_backend({"backend": "hdf5", "path": piece})
    assert be.stored_axes()[0] == "zyx"
    off, prov = resolve_offset(be, None, field="voxel_offset", order="xyz", ndim=3)
    assert off == (3, 2, 1) and "read as xyz" in prov
    assert "recorded in the source" not in prov, "an explicit order is not the file's"


# --------------------------------------------------------------------------- #
# a box out of a volume, at a level
# --------------------------------------------------------------------------- #
@pytest.fixture
def pyramid(tmp_path):
    """A 3-level anisotropic zarr group, so a level's voxel size is not 2**level."""
    from em_volume_tools import convert
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    data = np.arange(32 * 32 * 32, dtype=np.uint16).reshape(32, 32, 32)
    src = str(tmp_path / "p.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint16",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "pyr.zarr")
    convert(src, dst, voxel_size=(40, 8, 8), profile="local", chunk=(8, 8, 8), min_dim=8,
            delete_existing=True)
    return dst, data


def test_a_zarr_group_can_be_packed_at_all(tmp_path, pyramid):
    """It could not: the spec was built straight from the path, and a group is not an array.

    tensorstore's error was a wall of driver text about `node_type`, which says nothing
    about what the caller did wrong.
    """
    vol, data = pyramid
    out = str(tmp_path / "grp.h5")
    pack_hdf5(vol, out, voxel_size=(40, 8, 8))
    np.testing.assert_array_equal(_h5(out)[0], data)


def test_the_level_supplies_the_voxel_size_it_records(tmp_path, pyramid):
    """No retyping a scale the volume already knows — and never 2**level for it."""
    vol, _ = pyramid
    out = str(tmp_path / "lvl.h5")
    plan = pack_hdf5(vol, out, level=1)
    assert plan["voxel_size"] == (40.0, 16.0, 16.0), "z is coarse already, so (1,2,2)"
    assert list(_h5(out)[2]["voxel_size"]) == [40, 16, 16]

    assert pack_hdf5(vol, str(tmp_path / "l2.h5"), level=2)["voxel_size"] == \
        (40.0, 32.0, 32.0)


def test_a_box_is_packed_and_its_origin_becomes_the_offset(tmp_path, pyramid):
    """The one number nobody should type twice: where the piece came from."""
    vol, data = pyramid
    out = str(tmp_path / "box.h5")
    plan = pack_hdf5(vol, out, crop_start=(4, 8, 12), crop_stop=(12, 16, 20))

    assert plan["shape"] == (8, 8, 8)
    assert plan["voxel_offset"] == (4, 8, 12), "taken from the crop origin"
    np.testing.assert_array_equal(_h5(out)[0], data[4:12, 8:16, 12:20])
    assert list(_h5(out)[1]["voxel_offset"]) == [4, 8, 12]


def test_an_explicit_offset_overrides_the_crop_origin(tmp_path, pyramid):
    """Extracting from one volume to place into another somewhere else is legitimate."""
    vol, _ = pyramid
    plan = pack_hdf5(vol, str(tmp_path / "o.h5"), crop_start=(4, 4, 4),
                     crop_stop=(8, 8, 8), voxel_offset=(0, 0, 0))
    assert plan["voxel_offset"] == (0, 0, 0)


def test_a_box_at_a_level_round_trips_into_that_level(tmp_path, pyramid):
    """Extract a region at level 1, write it back at level 1, unchanged."""
    from em_volume_tools import write_subvolume

    vol, _ = pyramid
    piece = str(tmp_path / "rt.h5")
    pack_hdf5(vol, piece, level=1, crop_start=(2, 2, 2), crop_stop=(10, 10, 10))
    before = open_backend({"backend": "zarr3", "path": f"{vol}/1"}).read_region(
        (slice(2, 10),) * 3).copy()

    result = write_subvolume(vol, piece, level=1)
    assert result["offset"] == (2, 2, 2)
    after = open_backend({"backend": "zarr3", "path": f"{vol}/1"}).read_region(
        (slice(2, 10),) * 3)
    np.testing.assert_array_equal(after, before)


def test_a_level_on_a_source_that_has_none_is_refused(tmp_path):
    src, _ = _stack(tmp_path)
    with pytest.raises(ValueError, match="needs a multiscale volume"):
        pack_hdf5(src, str(tmp_path / "x.h5"), voxel_size=(8, 8, 8), level=2)


def test_a_level_the_volume_does_not_have_is_refused(tmp_path, pyramid):
    vol, _ = pyramid
    with pytest.raises(ValueError, match="has no level 9"):
        pack_hdf5(vol, str(tmp_path / "x.h5"), level=9)


def test_a_source_recording_no_scale_still_demands_one(tmp_path):
    src, _ = _stack(tmp_path)
    with pytest.raises(ValueError, match="voxel_size is required"):
        pack_hdf5(src, str(tmp_path / "x.h5"))


# --------------------------------------------------------------------------- #
# what the fields are called
# --------------------------------------------------------------------------- #
def test_the_voxel_size_field_can_be_renamed_for_both_directions(tmp_path):
    """Other tools have their own word for it, and a file should stay readable by them.

    The same name is used to write and to read, so a repacked file keeps whatever spelling
    its siblings use.
    """
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "named_field.h5")
    pack_hdf5(src, out, voxel_size=(40, 8, 8), voxel_size_field="resolution")

    _, dattrs, rattrs = _h5(out)
    assert list(rattrs["resolution"]) == [40, 8, 8] and "voxel_size" not in rattrs
    assert list(dattrs["resolution"]) == [40, 8, 8]

    # and reading it back needs no --voxel-size, given the same field name
    again = str(tmp_path / "again.h5")
    plan = pack_hdf5(out, again, voxel_size_field="resolution")
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)


def test_an_hdf5_source_supplies_its_own_scale(tmp_path):
    """Repacking a file that already records a voxel size must not ask for it again."""
    src, _ = _stack(tmp_path)
    first = str(tmp_path / "first.h5")
    pack_hdf5(src, first, voxel_size=(40, 8, 8), voxel_offset=(1, 2, 3))

    plan = pack_hdf5(first, str(tmp_path / "second.h5"))
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)
    assert plan["axes"] == ("z", "y", "x"), "the recorded order comes along too"


def test_the_offset_field_can_be_renamed(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "off.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(1, 2, 3),
              offset_field="corner")
    _, dattrs, _ = _h5(out)
    assert list(dattrs["corner"]) == [1, 2, 3] and "voxel_offset" not in dattrs

    # `write` looks for voxel_offset, so it has to be told the new name
    from em_volume_tools.ops.write import resolve_offset

    be = open_backend({"backend": "hdf5", "path": out})
    assert be.stored_offset("voxel_offset") is None
    assert resolve_offset(be, None, field="corner", order=None, ndim=3)[0] == (1, 2, 3)


def test_a_renamed_frame_is_still_compared_on_append(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "cmp.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_size_field="resolution", dataset="a")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(4, 4, 4), voxel_size_field="resolution",
                  dataset="b")


def test_write_warns_when_the_pieces_scale_is_not_the_levels(tmp_path, pyramid, caplog):
    """A piece from another level fits, places cleanly, and is at the wrong resolution.

    Nothing else in `write` can see it: the shapes, dtype and bounds are all fine.
    """
    from em_volume_tools import write_subvolume

    vol, _ = pyramid
    piece = str(tmp_path / "lvl1.h5")
    pack_hdf5(vol, piece, level=1, crop_start=(0, 0, 0), crop_stop=(4, 4, 4))

    with caplog.at_level("WARNING"):
        result = write_subvolume(vol, piece, level=0)      # level 1 data into level 0
    assert "records a voxel size" in caplog.text and "check --level" in caplog.text
    assert result["scale_note"]

    caplog.clear()
    with caplog.at_level("WARNING"):
        result = write_subvolume(vol, piece, level=1)      # back where it belongs
    assert result["scale_note"] is None
    assert "records a voxel size" not in caplog.text


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_packs_and_reports(tmp_path, capsys):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "cli.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--src-format",
                     "image_stack", "--voxel-size", "40,8,8", "--offset", "2,4,6"]) == 0
    printed = capsys.readouterr().out
    assert "40x8x8 nm" in printed and "z 2  y 4  x 6" in printed
    assert "axes        zyx" in printed
    np.testing.assert_array_equal(_h5(out)[0], data)


def test_the_cli_dry_run_writes_nothing(tmp_path, capsys):
    import os

    src, _ = _stack(tmp_path)
    out = str(tmp_path / "none.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--src-format",
                     "image_stack", "--voxel-size", "8,8,8", "--dry-run"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert not os.path.exists(out)


def test_a_remote_path_is_refused_with_a_useful_error(tmp_path):
    """h5py has no object-store driver, and its own error does not say so.

    Left to it, it tries to *create a local file called* `s3://bucket/piece.h5` and reports
    `errno = 2, No such file or directory`.
    """
    src, _ = _stack(tmp_path)
    with pytest.raises(ValueError, match="ordinary filesystem path"):
        pack_hdf5(src, "s3://bucket/piece.h5", voxel_size=(8, 8, 8))
    with pytest.raises(ValueError, match="rclone|aws s3 cp"):
        open_backend({"backend": "hdf5", "path": "s3://bucket/piece.h5"})


def test_the_cli_refuses_a_bad_axes_string(tmp_path):
    src, _ = _stack(tmp_path)
    with pytest.raises(SystemExit, match="permutation of zyx"):
        cli.main(["to-hdf5", "--src", src, "--out", str(tmp_path / "b.h5"),
                  "--src-format", "image_stack", "--voxel-size", "8,8,8",
                  "--axes", "zyz"])
