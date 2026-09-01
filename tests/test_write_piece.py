"""`neu_vol.write_piece`: an in-memory :class:`neu_lib.Piece` back out to HDF5.

The test that matters is the **round trip**, in two forms, because that is the whole claim:

* ``read_piece`` -> transform -> ``write_piece`` -> ``read_piece`` returns the same voxels
  in the same frame, under the same dataset name, still knowing what it is. Everything a
  cleaning pass over a bag of ground-truth crops needs, with no coordinate retyped.
* the file it writes is the file ``pack_hdf5`` writes, so ``neu-vol write`` places it with
  no ``--offset`` and no ``--offset-order``.

The second is what the shared layout helpers in ``ops/pack.py`` are for, and the reason
there is a test asserting the two writers agree attribute for attribute: a divergence there
would not fail anything at write time. It would put two datasets of one file in
disagreeing coordinate systems, and the piece that read back mirrored through the z=x
diagonal would look entirely plausible.
"""

import numpy as np
import pytest

from neu_lib import Frame, Piece
from neu_vol import create_volume, pack_hdf5, read_piece, write_piece
from neu_vol.ops.pack import DEFAULT_DATASET


def _piece(shape=(4, 5, 6), *, voxel=(32.0, 32.0, 32.0), origin=(320.0, 640.0, 960.0),
           kind="segmentation", name="gt/z07901", dtype="uint64"):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return Piece(array=data, frame=Frame(voxel_size_nm=voxel, origin_nm=origin),
                 kind=kind, name=name)


def _attrs(path, dataset):
    import h5py

    with h5py.File(path, "r") as f:
        return ({k: f[dataset].attrs[k] for k in f[dataset].attrs},
                {k: f.attrs[k] for k in f.attrs})


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #
def test_a_piece_survives_the_round_trip_whole(tmp_path):
    """Voxels, frame, kind and dataset name, with nothing passed but the destination."""
    out = str(tmp_path / "cleaned.h5")
    piece = _piece()

    plan = write_piece(piece, out)
    assert plan["dataset"] == "/z07901", "the array half of the piece's own name"

    back = read_piece(out, dataset="/z07901")
    np.testing.assert_array_equal(back.array, piece.array)
    assert back.frame == piece.frame, "voxel size AND origin, not just the shape"
    assert back.kind == "segmentation", \
        "a cleaned segmentation must not come back as an unknown kind"
    assert back.dtype == piece.dtype


def test_a_kind_this_package_did_not_write_is_ignored(tmp_path):
    """`kind` is a plain word; only the vocabulary is evidence.

    Trusting a foreign value is the one that authorises averaging label ids into ids that
    were never in the data, so an unrecognised one reads as unrecorded rather than raising.
    """
    import h5py

    out = str(tmp_path / "foreign.h5")
    write_piece(_piece(kind=None), out)
    with h5py.File(out, "r+") as f:
        f["/z07901"].attrs["kind"] = "labels"        # another tool's vocabulary

    assert read_piece(out, dataset="/z07901").kind is None
    assert read_piece(out, dataset="/z07901", kind="segmentation").kind == "segmentation", \
        "the caller can still say"


def test_the_transform_in_the_middle_is_the_point(tmp_path):
    """The workflow this exists for: read a crop, change it, write it back."""
    src, out = str(tmp_path / "gt.h5"), str(tmp_path / "gt_cleaned.h5")
    write_piece(_piece(name="gt/z07901"), src)

    piece = read_piece(src, dataset="/z07901")
    cleaned = piece.apply(lambda a: np.where(a < 10, 0, a))
    write_piece(cleaned, out)

    back = read_piece(out, dataset="/z07901")
    assert back.array.min() == 0 and (back.array[back.array > 0] >= 10).all()
    assert back.frame == piece.frame, "a transform must not move the piece"


def test_a_bag_of_crops_accumulates_into_one_file(tmp_path):
    """Each piece keeps its own offset; the file keeps one coordinate system."""
    out = str(tmp_path / "set.h5")
    write_piece(_piece(name="gt/z07901", origin=(320.0, 640.0, 960.0)), out)
    write_piece(_piece(shape=(3, 3, 3), name="gt/z08800",
                       origin=(3200.0, 6400.0, 9600.0)), out)

    from neu_vol.backends.hdf5 import datasets

    assert sorted(datasets(out)) == ["/z07901", "/z08800"]
    assert read_piece(out, dataset="/z07901").origin_voxel == (10, 20, 30)
    assert read_piece(out, dataset="/z08800").origin_voxel == (100, 200, 300)


# --------------------------------------------------------------------------- #
# it writes the same file `pack_hdf5` writes
# --------------------------------------------------------------------------- #
def test_the_two_writers_agree_attribute_for_attribute(tmp_path):
    """One layout, two doors — a divergence here fails nothing and mirrors data later."""
    piece = _piece(kind=None)
    mine = str(tmp_path / "mine.h5")
    write_piece(piece, mine, dataset=DEFAULT_DATASET)

    theirs = str(tmp_path / "theirs.h5")
    pack_hdf5(mine, theirs, dataset=DEFAULT_DATASET)

    a_dset, a_root = _attrs(mine, DEFAULT_DATASET)
    b_dset, b_root = _attrs(theirs, DEFAULT_DATASET)
    assert set(a_dset) == set(b_dset) and set(a_root) == set(b_root)
    for key in a_dset:
        np.testing.assert_array_equal(np.asarray(a_dset[key]), np.asarray(b_dset[key]))
    for key in a_root:
        np.testing.assert_array_equal(np.asarray(a_root[key]), np.asarray(b_root[key]))


def test_what_lands_in_the_file(tmp_path):
    out = str(tmp_path / "p.h5")
    write_piece(_piece(voxel=(40.0, 8.0, 8.0), origin=(40.0 * 16, 8.0 * 32, 8.0 * 64)), out)

    dset, root = _attrs(out, "/z07901")
    assert list(dset["voxel_offset"]) == [16, 32, 64]
    assert list(dset["voxel_size"]) == [40, 8, 8]
    assert list(dset["offset"]) == [16 * 40, 32 * 8, 64 * 8], "physical, from the voxels"
    assert dset["axes"] == "zyx" and root["axes"] == "zyx", \
        "a Piece is zyx by construction, so there is no order to guess"
    assert root["units"] == "nm" and dset["kind"] == "segmentation"
    assert list(root["voxel_size"]) == [40, 8, 8], "the file's frame, not just the array's"


def test_neu_vol_write_places_it_with_no_arguments(tmp_path):
    """The claim `voxel_offset` and `axes` exist for, end to end."""
    from neu_vol import write_subvolume
    from neu_vol.backends.base import open_backend

    piece = _piece(shape=(4, 6, 8), voxel=(8.0, 8.0, 8.0), origin=(32.0, 48.0, 64.0),
                   dtype="uint8", kind="segmentation")
    path = str(tmp_path / "piece.h5")
    write_piece(piece, path)

    vol = str(tmp_path / "vol.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint8",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, path, dataset="/z07901")

    assert result["offset"] == (4, 6, 8)
    assert "recorded in the source" in result["offset_from"], \
        "the order came from the file's `axes`, not from a default"
    be = open_backend({"backend": "zarr3", "path": f"{vol}/0"})
    np.testing.assert_array_equal(
        be.read_region((slice(4, 8), slice(6, 12), slice(8, 16))), piece.array)


# --------------------------------------------------------------------------- #
# where the dataset name comes from
# --------------------------------------------------------------------------- #
def test_a_piece_named_after_a_volume_gets_the_default_dataset(tmp_path):
    """No array name to reuse, so the default the reader also assumes."""
    out = str(tmp_path / "crop.h5")
    plan = write_piece(_piece(name="seg_neuprint_export_ts"), out)

    assert plan["dataset"] == DEFAULT_DATASET
    from neu_vol.backends.hdf5 import sole_dataset

    assert sole_dataset(out) == DEFAULT_DATASET, "readable with no arguments"


def test_an_unnamed_piece_gets_the_default_dataset(tmp_path):
    out = str(tmp_path / "anon.h5")
    assert write_piece(_piece(name=None), out)["dataset"] == DEFAULT_DATASET


def test_the_dataset_argument_wins_and_may_nest(tmp_path):
    out = str(tmp_path / "named.h5")
    assert write_piece(_piece(), out, dataset="gt/piece0")["dataset"] == "/gt/piece0"
    assert read_piece(out, dataset="/gt/piece0").shape == (4, 5, 6)


# --------------------------------------------------------------------------- #
# what it refuses, and why each refusal is not pedantry
# --------------------------------------------------------------------------- #
def test_a_disagreeing_frame_is_refused(tmp_path):
    out = str(tmp_path / "set.h5")
    write_piece(_piece(name="gt/a"), out)
    with pytest.raises(ValueError, match="two coordinate systems"):
        write_piece(_piece(name="gt/b", voxel=(8.0, 8.0, 8.0), origin=(0.0, 0.0, 0.0)), out)


def test_a_name_already_in_use_needs_overwrite(tmp_path):
    out = str(tmp_path / "set.h5")
    write_piece(_piece(), out)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_piece(_piece(), out)

    replaced = _piece().apply(lambda a: a * 0 + 7)
    assert write_piece(replaced, out, overwrite=True)["replacing"] is True
    assert (read_piece(out, dataset="/z07901").array == 7).all()


def test_a_fractional_origin_raises_rather_than_rounding(tmp_path):
    """Half a voxel of drift against whatever the piece must line up with, silently."""
    piece = _piece(origin=(330.0, 640.0, 960.0))          # 330 / 32 is not whole
    with pytest.raises(ValueError, match="whole number"):
        write_piece(piece, str(tmp_path / "p.h5"))


def test_a_destination_that_is_not_an_hdf5_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="recognise a container BY NAME"):
        write_piece(_piece(), str(tmp_path / "piece.dat"))


def test_a_remote_destination_is_refused_with_guidance(tmp_path):
    with pytest.raises(ValueError, match="no object-store driver"):
        write_piece(_piece(), "s3://my-bucket/piece.h5")


def test_the_backend_says_where_hdf5_writing_actually_happens(tmp_path):
    """`write_region` is not a stub waiting for someone; it is a signpost."""
    from neu_vol.backends.base import open_backend

    out = str(tmp_path / "p.h5")
    write_piece(_piece(), out)
    be = open_backend({"backend": "hdf5", "path": out, "dataset": "/z07901"})
    with pytest.raises(NotImplementedError, match="write_piece"):
        be.write_region((slice(0, 1), slice(0, 1), slice(0, 1)), np.zeros((1, 1, 1)))


# --------------------------------------------------------------------------- #
# reading a file and then writing to it, in one process
#
# The cleaning pass IS this: read every crop, transform, write. And a re-run of a notebook
# cell is this. An `HDF5Backend` holds an open `h5py.File` for the life of the backend
# cache, and HDF5 refuses to open a file for writing while any handle has it open
# read-only — so both of those failed on the write, with an error naming the file rather
# than the reader still holding it.
# --------------------------------------------------------------------------- #
def test_a_file_can_be_rewritten_after_being_read(tmp_path):
    out = str(tmp_path / "loop.h5")
    write_piece(_piece(), out)

    first = read_piece(out, dataset="/z07901")          # caches an open read handle
    assert first.array.max() == 4 * 5 * 6 - 1

    write_piece(first.apply(lambda a: a * 0 + 7), out, overwrite=True)
    assert read_piece(out, dataset="/z07901").array.max() == 7, \
        "and the second read must not be served from the handle held over the rewrite"


def test_reading_a_source_does_not_block_packing_a_different_file(tmp_path):
    """The same hazard through `pack_hdf5`, which shares the release."""
    src, out = str(tmp_path / "src.h5"), str(tmp_path / "packed.h5")
    write_piece(_piece(), src)
    read_piece(src, dataset="/z07901")                  # a handle on the source

    pack_hdf5(src, out, src_dataset="/z07901")
    assert read_piece(out, dataset=DEFAULT_DATASET).shape == (4, 5, 6)


def test_packing_a_file_into_itself_is_refused_plainly(tmp_path):
    """It never worked; the error used to be about the file being open read-only."""
    path = str(tmp_path / "self.h5")
    write_piece(_piece(), path)
    with pytest.raises(ValueError, match="same file"):
        pack_hdf5(path, path, src_dataset="/z07901", dataset="/copy")


def test_release_backends_reports_and_closes_what_it_dropped(tmp_path):
    from neu_vol import release_backends
    from neu_vol.backends.base import open_backend

    out = str(tmp_path / "held.h5")
    write_piece(_piece(), out)
    be = open_backend({"backend": "hdf5", "path": out, "dataset": "/z07901"})

    assert release_backends(str(tmp_path / "nothing.h5")) == 0, "matched on the path"
    assert release_backends(out) == 1
    with pytest.raises(Exception):
        be.read_region((slice(0, 1), slice(0, 1), slice(0, 1)))    # the handle is closed
    assert open_backend({"backend": "hdf5", "path": out, "dataset": "/z07901"}) is not be, \
        "dropped from the cache, so the next open is a fresh handle"


def test_a_view_over_the_file_is_released_too(tmp_path):
    """A crop or remap backend holds the same handle one level down."""
    from neu_vol import release_backends
    from neu_vol.backends.base import open_backend

    out = str(tmp_path / "viewed.h5")
    write_piece(_piece(), out)
    open_backend({"backend": "remap", "source": {"backend": "hdf5", "path": out,
                                                 "dataset": "/z07901"},
                  "values": [1], "to": 0})
    assert release_backends(out) >= 1, "the nested spec names the path"


# --------------------------------------------------------------------------- #
# the shapes of array a Piece is allowed to hold
# --------------------------------------------------------------------------- #
def test_the_write_is_blocked_so_a_lazy_piece_streams(tmp_path):
    """Several blocks, and the array reassembled from them is the original."""
    out = str(tmp_path / "blocked.h5")
    piece = _piece(shape=(8, 4, 4))
    plan = write_piece(piece, out, block_bytes=4 * 4 * 8)      # two z-planes per block

    assert plan["blocks"] > 1, "the ceiling has to actually bite for this to prove anything"
    np.testing.assert_array_equal(read_piece(out, dataset="/z07901").array, piece.array)


def test_a_lazy_array_is_read_a_block_at_a_time(tmp_path):
    """A Piece may hold an open h5py dataset; writing it must not materialise it whole."""
    import h5py

    src = str(tmp_path / "src.h5")
    with h5py.File(src, "w") as f:
        f.create_dataset("/data", data=np.arange(8 * 4 * 4, dtype="uint64").reshape(8, 4, 4))

    with h5py.File(src, "r") as f:
        lazy = Piece(array=f["/data"], frame=Frame(voxel_size_nm=(8.0, 8.0, 8.0),
                                                  origin_nm=(0.0, 0.0, 0.0)),
                     kind="segmentation", name="src/data")
        assert not isinstance(lazy.array, np.ndarray), "the point of the test"
        out = str(tmp_path / "lazy.h5")
        write_piece(lazy, out, block_bytes=4 * 4 * 8)

    np.testing.assert_array_equal(read_piece(out, dataset="/data").array,
                                  np.arange(8 * 4 * 4, dtype="uint64").reshape(8, 4, 4))


def test_a_channel_axis_is_not_a_position(tmp_path):
    out = str(tmp_path / "multi.h5")
    data = np.zeros((2, 4, 5, 6), dtype="float32")
    piece = Piece(array=data, frame=Frame(voxel_size_nm=(32.0, 32.0, 32.0),
                                          origin_nm=(320.0, 640.0, 960.0)),
                  kind="probability", name="p/data")
    write_piece(piece, out)

    dset, _root = _attrs(out, "/data")
    assert list(dset["voxel_offset"]) == [0, 10, 20, 30], "the channel axis leads with 0"
    assert list(dset["voxel_size"]) == [32, 32, 32], "and is not a physical dimension"
    assert read_piece(out, dataset="/data").shape == (2, 4, 5, 6)


# --------------------------------------------------------------------------- #
# planning without writing
# --------------------------------------------------------------------------- #
def test_dry_run_touches_nothing(tmp_path):
    import os

    out = str(tmp_path / "p.h5")
    plan = write_piece(_piece(), out, dry_run=True)

    assert not os.path.exists(out)
    assert plan["dataset"] == "/z07901" and plan["voxel_offset"] == (10, 20, 30)
    assert plan["kind"] == "segmentation" and plan["dtype"] == "uint64"


def test_the_dtype_may_be_widened_on_the_way_out(tmp_path):
    out = str(tmp_path / "wide.h5")
    write_piece(_piece(dtype="uint16"), out, dtype="uint64")
    assert read_piece(out, dataset="/z07901").dtype == np.dtype("uint64")
