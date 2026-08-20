"""`write`: one subvolume into an existing volume, at a voxel offset and one level.

Two things carry the weight here. **Placement** — the piece lands exactly where it was
asked to and nothing else in the volume moves — and **tiling** — no two tiles ever
share a destination chunk, because tiles that do would race if this were ever run
concurrently, and a lost partial-chunk write leaves no trace to find afterwards.
"""

import os

import numpy as np
import pytest

from neu_vol import create_volume, write_subvolume, write_subvolumes
from neu_vol.backends.base import open_backend
from neu_vol.ops.write import _tiles, plan_subvolume_write, source_spec


def _volume(tmp_path, *, shape=(64, 64, 64), dtype="uint64", chunk=(16, 16, 16),
            voxel=(8.0, 8.0, 8.0), levels=3, name="vol"):
    dst = str(tmp_path / f"{name}.zarr")
    create_volume(dst, shape=shape, dtype=dtype, voxel_size=voxel, chunk=chunk,
                  levels=levels, min_dim=8, kind="segmentation")
    return dst


def _h5(tmp_path, data, *, dataset="labels", name="piece"):
    import h5py

    path = str(tmp_path / f"{name}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=data)
    return path


def _png_stack(tmp_path, data, *, name="stack"):
    import imageio.v3 as iio

    d = tmp_path / name
    d.mkdir()
    for z in range(data.shape[0]):
        iio.imwrite(str(d / f"s{z:03d}.png"), data[z])
    return str(d)


def _level(volume, level):
    return open_backend({"backend": "zarr3", "path": os.path.join(volume, str(level))})


def _region(start, shape):
    return tuple(slice(a, a + s) for a, s in zip(start, shape))


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def test_an_hdf5_piece_lands_where_it_was_put_and_nowhere_else(tmp_path):
    vol = _volume(tmp_path)
    piece = np.random.default_rng(1).integers(1, 1000, (8, 12, 20), dtype=np.uint64)
    write_subvolume(vol, _h5(tmp_path, piece), (16, 16, 16))

    be = _level(vol, 0)
    np.testing.assert_array_equal(be.read_region(_region((16, 16, 16), piece.shape)),
                                  piece)
    whole = be.read_region((slice(0, 64),) * 3)
    whole[_region((16, 16, 16), piece.shape)] = 0
    assert not whole.any(), "something outside the region was written"


def test_an_image_stack_piece_round_trips(tmp_path):
    vol = _volume(tmp_path, dtype="uint8")
    piece = np.random.default_rng(2).integers(0, 255, (5, 24, 24), dtype=np.uint8)
    write_subvolume(vol, _png_stack(tmp_path, piece), (3, 8, 8))
    np.testing.assert_array_equal(
        _level(vol, 0).read_region(_region((3, 8, 8), piece.shape)), piece)


def test_several_pieces_coexist_in_one_volume(tmp_path):
    """The whole point of create-then-write: independent pieces, one frame."""
    vol = _volume(tmp_path)
    rng = np.random.default_rng(3)
    pieces = {(0, 0, 0): rng.integers(1, 9, (8, 8, 8), dtype=np.uint64),
              (16, 16, 16): rng.integers(10, 19, (8, 8, 8), dtype=np.uint64),
              (40, 4, 52): rng.integers(20, 29, (8, 8, 8), dtype=np.uint64)}
    for i, (at, data) in enumerate(pieces.items()):
        write_subvolume(vol, _h5(tmp_path, data, name=f"p{i}"), at)
    be = _level(vol, 0)
    for at, data in pieces.items():
        np.testing.assert_array_equal(be.read_region(_region(at, data.shape)), data)


def test_writing_a_level_leaves_every_other_level_alone(tmp_path):
    """Single-scale on purpose — coarsening a patch is a separate decision."""
    vol = _volume(tmp_path)
    piece = np.full((8, 8, 8), 7, np.uint64)
    write_subvolume(vol, _h5(tmp_path, piece), (0, 0, 0), level=1)
    assert _level(vol, 1).read_region(_region((0, 0, 0), piece.shape)).max() == 7
    assert not _level(vol, 0).read_region((slice(0, 64),) * 3).any()
    assert not _level(vol, 2).read_region((slice(0, 16),) * 3).any()


# --------------------------------------------------------------------------- #
# offsets across levels
# --------------------------------------------------------------------------- #
def test_an_offset_can_be_given_in_another_levels_voxels(tmp_path):
    """You read coordinates off level 0 but write the piece at level 2."""
    vol = _volume(tmp_path)
    piece = np.full((4, 4, 4), 5, np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (16, 32, 48), level=2,
                        offset_level=0)
    assert r["start"] == (4, 8, 12), "8 nm -> 32 nm is a factor of 4"
    np.testing.assert_array_equal(
        _level(vol, 2).read_region(_region((4, 8, 12), piece.shape)), piece)


def test_an_offset_that_is_not_a_whole_coarse_voxel_is_refused(tmp_path):
    """Rounding it would shift the piece by up to half a coarse voxel, silently."""
    vol = _volume(tmp_path)
    piece = np.ones((2, 2, 2), np.uint64)
    with pytest.raises(ValueError, match="not a whole number of level-2 voxels"):
        write_subvolume(vol, _h5(tmp_path, piece), (17, 0, 0), level=2, offset_level=0)


def test_the_offset_is_taken_at_the_written_level_by_default(tmp_path):
    vol = _volume(tmp_path)
    piece = np.full((4, 4, 4), 3, np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (8, 8, 8), level=1)
    assert r["start"] == (8, 8, 8)


# --------------------------------------------------------------------------- #
# a batch of sources
#
# The pieces arrive as a set of files, each carrying its own offset, so the batch is
# the natural unit. What it buys over a shell loop is that ALL of them are checked
# before ANY is written.
# --------------------------------------------------------------------------- #
def test_a_batch_of_sources_each_lands_at_its_own_stored_offset(tmp_path):
    vol = _volume(tmp_path)
    rng = np.random.default_rng(11)
    pieces = {(0, 0, 0): None, (16, 24, 32): None, (40, 8, 8): None}
    srcs = []
    for i, at in enumerate(pieces):
        pieces[at] = rng.integers(1, 99, (8, 10, 12), dtype=np.uint64)
        srcs.append(_h5_with_offset(tmp_path, pieces[at], at, name=f"batch{i}"))

    results = write_subvolumes(vol, srcs)
    assert [r["start"] for r in results] == list(pieces)
    got = _level(vol, 0).read_region((slice(0, 64),) * 3)
    for at, data in pieces.items():
        np.testing.assert_array_equal(got[_region(at, data.shape)], data)
        got[_region(at, data.shape)] = 0
    assert not got.any(), "something outside the pieces was written"


def test_one_bad_source_stops_the_batch_before_anything_is_written(tmp_path):
    """Planning resolves offsets and checks bounds and touches nothing, so doing all
    of it first means a mistake in the last file is caught while the volume is still
    clean — rather than after the earlier pieces have already landed in it."""
    vol = _volume(tmp_path, shape=(32, 32, 32))
    good = _h5_with_offset(tmp_path, np.ones((8, 8, 8), np.uint64), (0, 0, 0),
                           name="good")
    bad = _h5_with_offset(tmp_path, np.ones((8, 8, 8), np.uint64), (30, 30, 30),
                          name="bad")
    with pytest.raises(ValueError, match="does not fit"):
        write_subvolumes(vol, [good, bad])
    assert not _level(vol, 0).read_region((slice(0, 32),) * 3).any(), \
        "the good source was written before the bad one was checked"


def test_offsets_may_be_given_per_source_or_left_to_the_sources(tmp_path):
    vol = _volume(tmp_path)
    a = _h5(tmp_path, np.full((4, 4, 4), 1, np.uint64), name="pa")
    b = _h5(tmp_path, np.full((4, 4, 4), 2, np.uint64), name="pb")
    results = write_subvolumes(vol, [a, b], [(0, 0, 0), (16, 16, 16)])
    assert [r["start"] for r in results] == [(0, 0, 0), (16, 16, 16)]


def test_a_partial_list_of_offsets_is_refused(tmp_path):
    """One offset for three sources cannot mean anything sensible — pieces that all
    belong at the same place are one piece."""
    vol = _volume(tmp_path)
    src = _h5(tmp_path, np.ones((4, 4, 4), np.uint64))
    with pytest.raises(ValueError, match="2 source.* but 1 offset"):
        write_subvolumes(vol, [src, src], [(0, 0, 0)])


def test_overlapping_sources_are_reported_but_allowed(tmp_path):
    """Overwriting part of an earlier piece can be deliberate. Silence would not be:
    the result looks identical either way, so a mistyped offset leaves no trace."""
    vol = _volume(tmp_path)
    a = _h5_with_offset(tmp_path, np.full((8, 8, 8), 1, np.uint64), (0, 0, 0), name="oa")
    b = _h5_with_offset(tmp_path, np.full((4, 4, 4), 2, np.uint64), (4, 4, 4), name="ob")
    far = _h5_with_offset(tmp_path, np.full((4, 4, 4), 3, np.uint64), (32, 32, 32),
                          name="oc")
    results = write_subvolumes(vol, [a, b, far])
    assert results[0]["overlaps"] == [(0, 1)], "only the two that really overlap"
    got = _level(vol, 0).read_region((slice(0, 64),) * 3)
    assert got[4, 4, 4] == 2, "the later source wins where they meet"
    assert got[0, 0, 0] == 1


def test_a_single_source_batch_behaves_like_the_single_write(tmp_path):
    vol = _volume(tmp_path)
    piece = np.ones((4, 4, 4), np.uint64)
    results = write_subvolumes(vol, [_h5(tmp_path, piece)], [(8, 8, 8)])
    assert len(results) == 1 and results[0]["written"] == results[0]["num_tiles"]


def test_a_dry_run_batch_writes_nothing_and_still_checks_everything(tmp_path):
    vol = _volume(tmp_path)
    srcs = [_h5_with_offset(tmp_path, np.ones((4, 4, 4), np.uint64), at, name=f"d{i}")
            for i, at in enumerate([(0, 0, 0), (16, 16, 16)])]
    results = write_subvolumes(vol, srcs, dry_run=True)
    assert all(r["dry_run"] and r["written"] == 0 for r in results)
    assert [r["start"] for r in results] == [(0, 0, 0), (16, 16, 16)]
    assert not _level(vol, 0).read_region((slice(0, 64),) * 3).any()


# --------------------------------------------------------------------------- #
# an offset the source already knows
#
# Writers routinely record where a subvolume came from beside the array. Re-typing
# that by hand is tedious and a chance to mistype a coordinate, so the offset may be
# omitted and read from the source. It is an optional *backend* capability
# (`stored_offset`), not an HDF5 branch in the write op — HDF5 is just the only
# backend that answers today.
# --------------------------------------------------------------------------- #
def _h5_with_offset(tmp_path, data, offset, *, where="dataset", name="off",
                    field="voxel_offset", dataset="main"):
    """An HDF5 file recording its offset in one of the three places writers use."""
    import h5py

    path = str(tmp_path / f"{name}.h5")
    with h5py.File(path, "w") as f:
        d = f.create_dataset(dataset, data=data)
        if where == "dataset":
            f.create_dataset(field, data=np.asarray(offset))
        elif where == "root_attr":
            f.attrs[field] = np.asarray(offset)
        elif where == "dataset_attr":
            d.attrs[field] = np.asarray(offset)
    return path


def test_the_offset_can_come_from_the_source_instead_of_the_command_line(tmp_path):
    vol = _volume(tmp_path)
    piece = np.random.default_rng(10).integers(1, 99, (8, 10, 12), dtype=np.uint64)
    r = write_subvolume(vol, _h5_with_offset(tmp_path, piece, (16, 24, 32)))
    assert r["start"] == (16, 24, 32)
    assert r["offset_from"] == "from the source, /voxel_offset"
    np.testing.assert_array_equal(
        _level(vol, 0).read_region(_region((16, 24, 32), piece.shape)), piece)


@pytest.mark.parametrize("where, shown", [
    ("dataset", "/voxel_offset"),
    ("root_attr", "/.attrs['voxel_offset']"),
    ("dataset_attr", "/main.attrs['voxel_offset']"),
])
def test_all_three_places_a_writer_might_put_it_are_searched(tmp_path, where, shown):
    """And which one it came from is reported, because they can disagree."""
    vol = _volume(tmp_path)
    piece = np.ones((4, 4, 4), np.uint64)
    r = write_subvolume(vol, _h5_with_offset(tmp_path, piece, (8, 8, 8), where=where,
                                             name=where), dry_run=True)
    assert r["start"] == (8, 8, 8)
    assert r["offset_from"] == f"from the source, {shown}"


def test_the_most_specific_recorded_offset_wins(tmp_path):
    """A dataset's own attribute beats a file-wide one, which beats a stray dataset."""
    import h5py

    path = str(tmp_path / "layered.h5")
    with h5py.File(path, "w") as f:
        d = f.create_dataset("main", data=np.ones((4, 4, 4), np.uint64))
        f.create_dataset("voxel_offset", data=np.array([1, 1, 1]))
        f.attrs["voxel_offset"] = np.array([2, 2, 2])
        d.attrs["voxel_offset"] = np.array([3, 3, 3])
    r = write_subvolume(_volume(tmp_path), path, dry_run=True)
    assert r["start"] == (3, 3, 3)


def test_an_explicit_offset_beats_the_stored_one(tmp_path):
    vol = _volume(tmp_path)
    piece = np.ones((4, 4, 4), np.uint64)
    r = write_subvolume(vol, _h5_with_offset(tmp_path, piece, (16, 24, 32)), (0, 0, 0),
                        dry_run=True)
    assert r["start"] == (0, 0, 0) and r["offset_from"] == "given"


def test_a_stored_offset_can_be_read_as_xyz(tmp_path):
    """`voxel_offset` is precomputed's field name, and precomputed means XYZ.

    Nothing in the file distinguishes the two orders, so it is asked for rather than
    guessed — reversed, the piece lands mirrored through the z=x diagonal and nothing
    downstream can tell.
    """
    vol = _volume(tmp_path)
    piece = np.ones((4, 4, 4), np.uint64)
    src = _h5_with_offset(tmp_path, piece, (16, 24, 32))
    r = write_subvolume(vol, src, offset_order="xyz", dry_run=True)
    assert r["start"] == (32, 24, 16)
    assert "read as xyz" in r["offset_from"], "a reversal must be visible in the report"


def test_a_field_name_other_than_voxel_offset_can_be_named(tmp_path):
    vol = _volume(tmp_path)
    src = _h5_with_offset(tmp_path, np.ones((4, 4, 4), np.uint64), (4, 4, 4),
                          field="origin")
    with pytest.raises(ValueError, match="records none"):
        write_subvolume(vol, src, dry_run=True)
    r = write_subvolume(vol, src, offset_field="origin", dry_run=True)
    assert r["start"] == (4, 4, 4)


def test_no_offset_anywhere_says_where_it_looked(tmp_path):
    vol = _volume(tmp_path)
    with pytest.raises(ValueError, match="records none"):
        write_subvolume(vol, _h5(tmp_path, np.ones((4, 4, 4), np.uint64)), dry_run=True)


def test_a_source_that_cannot_store_an_offset_fails_the_same_way(tmp_path):
    """Image stacks expose no `stored_offset`; the op must not assume every backend does."""
    vol = _volume(tmp_path, dtype="uint8")
    stack = _png_stack(tmp_path, np.zeros((2, 4, 4), np.uint8))
    with pytest.raises(ValueError, match="records none"):
        write_subvolume(vol, stack, dry_run=True)


def test_a_stored_offset_that_is_not_whole_voxels_is_an_error(tmp_path):
    """Rounding a fractional offset would move the piece with nothing to show for it."""
    vol = _volume(tmp_path)
    src = _h5_with_offset(tmp_path, np.ones((4, 4, 4), np.uint64), (1.5, 2.0, 3.0))
    with pytest.raises(ValueError, match="not whole voxels"):
        write_subvolume(vol, src, dry_run=True)


def test_a_stored_offset_of_the_wrong_length_is_an_error(tmp_path):
    vol = _volume(tmp_path)
    src = _h5_with_offset(tmp_path, np.ones((4, 4, 4), np.uint64), (1, 2))
    with pytest.raises(ValueError, match="has 2 entries"):
        write_subvolume(vol, src, dry_run=True)


def test_an_unknown_offset_order_is_rejected_rather_than_ignored(tmp_path):
    vol = _volume(tmp_path)
    with pytest.raises(ValueError, match="must be 'zyx' or 'xyz'"):
        write_subvolume(vol, _h5(tmp_path, np.ones((4, 4, 4), np.uint64)), (0, 0, 0),
                        offset_order="yxz", dry_run=True)


# --------------------------------------------------------------------------- #
# tiling: the invariant that makes concurrent writes thinkable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("start, stop", [(0, 96), (7, 96), (0, 89), (13, 51)])
def test_tiles_cover_the_region_exactly_once(start, stop):
    got = np.zeros(stop - start, int)
    for (s,) in _tiles((start,), (stop,), (32,)):
        got[s.start - start: s.stop - start] += 1
    assert (got == 1).all()


def test_no_two_tiles_ever_share_a_destination_chunk():
    """Tiles are cut on the GLOBAL chunk grid, not from the region's own start.

    Cutting from the start would put an interior boundary mid-chunk, so two tiles
    would read-modify-write the same chunk — harmless in a serial loop, and a lost
    update the moment anything runs them in parallel.
    """
    chunk, unit = 16, 32
    owners = {}
    for (s,) in _tiles((24,), (200,), (unit,)):
        for c in range(s.start // chunk, -(-s.stop // chunk)):
            owners.setdefault(c, []).append((s.start, s.stop))
    shared = {c: t for c, t in owners.items() if len(t) > 1}
    assert not shared, f"chunks written by more than one tile: {shared}"


def test_a_large_piece_is_split_into_several_writes_and_still_round_trips(tmp_path):
    vol = _volume(tmp_path, shape=(64, 64, 64), dtype="uint8", chunk=(16, 16, 16))
    piece = np.random.default_rng(4).integers(0, 255, (48, 48, 48), dtype=np.uint8)
    r = write_subvolume(vol, _h5(tmp_path, piece), (16, 16, 16), max_bytes=16 * 16 * 16)
    assert r["num_tiles"] > 1
    np.testing.assert_array_equal(
        _level(vol, 0).read_region(_region((16, 16, 16), piece.shape)), piece)


def test_a_chunk_aligned_region_reports_itself_as_aligned(tmp_path):
    vol = _volume(tmp_path, chunk=(16, 16, 16))
    piece = np.ones((16, 32, 16), np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (16, 0, 32), dry_run=True)
    assert r["misaligned_axes"] == []


def test_an_unaligned_region_is_reported_per_axis(tmp_path):
    """Reported, not refused — it is correct, just read-modify-write. But nothing
    downstream can detect the loss if two such writes overlap, so it must be said."""
    vol = _volume(tmp_path, chunk=(16, 16, 16))
    piece = np.ones((16, 8, 16), np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (16, 4, 32), dry_run=True)
    assert r["misaligned_axes"] == [1]


def test_a_region_ending_at_the_volume_edge_counts_as_aligned(tmp_path):
    """There is no neighbouring data in that last chunk to lose."""
    vol = _volume(tmp_path, shape=(40, 40, 40), chunk=(16, 16, 16))
    piece = np.ones((8, 8, 8), np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (32, 32, 32), dry_run=True)
    assert r["misaligned_axes"] == []


# --------------------------------------------------------------------------- #
# a partial chunk is merged, never replaced
#
# The reason this is safe: TensorStore read-modify-writes a chunk that a write only
# partly covers — it fetches what is stored, overlays the new region, writes the whole
# chunk back. Nothing here has to implement that, but everything here depends on it,
# and if it ever stopped being true the loss would be invisible: the piece you just
# wrote would look perfect and its neighbours would be gone.
#
# Parametrized over the storage shapes because the merge happens at different layers —
# a plain chunk, a shard's inner chunk, and a compressed_segmentation block are three
# different code paths inside TensorStore.
# --------------------------------------------------------------------------- #
def _single_level(tmp_path, storage, name):
    """One-level volume in each storage shape, with a spec that opens level 0."""
    dst = str(tmp_path / f"{name}.{storage}")
    kw = dict(shape=(32, 32, 32), dtype="uint64", voxel_size=(8.0, 8.0, 8.0),
              chunk=(8, 8, 8), levels=1)
    if storage == "sharded":
        create_volume(dst, shard=(16, 16, 16), **kw)
    elif storage == "compressed_seg":
        create_volume(dst, format="precomputed", kind="segmentation", **kw)
    elif storage == "precomputed_raw":
        create_volume(dst, format="precomputed", encoding="raw", **kw)
    else:
        create_volume(dst, **kw)
    spec = ({"backend": "neuroglancer_precomputed", "path": dst, "scale_index": 0}
            if "precomputed" in storage or storage == "compressed_seg"
            else {"backend": "zarr3", "path": os.path.join(dst, "0")})
    return dst, spec


STORAGE = ["plain", "sharded", "precomputed_raw", "compressed_seg"]


@pytest.mark.parametrize("storage", STORAGE)
def test_a_partial_chunk_write_keeps_the_data_already_in_that_chunk(tmp_path, storage):
    """Write into the middle of populated chunks; only the covered voxels may change."""
    from neu_vol.backends.base import clear_backend_cache

    vol, spec = _single_level(tmp_path, storage, "keep")
    base = np.random.default_rng(8).integers(1, 200, (32, 32, 32), dtype=np.uint64)
    open_backend(spec).write_region((slice(0, 32),) * 3, base)

    piece = np.full((6, 6, 6), 999, np.uint64)
    write_subvolume(vol, _h5(tmp_path, piece, name=f"p_{storage}"), (5, 5, 5))

    clear_backend_cache()                       # read it back as a fresh reader would
    got = open_backend(spec).read_region((slice(0, 32),) * 3)
    want = base.copy()
    want[_region((5, 5, 5), piece.shape)] = 999
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("storage", STORAGE)
def test_a_second_piece_sharing_a_chunk_does_not_erase_the_first(tmp_path, storage):
    """The real workflow: pieces arrive one at a time, and two of them share a chunk.

    At chunk 8 these span 5-11 and 11-17, so they share the chunk at index 1 on every
    axis. Sequential, which is the supported way — running them at once is the hazard
    `misaligned_axes` exists to warn about.
    """
    from neu_vol.backends.base import clear_backend_cache

    vol, spec = _single_level(tmp_path, storage, "share")
    for at, val in (((5, 5, 5), 111), ((11, 11, 11), 222)):
        piece = np.full((6, 6, 6), val, np.uint64)
        write_subvolume(vol, _h5(tmp_path, piece, name=f"{storage}{val}"), at)

    clear_backend_cache()
    got = open_backend(spec).read_region((slice(0, 32),) * 3)
    assert (got[_region((5, 5, 5), (6, 6, 6))] == 111).all(), "the first piece was eaten"
    assert (got[_region((11, 11, 11), (6, 6, 6))] == 222).all()
    rest = got.copy()
    rest[_region((5, 5, 5), (6, 6, 6))] = 0
    rest[_region((11, 11, 11), (6, 6, 6))] = 0
    assert not rest.any(), "voxels outside both pieces changed"


# --------------------------------------------------------------------------- #
# dtypes
# --------------------------------------------------------------------------- #
def test_a_widening_conversion_happens_without_being_asked(tmp_path):
    vol = _volume(tmp_path, dtype="uint64")
    piece = np.random.default_rng(5).integers(0, 255, (8, 8, 8), dtype=np.uint8)
    write_subvolume(vol, _h5(tmp_path, piece), (0, 0, 0))
    out = _level(vol, 0).read_region(_region((0, 0, 0), piece.shape))
    assert out.dtype == np.uint64
    np.testing.assert_array_equal(out, piece)


def test_a_narrowing_conversion_takes_saying_so(tmp_path):
    vol = _volume(tmp_path, dtype="uint8")
    piece = _h5(tmp_path, np.array([[[300]]], np.uint64))
    with pytest.raises(ValueError, match="without possible loss"):
        write_subvolume(vol, piece, (0, 0, 0))
    write_subvolume(vol, piece, (0, 0, 0), cast=True)
    assert _level(vol, 0).read_region(_region((0, 0, 0), (1, 1, 1)))[0, 0, 0] == 300 % 256


# --------------------------------------------------------------------------- #
# saying no clearly
# --------------------------------------------------------------------------- #
def test_a_piece_that_does_not_fit_is_refused_with_the_numbers(tmp_path):
    vol = _volume(tmp_path, shape=(32, 32, 32))
    piece = _h5(tmp_path, np.ones((8, 8, 8), np.uint64))
    with pytest.raises(ValueError, match="does not fit"):
        write_subvolume(vol, piece, (28, 0, 0))
    with pytest.raises(ValueError, match="does not fit"):
        write_subvolume(vol, piece, (-1, 0, 0))


def test_a_level_that_does_not_exist_lists_the_ones_that_do(tmp_path):
    vol = _volume(tmp_path, levels=2)
    piece = np.ones((4, 4, 4), np.uint64)
    with pytest.raises(ValueError, match=r"no level 5; present: \[0, 1\]"):
        write_subvolume(vol, _h5(tmp_path, piece), (0, 0, 0), level=5)


def test_writing_to_somewhere_that_is_not_a_volume_says_to_create_one(tmp_path):
    piece = np.ones((4, 4, 4), np.uint64)
    with pytest.raises(FileNotFoundError, match="neu-vol create"):
        write_subvolume(str(tmp_path / "nothing"), _h5(tmp_path, piece), (0, 0, 0))


def test_dry_run_checks_everything_and_writes_nothing(tmp_path):
    vol = _volume(tmp_path)
    piece = np.ones((8, 8, 8), np.uint64)
    r = write_subvolume(vol, _h5(tmp_path, piece), (8, 8, 8), dry_run=True)
    assert r["start"] == (8, 8, 8) and r["num_tiles"] >= 1 and r["written"] == 0
    assert not _level(vol, 0).read_region((slice(0, 64),) * 3).any()


# --------------------------------------------------------------------------- #
# working out what the source is
# --------------------------------------------------------------------------- #
def test_hdf5_picks_the_only_volumetric_dataset_by_itself(tmp_path):
    path = _h5(tmp_path, np.zeros((4, 4, 4), np.uint64), dataset="deep/nested/seg")
    assert source_spec(path)["dataset"] == "/deep/nested/seg"


def test_hdf5_with_several_datasets_asks_which_one(tmp_path):
    import h5py

    path = str(tmp_path / "two.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.zeros((4, 4, 4), np.uint8))
        f.create_dataset("b", data=np.zeros((4, 4, 4), np.uint8))
    with pytest.raises(KeyError, match="say which one"):
        source_spec(path)
    assert source_spec(path, dataset="/b")["dataset"] == "/b"


def test_a_directory_of_images_is_an_image_stack(tmp_path):
    stack = _png_stack(tmp_path, np.zeros((2, 4, 4), np.uint8))
    assert source_spec(stack) == {"backend": "image_stack", "source": stack}
    assert source_spec(stack + "/*.png")["backend"] == "image_stack"


def test_a_real_volume_is_detected_as_one_not_guessed_at(tmp_path):
    vol = _volume(tmp_path, name="src")
    assert source_spec(vol)["backend"] == "zarr3"


def test_an_unrecognizable_source_says_to_name_the_format(tmp_path):
    path = str(tmp_path / "mystery.dat")
    open(path, "wb").write(b"\x00")
    with pytest.raises(ValueError, match="src_format"):
        source_spec(path)


def test_a_region_of_another_volume_can_be_the_source(tmp_path):
    """`--src` need not be a file: one volume's level is a perfectly good piece."""
    src = _volume(tmp_path, shape=(16, 16, 16), levels=1, name="donor")
    data = np.random.default_rng(6).integers(1, 99, (16, 16, 16), dtype=np.uint64)
    _level(src, 0).write_region((slice(0, 16),) * 3, data)
    dst = _volume(tmp_path, name="target")
    write_subvolume(dst, os.path.join(src, "0"), (16, 16, 16),
                    src_format="zarr3")
    np.testing.assert_array_equal(
        _level(dst, 0).read_region(_region((16, 16, 16), (16, 16, 16))), data)


def test_a_precomputed_destination_works_the_same_way(tmp_path):
    """`create` only makes zarr, but `write` places pieces into either format —
    precomputed addresses a level by scale_index rather than by subdirectory."""
    from neu_vol import convert
    from neu_vol.profiles import StorageProfile, zarr3_create_spec
    from neu_vol.backends.tensorstore import TensorStoreBackend

    src = str(tmp_path / "pc.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, (32, 32, 32), "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region((slice(0, 32),) * 3, np.zeros((32, 32, 32), np.uint8))
    vol = str(tmp_path / "dst.precomputed")
    convert(src, vol, voxel_size=(8, 8, 8), kind="image", min_dim=8,
            profile=StorageProfile("neuroglancer_precomputed", chunk=(8, 8, 8),
                                   compressor="gzip"),
            chunk=(8, 8, 8), delete_existing=True)

    piece = np.random.default_rng(7).integers(1, 250, (8, 8, 8), dtype=np.uint8)
    write_subvolume(vol, _h5(tmp_path, piece), (8, 8, 8), level=1)
    back = open_backend({"backend": "neuroglancer_precomputed", "path": vol,
                         "scale_index": 1})
    np.testing.assert_array_equal(back.read_region(_region((8, 8, 8), piece.shape)),
                                  piece)


def test_the_plan_reports_the_source_and_destination_it_resolved(tmp_path):
    vol = _volume(tmp_path)
    path = _h5(tmp_path, np.ones((8, 8, 8), np.uint64))
    plan = plan_subvolume_write(vol, path, (8, 8, 8), level=1)
    assert plan["src_spec"] == {"backend": "hdf5", "path": path, "dataset": "/labels"}
    assert plan["dst_spec"]["path"].endswith("/1")
    assert plan["dst_shape"] == (32, 32, 32) and plan["stop"] == (16, 16, 16)
