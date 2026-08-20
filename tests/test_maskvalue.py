"""Background that is not 0: `--background` at ingest, and `mask-by-value` afterwards.

The assertion that matters is not that the values change — that part is easy — it is that
**the storage** does. A tool numbering labels from 0 makes background 1, and an
all-background block of 1s is not all-fill, so it gets *stored*: the volume then has a chunk
object everywhere data was written and "which chunks exist" stops answering "where is the
data". That question is what `bboxes-json`, `relabel`, `downsample --sparse` and
neu-morpho's occupancy all ask, so the tests below check stored keys, not just voxels.
"""

import numpy as np
import pytest

from neu_vol import cli, convert, create_volume, mask_values, pack_hdf5
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.location import list_keys
from neu_vol.profiles import zarr3_create_spec


def _source(path, data, chunk=(8, 8, 8)):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=chunk),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return path


def _read(volume, level=0):
    be = open_backend({"backend": "zarr3", "path": f"{volume}/{level}"})
    return be.read_region(tuple(slice(0, s) for s in be.shape))


def _chunk_keys(volume, level=0):
    return sorted(k for k in list_keys(volume, str(level)) if k.startswith("c"))


@pytest.fixture
def background_one():
    """Labels 2 and 3 in one corner; everything else is background, spelled 1."""
    data = np.ones((16, 16, 16), dtype=np.uint32)
    data[0:4, 0:4, 0:4] = 2
    data[0:4, 4:8, 0:4] = 3
    return data


# --------------------------------------------------------------------------- #
# at ingest, which is where it belongs
# --------------------------------------------------------------------------- #
def test_without_background_every_block_is_stored(tmp_path, background_one):
    """The problem, stated as a test so the fix has something to be measured against."""
    src = _source(str(tmp_path / "s.zarr"), background_one)
    dst = str(tmp_path / "dense.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, profile="local", delete_existing=True)
    assert len(_chunk_keys(dst)) == 8, "all 8 blocks stored, though 7 hold only background"


def test_background_at_ingest_restores_the_sparsity(tmp_path, background_one):
    src = _source(str(tmp_path / "s.zarr"), background_one)
    dst = str(tmp_path / "sparse.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, profile="local", background=[1], delete_existing=True)

    assert len(_chunk_keys(dst)) == 1, "only the block holding real labels is stored"
    out = _read(dst)
    assert set(np.unique(out)) == {0, 2, 3}
    np.testing.assert_array_equal(out[0:4, 0:4, 0:4], 2)
    np.testing.assert_array_equal(out[0:4, 4:8, 0:4], 3)
    assert not out[8:].any(), "background became 0, not something else"


def test_several_background_values_can_be_given(tmp_path):
    data = np.full((8, 8, 8), 1, dtype=np.uint32)
    data[0, 0, 0] = 5
    data[1, 1, 1] = 99                        # a second "not really a label" value
    src = _source(str(tmp_path / "m.zarr"), data)
    dst = str(tmp_path / "m_out.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, profile="local", background=[1, 99], delete_existing=True)
    assert set(np.unique(_read(dst))) == {0, 5}


def test_write_applies_it_to_the_piece_being_placed(tmp_path, background_one):
    """The path the user's data actually takes: a segmented piece into a GT volume."""
    from neu_vol import write_subvolume

    piece = str(tmp_path / "piece.h5")
    pack_hdf5(_source(str(tmp_path / "p.zarr"), background_one), piece,
              voxel_size=(8, 8, 8), voxel_offset=(0, 0, 0))
    vol = str(tmp_path / "gt.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint32",
                  chunk=(8, 8, 8), kind="segmentation", levels=1)

    write_subvolume(vol, piece, background=[1])
    assert set(np.unique(_read(vol))) == {0, 2, 3}
    assert len(_chunk_keys(vol)) == 1, "the all-background blocks were never stored"


def test_to_hdf5_can_correct_it_while_packing(tmp_path, background_one):
    """Then the packed file is already right and nothing downstream needs to know."""
    import h5py

    src = _source(str(tmp_path / "h.zarr"), background_one)
    out = str(tmp_path / "fixed.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), background=[1])
    with h5py.File(out, "r") as f:
        assert set(np.unique(f["/data"][()])) == {0, 2, 3}


def test_the_remap_view_keeps_the_sources_recorded_metadata(tmp_path, background_one):
    """A remap changes values, not geometry, so `stored_offset` must survive the wrapper.

    Swallowing it would silently turn "the offset came from the file" back into "no offset
    given" — and `write` would then demand one.
    """
    from neu_vol import write_subvolume

    piece = str(tmp_path / "keeps.h5")
    pack_hdf5(_source(str(tmp_path / "k.zarr"), background_one), piece,
              voxel_size=(8, 8, 8), voxel_offset=(0, 4, 8))
    vol = str(tmp_path / "k_vol.zarr")
    create_volume(vol, shape=(32, 32, 32), voxel_size=(8, 8, 8), dtype="uint32",
                  chunk=(8, 8, 8), kind="segmentation", levels=1)

    result = write_subvolume(vol, piece, background=[1])   # no offset given
    assert result["offset"] == (0, 4, 8)
    assert "voxel_offset" in result["offset_from"]


# --------------------------------------------------------------------------- #
# afterwards, for data that has already landed
# --------------------------------------------------------------------------- #
@pytest.fixture
def landed(tmp_path, background_one):
    """A volume already written with background 1 — the situation to be repaired."""
    src = _source(str(tmp_path / "l.zarr"), background_one)
    dst = str(tmp_path / "landed.zarr")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, profile="local", delete_existing=True)
    return dst


def test_out_gives_a_corrected_and_sparse_copy(tmp_path, landed):
    out = str(tmp_path / "fixed.zarr")
    result = mask_values(landed, [1], out=out)

    assert set(np.unique(_read(out))) == {0, 2, 3}
    assert result["voxels_replaced"] == 16 ** 3 - 2 * 4 * 4 * 4
    assert len(_chunk_keys(out)) == 1, \
        "the blocks that became all-background were elided, not written"
    assert len(_chunk_keys(landed)) == 8, "the original is untouched"


def test_in_place_restores_the_storage_too(tmp_path, landed):
    """Writing zeros over a stored chunk REMOVES the object, so in place fixes both.

    Worth pinning, because assuming the opposite is easy: `convert --mask-bbox` genuinely
    cannot clear what is already at a key, but only because its block worker returns
    "empty" and never issues the write. Here the zeros are written, and the chunk goes.
    """
    mask_values(landed, [1], in_place=True)
    assert set(np.unique(_read(landed))) == {0, 2, 3}, "values corrected"
    assert len(_chunk_keys(landed)) == 1, "and the all-background chunks are gone"


def test_a_value_that_is_not_present_is_reported(tmp_path, landed, caplog):
    """Almost always a wrong guess at the background value, and otherwise a no-op."""
    with caplog.at_level("WARNING"):
        result = mask_values(landed, [7], out=str(tmp_path / "none.zarr"))
    assert result["voxels_replaced"] == 0
    assert "none of the values" in caplog.text


def test_merging_onto_an_existing_value_is_reported(tmp_path, background_one, caplog):
    """If 0 was a real label, mapping 1 onto it has just made two labels one."""
    data = background_one.copy()
    data[8:12, 8:12, 8:12] = 0                       # 0 already means something here
    src = _source(str(tmp_path / "z.zarr"), data)
    vol = str(tmp_path / "z_vol.zarr")
    convert(src, vol, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            multiscale=False, profile="local", delete_existing=True)

    with caplog.at_level("WARNING"):
        result = mask_values(vol, [1], out=str(tmp_path / "z_out.zarr"))
    assert result["voxels_already_background"] == 4 ** 3
    assert "merged two labels" in caplog.text


def test_replacing_with_a_value_being_replaced_is_refused(landed):
    with pytest.raises(ValueError, match="both a value to replace and the replacement"):
        mask_values(landed, [1], out="/x", to=1)


def test_out_and_in_place_are_mutually_exclusive(landed):
    with pytest.raises(ValueError, match="either out= .* or in_place"):
        mask_values(landed, [1], out="/x", in_place=True)
    with pytest.raises(ValueError, match="either out= .* or in_place"):
        mask_values(landed, [1])


def test_levels_above_are_reported_as_stale(tmp_path, background_one, caplog):
    """Correcting level 0 leaves the pyramid holding background as a label."""
    src = _source(str(tmp_path / "p.zarr"), background_one)
    vol = str(tmp_path / "pyr.zarr")
    convert(src, vol, voxel_size=(8, 8, 8), kind="segmentation", chunk=(8, 8, 8),
            min_dim=8, profile="local", delete_existing=True)

    with caplog.at_level("WARNING"):
        result = mask_values(vol, [1], in_place=True)
    assert result["stale_levels"] == [1]
    assert "still hold the old values" in caplog.text


def test_dry_run_counts_without_writing(tmp_path, landed):
    out = str(tmp_path / "planned.zarr")
    result = mask_values(landed, [1], out=out, dry_run=True)
    assert result["voxels_replaced"] > 0 and result["dry_run"]
    import os

    assert not os.path.exists(out)


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_reports_what_it_replaced(tmp_path, landed, capsys):
    out = str(tmp_path / "cli.zarr")
    assert cli.main(["mask-by-value", landed, "--values", "1", "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "replacing   [1] -> 0" in printed and "replaced " in printed
    assert set(np.unique(_read(out))) == {0, 2, 3}


def test_the_cli_says_in_place_is_not_recoverable(tmp_path, landed, capsys):
    assert cli.main(["mask-by-value", landed, "--values", "1", "--in-place"]) == 0
    assert "not recoverable" in capsys.readouterr().out


def test_the_cli_needs_a_destination(landed):
    with pytest.raises(SystemExit):
        cli.main(["mask-by-value", landed, "--values", "1"])
