"""Which chunks a sparse volume stores, and the boxes that cover them.

The box cover carries the weight: a wrong box is a claim that there is data somewhere there
isn't, or — worse — hides a region by merging it into a neighbour.

Turning these boxes into a viewer layer is `em-ngl bboxes` and is tested there.
"""

import numpy as np
import pytest

from em_volume_tools import convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.ops.annotate import (NoOccupancy, _precomputed_cell, _zarr_cell,
                                          labeled_regions, maximal_boxes)
from em_volume_tools.profiles import zarr3_create_spec

# The real geometry from sample3's gt_v1: two 3x3x3-chunk blocks whose footprints meet
# at one z boundary but are offset in y, so their union is not a box. Connected
# components reported these as ONE region spanning both plus the empty corner between.
BLOCK_A = {(x, y, z) for x in range(1, 4) for y in range(0, 3) for z in range(0, 3)}
BLOCK_B = {(x, y, z) for x in range(0, 3) for y in range(1, 4) for z in range(3, 6)}


# --------------------------------------------------------------------------- #
# the box cover
# --------------------------------------------------------------------------- #
def test_two_blocks_meeting_at_a_face_stay_two_boxes():
    boxes = maximal_boxes(BLOCK_A | BLOCK_B)
    assert len(boxes) == 2, "the gt07/gt08 merge is back"
    assert ((0, 1, 3), (3, 4, 6)) in boxes
    assert ((1, 0, 0), (4, 3, 3)) in boxes


def test_a_solid_block_is_one_box():
    cells = {(x, y, z) for x in range(2) for y in range(3) for z in range(4)}
    assert maximal_boxes(cells) == [((0, 0, 0), (2, 3, 4))]


def test_genuinely_contiguous_regions_are_one_box():
    """Two blocks written face to face *in line* are one region, by design.

    Nothing in the stored chunks distinguishes that from a single write of twice the
    size, and a box over both is a true statement about where the data is.
    """
    left = {(0, y, z) for y in range(2) for z in range(2)}
    right = {(1, y, z) for y in range(2) for z in range(2)}
    assert maximal_boxes(left | right) == [((0, 0, 0), (2, 2, 2))]


@pytest.mark.parametrize("seed", range(6))
def test_a_box_never_claims_an_absent_cell(seed):
    """The invariant that makes the output trustworthy, over scattered occupancy."""
    rng = np.random.default_rng(seed)
    cells = {tuple(int(v) for v in c)
             for c in rng.integers(0, 5, size=(30, 3))}
    boxes = maximal_boxes(cells)
    covered = set()
    for lo, hi in boxes:
        box = {(x, y, z) for x in range(lo[0], hi[0])
               for y in range(lo[1], hi[1]) for z in range(lo[2], hi[2])}
        assert box <= cells, "a box covers a cell with nothing stored in it"
        covered |= box
    assert covered == cells, "some occupied cell was left out of every box"


# --------------------------------------------------------------------------- #
# chunk keys
# --------------------------------------------------------------------------- #
def test_precomputed_key_is_xyz_and_becomes_a_zyx_cell():
    cell = _precomputed_cell("128-256_0-128_384-512", (128, 128, 128))
    assert cell == (3, 0, 1), "key is xyz, cells are zyx"


def test_cloudvolume_gz_suffixed_keys_still_parse():
    """A gzipped-chunk volume is exactly the kind most likely to be sparse."""
    assert _precomputed_cell("0-128_0-128_0-128.gz", (128,) * 3) == (0, 0, 0)


@pytest.mark.parametrize("key", ["c/1/2/3", "c.1.2.3"])
def test_zarr_keys_parse_with_either_separator(key):
    assert _zarr_cell(key, 3) == (1, 2, 3)


def test_zarr_channel_axis_is_dropped():
    assert _zarr_cell("c/0/1/2/3", 3) == (1, 2, 3)


def test_non_chunk_keys_are_ignored():
    assert _zarr_cell("zarr.json", 3) is None
    assert _precomputed_cell("info", (128,) * 3) is None


def test_a_sharded_level_says_so_rather_than_reporting_nothing(monkeypatch, tmp_path):
    """Shards hide which of their chunks exist, so occupancy is unanswerable.

    Silently returning zero regions would read as "this volume is empty", which is the
    one wrong answer worth guarding against.
    """
    from em_volume_tools.ops import annotate

    monkeypatch.setattr(annotate, "list_keys", lambda *a, **k: ["0.shard", "1.shard"])
    with pytest.raises(NoOccupancy, match="SHARDED"):
        annotate.occupied_cells(str(tmp_path), "zarr3", 0, (8, 8, 8))


# --------------------------------------------------------------------------- #
# against real volumes
# --------------------------------------------------------------------------- #
def _sparse(tmp_path, name, *, profile, chunk=(8, 8, 8)):
    """A real two-level volume, 32^3, holding two separated 8^3 labeled blocks.

    Built through `convert` so the levels, `info`/`zarr.json` and the elision of
    all-fill chunks are the production ones — occupancy here means what it means in a
    real run.
    """
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[0:8, 0:8, 0:8] = 3                       # cell (0,0,0)
    seg[16:24, 24:32, 8:16] = 4                  # cell (2,3,1)
    seg[16:24, 24:32, 8:16][0, 0, 0] = 5         # a second label, to count
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, seg.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in seg.shape), seg)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation", profile=profile,
            chunk=chunk, factors=[(2, 2, 2)], min_dim=8, delete_existing=True)
    return dst


@pytest.mark.parametrize("profile", ["local", "local-neuroglancer"])
def test_finds_the_written_regions_in_both_formats(tmp_path, profile):
    """The formats key their chunks completely differently; the answer must not."""
    dst = _sparse(tmp_path, "vol", profile=profile)
    regions, ctx = labeled_regions(dst, tighten_level=1)

    assert len(regions) == 2, [r["lo"] for r in regions]
    assert ctx["n_chunks"] == 2, "only the two non-fill chunks should exist"
    lows = sorted(tuple(r["lo"]) for r in regions)
    assert lows == [(0, 0, 0), (16, 24, 8)]
    by_lo = {tuple(r["lo"]): r for r in regions}
    assert tuple(by_lo[(0, 0, 0)]["hi"]) == (8, 8, 8)
    assert tuple(by_lo[(16, 24, 8)]["hi"]) == (24, 32, 16)
    # tightening reads level 1, where both labels of the second block survive
    assert by_lo[(16, 24, 8)]["n_labels"] >= 1


def test_no_tighten_skips_the_reads_and_reports_no_label_counts(tmp_path):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    regions, _ = labeled_regions(dst, tighten_level=None)
    assert len(regions) == 2
    assert all(r["n_labels"] is None for r in regions)
    # boxes stay on the chunk grid, which here is what the data happens to fill
    assert sorted(tuple(r["lo"]) for r in regions) == [(0, 0, 0), (16, 24, 8)]


def test_tightening_shrinks_a_box_to_its_data(tmp_path):
    """A chunk holding one labeled voxel should not annotate the whole chunk."""
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[8:24, 8:24, 8:24] = 9          # spans chunks, but leaves margins inside them
    src = str(tmp_path / "s.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, seg.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in seg.shape), seg)
    dst = str(tmp_path / "v")
    convert(src, dst, voxel_size=(8, 8, 8), kind="segmentation",
            profile="local-neuroglancer", chunk=(16, 16, 16), factors=[(2, 2, 2)],
            min_dim=8, delete_existing=True)

    loose, _ = labeled_regions(dst, tighten_level=None)
    tight, _ = labeled_regions(dst, tighten_level=1)
    assert tuple(loose[0]["lo"]) == (0, 0, 0) and tuple(loose[0]["hi"]) == (32, 32, 32)
    assert tuple(tight[0]["lo"]) == (8, 8, 8) and tuple(tight[0]["hi"]) == (24, 24, 24)


def test_an_absent_occupancy_level_is_an_error(tmp_path):
    """Occupancy at another level answers a different question, so it is not guessed."""
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    with pytest.raises(ValueError, match="no level 5"):
        labeled_regions(dst, level=5, tighten_level=None)


def test_tightening_falls_back_to_the_deepest_level_there_is(tmp_path):
    """A single-level volume is the normal state of one `create` made and `write` filled.

    Erroring because the default tighten level is absent would refuse to annotate
    exactly those volumes. Clamping goes finer, so the boxes get more exact, not less —
    but it has to be reported, or a slow run looks unexplained.
    """
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    regions, ctx = labeled_regions(dst, tighten_level=7)
    assert ctx["tighten_level"] == max(ctx["levels"])
    assert ctx["tighten_clamped_from"] == 7
    assert len(regions) == 2
    # clamped to a real level, so the bounds are still the tight ones
    assert sorted(tuple(r["lo"]) for r in regions) == [(0, 0, 0), (16, 24, 8)]


