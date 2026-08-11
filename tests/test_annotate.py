"""Deriving a neuroglancer annotation layer from which chunks a sparse volume stores.

Two things carry the weight here. The box cover, because a wrong box is a claim that
there is data somewhere there isn't (or, worse, hides a region by merging it into a
neighbour); and the zyx/xyz flip, because getting it wrong mirrors every annotation
through the z=x diagonal and still produces a layer that loads.
"""

import json
import os

import numpy as np
import pytest

from em_volume_tools import cli, convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.ops.annotate import (NoOccupancy, _precomputed_cell, _zarr_cell,
                                          annotation_layer, labeled_regions,
                                          maximal_boxes, output_dimensions, render)
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


# --------------------------------------------------------------------------- #
# the JSON
# --------------------------------------------------------------------------- #
def test_layer_coordinates_are_flipped_to_xyz():
    """zyx in memory, xyz on the wire. Reversed, every box is mirrored."""
    regions = [{"lo": (1, 2, 3), "hi": (4, 5, 6), "n_labels": 7}]
    layer = annotation_layer(regions, {d: [8e-9, "m"] for d in "xyz"})
    ann = layer["annotations"][0]
    assert ann["pointA"] == [3, 2, 1]
    assert ann["pointB"] == [6, 5, 4]


def test_points_sit_at_the_centre():
    regions = [{"lo": (0, 0, 0), "hi": (10, 20, 30), "n_labels": None}]
    ann = annotation_layer(regions, {}, kind="point")["annotations"][0]
    assert ann["type"] == "point"
    assert ann["point"] == [15.0, 10.0, 5.0]         # xyz


def test_layer_declares_its_own_dimensions():
    """So the layer can be pasted into any state of the volume, whatever the viewer is
    displaying in — the coordinates are read in the layer's frame, not the global one."""
    layer = annotation_layer([], {"x": [8e-9, "m"]})
    assert layer["source"]["url"] == "local://annotations"
    assert layer["source"]["transform"]["outputDimensions"] == {"x": [8e-9, "m"]}
    assert layer["tab"] == "annotations"


def test_voxel_size_becomes_metres():
    dims, warning = output_dimensions((8.0, 4.0, 4.0), "nm")     # zyx
    assert warning is None
    assert dims == {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [8e-9, "m"]}


@pytest.mark.parametrize("units", [None, "furlong"])
def test_unusable_units_warn_instead_of_inventing_a_scale(units):
    """A wrong scale would place the boxes somewhere plausible and wrong; unitless at
    least fails visibly, and the warning names the flag that fixes it."""
    voxel = None if units is None else (8.0, 8.0, 8.0)
    dims, warning = output_dimensions(voxel, units)
    assert dims == {d: [1, ""] for d in "xyz"}
    assert "--voxel-size" in warning


def test_render_stays_valid_json_with_one_line_per_annotation():
    """Pins a bug that produced a loadable-looking layer with no annotations in it.

    The renderer swaps each annotation for a placeholder, dumps, then substitutes the
    one-line form back. A placeholder built from control characters is re-escaped by
    json.dumps, so every substitution missed and the annotations shipped as the
    placeholder *strings* — valid JSON, twelve entries, none of them an annotation.
    """
    regions = [{"lo": (0, 0, 0), "hi": (8, 8, 8), "n_labels": 2},
               {"lo": (8, 8, 8), "hi": (16, 16, 16), "n_labels": 3}]
    layer = annotation_layer(regions, {d: [8e-9, "m"] for d in "xyz"})
    text = render(layer)

    assert "\\u0000" not in text and "em_vol_annotation" not in text
    back = json.loads(text)
    assert [a["type"] for a in back["annotations"]] == \
        ["axis_aligned_bounding_box"] * 2
    assert back["annotations"][0]["pointB"] == [8, 8, 8]
    for ann in back["annotations"]:
        assert sum(1 for line in text.splitlines()
                   if line.strip().startswith(f'{{"type": "axis_aligned'
                                              f'_bounding_box", "id": "{ann["id"]}"')) == 1


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_stdout_is_only_json_so_it_can_be_redirected(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    assert cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1"])) == 0
    captured = capsys.readouterr()
    layer = json.loads(captured.out)          # the whole of stdout, or this raises
    assert layer["type"] == "annotation"
    assert len(layer["annotations"]) == 2
    assert "region(s)" in captured.err, "the summary belongs on stderr"


def test_out_writes_the_file_and_leaves_stdout_empty(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    out = str(tmp_path / "layer.json")
    cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1", "--out", out]))
    assert capsys.readouterr().out == ""
    with open(out) as f:
        assert len(json.load(f)["annotations"]) == 2


def test_state_is_loadable_and_carries_the_volume_layer(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1", "--state"]))
    state = json.loads(capsys.readouterr().out)
    assert [lyr["type"] for lyr in state["layers"]] == ["segmentation", "annotation"]
    assert state["layers"][0]["source"] == f"precomputed://{dst}"
    assert state["selectedLayer"]["layer"] == state["layers"][1]["name"]
    # the view opens on the first region rather than at the origin of an empty frame
    assert state["position"] != [0, 0, 0]


def test_state_names_the_zarr_driver_for_a_zarr_volume(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local")
    cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1", "--state"]))
    state = json.loads(capsys.readouterr().out)
    assert state["layers"][0]["source"] == f"zarr://{dst}"


def test_label_and_name_flow_through(tmp_path, capsys):
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1", "--label", "gt",
         "--name", "gt-chunks", "--color", "#ff0000"]))
    layer = json.loads(capsys.readouterr().out)
    assert layer["name"] == "gt-chunks"
    assert layer["annotationColor"] == "#ff0000"
    assert [a["id"] for a in layer["annotations"]] == ["gt00", "gt01"]


def test_an_empty_volume_reports_nothing_to_annotate(tmp_path, capsys):
    """Distinguishable from a failure: exit 1, and it says why."""
    from em_volume_tools.ops.create import create_volume

    dst = str(tmp_path / "empty")
    create_volume(dst, format="precomputed", shape=(32, 32, 32), dtype="uint64",
                  voxel_size=(8, 8, 8), chunk=(8, 8, 8), kind="segmentation")
    assert cli.cmd_bboxes_json(cli._parse_args(["bboxes-json", dst])) == 1
    assert "nothing is stored" in capsys.readouterr().err


def test_a_missing_volume_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="no volume found"):
        cli.cmd_bboxes_json(cli._parse_args(
            ["bboxes-json", str(tmp_path / "nope")]))


def test_tighten_defaults_to_the_footprint_level(tmp_path, capsys):
    """Exact boxes by default, and the read cost scales with the level already chosen.

    Defaulting to a fixed coarse level made the common invocation quantize every bound
    to that level's voxel — the reason extents came out 252 instead of 256 on real data
    and had to be explained. Defaulting to --level means exact at the default --level 0,
    and no worse than the listing already costs when a coarser level is asked for.
    """
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[8:24, 8:24, 8:24] = 9          # inside chunk boundaries on every side
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

    args = cli._parse_args(["bboxes-json", dst])
    assert args.tighten_level is None, "the default must resolve against --level"
    assert cli.cmd_bboxes_json(args) == 0
    ann = json.loads(capsys.readouterr().out)["annotations"][0]
    assert ann["pointA"] == [8, 8, 8] and ann["pointB"] == [24, 24, 24]


def test_out_goes_through_the_kvstore_not_open(tmp_path, capsys):
    """So `--out s3://...` works at all.

    Pinned with a local path whose parent does not exist: the file driver creates it,
    while `open()` would raise. That is the same code path a remote --out takes, and it
    is the only part of it a test can exercise without a bucket.
    """
    dst = _sparse(tmp_path, "vol", profile="local-neuroglancer")
    out = str(tmp_path / "does" / "not" / "exist" / "layer.json")
    assert cli.cmd_bboxes_json(cli._parse_args(
        ["bboxes-json", dst, "--tighten-level", "1", "--out", out])) == 0
    with open(out) as f:
        assert len(json.load(f)["annotations"]) == 2
    assert capsys.readouterr().out == "", "--out must keep stdout clean"


def test_bboxes_json_is_wired_to_the_subcommand():
    assert cli._parse_args(["bboxes-json", "v"]).func is cli.cmd_bboxes_json
    assert not os.path.exists("v"), "parsing must not touch the volume"
