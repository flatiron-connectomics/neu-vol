"""Assembling a neuroglancer state and encoding it into a URL.

The failure this guards against is not a crash. A state whose `dimensions` or source
scheme is wrong loads perfectly and shows the data in the wrong place, or shows nothing
and blames the store — so the tests check the *values*, not that a URL came out.
"""

import json
import urllib.parse

import numpy as np
import pytest

from em_volume_tools import cli, convert
from em_volume_tools.backends.tensorstore import TensorStoreBackend
from em_volume_tools.ops.ngurl import (DEFAULT_VIEWER, VolumeProblem, build_state,
                                       load_layer, parse_url, state_url, volume_layer)
from em_volume_tools.profiles import zarr3_create_spec


def _volume(tmp_path, name, *, kind="image", profile="local-neuroglancer",
            voxel=(8, 8, 8), dtype="uint8"):
    data = np.ones((16, 16, 16), dtype)
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, dtype,
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=voxel, kind=kind, profile=profile, chunk=(8, 8, 8),
            min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def test_the_source_scheme_comes_from_the_format(tmp_path):
    """`precomputed://` vs `zarr://` — get it wrong and the layer simply fails to load."""
    pre = _volume(tmp_path, "pre", profile="local-neuroglancer")
    zar = _volume(tmp_path, "zar", profile="local")
    assert volume_layer(pre)[0]["source"] == f"precomputed://{pre}"
    assert volume_layer(zar)[0]["source"] == f"zarr://{zar}"


def test_the_recorded_kind_decides_image_versus_segmentation(tmp_path):
    """A segmentation drawn as an image is a grey mush, and easy to miss on a small ROI."""
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    img = _volume(tmp_path, "img", kind="image")
    assert volume_layer(seg)[0]["type"] == "segmentation"
    assert volume_layer(img)[0]["type"] == "image"
    # ...and an explicit kind overrides it
    assert volume_layer(seg, kind="image")[0]["type"] == "image"


def test_segment_ids_are_strings(tmp_path):
    """JSON numbers are doubles, so a real uint64 body id above 2**53 would arrive
    rounded — and silently select a different segment."""
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    big = 2 ** 60 + 1
    layer, _ = volume_layer(seg, segments=[1, big])
    assert layer["segments"] == ["1", str(big)]
    assert json.loads(json.dumps(layer))["segments"][1] == str(big)


def test_a_missing_volume_is_named(tmp_path):
    with pytest.raises(VolumeProblem, match="no volume found"):
        volume_layer(str(tmp_path / "nope"))


def test_the_frame_is_reported_for_the_state_to_use(tmp_path):
    vol = _volume(tmp_path, "v", voxel=(40, 8, 8))
    _layer, frame = volume_layer(vol)
    assert tuple(frame["voxel_size"]) == (40, 8, 8)      # zyx
    assert frame["units"] == "nm"


# --------------------------------------------------------------------------- #
# loading layer files
# --------------------------------------------------------------------------- #
def test_a_layer_file_is_accepted_as_a_layer_or_a_whole_state(tmp_path):
    """`bboxes-json` can emit either shape; which one the caller happened to make must
    not matter."""
    layer = {"type": "annotation", "name": "a", "annotations": []}
    bare, wrapped = tmp_path / "bare.json", tmp_path / "state.json"
    bare.write_text(json.dumps(layer))
    wrapped.write_text(json.dumps({"dimensions": {}, "layers": [layer]}))
    assert load_layer(str(bare)) == [layer]
    assert load_layer(str(wrapped)) == [layer]


def test_a_file_that_is_neither_says_which_it_expected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}')
    with pytest.raises(VolumeProblem, match="not a neuroglancer layer or state"):
        load_layer(str(bad))


# --------------------------------------------------------------------------- #
# the state and the URL
# --------------------------------------------------------------------------- #
def test_position_is_reversed_to_xyz():
    """zyx in memory, xyz on the wire. Reversed, the link opens somewhere else entirely."""
    state, _ = build_state([{"type": "image", "name": "i"}],
                           voxel_size_zyx=(8, 8, 8), units="nm",
                           position_zyx=(100, 200, 300))
    assert state["position"] == [300.0, 200.0, 100.0]


def test_dimensions_come_from_the_voxel_size():
    state, warning = build_state([], voxel_size_zyx=(40, 8, 8), units="nm")
    assert warning is None
    assert state["dimensions"] == {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [40e-9, "m"]}


def test_the_url_round_trips():
    state, _ = build_state([{"type": "image", "name": "i"}], voxel_size_zyx=(8, 8, 8),
                           units="nm", position_zyx=(1, 2, 3), layout="xy-3d")
    url = state_url(state)
    assert url.startswith(DEFAULT_VIEWER.rstrip("/") + "/#!")
    assert parse_url(url) == state


def test_the_fragment_is_percent_encoded():
    """Raw JSON in a fragment survives only until something in the chain — a chat client,
    a wiki, a shell — decides what to do with the braces and quotes."""
    state, _ = build_state([{"type": "image", "name": "i"}], voxel_size_zyx=(8, 8, 8),
                           units="nm")
    fragment = state_url(state).split("#!", 1)[1]
    assert "{" not in fragment and '"' not in fragment
    assert json.loads(urllib.parse.unquote(fragment))["layers"][0]["name"] == "i"


def test_parse_url_rejects_a_non_state_url():
    with pytest.raises(ValueError, match="no '#!' fragment"):
        parse_url("https://neuroglancer-demo.appspot.com/")


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_layers_are_ordered_image_then_segmentation_then_files(tmp_path, capsys):
    """The segmentation has to draw over the image, and the boxes over both."""
    img = _volume(tmp_path, "img")
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    assert cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--seg", seg, "--image", img, "--layer", str(box)])) == 0
    state = parse_url(capsys.readouterr().out.strip())
    assert [lyr["name"] for lyr in state["layers"]] == ["img", "seg", "boxes"]


def test_segments_apply_to_the_seg_volumes_in_order(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--seg", a, "--segments", "1,2", "--seg", b,
         "--segments", "7"]))
    state = parse_url(capsys.readouterr().out.strip())
    picked = {lyr["name"]: lyr.get("segments") for lyr in state["layers"]}
    assert picked == {"a": ["1", "2"], "b": ["7"]}


def test_position_order_xyz_lets_you_paste_from_the_viewer(tmp_path, capsys):
    """Neuroglancer displays xyz, so the numbers a user copies are xyz."""
    vol = _volume(tmp_path, "v")
    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--image", vol, "--position", "1,2,3",
         "--position-order", "xyz"]))
    assert parse_url(capsys.readouterr().out.strip())["position"] == [1.0, 2.0, 3.0]

    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--image", vol, "--position", "1,2,3"]))      # zyx default
    assert parse_url(capsys.readouterr().out.strip())["position"] == [3.0, 2.0, 1.0]


def test_layer_files_alone_need_an_explicit_voxel_size(tmp_path, capsys):
    """A layer carries its own frame but does not establish the viewer's, and a
    disagreeing `dimensions` block loads fine while misplacing everything."""
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    with pytest.raises(SystemExit, match="no voxel size available"):
        cli.cmd_ng_url_gen(cli._parse_args(["ng-url-gen", "--layer", str(box)]))

    assert cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--layer", str(box), "--voxel-size", "8,8,8"])) == 0
    assert parse_url(capsys.readouterr().out.strip())["dimensions"]["x"] == [8e-9, "m"]


def test_nothing_to_show_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="nothing to show"):
        cli.cmd_ng_url_gen(cli._parse_args(["ng-url-gen"]))


def test_select_last_opens_the_boxes_panel(tmp_path, capsys):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--seg", vol, "--layer", str(box), "--select-last"]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["selectedLayer"] == {"visible": True, "layer": "boxes"}


def test_out_and_state_out_write_through_the_kvstore(tmp_path, capsys):
    """Parents that do not exist are created, which is what makes s3:// work too."""
    vol = _volume(tmp_path, "v")
    url_path = str(tmp_path / "deep" / "link.txt")
    state_path = str(tmp_path / "deep" / "state.json")
    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--image", vol, "--out", url_path,
         "--state-out", state_path]))
    assert capsys.readouterr().out == "", "--out must keep stdout clean"
    with open(url_path) as f:
        url = f.read().strip()
    with open(state_path) as f:
        assert json.load(f) == parse_url(url)


def test_the_viewer_base_is_overridable(tmp_path, capsys):
    vol = _volume(tmp_path, "v")
    cli.cmd_ng_url_gen(cli._parse_args(
        ["ng-url-gen", "--image", vol, "--viewer", "https://ng.example.org/"]))
    assert capsys.readouterr().out.startswith("https://ng.example.org/#!")


def test_ng_url_gen_is_wired():
    assert cli._parse_args(["ng-url-gen", "--image", "v"]).func is cli.cmd_ng_url_gen
