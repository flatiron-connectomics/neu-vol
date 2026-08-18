"""Linking a precomputed segmentation volume to its mesh / skeleton / property sidecars.

Moved here from em-seg-morpho, which still re-exports it: it edits a *volume's* ``info``,
and more than one consumer needs it. em-seg-morpho's own tests exercise it against a real
generated volume; these cover the parts that are pure ``info`` bookkeeping, including the
``segment_properties`` ``@type`` check that had no coverage before.
"""

import pytest

from em_volume_tools.location import read_json, write_json
from em_volume_tools.ops.subresources import SUBRESOURCE_TYPES, link_subresources


def _volume(tmp_path, **subs):
    """A segmentation `info` plus a subresource `info` per named @type."""
    out = str(tmp_path / "vol")
    write_json(out, {"type": "segmentation", "data_type": "uint64",
                     "num_channels": 1, "scales": []}, "info")
    for sub, at_type in subs.items():
        write_json(out, {"@type": at_type}, sub, "info")
    return out


def test_segment_properties_is_linked_like_mesh_and_skeletons(tmp_path):
    out = _volume(tmp_path, segment_properties="neuroglancer_segment_properties")
    assert link_subresources(out, segment_properties="segment_properties") == {
        "segment_properties": "segment_properties"}
    assert read_json(out, "info")["segment_properties"] == "segment_properties"


def test_a_wrong_segment_properties_type_is_caught_not_written(tmp_path):
    """The gap this move closed: the key was accepted but its @type was never checked,
    so pointing a volume at, say, a mesh directory produced a layer that silently shows
    no properties."""
    out = _volume(tmp_path, sp="neuroglancer_multilod_draco")
    with pytest.raises(ValueError,
                       match="expected 'neuroglancer_segment_properties'"):
        link_subresources(out, segment_properties="sp")
    # and the info is left alone
    assert "segment_properties" not in read_json(out, "info")


def test_every_linkable_key_has_a_declared_type():
    """A key with no entry in the map is accepted without validation — which is exactly
    how segment_properties went unchecked."""
    import inspect

    params = inspect.signature(link_subresources).parameters
    keys = [n for n in params if n != "volume_dir"]
    assert set(keys) == set(SUBRESOURCE_TYPES)


def test_all_three_keys_can_be_set_at_once(tmp_path):
    out = _volume(tmp_path, mesh="neuroglancer_multilod_draco",
                  skeleton="neuroglancer_skeletons",
                  props="neuroglancer_segment_properties")
    assert link_subresources(out, mesh="mesh", skeletons="skeleton",
                             segment_properties="props") == {
        "mesh": "mesh", "skeletons": "skeleton", "segment_properties": "props"}
    info = read_json(out, "info")
    assert (info["mesh"], info["skeletons"], info["segment_properties"]) == (
        "mesh", "skeleton", "props")


def test_a_missing_subresource_info_is_refused(tmp_path):
    out = _volume(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        link_subresources(out, segment_properties="nope")


def test_an_image_volume_is_refused(tmp_path):
    out = _volume(tmp_path, sp="neuroglancer_segment_properties")
    info = read_json(out, "info")
    info["type"] = "image"
    write_json(out, info, "info")
    with pytest.raises(ValueError, match="not a segmentation volume"):
        link_subresources(out, segment_properties="sp")
