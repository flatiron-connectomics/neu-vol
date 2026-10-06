"""DVID ROIs as a positioned label volume, without a DVID server.

The fetch is stubbed; the UNPACK is neuclease's real one, fed real ``ranges`` rows, because
the thing most likely to be silently wrong is the box convention — which block the array's
voxel ``(0, 0, 0)`` is — and a stubbed unpack would only restate the assumption.
"""

import numpy as np
import pytest

from neu_lib import mask_predicate
from neu_vol import dvid

URL = "dvid://dvid.example.org/93fdbc/synapses"
SPEC = dvid.parse_url(URL)

#: ``ranges`` rows are ``[z, y, x0, x1]`` in 32-voxel blocks, ``x1`` inclusive. A and B
#: share block (2, 3, 5), so B — named later — must hold it.
RANGES = {
    "A": np.array([[2, 3, 4, 5]]),
    "B": np.array([[2, 3, 5, 6], [3, 3, 6, 6]]),
    "EMPTY": np.zeros((0, 4), dtype=int),
}


@pytest.fixture()
def fake_rois(monkeypatch):
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd
    import neuclease.dvid.roi as ndroi

    monkeypatch.setattr(nd, "fetch_repo_info", lambda server, uuid, **k: {
        "DataInstances": {
            "synapses": {"Base": {"TypeName": "annotation"}},
            **{name: {"Base": {"TypeName": "roi"}} for name in ("A", "B", "EMPTY", "ME(L)")},
        }})

    def fetch_roi_ranges_and_boxes(server, uuid, rois, **k):
        ranges = {n: RANGES[n] for n in rois}
        boxes = {n: np.array([r[:, (0, 1, 2)].min(axis=0), 1 + r[:, (0, 1, 3)].max(axis=0)])
                 for n, r in ranges.items() if len(r)}
        return ranges, boxes

    monkeypatch.setattr(ndroi, "fetch_roi_ranges_and_boxes", fetch_roi_ranges_and_boxes)


# --------------------------------------------------------------------------- #
# validating the ROI set before anything expensive
# --------------------------------------------------------------------------- #
def test_available_rois_lists_only_the_roi_instances(fake_rois):
    assert dvid.available_rois(SPEC) == ["A", "B", "EMPTY", "ME(L)"]


def test_a_typo_is_caught_up_front_with_close_matches(fake_rois):
    with pytest.raises(ValueError, match=r"named ME\(Q\).*ME\(L\)"):
        dvid.resolve_roi_set(SPEC, ["A", "ME(Q)"])


def test_an_empty_roi_set_explains_why_there_is_no_default(fake_rois):
    with pytest.raises(ValueError, match="deliberately no default"):
        dvid.resolve_roi_set(SPEC, [])


def test_a_repeated_roi_is_refused(fake_rois):
    with pytest.raises(ValueError, match="repeats A"):
        dvid.resolve_roi_set(SPEC, ["A", "A"])


def test_the_order_given_is_preserved(fake_rois):
    """It decides which ROI wins in an overlap, so it must not be sorted."""
    assert dvid.resolve_roi_set(SPEC, ["B", "A"]) == ["B", "A"]


# --------------------------------------------------------------------------- #
# the positioned volume
# --------------------------------------------------------------------------- #
def test_roi_frame_is_one_block_per_voxel_starting_at_the_box():
    frame = dvid.roi_frame((2, 3, 4), (8.0, 8.0, 8.0), origin_nm=(0.0, 0.0, 100.0))
    assert frame.voxel_size_nm == (256.0, 256.0, 256.0)
    assert frame.origin_nm == (512.0, 768.0, 1124.0)


def test_roi_frame_keeps_anisotropy():
    frame = dvid.roi_frame((1, 1, 1), (40.0, 8.0, 8.0))
    assert frame.voxel_size_nm == (1280.0, 256.0, 256.0)
    assert frame.origin_nm == (1280.0, 256.0, 256.0)


def test_roi_volume_places_blocks_where_dvid_says_they_are(fake_rois):
    vol = dvid.roi_volume(SPEC, ["A", "B"], voxel_size_nm=(8, 8, 8))
    assert vol.names == ("A", "B")
    assert vol.labels.shape == (2, 1, 3)                 # z 2..3, y 3, x 4..6
    inside_a = mask_predicate(vol.mask("A"), vol.frame)
    inside_b = mask_predicate(vol.mask("B"), vol.frame)

    def centre(block_zyx):
        return (np.asarray(block_zyx, float) + 0.5) * 256.0

    pts = np.array([centre((2, 3, 4)), centre((2, 3, 5)), centre((2, 3, 6)),
                    centre((3, 3, 6)), centre((2, 3, 3)), centre((2, 4, 4))])
    assert inside_a(pts).tolist() == [True, False, False, False, False, False]
    assert inside_b(pts).tolist() == [False, True, True, True, False, False]


def test_the_later_roi_wins_an_overlap_and_the_overlap_is_recorded(fake_rois):
    vol = dvid.roi_volume(SPEC, ["A", "B"], voxel_size_nm=(8, 8, 8))
    assert vol.overlaps == (("A", "B", 1),)
    assert vol.mask().sum() == 4                         # the union is unaffected


def test_mask_rejects_a_name_it_does_not_hold(fake_rois):
    vol = dvid.roi_volume(SPEC, ["A"], voxel_size_nm=(8, 8, 8))
    with pytest.raises(KeyError, match="holds"):
        vol.mask("B")


def test_an_empty_roi_raises_rather_than_shrinking_the_region(fake_rois):
    with pytest.raises(ValueError, match="EMPTY.*hold no blocks"):
        dvid.roi_volume(SPEC, ["A", "EMPTY"], voxel_size_nm=(8, 8, 8))
