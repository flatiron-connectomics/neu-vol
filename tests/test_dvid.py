"""The DVID source path, exercised without a DVID server.

Everything here is address resolution, detection and metadata shape — the plumbing that
had to be carved out because DVID is *not* a kvstore, and the part most likely to break
silently. The parts that genuinely need a server (reading voxels) are verified by hand
against a real instance; see NOTES-TODO for the measurements and the end-to-end check.
"""

import numpy as np
import pytest

from em_volume_tools import dvid as canonical_dvid
from em_volume_tools import source_metadata as sm
from em_volume_tools.backends import dvid

URL = "dvid://dvid.example.org/93fdbc:main/labels"

#: A believable `/info` for a labelmap, matching the real instance's shape.
INFO = {
    "Base": {"TypeName": "labelmap"},
    "Extended": {
        "BlockSize": [64, 64, 64],
        "VoxelSize": [8, 8, 8],
        "VoxelUnits": ["nanometers"] * 3,
        "MaxDownresLevel": 3,
        "MinPoint": [1024, 1536, 0],
        "MaxPoint": [12799, 9215, 8191],
    },
}


# --------------------------------------------------------------------------- #
# URL form
# --------------------------------------------------------------------------- #
def test_url_splits_into_server_uuid_instance():
    assert dvid.parse_url(URL) == {
        "server": "dvid.example.org",
        "uuid": "93fdbc:main",
        "instance": "labels",
    }


def test_a_port_and_a_branch_do_not_add_segments():
    """Both carry a ':', which is why the URL splits on '/' alone."""
    assert dvid.parse_url("dvid://emdata3:8900/abc9:main/segmentation") == {
        "server": "emdata3:8900", "uuid": "abc9:main", "instance": "segmentation"}


def test_https_scheme_is_carried_to_the_server_string():
    """neuclease prepends http:// to a bare host, so TLS needs its own scheme."""
    spec = dvid.parse_url("dvid+https://example.org/abc9/labels")
    assert spec["server"] == "https://example.org"
    assert dvid.spec_url(spec) == "dvid+https://example.org/abc9/labels"


def test_spec_url_round_trips():
    assert dvid.spec_url(dvid.parse_url(URL)) == URL


@pytest.mark.parametrize("bad", [
    "dvid://server/uuid",                 # no instance
    "dvid://server/uuid/instance/extra",  # too many
    "dvid://server",
    "s3://bucket/prefix",                 # not DVID at all
])
def test_a_malformed_url_raises_and_says_what_was_expected(bad):
    with pytest.raises(ValueError) as e:
        dvid.parse_url(bad)
    assert "dvid://" in str(e.value)


def test_is_url_recognises_both_schemes_and_nothing_else():
    assert dvid.is_url(URL) and dvid.is_url("dvid+https://a/b/c")
    assert not dvid.is_url("/local/path")
    assert not dvid.is_url("s3://bucket/prefix")
    assert not dvid.is_url({"driver": "file", "path": "/x"})


# --------------------------------------------------------------------------- #
# detection and spec building
# --------------------------------------------------------------------------- #
def test_detect_backend_answers_from_the_scheme_without_touching_a_store(monkeypatch):
    """The carve-out that matters: `to_kvstore` would read the URL as a local path,
    probe the filesystem, find nothing and report "could not detect source format"."""
    def explode(*a, **k):                       # nothing may reach the kvstore layer
        raise AssertionError("detect_backend touched a store for a DVID URL")

    monkeypatch.setattr(sm, "to_kvstore", explode)
    assert sm.detect_backend(URL) == "dvid"


def test_location_spec_builds_the_three_forms():
    assert sm.location_spec(URL, "dvid")["backend"] == "dvid"
    assert sm.location_spec(URL, "dvid")["instance"] == "labels"
    assert sm.location_spec("/some/dir", "image_stack") == {
        "backend": "image_stack", "source": "/some/dir"}
    assert sm.location_spec("/v", "zarr3") == {"backend": "zarr3", "path": "/v"}


def test_level_spec_selects_a_dvid_scale_by_index():
    spec = sm.level_spec(URL, "dvid", 2)
    assert spec["scale_index"] == 2 and spec["instance"] == "labels"


def test_no_format_can_shadow_a_dvid_instance():
    """`other_format_markers` would otherwise read `dvid://...` as a local path."""
    assert sm.other_format_markers(URL, "dvid") == []


# --------------------------------------------------------------------------- #
# metadata, against a stubbed /info
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_cached_nodes():
    """Ref resolutions are memoized per process, so a leaked entry makes one test's
    answer another test's premise. Cheap to clear; confusing not to."""
    dvid.clear_node_cache()
    yield
    dvid.clear_node_cache()


@pytest.fixture()
def stub_info(monkeypatch):
    """Stub BOTH the instance info and the ref resolution.

    Stubbing only the former left `_read_dvid` calling the real `resolve_node`, which
    reached out to a live DVID server — so the suite needed the network, took a network
    round trip per test, and cached a real uuid that then leaked into tests expecting a
    fake one. Tests here must not touch a server at all.

    Patched on `em_volume_tools.dvid`, the module that *defines* them. `backends.dvid`
    re-exports the same names, but a re-export is a separate binding: patching it would
    leave `source_metadata` — which imports from the canonical module — calling the real
    thing. Both call sites reach these through the canonical module attribute so that one
    patch here covers everything; see `test_stubbing_the_canonical_module_covers_...`.
    """
    monkeypatch.setattr(canonical_dvid, "instance_info", lambda spec: INFO)
    monkeypatch.setattr(canonical_dvid, "resolve_node",
                        lambda spec, prefer_locked=False: {
                            "ref": str(spec.get("uuid")),
                            "uuid": "846e3a" if prefer_locked else "d38898",
                            "locked": bool(prefer_locked),
                            "walked": 1 if prefer_locked else 0})


def test_geometry_is_origin_anchored_not_translated_by_minpoint(stub_info):
    """MinPoint says where data starts; it is not a translation to apply. Anchoring at
    the origin keeps a voxel index meaning the same thing here and in DVID, and the
    empty margin costs one ~30 ms round trip per task and nothing on disk."""
    geom = dvid.geometry(INFO, dvid.parse_url(URL))
    assert geom["shape0"] == (8192, 9216, 12800)      # zyx, MaxPoint + 1
    assert geom["min_point"] == (0, 1536, 1024)       # recorded, not subtracted
    assert geom["chunks"] == (64, 64, 64)
    assert geom["voxel_size"] == (8.0, 8.0, 8.0)


def test_read_source_metadata_reports_a_segmentation(stub_info):
    """Never `image`: averaging label ids invents ids that were never in the data, and
    the failure is silent."""
    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["kind"] == "segmentation"
    assert meta["voxel_size"] == (8.0, 8.0, 8.0)
    assert meta["offset"] == (0.0, 0.0, 0.0)
    assert meta["units"] == "nm" and meta["has_channels"] is False


def test_data_spec_carries_the_dvid_backend_through(stub_info):
    """Invariant 9: `data_spec` selects the reader, so it must not contradict what
    detection decided — the bug that made every block read as zeros for .gz precomputed."""
    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["data_spec"]["backend"] == "dvid"
    assert meta["data_spec"]["instance"] == "labels"


def test_level_voxel_sizes_follow_dvids_own_downres_depth(stub_info):
    sizes = sm.read_level_voxel_sizes(sm.location_spec(URL, "dvid"))
    assert sizes == [(8.0,) * 3, (16.0,) * 3, (32.0,) * 3, (64.0,) * 3]


def test_level_shape_halves_every_axis():
    assert dvid.level_shape((8192, 9216, 12800), 0) == (8192, 9216, 12800)
    assert dvid.level_shape((8192, 9216, 12800), 3) == (1024, 1152, 1600)
    assert dvid.level_shape((10, 10, 10), 2) == (3, 3, 3)      # ceil, not floor


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_a_non_labelmap_instance_is_refused():
    info = {**INFO, "Base": {"TypeName": "uint8blk"}}
    with pytest.raises(ValueError, match="labelmap"):
        dvid.geometry(info, dvid.parse_url(URL))


def test_an_unwritten_instance_says_so_rather_than_reading_as_empty():
    """DVID returns null extents for an instance created but never written. Treated as
    an error: a zero-shaped volume would convert successfully and produce nothing."""
    info = {**INFO, "Extended": {**INFO["Extended"], "MinPoint": None, "MaxPoint": None}}
    with pytest.raises(ValueError, match="never written"):
        dvid.geometry(info, dvid.parse_url(URL))


def test_a_negative_minpoint_is_refused_rather_than_clipped():
    info = {**INFO, "Extended": {**INFO["Extended"], "MinPoint": [-64, 0, 0]}}
    with pytest.raises(ValueError, match="negative MinPoint"):
        dvid.geometry(info, dvid.parse_url(URL))


def test_writing_to_dvid_is_refused(stub_info):
    be = dvid.DVIDBackend(dvid.parse_url(URL))
    with pytest.raises(TypeError, match="read-only"):
        be.write_region((slice(0, 1),) * 3, np.zeros((1, 1, 1), np.uint64))


def test_a_scale_above_maxdownreslevel_is_refused(stub_info):
    with pytest.raises(ValueError, match="MaxDownresLevel"):
        dvid.DVIDBackend({**dvid.parse_url(URL), "scale_index": 4})


def test_supervoxels_on_a_non_dvid_source_raises_rather_than_being_ignored(tmp_path):
    """Silently dropping it would copy agglomerated bodies while the caller believed
    they had asked for supervoxels, and nothing downstream could tell."""
    from em_volume_tools import convert

    with pytest.raises(ValueError, match="DVID sources only"):
        convert({"backend": "zarr3", "path": str(tmp_path / "x")},
                str(tmp_path / "out"), supervoxels=True, voxel_size=(8, 8, 8))


def test_the_backend_is_registered_under_its_tag():
    from em_volume_tools.backends.base import _OPENERS

    assert dvid.TAG in _OPENERS


# --------------------------------------------------------------------------- #
# ref resolution: a branch ref names a different node tomorrow
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_dag(monkeypatch):
    """A lock-and-spawn repo: HEAD open, every ancestor locked.

    Matches a production instance, where 447 of 448 nodes on `main` are locked and the open
    node is HEAD.
    """
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd

    locked = {"d38898": False, "846e3a": True, "5ad376": True,
              # The full-length form, for the placeholder-truncation tests.
              "d38898ac94c8400baeb15103bab7f850": False}
    refs = {"93fdbc:main": "d38898", "93fdbc:main~1": "846e3a",
            "93fdbc:main~2": "5ad376"}

    def resolve_ref(server, ref, expand=False, **k):
        # A concrete uuid resolves to itself, as the real one does; anything else must
        # be a ref we know about.
        return refs[ref] if ref in refs else ref if ref in locked else _unknown(ref)

    def _unknown(ref):
        raise RuntimeError(f'"{ref}" is not a known uuid')

    monkeypatch.setattr(nd, "resolve_ref", resolve_ref)
    monkeypatch.setattr(nd, "fetch_commit", lambda server, uuid, **k: locked[uuid])
    return refs


def test_a_branch_ref_resolves_to_head_and_reports_it_open(fake_dag):
    node = dvid.resolve_node(dvid.parse_url(URL))
    assert node == {"ref": "93fdbc:main", "uuid": "d38898", "locked": False, "walked": 0}


def test_prefer_locked_walks_back_to_the_newest_locked_node(fake_dag):
    node = dvid.resolve_node(dvid.parse_url(URL), prefer_locked=True)
    assert node["uuid"] == "846e3a" and node["locked"] and node["walked"] == 1


def test_prefer_locked_is_a_no_op_when_the_ref_is_already_locked(fake_dag):
    url = "dvid://dvid.example.org/93fdbc:main~1/labels"
    node = dvid.resolve_node(dvid.parse_url(url), prefer_locked=True)
    assert node["uuid"] == "846e3a" and node["walked"] == 0


def test_prefer_locked_from_a_bare_uuid_says_why_it_cannot_walk(fake_dag):
    """DVID's ref~N syntax needs repo:branch; a bare uuid already names one node."""
    url = "dvid://dvid.example.org/d38898/labels"
    with pytest.raises(ValueError, match="repo:branch"):
        dvid.resolve_node(dvid.parse_url(url), prefer_locked=True)


def test_the_resolved_uuid_reaches_the_workers_not_the_ref(fake_dag, stub_info):
    """The correctness point: workers reopen from `data_spec`. If it carried the ref, a
    lock-and-spawn mid-run would move HEAD and later blocks would come from a different
    node than earlier ones, with nothing in the output to show it."""
    meta = sm.read_source_metadata({**sm.location_spec(URL, "dvid"),
                                    "prefer_locked": True})
    assert meta["data_spec"]["uuid"] == "846e3a"        # not '93fdbc:main'
    assert "prefer_locked" not in meta["data_spec"]     # applied, not carried


def test_prefer_locked_on_a_non_dvid_source_raises(tmp_path):
    from em_volume_tools import convert

    with pytest.raises(ValueError, match="DVID sources only"):
        convert({"backend": "zarr3", "path": str(tmp_path / "x")},
                str(tmp_path / "out"), prefer_locked=True, voxel_size=(8, 8, 8))


# --------------------------------------------------------------------------- #
# what the output records, and what `copy` refuses
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_repo(monkeypatch):
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_maxlabel", lambda *a, **k: 139103922, raising=False)
    monkeypatch.setattr(nd, "fetch_repo_info", lambda *a, **k: {
        "DAG": {"Nodes": {"846e3a": {"Branch": "main", "Note": "Locked via fivol API",
                                     "VersionID": 452, "Created": "c", "Updated": "u",
                                     "Log": ["enormous", "do not record"]}}}},
                        raising=False)


def test_provenance_names_the_resolved_node_and_what_was_asked_for(fake_dag, fake_repo):
    """Both matter: the uuid is what was read, the ref is how it was chosen. 'the newest
    locked node on main' and 'this node' are different claims even when they agree today."""
    from em_volume_tools.ops import provenance as prov

    spec = {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
            "requested_ref": "93fdbc:main", "ancestors_walked": 1}
    rec = prov.build_record(src_spec=spec, dst="/out", kind="segmentation")
    src = rec["source"]
    assert src["uuid"] == "846e3a" and src["requested"] == "93fdbc:main"
    assert src["locked"] is True and src["ancestors_walked"] == 1
    assert src["url"].endswith("/846e3a/labels")      # pinned, not the branch ref
    assert src["maxlabel"] == 139103922
    assert src["node"]["Branch"] == "main" and src["node"]["VersionID"] == 452
    assert "Log" not in src["node"]                   # the mutation log can be enormous
    assert rec["run"]["kind"] == "segmentation"


def test_provenance_survives_a_server_that_will_not_answer(fake_dag, monkeypatch):
    """A completed conversion must not be failed by a missing extra field."""
    import neuclease.dvid as nd
    from em_volume_tools.ops import provenance as prov

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(nd, "fetch_maxlabel", boom, raising=False)
    monkeypatch.setattr(nd, "fetch_repo_info", boom, raising=False)
    src = prov.build_record(src_spec={**sm.location_spec(URL, "dvid"),
                                      "uuid": "846e3a"}, dst="/out")["source"]
    assert src["uuid"] == "846e3a"
    assert "nope" in src["maxlabel_error"] and "nope" in src["node_error"]


def test_provenance_falls_back_to_describing_any_other_spec():
    from em_volume_tools.ops import provenance as prov

    rec = prov.build_record(src_spec={"backend": "zarr3", "path": "/v"}, dst="/out")
    assert rec["source"] == {"source": "zarr3", "path": "/v"}


def test_copy_refuses_a_dvid_source_and_points_at_convert():
    from em_volume_tools import cli

    args = cli._parse_args(["copy", "--src", URL, "--dst", "/out"])
    with pytest.raises(SystemExit) as e:
        cli.cmd_copy(args)
    assert "convert" in str(e.value) and "not a storage format" in str(e.value)


# --------------------------------------------------------------------------- #
# what `em-vol info` reports
# --------------------------------------------------------------------------- #
def test_node_summary_gives_both_candidate_versions(fake_dag):
    """`info` shows both because they are different answers to "which version would I
    get", and choosing between them is choosing whether the export is reproducible."""
    s = dvid.node_summary(dvid.parse_url(URL))
    assert s["head"]["uuid"] == "d38898" and s["head"]["locked"] is False
    assert s["locked"]["uuid"] == "846e3a" and s["locked"]["walked"] == 1


def test_node_summary_collapses_when_the_ref_is_already_locked(fake_dag):
    url = "dvid://dvid.example.org/93fdbc:main~1/labels"
    s = dvid.node_summary(dvid.parse_url(url))
    assert s["head"]["uuid"] == s["locked"]["uuid"] == "846e3a"


def test_node_summary_reports_rather_than_raises_when_no_locked_node_is_reachable(
        fake_dag):
    """A bare uuid cannot be walked back from on a multi-repo server. `info` must still
    print the rest of what it knows."""
    url = "dvid://dvid.example.org/d38898/labels"
    s = dvid.node_summary(dvid.parse_url(url))
    assert s["head"]["uuid"] == "d38898"
    assert s["locked"] is None and "repo:branch" in s["locked_error"]


def test_info_prints_both_uuids_in_full(fake_dag, stub_info, capsys):
    """Printed in full because building a destination name from one is the usual reason
    to run this."""
    from em_volume_tools import cli

    cli.cmd_info(cli._parse_args(["info", URL]))
    out = capsys.readouterr().out
    assert "d38898" in out and "846e3a" in out
    assert "OPEN" in out and "--dvid-locked" in out


def test_info_summarises_a_provenance_file_and_can_print_it_whole(tmp_path, capsys):
    import json

    from em_volume_tools import cli

    vol = tmp_path / "v"
    vol.mkdir()
    (vol / "provenance.json").write_text(json.dumps(
        {"written": "2026-08-15T00:00:00+00:00",
         "source": {"source": "dvid", "url": "dvid://s/abc/labels", "uuid": "abc",
                    "locked": True, "maxlabel": 7}}))

    cli._print_provenance(str(vol), False)
    brief = capsys.readouterr().out
    assert "dvid://s/abc/labels" in brief and "locked" in brief
    assert "maxlabel" not in brief                 # summary, not the whole document

    cli._print_provenance(str(vol), True)
    assert "maxlabel" in capsys.readouterr().out


def test_info_says_nothing_about_provenance_when_there_is_none(tmp_path, capsys):
    from em_volume_tools import cli

    cli._print_provenance(str(tmp_path), False)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# naming a destination after the node it came from
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dvid_src(fake_dag):
    """A resolved DVID spec, as `read_source_metadata` hands one on."""
    return {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
            "requested_ref": "93fdbc:main"}


def test_uuid_defaults_to_eight_characters(dvid_src):
    from em_volume_tools.ops.naming import expand

    assert expand("/d/seg_{uuid}", dvid_src) == "/d/seg_846e3a"   # stub uuid is short


def test_uuid_length_and_full_are_selectable(fake_dag):
    from em_volume_tools.ops.naming import expand

    src = {**sm.location_spec(URL, "dvid"), "uuid": "d38898ac94c8400baeb15103bab7f850"}
    assert expand("{uuid}", src) == "d38898ac"
    assert expand("{uuid:6}", src) == "d38898"
    assert expand("{uuid:full}", src) == "d38898ac94c8400baeb15103bab7f850"


def test_branch_and_instance_are_available(dvid_src):
    from em_volume_tools.ops.naming import expand

    assert expand("/d/{instance}_{branch}", dvid_src) == "/d/labels_main"


def test_a_destination_without_placeholders_is_untouched():
    from em_volume_tools.ops.naming import expand

    # Also must not resolve anything — an ordinary destination costs no request.
    assert expand("/plain/path", {"backend": "zarr3"}) == "/plain/path"


def test_expansion_is_idempotent(dvid_src):
    """The CLI expands before deriving targets; `convert` may expand again."""
    from em_volume_tools.ops.naming import expand

    once = expand("/d/seg_{uuid}", dvid_src)
    assert expand(once, dvid_src) == once


def test_a_placeholder_on_a_source_that_has_none_raises():
    from em_volume_tools.ops.naming import expand

    with pytest.raises(ValueError, match="supplies none"):
        expand("/d/seg_{uuid}", {"backend": "zarr3", "path": "/v"})


def test_an_unknown_placeholder_lists_the_real_ones(dvid_src):
    from em_volume_tools.ops.naming import expand

    with pytest.raises(ValueError, match=r"\{uuid\}"):
        expand("/d/{nonesuch}", dvid_src)


@pytest.mark.parametrize("bad", ["{uuid:0}", "{uuid:xyz}", "{uuid:-2}"])
def test_a_bad_length_spec_raises(dvid_src, bad):
    from em_volume_tools.ops.naming import expand

    with pytest.raises(ValueError, match="positive number"):
        expand(bad, dvid_src)


def test_values_are_sanitised_for_paths_and_object_keys(fake_dag):
    """A ref can carry '~'; an object-store key is happier without it."""
    from em_volume_tools.ops.naming import expand

    src = {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
           "requested_ref": "93fdbc:main~1"}
    assert expand("{branch}", src) == "main-1"


def test_convert_kind_defaults_to_none_so_the_source_can_decide():
    """`--kind` used to default to 'image' for `convert`, which downgraded a source that
    recorded 'segmentation' and averaged its label ids."""
    from em_volume_tools import cli

    assert cli._parse_args(["convert", "--src", "a", "--dst", "b"]).kind is None


# --------------------------------------------------------------------------- #
# the addressing / labelmap split
# --------------------------------------------------------------------------- #
def test_the_backend_reexports_the_shared_names_without_copying_them():
    """One implementation, two import paths. If these ever drift, the module that got
    patched and the module that got called stop being the same thing."""
    for name in ("parse_url", "is_url", "spec_url", "resolve_node", "node_summary",
                 "instance_info", "node_provenance", "clear_node_cache",
                 "check_instance_type"):
        assert getattr(dvid, name) is getattr(canonical_dvid, name), name


def test_stubbing_the_canonical_module_covers_every_call_site(monkeypatch):
    """The regression guard for the addressing split.

    `source_metadata` and the backend both reach these through
    `em_volume_tools.dvid.<name>`, so one patch governs both. When the backend held its
    own copies, patching it left `source_metadata` calling the real function — the suite
    passed only because a live DVID server happened to answer, which is the failure this
    test exists to make impossible.
    """
    calls = []

    def spy_info(spec):
        calls.append("instance_info")
        return INFO

    def spy_node(spec, prefer_locked=False):
        calls.append("resolve_node")
        return {"ref": "r", "uuid": "d38898", "locked": True, "walked": 0}

    monkeypatch.setattr(canonical_dvid, "instance_info", spy_info)
    monkeypatch.setattr(canonical_dvid, "resolve_node", spy_node)

    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["kind"] == "segmentation"
    assert "instance_info" in calls and "resolve_node" in calls

    calls.clear()
    dvid.DVIDBackend(sm.location_spec(URL, "dvid"))
    assert "instance_info" in calls


def test_instance_type_and_syncs_are_read_from_base():
    ann = {"Base": {"TypeName": "annotation", "Syncs": ["labels"]}}
    assert canonical_dvid.instance_type(ann) == "annotation"
    assert canonical_dvid.synced_instances(ann) == ["labels"]
    # A missing or null Syncs is a plain empty list, not None — callers iterate it.
    assert canonical_dvid.synced_instances({"Base": {"TypeName": "keyvalue"}}) == []
    assert canonical_dvid.synced_instances({"Base": {"Syncs": None}}) == []


def test_check_instance_type_names_what_it_wanted():
    spec = canonical_dvid.parse_url(URL)
    assert canonical_dvid.check_instance_type(
        {"Base": {"TypeName": "annotation"}}, spec, "annotation") == "annotation"
    with pytest.raises(ValueError, match="expected annotation"):
        canonical_dvid.check_instance_type({"Base": {"TypeName": "keyvalue"}},
                                          spec, "annotation")
    # More than one acceptable type reads as a list, not a tuple repr.
    with pytest.raises(ValueError, match="expected annotation or keyvalue"):
        canonical_dvid.check_instance_type({"Base": {"TypeName": "roi"}}, spec,
                                          "annotation", "keyvalue")


def test_node_provenance_carries_no_labelmap_fields(fake_repo, fake_dag):
    """`maxlabel` and `supervoxels` describe a label array; the shared record must not
    claim them, or a synapse table's provenance would assert things about labels."""
    spec = canonical_dvid.parse_url(URL)
    node = canonical_dvid.resolve_node(spec)
    rec = canonical_dvid.node_provenance(spec, node)
    assert "maxlabel" not in rec and "supervoxels" not in rec
    assert rec["uuid"] == "d38898" and rec["source"] == "dvid"
    # The labelmap backend is what adds them.
    assert "supervoxels" in dvid.provenance(spec, node)
