"""The webknossos source, against a stubbed server.

What is pinned, because each fails silently rather than loudly:

- the AXIS FLIP: webknossos serves Fortran-ordered (x, y, z), and reading it as zyx
  without the flip mirrors the volume through the z=x diagonal (invariant ZYX-XYZ);
- the REQUEST COORDINATES: position in mag-1 voxels, extent in the mag's own — mixing
  them reads the wrong place at every mag but the first;
- the MAPPING reaches the server, and no mapping means supervoxels;
- the TOKEN never enters a spec, so it cannot reach a manifest or provenance.json.

The stub replaces `neu_vol.webknossos._request` — the one module attribute every call goes
through, so one patch point covers metadata and data alike (the rule NOTES-neu-vol.md
records for the DVID modules).
"""

from __future__ import annotations

import json
import os
import urllib.parse

import numpy as np
import pytest

from neu_vol import source_metadata as sm
from neu_vol import webknossos as wk
from neu_vol.backends import webknossos as wkb
from neu_vol.backends.base import clear_backend_cache, open_backend

URL = "https://wk.example.org"
DATASTORE = "https://data.wk.example.org"
TOKEN = "secret-token-123"
DSID = "0123456789abcdef01234567"

# Bounding box in xyz, as webknossos records it. Deliberately not at the origin and
# deliberately unequal per axis, so a transposed read cannot pass by symmetry.
TOP_LEFT = (4, 2, 1)
SIZE = (10, 7, 5)
HI = tuple(a + b for a, b in zip(TOP_LEFT, SIZE))         # (14, 9, 6)
MAGS = [(1, 1, 1), (2, 2, 1)]                              # the second is anisotropic


def _volume_xyz() -> np.ndarray:
    vol = np.zeros(HI, dtype=np.uint32)
    x0, y0, z0 = TOP_LEFT
    inner = np.arange(1, np.prod(SIZE) + 1, dtype=np.uint32).reshape(SIZE)
    vol[x0:, y0:, z0:] = inner
    return vol


def _mapped(a: np.ndarray) -> np.ndarray:
    return np.where(a > 0, a % 7 + 1, 0).astype(a.dtype)


class FakeServer:
    def __init__(self):
        self.xyz = _volume_xyz()
        self.data_requests: list[dict] = []
        self.tokens: set[str | None] = set()
        self.failed_buckets = "[]"
        self.datasets = [{"id": DSID, "name": "specimen", "directoryName": "specimen",
                          "owningOrganization": "lab", "created": 1790000000000,
                          "lastUsedByUser": 1791000000000, "usedStorageBytes": 3 * 2**30,
                          "segmentationLayerNames": ["segmentation"],
                          "colorLayerNames": []}]
        self.list_queries: list[dict] = []

    def info(self):
        return {
            "id": DSID, "name": "specimen", "owningOrganization": "lab",
            "description": "Workflow Report: [abc](https://wk.example.org/workflows/abc)",
            "uploaderFullName": "A. Uploader", "created": 1790000000000,
            "lastUsedByUser": 1791000000000, "tags": [], "usedStorageBytes": 3 * 2**30,
            "dataStore": {"url": DATASTORE},
            "dataSource": {
                "scale": {"factor": [8.0, 8.0, 30.0], "unit": "nanometer"},
                "dataLayers": [{
                    "name": "segmentation", "category": "segmentation",
                    "elementClass": "uint32", "numChannels": 1,
                    # The form the real server sends (a bigint wrapper), not a number.
                    "largestSegmentId": {"customJsonEncoding": "bigint",
                                         "value": str(int(self.xyz.max()))},
                    "boundingBox": {"topLeft": list(TOP_LEFT), "width": SIZE[0],
                                    "height": SIZE[1], "depth": SIZE[2]},
                    "mags": [{"mag": list(m)} for m in MAGS],
                    "defaultViewConfiguration": {
                        "mapping": {"name": "agglomerate_view_70", "type": "HDF5"}},
                }],
            },
        }

    def __call__(self, url, *, token, query=None, timeout=None, data=None,
                 content_type=None):
        self.tokens.add(token)
        q = dict(query or {})
        if url.endswith("/agglomeratesForSegments"):
            assert content_type == "application/x-protobuf"
            self.table_requests = getattr(self, "table_requests", 0) + 1
            ids = wk.decode_list_of_long(data)
            return wk.encode_list_of_long(_mapped(ids)), {}
        if url == f"{URL}/api/buildinfo":
            return json.dumps({"httpApiVersioning": {
                "currentApiVersion": 16, "oldestSupportedApiVersion": 5}}).encode(), {}
        if url == f"{URL}/api/v16/datasets":
            self.list_queries.append(q)
            hits = [d for d in self.datasets if q.get("searchQuery", "") in d["name"]]
            return json.dumps(hits).encode(), {}
        if url == f"{URL}/api/v16/datasets/{DSID}":
            return json.dumps(self.info()).encode(), {}
        base = f"{DATASTORE}/data/v15/datasets/{DSID}/layers/segmentation"
        if url == f"{base}/agglomerates":
            return json.dumps(["agglomerate_view_70", "agglomerate_view_10"]).encode(), {}
        if url == f"{base}/mappings":
            return b"[]", {}
        if url == f"{base}/data":
            self.data_requests.append(q)
            mag = tuple(int(v) for v in q["mag"].split("-"))
            arr = self.xyz[::mag[0], ::mag[1], ::mag[2]]
            x, y, z = (q[a] // m for a, m in zip("xyz", mag))
            sub = arr[x:x + q["width"], y:y + q["height"], z:z + q["depth"]]
            assert sub.shape == (q["width"], q["height"], q["depth"]), "request off the end"
            if q.get("mappingName"):
                sub = _mapped(sub)
            if self.failed_buckets != "[]":
                sub = np.zeros_like(sub)        # what the real server does: 200, zeros
            return sub.tobytes(order="F"), {"MISSING-BUCKETS": "[]",
                                            "failure-bucket-indices": self.failed_buckets,
                                            "empty-bucket-indices": "[]"}
        raise AssertionError(f"unexpected request {url}")


@pytest.fixture()
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setenv("WK_TOKEN", TOKEN)
    monkeypatch.setenv("WK_URL", URL)
    monkeypatch.setattr(wk, "_request", fake)
    wk.clear_cache()
    clear_backend_cache()
    yield fake
    wk.clear_cache()
    clear_backend_cache()


def _zyx(xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(xyz.transpose(2, 1, 0))


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #
def test_url_splits_into_dataset_and_layer():
    assert wk.parse_url("wk://specimen/segmentation") == {
        "dataset": "specimen", "layer": "segmentation"}


@pytest.mark.parametrize("bad", ["wk://specimen", "wk://host/specimen/segmentation",
                                 "dvid://a/b/c"])
def test_a_malformed_url_raises(bad):
    with pytest.raises(ValueError):
        wk.parse_url(bad)


def test_detect_backend_answers_from_the_scheme_without_touching_a_store(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("a wk:// URL reached to_kvstore")

    monkeypatch.setattr(sm, "to_kvstore", explode)
    assert sm.detect_backend("wk://specimen/segmentation") == "webknossos"
    assert sm.other_format_markers("wk://specimen/segmentation", "webknossos") == []


def test_level_spec_selects_a_mag_by_index():
    assert sm.level_spec("wk://specimen/segmentation", "webknossos", 1) == {
        "backend": "webknossos", "dataset": "specimen", "layer": "segmentation",
        "scale_index": 1}


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def test_credentials_come_from_a_file_when_the_environment_has_none(tmp_path, monkeypatch):
    monkeypatch.delenv("WK_TOKEN", raising=False)
    f = tmp_path / "credentials.toml"
    f.write_text('url = "https://wk.example.org/"\ntoken = "abc"\norganization = "lab"\n')
    f.chmod(0o600)
    monkeypatch.setenv(wk.ENV_CREDENTIALS, str(f))
    assert wk.credentials() == {"url": "https://wk.example.org", "token": "abc",
                                "organization": "lab"}


def test_missing_credentials_say_what_file_to_create(tmp_path, monkeypatch):
    monkeypatch.delenv("WK_TOKEN", raising=False)
    monkeypatch.setenv(wk.ENV_CREDENTIALS, str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError, match="token = "):
        wk.credentials()


def test_a_world_readable_credentials_file_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("WK_TOKEN", raising=False)
    f = tmp_path / "credentials.toml"
    f.write_text('token = "abc"\n')
    f.chmod(0o644)
    monkeypatch.setenv(wk.ENV_CREDENTIALS, str(f))
    wk.credentials()
    assert "chmod 600" in caplog.text


# --------------------------------------------------------------------------- #
# geometry and reads
# --------------------------------------------------------------------------- #
def test_levels_are_mags_with_true_anisotropic_voxel_sizes(server):
    geom = wkb.geometry(server.info(), "segmentation")
    assert [lv["voxel_size"] for lv in geom["levels"]] == [(30.0, 8.0, 8.0),
                                                           (30.0, 16.0, 16.0)]
    # Origin-anchored zyx: shape reaches the bounding box's far corner.
    assert geom["levels"][0]["shape"] == (6, 9, 14)
    assert geom["levels"][1]["shape"] == (6, 5, 7)
    assert geom["kind"] == "segmentation"
    assert geom["default_mapping"] == "agglomerate_view_70"


def test_a_full_read_is_the_volume_in_zyx_not_mirrored(server):
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    got = be.read_region(tuple(slice(0, s) for s in be.shape))
    np.testing.assert_array_equal(got, _zyx(server.xyz))


def test_a_box_outside_the_bounding_box_is_zeros_without_a_request(server):
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    got = be.read_region((slice(0, 1), slice(0, 2), slice(0, 4)))  # z<1, y<2, x<4
    assert not got.any()
    assert server.data_requests == []


def test_a_partial_overlap_requests_only_the_part_inside(server):
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    region = (slice(0, 3), slice(1, 5), slice(2, 9))
    got = be.read_region(region)
    np.testing.assert_array_equal(got, _zyx(server.xyz)[region])
    (q,) = server.data_requests
    assert (q["x"], q["y"], q["z"]) == TOP_LEFT
    assert (q["width"], q["height"], q["depth"]) == (5, 3, 2)


def test_buckets_the_server_failed_to_load_retry_rather_than_read_as_zeros(server):
    """A failed bucket arrives zero-filled with a 200; only a header says so. Ignored, it
    was recorded as an empty block and a dense layer came out with holes."""
    from neu_vol.retry import is_transient

    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    server.failed_buckets = "[0, 3]"
    with pytest.raises(RuntimeError, match="failure-bucket-indices") as e:
        be.read_region(tuple(slice(0, s) for s in be.shape))
    assert is_transient(e.value)


def test_a_large_read_is_split_into_request_sized_pieces_on_a_grid(server, monkeypatch):
    """One huge request outruns the server's deadline and comes back zero-filled."""
    monkeypatch.setattr(wkb, "REQUEST_CHUNK", (2, 4, 4))
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    got = be.read_region(tuple(slice(0, s) for s in be.shape))
    np.testing.assert_array_equal(got, _zyx(server.xyz))
    assert len(server.data_requests) > 1
    for q in server.data_requests:
        assert q["width"] <= 4 and q["height"] <= 4 and q["depth"] <= 2
        assert q["x"] % 4 == 0 or q["x"] == TOP_LEFT[0]


def test_a_slow_response_is_retried_not_trusted(server, monkeypatch):
    from neu_vol.retry import is_transient

    monkeypatch.setattr(wkb, "SLOW_REQUEST_S", -1.0)
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation"})
    with pytest.raises(RuntimeError, match="zero-filled") as e:
        be.read_region(tuple(slice(0, s) for s in be.shape))
    assert is_transient(e.value)


def test_a_coarse_mag_positions_in_mag1_voxels_and_sizes_in_its_own(server):
    be = open_backend({"backend": "webknossos", "dataset": "specimen",
                       "layer": "segmentation", "scale_index": 1})
    got = be.read_region(tuple(slice(0, s) for s in be.shape))
    np.testing.assert_array_equal(got, _zyx(server.xyz[::2, ::2, ::1]))
    q = server.data_requests[-1]
    assert q["mag"] == "2-2-1"
    assert (q["x"], q["y"], q["z"]) == (4, 2, 1)          # mag-1 voxels
    assert (q["width"], q["height"], q["depth"]) == (5, 4, 5)   # mag (2,2,1) voxels


def test_no_mapping_reads_supervoxels_and_a_mapping_reaches_the_server(server):
    raw = open_backend({"backend": "webknossos", "dataset": "specimen",
                        "layer": "segmentation"})
    mapped = open_backend({"backend": "webknossos", "dataset": "specimen",
                           "layer": "segmentation", "mapping": "agglomerate_view_70"})
    full = tuple(slice(0, s) for s in raw.shape)
    np.testing.assert_array_equal(raw.read_region(full), _zyx(server.xyz))
    np.testing.assert_array_equal(mapped.read_region(full), _zyx(_mapped(server.xyz)))
    assert "mappingName" not in server.data_requests[0] or \
        server.data_requests[0]["mappingName"] is None


def test_requests_ask_for_gzip_and_decompress_it(monkeypatch):
    """Without it a read is wire-bound: measured 10x slower on real label data."""
    import gzip
    import io
    import urllib.request

    seen = {}

    class Resp(io.BytesIO):
        headers = {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        seen.update(req.headers)
        return Resp(gzip.compress(b"payload"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    body, _ = wk._request(f"{URL}/x", token=TOKEN)
    assert body == b"payload"
    assert seen.get("Accept-encoding") == "gzip"


def test_a_server_failure_is_classified_for_retry(monkeypatch):
    import urllib.error
    import urllib.request

    from neu_vol.retry import is_transient

    def fail(code):
        def urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
        return urlopen

    for code, transient in ((503, True), (403, False), (404, False)):
        monkeypatch.setattr(urllib.request, "urlopen", fail(code))
        with pytest.raises(RuntimeError) as e:
            wk._request(f"{URL}/x", token=TOKEN, query={"a": 1})
        assert is_transient(e.value) is transient
        assert TOKEN not in str(e.value)


# --------------------------------------------------------------------------- #
# metadata, resolution, provenance
# --------------------------------------------------------------------------- #
def test_data_spec_carries_the_resolved_id_and_backend_but_never_the_token(server):
    meta = sm.read_source_metadata(sm.location_spec("wk://specimen/segmentation",
                                                    "webknossos"))
    ds = meta["data_spec"]
    assert ds["backend"] == "webknossos" and ds["dataset_id"] == DSID
    assert ds["url"] == URL and ds["datastore"] == DATASTORE
    assert TOKEN not in json.dumps(ds) and TOKEN not in json.dumps(meta["provenance_spec"])
    assert meta["kind"] == "segmentation" and meta["voxel_size"] == (30.0, 8.0, 8.0)


def test_an_unknown_mapping_is_refused_up_front_naming_the_real_ones(server):
    spec = {**sm.location_spec("wk://specimen/segmentation", "webknossos"),
            "mapping": "agglomerate_view_71"}
    with pytest.raises(ValueError, match="agglomerate_view_10, agglomerate_view_70"):
        sm.read_source_metadata(spec)


def test_an_ambiguous_dataset_name_raises_rather_than_picking(server):
    server.datasets.append({"id": "f" * 24, "name": "specimen", "owningOrganization": "x"})
    with pytest.raises(ValueError, match="ambiguous"):
        wk.resolve_dataset_id("specimen")


# --------------------------------------------------------------------------- #
# listing (wk-ls)
# --------------------------------------------------------------------------- #
def _other_orgs(server):
    """Datasets a real server lists beside yours: other organizations' public ones."""
    server.datasets += [
        {"id": "a" * 24, "name": "specimen_public", "owningOrganization": "elsewhere",
         "created": 1600000000000, "segmentationLayerNames": ["seg"],
         "colorLayerNames": ["image"]},
        {"id": "b" * 24, "name": "specimen_5_v2", "owningOrganization": "lab",
         "created": 1795000000000, "segmentationLayerNames": ["gt", "specimen5"],
         "colorLayerNames": ["image"]},
    ]


def test_listing_is_one_request_filtered_to_the_organization_and_oldest_first(server):
    _other_orgs(server)
    found = wk.list_datasets(organization="lab")
    assert [d["name"] for d in found] == ["specimen", "specimen_5_v2"]
    assert len(server.list_queries) == 1
    assert {d["owningOrganization"] for d in wk.list_datasets()} == {"lab", "elsewhere"}


@pytest.mark.parametrize("pattern, names", [
    ("SPECIMEN_5", ["specimen_5_v2"]),           # substring, case-insensitive
    ("specimen", ["specimen", "specimen_5_v2"]),
    ("spec*v2", ["specimen_5_v2"]),              # glob: the whole name must match
    ("*imen", ["specimen"]),
])
def test_a_pattern_is_a_substring_or_a_glob(server, pattern, names):
    _other_orgs(server)
    assert [d["name"] for d in wk.list_datasets(pattern, organization="lab")] == names


def test_dates_are_utc_days_and_a_missing_one_is_a_dash():
    assert wk.iso_date(1790000000000) == "2026-09-21"
    assert wk.iso_date(None) == "-" and wk.iso_date(0) == "-"


def test_a_summary_carries_what_tells_datasets_apart(server):
    s = wk.dataset_summary(DSID)
    assert s["description"].startswith("Workflow Report")
    assert s["uploader"] == "A. Uploader" and s["created"] == "2026-09-21"
    (lyr,) = s["layers"]
    assert lyr["top_left_xyz"] == list(TOP_LEFT) and lyr["size_xyz"] == list(SIZE)
    assert lyr["default_mapping"] == "agglomerate_view_70"
    assert lyr["mappings"]["agglomerate"] == ["agglomerate_view_10", "agglomerate_view_70"]


def _wk_ls(capsys, *argv):
    from neu_vol.cli import main

    assert main(["wk-ls", *argv]) == 0
    return capsys.readouterr().out


def test_wk_ls_uses_the_credentials_organization(server, tmp_path, monkeypatch, capsys):
    _other_orgs(server)
    monkeypatch.delenv("WK_TOKEN", raising=False)
    f = tmp_path / "credentials.toml"
    f.write_text(f'url = "{URL}"\ntoken = "{TOKEN}"\norganization = "lab"\n')
    f.chmod(0o600)
    monkeypatch.setenv(wk.ENV_CREDENTIALS, str(f))
    out = _wk_ls(capsys)
    assert "specimen_5_v2" in out and "specimen_public" not in out
    assert "seg: gt, specimen5 | color: image" in out and "2026-09-21" in out
    assert "2 dataset(s) in organization lab" in out
    assert TOKEN not in out
    assert "specimen_public" in _wk_ls(capsys, "--all-orgs")


def test_wk_ls_without_an_organization_lists_everything_and_says_how_to_narrow(
        server, capsys):
    _other_orgs(server)
    out = _wk_ls(capsys)
    assert "specimen_public" in out and "set `organization`" in out
    assert "specimen_public" not in _wk_ls(capsys, "--org", "lab")


def test_wk_ls_details_shows_the_box_views_and_description(server, capsys):
    out = _wk_ls(capsys, "specimen", "--org", "lab", "--details")
    assert "description: Workflow Report" in out and "by A. Uploader" in out
    assert "box xyz 4,2,1 + 10x7x5" in out
    assert "views: agglomerate_view_10, agglomerate_view_70  (default: agglomerate_view_70)" \
        in out
    assert TOKEN not in out


def test_wk_ls_details_refuses_a_broad_match_before_asking_for_any(server, capsys):
    from neu_vol.cli import main

    _other_orgs(server)
    server.info = lambda: pytest.fail("fetched a dataset record despite the cap")
    with pytest.raises(SystemExit, match="--max-details 1"):
        main(["wk-ls", "--all-orgs", "--details", "--max-details", "1"])


def test_wk_mapping_on_another_source_raises_rather_than_being_ignored(tmp_path):
    from neu_vol import convert

    with pytest.raises(ValueError, match="webknossos sources only"):
        convert({"backend": "zarr3", "path": str(tmp_path / "x")},
                str(tmp_path / "y"), wk_mapping="agglomerate_view_70")


@pytest.mark.parametrize("mapping", [None, "agglomerate_view_70"])
def test_convert_writes_the_layer_zyx_and_records_what_it_read(server, tmp_path, mapping):
    from neu_vol import convert

    dst = str(tmp_path / "out")
    convert("wk://specimen/segmentation", dst, profile="local", chunk=(4, 4, 4),
            multiscale=False, wk_mapping=mapping, delete_existing=True)
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    want = _zyx(_mapped(server.xyz) if mapping else server.xyz)
    np.testing.assert_array_equal(lvl0.read_region(tuple(slice(0, s) for s in lvl0.shape)),
                                  want)
    with open(os.path.join(dst, "provenance.json")) as fh:
        text = fh.read()
    rec = json.loads(text)["source"]
    assert rec["dataset_id"] == DSID and rec["mapping"] == mapping
    assert rec["ids"] == ("agglomerates" if mapping else "supervoxels (raw ids)")
    assert TOKEN not in text


def test_the_backend_is_registered_under_its_tag():
    from neu_vol.backends.base import _OPENERS

    assert "webknossos" in _OPENERS


# --------------------------------------------------------------------------- #
# agglomerate tables and mapped views
# --------------------------------------------------------------------------- #
IDS = np.array([0, 1, 8, 10, 127, 128, 300, 2**32 + 5, 2**63, 2**64 - 1], np.uint64)


def test_list_of_long_round_trips_including_a_lone_0x0a_byte():
    # 10 encodes as the single byte 0x0A — also the packed-form tag. Decoding by byte
    # rather than by the first tag would misread it.
    data = wk.encode_list_of_long(IDS)
    np.testing.assert_array_equal(wk.decode_list_of_long(data), IDS)
    assert wk.decode_list_of_long(b"").size == 0


def test_list_of_long_matches_the_protobuf_library_both_forms():
    pytest.importorskip("google.protobuf")
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    pool = descriptor_pool.DescriptorPool()
    classes = {}
    for packed in (False, True):
        fd = descriptor_pb2.FileDescriptorProto(name=f"l{packed}.proto",
                                                package=f"p{int(packed)}", syntax="proto2")
        field = fd.message_type.add(name="ListOfLong").field.add(
            name="items", number=1, type=4, label=3)                 # repeated uint64
        field.options.packed = packed
        pool.Add(fd)
        classes[packed] = message_factory.GetMessageClass(
            pool.FindMessageTypeByName(f"p{int(packed)}.ListOfLong"))
    items = [int(i) for i in IDS]
    # Unpacked is proto2's default and what the server sends: byte-identical.
    assert wk.encode_list_of_long(IDS) == classes[False](items=items).SerializeToString()
    for packed in (False, True):
        np.testing.assert_array_equal(
            wk.decode_list_of_long(classes[packed](items=items).SerializeToString()), IDS)


@pytest.fixture()
def sv_volume(server, tmp_path):
    """The layer's supervoxels, converted the way a user would: no mapping."""
    from neu_vol import convert

    dst = str(tmp_path / "sv.zarr")
    convert("wk://specimen/segmentation", dst, profile="local", chunk=(4, 4, 4),
            min_dim=4, delete_existing=True)
    return dst


def test_tables_and_descriptors_are_written_and_a_rerun_refetches_nothing(
        server, sv_volume, tmp_path):
    from neu_vol.ops.wk_mappings import fetch_mapping_tables

    out = str(tmp_path / "views")
    index = fetch_mapping_tables("wk://specimen/segmentation", out, volume=sv_volume)
    n_ids = int(server.xyz.max()) + 1
    assert index["n_ids"] == n_ids and index["dataset_id"] == DSID
    assert set(index["views"]) == {"agglomerate_view_10", "agglomerate_view_70"}
    table = np.load(os.path.join(out, "agglomerate_view_70.npy"))
    np.testing.assert_array_equal(table, _mapped(np.arange(n_ids, dtype=np.uint64)))
    assert os.path.isfile(os.path.join(out, "agglomerate_view_70.json"))

    before = server.table_requests
    fetch_mapping_tables("wk://specimen/segmentation", out, volume=sv_volume)
    assert server.table_requests == before


def test_an_unknown_view_is_refused_before_fetching(server, tmp_path):
    from neu_vol.ops.wk_mappings import fetch_mapping_tables

    with pytest.raises(ValueError, match="agglomerate_view_10, agglomerate_view_70"):
        fetch_mapping_tables("wk://specimen/segmentation", str(tmp_path / "v"),
                             mappings=["agglomerate_view_99"])
    assert getattr(server, "table_requests", 0) == 0


@pytest.fixture()
def view(server, sv_volume, tmp_path):
    from neu_vol import mapped
    from neu_vol.ops.wk_mappings import fetch_mapping_tables

    out = str(tmp_path / "views")
    fetch_mapping_tables("wk://specimen/segmentation", out, volume=sv_volume,
                         mappings=["agglomerate_view_70"])
    mapped.clear_cache()
    return os.path.join(out, "agglomerate_view_70.json")


def test_a_descriptor_is_a_volume_everywhere_a_location_is_taken(server, view, sv_volume):
    from neu_vol import read_piece
    from neu_vol.source_metadata import describe, detect_backend

    assert detect_backend(view) == "mapped"
    d = describe(view)
    assert d["format"] == "mapped" and d["meta"]["kind"] == "segmentation"
    # The view's levels are its source's, whatever pyramid that volume was given.
    assert d["shape"] == (6, 9, 14)
    assert d["levels"] == describe(sv_volume)["levels"]
    piece = read_piece(view)
    np.testing.assert_array_equal(np.asarray(piece.array), _zyx(_mapped(server.xyz)))
    assert tuple(piece.frame.voxel_size_nm) == (30.0, 8.0, 8.0)


def test_a_coarse_level_maps_the_sources_own_coarse_level(server, view, sv_volume):
    from neu_vol.source_metadata import level_spec

    lvl1 = open_backend(level_spec(view, "mapped", 1))
    src1 = open_backend(level_spec(sv_volume, "zarr3", 1))
    full = tuple(slice(0, s) for s in lvl1.shape)
    np.testing.assert_array_equal(lvl1.read_region(full), _mapped(src1.read_region(full)))


def test_converting_a_view_writes_the_mapped_labels_and_records_both_halves(
        server, view, tmp_path):
    from neu_vol import convert

    dst = str(tmp_path / "view70.zarr")
    convert(view, dst, profile="local", chunk=(4, 4, 4), multiscale=False,
            delete_existing=True)
    lvl0 = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(lvl0.read_region(tuple(slice(0, s) for s in lvl0.shape)),
                                  _zyx(_mapped(server.xyz)))
    assert lvl0.dtype == np.uint32              # values fit, so the source's dtype is kept
    with open(os.path.join(dst, "provenance.json")) as fh:
        rec = json.load(fh)["source"]
    assert rec["source"] == "mapped" and rec["mapping"] == "agglomerate_view_70"
    assert rec["origin"]["dataset_id"] == DSID and rec["of"]["source"] == "zarr3"


def test_a_volume_holding_ids_past_the_table_raises(server, view):
    from neu_vol import mapped

    path = mapped.read_descriptor(view)["table"]
    np.save(path, np.zeros(5, np.uint64))       # far too short for this volume
    mapped.clear_cache()
    clear_backend_cache()
    # Built the way production builds it (invariant SPEC-FORMS): through the metadata.
    be = open_backend(sm.read_source_metadata(sm.location_spec(view, "mapped"))["data_spec"])
    with pytest.raises(ValueError, match="different segmentation"):
        be.read_region(tuple(slice(0, s) for s in be.shape))


def test_concurrent_fetches_produce_the_same_tables_and_index(server, sv_volume, tmp_path):
    from neu_vol.ops.wk_mappings import fetch_mapping_tables

    serial = fetch_mapping_tables("wk://specimen/segmentation", str(tmp_path / "a"),
                                  volume=sv_volume, threads=1)
    threaded = fetch_mapping_tables("wk://specimen/segmentation", str(tmp_path / "b"),
                                    volume=sv_volume, threads=2)
    assert set(threaded["views"]) == set(serial["views"])
    for name in serial["views"]:
        np.testing.assert_array_equal(np.load(tmp_path / "a" / f"{name}.npy"),
                                      np.load(tmp_path / "b" / f"{name}.npy"))
        assert (tmp_path / "b" / f"{name}.json").is_file()


def test_read_scales_answers_for_a_view_with_its_sources_levels(server, view, sv_volume):
    """Callers that size or place a volume (coverage checks, crops) go through
    read_scales, and a view has exactly its source's geometry."""
    from neu_vol import read_scales

    got, want = read_scales(view), read_scales(sv_volume)
    assert [(s.shape, s.frame) for s in got] == [(s.shape, s.frame) for s in want]
