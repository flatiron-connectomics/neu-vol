"""Addressing a webknossos layer, its credentials, and the few REST calls behind it.

A webknossos dataset is read over HTTP, the way a DVID instance is: there is no store to
open, so it is addressed by URL — ``wk://DATASET/LAYER`` — and the server it lives on and
the token that opens it come from a **credentials file**, never from the URL or a spec.
The array view is ``backends/webknossos.py``; this module is everything that is not the
array, so a caller wanting the mapping list or the dataset record needs no backend.

**Stdlib only.** The ``webknossos`` package is the official client, but it pins older
``click``/``httpx``/``networkx``/``rich``/``psutil`` than this environment holds, and the
part of it needed here is four GET routes. ``urllib`` does those without touching the env.
The routes and API versions below are the ones ``webknossos`` 4.2.2 itself uses.

**Supervoxels are the default, and that is a deliberate difference from the viewer.** A
segmentation layer stores an over-segmentation; neurons exist only once an *agglomerate
mapping* is applied, and the layer usually names one as its default view. ``webknossos``'
own ``download()`` never applies it, so a download is supervoxels whatever the viewer
showed. Here the mapping is a spec field (``mapping``) applied **by the server on every
read**, and leaving it unset reads supervoxels — the right input for evaluating
agglomeration views against each other, since each view is then a relabelling of one
volume. ``convert`` says so whenever a layer has a default view that is not being applied.

Credentials, first hit wins:

1. ``$WK_TOKEN`` (+ ``$WK_URL``) — the variables the ``webknossos`` package reads, so an
   existing setup carries over.
2. ``$NEU_WK_CREDENTIALS``, else ``~/.config/webknossos/credentials.toml``:

   .. code-block:: toml

      # chmod 600 — this is a secret
      url = "https://webknossos.org"
      token = "..."              # https://webknossos.org/account/token
      organization = "..."       # optional: narrows dataset-name lookup

A file in the home directory rather than in a repo or the workspace, for the same reason as
``~/.aws/credentials`` (see ``aws.py``): it is a secret, the repos are public, and home is
shared with SLURM workers, which each read it themselves — the token is never put into a
spec, a manifest, a job script or ``provenance.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np

log = logging.getLogger(__name__)

SCHEME = "wk://"

#: The REST API versions spoken here — the ones `webknossos` 4.2.2 uses (its latest
#: `WkApiClient` / `DatastoreApiClient`). Checked against the server's supported range
#: once per process, so an incompatible server fails with a message rather than a 404.
WK_API_VERSION = 16
DATASTORE_API_VERSION = 15

ENV_CREDENTIALS = "NEU_WK_CREDENTIALS"
DEFAULT_CREDENTIALS = "~/.config/webknossos/credentials.toml"

#: A socket timeout only. Data requests are capped at 128^3 and a slow one is retried
#: long before this (``backends/webknossos.SLOW_REQUEST_S``); this bounds the table POSTs.
TIMEOUT_S = 600

_DATASET_ID = re.compile(r"^[0-9a-f]{24}$")

#: webknossos `elementClass` -> numpy dtype. `uint24` (packed RGB) is deliberately absent.
ELEMENT_CLASSES = {
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "float": "float32", "double": "float64",
}


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #
def is_url(location: Any) -> bool:
    """True if this location names a webknossos layer rather than a store."""
    return isinstance(location, str) and location.startswith(SCHEME)


def parse_url(url: str) -> dict[str, str]:
    """``wk://DATASET/LAYER`` -> ``{"dataset", "layer"}``.

    ``DATASET`` is a dataset name or its 24-hex id. The server is not part of the URL: it
    comes from the credentials, which is where its token is too.
    """
    if not is_url(url):
        raise ValueError(f"not a webknossos URL: {url!r} (expected wk://DATASET/LAYER)")
    parts = [p for p in url[len(SCHEME):].split("/") if p]
    if len(parts) != 2:
        raise ValueError(
            f"malformed webknossos URL {url!r}: expected exactly two segments after "
            f"{SCHEME} — the dataset (name or id) and the layer, as in "
            f"wk://my_dataset/segmentation — but got {len(parts)} ({parts!r}). The "
            f"server comes from the credentials file, not the URL.")
    return {"dataset": parts[0], "layer": parts[1]}


def spec_url(spec: Mapping[str, Any]) -> str:
    """The ``wk://`` URL for a spec — for messages."""
    return f"{SCHEME}{spec.get('dataset')}/{spec.get('layer')}"


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def credentials_path() -> Path:
    return Path(os.environ.get(ENV_CREDENTIALS) or DEFAULT_CREDENTIALS).expanduser()


def credentials() -> dict[str, str | None]:
    """``{"url", "token", "organization"}``, from the environment or the credentials file.

    Raises with the file's expected shape when nothing is configured: the first run on a
    new machine is exactly when that is the useful answer.
    """
    if os.environ.get("WK_TOKEN"):
        return {"url": os.environ.get("WK_URL", "https://webknossos.org").rstrip("/"),
                "token": os.environ["WK_TOKEN"], "organization": None}
    path = credentials_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"no webknossos credentials: $WK_TOKEN is unset and {path} does not exist. "
            f"Create it (chmod 600), with your token from <server>/account/token:\n\n"
            f'    url = "https://webknossos.org"\n'
            f'    token = "..."\n'
            f'    organization = "..."   # optional\n')
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by others (mode %o); it holds a token — chmod 600 it",
                    path, stat.S_IMODE(mode))
    import tomllib

    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if not data.get("token"):
        raise ValueError(f"{path} sets no `token`")
    return {"url": str(data.get("url") or "https://webknossos.org").rstrip("/"),
            "token": str(data["token"]), "organization": data.get("organization")}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _request(url: str, *, token: str | None, query: Mapping[str, Any] | None = None,
             timeout: float = TIMEOUT_S, data: bytes | None = None,
             content_type: str | None = None) -> tuple[bytes, Mapping[str, str]]:
    """GET ``url`` (POST ``data`` when given); ``(body, headers)``.

    Failures are re-raised with the status vocabulary ``neu_vol.retry`` classifies on, so a
    block read retries a 503 or a reset connection and fails fast on a 403 or a 404.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    if query:
        url = f"{url}?{urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})}"
    # gzip is a 10x, not a nicety: label data compresses ~20x, and without it a read is
    # bound by the wire. Measured on a 256^3 uint32 box: 67 MB in 21.9 s plain, 3.7 MB in
    # 2.0 s gzipped — and the server-side mapping cost nothing either way. urllib does not
    # ask for it on its own (httpx, which `webknossos` uses, does).
    headers = {"Accept-Encoding": "gzip"}
    if token:
        headers["X-Auth-Token"] = token
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, headers=headers, data=data)
    # The token must never reach a message: errors below name the route, not the request.
    route = url.split("?", 1)[0]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body, got = resp.read(), dict(resp.headers)
        if (got.get("Content-Encoding") or got.get("content-encoding")) == "gzip":
            import gzip

            body = gzip.decompress(body)
        return body, got
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        status = ("PERMISSION_DENIED" if e.code in (401, 403) else
                  "NOT_FOUND" if e.code == 404 else
                  "UNAVAILABLE" if e.code == 429 or e.code >= 500 else
                  "INVALID_ARGUMENT")
        raise RuntimeError(f"{status}: webknossos HTTP {e.code} from {route}: {body}") \
            from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RuntimeError(f"UNAVAILABLE: webknossos request to {route} failed: {e}") \
            from None


def _get_json(url: str, *, token: str | None, query: Mapping[str, Any] | None = None) -> Any:
    body, _ = _request(url, token=token, query=query)
    return json.loads(body)


#: Checked once per server per process.
_VERSION_OK: set[str] = set()


def _check_api_version(url: str) -> None:
    if url in _VERSION_OK:
        return
    info = _get_json(f"{url}/api/buildinfo", token=None)
    v = info.get("httpApiVersioning") or {}
    lo, hi = v.get("oldestSupportedApiVersion"), v.get("currentApiVersion")
    if isinstance(lo, int) and isinstance(hi, int) and not lo <= WK_API_VERSION <= hi:
        raise RuntimeError(
            f"FAILED_PRECONDITION: {url} supports API versions {lo}..{hi}; this module "
            f"speaks {WK_API_VERSION}. Compare with the routes in the current `webknossos` "
            f"package's client/api_client and update the version constants here.")
    _VERSION_OK.add(url)


def api(url: str) -> str:
    return f"{url}/api/v{WK_API_VERSION}"


def data_api(datastore: str) -> str:
    return f"{datastore.rstrip('/')}/data/v{DATASTORE_API_VERSION}"


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def resolve_dataset_id(dataset: str, *, creds: Mapping[str, Any] | None = None) -> str:
    """A dataset name (or id) -> its id. Ambiguity raises rather than picking one.

    Names are not unique across organizations, and ``webknossos`` itself resolves an
    ambiguous name by taking the first and warning — which on a re-run could be a
    different dataset. An id is what goes into ``data_spec`` and the provenance.
    """
    creds = creds or credentials()
    if _DATASET_ID.match(dataset):
        return dataset
    _check_api_version(creds["url"])
    found = _get_json(f"{api(creds['url'])}/datasets", token=creds["token"],
                      query={"searchQuery": dataset, "isActive": "true",
                             "organizationId": creds.get("organization")})
    exact = [d for d in found
             if dataset in (d.get("name"), d.get("directoryName"))]
    if len(exact) == 1:
        return str(exact[0]["id"])
    if not exact:
        near = ", ".join(sorted({str(d.get("name")) for d in found})[:8]) or "none"
        raise ValueError(
            f"NOT_FOUND: no dataset named {dataset!r} visible to this token on "
            f"{creds['url']}. Close matches: {near}")
    listed = "; ".join(f"{d['id']} ({d.get('owningOrganization')})" for d in exact)
    raise ValueError(
        f"dataset name {dataset!r} is ambiguous on {creds['url']}: {listed}. Use the id "
        f"in the URL (wk://<id>/<layer>) or set `organization` in the credentials file.")


#: Dataset records, per (server, id), per process. One request answers every level, so
#: a worker opening eight levels' backends asks once.
_DATASETS: dict[tuple[str, str], dict] = {}


def clear_cache() -> None:
    """Drop memoized dataset records and version checks (tests, long-lived processes)."""
    _DATASETS.clear()
    _VERSION_OK.clear()


def dataset_info(dataset_id: str, *, creds: Mapping[str, Any] | None = None) -> dict:
    """The server's record for one dataset (``GET /datasets/{id}``)."""
    creds = creds or credentials()
    key = (creds["url"], dataset_id)
    if key not in _DATASETS:
        _check_api_version(creds["url"])
        rec = _get_json(f"{api(creds['url'])}/datasets/{dataset_id}", token=creds["token"])
        if (rec.get("dataSource") or {}).get("status") and \
                not (rec.get("dataSource") or {}).get("dataLayers"):
            raise RuntimeError(
                f"FAILED_PRECONDITION: dataset {dataset_id} is unusable: "
                f"{rec['dataSource']['status']}")
        _DATASETS[key] = rec
    return _DATASETS[key]


def layer_record(info: Mapping[str, Any], layer: str) -> dict:
    layers = (info.get("dataSource") or {}).get("dataLayers") or []
    for rec in layers:
        if rec.get("name") == layer:
            return rec
    raise ValueError(
        f"NOT_FOUND: dataset {info.get('name')!r} has no layer {layer!r}; it has "
        f"{', '.join(repr(r.get('name')) for r in layers) or 'none'}")


def default_mapping(layer: Mapping[str, Any]) -> str | None:
    """The mapping the viewer applies to this layer by default, if any."""
    view = layer.get("defaultViewConfiguration") or {}
    return (view.get("mapping") or {}).get("name")


def mappings(spec: Mapping[str, Any], *, creds: Mapping[str, Any] | None = None) -> dict:
    """``{"agglomerate": [...], "json": [...]}`` — every mapping the layer can be read with.

    Both kinds are accepted by the data route's ``mappingName``; agglomerate files are what
    a vendor delivers as views at several thresholds, JSON mappings are uploaded tables.
    """
    creds = creds or credentials()
    info = dataset_info(spec["dataset_id"], creds=creds)
    base = (f"{data_api(info['dataStore']['url'])}/datasets/{spec['dataset_id']}"
            f"/layers/{spec['layer']}")
    out = {}
    for kind, route in (("agglomerate", "agglomerates"), ("json", "mappings")):
        try:
            names = _get_json(f"{base}/{route}", token=creds["token"])
        except RuntimeError as e:
            if "NOT_FOUND" not in str(e):
                raise
            names = []
        out[kind] = sorted(names, key=_natural)
    return out


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


# --------------------------------------------------------------------------- #
# listing
# --------------------------------------------------------------------------- #
def list_datasets(pattern: str | None = None, *, organization: str | None = None,
                  creds: Mapping[str, Any] | None = None) -> list[dict]:
    """The active datasets this token can see, oldest first — ONE request.

    The server's list includes every public dataset on it, other organizations' too
    (webknossos.org answers with hundreds), so ``organization`` filters to one. Its records
    carry the layer names, dates and size but no description; `dataset_summary` adds that
    at one request per dataset.

    ``pattern`` matches the name case-insensitively: a glob if it has ``*?[``, else a
    substring.
    """
    import fnmatch

    creds = creds or credentials()
    _check_api_version(creds["url"])
    found = _get_json(f"{api(creds['url'])}/datasets", token=creds["token"],
                      query={"isActive": "true", "organizationId": organization})
    if organization:
        found = [d for d in found if d.get("owningOrganization") == organization]
    if pattern:
        p = pattern.lower()
        glob = any(c in p for c in "*?[")
        found = [d for d in found
                 if (fnmatch.fnmatch(str(d.get("name", "")).lower(), p) if glob
                     else p in str(d.get("name", "")).lower())]
    return sorted(found, key=lambda d: (d.get("created") or 0, str(d.get("name"))))


def iso_date(ms: Any) -> str:
    """A webknossos timestamp (ms since the epoch) as a UTC date; ``-`` if absent."""
    import datetime as dt

    if not isinstance(ms, (int, float)) or ms <= 0:
        return "-"
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def dataset_summary(dataset_id: str, *, creds: Mapping[str, Any] | None = None) -> dict:
    """What a dataset is, for choosing one: its record's description, uploader and dates,
    and per layer the box, mags, default view and every mapping it can be read with.

    One request for the record plus two per segmentation layer (the mapping lists).
    """
    creds = creds or credentials()
    info = dataset_info(dataset_id, creds=creds)
    source = info.get("dataSource") or {}
    layers = []
    for rec in source.get("dataLayers") or []:
        box = rec.get("boundingBox") or {}
        layer = {
            "name": rec.get("name"), "category": rec.get("category"),
            "dtype": rec.get("elementClass"),
            "top_left_xyz": box.get("topLeft"),
            "size_xyz": [box.get("width"), box.get("height"), box.get("depth")],
            "mags": [m.get("mag") for m in rec.get("mags") or []],
            "default_mapping": default_mapping(rec),
        }
        if rec.get("category") == "segmentation":
            layer["mappings"] = mappings({"dataset_id": dataset_id, "layer": rec["name"]},
                                         creds=creds)
        layers.append(layer)
    return {
        "id": info.get("id", dataset_id), "name": info.get("name"),
        "organization": info.get("owningOrganization"),
        "description": info.get("description"), "uploader": info.get("uploaderFullName"),
        "created": iso_date(info.get("created")),
        "last_used": iso_date(info.get("lastUsedByUser")),
        "tags": info.get("tags") or [], "bytes": info.get("usedStorageBytes"),
        "voxel_size": (source.get("scale") or {}).get("factor"),
        "voxel_unit": (source.get("scale") or {}).get("unit"),
        "layers": layers,
    }


def layer_base(spec: Mapping[str, Any]) -> str:
    """The datastore URL of a resolved spec's layer; every per-layer route hangs off it."""
    info = dataset_info(spec["dataset_id"])
    return (f"{data_api(info['dataStore']['url'])}/datasets/{spec['dataset_id']}"
            f"/layers/{spec['layer']}")


# --------------------------------------------------------------------------- #
# agglomerate tables
# --------------------------------------------------------------------------- #
# The datastore's `agglomeratesForSegments` takes and returns a protobuf `ListOfLong`:
#     syntax = "proto2"; message ListOfLong { repeated uint64 items = 1; }
# proto2 `repeated` is UNPACKED by default — each item is its own field, tag byte 0x08
# then a varint — but a parser must accept the packed form too (tag 0x0A, a length,
# then bare varints). Encoded here with numpy rather than generated code: one message
# type, and a few million items, which a per-item Python loop would make slow.
_MAX_VARINT = 10


def encode_list_of_long(ids: np.ndarray) -> bytes:
    """``ListOfLong`` bytes for these ids, unpacked (what the server's own encoder emits)."""
    v = np.asarray(ids, dtype=np.uint64).ravel()
    if v.size == 0:
        return b""
    shifts = (np.arange(_MAX_VARINT, dtype=np.uint64) * np.uint64(7))
    groups = ((v[:, None] >> shifts[None, :]) & np.uint64(0x7F)).astype(np.uint8)
    # Bytes needed per value: up to and including its highest non-zero 7-bit group.
    nonzero = groups != 0
    n = np.where(nonzero.any(axis=1),
                 _MAX_VARINT - np.argmax(nonzero[:, ::-1], axis=1), 1)
    used = np.arange(_MAX_VARINT)[None, :] < n[:, None]
    more = np.arange(_MAX_VARINT)[None, :] < (n - 1)[:, None]
    groups |= (more.astype(np.uint8) << 7)
    out = np.concatenate([np.full((v.size, 1), 0x08, np.uint8), groups], axis=1)
    keep = np.concatenate([np.ones((v.size, 1), bool), used], axis=1)
    return out[keep].tobytes()


def _decode_varints(buf: np.ndarray) -> np.ndarray:
    """Every varint in ``buf``, which must hold nothing else."""
    if buf.size == 0:
        return np.zeros(0, np.uint64)
    ends = np.flatnonzero((buf & 0x80) == 0)
    if ends.size == 0 or ends[-1] != buf.size - 1:
        raise ValueError("truncated varint in a ListOfLong")
    starts = np.concatenate([[0], ends[:-1] + 1])
    group = np.repeat(np.arange(ends.size), ends - starts + 1)
    pos = np.arange(buf.size) - starts[group]
    parts = (buf & 0x7F).astype(np.uint64) << (pos.astype(np.uint64) * np.uint64(7))
    return np.add.reduceat(parts, starts)


def decode_list_of_long(data: bytes) -> np.ndarray:
    """The ``items`` of a ``ListOfLong``, packed or unpacked, as uint64."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return np.zeros(0, np.uint64)
    # The form is decided by the first byte, because it cannot be decided per byte: in an
    # unpacked stream the value 10 is the lone byte 0x0A, which is also the packed tag.
    # An encoder uses one form for a whole field, so the first tag answers for all of it.
    if buf[0] == 0x08:                              # unpacked: tag + varint, repeated
        vals = _decode_varints(buf)
        if vals.size % 2 or not (vals[::2] == 8).all():
            raise ValueError("unexpected field in an unpacked ListOfLong")
        return vals[1::2]
    if buf[0] != 0x0A:
        raise ValueError(f"unexpected tag byte {buf[0]:#x} in a ListOfLong")
    out, i = [], 0
    while i < buf.size:                             # packed: tag, length, varints
        if buf[i] != 0x0A:
            raise ValueError(f"unexpected tag byte {buf[i]:#x} in a packed ListOfLong")
        j = i + 1
        while buf[j] & 0x80:
            j += 1
        length = int(_decode_varints(buf[i + 1:j + 1])[0])
        out.append(_decode_varints(buf[j + 1:j + 1 + length]))
        i = j + 1 + length
    return np.concatenate(out)


#: Ids per `agglomeratesForSegments` request: ~0.5 MB of body, comfortably under any
#: proxy limit, and few enough requests that a 2M-supervoxel table is a couple of dozen.
TABLE_BATCH = 100_000


def agglomerate_table(spec: Mapping[str, Any], mapping: str, n_ids: int, *,
                      batch: int = TABLE_BATCH) -> np.ndarray:
    """``table[sv] = agglomerate id`` for every supervoxel id ``0 .. n_ids - 1``.

    Dense, indexed by supervoxel id, so applying it is ``table[volume]``. Id 0 is
    background and is not asked about; it maps to 0.
    """
    creds = credentials()
    url = f"{layer_base(spec)}/agglomerates/{mapping}/agglomeratesForSegments"
    table = np.zeros(int(n_ids), dtype=np.uint64)
    for lo in range(1, int(n_ids), batch):
        ids = np.arange(lo, min(lo + batch, int(n_ids)), dtype=np.uint64)
        body, _ = _request(url, token=creds["token"], data=encode_list_of_long(ids),
                           content_type="application/x-protobuf")
        got = decode_list_of_long(body)
        if got.size != ids.size:
            raise RuntimeError(f"UNAVAILABLE: {mapping}: asked about {ids.size} "
                               f"supervoxels, got {got.size} answers")
        table[lo:lo + ids.size] = got
    return table


def check_mapping(spec: Mapping[str, Any], mapping: str) -> None:
    """Refuse a mapping the layer does not have, naming the ones it does.

    Checked up front because the alternative is a 400 from every task of a run — or, if a
    server ever ignored an unknown name, a volume of supervoxels labelled as a view.
    """
    have = mappings(spec)
    if mapping not in have["agglomerate"] + have["json"]:
        listed = ", ".join(have["agglomerate"] + have["json"]) or "none"
        raise ValueError(
            f"{spec_url(spec)} has no mapping {mapping!r}. Available: {listed}")
