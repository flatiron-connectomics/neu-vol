"""Read-only backend over a DVID ``labelmap`` instance.

DVID is not a kvstore, so this backend is addressed by ``server``/``uuid``/``instance``
rather than by a path — see :func:`parse_url` for the URL form and
``source_metadata.detect_backend`` for where it is recognised.

**Blocks that hold no labels are omitted from DVID's response**, which is why there is
no occupancy prefilter anywhere in this package for DVID sources: an empty region costs
a round trip and no bandwidth (measured at the 512^3 task size: 0.03 s and zero bytes,
against 0.4-0.6 s and ~12 MB for a full one). Filtering them out would save the cheap
tasks and leave the expensive ones untouched. See NOTES-TODO for the measurements.

Reads go through ``fetch_labelmap_voxels``, which requests ``?compression=blocks`` —
DVID's *stored* form, so the server never decompresses. Inflation happens here, in this
process, via libdvid; under dask that is the worker, so it is already spread across the
fleet. It is also the dominant per-task cost (~1.4 s of a 1.9 s 512^3 read), which means
throughput is bounded by DVID's serving rate rather than by our CPU. DVID is typically a
**shared** service and answers overload with 503 — see ``retry.TRANSIENT_MARKERS``.

Read-only: this package converts a labelmap into a neuroglancer volume; it never writes
segmentation back to DVID, which is the proofreading tool's job.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "dvid"

#: Accepted URL schemes -> the scheme handed to neuclease. A bare host gets ``http://``
#: prepended by neuclease's own ``dvid_api_wrapper``, so ``dvid://`` needs no prefix;
#: ``dvid+https://`` exists because that wrapper cannot be told to use TLS otherwise.
_SCHEMES = {"dvid+https://": "https://", "dvid://": ""}

_MISSING = (
    "Reading a DVID labelmap needs the `neuclease` package, which is not installed "
    "here.\n\n"
    "neuclease cannot be a pip dependency: it needs `libdvid-cpp`, `vigra` and "
    "`dvidutils`, which are conda-only on flyem-forge — and libdvid is load-bearing "
    "rather than incidental, since it is what inflates DVID's compressed label "
    "blocks. Install it into the environment instead:\n"
    "    mamba install -n em-lib -c flyem-forge -c conda-forge neuclease"
)


def parse_url(url: str) -> dict[str, str]:
    """``dvid://<server>/<uuid>/<instance>`` -> the spec fields.

    Three path segments, always, because all three are required to address an
    instance and none of them may contain a ``/``:

    - ``server`` may carry a port (``emdata3:8900``) and may be a bare host
      (``dvid.example.org``), in which case neuclease prepends
      ``http://``. Use the ``dvid+https://`` scheme for a TLS server.
    - ``uuid`` may be a bare uuid (``93fdbc``), an abbreviated one, or the
      ``uuid:branch`` form DVID accepts (``93fdbc:main``).
    - ``instance`` is the labelmap instance name (``labels``, ``segmentation``).

    Both colons above are why this splits on ``/`` only: a port and a branch
    qualifier both contain ``:``, and neither contains a slash.
    """
    for scheme, prefix in _SCHEMES.items():
        if url.startswith(scheme):
            rest, server_prefix = url[len(scheme):], prefix
            break
    else:
        raise ValueError(
            f"not a DVID URL: {url!r} (expected dvid://server/uuid/instance)")

    parts = [p for p in rest.split("/") if p != ""]
    if len(parts) != 3:
        raise ValueError(
            f"malformed DVID URL {url!r}: expected exactly three segments after the "
            f"scheme — server, uuid and instance, as in "
            f"dvid://dvid.example.org/93fdbc:main/labels — but got "
            f"{len(parts)} ({parts!r}). A port on the server and a ':branch' on the "
            f"uuid are fine; neither adds a segment.")
    server, uuid, instance = parts
    return {"server": server_prefix + server, "uuid": uuid, "instance": instance}


def is_url(location: Any) -> bool:
    """True if this location names a DVID instance rather than a store."""
    return isinstance(location, str) and location.startswith(tuple(_SCHEMES))


def spec_url(spec: Mapping[str, Any]) -> str:
    """The ``dvid://`` URL for a spec — for messages and provenance."""
    server = str(spec["server"])
    scheme = "dvid+https://" if server.startswith("https://") else "dvid://"
    bare = server.split("://", 1)[-1]
    return f"{scheme}{bare}/{spec['uuid']}/{spec['instance']}"


#: How far back to walk looking for a locked node. In a nightly lock-and-spawn repo the
#: answer is 1 (measured on dvid.example.org: 448 nodes on `main`, 447 locked, the single open
#: one being HEAD), so a bound this size only ever matters if something is wrong.
_MAX_LOCK_WALK = 100


#: Resolutions already made, keyed by (server, ref, prefer_locked). A ref costs two
#: HTTP requests to resolve (resolve_ref + fetch_commit) and several call sites ask
#: independently — the CLI's crop helpers peek at source metadata before `convert` runs.
#: Caching also keeps the "this node is OPEN" warning to one line instead of one per
#: caller. Per process, like every other cache here.
_NODES: dict[tuple[str, str, bool], dict] = {}


def clear_node_cache() -> None:
    """Drop memoized ref resolutions (tests, and any long-lived process)."""
    _NODES.clear()


def resolve_node(spec: Mapping[str, Any], *, prefer_locked: bool = False) -> dict:
    """Resolve this spec's ``uuid`` **ref** to a concrete node, and say if it is locked.

    ``uuid`` may be a branch ref (``93fdbc:main``) rather than a node id, and a branch
    ref resolves to the branch HEAD — which in a lock-and-spawn workflow is the **open**
    node. Two consequences, and the first is a correctness bug rather than a nicety:

    - **The ref must be resolved once, in the driver, not per worker.** Workers reopen
      the backend from ``data_spec``; if that carried the ref, a lock-and-spawn landing
      mid-run would move HEAD and later workers would read a *different node* than
      earlier ones, producing a volume stitched from two versions with nothing to show
      it. So ``read_source_metadata`` resolves and puts the concrete uuid in
      ``data_spec`` — the same discipline as invariant 9.
    - **An open node is mutable**, so a pull from one is not reproducible even if
      nothing goes wrong. That is worth saying out loud, which the caller does.

    ``prefer_locked`` walks back to the newest locked node — DVID's own ``ref~N``
    ancestor syntax, which needs the ``repo:branch`` form; a bare uuid cannot be walked
    from on a multi-repo server, and is already a pinned node anyway.
    """
    try:
        from neuclease.dvid import fetch_commit, resolve_ref
    except ImportError as exc:
        raise ImportError(_MISSING) from exc

    server, ref, _instance = _address(spec)
    cached = _NODES.get((server, ref, bool(prefer_locked)))
    if cached is not None:
        return dict(cached)

    def _remember(node: dict) -> dict:
        _NODES[(server, ref, bool(prefer_locked))] = dict(node)
        return node

    uuid = resolve_ref(server, ref, expand=True)
    locked = bool(fetch_commit(server, uuid))
    if locked or not prefer_locked:
        return _remember({"ref": ref, "uuid": uuid, "locked": locked, "walked": 0})

    if ":" not in ref:
        raise ValueError(
            f"cannot find a locked node from {ref!r}: walking ancestors uses DVID's "
            f"'ref~N' syntax, which needs the repo:branch form (e.g. 93fdbc:main). A "
            f"bare uuid already names one node — drop the locked-node request, or give "
            f"the branch it is on.")
    for n in range(1, _MAX_LOCK_WALK + 1):
        candidate = resolve_ref(server, f"{ref}~{n}", expand=True)
        if fetch_commit(server, candidate):
            return _remember(
                {"ref": ref, "uuid": candidate, "locked": True, "walked": n})
    raise ValueError(
        f"no locked node within {_MAX_LOCK_WALK} ancestors of {ref!r}; every one of "
        f"them is still open, which is not what a lock-and-spawn repo looks like")


#: Node fields worth keeping from the repo DAG. `Log` is deliberately omitted — it is
#: the full mutation log and can be enormous.
_NODE_FIELDS = ("Branch", "Note", "VersionID", "Created", "Updated")


def node_summary(spec: Mapping[str, Any]) -> dict:
    """Both candidate nodes for a ref: what it points at now, and the newest locked one.

    ``em-vol info`` shows both because they are different answers to "which version
    would I get", and the choice between them is the choice between a reproducible
    export and a convenient one. Returns ``{"head": node, "locked": node | None,
    "locked_error": str | None}``; ``locked`` is the same object as ``head`` when the
    ref already points at a locked node, and ``None`` when no locked node is reachable
    (a bare uuid on a multi-repo server cannot be walked back from).
    """
    head = resolve_node(spec)
    if head["locked"]:
        return {"head": head, "locked": head, "locked_error": None}
    try:
        return {"head": head, "locked": resolve_node(spec, prefer_locked=True),
                "locked_error": None}
    except Exception as exc:                                   # noqa: BLE001
        return {"head": head, "locked": None,
                "locked_error": f"{type(exc).__name__}: {exc}"}


def provenance(spec: Mapping[str, Any], node: Mapping[str, Any]) -> dict:
    """What this pull came from, for the record written beside the output.

    The resolved ``uuid`` is the load-bearing field: a branch ref means something
    different tomorrow, and the whole point of pulling successive proofreading
    snapshots is being able to say which one you have. ``requested`` keeps the ref that
    was actually typed, since "the newest locked node on main" and "this node" are
    different provenance claims even when they name the same uuid today.

    There is **no per-instance mutation id** to record: DVID's mutation id and lastmod
    endpoints are both per-*body*. What characterises a snapshot instead is the node's
    own DAG metadata plus ``maxlabel``, which advances as proofreading creates bodies.
    """
    server, ref, instance = _address(spec)
    out = {
        "source": "dvid",
        # Pinned to the resolved node, so this URL is reproducible as written.
        "url": spec_url({**dict(spec), "uuid": node["uuid"]}),
        "server": server,
        "requested": node.get("ref", ref),
        "uuid": node["uuid"],
        "locked": node["locked"],
        "ancestors_walked": node.get("walked", 0),
        "instance": instance,
        "supervoxels": bool(spec.get("supervoxels", False)),
    }
    # Both of these are nice to have and neither is worth failing a completed run over,
    # so each degrades to a recorded error rather than an exception.
    try:
        from neuclease.dvid import fetch_repo_info

        info = fetch_repo_info(server, node["uuid"])
        dag_node = info["DAG"]["Nodes"][node["uuid"]]
        out["node"] = {k: dag_node.get(k) for k in _NODE_FIELDS}
    except Exception as exc:                                   # noqa: BLE001
        out["node_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        from neuclease.dvid import fetch_maxlabel

        out["maxlabel"] = int(fetch_maxlabel(server, node["uuid"], instance))
    except Exception as exc:                                   # noqa: BLE001
        out["maxlabel_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def _address(spec: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        return str(spec["server"]), str(spec["uuid"]), str(spec["instance"])
    except KeyError as e:
        raise ValueError(
            f"DVID spec is missing {e}; it needs 'server', 'uuid' and 'instance'") from e


def instance_info(spec: Mapping[str, Any]) -> dict:
    """DVID's ``/info`` for this instance. Separate so metadata readers need no backend.

    ``read_source_metadata`` calls this rather than opening a backend, keeping the
    "what is this volume" path one HTTP request rather than a full open.
    """
    try:
        from neuclease.dvid import fetch_instance_info
    except ImportError as exc:
        raise ImportError(_MISSING) from exc

    return fetch_instance_info(*_address(spec))


def geometry(info: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    """Shape / chunks / voxel size / max scale, all zyx, from DVID's ``/info``.

    The volume is presented **origin-anchored**: shape is ``MaxPoint + 1``, so a voxel
    index means the same thing here as it does in DVID and in any other view of the
    same dataset. The margin below ``MinPoint`` is empty, which costs one 30 ms round
    trip per task and nothing on disk — the alternative, translating by ``MinPoint``,
    would buy a slightly smaller volume at the price of every coordinate disagreeing
    with DVID's own.
    """
    ext = info.get("Extended", {})
    type_name = info.get("Base", {}).get("TypeName")
    if type_name != "labelmap":
        raise ValueError(
            f"{spec_url(spec)} is a {type_name!r} instance; this backend reads "
            f"'labelmap' only")

    lo_xyz, hi_xyz = ext.get("MinPoint"), ext.get("MaxPoint")
    if lo_xyz is None or hi_xyz is None:
        raise ValueError(
            f"{spec_url(spec)} reports no extents (MinPoint/MaxPoint are null), which "
            f"DVID does for an instance that has been created but never written to. "
            f"There is nothing to read.")
    lo = tuple(int(v) for v in lo_xyz[::-1])
    if any(v < 0 for v in lo):
        raise ValueError(
            f"{spec_url(spec)} has a negative MinPoint {lo_xyz!r}. This backend "
            f"presents DVID's own coordinates origin-anchored, which would clip data "
            f"at negative indices. Support for a translated frame is not implemented.")

    block = tuple(int(v) for v in ext["BlockSize"][::-1])
    voxel = tuple(float(v) for v in ext["VoxelSize"][::-1])
    return {
        "shape0": tuple(int(v) + 1 for v in hi_xyz[::-1]),
        "min_point": lo,
        "chunks": block,
        "voxel_size": voxel,
        "max_scale": int(ext.get("MaxDownresLevel", 0)),
        "dtype": np.dtype("uint64"),
    }


def level_shape(shape0, scale: int) -> tuple[int, ...]:
    """Shape at a DVID downres level.

    DVID halves every axis per level, so unlike a general pyramid this factor really is
    ``2**scale`` — it is DVID's documented model rather than the assumption CLAUDE.md
    invariant 1 warns against, and it must not be copied to any other backend.
    """
    return tuple(-(-int(s) // 2 ** int(scale)) for s in shape0)


class DVIDBackend:
    """Read-only ``(z, y, x)`` view over one scale of a DVID labelmap instance."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._server, self._uuid, self._instance = _address(spec)
        self._scale = int(spec.get("scale_index", 0))
        self._supervoxels = bool(spec.get("supervoxels", False))

        geom = geometry(instance_info(spec), spec)
        if self._scale > geom["max_scale"]:
            raise ValueError(
                f"{spec_url(spec)} has scales 0..{geom['max_scale']} "
                f"(MaxDownresLevel), but scale {self._scale} was requested")
        self._shape = level_shape(geom["shape0"], self._scale)
        self._chunks = geom["chunks"]
        self._dtype = geom["dtype"]
        self.voxel_size = tuple(v * 2 ** self._scale for v in geom["voxel_size"])

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "DVIDBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    def read_region(self, region: Region) -> np.ndarray:
        try:
            from neuclease.dvid import fetch_labelmap_voxels
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        box = np.array([[int(s.start) for s in region],
                        [int(s.stop) for s in region]], dtype=np.int64)
        # fetch_labelmap_voxels rounds the request out to DVID's 64-voxel block grid and
        # truncates the result back, so an unaligned region is correct but wasteful. Our
        # task shapes are whole multiples of `chunks`, so in practice no rounding occurs.
        out = fetch_labelmap_voxels(
            self._server, self._uuid, self._instance, box,
            scale=self._scale, supervoxels=self._supervoxels)
        return np.ascontiguousarray(out)

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError(
            "DVID labelmap instances are read-only here. Proofreading edits belong in "
            "the proofreading tool; this package only pulls snapshots out.")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, DVIDBackend.open)
