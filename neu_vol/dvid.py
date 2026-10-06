"""Addressing a DVID instance, and resolving which *version* of it you get.

This module is deliberately **instance-type agnostic**. Everything here works the same
for a ``labelmap``, an ``annotation`` or a ``keyvalue`` instance, because none of it
touches the data — it addresses an instance, resolves a ref to a node, and records what
that node was. The labelmap-specific parts (geometry, the array view) live in
``backends/dvid.py``, which imports from here.

That split exists because consumers above this package need the version machinery
without wanting an array. ``neu-mark`` pulls synapse elements and body annotations
out of DVID; it needs :func:`parse_url`, :func:`resolve_node`, :func:`node_summary` and
:func:`node_provenance` exactly as ``convert`` does, and reaching into a *backend* module
for them would be reaching past the thing it wanted.

**The one non-obvious rule: resolve the ref once, in the driver.** A ``uuid`` may be a
branch ref, and a branch ref names a node that moves. See :func:`resolve_node`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neu_lib import Frame

#: Accepted URL schemes -> the scheme handed to neuclease. A bare host gets ``http://``
#: prepended by neuclease's own ``dvid_api_wrapper``, so ``dvid://`` needs no prefix;
#: ``dvid+https://`` exists because that wrapper cannot be told to use TLS otherwise.
_SCHEMES = {"dvid+https://": "https://", "dvid://": ""}

#: Raised as an ImportError message wherever neuclease is first needed. Spelled out
#: because "pip install neuclease" is the natural next move and it does not work.
MISSING = (
    "Talking to DVID needs the `neuclease` package, which is not installed here.\n\n"
    "neuclease cannot be a pip dependency: it needs `libdvid-cpp`, `vigra` and "
    "`dvidutils`, which are conda-only on flyem-forge — and libdvid is load-bearing "
    "rather than incidental, since it is what inflates DVID's compressed label "
    "blocks. Install it into the environment instead:\n"
    "    mamba install -n neu-env -c flyem-forge -c conda-forge neuclease"
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
    - ``instance`` is the data instance name — ``labels`` for a labelmap,
      ``synapses`` for point annotations, ``labels_annotations`` for a keyvalue.
      Nothing here cares which; the caller checks the type it needs.

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


def address(spec: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(server, uuid, instance)`` — the argument triple every neuclease call takes."""
    try:
        return str(spec["server"]), str(spec["uuid"]), str(spec["instance"])
    except KeyError as e:
        raise ValueError(
            f"DVID spec is missing {e}; it needs 'server', 'uuid' and 'instance'") from e


#: How far back to walk looking for a locked node. In a nightly lock-and-spawn repo the
#: answer is 1 (measured on a production instance: 448 nodes on `main`, 447 locked, the
#: single open one being HEAD), so a bound this size only ever matters if something is wrong.
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
      ``data_spec`` — the same discipline as invariant 9. The identical rule applies to
      a table of synapses: half from one node and half from the next is not a snapshot.
    - **An open node is mutable**, so a pull from one is not reproducible even if
      nothing goes wrong. That is worth saying out loud, which the caller does.

    ``prefer_locked`` walks back to the newest locked node — DVID's own ``ref~N``
    ancestor syntax, which needs the ``repo:branch`` form; a bare uuid cannot be walked
    from on a multi-repo server, and is already a pinned node anyway.
    """
    try:
        from neuclease.dvid import fetch_commit, resolve_ref
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, ref, _instance = address(spec)
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

    ``neu-vol info`` shows both because they are different answers to "which version
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


def node_provenance(spec: Mapping[str, Any], node: Mapping[str, Any]) -> dict:
    """What a pull came from, for the record written beside the output.

    The resolved ``uuid`` is the load-bearing field: a branch ref means something
    different tomorrow, and the whole point of pulling successive proofreading
    snapshots is being able to say which one you have. ``requested`` keeps the ref that
    was actually typed, since "the newest locked node on main" and "this node" are
    different provenance claims even when they name the same uuid today.

    There is **no per-instance mutation id** to record: DVID's mutation id and lastmod
    endpoints are both per-*body*. What characterises a snapshot instead is the node's
    own DAG metadata, which is all this records. Callers add whatever else describes
    their own read — the labelmap backend appends ``maxlabel`` and ``supervoxels``,
    ``neu-mark`` appends element counts.
    """
    server, ref, instance = address(spec)
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
    }
    # Nice to have, and not worth failing a completed run over, so this degrades to a
    # recorded error rather than an exception.
    try:
        from neuclease.dvid import fetch_repo_info

        info = fetch_repo_info(server, node["uuid"])
        dag_node = info["DAG"]["Nodes"][node["uuid"]]
        out["node"] = {k: dag_node.get(k) for k in _NODE_FIELDS}
    except Exception as exc:                                   # noqa: BLE001
        out["node_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def instance_info(spec: Mapping[str, Any]) -> dict:
    """DVID's ``/info`` for this instance. Separate so metadata readers need no backend.

    ``read_source_metadata`` calls this rather than opening a backend, keeping the
    "what is this volume" path one HTTP request rather than a full open.
    """
    try:
        from neuclease.dvid import fetch_instance_info
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    return fetch_instance_info(*address(spec))


def instance_type(info: Mapping[str, Any]) -> str | None:
    """The instance's DVID type name — ``labelmap``, ``annotation``, ``keyvalue``, …"""
    return info.get("Base", {}).get("TypeName")


def synced_instances(info: Mapping[str, Any]) -> list[str]:
    """Instances this one is synced to, from ``Base.Syncs``.

    Load-bearing for annotation instances: DVID's ``/label/<id>`` endpoint — the only
    way to ask "which elements are in this body" — exists **only** when the annotation
    instance is synced to a labelmap. An unsynced instance answers with an error rather
    than an empty list, so callers check this up front instead of mid-fetch.
    """
    return list(info.get("Base", {}).get("Syncs") or [])


def check_instance_type(info: Mapping[str, Any], spec: Mapping[str, Any],
                        *expected: str) -> str:
    """Require the instance to be one of ``expected``, else raise. Returns its type."""
    actual = instance_type(info)
    if actual not in expected:
        want = expected[0] if len(expected) == 1 else (
            " or ".join([", ".join(expected[:-1]), expected[-1]]))
        raise ValueError(
            f"{spec_url(spec)} is a {actual!r} instance; expected {want}")
    return str(actual)


# --------------------------------------------------------------------------- #
# ROIs
# --------------------------------------------------------------------------- #
ROI_INSTANCE = "roi"

#: DVID stores an ROI as a set of whole blocks of the scale-0 grid, 32 voxels on a side,
#: and its ``ranges`` endpoint answers in block coordinates. So an ROI's native resolution
#: is one block per voxel — "scale 5" in neuclease's terms — and nothing finer exists.
ROI_BLOCK_VOXELS = 32


def available_rois(spec: Mapping[str, Any]) -> list[str]:
    """Every ``roi`` instance on this node."""
    try:
        from neuclease.dvid import fetch_repo_info
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, _instance = address(spec)
    repo = fetch_repo_info(server, uuid)
    return sorted(name for name, d in (repo.get("DataInstances") or {}).items()
                  if d.get("Base", {}).get("TypeName") == ROI_INSTANCE)


def resolve_roi_set(spec: Mapping[str, Any], rois: Sequence[str]) -> list[str]:
    """Validate an ROI name list against the node, before anything expensive happens.

    Checked up front because building a combined volume fetches every named ROI — ~1 s
    each on a real server — and a typo would otherwise surface as a failure or, worse, as
    a silently smaller region. The order given is kept: it decides which ROI wins where
    two intersect.
    """
    wanted = [str(r).strip() for r in rois if str(r).strip()]
    if not wanted:
        raise ValueError(
            "no ROIs given. There is deliberately no default: the combined ROI volume is "
            "built by overwriting, so asking for every ROI on the node would label a point "
            "in ME(L) as whichever of ME(L) / OL(L) / all_neuropils was written last.")
    duplicated = sorted({r for r in wanted if wanted.count(r) > 1})
    if duplicated:
        raise ValueError(f"ROI list repeats {', '.join(duplicated)}")
    have = set(available_rois(spec))
    missing = [r for r in wanted if r not in have]
    if missing:
        raise ValueError(
            f"no roi instance on this node named {', '.join(missing)}. "
            f"{len(have)} are available; the closest are "
            f"{', '.join(_closest(missing[0], have))}.")
    return wanted


def _closest(name: str, candidates: Iterable[str], n: int = 5) -> list[str]:
    import difflib

    return difflib.get_close_matches(name, sorted(candidates), n=n, cutoff=0.4) or \
        sorted(candidates)[:n]


def fetch_roi_ranges(spec: Mapping[str, Any], rois: Sequence[str], *,
                     processes: int = 0) -> tuple[list[str], dict, dict]:
    """``(names, ranges, boxes)`` — the expensive half of building an ROI volume.

    One request per ROI. Kept separate from the unpack because a caller measuring overlap
    unpacks the same ranges twice in different orders, and should not refetch to do it.
    ``boxes`` omits any ROI that holds no blocks.
    """
    try:
        from neuclease.dvid.roi import fetch_roi_ranges_and_boxes
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    names = resolve_roi_set(spec, rois)
    server, uuid, _instance = address(spec)
    ranges, boxes = fetch_roi_ranges_and_boxes(server, uuid, names, processes=processes)
    return names, ranges, boxes


def roi_frame(box_lo_blocks: Sequence[int], voxel_size_nm: Sequence[float],
              origin_nm: Sequence[float] = (0.0, 0.0, 0.0)) -> Frame:
    """The :class:`Frame` of an ROI array whose voxel ``(0, 0, 0)`` is block ``box_lo``.

    ``voxel_size_nm`` and ``origin_nm`` describe the **scale-0 grid** the DVID node
    indexes (the same grid its synapse coordinates are in). DVID records neither, which is
    why they are arguments: an ROI is integer blocks, and only the caller's volume knows
    how large a voxel is.
    """
    size = np.asarray(voxel_size_nm, dtype=np.float64) * ROI_BLOCK_VOXELS
    lo = np.asarray(box_lo_blocks, dtype=np.float64)
    return Frame(tuple(size), tuple(np.asarray(origin_nm, dtype=np.float64) + lo * size))


@dataclass(frozen=True)
class RoiVolume:
    """A combined ROI label array, positioned in nm.

    ``labels[z, y, x] == i`` means block ``(z, y, x)`` belongs to ``names[i - 1]``; 0 is
    in none of them. Where ROIs intersect, the one named **later** holds the voxel, and
    ``overlaps`` records each intersecting pair with its size in blocks, so a caller can
    decide whether that matters rather than never learning it happened.
    """

    labels: np.ndarray
    frame: Frame
    names: tuple[str, ...]
    overlaps: tuple[tuple[str, str, int], ...] = ()

    def mask(self, *names: str) -> np.ndarray:
        """Boolean array of the named ROIs (all of them when none are named)."""
        if not names:
            return self.labels != 0
        unknown = [n for n in names if n not in self.names]
        if unknown:
            raise KeyError(f"no ROI {unknown} in this volume; it holds {list(self.names)}")
        return np.isin(self.labels, [self.names.index(n) + 1 for n in names])


def roi_volume(spec: Mapping[str, Any], rois: Sequence[str], *,
               voxel_size_nm: Sequence[float],
               origin_nm: Sequence[float] = (0.0, 0.0, 0.0),
               processes: int = 0) -> RoiVolume:
    """Fetch the named ROIs as one label array with its :class:`Frame`.

    ``spec`` addresses any instance on the node (ROIs are per node, not per instance).
    ``voxel_size_nm`` / ``origin_nm`` are the scale-0 grid's — see :func:`roi_frame`.

    An ROI holding no blocks **raises**: it would otherwise contribute nothing and the
    result would look like a smaller region rather than a broken name.
    """
    try:
        from neuclease.dvid.roi import unpack_roi_ranges_to_combined_volume
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    names, ranges, boxes = fetch_roi_ranges(spec, rois, processes=processes)
    empty = [n for n in names if n not in boxes]
    if empty:
        raise ValueError(f"ROI(s) {', '.join(empty)} on {spec_url(spec)} hold no blocks")
    labels, box, overlaps = unpack_roi_ranges_to_combined_volume(names, ranges, boxes)
    pairs = () if overlaps is None or not len(overlaps) else tuple(
        (str(a), str(b), int(n)) for a, b, n in
        overlaps[["roi_a", "roi_b", "overlap"]].itertuples(index=False))
    return RoiVolume(labels=np.asarray(labels), frame=roi_frame(box[0], voxel_size_nm,
                                                                 origin_nm),
                     names=tuple(names), overlaps=pairs)
