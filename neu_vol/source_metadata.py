"""Read coordinate metadata + the level-0 data location from a source volume.

``convert`` uses this so you don't have to re-specify ``voxel_size``/``offset``
when the source already carries them (OME-NGFF zarr groups, precomputed ``info``).
Any field the caller passes explicitly overrides what's read here. Sources
without reliable metadata (image stacks, bare arrays, HDF5) return ``None``.

Returned dict keys: ``data_spec`` (the array/scale spec to actually read),
``voxel_size``/``offset`` (canonical z,y,x), ``units``, ``spatial_axes``,
``has_channels``.

**Two tiers, and the difference is cost.** Everything above ``describe`` reads only
small metadata documents through the kvstore — one object per call. ``describe`` and
``existing_levels`` at the bottom compose those into a whole-volume picture, and to do
it they *open every level* through a backend. That is not free: measured on an
8-level S3 zarr, the array opens took 15.8 s against 1.4 s for the metadata. Ask the
narrow question when the narrow answer will do.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .location import join, spec_kvstore, to_kvstore


#: Precomputed written by CloudVolume, whose chunk keys carry a ``.gz`` suffix.
#: tensorstore requests the *unsuffixed* key, finds nothing, and returns the fill
#: value — so the volume reads as all zeros with no error anywhere. A full 1.9 M-block
#: conversion was lost to exactly this before it was detected.
PRECOMPUTED_GZ = "neuroglancer_precomputed_gz"


def _first_chunk_key(scale: Mapping[str, Any]) -> str | None:
    """The key of a precomputed scale's origin chunk, from its metadata alone.

    ``x0-x1_y0-y1_z0-z1`` in xyz, where the extent is clipped to the scale's size —
    a scale smaller than one chunk stores `0-88_0-71_0-108`, not `0-128_...`.
    """
    try:
        off = [int(v) for v in scale.get("voxel_offset", [0, 0, 0])]
        size = [int(v) for v in scale["size"]]
        chunk = [int(v) for v in scale["chunk_sizes"][0]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return "_".join(f"{off[a]}-{off[a] + min(chunk[a], size[a])}" for a in range(3))


def precomputed_chunks_are_gzipped(location: str | Mapping[str, Any],
                                   scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's chunk objects are ``.gz``-suffixed.

    CloudVolume gzips chunks and appends ``.gz`` to the key, which is legal for
    something that serves them over HTTP with ``Content-Encoding: gzip`` but is not what
    the precomputed spec addresses — tensorstore asks for the unsuffixed key and reads
    the fill value, so every block comes back as zeros with nothing raised.

    **Answered with two existence checks, not a listing.** This used to list the scale
    prefix with ``limit=4``, which reads as bounded and is not: ``list_keys`` awaits the
    whole listing before Python truncates it, so on a dense image volume's finest scale
    it enumerated millions of keys. Measured on an 8-level EM volume on S3: **51 s, in
    `detect_backend`, which nearly every op calls before doing anything.** Probing the
    one key the metadata predicts is O(1) and needs no enumeration at all.

    Falls back to a listing only when the origin chunk is absent — legitimate on a
    sparse volume — and then over the caller's chosen scale, which is why
    :func:`detect_backend` hands it the *coarsest*: it holds the fewest chunks.
    """
    from .location import exists, list_keys

    key = _first_chunk_key(scale)
    if key is not None:
        prefix = scale.get("key", "")
        if exists(location, prefix, key):
            return False
        if exists(location, prefix, key + ".gz"):
            return True

    for name in (k.rsplit("/", 1)[-1] for k in list_keys(location, scale.get("key", ""))):
        # A chunk key looks like `0-2048_0-2048_0-128`; anything else (a shard index,
        # a stray file) is not evidence either way.
        if name and "_" in name and "-" in name:
            return name.endswith(".gz")
    return False


#: The object whose presence identifies each volume format at a location. Order
#: matters and matches :func:`detect_backend`: ``info`` is checked first, so a
#: directory holding both markers reads as precomputed.
FORMAT_MARKERS = {"neuroglancer_precomputed": "info", "zarr3": "zarr.json"}


def other_format_markers(location: str | Mapping[str, Any], fmt: str) -> list[str]:
    """Markers of volume formats *other* than ``fmt`` already present at ``location``.

    Two volumes in one directory is not hypothetical: creating a volume writes only
    its own marker, so making a precomputed volume where a zarr one already lives
    leaves ``info`` and ``zarr.json`` side by side. :func:`detect_backend` checks
    ``info`` first, so from then on the zarr is unreachable through every path in this
    package while its chunks still occupy the store — and nothing says so.
    """
    # A DVID instance has no keyspace to shadow: the question this answers — "is there
    # a second volume of another format underneath this one" — cannot arise, and asking
    # it would mean reading `dvid://...` as a local path.
    if fmt == "dvid":
        return []
    kv = to_kvstore(location)
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    fmt = "neuroglancer_precomputed" if str(fmt).startswith("neuroglancer_precomputed") \
        else fmt
    return [marker for f, marker in FORMAT_MARKERS.items()
            if f != fmt and _read_key(kv, marker) is not None]


def detect_backend(location: str | Mapping[str, Any]) -> str | None:
    """Detect a source's format from its marker file (no data read).

    ``info`` -> neuroglancer-precomputed; ``zarr.json`` -> zarr v3;
    ``.zarray``/``.zgroup`` -> zarr v2; a ``dvid://`` URL -> ``dvid``. Returns ``None``
    if none match.

    Precomputed whose chunks are ``.gz``-suffixed reports :data:`PRECOMPUTED_GZ`
    instead, so callers fail loudly rather than reading zeros.
    """
    # DVID is decided by scheme, BEFORE `to_kvstore`, because it is not a store at all:
    # `to_kvstore` would read `dvid://server/uuid/instance` as a *local file path*,
    # probe the filesystem for `info`/`zarr.json`, find nothing and return None — which
    # reaches the user as "could not detect source format", pointing nowhere near the
    # real problem.
    # From `.dvid`, not `.backends.dvid`: this runs on nearly every op's first step, and
    # the backend module pulls in numpy and registers itself just to answer a string test.
    from .dvid import is_url as _is_dvid_url

    if _is_dvid_url(location):
        return "dvid"

    kv = to_kvstore(location)
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    raw = _read_key(kv, "info")
    if raw is not None:
        try:
            scales = json.loads(raw)["scales"]
            # The COARSEST scale, not the finest: gzipping is a property of how the
            # whole volume was written, so any scale answers the question, and this one
            # holds the fewest chunks if the probe has to fall back to a listing.
            coarsest = max(scales, key=lambda s: tuple(s["resolution"]))
        except Exception:
            return "neuroglancer_precomputed"
        if precomputed_chunks_are_gzipped(kv, coarsest):
            return PRECOMPUTED_GZ
        return "neuroglancer_precomputed"
    if _read_key(kv, "zarr.json") is not None:
        return "zarr3"
    if _read_key(kv, ".zarray") is not None or _read_key(kv, ".zgroup") is not None:
        return "zarr2"
    return None


def _kvstore_of(spec: Mapping[str, Any], *, trailing_slash: bool = True) -> dict[str, Any]:
    kv = spec_kvstore(spec)  # handles 'kvstore', local 'path', and s3://... URLs
    if trailing_slash and "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    return kv


def _read_key(kvstore: Mapping[str, Any], key: str) -> bytes | None:
    """Read one metadata object.

    Goes through ``location.read_bytes`` rather than opening a kvstore directly,
    so it gets the S3 credential bootstrap and the per-prefix store cache like
    every other store-opening path. Opening raw here meant the credential chain
    fell through to the EC2 metadata service — which does not exist on Rusty — and
    waited out a socket timeout on *each* of ``detect_backend``'s four probes.
    """
    from .location import read_bytes

    return read_bytes(dict(kvstore), key)


def location_spec(location: str, fmt: str) -> dict[str, Any]:
    """The backend spec addressing ``location``, in the form ``fmt`` expects.

    Three forms, because not every source is a store: most take ``path``, an image
    stack takes ``source`` (a directory or glob), and DVID takes
    ``server``/``uuid``/``instance``. Everything that turns a user-supplied location
    into a spec goes through here, so a new non-store backend is one branch rather
    than a search for every place that wrote ``{"backend": fmt, "path": ...}``.
    """
    if fmt == "dvid":
        from .dvid import parse_url

        return {"backend": fmt, **parse_url(location)}
    # The image-stack backend is never auto-detected — a directory of PNGs looks like
    # any other directory — so it only arrives here when asked for by name.
    if fmt == "image_stack":
        return {"backend": fmt, "source": location}
    return {"backend": fmt, "path": location}


def read_source_metadata(spec: Mapping[str, Any]) -> dict | None:
    backend = spec.get("backend")
    if backend == "zarr3":
        return _read_zarr_ome(spec)
    if backend == "dvid":
        return _read_dvid(spec)
    # PRECOMPUTED_GZ reads the SAME `info` document — only the chunk keys differ, and
    # metadata does not live in chunks. Omitting it here silently loses voxel_size,
    # offset and kind, which makes `convert` demand --voxel-size and, worse, lose the
    # image/segmentation distinction that decides mean vs mode downsampling.
    if backend in ("neuroglancer_precomputed", PRECOMPUTED_GZ):
        return _read_precomputed(spec)
    return None


def read_level_voxel_sizes(spec: Mapping[str, Any]) -> list[tuple[float, ...]] | None:
    """Each level's own voxel size, from the volume's metadata. ``None`` if absent.

    **Not derived from shape ratios.** Shapes are ceil-divided, so a level-0 extent of
    13750 over a factor of 4 stores 3438, and 13750/3438 is 3.9994 — deriving from that
    reports 31.9953 nm for a level that is exactly 32. Real pyramids are also
    anisotropic, so the ``2**level`` shortcut is wrong for a different reason. Both
    formats record the true value per level; read it.

    precomputed lists a ``resolution`` per entry of ``info["scales"]``; OME-NGFF lists
    a scale transform per ``multiscales[0]["datasets"]`` entry. Returned finest-first
    in ``(z, y, x)``, spatial axes only.
    """
    backend = spec.get("backend")
    if backend == "dvid":
        # DVID's downres really is a factor of 2 on every axis at every level — that is
        # its documented model, not the `2**level` assumption invariant 1 warns about,
        # and it must not be generalised to any other backend. The pyramid depth is
        # whatever the instance recorded in MaxDownresLevel.
        from .backends.dvid import geometry
        from .dvid import instance_info

        geom = geometry(instance_info(spec), spec)
        return [tuple(v * 2 ** i for v in geom["voxel_size"])
                for i in range(geom["max_scale"] + 1)]
    if backend in ("neuroglancer_precomputed", PRECOMPUTED_GZ):   # same `info`
        raw = _read_key(_kvstore_of(spec), "info")
        if raw is None:
            return None
        scales = json.loads(raw)["scales"]
        ordered = sorted(scales, key=lambda s: tuple(s["resolution"]))
        return [tuple(float(v) for v in s["resolution"][::-1]) for s in ordered]
    if backend == "zarr3":
        raw = _read_key(_kvstore_of(spec), "zarr.json")
        if raw is None:
            return None
        ome = json.loads(raw).get("attributes", {}).get("ome")
        if not ome:
            return None
        ms = ome["multiscales"][0]
        spatial = [i for i, a in enumerate(ms["axes"]) if a.get("type") == "space"]
        out = []
        for ds in ms["datasets"]:
            scale = next(t["scale"] for t in ds["coordinateTransformations"]
                         if t["type"] == "scale")
            out.append(tuple(float(scale[i]) for i in spatial))
        return out
    return None


def _read_zarr_ome(spec: Mapping[str, Any]) -> dict | None:
    raw = _read_key(_kvstore_of(spec), "zarr.json")
    if raw is None:
        return None
    meta = json.loads(raw)
    attrs = meta.get("attributes", {})
    ome = attrs.get("ome")
    if meta.get("node_type") != "group" or not ome:
        return None  # bare array (or non-OME group) -> no coordinate metadata
    ms = ome["multiscales"][0]
    axes = ms["axes"]
    ds0 = ms["datasets"][0]
    ct = ds0["coordinateTransformations"]
    scale = next(t["scale"] for t in ct if t["type"] == "scale")
    translation = next((t["translation"] for t in ct if t["type"] == "translation"),
                       [0.0] * len(scale))

    spatial = [i for i, a in enumerate(axes) if a.get("type") == "space"]
    has_channels = any(a.get("type") == "channel" for a in axes)
    return {
        "data_spec": {"backend": "zarr3", "kvstore": join(spec_kvstore(spec), ds0["path"])},
        "voxel_size": tuple(scale[i] for i in spatial),
        "offset": tuple(translation[i] for i in spatial),
        "units": axes[spatial[0]].get("unit"),
        "spatial_axes": tuple(axes[i]["name"] for i in spatial),
        "has_channels": has_channels,
        # The multiscales "type" is the downsampling method; we write `kind` there,
        # but another writer may put something else, so only trust our own values.
        "kind": ms.get("type") if ms.get("type") in ("image", "segmentation") else None,
    }


def _read_dvid(spec: Mapping[str, Any]) -> dict | None:
    """Coordinate metadata for a DVID labelmap, from one ``/info`` request.

    ``kind`` is **always** ``segmentation``: a labelmap is one by definition, and the
    consequence of getting it wrong is silent — the pyramid would average label ids
    into ids that were never in the data (CLAUDE.md invariant 9's sibling hazard, and
    the reason `copy` exists at all).

    ``offset`` is 0: the backend presents DVID's coordinates origin-anchored, so the
    output frame already matches DVID's own. ``MinPoint`` is a fact about where data
    *starts*, not a translation to apply.

    **The branch ref is resolved to a concrete uuid here**, and that is what goes into
    ``data_spec`` — so every worker reads the node the driver chose. Leaving the ref in
    place would let a lock-and-spawn landing mid-run move HEAD, and the output would be
    stitched from two versions of the segmentation with nothing to show it. Reading from
    an *open* node is legal but not reproducible, so it is warned about.
    """
    import logging

    from .backends.dvid import geometry
    from .dvid import instance_info, resolve_node

    node = resolve_node(spec, prefer_locked=bool(spec.get("prefer_locked")))
    # Everything downstream addresses the concrete node. `prefer_locked` is dropped
    # because it has now been *applied* — carrying it would invite a second resolution
    # on a worker, which is the drift this exists to prevent.
    resolved = {k: v for k, v in spec.items() if k != "prefer_locked"}
    resolved["uuid"] = node["uuid"]
    # Keep what was actually asked for. "the newest locked node on main" and "this node"
    # are different provenance claims even when they name the same uuid today, and the
    # resolved spec is what the record is built from.
    resolved["requested_ref"] = node["ref"]
    resolved["ancestors_walked"] = node["walked"]

    # Resolution is reported; the fact that a node is OPEN is *warned about* by the op
    # that is going to export it, not here. This function is also on the path of
    # read-only inspection (`neu-vol info`, the CLI's crop peeks), where a warning about
    # reproducibility is noise — `info` prints the open/locked status in its own table.
    log = logging.getLogger(__name__)
    if node["uuid"] != str(spec.get("uuid")):
        log.info("dvid %s resolved to node %s (%s)%s", spec.get("uuid"), node["uuid"],
                 "locked" if node["locked"] else "OPEN",
                 f", {node['walked']} ancestor(s) back" if node["walked"] else "")

    geom = geometry(instance_info(resolved), resolved)
    return {
        # Carries the caller's backend through, like every other reader here
        # (invariant 9): `data_spec` selects the reader and must not contradict what
        # detection decided.
        "data_spec": {**resolved, "backend": spec.get("backend") or "dvid",
                      "scale_index": int(spec.get("scale_index", 0))},
        "provenance_spec": resolved,
        "voxel_size": geom["voxel_size"],
        "offset": (0.0,) * len(geom["voxel_size"]),
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": False,
        "kind": "segmentation",
    }


def _read_precomputed(spec: Mapping[str, Any]) -> dict | None:
    raw = _read_key(_kvstore_of(spec), "info")
    if raw is None:
        return None
    info = json.loads(raw)
    scales = info["scales"]
    # level 0 = finest scale (smallest resolution).
    idx, sc0 = min(enumerate(scales), key=lambda kv: tuple(kv[1]["resolution"]))
    res_xyz = sc0["resolution"]
    voxel_size = tuple(float(v) for v in res_xyz[::-1])          # -> (z, y, x)
    voxel_offset_xyz = sc0.get("voxel_offset", [0, 0, 0])
    offset = tuple(float(o) * v for o, v in zip(voxel_offset_xyz[::-1], voxel_size))
    nch = int(info.get("num_channels", 1))
    return {
        # Carry the CALLER's backend through. Hardcoding "neuroglancer_precomputed"
        # here silently routed .gz volumes back to tensorstore — detection said
        # PRECOMPUTED_GZ, but the spec the workers actually read through said
        # otherwise, so every block read as zeros. `data_spec` selects the reader;
        # it must not contradict what detection decided.
        "data_spec": {"backend": spec.get("backend") or "neuroglancer_precomputed",
                      "kvstore": spec_kvstore(spec), "scale_index": idx},
        "voxel_size": voxel_size,
        "offset": offset,
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": nch > 1,
        # precomputed records the volume type directly, which is what decides
        # whether downsampling may average (image) or must take a mode (labels).
        "kind": "segmentation" if info.get("type") == "segmentation" else "image",
    }


# --------------------------------------------------------------------------- #
# The whole-volume picture: format + coordinates + the levels actually present.
#
# These compose the readers above and additionally OPEN each level, which is the
# expensive part (see the module docstring). `info`, `downsample`, `create --like`
# and `write` all go through here, so none of them can disagree about what is on disk.
# --------------------------------------------------------------------------- #
def level_spec(volume: str, fmt: str, level: int) -> dict[str, Any]:
    """The backend spec for one level of a multiscale volume.

    The two formats address a level differently: zarr v3 puts each level in its own
    subdirectory, while precomputed keeps every scale under one ``info`` and selects
    with ``scale_index``. Everything that opens a level goes through here rather than
    building the path itself.
    """
    if fmt == "zarr3":
        return {"backend": fmt, "path": f"{volume.rstrip('/')}/{level}"}
    if fmt == "dvid":
        # A DVID level is a downres scale of one instance, addressed the same way
        # precomputed addresses a scale — by index, not by a distinct location.
        return {**location_spec(volume, fmt), "scale_index": int(level)}
    return {"backend": fmt, "path": volume, "scale_index": int(level)}


def describe(volume: str) -> dict:
    """Detected format, coordinate metadata and the levels actually present.

    Raises ``FileNotFoundError`` if nothing at ``volume`` looks like a volume.
    """
    from .backends.base import open_backend

    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")
    spec = location_spec(volume, fmt)
    meta = read_source_metadata(spec)
    level0 = open_backend(meta["data_spec"] if meta else spec)
    shape = tuple(int(s) for s in level0.shape)
    return {"format": fmt, "meta": meta, "shape": shape,
            "dtype": str(getattr(level0, "dtype", "?")),
            "has_channels": bool(meta["has_channels"]) if meta else False,
            "level_voxel_sizes": read_level_voxel_sizes(spec),
            "levels": existing_levels(volume, fmt),
            # One extra small read, against several array opens — worth it to notice
            # a second volume shadowed underneath this one.
            "other_markers": other_format_markers(volume, fmt)}


def existing_levels(volume: str, fmt: str, probe: int = 12) -> dict[int, dict]:
    """``{level: {"shape", "chunks", "read_chunks"}}`` for the levels that open.

    Probes upward until one misses. The multiscale group metadata is written at the
    very end of a conversion, so an in-flight volume has levels but no group metadata
    — probing is what makes this work on a run that is still going.

    Chunking is **per level**, not a property of the volume: it lives in each level's
    own array metadata (zarr's ``zarr.json``, precomputed's per-scale ``chunk_sizes``),
    and a conversion is free to chunk levels differently. So it is read here, where
    each level is opened anyway, rather than assumed from level 0.

    ``chunks`` is the *write* chunk and ``read_chunks`` the *read* chunk. They differ
    only when the level is sharded, where the write chunk is the shard and the read
    chunk is the unit actually fetched — which is the number that governs read
    amplification, so both are worth having.
    """
    from .backends.base import open_backend

    out: dict[int, dict] = {}
    for i in range(probe):
        try:
            be = open_backend(level_spec(volume, fmt, i))
            shape = tuple(int(s) for s in be.shape)
        except Exception:
            break
        # chunks come from the same open, but a backend need not expose them
        def _maybe(attr):
            try:
                return tuple(int(c) for c in getattr(be, attr))
            except Exception:
                return None
        out[i] = {"shape": shape, "chunks": _maybe("chunks"),
                  "read_chunks": _maybe("read_chunks")}
    return out
