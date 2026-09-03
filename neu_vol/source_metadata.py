"""Read coordinate metadata + the level-0 data location from a source volume.

``convert`` uses this so you don't have to re-specify ``voxel_size``/``offset``
when the source already carries them (OME-NGFF zarr groups, precomputed ``info``,
and an HDF5 file that records its own frame — which is what ``neu-vol to-hdf5``
writes). Any field the caller passes explicitly overrides what's read here. Sources
that record nothing (image stacks, bare arrays, an HDF5 file with no frame attributes)
return ``None``.

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
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .location import join, spec_kvstore, to_kvstore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:                       # the class is imported inside `describe`, where
    from .report import Description     # it is used, so the module stays import-light


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


def precomputed_chunks_are_gzipped(
        location: str | Mapping[str, Any],
        scales: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> bool:
    """True if these precomputed scales' chunk objects are ``.gz``-suffixed.

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

    Falls back to a listing only when no origin chunk is found — legitimate on a sparse
    volume — and then over the *first* scale given, which is why :func:`detect_backend`
    orders them coarsest-first: that one holds the fewest chunks.

    ``scales`` is one entry of ``info["scales"]`` or several in preference order. **With
    several, every O(1) origin probe is tried before any listing**, which is what makes
    a volume whose ``info`` declares scales it never wrote answerable at all. A
    ``sample3`` neuropil mask declares seven scales and stores exactly one (``64_64_64``,
    18 chunks): asked about the coarsest alone this found no origin chunk, listed an
    empty prefix, and reported the volume as un-gzipped — so every block read as zeros,
    the precise failure the two-check probe exists to prevent. Sweeping the origins
    first costs 14 existence checks and 53 ms there, and still answers a dense volume on
    the very first probe, so the 51 s enumeration stays gone.
    """
    from .location import exists, list_keys

    entries = [scales] if isinstance(scales, Mapping) else list(scales)

    for scale in entries:
        key = _first_chunk_key(scale)
        if key is None:
            continue
        prefix = scale.get("key", "")
        if exists(location, prefix, key):
            return False
        if exists(location, prefix, key + ".gz"):
            return True

    # Every scale, in the order given, until one yields a chunk name — NOT just the
    # first. Stopping at the first was the same origin-chunk assumption in another
    # costume: a volume sparse at the origin of every scale falls through pass 1
    # entirely, and if its cheapest scale is one of the unwritten ones the listing
    # finds nothing and reports the volume as un-gzipped. That is what happened to a
    # registered neuropil mask — declared five scales, wrote only `64_64_64`, and none
    # of its 26 chunks at the origin. Detection answered `neuroglancer_precomputed`,
    # tensorstore asked for unsuffixed keys, and all 80 level-0 tasks recorded "empty":
    # a full run that wrote nothing and raised nothing.
    #
    # Cost is unchanged where it matters. An unwritten scale's prefix is empty and
    # lists instantly, and a volume whose chunks ARE at the origin never reaches here.
    for scale in entries:
        for name in (k.rsplit("/", 1)[-1]
                     for k in list_keys(location, scale.get("key", ""))):
            # A chunk key looks like `0-2048_0-2048_0-128`; anything else (a shard
            # index, a stray file) is not evidence either way.
            if name and "_" in name and "-" in name:
                return name.endswith(".gz")
    return False


def scale_stores_chunks(location: str | Mapping[str, Any],
                        scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's ORIGIN chunk object is stored.

    The same two O(1) existence checks :func:`precomputed_chunks_are_gzipped` makes,
    asked for the other reason — either suffix counts, since a CloudVolume-written
    scale is still a written one.

    **A True is evidence of a written scale; a False is not proof of an empty one.** A
    sparse volume may legitimately store nothing at its origin, which is why
    :func:`finest_populated_scale` falls back rather than raising and why
    :func:`require_populated_scale` needs a second scale to disagree with before it
    will refuse anything.
    """
    from .location import exists

    key = _first_chunk_key(scale)
    if key is None:
        return False
    prefix = scale.get("key", "")
    return bool(exists(location, prefix, key) or exists(location, prefix, key + ".gz"))


def scale_is_empty(location: str | Mapping[str, Any],
                   scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's prefix holds NO chunk object at all.

    The conclusive form of the question :func:`scale_stores_chunks` answers cheaply,
    and the two exist as a pair because **the origin chunk is not a sound test for an
    unwritten scale**. A sparse volume elides all-background chunks, so a ground-truth
    crop sitting away from the origin stores nothing there while being perfectly
    written — reading the origin alone, such a scale is indistinguishable from one that
    was only ever declared.

    Costs a listing, so it is used only after the cheap probe has already failed. Note
    it is **cheap exactly when it matters**: an unwritten scale's prefix is empty and
    the listing returns immediately. The expensive case is a genuinely sparse volume
    whose origin happens to be background, where the full prefix is enumerated
    (``limit`` does not bound a listing — invariant STORE-ACCESS). Naming the level with
    ``--src-level`` skips the probe entirely and costs nothing.
    """
    from .location import list_keys

    for key in list_keys(location, scale.get("key", "")):
        name = key.rsplit("/", 1)[-1]
        # A chunk key looks like `0-2048_0-2048_0-128`; a shard index or a stray file
        # is not a chunk and says nothing about whether this scale holds data.
        if name and "_" in name and "-" in name:
            return False
    return True


def finest_populated_scale(location: str | Mapping[str, Any],
                           scales: Sequence[Mapping[str, Any]]) -> int | None:
    """Index into ``scales`` of the finest one that actually holds data.

    ``None`` when no scale stores anything findable, which is a caller's problem to
    decide about rather than an answer.

    **An `info` may declare scales that were never written**, and level 0 is then a
    scale that does not exist. A sample3 neuropil mask declares seven (8 nm through
    512 nm) and stores exactly one — ``64_64_64``, 18 chunk objects. The finest scale
    opened without complaint and reported the full ``(11260, 9000, 13750)`` extent its
    ``info`` claims, so ``convert`` read *that*, every block came back as the fill
    value, and the copy would have been a structurally perfect volume of zeros.

    Three tiers, cheapest first, and an ordinary volume never leaves the first:

    1. Probe origin chunks finest-first. The finest scale is normally written like
       every other, so this answers in ONE existence check and nothing else runs.
    2. If some coarser scale has an origin chunk and finer ones do not, those finer
       scales are either unwritten or merely sparse at the origin — which the origin
       cannot distinguish. :func:`scale_is_empty` settles it, and settles it instantly
       in the unwritten case, because an unwritten prefix is empty.
    3. If NO scale has an origin chunk, fall back to the conclusive test across every
       scale, finest-first. A volume can be sparse at the origin of *all* of its
       scales and still be perfectly written — a registered neuropil mask whose data
       starts 512 voxels in stores 26 chunks at 64 nm, none of them the origin one, so
       tier 1 sees nothing anywhere. Giving up here fell back to the finest declared
       scale, which is the very failure this function exists to prevent.

    Tier 2 is what stops a sparse ground-truth volume — no origin chunk at scale 0, one
    at scale 1, every scale correctly written — from being read at the wrong scale.
    """
    order = sorted(range(len(scales)), key=lambda i: tuple(scales[i]["resolution"]))
    for pos, i in enumerate(order):
        if not scale_stores_chunks(location, scales[i]):
            continue
        if pos == 0:
            return i                     # the finest scale holds data: nothing to weigh
        # Finer scales exist on paper but had no origin chunk. Sparse, or never written?
        for j in order[:pos]:
            if not scale_is_empty(location, scales[j]):
                return j                 # sparse, and genuinely the finest data present
        return i

    # Tier 3. Nothing was found at any origin, so every scale is still in question and
    # only a listing can separate "sparse here" from "never written". Finest-first, so
    # the answer is the same one tiers 1-2 would have given.
    for i in order:
        if not scale_is_empty(location, scales[i]):
            return i
    return None


def require_populated_scale(spec: Mapping[str, Any], *, op: str = "convert") -> None:
    """Raise if ``spec`` names a precomputed scale this volume never wrote.

    The failure this prevents is entirely silent. An unwritten scale opens, reports the
    extent its ``info`` declares, and reads as the fill value at every block — so the
    run succeeds, the manifest fills up, and the output is a correct-looking volume of
    zeros. Nothing downstream can tell it from a volume that really is empty.

    **Raises only on conclusive evidence**: this scale's prefix holds no chunk object
    at all, while some other scale's does. An absent *origin* chunk is not enough —
    a sparse volume elides background chunks and may legitimately store nothing there,
    and refusing one of those would be a worse failure than the one being prevented.
    The cheap origin probe runs first and returns immediately on any ordinary volume,
    so the listing behind the conclusive form is only reached when there is already
    something to explain.
    """
    if not str(spec.get("backend", "")).startswith("neuroglancer_precomputed"):
        return
    kv = _kvstore_of(spec)
    raw = _read_key(kv, "info")
    if raw is None:
        return
    scales = json.loads(raw)["scales"]
    idx = int(spec.get("scale_index") or 0)
    if not 0 <= idx < len(scales):
        raise ValueError(
            f"`{op}`: source level {idx} does not exist; its `info` declares "
            f"{len(scales)} scale(s), 0-{len(scales) - 1}")
    if scale_stores_chunks(kv, scales[idx]) or not scale_is_empty(kv, scales[idx]):
        return

    populated = [i for i in range(len(scales))
                 if i != idx and (scale_stores_chunks(kv, scales[i])
                                  or not scale_is_empty(kv, scales[i]))]
    if not populated:
        logger.warning(
            "source scale %s (%s) holds no chunk objects, and neither does any other "
            "scale — this volume appears to be entirely empty, so reading it will "
            "produce nothing but the fill value.",
            idx, scales[idx].get("key"))
        return

    listed = ", ".join(f"{i} ({scales[i].get('key')})" for i in populated)
    raise ValueError(
        f"`{op}`: source level {idx} ({scales[idx].get('key')}) stores no data — its "
        f"`info` declares the scale, but nothing was ever written there, so every "
        f"block would read as the fill value and the output would be a volume of "
        f"zeros with nothing raised.\n\n"
        f"  levels that DO store data: {listed}\n\n"
        f"Pass --src-level to choose one explicitly.")


#: The object whose presence identifies each volume format at a location. Order
#: matters and matches :func:`detect_backend`: ``info`` is checked first, so a
#: directory holding both markers reads as precomputed.
FORMAT_MARKERS = {"neuroglancer_precomputed": "info", "zarr3": "zarr.json"}

#: Formats that are ONE array in ONE container: no pyramid, and no chunk keyspace to
#: enumerate. Detection reaches them (see :func:`detect_file_backend`) so they can be
#: *inspected* and *read*, but every op that rewrites a volume in place —
#: ``downsample``, ``relabel``, ``mask-by-value``, ``write``, ``progress`` — needs the
#: other kind and says so through :func:`require_chunked_volume`. Without that guard
#: they fail deep inside an occupancy listing that finds no chunk keys, which reads as
#: "this volume is empty" rather than "this is not that kind of volume".
SINGLE_LEVEL_FORMATS = frozenset({"hdf5", "image_stack"})


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
    # it would mean reading `dvid://...` as a local path. Nor does a single container:
    # an HDF5 file is a file, and probing `piece.h5/info` is two reads that can only
    # miss. An image stack's directory *could* hold a marker, but then detection would
    # have returned that format instead, so there is nothing left to shadow.
    if fmt == "dvid" or fmt in SINGLE_LEVEL_FORMATS:
        return []
    kv = to_kvstore(location)
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    fmt = "neuroglancer_precomputed" if str(fmt).startswith("neuroglancer_precomputed") \
        else fmt
    return [marker for f, marker in FORMAT_MARKERS.items()
            if f != fmt and _read_key(kv, marker) is not None]


def detect_file_backend(location: str | Mapping[str, Any]) -> str | None:
    """``hdf5`` or ``image_stack`` for a local file, glob or directory of slices.

    These two have **no marker object**, so they cannot be detected the way the store
    formats are: HDF5's signature is inside the file, and a stack of PNGs is a directory
    with nothing in it that says so. What is left is the name plus one ``stat`` — which
    is why this runs *last*, after every marker probe has missed, and why it is
    deliberately narrow:

    * a **glob** is an image stack and nothing else — no other format is ever addressed
      with one;
    * a **file** is judged by extension, and only the two known sets count. An unknown
      extension returns ``None`` rather than being guessed at;
    * a **directory** must actually contain a slice. "Any directory is an image stack"
      is what ``ops/write.py`` used to assume, and it turns every typo into a stack whose
      reader then reports "no image files matched" — so the listing is the evidence, and
      an empty or unrelated directory is still ``None``.

    **Local paths only.** Neither reader has an object-store driver — h5py has none at
    all, and the stack reader globs the filesystem — so detecting either on ``s3://``
    would name a backend that cannot open it. The probes are also ``os.stat`` calls,
    which is why the locality test comes first rather than being a detail of the readers.
    """
    import os
    from glob import has_magic

    from .backends.hdf5 import HDF5_EXTENSIONS
    from .backends.imagestack import IMAGE_EXTENSIONS
    from .location import is_local, local_path

    if not is_local(location):
        return None
    path = local_path(location).rstrip("/")
    if not path:
        return None
    if has_magic(path):
        return "image_stack"
    if os.path.isfile(path):
        low = path.lower()
        if low.endswith(HDF5_EXTENSIONS):
            return "hdf5"
        if low.endswith(IMAGE_EXTENSIONS):
            return "image_stack"      # a single multipage TIFF is a stack
        return None
    if not os.path.isdir(path):
        return None
    try:
        names = os.listdir(path)
    except OSError:
        return None
    found = {os.path.splitext(n)[1].lower() for n in names} & set(IMAGE_EXTENSIONS)
    if not found:
        return None
    if len(found) > 1:
        # The reader takes every slice extension it knows and sorts them together, so a
        # mixed directory interleaves two stacks into one volume. It is a legal thing to
        # ask for and almost never what anyone means, and nothing later can see it.
        import logging

        logging.getLogger(__name__).warning(
            "%s holds slices of more than one type (%s); they will be read as ONE stack, "
            "sorted together. If they are two stacks, point at a glob of one of them",
            path, ", ".join(sorted(found)))
    return "image_stack"


def detect_backend(location: str | Mapping[str, Any]) -> str | None:
    """Detect a source's format from its marker file (no data read).

    ``info`` -> neuroglancer-precomputed; ``zarr.json`` -> zarr v3;
    ``.zarray``/``.zgroup`` -> zarr v2; a ``dvid://`` URL -> ``dvid``. Returns ``None``
    if none match.

    Precomputed whose chunks are ``.gz``-suffixed reports :data:`PRECOMPUTED_GZ`
    instead, so callers fail loudly rather than reading zeros.

    **A marker always wins.** The two container formats — ``hdf5`` and ``image_stack``
    — have no marker to probe for and are recognised from the name by
    :func:`detect_file_backend`, which runs only after every marker probe has missed. So
    a directory holding both an ``info`` and a stray TIFF is still precomputed, and the
    order cannot be reversed by adding a file.
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
            # COARSEST FIRST, then the rest. Gzipping is a property of how the whole
            # volume was written, so any scale that STORES a chunk answers the question
            # — but one that stores nothing answers nothing, and an `info` may well
            # declare scales that were never written. So all of them are offered, in the
            # order that keeps the listing fallback on the cheapest prefix.
            ordered = sorted(scales, key=lambda s: tuple(s["resolution"]), reverse=True)
        except Exception:
            return "neuroglancer_precomputed"
        if precomputed_chunks_are_gzipped(kv, ordered):
            return PRECOMPUTED_GZ
        return "neuroglancer_precomputed"
    if _read_key(kv, "zarr.json") is not None:
        return "zarr3"
    if _read_key(kv, ".zarray") is not None or _read_key(kv, ".zgroup") is not None:
        return "zarr2"
    # `location`, not `kv`: the trailing slash added above makes an ordinary file path
    # fail every `os.path.isfile` test, and a spec's own kvstore has already been
    # normalised past the form the filesystem answers about.
    return detect_file_backend(location)


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


def location_spec(location: str, fmt: str, *,
                  dataset: str | None = None) -> dict[str, Any]:
    """The backend spec addressing ``location``, in the form ``fmt`` expects.

    Four forms, because not every source is a store: most take ``path``, an image
    stack takes ``source`` (a directory or glob), HDF5 takes ``path`` plus the
    ``dataset`` inside the container, and DVID takes
    ``server``/``uuid``/``instance``. Everything that turns a user-supplied location
    into a spec goes through here, so a new non-store backend is one branch rather
    than a search for every place that wrote ``{"backend": fmt, "path": ...}``.

    ``dataset`` names the array inside an HDF5 file and is resolved by
    :func:`~neu_vol.backends.hdf5.sole_dataset` when it is not given — which answers
    only when the file holds exactly one candidate and otherwise raises listing them.
    That is deliberate: a container is ambiguous in a way a path is not, and guessing
    which array someone meant is worse than asking. Ignored for every other format.
    """
    if fmt == "dvid":
        from .dvid import parse_url

        return {"backend": fmt, **parse_url(location)}
    if fmt == "image_stack":
        return {"backend": fmt, "source": location}
    if fmt == "hdf5":
        if dataset is None:
            from .backends.hdf5 import sole_dataset

            dataset = sole_dataset(location)
        return {"backend": fmt, "path": location, "dataset": dataset}
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
    if backend == "hdf5":
        return _read_hdf5(spec)
    # An image stack records no physical scale at all — an image file has pixels and
    # nothing else — so there is genuinely nothing to read, and `convert` asking for
    # --voxel-size is the correct outcome rather than a shortcoming.
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
    if backend == "hdf5":
        # One level, and it is the file's own. Returned as a list of one so a caller
        # asking "what does level N measure" gets the same shape of answer for every
        # format — and so `--bbox-scale 2` against an HDF5 piece reports "the source
        # records only 1 level(s)" rather than "records none", which is a different
        # and misleading complaint.
        meta = _read_hdf5(spec)
        return [tuple(meta["voxel_size"])] if meta else None
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


def _read_hdf5(spec: Mapping[str, Any]) -> dict | None:
    """The frame an HDF5 file records about itself, or ``None`` if it records none.

    An HDF5 file is a container, not a format with a metadata document, so there is no
    ``info`` to read — the numbers live in attributes beside the array, and
    ``HDF5Backend.stored_*`` is what finds them (dataset attributes, then the root
    group's, then a top-level dataset of that name; writers use all three).
    ``neu-vol to-hdf5`` writes ``voxel_size``, ``units``, ``axes`` and ``voxel_offset``,
    so a file this package packed round-trips through ``neu-vol info``, ``create --like``
    and ``convert`` without anyone retyping a coordinate.

    **``voxel_size`` is the gate.** Without it there is no frame — an offset in voxels
    means nothing physical and ``units`` describes numbers that are not there — so the
    whole dict is withheld rather than half-filled, which is what keeps ``convert``'s
    "voxel_size is required" the honest answer for a bare array.

    **The axis order is read, never assumed, and an uninterpretable one raises.** The
    field name ``voxel_offset`` is precomputed's, where it means *xyz*, while everything
    in this package is zyx — so the numbers alone cannot say. A recorded ``axes``
    settles it; nothing recorded falls back to zyx, which is what the reader already
    assumes of the array itself, and warns when the fallback could matter (an
    anisotropic vector is a different volume reversed; an isotropic one is not). An
    ``axes`` that is neither order is refused outright: it is a *stated* fact that this
    cannot honour, and quietly substituting zyx for it would mirror the data through the
    z=x diagonal with nothing downstream able to tell.
    """
    import logging

    from .backends.base import open_backend

    backend = open_backend(dict(spec))
    size = backend.stored_voxel_size(spec.get("voxel_size_field", "voxel_size"))
    if not size:
        return None
    log = logging.getLogger(__name__)

    stated = backend.stored_axes()
    if stated and stated[0] not in ("zyx", "xyz"):
        raise ValueError(
            f"{spec.get('path')} records axes {stated[0]!r} ({stated[1]}), which is "
            f"neither 'zyx' nor 'xyz', so nothing here can say which way round its "
            f"voxel_size and voxel_offset are. Fix the attribute, or read the file with "
            f"an explicit spec")
    order = stated[0] if stated else "zyx"

    voxel_size = tuple(float(v) for v in size[0])
    if not stated and voxel_size != voxel_size[::-1]:
        log.warning(
            "%s records voxel_size %s but no `axes`, so it is read as zyx (this "
            "package's convention). Reversed it would be %s — a different volume. "
            "`neu-vol to-hdf5` records `axes`; add one to remove the doubt",
            spec.get("path"), voxel_size, voxel_size[::-1])
    if order == "xyz":
        voxel_size = voxel_size[::-1]

    found = backend.stored_offset(spec.get("offset_field", "voxel_offset"))
    voxel_offset = tuple(int(v) for v in found[0]) if found else (0,) * len(voxel_size)
    if found and order == "xyz":
        voxel_offset = voxel_offset[::-1]

    units = backend.stored_units()
    shape = tuple(int(s) for s in backend.shape)
    axes = tuple(order)

    # An HDF5 file has no agreed-on place to say image or segmentation, so a `kind` this
    # package did not write is not evidence — `kind` is a plain word another tool may have
    # used for something else. Only a member of the vocabulary is trusted, the same rule
    # `_read_zarr_ome` applies to the multiscales `type`, and anything else reads as
    # unrecorded rather than raising: it costs a `convert --kind`, where trusting it wrongly
    # would average label ids into ids that were never in the data.
    from neu_lib import KINDS

    stated_kind = backend.stored_kind()
    kind = stated_kind[0] if stated_kind and stated_kind[0] in KINDS else None
    if stated_kind and kind is None:
        log.info("%s records kind %r (%s), which is not one of %s, so it is ignored",
                 spec.get("path"), stated_kind[0], stated_kind[1], ", ".join(KINDS))
    return {
        # The spec as given: an HDF5 dataset IS the array, so there is no level or scale
        # to descend into, and `data_spec` carries the resolved `dataset` through the
        # same way every other reader carries the caller's backend (invariant 9).
        "data_spec": dict(spec),
        "voxel_size": voxel_size,
        "offset": tuple(float(o) * v for o, v in zip(voxel_offset, voxel_size)),
        "voxel_offset": voxel_offset,
        "units": units[0] if units else None,
        "spatial_axes": axes,
        "has_channels": len(shape) > len(axes),
        # Recorded when the file says so in this package's own vocabulary, and `None`
        # otherwise — never inferred from the dtype, which is the guess that averages label
        # ids into ids that were never in the data. Unrecorded means `convert --kind`.
        "kind": kind,
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
    kv = _kvstore_of(spec)
    raw = _read_key(kv, "info")
    if raw is None:
        return None
    info = json.loads(raw)
    scales = info["scales"]
    finest = min(range(len(scales)), key=lambda i: tuple(scales[i]["resolution"]))

    # Level 0 = the finest scale that was actually WRITTEN, not simply the finest one
    # declared. `info` is a claim about the volume's layout and the chunk objects are
    # the volume; where they disagree, believe the store. An explicit `scale_index`
    # (what `--src-level` sets) always wins, so the caller can name a scale the probe
    # would not have chosen — including one it thinks is empty, which `neu-vol write`
    # legitimately does when filling a level that does not exist yet.
    idx = spec.get("scale_index")
    if idx is None:
        idx = finest_populated_scale(kv, scales)
        if idx is None:
            idx = finest
        elif idx != finest:
            logger.warning(
                "level 0 is scale %s (%s): the %s finer scale(s) this volume's `info` "
                "declares store no data. Reading %s would have returned the fill value "
                "everywhere. Override with --src-level.",
                idx, scales[idx].get("key"), idx, scales[finest].get("key"))
    idx = int(idx)
    if not 0 <= idx < len(scales):
        raise ValueError(
            f"source level {idx} does not exist: this volume's `info` declares "
            f"{len(scales)} scale(s), 0-{len(scales) - 1}")
    sc0 = scales[idx]
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
def require_chunked_volume(fmt: str, location: str, op: str) -> str:
    """``fmt`` if it is a chunked multiscale volume; a useful error if it is not.

    Detection now reaches HDF5 files and image stacks (:func:`detect_file_backend`), so
    an op that rewrites a volume in place can be handed one — and every such op finds
    out the same confusing way, deep inside an occupancy listing that finds no chunk
    keys and reports the volume as holding no data. The formats are not deficient
    volumes, they are a different thing: one array in one container, with no pyramid to
    downsample and no keyspace whose contents answer "where is the data".

    Called by the ops that *write*; the read-only ones (``info``, ``align-bbox``,
    ``create --like``, ``convert``, ``to-hdf5``) deliberately do not, since inspecting
    and reading these is the whole point of detecting them.
    """
    if fmt in SINGLE_LEVEL_FORMATS:
        raise ValueError(
            f"`{op}` needs a chunked multiscale volume (zarr or neuroglancer-"
            f"precomputed), and {location} is {fmt}: one array in one container, with no "
            f"pyramid and no chunk objects to count. Convert it into a volume first — "
            f"`neu-vol convert {location} <dst> --voxel-size z,y,x` — and run `{op}` on "
            f"that. To go the other way, `neu-vol write` places an HDF5 piece INTO a "
            f"volume.")
    return fmt


def level_spec(volume: str, fmt: str, level: int,
               *, dataset: str | None = None) -> dict[str, Any]:
    """The backend spec for one level of a multiscale volume.

    The two formats address a level differently: zarr v3 puts each level in its own
    subdirectory, while precomputed keeps every scale under one ``info`` and selects
    with ``scale_index``. Everything that opens a level goes through here rather than
    building the path itself.

    A single-container format has exactly one level and any other is an **error**, not
    a level that happens to be missing. That matters because the specs are permissive:
    ``HDF5Backend`` ignores a key it does not know, so a spec carrying ``scale_index=7``
    opened level 0 and reported it as level 7 — which made :func:`existing_levels` probe
    twelve levels, open the same array twelve times, and report a one-level file as a
    twelve-level pyramid of identical shapes.
    """
    if fmt in SINGLE_LEVEL_FORMATS:
        if int(level) != 0:
            raise ValueError(
                f"{volume} is {fmt}, which is a single array and has only level 0; "
                f"level {level} does not exist")
        return location_spec(volume, fmt, dataset=dataset)
    if fmt == "zarr3":
        return {"backend": fmt, "path": f"{volume.rstrip('/')}/{level}"}
    if fmt == "dvid":
        # A DVID level is a downres scale of one instance, addressed the same way
        # precomputed addresses a scale — by index, not by a distinct location.
        return {**location_spec(volume, fmt), "scale_index": int(level)}
    return {"backend": fmt, "path": volume, "scale_index": int(level)}


def describe(volume: str, *, dataset: str | None = None,
             level: int | None = None) -> "Description":
    """Detected format, coordinate metadata and the levels actually present.

    Returns a :class:`~neu_vol.report.Description`, which **is** a dict — subscript it as
    before — that additionally renders itself as the table ``neu-vol info`` prints, so
    ``print(describe(v))`` in a notebook is legible and ``describe(v).frame()`` is a
    DataFrame of the per-level (or per-dataset) rows.

    Works for every format detection reaches, which includes an HDF5 file and a
    directory or glob of 2D slices. Those have one level and may record no coordinate
    metadata at all, so ``meta`` is ``None`` and ``level_voxel_sizes`` empty for them —
    ``shape`` and ``dtype`` come from opening the array itself.

    **An HDF5 file may be a CONTAINER of several arrays, and that is described too.** A
    bag of annotated crops, each keeping its own ``voxel_offset`` so it can be placed
    back, is what an HDF5 file here usually is — so refusing to describe one until a
    ``dataset`` is named would refuse exactly the file you run this on to *find out* the
    names. Instead:

    * ``datasets`` is ``{name: {"shape", "dtype", "chunks", <recorded frame>}}`` for
      every HDF5 file, however many it holds, and ``None`` for other formats. Frame
      values there are **as recorded**, each with its own ``axes``;
    * ``dataset`` is the array the rest of the dict describes, or ``None``;
    * with one candidate — or with ``dataset`` given — that array is resolved and every
      field is filled as usual;
    * with several and none named, ``shape`` / ``dtype`` / ``meta`` / ``level_voxel_sizes``
      are ``None`` and ``levels`` is empty, because thirteen differently-shaped arrays
      have no single answer to any of them. Callers that need one array should say so
      through :func:`require_one_array`, which names the datasets and how to choose.

    Raises ``FileNotFoundError`` if nothing at ``volume`` looks like a volume.
    """
    from .backends.base import open_backend
    from .logs import quiet_reads
    from .report import Description

    # Wrapped because `describe` IS the entry point from a notebook's point of view —
    # there is no `main()` to wrap, and an S3 open logs two `AuthCredentialsProvider`
    # lines at `E` severity per prefix that are not failures, only the two providers that
    # missed before the environment one succeeded. Erik's rule: a user must not have to
    # invoke anything to get legible output, and a manual call is worse than the noise,
    # because forgetting it after a kernel restart looks identical to a broken filter.
    # `quiet_reads` nests and is thread-safe, so the CLI wrapping `main()` too is fine.
    with quiet_reads():
        return _describe(volume, dataset=dataset, level=level)


def _describe(volume: str, *, dataset: str | None = None,
              level: int | None = None) -> "Description":
    from .backends.base import open_backend
    from .report import Description

    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")

    entries = None
    if fmt == "hdf5":
        from .backends.hdf5 import describe_datasets

        entries = describe_datasets(volume)
        if not entries:
            raise FileNotFoundError(f"{volume} contains no 3D+ dataset")
        if dataset is None and len(entries) > 1:
            # Everything below needs ONE array. Reported rather than raised: the listing
            # is the useful answer here, and it is the only way to learn the names.
            return Description(
                location=volume, format=fmt, meta=None, shape=None, dtype=None,
                has_channels=False, level_voxel_sizes=None, levels={},
                spec={"backend": fmt, "path": volume},
                dataset=None, datasets=entries, other_markers=[])

    spec = location_spec(volume, fmt, dataset=dataset)
    if level is not None:
        # `shape`, `dtype` and `meta` then describe THAT scale, which is what a caller
        # deriving a copy's parameters from the source needs. `levels` below still maps
        # the volume's whole declared pyramid — it answers a different question.
        spec["scale_index"] = int(level)
    meta = read_source_metadata(spec)
    level0 = open_backend(meta["data_spec"] if meta else spec)
    shape = tuple(int(s) for s in level0.shape)
    # A single-container format's one level IS `meta`, so it is reused rather than read
    # again: `read_level_voxel_sizes` would reopen the file and re-emit any warning the
    # first read already made (an HDF5 file recording no `axes` warns about the zyx
    # fallback, and hearing it twice reads as two problems).
    if fmt in SINGLE_LEVEL_FORMATS:
        per_level = [tuple(meta["voxel_size"])] if meta else None
    else:
        per_level = read_level_voxel_sizes(spec)
    return Description(
        # What was described. Carried so the result can render its own first line, and
        # so a caller holding one no longer has to remember what it asked about.
        location=volume,
        format=fmt, meta=meta, shape=shape,
        dtype=str(getattr(level0, "dtype", "?")),
        has_channels=bool(meta["has_channels"]) if meta else False,
        level_voxel_sizes=per_level,
        levels=existing_levels(volume, fmt, dataset=dataset),
        # The resolved spec, so a caller that has already paid for the detection can
        # read the volume without repeating it — and so `info` can say WHICH dataset
        # of an HDF5 container it just described.
        spec=spec,
        dataset=spec.get("dataset"),
        # WHICH source scale `shape`/`dtype`/`meta` above describe. Usually 0, but a
        # volume whose `info` declares scales it never wrote resolves to the finest one
        # that stores data, and a caller reading level-0 properties out of `levels`
        # needs to look them up under this rather than under 0.
        source_level=int((meta or {}).get("data_spec", {}).get("scale_index") or 0),
        datasets=entries,
        # One extra small read, against several array opens — worth it to notice
        # a second volume shadowed underneath this one.
        other_markers=other_format_markers(volume, fmt))


def require_one_array(described: Mapping[str, Any], location: str, op: str) -> dict:
    """``described`` if it resolved a single array; a useful error if it did not.

    :func:`describe` describes a multi-array HDF5 container rather than refusing to look
    at one, which means ``shape`` / ``dtype`` / ``meta`` can legitimately be ``None``.
    Anything that goes on to *read* the array has to say so, or it hits a ``TypeError``
    on ``None`` somewhere in its own arithmetic — which says nothing about the container.
    """
    if described.get("shape") is not None:
        # The same object, not a copy: it is a `Description`, and copying it into a plain
        # dict would take the rendering off a value a caller may still want to print.
        return described
    listed = ", ".join(sorted(described.get("datasets") or ()))
    raise ValueError(
        f"`{op}` needs one array and {location} is a container of "
        f"{len(described.get('datasets') or ())}: {listed}. `neu-vol info {location}` "
        f"lists them with their shapes and offsets; pass --dataset where the command "
        f"takes one, or point at a file holding a single array.")


def existing_levels(volume: str, fmt: str, probe: int = 12,
                    *, dataset: str | None = None) -> dict[int, dict]:
    """``{level: {"shape", "chunks", "read_chunks"}}`` for the levels that open.

    Probes upward until one misses. The multiscale group metadata is written at the
    very end of a conversion, so an in-flight volume has levels but no group metadata
    — probing is what makes this work on a run that is still going.

    A single-container format is not probed at all: it has one level by construction,
    and :func:`level_spec` refuses any other, so the loop below would stop after one
    iteration anyway — ``probe`` just stops being a number of store opens.

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
    for i in range(1 if fmt in SINGLE_LEVEL_FORMATS else probe):
        try:
            be = open_backend(level_spec(volume, fmt, i, dataset=dataset))
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
