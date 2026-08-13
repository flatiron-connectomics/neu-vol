"""Assemble a neuroglancer viewer state and encode it into a shareable URL.

Neuroglancer keeps its entire state in the URL fragment, so a link *is* the state: which
volumes are loaded, where the view is, which segments are selected. Building one by hand
means getting the source scheme, the coordinate space and the layer shapes right, which
is exactly the sort of thing that fails silently — a wrong `dimensions` block puts every
layer in the wrong place and still loads.

So this reads what it can from the volumes themselves. The source scheme comes from
``detect_backend``, and the coordinate space from the first layer's recorded voxel size,
because a state's ``dimensions`` must agree with the data or nothing lines up.

**The fragment is never sent to a server.** Everything after ``#!`` stays in the browser,
so a link carries no data anywhere — but it does mean the whole state travels in the URL,
and a large inline annotation layer makes for a long one (see :data:`LONG_URL`).

Composes with ``bboxes-json``: that command writes a layer, ``--layer`` inlines it here.
Neither needs to know about the other beyond a file of JSON.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Sequence

from ..source_metadata import PRECOMPUTED_GZ, detect_backend, read_source_metadata

DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"

# Where URLs start being awkward to paste into things — mail clients wrap, some chat
# tools truncate, and a few proxies cap the request line even though a fragment is never
# sent. Not an error, just worth saying out loud.
LONG_URL = 8000

# How neuroglancer addresses each backend. The classic `scheme://` form rather than the
# newer `kvstore|adapter:` pipeline syntax: both work in current builds, only this one
# works in older ones, and a link is the thing most likely to be opened by someone
# running a different viewer.
SCHEME = {
    "neuroglancer_precomputed": "precomputed",
    PRECOMPUTED_GZ: "precomputed",
    "zarr3": "zarr",
}

# Neuroglancer's own layout names. Passed through rather than validated against a
# hardcoded list would be friendlier to new ones, but a typo here produces a viewer that
# silently falls back, which is worse than being told.
LAYOUTS = ("4panel", "xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "3d")


# A nominal viewport edge in pixels, used only to turn "fit the volume" into a zoom
# number. The real panel size is not knowable here — it depends on the window and the
# layout — so this errs large, which errs zoomed OUT. Being a factor of two off is
# harmless; starting at the origin corner at one voxel per pixel is not.
NOMINAL_VIEWPORT_PX = 1000

# Leave a little space around the volume rather than cropping it to the panel edge.
FIT_MARGIN = 1.15


class VolumeProblem(RuntimeError):
    """A named volume could not be used as a layer source."""


def volume_extent(volume: str, fmt: str) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` in level-0 voxels, or None if not determinable.

    Read as cheaply as the format allows: precomputed carries every scale's ``size`` and
    ``voxel_offset`` in the one ``info`` this has already fetched, so it costs nothing,
    while zarr needs the level-0 array's own metadata — one open, not the every-level
    probe that :func:`describe` does.
    """
    from ..location import read_json

    if str(fmt).startswith("neuroglancer_precomputed"):
        info = read_json(volume.rstrip("/") + "/info") or {}
        scales = info.get("scales")
        if not scales:
            return None
        finest = min(scales, key=lambda s: tuple(s["resolution"]))
        size = [int(v) for v in finest["size"]][::-1]                    # xyz -> zyx
        off = [int(v) for v in finest.get("voxel_offset", [0, 0, 0])][::-1]
        return tuple(size), tuple(off)

    meta = read_source_metadata({"backend": fmt, "path": volume})
    if not meta:
        return None
    try:
        from ..backends.base import open_backend

        shape = tuple(int(s) for s in open_backend(meta["data_spec"]).shape)
    except Exception:
        return None
    spatial = shape[-3:] if len(shape) > 3 else shape       # drop a channel axis
    return spatial, (0, 0, 0)


def annotation_extent(layers: Sequence[dict]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` covering the annotations in ``layers``.

    The fallback when every layer came from a file: a bounding-box layer knows where the
    data is even though no volume was named, and framing those boxes is a better opening
    view than the origin.
    """
    lo = [None, None, None]
    hi = [None, None, None]
    for layer in layers:
        for ann in layer.get("annotations", []) or []:
            pts = [ann[k] for k in ("pointA", "pointB", "point", "center") if k in ann]
            for p in pts:
                for a in range(3):
                    v = float(p[2 - a])                     # stored xyz, wanted zyx
                    lo[a] = v if lo[a] is None else min(lo[a], v)
                    hi[a] = v if hi[a] is None else max(hi[a], v)
    if any(v is None for v in lo):
        return None
    extent = tuple(max(1.0, hi[a] - lo[a]) for a in range(3))
    return extent, tuple(lo)


def default_view(extent_zyx: Sequence[float],
                 offset_zyx: Sequence[float] = (0, 0, 0)) -> tuple[list, float, float]:
    """``(centre_zyx, cross_section_scale, projection_scale)`` framing a whole volume.

    Neuroglancer with no ``position`` opens at the origin **corner** and with no
    ``crossSectionScale`` opens at one voxel per pixel, which on a 13750-voxel volume is
    a view of its empty edge. Both scales are in canonical voxels — per viewport pixel
    for the cross sections, across the viewport for the projection — so fitting is a
    division by a nominal panel size.
    """
    centre = [float(o) + float(e) / 2 for o, e in zip(offset_zyx, extent_zyx)]
    span = max(float(e) for e in extent_zyx) * FIT_MARGIN
    return centre, span / NOMINAL_VIEWPORT_PX, span


def volume_layer(volume: str, *, kind: str | None = None, name: str | None = None,
                 segments: Sequence[int] | None = None,
                 opacity: float | None = None) -> tuple[dict, dict]:
    """One layer for ``volume``, plus what its metadata says about the frame.

    ``kind`` overrides the volume's own record of whether it is an image or a
    segmentation; without it the recorded value decides, because a segmentation shown as
    an image is a grey mush and the mistake is easy to miss on a small ROI.
    """
    volume = volume.rstrip("/")
    fmt = detect_backend(volume)
    if fmt is None:
        raise VolumeProblem(f"no volume found at {volume}")
    scheme = SCHEME.get(fmt)
    if scheme is None:
        raise VolumeProblem(f"{volume} is {fmt}, which neuroglancer cannot read")

    # `read_source_metadata`, NOT `describe`: this needs the voxel size, the units and
    # the recorded kind, all of which are in `info`. `describe` additionally OPENS EVERY
    # LEVEL and probes for a foreign marker — the expensive tier documented in
    # source_metadata — and a link needs none of it. Two volumes here meant ~20 store
    # opens for numbers already read.
    meta = read_source_metadata({"backend": fmt, "path": volume}) or {}
    resolved = kind or meta.get("kind") or "image"
    layer: dict[str, Any] = {
        "type": "segmentation" if resolved == "segmentation" else "image",
        "name": name or volume.rsplit("/", 1)[-1],
        "source": f"{scheme}://{volume}",
    }
    if segments:
        # Strings, not ints: neuroglancer segment ids are uint64 and JSON numbers are
        # doubles, so a real id above 2**53 would arrive rounded.
        layer["segments"] = [str(int(s)) for s in segments]
    if opacity is not None:
        layer["opacity"] = float(opacity)
    return layer, {"voxel_size": meta.get("voxel_size"), "units": meta.get("units"),
                   "format": fmt}


def load_layer(path: str) -> dict:
    """A layer (or a whole state's worth of layers) from a JSON file.

    Accepts what ``bboxes-json`` writes either way round — a bare layer object, or a
    full state whose ``layers`` are taken — so it does not matter which of the two the
    caller happened to generate.
    """
    from ..location import read_bytes

    raw = read_bytes(path)
    if raw is None:
        raise VolumeProblem(f"no such file: {path}")
    obj = json.loads(raw)
    if isinstance(obj, dict) and "layers" in obj:
        return obj["layers"]
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict) or "type" not in obj:
        raise VolumeProblem(
            f"{path} is not a neuroglancer layer or state: expected an object with a "
            f"'type' (a layer) or a 'layers' array (a state)")
    return [obj]


def build_state(layers: Sequence[dict], *, voxel_size_zyx=None, units: str | None = None,
                position_zyx: Sequence[float] | None = None,
                layout: str = "4panel",
                cross_section_scale: float | None = None,
                projection_scale: float | None = None,
                selected: str | None = None,
                show_slices: bool | None = None,
                frame: tuple | None = None) -> dict:
    """A viewer state. ``position_zyx`` is zyx and is reversed here, like every coordinate.

    ``dimensions`` has to agree with the layers or nothing lines up, so it is derived
    from a volume's recorded voxel size rather than assumed.

    ``frame`` is an ``(extent_zyx, offset_zyx)`` pair used to fill in whichever of the
    position and the two zooms the caller did not specify — see :func:`default_view`.
    Without it the state carries none of the three and neuroglancer opens at the origin
    corner, fully zoomed in.

    ``show_slices=False`` sets neuroglancer's ``showSlices``, which hides the cross-section
    planes *inside the 3D panel* — worth it when the point of the link is meshes or
    skeletons, which the slices otherwise sit across. It does not touch the 2D panels; use
    ``layout="3d"`` for that. Left as ``None`` the key is omitted and the viewer's own
    default (shown) applies.
    """
    from .annotate import output_dimensions

    dims, warning = output_dimensions(voxel_size_zyx, units)
    state: dict[str, Any] = {"dimensions": dims, "layers": list(layers),
                             "layout": layout}

    fitted = default_view(*frame) if frame else (None, None, None)
    position = position_zyx if position_zyx is not None else fitted[0]
    cross = cross_section_scale if cross_section_scale is not None else fitted[1]
    projection = projection_scale if projection_scale is not None else fitted[2]

    if position is not None:
        state["position"] = [float(v) for v in tuple(position)[::-1]]
    if cross is not None:
        state["crossSectionScale"] = float(cross)
    if projection is not None:
        state["projectionScale"] = float(projection)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
    if show_slices is not None:
        # Written only when asked for, like the position and the two zooms above: a state
        # that carries no `showSlices` gets neuroglancer's own default (true), which is
        # what someone opening the link would otherwise expect.
        state["showSlices"] = bool(show_slices)
    return state, warning


def state_url(state: dict, viewer: str = DEFAULT_VIEWER) -> str:
    """``<viewer>#!<url-encoded state>``.

    Percent-encoded rather than raw: neuroglancer accepts both, but a raw fragment
    survives only until something in the chain — a chat client, a wiki, a shell — decides
    what to do with the quotes and braces in it.
    """
    encoded = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
    return f"{viewer.rstrip('/')}/#!{encoded}"


def parse_url(url: str) -> dict:
    """The state back out of a neuroglancer URL, for inspecting or editing one."""
    if "#!" not in url:
        raise ValueError("not a neuroglancer state URL: no '#!' fragment")
    fragment = url.split("#!", 1)[1]
    return json.loads(urllib.parse.unquote(fragment))
