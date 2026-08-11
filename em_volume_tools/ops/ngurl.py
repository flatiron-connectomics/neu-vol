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

from ..source_metadata import PRECOMPUTED_GZ, describe, detect_backend

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


class VolumeProblem(RuntimeError):
    """A named volume could not be used as a layer source."""


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

    d = describe(volume)
    meta = d["meta"] or {}
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
                   "shape": d["shape"], "format": fmt}


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
                selected: str | None = None) -> dict:
    """A viewer state. ``position_zyx`` is zyx and is reversed here, like every coordinate.

    ``dimensions`` has to agree with the layers or nothing lines up, so it is derived
    from a volume's recorded voxel size rather than assumed.
    """
    from .annotate import output_dimensions

    dims, warning = output_dimensions(voxel_size_zyx, units)
    state: dict[str, Any] = {"dimensions": dims, "layers": list(layers),
                             "layout": layout}
    if position_zyx is not None:
        state["position"] = [float(v) for v in tuple(position_zyx)[::-1]]
    if cross_section_scale is not None:
        state["crossSectionScale"] = float(cross_section_scale)
    if projection_scale is not None:
        state["projectionScale"] = float(projection_scale)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
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
