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
from typing import Any, Mapping, Sequence

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

#: The `@type` a precomputed annotation source declares. Checked rather than assumed,
#: because an annotation source and a volume are both addressed `precomputed://` and both
#: have an `info` — pointing --annotations at a volume would otherwise produce a layer that
#: loads and draws nothing.
ANNOTATION_TYPE = "neuroglancer_annotations_v1"

#: Shaders for a precomputed annotation layer, by name. A shader lives in the viewer state,
#: not in the source, so a link is the only place it can be shipped — which is why these are
#: here rather than in whatever wrote the annotations.
#:
#: Each entry names the properties it reads. `pick_shader` will not apply one whose
#: properties the source does not declare: a shader referencing a `prop_` that does not
#: exist fails to compile and neuroglancer then draws NOTHING, with the error only visible
#: in the layer's shader tab.
#: The endpoint markers, not the line, are what stays legible. A synapse is a few hundred
#: nanometres long, so at any zoom that shows more than one the line is sub-pixel — and a line
#: drawn in a blend of the two endpoint colours then swamps the markers and reads as a single
#: flat colour. So `show_pre`/`show_post` gate the two markers independently and the line is
#: drawn only when BOTH are on, which is also what makes one source usable as two layers.
_SYNAPSE_SHADER = """\
#uicontrol bool show_pre checkbox(default=true)
#uicontrol bool show_post checkbox(default=true)

#uicontrol float pre_size slider(min=0.0, max=20.0, default=6.0)
#uicontrol float post_size slider(min=0.0, max=20.0, default=4.0)

#uicontrol vec3 pre_color color(default="#ff2000")
#uicontrol vec3 post_color color(default="#00c0ff")
#uicontrol vec3 line_color color(default="#ffffff")

#uicontrol float min_conf slider(min=0.0, max=1.0, default=0.0, step=0.01)

void main() {
  // NaN fails every comparison, so an UNKNOWN confidence is never hidden here. Unknown is
  // not the same as low, and this is where that distinction survives into the viewer.
  if (prop_conf_pre() < min_conf || prop_conf_post() < min_conf) discard;

  setEndpointMarkerColor(vec4(pre_color, 1.0), vec4(post_color, 1.0));
  setLineColor(line_color);

  // The line is only meaningful when both ends are shown; otherwise it points at something
  // deliberately hidden, and at these scales its colour would drown the markers.
  if (show_pre && show_post) {
    setLineWidth(1.0);
    setEndpointMarkerSize(pre_size, post_size);
  } else if (show_pre) {
    setLineWidth(0.0);
    setEndpointMarkerSize(pre_size, 0.0);
  } else if (show_post) {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, post_size);
  } else {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, 0.0);
  }
}
"""

ANNOTATION_SHADERS: dict[str, dict[str, Any]] = {
    "synapse": {
        "properties": ("conf_pre", "conf_post"),
        "doc": "pre/post endpoint markers, independently toggleable, with a confidence "
               "threshold",
        "source": _SYNAPSE_SHADER,
    },
}

#: `shaderControls` overrides for the two halves of a split pair. Set in the STATE rather than
#: by generating two shaders, so both layers carry the same code and a user editing one can see
#: exactly which control the other flipped.
SPLIT_CONTROLS = {
    "pre": {"show_pre": True, "show_post": False},
    "post": {"show_pre": False, "show_post": True},
}


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


def read_annotation_info(source: str) -> dict:
    """The ``info`` of a precomputed annotation source, refusing anything else.

    A volume and an annotation source are both ``precomputed://`` with an ``info`` at the
    root, so nothing about the URL distinguishes them — and an annotation layer pointed at a
    volume loads happily and draws nothing at all. The ``@type`` is the only honest check.
    """
    from ..location import read_json

    source = source.rstrip("/")
    info = read_json(source, "info")
    if info is None:
        raise VolumeProblem(f"no info at {source}")
    if info.get("@type") != ANNOTATION_TYPE:
        raise VolumeProblem(
            f"{source} is {info.get('@type') or 'not an annotation source'}, not "
            f"{ANNOTATION_TYPE}. Volumes go to --image or --seg; --annotations wants a "
            f"precomputed ANNOTATION source (what `em-annot annotation-source` writes).")
    return info


def pick_shader(info: Mapping[str, Any], name: str | None) -> tuple[str | None, str | None]:
    """``(shader source, why)`` for an annotation layer.

    ``name`` may be a built-in name, a path to a file of GLSL, ``"none"``, or ``None`` to
    choose automatically — which means the first built-in whose properties the source
    actually declares. Automatic rather than a fixed default because a shader naming a
    ``prop_`` the source lacks does not degrade: it fails to compile and the layer draws
    nothing, with the error only in the shader tab.
    """
    declared = {p["id"] for p in info.get("properties", []) or []}
    if name == "none":
        return None, None
    if name is None:
        for key, entry in ANNOTATION_SHADERS.items():
            if declared.issuperset(entry["properties"]):
                return entry["source"], f"auto-selected {key!r} ({entry['doc']})"
        return None, ("no built-in shader matches this source's properties "
                      + (f"({', '.join(sorted(declared))})" if declared else "(none)"))
    if name in ANNOTATION_SHADERS:
        entry = ANNOTATION_SHADERS[name]
        missing = sorted(set(entry["properties"]) - declared)
        if missing:
            raise VolumeProblem(
                f"shader {name!r} reads {', '.join(missing)}, which this source does not "
                f"declare. A shader naming a property that is absent fails to compile and "
                f"the layer draws nothing. Declared: "
                + (", ".join(sorted(declared)) or "no properties"))
        return entry["source"], f"built-in {name!r}"

    from ..location import read_bytes

    raw = read_bytes(name)
    if raw is None:
        raise VolumeProblem(
            f"--annotation-shader {name!r} is neither a built-in name ("
            + ", ".join(ANNOTATION_SHADERS) + ", none) nor a readable file")
    return raw.decode(), f"from {name}"


def annotation_layer(source: str, *, name: str | None = None,
                     shader: str | None = None,
                     linked_segmentation: str | None = None,
                     filter_by_segmentation: bool = True,
                     filter_relationships: Sequence[str] | None = None,
                     controls: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """One annotation layer for a precomputed annotation source, plus its frame.

    ``linked_segmentation`` is the name of a segmentation layer in the same state, and it is
    **what makes the relationship index do anything in the viewer**. The source's
    ``relationships`` are keyed on segment id, but neuroglancer only consults them once each
    relationship is bound to a layer whose selected segments it can read; without the binding
    the layer draws every annotation and "this body's synapses" is not available at all.

    ``filter_relationships`` narrows which of them the filter uses. Every relationship stays
    *bound* regardless — binding is what makes a relationship usable, filtering is what decides
    whether it restricts the view — and filtering on a subset is what turns one source into
    "this body's outputs" and "this body's inputs" as separate layers.
    """
    source = source.rstrip("/")
    info = read_annotation_info(source)
    layer: dict[str, Any] = {
        "type": "annotation",
        "name": name or source.rsplit("/", 1)[-1],
        "source": f"precomputed://{source}",
    }
    shader_source, why = pick_shader(info, shader)
    if shader_source:
        layer["shader"] = shader_source
    if controls:
        layer["shaderControls"] = dict(controls)

    relationships = [r["id"] for r in info.get("relationships", []) or []]
    if linked_segmentation and relationships:
        layer["linkedSegmentationLayer"] = {r: linked_segmentation for r in relationships}
        if filter_by_segmentation:
            picked = [r for r in (filter_relationships or relationships)
                      if r in relationships]
            if filter_relationships and not picked:
                raise VolumeProblem(
                    f"none of {', '.join(filter_relationships)} is a relationship of "
                    f"{source}; it declares " + (", ".join(relationships) or "none"))
            layer["filterBySegmentation"] = picked

    return layer, {"info": info, "shader": why, "relationships": relationships}


#: The relationship whose *own* end of the line is the given side, and the shader controls that
#: show only that end. Filtering on `body_pre` gives the selected body's OUTPUTS; on `body_post`
#: its INPUTS. Two layers on one source, which is how the reference dataset presents them.
SPLIT_SIDES = (("pre", "body_pre"), ("post", "body_post"))


def annotation_layer_pair(source: str, *, name: str | None = None,
                          shader: str | None = None,
                          linked_segmentation: str | None = None,
                          filter_by_segmentation: bool = True) -> tuple[list[dict], dict]:
    """Two layers on one source: the selected body's outputs, then its inputs.

    A single layer filtered on both relationships answers "every synapse touching this body",
    which conflates the two directions — and drawn together the endpoint markers overlap at any
    zoom that shows more than a few. Splitting costs nothing on the store (one source, two
    layers) and each half then has its own visibility, colour and marker size.
    """
    base = (name or source.rstrip("/").rsplit("/", 1)[-1])
    layers, info = [], None
    for side, relationship in SPLIT_SIDES:
        layer, info = annotation_layer(
            source, name=f"{base}-{side}", shader=shader,
            linked_segmentation=linked_segmentation,
            filter_by_segmentation=filter_by_segmentation,
            filter_relationships=[relationship], controls=SPLIT_CONTROLS[side])
        layers.append(layer)
    return layers, info


def annotation_source_extent(info: Mapping[str, Any]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` from an annotation source's declared bounds.

    The format requires ``lower_bound``/``upper_bound``, so an annotations-only link can
    still open framed on its data rather than at the origin corner — the same job
    :func:`volume_extent` does for a volume.
    """
    lower, upper = info.get("lower_bound"), info.get("upper_bound")
    if not lower or not upper or len(lower) != 3 or len(upper) != 3:
        return None
    lo = [float(v) for v in reversed(lower)]            # stored xyz, wanted zyx
    hi = [float(v) for v in reversed(upper)]
    return tuple(max(1.0, h - l) for l, h in zip(lo, hi)), tuple(lo)


def annotation_source_voxel_size(info: Mapping[str, Any]) -> tuple | None:
    """Level-0 voxel size in nm, zyx, from the source's ``dimensions``.

    ``dimensions`` is ``{axis: [scale, unit]}`` in SI, so metres become nanometres here.
    Lets an annotations-only link establish the viewer's frame without --voxel-size.
    """
    dims = info.get("dimensions")
    if not isinstance(dims, Mapping):
        return None
    try:
        scales = [dims[axis] for axis in ("x", "y", "z")]
    except KeyError:
        return None
    out = []
    for scale, unit in scales:
        if unit != "m":
            return None
        out.append(float(scale) * 1e9)
    return tuple(reversed(out))                        # xyz read, zyx returned


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
