"""Read each pyramid level's shape, voxel size and origin from the source metadata.

Every level must be described by its **own voxel size in nm**, never by an assumed
``2**scale`` factor — real pyramids are anisotropic and non-standard downsample factors
are common. That only helps if the number comes from the data, so this reads it:

- **precomputed**: each entry of ``info["scales"]`` carries its own ``resolution``
  (xyz nm), ``size`` (xyz voxels) and optional ``voxel_offset`` (xyz voxels of THIS
  level, so the nm origin is the product).
- **zarr / OME-NGFF**: each multiscale dataset carries its own ``scale`` coordinate
  transformation, and an optional ``translation`` which is **already physical** and must
  not be multiplied by the voxel size.

Each level comes back carrying a :class:`~neu_lib.Frame`, so "what does a voxel here
mean in nm" has one answer and callers stop rebuilding one from a bare voxel size.
**The offsets used to be dropped**, which made every level claim to start at nm zero —
correct for a volume written from the origin, wrong for anything cropped, and silent
either way.

Scales are ordered finest-first, so index 0 is full resolution and matches
``scale_index`` in a precomputed spec.

This lives in neu-vol because it opens a store. It was in neu-morpho, which meant
neu-draw imported a meshing package to learn a volume's voxel size.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from neu_lib import Frame, ScaleInfo


def read_scales(spec: str | Mapping[str, Any]) -> list[ScaleInfo]:
    """All pyramid levels of a source, finest first. Raises if metadata is absent.

    Store logging is filtered for the duration: this is called straight from notebooks
    (it is why the function moved down here from neu-morpho), and an S3 open emits two
    benign `AuthCredentialsProvider` lines per prefix. See `source_metadata.describe`.
    """
    from .logs import quiet_reads
    from .source_metadata import detect_backend

    with quiet_reads():
        return _read_scales(spec)


def _read_scales(spec: str | Mapping[str, Any]) -> list[ScaleInfo]:
    from .source_metadata import detect_backend

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = _kvstore(spec)
    # detect_backend takes a *location*, not a spec — handing it the spec dict
    # loses the kvstore driver and tensorstore then refuses to open it.
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return _precomputed_scales(kv)
    if backend in ("zarr3", "zarr2"):
        return _zarr_scales(kv)
    raise ValueError(
        f"cannot read scale metadata for backend {backend!r}; "
        "pass voxel sizes explicitly instead")


def _kvstore(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Directory kvstore for a spec (with a trailing slash, so keys append)."""
    from .location import spec_kvstore

    kv = dict(spec_kvstore(spec))
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    return kv


def _read_key(kv: Mapping[str, Any], key: str) -> bytes | None:
    """Read one metadata key through ``location``, never ``ts.KvStore.open`` directly.

    Opening here was a fourth store-opening path that skipped both of the things
    ``location`` exists to guarantee: the **per-prefix store cache**, so every call paid
    a fresh open (visible as two S3 credential-provider log lines and a round trip per
    ``read_scales``), and **``ensure_credentials``**, which makes it a latent 403 in any
    process that has not otherwise bootstrapped (invariant 8). Same class of bug as the
    one already fixed in ``source_metadata._read_key``.
    """
    from .location import read_bytes

    return read_bytes(kv, key)


def _precomputed_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    raw = _read_key(kv, "info")
    if raw is None:
        raise ValueError(f"no precomputed 'info' at {kv}")
    scales = json.loads(raw)["scales"]
    # finest first; `scale_index` in a tensorstore spec indexes this same order
    ordered = sorted(scales, key=lambda s: tuple(s["resolution"]))
    out = []
    for i, s in enumerate(ordered):
        voxel = tuple(float(v) for v in s["resolution"][::-1])           # xyz -> zyx
        # `voxel_offset` is in THIS level's voxels, so the nm origin is the product.
        # Absent on a volume written from the origin, which is why dropping it went
        # unnoticed: every test volume started at zero.
        off = tuple(int(v) for v in s.get("voxel_offset", (0, 0, 0))[::-1])
        out.append(ScaleInfo(
            index=i,
            shape=tuple(int(v) for v in s["size"][::-1]),
            frame=Frame(voxel_size_nm=voxel,
                        origin_nm=tuple(o * v for o, v in zip(off, voxel))),
            key=str(s.get("key", ""))))
    return out


def _zarr_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    from .location import join

    raw = _read_key(kv, "zarr.json")
    if raw is None:
        raise ValueError(f"no 'zarr.json' at {kv} (zarr v2 OME groups not supported here)")
    meta = json.loads(raw)
    ome = meta.get("attributes", {}).get("ome")
    if meta.get("node_type") != "group" or not ome:
        raise ValueError("source is a bare zarr array, not an OME multiscale group; "
                         "pass voxel sizes explicitly")
    ms = ome["multiscales"][0]
    axes = ms["axes"]
    spatial = [i for i, a in enumerate(axes) if a.get("type") == "space"]

    out = []
    for i, ds in enumerate(ms["datasets"]):
        transforms = ds["coordinateTransformations"]
        scale = next(t["scale"] for t in transforms if t["type"] == "scale")
        # OME's optional `translation`, the zarr spelling of precomputed's
        # `voxel_offset` — except it is already in PHYSICAL units, so unlike the
        # precomputed case it must not be multiplied by the voxel size.
        translation = next((t["translation"] for t in transforms
                            if t["type"] == "translation"), None)
        sub = _read_key(join(kv, ds["path"] + "/"), "zarr.json")
        shape = json.loads(sub)["shape"] if sub else None
        out.append(ScaleInfo(
            index=i,
            shape=tuple(int(shape[a]) for a in spatial) if shape else (0, 0, 0),
            frame=Frame(
                voxel_size_nm=tuple(float(scale[a]) for a in spatial),
                origin_nm=(tuple(float(translation[a]) for a in spatial)
                           if translation else (0.0, 0.0, 0.0))),
            key=str(ds["path"])))
    return out


def scale_spec(spec: str | Mapping[str, Any], scale_index: int) -> dict:
    """A read spec pinned to one pyramid level.

    precomputed selects the level with ``scale_index``; zarr addresses the level's
    subgroup by path, so the two need different treatment.

    **Always go through this rather than hand-writing a spec.** The key is
    ``scale_index``, and an unrecognised one is *silently ignored*: a spec carrying
    ``{"scale": 2}`` opens at full resolution and reports the scale-0 shape, so the
    coordinates you pass are then interpreted 4x too fine and read the wrong place
    entirely. That fails as empty data, not as an error.
    """
    from .source_metadata import detect_backend
    from .location import join

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = _kvstore(spec)
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return {"backend": "neuroglancer_precomputed", "kvstore": kv, "scale_index": scale_index}
    scales = read_scales(spec)
    if not 0 <= scale_index < len(scales):
        raise IndexError(f"scale {scale_index} out of range (source has {len(scales)})")
    return {"backend": backend or "zarr3", "kvstore": join(kv, scales[scale_index].key)}


def describe_scales(spec: str | Mapping[str, Any]) -> str:
    """Human-readable pyramid listing — print this before committing to a run.

    Named apart from :func:`neu_vol.describe`, which is a different and much more
    expensive thing: that one OPENS EVERY LEVEL to probe for a foreign marker, while
    this reads one ``info`` or ``zarr.json``.
    """
    scales = read_scales(spec)
    lines = [f"{'scale':>5}  {'shape (z,y,x)':>24}  {'voxel nm (z,y,x)':>22}  key"]
    for s in scales:
        lines.append(f"{s.index:>5}  {str(s.shape):>24}  "
                     f"{str(tuple(round(v, 3) for v in s.voxel_size)):>22}  {s.key}")
    return "\n".join(lines)
