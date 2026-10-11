"""Read-only backend over one layer of a webknossos dataset.

Addressed by ``wk://DATASET/LAYER``; addressing, credentials and the REST calls live in
``neu_vol/webknossos.py`` — see there for why the token never enters a spec, and why no
mapping (i.e. supervoxels) is the default.

**Presented zyx and origin-anchored, like DVID.** webknossos speaks x,y,z everywhere and
serves voxels Fortran-ordered as ``(c, x, y, z)``, which is the same bytes as a C-ordered
``(z, y, x)`` array for one channel — so the reshape below *is* the axis flip (invariant
ZYX-XYZ). The volume's shape runs from voxel 0 to the layer's bounding box's far corner,
so a voxel index means the same thing here as in the viewer; reads outside the bounding
box are zeros **without a request**, so the empty margin costs nothing.

**Levels are the layer's mags, finest first**, each with its true voxel size (the
dataset's ``scale`` times the mag), never ``2**level`` — webknossos mags may be
anisotropic (``2-2-1``), invariant NM-SPACE.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .. import webknossos as _wk
from .base import Region, register_backend

TAG = "webknossos"

#: The largest box sent in one request, and the read unit reported to the task planner.
#: **Not 512^3**, which `webknossos`' own download uses: the server's throughput is shared
#: (~100-125 MB/s total, measured 2026-10-11, however split), so a request's time is its
#: size times the concurrency, and past a deadline of ~47 s the server returns buckets it
#: has not loaded as ZEROS, with a 200 and every bucket header empty. 12 concurrent 512^3
#: requests took 47-49 s and 9 of 12 came back with thousands of zero 32^3 buckets; 48
#: concurrent 128^3 requests took <= 5 s and lost none, at the same throughput.
REQUEST_CHUNK = (128, 128, 128)

#: A response slower than this is retried rather than trusted, since the zero-filling
#: above is otherwise silent. Normal 128^3 requests take 1-5 s under load.
SLOW_REQUEST_S = 30.0


def _mag(entry: Any) -> tuple[int, int, int]:
    """A layer's ``mags`` entry (``{"mag": [x,y,z]}``, or a bare list) -> xyz ints."""
    m = entry.get("mag") if isinstance(entry, Mapping) else entry
    if isinstance(m, str):
        m = [int(v) for v in m.split("-")]
    return tuple(int(v) for v in m)          # type: ignore[return-value]


def _voxel_size_xyz(info: Mapping[str, Any]) -> tuple[float, float, float]:
    scale = (info.get("dataSource") or {}).get("scale")
    if isinstance(scale, Mapping):
        unit = scale.get("unit", "nanometer")
        if unit not in ("nanometer", "nm"):
            raise ValueError(
                f"dataset {info.get('name')!r} has voxel size in {unit!r}; only "
                f"nanometres are supported (invariant NM-SPACE)")
        scale = scale["factor"]
    return tuple(float(v) for v in scale)    # type: ignore[return-value]


def geometry(info: Mapping[str, Any], layer: str) -> dict:
    """Everything about a layer's array, zyx: per-level shape, box, voxel size; dtype; kind."""
    rec = _wk.layer_record(info, layer)
    element = rec.get("elementClass")
    if element not in _wk.ELEMENT_CLASSES:
        raise ValueError(f"layer {layer!r} has elementClass {element!r}, which this "
                         f"backend cannot read (supported: {sorted(_wk.ELEMENT_CLASSES)})")
    channels = int(rec.get("numChannels") or 1)
    if channels != 1:
        raise ValueError(f"layer {layer!r} has {channels} channels; only single-channel "
                         f"layers are supported")

    bb = rec["boundingBox"]
    lo_xyz = np.array(bb["topLeft"], dtype=np.int64)
    hi_xyz = lo_xyz + np.array([bb["width"], bb["height"], bb["depth"]], dtype=np.int64)
    raw = rec.get("mags") or rec.get("resolutions") or [[1, 1, 1]]
    mags = sorted({_mag(m) for m in raw}, key=lambda m: (max(m), m))
    vox_xyz = np.array(_voxel_size_xyz(info))

    levels = []
    for m in mags:
        mv = np.array(m, dtype=np.int64)
        levels.append({
            "mag": m,
            # Origin-anchored: index 0 is the dataset's voxel 0 at every level.
            "shape": tuple(int(v) for v in (-(-hi_xyz // mv))[::-1]),
            "box_lo": tuple(int(v) for v in (lo_xyz // mv)[::-1]),
            "box_hi": tuple(int(v) for v in (-(-hi_xyz // mv))[::-1]),
            "voxel_size": tuple(float(v) for v in (vox_xyz * mv)[::-1]),
        })
    return {
        "levels": levels,
        "dtype": np.dtype(_wk.ELEMENT_CLASSES[element]),
        "kind": "segmentation" if rec.get("category") == "segmentation" else "image",
        "default_mapping": _wk.default_mapping(rec),
        "largest_segment_id": _bigint(rec.get("largestSegmentId")),
    }


def _bigint(value: Any) -> int | None:
    """webknossos sends a uint64 as ``{"customJsonEncoding": "bigint", "value": "123"}``
    (JSON numbers stop being exact at 2**53); older servers send a bare number."""
    if isinstance(value, Mapping):
        value = value.get("value")
    return None if value is None else int(value)


class WebknossosBackend:
    """Read-only ``(z, y, x)`` view over one mag of a webknossos layer."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        if not spec.get("dataset_id"):
            # A spec straight from a URL: resolve here so a bare backend works too. The
            # convert path always arrives resolved, via `read_source_metadata`.
            self._spec["dataset_id"] = _wk.resolve_dataset_id(str(spec["dataset"]))
        self._creds = _wk.credentials()
        info = _wk.dataset_info(self._spec["dataset_id"], creds=self._creds)
        self._layer = str(spec["layer"])
        self._mapping = spec.get("mapping")
        geom = geometry(info, self._layer)
        level = int(spec.get("scale_index", 0))
        if not 0 <= level < len(geom["levels"]):
            raise ValueError(
                f"{_wk.spec_url(spec)} has {len(geom['levels'])} mags "
                f"({', '.join('-'.join(map(str, lv['mag'])) for lv in geom['levels'])}); "
                f"level {level} was requested")
        lv = geom["levels"][level]
        self._mag = lv["mag"]
        self._shape = lv["shape"]
        self._lo, self._hi = lv["box_lo"], lv["box_hi"]
        self._dtype = geom["dtype"]
        self.voxel_size = lv["voxel_size"]
        self._url = (f"{_wk.data_api(info['dataStore']['url'])}/datasets/"
                     f"{self._spec['dataset_id']}/layers/{self._layer}/data")

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "WebknossosBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return REQUEST_CHUNK

    def read_region(self, region: Region) -> np.ndarray:
        start = [int(s.start or 0) for s in region]
        stop = [int(s.stop if s.stop is not None else n) for s, n in zip(region, self._shape)]
        out = np.zeros([b - a for a, b in zip(start, stop)], dtype=self._dtype)
        lo = [max(a, b) for a, b in zip(start, self._lo)]
        hi = [min(a, b) for a, b in zip(stop, self._hi)]
        if any(h <= l for l, h in zip(lo, hi)):
            return out                      # wholly outside the bounding box: no request
        # At most REQUEST_CHUNK per request, on its grid, so a big read cannot outrun the
        # server's deadline (see REQUEST_CHUNK).
        grids = [range(l - l % c, h, c) for l, h, c in zip(lo, hi, REQUEST_CHUNK)]
        for z in grids[0]:
            for y in grids[1]:
                for x in grids[2]:
                    a = [max(v, l) for v, l in zip((z, y, x), lo)]
                    b = [min(v + c, h) for v, c, h in zip((z, y, x), REQUEST_CHUNK, hi)]
                    out[tuple(slice(p - s, q - s) for p, q, s in zip(a, b, start))] = \
                        self._fetch(a, b)
        return out

    def _fetch(self, lo, hi) -> np.ndarray:
        """One request for the zyx box ``[lo, hi)`` at this mag."""
        import time

        size = [h - l for l, h in zip(lo, hi)]
        mz, my, mx = self._mag[::-1]
        t0 = time.monotonic()
        body, headers = _wk._request(self._url, token=self._creds["token"], query={
            "mag": "-".join(str(v) for v in self._mag),
            # The position is in MAG-1 voxels, the extent in this mag's voxels — the
            # convention of webknossos' own download.
            "x": lo[2] * mx, "y": lo[1] * my, "z": lo[0] * mz,
            "width": size[2], "height": size[1], "depth": size[0],
            "mappingName": self._mapping,
        })
        elapsed = time.monotonic() - t0
        if elapsed > SLOW_REQUEST_S:
            raise RuntimeError(
                f"UNAVAILABLE: webknossos took {elapsed:.0f} s for a {tuple(size)} box "
                f"(> {SLOW_REQUEST_S:.0f} s); a slow response may hold buckets the server "
                f"zero-filled without saying so — retrying. Persistent? Use fewer workers.")
        # The server's own report of buckets it could not load, which it otherwise serves
        # ZERO-FILLED with a 200. Not sufficient on its own: the deadline zero-filling under
        # REQUEST_CHUNK leaves these headers empty, hence the timing check below too.
        # Raising UNAVAILABLE makes the block retry (invariant FAULT-POLICY).
        lower = {k.lower(): v for k, v in headers.items()}
        for name in ("failure-bucket-indices", "missing-buckets"):
            bad = lower.get(name)
            if bad not in (None, "", "[]"):
                raise RuntimeError(f"UNAVAILABLE: webknossos reported {name} {bad[:200]} "
                                   f"for {_wk.spec_url(self._spec)}")
        expected = int(np.prod(size)) * self._dtype.itemsize
        if len(body) != expected:
            raise RuntimeError(
                f"UNAVAILABLE: webknossos returned {len(body)} bytes for a "
                f"{tuple(size)} {self._dtype} box; expected {expected}")
        # Fortran (x, y, z) == C (z, y, x): this reshape is the axis flip.
        return np.frombuffer(body, dtype=self._dtype).reshape(size)

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("webknossos layers are read-only here; this package pulls data "
                        "out of webknossos, it does not upload")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, WebknossosBackend.open)
