"""Read-only backend over a stack of 2D images or a multipage TIFF.

Sources:
  - a glob / directory of ordered 2D image files (TIFF/PNG), one per Z slice, or
  - a single multipage TIFF.

Presents the stack as a 3D ``(z, y, x)`` volume. Read-only: an ingest *source*,
never a write target. v1 assumes single-channel 2D slices.
"""

from __future__ import annotations

import glob as _glob
import os
import re
from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "image_stack"

#: Slice extensions this reader accepts. Public because it is also the *detection*
#: vocabulary: ``source_metadata.detect_file_backend`` asks whether a directory holds
#: any of these, and the two must not drift — a directory detected as a stack whose
#: files this reader then filters out would fail with "no image files matched".
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png")


def _natural_key(s: str) -> list:
    """Sort key so ``z2`` precedes ``z10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_files(source: str) -> list[str]:
    if _glob.has_magic(source):
        files = _glob.glob(source)
    elif os.path.isdir(source):
        files = [
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"no image files matched {source!r}")
    return sorted(files, key=_natural_key)


def _lift_pillow_bomb_guard() -> None:
    """Let Pillow open slices larger than its 179 Mpx "decompression bomb" limit.

    That guard exists to stop a small hostile file from expanding to gigabytes in a
    web service. An EM section is legitimately far past it — a 16648x13544 slice is
    225 Mpx — so with the guard in place every read of a real dataset fails with
    ``DecompressionBombError`` and no conversion is possible.

    This is a deliberate trade for trusted, local scientific data, and it is scoped to
    this reader rather than set process-wide by a caller who may not expect it.
    """
    try:
        from PIL import Image
    except ImportError:      # tifffile-only install; nothing to lift
        return
    Image.MAX_IMAGE_PIXELS = None


def _read_image(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        arr = tifffile.imread(path)
    else:
        import imageio.v3 as iio

        _lift_pillow_bomb_guard()      # imageio delegates PNG decoding to Pillow
        arr = iio.imread(path)
    if arr.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2D single-channel slice, got shape {arr.shape} "
            "(multi-channel stacks are not supported in v1)"
        )
    return arr


class ImageStackBackend:
    """Lazy 3D ``(z, y, x)`` view over ordered 2D files or a multipage TIFF."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        source = str(spec["source"])
        files = _list_files(source)

        # Single TIFF with multiple pages -> multipage stack.
        self._multipage = False
        self._files = files
        if len(files) == 1 and files[0].lower().endswith((".tif", ".tiff")):
            import tifffile

            with tifffile.TiffFile(files[0]) as tf:
                shp = tf.series[0].shape
                self._dtype = np.dtype(tf.series[0].dtype)
                if len(shp) == 3:
                    self._multipage = True
                    self._nz, self._ny, self._nx = (int(x) for x in shp)
                elif len(shp) == 2:
                    self._nz = 1
                    self._ny, self._nx = int(shp[0]), int(shp[1])
                else:
                    raise ValueError(f"{files[0]}: unsupported series shape {shp}")
            self._path = files[0]
            return

        # One 2D image per file.
        first = _read_image(files[0])
        self._dtype = first.dtype
        self._ny, self._nx = int(first.shape[0]), int(first.shape[1])
        self._nz = len(files)

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "ImageStackBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._nz, self._ny, self._nx)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        # Natural read granularity: one full slice.
        return (1, self._ny, self._nx)

    def read_region(self, region: Region) -> np.ndarray:
        zs, ys, xs = region
        z0, z1 = zs.start, zs.stop
        out = np.empty((z1 - z0, ys.stop - ys.start, xs.stop - xs.start), dtype=self._dtype)
        if self._multipage:
            import tifffile

            with tifffile.TiffFile(self._path) as tf:
                for i, z in enumerate(range(z0, z1)):
                    out[i] = tf.pages[z].asarray()[ys, xs]
        else:
            for i, z in enumerate(range(z0, z1)):
                out[i] = _read_image(self._files[z])[ys, xs]
        return out

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("image stacks are read-only ingest sources")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, ImageStackBackend.open)
