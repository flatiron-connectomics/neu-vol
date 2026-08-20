"""HDF5-backed reader (h5py).

Reads a dataset from an HDF5 file as a volume (docs/DESIGN.md §2). Read-focused
for v1; the dataset is assumed to be in canonical ``(z, y, x)`` / ``(c, z, y, x)``
order. Each backend instance holds an open file handle, and is reopened from its
spec on each dask worker (specs are picklable; open handles are not).

HDF5 is a container rather than an array, so two things a path alone cannot say are
answered here: :func:`sole_dataset` picks the dataset when there is exactly one, and
:meth:`HDF5Backend.stored_offset` reads a voxel offset the writer recorded beside the
array. The second is an **optional** backend capability — ``ops/write.py`` asks any
backend for it and does not care that only this one answers today.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "hdf5"


def _as_offset(value: Any, where: str) -> tuple[int, ...]:
    """A stored offset as whole voxels. Anything else is an error, not a rounding."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        raise ValueError(f"{where} is empty")
    if not np.all(arr == np.rint(arr.astype("float64"))):
        raise ValueError(f"{where} is {arr.tolist()}, which is not whole voxels")
    return tuple(int(v) for v in np.rint(arr.astype("float64")))


def require_local_path(path: str, what: str = "an HDF5 file") -> str:
    """``path`` if it is an ordinary filesystem path; a useful error if it is not.

    h5py needs a real path — there is no kvstore in front of it as there is for zarr and
    precomputed, and a container that cannot be read a chunk at a time would be a poor fit
    for one anyway. Without this check the failure is h5py's: it tries to *create a local
    file called* ``s3://bucket/piece.h5`` and reports ``errno = 2, No such file or
    directory``, which says nothing about what is actually wrong.
    """
    from ..location import is_local

    if not is_local(path):
        raise ValueError(
            f"{what} must be an ordinary filesystem path, and {path!r} is not: h5py has no "
            f"object-store driver, so nothing here can read or write HDF5 on one. Copy the "
            f"file to local storage first (`rclone copy`, `aws s3 cp`), or write it locally "
            f"and upload it afterwards.")
    return path


def _as_floats(value: Any, where: str) -> tuple[float, ...]:
    """A stored physical vector as floats. Unlike an offset, these are not whole voxels."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        raise ValueError(f"{where} is empty")
    return tuple(float(v) for v in arr.astype("float64"))


def axes_string(value: Any) -> str:
    """An axis-order attribute as ``"zyx"``, however h5py handed it over.

    ``"zyx"``, ``b"zyx"``, ``["z","y","x"]`` and ``[b"z",b"y",b"x"]`` are all things a
    writer may leave behind, and h5py's own round-tripping decides which one comes back.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value.replace(" ", "")
    return "".join(v.decode() if isinstance(v, bytes) else str(v)
                   for v in np.asarray(value).ravel())


def datasets(path: str, *, min_ndim: int = 3) -> list[str]:
    """Every dataset in the file with at least ``min_ndim`` dimensions, by path."""
    import h5py

    found: list[str] = []
    with h5py.File(require_local_path(path), "r") as f:
        f.visititems(lambda name, obj: found.append("/" + name)
                     if isinstance(obj, h5py.Dataset) and obj.ndim >= min_ndim else None)
    return found


def sole_dataset(path: str) -> str:
    """The file's one volumetric dataset, for callers that did not name one.

    HDF5 is a container, not an array, so a path alone is ambiguous. Guessing wrongly
    is worse than asking — so this only answers when there is exactly one candidate,
    and otherwise raises listing what it found.
    """
    found = datasets(path)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"{path} contains no 3D+ dataset")
    raise KeyError(f"{path} contains {len(found)} datasets ({', '.join(sorted(found))}); "
                   f"say which one")


class HDF5Backend:
    """View over a single dataset in an HDF5 file."""

    def __init__(self, spec: Mapping[str, Any]):
        import h5py

        self._spec = dict(spec)
        self._path = require_local_path(str(spec["path"]))
        self._dataset = str(spec.get("dataset", "/data"))
        self._file = h5py.File(self._path, "r")
        if self._dataset not in self._file:
            self._file.close()
            raise KeyError(f"dataset {self._dataset!r} not found in {self._path}")
        self._dset = self._file[self._dataset]

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "HDF5Backend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._dset.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dset.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        # HDF5 storage chunks if chunked, else treat the whole array as one chunk.
        return tuple(self._dset.chunks) if self._dset.chunks else self.shape

    def stored_offset(self, name: str = "voxel_offset"):
        """A voxel offset recorded in the file as ``(value, where_it_was_found)``.

        ``None`` if the file records none. Three places are searched, most specific
        first: the dataset's own attributes, the root group's attributes, then a
        top-level dataset of that name — writers put it in all three depending on
        which tool wrote the file.

        **The axis order is not knowable from here.** The name is precomputed's, where
        it means *xyz*; a file whose array is canonical ``(z, y, x)`` most likely means
        zyx. Nothing in the file distinguishes them, so this returns the numbers as
        stored and leaves the decision to the caller — which reports it, because
        getting it backwards places the subvolume mirrored through the z=x diagonal
        and nothing downstream can tell.
        """
        import h5py

        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return _as_offset(attrs[name], where), where
        node = self._file.get(name)
        if isinstance(node, h5py.Dataset):
            return _as_offset(node[()], f"/{name}"), f"/{name}"
        return None

    def stored_voxel_size(self, name: str = "voxel_size"):
        """A physical voxel size recorded in the file, as ``(value, where)``, or ``None``.

        Same three places and same order as :meth:`stored_offset`, and ``name`` is a
        parameter for the same reason: ``voxel_size`` is this package's own vocabulary, but
        a file written by something else may have called it whatever it liked, and the
        caller is the one who knows.
        """
        import h5py

        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return _as_floats(attrs[name], where), where
        node = self._file.get(name)
        if isinstance(node, h5py.Dataset):
            return _as_floats(node[()], f"/{name}"), f"/{name}"
        return None

    def stored_axes(self, name: str = "axes"):
        """The axis order the writer recorded, as ``(order, where)``, or ``None``.

        This is what makes :meth:`stored_offset`'s numbers unambiguous. Without it the
        order genuinely cannot be known — ``voxel_offset`` is precomputed's field name and
        means xyz there, while everything in this package is zyx — so ``ops/write`` has to
        ask. A file written by ``neu-vol to-hdf5`` says which, and then nobody has to.

        Searched on the dataset first and the root group second, as :meth:`stored_offset`
        does.
        """
        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return axes_string(attrs[name]), where
        return None

    def read_region(self, region: Region) -> np.ndarray:
        return np.asarray(self._dset[region])

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise NotImplementedError("HDF5 writing not supported yet")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, HDF5Backend.open)
