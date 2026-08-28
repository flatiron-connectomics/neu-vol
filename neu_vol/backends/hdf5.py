"""HDF5-backed reader (h5py).

Reads a dataset from an HDF5 file as a volume. Read-focused
for v1; the dataset is assumed to be in canonical ``(z, y, x)`` / ``(c, z, y, x)``
order. Each backend instance holds an open file handle, and is reopened from its
spec on each dask worker (specs are picklable; open handles are not).

HDF5 is a container rather than an array, so two things a path alone cannot say are
answered here: :func:`sole_dataset` picks the dataset when there is exactly one, and
:meth:`HDF5Backend.stored_offset` reads a voxel offset the writer recorded beside the
array. The second is an **optional** backend capability — ``ops/write.py`` asks any
backend for it and does not care that only this one answers today.

``stored_offset`` / ``stored_voxel_size`` / ``stored_axes`` / ``stored_units`` are the
four of those, and together they are what makes an HDF5 file describable:
``source_metadata.read_source_metadata`` composes them into the same coordinate-metadata
dict a precomputed ``info`` or an OME group yields, so ``neu-vol info`` and ``create
--like`` work on a file this package packed. They are read strictly — an axis order the
file states but nobody can interpret is an error, never a guess — because getting the
order wrong places the data mirrored through the z=x diagonal and nothing downstream can
tell.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "hdf5"

#: Extensions that name an HDF5 container. Public because it is the *detection*
#: vocabulary too: HDF5 has no marker object to probe for — the format signature is
#: inside the file — so ``source_metadata.detect_file_backend`` goes by the name, and the
#: two lists must not drift.
HDF5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".he5")


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
    from ..location import is_local, local_path

    if not is_local(path):
        raise ValueError(
            f"{what} must be an ordinary filesystem path, and {path!r} is not: h5py has no "
            f"object-store driver, so nothing here can read or write HDF5 on one. Copy the "
            f"file to local storage first (`rclone copy`, `aws s3 cp`), or write it locally "
            f"and upload it afterwards.")
    # Resolved, not just vetted: h5py takes a relative path but does **not** expand `~`, so
    # `~/data/piece.h5` reaches it verbatim and fails with errno 2 on a path that plainly
    # exists. `local_path` goes through `to_kvstore`, so this and every store path get the
    # same treatment.
    return local_path(path)


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


#: The attribute names read as a frame, and what each one means. Reported when a file
#: records none, because "no coordinate metadata" and "this file spells it differently"
#: look identical otherwise, and the field names are parameters everywhere else
#: (``voxel_size_field`` / ``offset_field``) precisely because another writer's choice is
#: not this package's to assume.
FRAME_ATTRIBUTES = {
    "voxel_size": "physical size of one voxel, in `units`",
    "voxel_offset": "where this array starts, in whole voxels",
    "offset": "the same origin in physical units",
    "units": "what `voxel_size` is measured in (this suite works in nm)",
    "axes": "the axis order of the three vectors above ('zyx' or 'xyz')",
}


def describe_datasets(path: str, *, min_ndim: int = 3) -> dict[str, dict]:
    """Every volumetric dataset in the file, with its geometry and recorded frame.

    ``{name: {"shape", "dtype", "chunks", "ndim", <frame attributes>}}``, in file order,
    from **one** file open — attributes only, so it is cheap however large the arrays are.

    This is the *container's* answer, and it exists because an HDF5 file is not usually
    one volume: a bag of ground-truth crops, each with its own ``voxel_offset``, is the
    ordinary shape of one here, and asking a 13-piece file to name its "sole dataset" can
    only fail. :func:`~neu_vol.source_metadata.describe` still describes one array —
    everything downstream of it needs exactly one — so this is what ``neu-vol info``
    falls back to in order to say *which* arrays there are to choose from.

    Frame values come back **exactly as recorded**, not reoriented into zyx: each entry
    carries its own ``axes``, and normalising here would mean either dropping that column
    or silently disagreeing with it. The single-array path
    (``source_metadata._read_hdf5``) is where the reorientation happens, and it is strict
    there. Values fall back to the root group's, which is where a file-wide
    ``voxel_size``/``units`` usually lives.
    """
    import h5py

    out: dict[str, dict] = {}
    with h5py.File(require_local_path(path), "r") as f:
        root = {k: v for k, v in f.attrs.items() if k in FRAME_ATTRIBUTES}

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim < min_ndim:
                return
            entry = {"shape": tuple(int(s) for s in obj.shape),
                     "dtype": str(obj.dtype),
                     "chunks": tuple(int(c) for c in obj.chunks) if obj.chunks else None,
                     "ndim": int(obj.ndim)}
            for key in FRAME_ATTRIBUTES:
                if key in obj.attrs:
                    value = obj.attrs[key]
                elif key in root:
                    value = root[key]
                else:
                    continue
                if key in ("units", "axes"):
                    entry[key] = (axes_string(value) if key == "axes"
                                  else (value.decode() if isinstance(value, bytes)
                                        else str(value)))
                else:
                    entry[key] = tuple(float(v) for v in np.asarray(value).ravel())
            out["/" + name] = entry

        f.visititems(visit)
    return out


def sole_dataset(path: str) -> str:
    """The file's one volumetric dataset, for callers that did not name one.

    HDF5 is a container, not an array, so a path alone is ambiguous. Guessing wrongly
    is worse than asking — so this only answers when there is exactly one candidate,
    and otherwise raises listing what it found. A bag of ground-truth crops in one file
    is the ordinary case, not an edge one, so the error says how to choose: `neu-vol
    info` lists them with their shapes and offsets, and every command that reads an HDF5
    array takes the name.
    """
    found = datasets(path)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"{path} contains no 3D+ dataset")
    listed = ", ".join(sorted(found))
    raise KeyError(
        f"{path} contains {len(found)} volumetric datasets ({listed}); say which one — "
        f"`neu-vol info {path}` lists them with their shapes and offsets, then pass "
        f"dataset= (`open_hdf5`, `describe`), "
        f"--dataset (`info`), --dataset (`write`), --src-dataset (`to-hdf5`), or "
        f"'dataset' in a backend spec")


def open_hdf5(path: str, dataset: str | None = None) -> "HDF5Backend":
    """One dataset of an HDF5 file, ready to read regions from.

    The front door for a file you are pointing at by hand::

        be = open_hdf5("piece.h5")                      # one dataset: found
        be = open_hdf5("gt_v1_eval.h5", "/z07901")      # a container: name it
        be.read_region((slice(0, 64), slice(0, 64), slice(0, 64)))

    ``dataset`` is optional only when the file holds exactly one 3D+ array; with several
    it is **required**, and the error lists them (:func:`sole_dataset`). That is not
    strictness for its own sake — a bag of annotated crops in one file is the ordinary
    arrangement here, and picking one of thirteen on the caller's behalf would be
    picking the wrong one twelve times out of thirteen.

    **Deliberately not a general "open anything" helper**, and deliberately not folded
    into :func:`~neu_vol.backends.base.open_backend`. The format is in the name, so this
    needs no detection at all — no marker probes, no store reads. ``open_backend`` stays
    what its docstring says it is: the spec-driven primitive a per-block dask task
    reopens on a worker, thousands of times a run, with the reader named rather than
    inferred (CLAUDE.md invariant 9). To *find out* what something is, use
    :func:`~neu_vol.source_metadata.describe`.

    Goes through ``open_backend`` rather than constructing the backend directly, so the
    handle joins the per-process cache: an ``HDF5Backend`` holds an open ``h5py.File``,
    and two calls for one array should not mean two file handles.
    """
    from ..logs import quiet_reads
    from .base import open_backend

    # A notebook-facing entry point with no `main()` to wrap; see `source_metadata.describe`.
    # HDF5 itself is local and silent, but `require_local_path` goes through `location`, and
    # a caller reaching for this in a loop should not have to think about it.
    with quiet_reads():
        path = require_local_path(path)
        return open_backend({"backend": TAG, "path": path,
                             "dataset": dataset if dataset is not None
                             else sole_dataset(path)})


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

    def stored_units(self, name: str = "units"):
        """The physical unit the writer recorded, as ``(unit, where)``, or ``None``.

        Attributes only, dataset before root — a unit is a string, so the
        top-level-dataset fallback the other three readers have would be a 1-element
        string array nobody writes. ``neu-vol to-hdf5`` records it beside
        ``voxel_size``, which is the pairing that makes those numbers mean anything:
        the rest of the suite works in nm, and a file whose voxel size is in
        micrometres is off by a thousand with nothing to show for it.
        """
        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                value = attrs[name]
                return (value.decode() if isinstance(value, bytes) else str(value)), where
        return None

    def read_region(self, region: Region) -> np.ndarray:
        return np.asarray(self._dset[region])

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise NotImplementedError("HDF5 writing not supported yet")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, HDF5Backend.open)
