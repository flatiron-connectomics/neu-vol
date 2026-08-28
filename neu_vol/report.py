"""How a :func:`~neu_vol.source_metadata.describe` result presents itself.

``describe`` returns a mapping, because that is what every caller in the package needs
— ``create --like`` reads ``shape`` and ``dtype``, ``downsample`` reads
``level_voxel_sizes``, the CLI reads all of it. But a mapping is the wrong thing to
*look at*: in a notebook a dict of nested tuples is exactly as much information as the
CLI's table and far harder to read, and printing one was the reason people ran
``neu-vol info`` in a subprocess instead.

So the result is a :class:`Description` — a real ``dict``, with the rendering attached.
``d["shape"]`` and ``dict(d)`` behave as before; ``print(d)`` and a bare ``d`` in a cell
give the table, and :meth:`Description.frame` gives a DataFrame of the per-row part.

**The rendering lives here rather than in the CLI, and that is the point of the module.**
``cmd_info`` used to hold the only copy, so anything else that wanted the same table had
to grow a second one — and two renderings of the same dict drift. The CLI now assembles
its output from :meth:`Description.header_lines` and :meth:`Description.table_lines`
with its own store-reading sections (DVID nodes, ``provenance.json``) in between, which
is the only reason those are separate methods rather than one.

pandas is **not** a dependency of this package and must not become one; ``frame`` imports
it lazily and says what to install if it is missing. Nothing else here needs it.
"""

from __future__ import annotations

#: A header row whose value is absent is **omitted** rather than printed empty: an
#: offset of ``None`` and an offset of zero mean different things, and a blank reads as
#: the latter. The one exception is the voxel size, where "not recorded" is the fact
#: worth stating, since it is what `convert` and `downsample` will ask for.
_MISSING_VOXEL_SIZE = "(not recorded — --voxel-size required for convert/downsample)"


def _vec(values, sep: str = "x") -> str:
    return sep.join(f"{v:g}" for v in values)


class Description(dict):
    """A ``describe`` result: a plain dict that also knows how to show itself.

    Subclassing ``dict`` rather than wrapping one is deliberate — every existing caller
    subscripts this, and a wrapper would have meant touching all of them to get a
    notebook nicety. ``dict(d)`` strips the presentation if something needs a bare
    mapping (JSON, a cache key).
    """

    #: What a bare ``d`` shows in a terminal REPL, and what ``print(d)`` gives. The
    #: table, not the mapping: the mapping is available by subscripting, and nobody reads
    #: a screenful of nested tuples on purpose.
    def __repr__(self) -> str:
        return self.text()

    __str__ = __repr__

    @property
    def is_container(self) -> bool:
        """True when this describes an HDF5 file of several arrays and resolved none.

        ``shape`` is the test because it is the field that cannot have an answer: a
        container of thirteen differently-shaped crops has no single shape, dtype or
        frame, so ``describe`` leaves all three ``None`` and fills ``datasets`` instead.
        """
        return self.get("shape") is None and bool(self.get("datasets"))

    # ------------------------------------------------------------------ text
    def header_lines(self) -> list[str]:
        """The location and the scalar facts, one ``label  value`` line each."""
        location = self.get("location") or "(volume)"
        if self.is_container:
            n = len(self["datasets"])
            return [f"{location}",
                    f"  format      {self['format']} (a container: {n} volumetric "
                    f"datasets)"]

        meta = self.get("meta") or {}
        units = meta.get("units") or "nm"
        out = [f"{location}", f"  format      {self['format']}"]
        # Which array of an HDF5 container this describes. A path names the container, so
        # without this line a file holding one dataset and a file where `dataset=` picked
        # one of five read identically.
        if self.get("dataset"):
            out.append(f"  dataset     {self['dataset']}")
        out.append(f"  dtype       {self['dtype']}")
        out.append(f"  kind        {meta.get('kind') or '(not recorded)'}")
        if meta.get("voxel_size"):
            out.append(f"  voxel size  {_vec(meta['voxel_size'])} {units}")
        else:
            out.append(f"  voxel size  {_MISSING_VOXEL_SIZE}")
        if meta.get("offset"):
            out.append(f"  offset      {tuple(meta['offset'])} {units}")
        # In voxels as well as physical units, because that is the number `neu-vol write
        # --offset` takes and the one a crop was expressed in.
        if meta.get("voxel_offset"):
            out.append(f"              {tuple(meta['voxel_offset'])} voxels, "
                       f"{''.join(meta.get('spatial_axes') or 'zyx')}")
        if self.get("has_channels"):
            out.append("  channels    yes (leading axis)")
        return out

    def warning_lines(self) -> list[str]:
        """Two volumes in one directory, if that is the situation.

        Whichever loses the detection order is unreachable through every path in this
        package while still occupying the store, and nothing else says so.
        """
        markers = self.get("other_markers") or []
        if not markers:
            return []
        return [f"\n  WARNING: this directory also contains {', '.join(markers)} — a "
                f"second volume of another format is shadowed\n  here and cannot be "
                f"opened while {self['format']} wins detection. Move or delete one of "
                f"them.\n"]

    def table_lines(self) -> list[str]:
        """The per-row part: one row per dataset for a container, else one per level."""
        if self.is_container:
            return self._container_lines()
        return self._level_lines()

    def _container_lines(self) -> list[str]:
        entries = self["datasets"]
        axis_orders = {e.get("axes") for e in entries.values()}
        units = {e.get("units") for e in entries.values() if e.get("units")}
        # The axis order is a column only when the rows disagree; where they all say the
        # same thing it is one line underneath, which is the usual case and much quieter.
        mixed = len(axis_orders) > 1
        out = [f"  {'dataset':<22} {'shape (z,y,x)':>20} {'dtype':>8} {'chunk':>14} "
               f"{'voxel':>12} {'voxel offset':>26}" + ("  axes" if mixed else "")]
        for name, e in entries.items():
            chunk = "x".join(str(c) for c in e["chunks"]) if e["chunks"] else "(contiguous)"
            voxel = _vec(e["voxel_size"]) if e.get("voxel_size") else "—"
            off = (str(tuple(int(v) for v in e["voxel_offset"]))
                   if e.get("voxel_offset") else "—")
            out.append(f"  {name:<22} {str(e['shape']):>20} {e['dtype']:>8} {chunk:>14} "
                       f"{voxel:>12} {off:>26}"
                       + (f"  {e.get('axes') or '?'}" if mixed else ""))
        if not mixed and (only := next(iter(axis_orders))):
            out.append(f"\n  axes        {only} (all datasets), voxel size in "
                       f"{'/'.join(sorted(units)) if units else 'unrecorded units'}")
        if not any(e.get("voxel_size") for e in entries.values()):
            out += self._no_frame_lines()
        first = next(iter(entries))
        out.append(f"\n  dataset={first!r} (--dataset {first}) for the full report on "
                   f"one of them")
        return out

    def _no_frame_lines(self) -> list[str]:
        """Which attribute names were searched, when none of them was there.

        "This file records no scale" and "this file spells it differently" look identical
        otherwise — and the names are parameters (``voxel_size_field`` /
        ``offset_field``) precisely because another writer's choice is not this package's
        to assume, so saying which ones were tried is the actionable half.
        """
        from .backends.hdf5 import FRAME_ATTRIBUTES

        out = ["\n  no dataset records a voxel size, so nothing here knows its physical",
               "  scale. The attributes read as a frame are:"]
        out += [f"      {key:<14} {what}" for key, what in FRAME_ATTRIBUTES.items()]
        out.append("  If this file spells them differently, pass the names: "
                   "--voxel-size-field /\n      --offset-field on the commands that read "
                   "one, or give --voxel-size directly.")
        return out

    def _level_lines(self) -> list[str]:
        levels = self.get("levels") or {}
        if not levels:
            return ["  levels      none found"]
        # Each level's OWN recorded voxel size — never derived from shape ratios, which
        # ceil-division makes inexact (32 nm reads back as 31.9953), and never 2**level,
        # which is wrong on the anisotropic pyramids that are common.
        per_level = self.get("level_voxel_sizes")
        sharded = any(lv["read_chunks"] and lv["chunks"] != lv["read_chunks"]
                      for lv in levels.values())
        out = [f"  {'level':>5}  {'shape':>24}  {'voxel nm':>20}  {'chunk':>17}"
               + (f"  {'shard':>17}" if sharded else "")]
        for i, lv in sorted(levels.items()):
            vox = (_vec(per_level[i]) if per_level and i < len(per_level)
                   else "(not recorded)")
            # With sharding the *read* chunk is the unit actually fetched, so that is
            # what belongs in the "chunk" column; the write chunk is the shard.
            chunk = lv["read_chunks"] or lv["chunks"]
            chunk_s = "x".join(str(c) for c in chunk) if chunk else "?"
            row = f"  {i:>5}  {str(lv['shape']):>24}  {vox:>20}  {chunk_s:>17}"
            if sharded:
                shard = lv["chunks"] if lv["chunks"] != lv["read_chunks"] else None
                row += f"  {('x'.join(str(c) for c in shard) if shard else '—'):>17}"
            out.append(row)
        return out

    def text(self) -> str:
        """The whole report as one string — the CLI's output minus its store reads.

        ``provenance.json`` and the DVID node summary are *not* here: both open a store,
        and a ``__repr__`` that makes network calls is a trap in a notebook, where
        anything can echo an object. ``neu-vol info`` adds them between the header and
        the table.
        """
        return "\n".join(self.header_lines() + self.warning_lines() + self.table_lines())

    # ------------------------------------------------------------------ notebook
    def _repr_html_(self) -> str:
        """Notebook rendering: the header verbatim, the rows as a real table if pandas
        is importable and as preformatted text if it is not."""
        import html

        head = f"<pre>{html.escape(chr(10).join(self.header_lines()))}</pre>"
        warn = "".join(f"<pre>{html.escape(w)}</pre>" for w in self.warning_lines())
        try:
            frame = self.frame()
        except Exception:                                        # noqa: BLE001
            return head + warn + f"<pre>{html.escape(chr(10).join(self.table_lines()))}</pre>"
        return head + warn + frame.to_html(border=0)

    def frame(self):
        """A DataFrame of the per-row part: one row per dataset, or one per level.

        The two shapes answer the two questions this object is ever asked. For an HDF5
        container it is the dataset listing — name, shape, dtype, chunking and each
        crop's own recorded frame — which is what you sort and filter when choosing one.
        For anything else it is the pyramid: one row per level with its own voxel size
        and chunking.

        Values are the **numbers**, not the formatted strings the text table shows, so
        the frame is something to compute with; ``print(d)`` is for reading.
        """
        pd = _pandas()
        if self.is_container:
            rows = []
            for name, e in self["datasets"].items():
                rows.append({"dataset": name, "shape": e["shape"], "dtype": e["dtype"],
                             "chunks": e["chunks"],
                             "voxel_size": e.get("voxel_size"),
                             "voxel_offset": (tuple(int(v) for v in e["voxel_offset"])
                                              if e.get("voxel_offset") else None),
                             "offset": e.get("offset"), "units": e.get("units"),
                             "axes": e.get("axes")})
            return pd.DataFrame(rows).set_index("dataset")

        per_level = self.get("level_voxel_sizes") or []
        rows = []
        for i, lv in sorted((self.get("levels") or {}).items()):
            read = lv.get("read_chunks")
            write = lv.get("chunks")
            rows.append({"level": i, "shape": lv["shape"],
                         "voxel_size": (tuple(per_level[i]) if i < len(per_level)
                                        else None),
                         # Same split the text table makes: with sharding the read chunk
                         # is the unit fetched and the write chunk is the shard.
                         "chunk": read or write,
                         "shard": write if read and write != read else None})
        return pd.DataFrame(rows).set_index("level")


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:                                     # pragma: no cover
        raise ImportError(
            "a DataFrame needs pandas, which neu-vol does not depend on (it is not "
            "needed to read or write a volume). `pip install pandas`, or use "
            "`print(description)` / `description.text()` for the same table as text."
        ) from e
    return pd
