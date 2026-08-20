"""Point a precomputed segmentation volume's ``info`` at its subresources.

A segmentation volume's ``info`` may carry ``mesh``, ``skeletons`` and
``segment_properties`` keys, each **naming a subdirectory** of the volume root. With them
set, one neuroglancer layer shows the labels together with their meshes, skeletons and
per-segment properties; without them each is a separate source the viewer has no reason
to associate with the labels.

This lives here rather than in a consumer because it edits a *volume's* ``info``, which is
this package's business, and because more than one consumer needs it: neu-morpho links
``mesh``/``skeletons`` after its seg stage, neu-mark links ``segment_properties``.
"""

from __future__ import annotations

from ..location import read_json, write_json

#: The ``@type`` each subresource's own ``info`` must declare. Pointing a volume at the
#: wrong directory is spec-legal and fails **silently** in the viewer (invariant 7), so it
#: is checked at write time instead of discovered in a browser.
SUBRESOURCE_TYPES = {
    "mesh": "neuroglancer_multilod_draco",
    "skeletons": "neuroglancer_skeletons",
    "segment_properties": "neuroglancer_segment_properties",
}


def link_subresources(volume_dir: str, *, mesh: str | None = None,
                      skeletons: str | None = None,
                      segment_properties: str | None = None) -> dict:
    """Set the subresource keys on a precomputed segmentation volume's ``info``.

    Each named subresource must exist and carry the right ``@type``. Returns the keys
    that were set.

    Existence is tested by reading ``<sub>/info``, not by listing a directory: object
    stores have no directories, and the subresource's own ``info`` is the thing that
    actually has to be there for the viewer.
    """
    info = read_json(volume_dir, "info")
    if info is None:
        raise FileNotFoundError(f"no precomputed 'info' at {volume_dir}")
    if info.get("type") != "segmentation":
        raise ValueError(f"{volume_dir}/info is not a segmentation volume "
                         f"(type={info.get('type')!r}); mesh/skeletons/"
                         f"segment_properties do not apply")

    changed: dict[str, str] = {}
    for key, sub in (("mesh", mesh), ("skeletons", skeletons),
                     ("segment_properties", segment_properties)):
        if sub is None:
            continue
        sub_info = read_json(volume_dir, sub, "info")
        if sub_info is None:
            raise FileNotFoundError(
                f"info would point {key!r} at {sub!r}, but {volume_dir}/{sub}/info "
                f"does not exist")
        want = SUBRESOURCE_TYPES.get(key)
        if want and sub_info.get("@type") != want:
            raise ValueError(f"{sub}/info has @type {sub_info.get('@type')!r}, "
                             f"expected {want!r} for the {key!r} key")
        info[key] = sub
        changed[key] = sub

    # A single kvstore write; no tmp+rename dance, because both the file driver
    # and object stores make an individual key write atomic.
    write_json(volume_dir, info, "info")
    return changed
