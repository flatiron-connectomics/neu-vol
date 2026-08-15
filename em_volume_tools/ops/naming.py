"""Placeholders in a destination path, resolved from the source.

``--dst /data/wasp_seg_{uuid:8}`` becomes ``/data/wasp_seg_d38898ac``. The point is
naming an export after the *node it came from*: a DVID branch ref means a different node
tomorrow, so a fixed destination name silently accumulates exports of different
segmentations under one path, and nothing afterwards can tell them apart. This is the
same problem ``provenance.json`` solves from the inside; a name solves it from the
outside, where a human browsing a directory sees it.

**Expanded once, in the caller, before anything uses the path.** The destination is used
to derive the format-suffixed targets, the progress manifest name and the resume check,
so expanding it late would leave those referring to a path with braces in it. Expansion
is idempotent — an expanded path has no placeholders left — so ``convert`` can call it
again harmlessly on a path the CLI already resolved.

Substitution is by regex, never ``str.format``: a real path may contain braces of its
own, and ``format`` would raise on them or, worse, interpret them.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: ``{name}`` or ``{name:spec}``. The NAME is narrow (a bare lowercase word) so an
#: unrelated brace in a path is left alone; the SPEC is anything up to the closing brace,
#: deliberately, so that a malformed one like ``{uuid:-2}`` is caught and reported rather
#: than failing to match and passing through as a literal directory name.
_FIELD = re.compile(r"\{([a-z_]+)(?::([^}]*))?\}")

#: Default length for ``{uuid}``. Eight hex characters, matching how these are referred
#: to in practice (`93fdbc`, git short shas) and short enough to read in a path. A
#: collision would need two nodes of one repo sharing a 32-bit prefix.
_UUID_DEFAULT = 8

#: What may appear in an expanded value. DVID uuids are hex and instance names are
#: usually plain, but a branch can carry `~`, and object-store keys are happier without
#: surprises.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def has_placeholder(dst: str) -> bool:
    return bool(_FIELD.search(str(dst)))


def _branch_of(ref: str) -> str:
    """The branch part of a ``repo:branch`` ref, or the whole ref when there is none.

    Taken from the ref rather than the node's DAG entry so this costs no extra request;
    the ref is what the caller typed, which is also what they mean by "the branch".
    """
    return ref.split(":", 1)[1] if ":" in ref else ref


def _values(src_spec: Mapping[str, Any]) -> dict[str, str]:
    """The substitutable values for this source, or ``{}`` if it offers none."""
    if src_spec.get("backend") != "dvid":
        return {}
    from ..backends.dvid import resolve_node

    # Cached, so this is free when the source metadata has already been read.
    node = resolve_node(src_spec)
    ref = str(src_spec.get("requested_ref") or src_spec.get("uuid") or "")
    return {
        "uuid": str(node["uuid"]),
        "instance": str(src_spec.get("instance", "")),
        "branch": _branch_of(ref),
    }


def _apply_spec(name: str, value: str, spec: str | None) -> str:
    if spec is None:
        return value[:_UUID_DEFAULT] if name == "uuid" else value
    if spec == "full":
        return value
    if spec.isdigit() and int(spec) > 0:
        return value[:int(spec)]
    raise ValueError(
        f"bad placeholder {{{name}:{spec}}} in --dst: the part after ':' must be a "
        f"positive number of characters, or 'full'")


def expand(dst: str, src_spec: Mapping[str, Any]) -> str:
    """Resolve ``{...}`` placeholders in ``dst`` against ``src_spec``.

    Returns ``dst`` unchanged when it holds no placeholder, without resolving anything —
    so an ordinary destination costs nothing.
    """
    dst = str(dst)
    if not has_placeholder(dst):
        return dst

    values = _values(src_spec)
    if not values:
        raise ValueError(
            f"--dst {dst!r} contains a placeholder, but the source "
            f"({src_spec.get('backend')!r}) supplies none. Placeholders name an export "
            f"after the version it came from, which only DVID sources have.")

    def sub(m: re.Match) -> str:
        name, spec = m.group(1), m.group(2)
        if name not in values:
            raise ValueError(
                f"unknown placeholder {{{name}}} in --dst; this source offers "
                f"{', '.join('{' + k + '}' for k in sorted(values))}")
        value = _apply_spec(name, values[name], spec)
        if not value:
            raise ValueError(
                f"placeholder {{{name}}} in --dst resolved to an empty string, which "
                f"would leave a nameless path component")
        return _UNSAFE.sub("-", value)

    return _FIELD.sub(sub, dst)
