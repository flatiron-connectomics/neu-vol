"""Destination/source locations: local paths, ``s3://``/``gs://`` URLs, kvstore dicts.

Everything downstream speaks TensorStore *kvstore specs* (plain dicts), so the ops
behave identically for local files and object stores. This module normalizes a
user-facing location into a kvstore and joins subpaths uniformly.
"""

from __future__ import annotations

from typing import Any, Mapping

_SCHEMES = {"s3://": "s3", "gs://": "gcs"}


def to_kvstore(location: str | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a location to a TensorStore kvstore spec.

    Accepts a local path, an ``s3://bucket/prefix`` / ``gs://bucket/prefix`` URL,
    or an existing kvstore dict (returned as-is).
    """
    if isinstance(location, Mapping):
        return dict(location)
    s = str(location)
    for scheme, driver in _SCHEMES.items():
        if s.startswith(scheme):
            bucket, _, prefix = s[len(scheme):].partition("/")
            return {"driver": driver, "bucket": bucket, "path": prefix}
    return {"driver": "file", "path": s}


def join(kvstore: Mapping[str, Any], *parts: str) -> dict[str, Any]:
    """Append path segments to a kvstore's ``path`` (leading '/' kept for files)."""
    kv = dict(kvstore)
    base = str(kv.get("path", ""))
    absolute = kv.get("driver") == "file" and base.startswith("/")
    segs = [base.strip("/")] if base.strip("/") else []
    segs += [str(p).strip("/") for p in parts if str(p).strip("/")]
    joined = "/".join(segs)
    kv["path"] = ("/" + joined) if absolute else joined
    return kv


def spec_kvstore(spec: Mapping[str, Any]) -> dict[str, Any]:
    """kvstore for a backend spec (handles 'kvstore', 'path', and scheme URLs)."""
    if "kvstore" in spec:
        return dict(spec["kvstore"])
    if "path" in spec:
        return to_kvstore(spec["path"])
    raise ValueError("spec needs 'kvstore' or 'path'")


def default_progress_path(dst: str | Mapping[str, Any]) -> str:
    """Default manifest path for a destination (local: alongside it; remote: cwd)."""
    kv = to_kvstore(dst)
    if kv.get("driver") == "file":
        return kv["path"].rstrip("/") + ".progress.jsonl"
    name = kv.get("path", "").rstrip("/").split("/")[-1] or kv.get("bucket", "volume")
    return name + ".progress.jsonl"
