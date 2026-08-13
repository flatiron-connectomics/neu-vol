"""Destination/source locations: local paths, ``s3://``/``gs://`` URLs, kvstore dicts.

Everything downstream speaks TensorStore *kvstore specs* (plain dicts), so the ops
behave identically for local files and object stores. This module normalizes a
user-facing location into a kvstore and joins subpaths uniformly.
"""

from __future__ import annotations

import hashlib
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


def is_local(location: str | Mapping[str, Any]) -> bool:
    """True if ``location`` resolves to an ordinary filesystem path.

    Callers that genuinely need POSIX semantics (sqlite, append-in-place logs,
    ``os.replace``) use this to reject object-store locations up front with a
    clear message, rather than failing deep inside a write.
    """
    return to_kvstore(location).get("driver") == "file"


def local_path(location: str | Mapping[str, Any]) -> str:
    """The filesystem path for a local location; raises if it is remote."""
    kv = to_kvstore(location)
    if kv.get("driver") != "file":
        raise ValueError(
            f"{location!r} is not a local path (driver={kv.get('driver')!r}); "
            "this operation needs a filesystem")
    return str(kv.get("path", ""))


# --------------------------------------------------------------------------- #
# Byte / JSON I/O over a kvstore
#
# One code path for local files and object stores, so callers never branch on
# the destination. Note the file driver creates missing parent directories on
# write, so these replace `os.makedirs` + `open` rather than supplementing it.
# --------------------------------------------------------------------------- #
def ensure_credentials(kvstore: Mapping[str, Any]) -> dict[str, Any]:
    """Bootstrap object-store credentials for a kvstore that is about to be opened.

    **Call this at every point where a store is opened, not where a spec is
    built.** tensorstore 0.1.84's S3 *profile* provider cannot read
    ``~/.aws/credentials``, so ``aws.ensure_aws_credentials`` copies the profile
    into ``AWS_*`` env vars for its *environment* provider — which means the
    bootstrap is **per process**, and a process that skips it gets a 403 on every
    write. That failure mode is nasty on dask: whether a given worker has
    bootstrapped depends on whether it happened to open an S3 store earlier in the
    run, so the same code path succeeds on some workers and 403s on others, and
    which is which changes with worker startup timing.
    """
    kv = dict(kvstore)
    if kv.get("driver") == "s3":
        from .aws import ensure_aws_credentials
        ensure_aws_credentials()
    return kv


# Opened stores, keyed by their *prefix* spec. A KvStore is a handle to a prefix,
# not to one object, so every read/write under that prefix can share one — and
# reopening is not free: each S3 open reconstructs the credential provider chain,
# which costs setup and emits two absl ERROR lines per open. Unbounded by design;
# the key space is the handful of prefixes one run writes to (volume, mesh,
# skeleton). Per process, so dask workers each build their own after forking.
_STORES: dict[str, Any] = {}


def _open_store(kvstore: Mapping[str, Any]):
    import json as _json

    import tensorstore as ts

    cache_key = _json.dumps(dict(kvstore), sort_keys=True, default=str)
    store = _STORES.get(cache_key)
    if store is None:
        store = ts.KvStore.open(ensure_credentials(kvstore)).result()
        _STORES[cache_key] = store
    return store


def _kv(location: str | Mapping[str, Any], *parts: str):
    kv = join(to_kvstore(location), *parts) if parts else to_kvstore(location)
    key = ""
    base = str(kv.get("path", ""))
    if base:
        # Split the final segment off as the key so the store is opened on the
        # containing prefix — a trailing separator is what keeps the key from
        # being concatenated onto the directory name.
        head, _, tail = base.rstrip("/").rpartition("/")
        kv["path"] = (head + "/") if head else ("/" if base.startswith("/") else "")
        key = tail
    return _open_store(kv), key


def read_bytes(location: str | Mapping[str, Any], *parts: str) -> bytes | None:
    """Read one object; ``None`` if it does not exist."""
    store, key = _kv(location, *parts)
    result = store.read(key).result()
    return bytes(result.value) if result.state == "value" else None


def write_bytes(location: str | Mapping[str, Any], data: bytes, *parts: str) -> None:
    """Write one object, creating parent directories for the file driver."""
    store, key = _kv(location, *parts)
    store.write(key, bytes(data)).result()


def list_keys(location: str | Mapping[str, Any], *parts: str,
              limit: int | None = None) -> list[str]:
    """Key names directly under a prefix.

    Goes through the cached, credential-bootstrapped opener like every other access
    here.

    **``limit`` truncates the RESULT, not the request, and does not bound the cost.**
    tensorstore's ``kvstore.list()`` returns a future of the complete listing, so the
    whole enumeration happens before Python can stop reading it — a prefix with millions
    of objects costs the same with ``limit=4`` as without. It is useful only for keeping
    the returned list small.

    So there is no cheap way to sample a large prefix. To answer a question about one
    object, probe it by name with :func:`exists`; to bound a listing, list a narrower
    prefix. Getting this wrong cost 51 s inside ``detect_backend`` on a dense volume,
    where the docstring here promised "one small request".
    """
    kv = join(to_kvstore(location), *parts) if parts else dict(to_kvstore(location))
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    store = _open_store(kv)
    out: list[str] = []
    for key in store.list().result():
        out.append(key.decode() if isinstance(key, bytes) else str(key))
        if limit is not None and len(out) >= limit:
            break
    return out


def exists(location: str | Mapping[str, Any], *parts: str) -> bool:
    """True if the object exists. One request; safe against object stores."""
    store, key = _kv(location, *parts)
    return store.read(key).result().state == "value"


def read_json(location: str | Mapping[str, Any], *parts: str):
    """Read and parse a JSON object; ``None`` if it does not exist."""
    import json

    raw = read_bytes(location, *parts)
    return None if raw is None else json.loads(raw.decode("utf-8"))


def write_json(location: str | Mapping[str, Any], obj: Any, *parts: str,
               indent: int | None = 2) -> None:
    """Serialize ``obj`` as JSON and write it."""
    import json

    write_bytes(location, json.dumps(obj, indent=indent).encode("utf-8"), *parts)


def default_progress_path(dst: str | Mapping[str, Any]) -> str:
    """Default manifest path for a destination (local: alongside it; remote: cwd).

    A manifest must be an ordinary file — it is appended to and fsynced per task — so for a
    remote destination it cannot live beside the volume and lands in the working directory
    instead. That is where the name has to do the disambiguating, and the **last path
    component alone is not enough**: two prefixes ending in the same word are entirely
    ordinary (``…/specimen3/gt_v1_eval`` and ``…/specimen5/gt_v1_eval``), and two runs
    sharing a manifest do not merely interleave a log. They read each other's records on
    resume and skip each other's blocks, and a ``--fresh`` run truncates the other's
    history mid-flight. Nothing about the output looks wrong afterwards.

    So a remote name carries a short digest of the **whole** location beside the readable
    tail: ``gt_v1_eval.4f2a91c7.progress.jsonl``. The digest covers only the identity fields
    — driver, bucket, path — rather than the kvstore dict, so a field added there later
    cannot rename the manifest a run is in the middle of appending to. The trailing slash is
    normalised away for the same reason: ``s3://b/k`` and ``s3://b/k/`` are one destination
    and must resolve to one manifest, or a resumed run would start over and `em-vol
    progress` would report on nothing.

    Local destinations keep their old name: a path beside the volume is already unique, and
    renaming those would orphan every manifest already on disk.
    """
    kv = to_kvstore(dst)
    if kv.get("driver") == "file":
        return kv["path"].rstrip("/") + ".progress.jsonl"
    path = str(kv.get("path", "")).rstrip("/")
    name = path.split("/")[-1] or str(kv.get("bucket", "volume"))
    identity = f"{kv.get('driver', '')}|{kv.get('bucket', '')}|{path}"
    digest = hashlib.blake2b(identity.encode(), digest_size=4).hexdigest()
    return f"{name}.{digest}.progress.jsonl"
