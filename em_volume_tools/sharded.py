"""Writing and reading ``neuroglancer_uint64_sharded_v1`` indexes.

A sharded index turns "one object per key" into a handful of ``.shard`` files. That is not a
nicety at scale: a precomputed annotation source keyed by annotation id needs one object per
annotation, and the reference male-CNS synapse dataset has ~312M of them — 1,024 shards of
~31 MB instead of 312 million objects. The same arithmetic bites a volume, where file count
is volume/chunk³ and ceph enforces inode quotas as well as capacity.

**None of the format is implemented here, deliberately.** tensorstore already ships a
``neuroglancer_uint64_sharded`` kvstore driver that reads *and writes* it, so this module is
a thin wrapper that opens one. That avoids reimplementing murmurhash3_x86_128, minishard
tables, the shard-index layout and the two gzip layers — every one of which produces a
spec-legal file that a viewer silently rejects when it is subtly wrong.

Verified against the published reference rather than only against ourselves: opening
``gs://flyem-male-cns/v1.0/male-cns-v1.0-synapses-precomputed/by_id/`` with its own sharding
spec and reading one annotation by key returned the same bytes as decoding that annotation
by hand out of the spatial index. Two independent paths, one answer.

## The two things a caller has to get right

- **Keys are 8-byte BIG-endian uint64** (:func:`key`). Everything else in precomputed is
  little-endian, so this is the one place the byte order flips.
- **The ``metadata`` block is the sharding spec that goes into your own ``info``.** Pass the
  *same dict* to both, or the file and the metadata describing it can drift — and a reader
  trusts the metadata.

``shard_bits=0`` with ``minishard_bits=0`` is legal and useful: it writes a single
``0.shard``, which is what the reference does for its coarsest spatial level. Shard
everything, and let the bit counts collapse where a level is small; one code path.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping

AT_TYPE = "neuroglancer_uint64_sharded_v1"

#: The only hash the format defines besides ``identity``. The reference uses this for every
#: index; ``identity`` is for keys that are already uniformly distributed.
DEFAULT_HASH = "murmurhash3_x86_128"


def key(chunk_id: int) -> bytes:
    """A uint64 key as the sharded kvstore expects it: 8 bytes, **big**-endian."""
    return struct.pack(">Q", int(chunk_id))


def sharding_spec(*, shard_bits: int, minishard_bits: int, preshift_bits: int = 0,
                  hash: str = DEFAULT_HASH, data_encoding: str = "gzip",
                  minishard_index_encoding: str = "gzip") -> dict[str, Any]:
    """The sharding metadata, for both the kvstore and the ``info`` that describes it.

    ``2**shard_bits`` is the number of ``.shard`` files. ``minishard_bits`` splits each shard
    internally, which bounds how much index a reader must fetch to find one key — the
    reference uses 9 for a 1,024-shard ``by_id`` and 0 for a single-file spatial level.
    """
    for name, value in (("shard_bits", shard_bits), ("minishard_bits", minishard_bits),
                        ("preshift_bits", preshift_bits)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    if hash not in (DEFAULT_HASH, "identity"):
        raise ValueError(f"hash must be {DEFAULT_HASH!r} or 'identity'; got {hash!r}")
    for name, value in (("data_encoding", data_encoding),
                        ("minishard_index_encoding", minishard_index_encoding)):
        if value not in ("raw", "gzip"):
            raise ValueError(f"{name} must be 'raw' or 'gzip'; got {value!r}")
    return {"@type": AT_TYPE, "hash": hash, "preshift_bits": preshift_bits,
            "shard_bits": shard_bits, "minishard_bits": minishard_bits,
            "data_encoding": data_encoding,
            "minishard_index_encoding": minishard_index_encoding}


def plan_sharding(n_keys: int, *, per_shard: int = 30_000,
                  max_shard_bits: int = 10) -> dict[str, Any]:
    """Sharding parameters for ``n_keys`` entries, aiming at ``per_shard`` keys per file.

    Chosen so a shard is a sensible object to fetch — the reference lands at ~300k keys and
    ~31 MB per shard, which is larger than we want for 1.4M annotations, so the default aims
    smaller. ``minishard_bits`` then keeps each minishard index modest, since a reader
    fetching one key must decode a whole minishard index.

    This is a heuristic, not a rule from the spec. Override it when you know better.
    """
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")
    shard_bits = 0
    while (n_keys >> shard_bits) > per_shard and shard_bits < max_shard_bits:
        shard_bits += 1
    keys_per_shard = max(1, n_keys >> shard_bits)
    # Aim for a few thousand keys per minishard: enough that the index is worth having,
    # small enough that reading one key does not decode a huge index.
    minishard_bits = 0
    while (keys_per_shard >> minishard_bits) > 2_000 and minishard_bits < 9:
        minishard_bits += 1
    return sharding_spec(shard_bits=shard_bits, minishard_bits=minishard_bits)


def open_kvstore(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
                 *parts: str):
    """A sharded kvstore rooted at ``location``/``parts``.

    ``location`` is anything :mod:`em_volume_tools.location` understands, so a local path and
    ``s3://`` behave the same. The trailing slash matters to the driver, so it is added here
    rather than left to the caller.
    """
    import tensorstore as ts

    from .location import ensure_credentials, join, to_kvstore

    base = join(to_kvstore(location), *parts) if parts else dict(to_kvstore(location))
    base = ensure_credentials(base)
    path = str(base.get("path", ""))
    if path and not path.endswith("/"):
        base = {**base, "path": path + "/"}
    return ts.KvStore.open({"driver": "neuroglancer_uint64_sharded",
                            "base": base, "metadata": dict(metadata)}).result()


def write_all(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
              entries: Iterable[tuple[int, bytes]], *parts: str) -> int:
    """Write ``(chunk_id, bytes)`` pairs into a sharded index. Returns how many.

    One transaction for the whole batch: the driver assembles each shard in memory and emits
    it as a single object, so writing key-by-key outside a transaction would rewrite a shard
    per key. That is the difference between one PUT per shard and one per annotation, which
    is the entire reason for sharding.
    """
    import tensorstore as ts

    kv = open_kvstore(location, metadata, *parts)
    written = 0
    with ts.Transaction() as txn:
        scoped = kv.with_transaction(txn)
        for chunk_id, payload in entries:
            scoped[key(chunk_id)] = bytes(payload)
            written += 1
    return written


def read_one(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
             chunk_id: int, *parts: str) -> bytes | None:
    """One entry by id, or ``None`` if absent. For tests and for spot-checking output."""
    kv = open_kvstore(location, metadata, *parts)
    result = kv.read(key(chunk_id)).result()
    return bytes(result.value) if result.state == "value" else None
