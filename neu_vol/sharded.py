"""Writing and reading ``neuroglancer_uint64_sharded_v1`` indexes.

A sharded index turns "one object per key" into a handful of ``.shard`` files. That is not a
nicety at scale: a precomputed annotation source keyed by annotation id needs one object per
annotation, and the reference male-CNS synapse dataset has ~312M of them — 1,024 shards of
~31 MB instead of 312 million objects. The same arithmetic bites a volume, where file count
is volume/chunk³ and ceph enforces inode quotas as well as capacity.

**Writing does not implement any of the format, deliberately.** tensorstore already ships a
``neuroglancer_uint64_sharded`` kvstore driver that reads *and writes* it, so :func:`write_all`
and :func:`read_one` are thin wrappers that open one. That avoids reimplementing
murmurhash3_x86_128, minishard tables, the shard-index layout and the two gzip layers — every
one of which produces a spec-legal file that a viewer silently rejects when it is subtly wrong.

**:class:`ShardReader` is the one exception, and it is narrow.** The driver addresses entries
by KEY, and that is all it can do — which is not enough for the sharded multi-resolution mesh
format, where the mesh fragment data is *not an indexed entry at all*. It sits immediately
before the object's manifest in the same shard file and is addressed by byte offsets relative
to it, so reading a mesh requires knowing where in the shard the manifest landed. There is no
tensorstore API for that, so the addressing — hash, shard index, minishard table — is
reimplemented here. Nothing about *writing* is, and :func:`read_one` deliberately stays on the
driver: ``test_shard_reader_agrees_with_tensorstore`` asserts the two paths return the same
bytes for the same key, which is what keeps this from being a second unverified implementation.

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
from typing import Any, Iterable, Mapping, Sequence

AT_TYPE = "neuroglancer_uint64_sharded_v1"

#: The only hash the format defines besides ``identity``. The reference uses this for every
#: index; ``identity`` is for keys that are already uniformly distributed.
DEFAULT_HASH = "murmurhash3_x86_128"


def key(chunk_id: int) -> bytes:
    """A uint64 key as the sharded kvstore expects it: 8 bytes, **big**-endian."""
    return struct.pack(">Q", int(chunk_id))


def morton_bits(grid_shape: Sequence[int]) -> list[int]:
    """How many bits the compressed Morton code spends on each dimension.

    The spec's own condition, ``2**i < grid_size[dim]``, which is ``ceil(log2(size))`` — and
    **zero** for a size-1 dimension. That is the "compressed" part: a flat grid spends no bits
    on the axis it does not subdivide.
    """
    out = []
    for size in grid_shape:
        bits = 0
        while (1 << bits) < int(size):
            bits += 1
        out.append(bits)
    return out


def compressed_morton_code(positions, grid_shape: Sequence[int]):
    """Compressed Morton codes for grid cell positions, as precomputed keys them.

    ``positions`` is ``(n, 3)`` or ``(3,)`` in the **same axis order as grid_shape**, which for
    precomputed means **xyz with x varying fastest**. Returns an ``int64`` array (or one int).

    This is how a **sharded** index addresses a grid cell — a volume chunk or an annotation
    source's spatial cell alike. It is emphatically *not* a row-major flattening, and the
    difference is a silent one: the two agree whenever at most one axis is subdivided, so a
    1×1×1 or 2×1×1 grid gives identical keys and any test on a small grid passes. They diverge
    as soon as two axes are, and then the objects are written at keys nothing ever requests —
    the data is all present and the viewer renders none of it. Measured on a real annotation
    source: agreement fell from every cell at the coarsest levels to 0 of 867 at the finest,
    leaving 2.9% of annotations reachable.

    Volumes never come through here — tensorstore keys those itself — so this exists for
    hand-built sharded indexes, which today means the annotation spatial index.
    """
    import numpy as np

    array = np.atleast_2d(np.asarray(positions, dtype=np.int64))
    if array.shape[-1] != len(grid_shape):
        raise ValueError(f"positions have {array.shape[-1]} axes, grid_shape has "
                         f"{len(grid_shape)}")
    per_dim = morton_bits(grid_shape)
    code = np.zeros(len(array), dtype=np.int64)
    j = 0
    for i in range(max(per_dim, default=0)):
        for dim, bits in enumerate(per_dim):
            if i < bits:
                code |= ((array[:, dim] >> i) & 1) << j
                j += 1
    return code if np.ndim(positions) > 1 else int(code[0])


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

    ``location`` is anything :mod:`neu_vol.location` understands, so a local path and
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


def clear(location: str | Mapping[str, Any], *parts: str) -> int:
    """Delete the ``.shard`` objects of an index. Returns how many were removed.

    Named for what it is rather than folded into a flag, because a partly written index is a
    real state: this is not atomic, so an interrupted rewrite leaves fewer shards than either
    version had. That is still better than the alternative it exists to prevent — see
    :func:`write_all`.

    **Only ``.shard`` objects are removed, and that restriction is load-bearing.** This used
    to delete every key under the prefix, which is harmless where an index owns its own
    directory — as every annotation index does — and destructive where it does not. A sharded
    mesh or skeleton lives in the *same* directory as the ``info`` that describes it, so a
    rewrite there took the ``info`` with it and left a subresource that reads as absent: the
    shards are all present and correct, nothing declares them, and every body comes back
    ``None``. The shard filename is fixed by the spec, so filtering on it is exact.
    """
    from .location import _kv, list_keys

    removed = 0
    for name in list_keys(location, *parts):
        if not name.endswith(".shard"):
            continue
        store, resolved = _kv(location, *parts, name)
        store.write(resolved, None).result()
        removed += 1
    return removed


def write_all(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
              entries: Iterable[tuple[int, bytes]], *parts: str,
              replace: bool = True) -> int:
    """Write ``(chunk_id, bytes)`` pairs into a sharded index. Returns how many.

    One transaction for the whole batch: the driver assembles each shard in memory and emits
    it as a single object, so writing key-by-key outside a transaction would rewrite a shard
    per key. That is the difference between one PUT per shard and one per annotation, which
    is the entire reason for sharding.

    **``replace`` deletes the existing shards first, and defaults to on because the
    alternative silently MERGES.** A sharded index is a key-value store: writing keys 1..N
    leaves any key already present and not in that set exactly where it was. So rewriting an
    index whose KEY SPACE changed — a spatial index regridded, a body list narrowed — produces
    a file containing both generations, and a reader asking for a key that only the old one
    used gets a stale answer. The object *names* are unchanged, so a listing looks right and
    the staleness is inside the shards. This cost a debugging session: a spatial index rewritten
    from row-major keys to Morton keys read back correct data at the wrong positions, because
    the old row-major entries were still there. Pass ``replace=False`` only to add to an index
    on purpose.
    """
    import tensorstore as ts

    if replace:
        clear(location, *parts)
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


# --------------------------------------------------------------------------- #
# reading by byte offset
# --------------------------------------------------------------------------- #

_M32 = 0xFFFFFFFF


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M32


def _fmix32(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _M32
    h ^= h >> 16
    return h


def murmurhash3_x86_128_low64(data: bytes, seed: int = 0) -> int:
    """The low 64 bits of MurmurHash3 x86_128 — the hash the sharded format specifies.

    Pure Python, which is fine: it is called once per key looked up, never per byte of
    data. The alternative was a compiled dependency (``mmh3``) in a package whose install
    story is already delicate, for one hash of one 8-byte value.

    Note ``x86_128``, **not** ``x64_128``: the two are different functions producing different
    digests, and picking the wrong one sends every read to a shard that does not hold the
    key — which reads back as "this object does not exist" rather than as an error.
    """
    c1, c2, c3, c4 = 0x239B961B, 0xAB0E9789, 0x38B34AE5, 0xA1E38B93
    h1 = h2 = h3 = h4 = seed & _M32
    length = len(data)
    nblocks = length // 16

    for i in range(nblocks):
        k1, k2, k3, k4 = struct.unpack_from("<IIII", data, i * 16)
        k1 = (_rotl32((k1 * c1) & _M32, 15) * c2) & _M32
        h1 ^= k1
        h1 = ((_rotl32(h1, 19) + h2) * 5 + 0x561CCD1B) & _M32
        k2 = (_rotl32((k2 * c2) & _M32, 16) * c3) & _M32
        h2 ^= k2
        h2 = ((_rotl32(h2, 17) + h3) * 5 + 0x0BCAA747) & _M32
        k3 = (_rotl32((k3 * c3) & _M32, 17) * c4) & _M32
        h3 ^= k3
        h3 = ((_rotl32(h3, 15) + h4) * 5 + 0x96CD1C35) & _M32
        k4 = (_rotl32((k4 * c4) & _M32, 18) * c1) & _M32
        h4 ^= k4
        h4 = ((_rotl32(h4, 13) + h1) * 5 + 0x32AC3B17) & _M32

    tail = data[nblocks * 16:]
    n = len(tail)
    k1 = k2 = k3 = k4 = 0
    if n >= 15:
        k4 ^= tail[14] << 16
    if n >= 14:
        k4 ^= tail[13] << 8
    if n >= 13:
        k4 ^= tail[12]
        h4 ^= (_rotl32((k4 * c4) & _M32, 18) * c1) & _M32
    if n >= 12:
        k3 ^= tail[11] << 24
    if n >= 11:
        k3 ^= tail[10] << 16
    if n >= 10:
        k3 ^= tail[9] << 8
    if n >= 9:
        k3 ^= tail[8]
        h3 ^= (_rotl32((k3 * c3) & _M32, 17) * c4) & _M32
    if n >= 8:
        k2 ^= tail[7] << 24
    if n >= 7:
        k2 ^= tail[6] << 16
    if n >= 6:
        k2 ^= tail[5] << 8
    if n >= 5:
        k2 ^= tail[4]
        h2 ^= (_rotl32((k2 * c2) & _M32, 16) * c3) & _M32
    if n >= 4:
        k1 ^= tail[3] << 24
    if n >= 3:
        k1 ^= tail[2] << 16
    if n >= 2:
        k1 ^= tail[1] << 8
    if n >= 1:
        k1 ^= tail[0]
        h1 ^= (_rotl32((k1 * c1) & _M32, 15) * c2) & _M32

    h1 ^= length
    h2 ^= length
    h3 ^= length
    h4 ^= length
    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    h3 = (h3 + h1) & _M32
    h4 = (h4 + h1) & _M32
    h1, h2, h3, h4 = _fmix32(h1), _fmix32(h2), _fmix32(h3), _fmix32(h4)
    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    return h1 | (h2 << 32)


def shard_location(chunk_id: int, metadata: Mapping[str, Any]) -> tuple[str, int]:
    """``(shard filename, minishard number)`` for a key, per the sharding spec.

    The filename is lowercase hex zero-padded to ``ceil(shard_bits / 4)`` digits plus
    ``.shard``; ``shard_bits=0`` therefore gives ``0.shard``, the single-shard case.
    """
    preshift = int(metadata.get("preshift_bits", 0))
    minishard_bits = int(metadata["minishard_bits"])
    shard_bits = int(metadata["shard_bits"])

    hashed = int(chunk_id) >> preshift
    how = metadata.get("hash", "identity")
    if how == DEFAULT_HASH:
        hashed = murmurhash3_x86_128_low64(struct.pack("<Q", hashed & 0xFFFFFFFFFFFFFFFF))
    elif how != "identity":
        raise ValueError(f"unknown sharded hash {how!r}; expected 'identity' or "
                         f"{DEFAULT_HASH!r}")

    minishard = hashed & ((1 << minishard_bits) - 1)
    shard = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
    width = -(-shard_bits // 4) or 1
    return f"{shard:0{width}x}.shard", minishard


class ShardReader:
    """Locates and reads entries of a sharded index, by key *and* by byte offset.

    Exists for the one thing tensorstore's driver cannot do — say **where** in its shard
    an entry lives. The sharded multi-resolution mesh format needs that, because a body's
    mesh fragment data is not an indexed entry: it sits immediately before the manifest
    and is addressed relative to it. :func:`read_one` remains the right call for anything
    that only wants a value.

    The shard index (one small read per shard) and each minishard index are cached on the
    instance, so fetching many bodies costs one pair of index reads per shard touched
    rather than per body. Reuse one reader across a batch; it holds no store handles of
    its own beyond what :mod:`neu_vol.location` already caches per process.
    """

    def __init__(self, location: str | Mapping[str, Any], metadata: Mapping[str, Any],
                 *parts: str):
        self.location = location
        self.metadata = dict(metadata)
        self.parts = tuple(parts)
        self._index_length = (1 << int(self.metadata["minishard_bits"])) * 16
        self._shard_index: dict[str, Any] = {}
        self._minishard_index: dict[tuple[str, int], Any] = {}

    def read_range(self, shard: str, start: int, end: int) -> bytes | None:
        """Raw bytes ``[start, end)`` of one shard file, undecoded."""
        from .location import read_range

        if end <= start:
            return b""
        return read_range(self.location, start, end, *self.parts, shard)

    def _shard_table(self, shard: str):
        if shard not in self._shard_index:
            import numpy as np

            raw = self.read_range(shard, 0, self._index_length)
            if raw is None or len(raw) < self._index_length:
                # A shard file that does not exist holds no keys. Short is equally
                # fatal and equally not an error to the caller: either way there is
                # nothing to look up here.
                self._shard_index[shard] = None
            else:
                self._shard_index[shard] = np.frombuffer(raw, dtype="<u8").reshape(-1, 2)
        return self._shard_index[shard]

    def _minishard_table(self, shard: str, minishard: int):
        if (shard, minishard) in self._minishard_index:
            return self._minishard_index[(shard, minishard)]

        import gzip

        import numpy as np

        table = self._shard_table(shard)
        result = None
        if table is not None and minishard < len(table):
            # Offsets in both index levels are relative to the END of the shard index.
            start = int(table[minishard][0]) + self._index_length
            end = int(table[minishard][1]) + self._index_length
            if end > start:
                raw = self.read_range(shard, start, end)
                if raw:
                    if self.metadata.get("minishard_index_encoding", "raw") == "gzip":
                        raw = gzip.decompress(raw)
                    flat = np.frombuffer(raw, dtype="<u8")
                    n = flat.size // 3
                    # Three parallel arrays, each delta-encoded against a different
                    # baseline: ids against the previous id, starts against the previous
                    # END (hence the sizes term), sizes stored outright.
                    ids = np.cumsum(flat[0:n])
                    sizes = flat[2 * n:3 * n]
                    starts = np.cumsum(flat[n:2 * n])
                    starts[1:] += np.cumsum(sizes[:-1])
                    starts += self._index_length
                    result = np.stack([ids, starts, sizes], axis=1)
        self._minishard_index[(shard, minishard)] = result
        return result

    def locate(self, chunk_id: int) -> tuple[str, int, int] | None:
        """``(shard, offset, size)`` of the STORED entry, or ``None`` if the key is absent.

        ``size`` is the encoded length on disk, so with ``data_encoding='gzip'`` it is the
        compressed size — which is exactly what the mesh format's arithmetic wants, since
        the fragment data abuts the stored bytes rather than the decoded ones.
        """
        import numpy as np

        shard, minishard = shard_location(chunk_id, self.metadata)
        table = self._minishard_table(shard, minishard)
        if table is None:
            return None
        rows = np.nonzero(table[:, 0] == np.uint64(int(chunk_id)))[0]
        if not len(rows):
            return None
        return shard, int(table[rows[0], 1]), int(table[rows[0], 2])

    def read(self, chunk_id: int) -> bytes | None:
        """One entry by id, decoded per ``data_encoding``; ``None`` if absent."""
        import gzip

        found = self.locate(chunk_id)
        if found is None:
            return None
        shard, offset, size = found
        raw = self.read_range(shard, offset, offset + size)
        if raw is None:
            return None
        if self.metadata.get("data_encoding", "raw") == "gzip":
            raw = gzip.decompress(raw)
        return raw
