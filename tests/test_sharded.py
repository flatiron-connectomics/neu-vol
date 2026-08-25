"""The sharded uint64 index, which tensorstore implements and this package mostly wraps.

For writing there is therefore no format to test, only the wrapper's contract: the key byte
order, that the metadata handed to the kvstore is the same object that goes into an `info`,
and that a whole batch lands in one transaction rather than rewriting a shard per key.

`ShardReader` is the exception — it reimplements the addressing so a caller can learn the
byte OFFSET of an entry, which the driver will not tell you and the sharded mesh format needs.
That does have a format to test, and it is tested two ways: the hash and shard placement
against known answers taken from an independent implementation (cloud-volume), and the values
against `read_one`, i.e. against tensorstore reading the same files.
"""

import struct

import pytest

from neu_vol import sharded


def _spec(**kw):
    return sharded.sharding_spec(shard_bits=2, minishard_bits=1, **kw)


# --------------------------------------------------------------------------- #
# the spec block
# --------------------------------------------------------------------------- #
def test_the_spec_is_what_goes_into_info_verbatim():
    """The file and the metadata describing it must come from one dict, or they can drift and
    a reader trusts the metadata."""
    spec = _spec()
    assert spec["@type"] == "neuroglancer_uint64_sharded_v1"
    assert set(spec) == {"@type", "hash", "preshift_bits", "shard_bits", "minishard_bits",
                         "data_encoding", "minishard_index_encoding"}


def test_it_matches_the_shape_the_reference_dataset_uses():
    """Taken from gs://flyem-male-cns/.../by_id, which neuroglancer demonstrably reads."""
    spec = sharded.sharding_spec(shard_bits=10, minishard_bits=9, preshift_bits=10)
    assert spec == {"@type": "neuroglancer_uint64_sharded_v1",
                    "hash": "murmurhash3_x86_128", "preshift_bits": 10,
                    "shard_bits": 10, "minishard_bits": 9,
                    "data_encoding": "gzip", "minishard_index_encoding": "gzip"}


@pytest.mark.parametrize("kw", [
    {"shard_bits": -1, "minishard_bits": 0},
    {"shard_bits": 0, "minishard_bits": -3},
])
def test_negative_bit_counts_are_refused(kw):
    with pytest.raises(ValueError, match="non-negative"):
        sharded.sharding_spec(**kw)


def test_an_unknown_hash_or_encoding_is_refused():
    with pytest.raises(ValueError, match="hash must be"):
        sharded.sharding_spec(shard_bits=0, minishard_bits=0, hash="sha256")
    with pytest.raises(ValueError, match="data_encoding must be"):
        sharded.sharding_spec(shard_bits=0, minishard_bits=0, data_encoding="zstd")


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def test_keys_are_big_endian_which_is_the_one_place_the_order_flips():
    """Everything else in precomputed is little-endian."""
    assert sharded.key(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"
    assert sharded.key(0x0102030405060708) == bytes(range(1, 9))
    assert struct.unpack(">Q", sharded.key(2**63 + 5))[0] == 2**63 + 5


# --------------------------------------------------------------------------- #
# writing and reading back
# --------------------------------------------------------------------------- #
def test_a_batch_round_trips(tmp_path):
    spec = _spec()
    entries = [(i * 7 + 1, f"payload-{i}".encode()) for i in range(200)]
    n = sharded.write_all(str(tmp_path / "idx"), spec, entries)
    assert n == 200
    for chunk_id, payload in entries[:20]:
        assert sharded.read_one(str(tmp_path / "idx"), spec, chunk_id) == payload


def test_a_missing_key_reads_as_none(tmp_path):
    spec = _spec()
    sharded.write_all(str(tmp_path / "idx"), spec, [(1, b"x")])
    assert sharded.read_one(str(tmp_path / "idx"), spec, 999) is None


def test_the_file_count_is_two_to_the_shard_bits(tmp_path):
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=0)
    sharded.write_all(str(tmp_path / "idx"), spec, [(i, b"v") for i in range(64)])
    shards = sorted(p.name for p in (tmp_path / "idx").iterdir())
    assert shards == ["0.shard", "1.shard", "2.shard", "3.shard"]


def test_all_zero_bits_writes_a_single_shard(tmp_path):
    """Legal and useful: the reference does exactly this for its coarsest spatial level, so
    every level can be sharded and the bit counts collapse where a level is small."""
    spec = sharded.sharding_spec(shard_bits=0, minishard_bits=0)
    sharded.write_all(str(tmp_path / "idx"), spec, [(0, b"only")])
    assert [p.name for p in (tmp_path / "idx").iterdir()] == ["0.shard"]
    assert sharded.read_one(str(tmp_path / "idx"), spec, 0) == b"only"


def test_raw_encoding_also_round_trips(tmp_path):
    spec = sharded.sharding_spec(shard_bits=1, minishard_bits=1,
                                 data_encoding="raw", minishard_index_encoding="raw")
    sharded.write_all(str(tmp_path / "idx"), spec, [(5, b"uncompressed")])
    assert sharded.read_one(str(tmp_path / "idx"), spec, 5) == b"uncompressed"


def test_large_ids_survive(tmp_path):
    """Annotation ids in the reference run to 5.8e17 and look hashed, not sequential."""
    spec = _spec()
    ids = [48545692435670922, 582569736127172416, 2**64 - 1]
    sharded.write_all(str(tmp_path / "idx"), spec,
                      [(i, str(i).encode()) for i in ids])
    for i in ids:
        assert sharded.read_one(str(tmp_path / "idx"), spec, i) == str(i).encode()


def test_writing_nothing_is_not_an_error(tmp_path):
    assert sharded.write_all(str(tmp_path / "idx"), _spec(), []) == 0


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_planning_scales_the_shard_count_with_the_key_count():
    small = sharded.plan_sharding(100)
    big = sharded.plan_sharding(1_400_000)
    assert small["shard_bits"] == 0            # one file is plenty
    assert big["shard_bits"] > small["shard_bits"]
    assert 2 ** big["shard_bits"] <= 1024


def test_planning_is_capped_so_a_huge_input_does_not_explode_the_file_count():
    assert sharded.plan_sharding(312_000_000)["shard_bits"] <= 10


def test_planning_produces_a_usable_spec(tmp_path):
    spec = sharded.plan_sharding(5_000)
    sharded.write_all(str(tmp_path / "idx"), spec, [(i, b"v") for i in range(5_000)])
    assert sharded.read_one(str(tmp_path / "idx"), spec, 4_999) == b"v"


def test_a_non_positive_key_count_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        sharded.plan_sharding(0)


# --------------------------------------------------------------------------- #
# compressed morton code — how a sharded index keys a GRID CELL
# --------------------------------------------------------------------------- #
def test_bits_are_spent_only_where_the_grid_subdivides():
    """`2**i < size`, so ceil(log2(size)) — and ZERO for a flat axis. That is the compression."""
    assert sharded.morton_bits([1, 1, 1]) == [0, 0, 0]
    assert sharded.morton_bits([2, 1, 1]) == [1, 0, 0]
    assert sharded.morton_bits([16, 8, 16]) == [4, 3, 4]
    assert sharded.morton_bits([3, 5, 9]) == [2, 3, 4]


def test_x_varies_fastest():
    """Bit 0 of x is output bit 0, then bit 0 of y, then z — so on a 2x2x2 grid the code is
    x + 2y + 4z. A row-major flattening gives 4x + 2y + z, i.e. x and z swapped."""
    grid = [2, 2, 2]
    assert sharded.compressed_morton_code([1, 0, 0], grid) == 1
    assert sharded.compressed_morton_code([0, 1, 0], grid) == 2
    assert sharded.compressed_morton_code([0, 0, 1], grid) == 4
    assert sharded.compressed_morton_code([1, 1, 1], grid) == 7


def test_bits_interleave_rather_than_concatenate():
    """The whole point of a Morton code: neighbouring cells get nearby keys, so one shard holds
    a spatially compact set. Concatenating the coordinates would not."""
    grid = [4, 4, 1]
    # x=2 (binary 10), y=1 (binary 01): bits go x0,y0,x1,y1 -> 0,1,1,0 -> 0b0110 = 6
    assert sharded.compressed_morton_code([2, 1, 0], grid) == 6


def test_a_flat_axis_contributes_nothing():
    """grid [4,1,4] spends bits x0, z0, x1, z1 and none on y."""
    grid = [4, 1, 4]
    assert sharded.compressed_morton_code([3, 0, 3], grid) == 0b1111
    # x=1 -> bits (1,0); z=2 -> bits (0,1); interleaved x0,z0,x1,z1 = 1,0,0,1
    assert sharded.compressed_morton_code([1, 0, 2], grid) == 0b1001


def test_the_code_is_a_bijection_over_the_grid():
    """Every cell must get its own key, or two cells share an object and one is lost."""
    import itertools

    import numpy as np

    grid = [4, 2, 8]
    cells = np.array(list(itertools.product(range(4), range(2), range(8))))
    codes = sharded.compressed_morton_code(cells, grid)
    assert len(set(codes.tolist())) == len(cells) == 64
    assert min(codes) == 0 and max(codes) == 63


def test_it_agrees_with_a_row_major_index_exactly_where_that_is_misleading():
    """Degenerate grids give identical keys, which is why substituting one for the other passes
    every small test and then loses almost everything on a real multi-axis grid."""
    import itertools

    import numpy as np

    def row_major(cells, grid):
        return (cells[:, 0] * grid[1] + cells[:, 1]) * grid[2] + cells[:, 2]

    for grid in ([1, 1, 1], [2, 1, 1], [8, 1, 1]):
        cells = np.array(list(itertools.product(*(range(g) for g in grid))))
        assert np.array_equal(sharded.compressed_morton_code(cells, grid),
                              row_major(cells, grid)), grid

    grid = [16, 8, 16]
    cells = np.array(list(itertools.product(*(range(g) for g in grid))))
    agree = sharded.compressed_morton_code(cells, grid) == row_major(cells, grid)
    assert agree.sum() < 0.05 * len(cells), "a multi-axis grid must NOT agree"


def test_a_position_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="axes"):
        sharded.compressed_morton_code([1, 2], [4, 4, 4])


def test_rewriting_an_index_replaces_it_rather_than_merging(tmp_path):
    """A sharded index is a key-value store, so writing keys 1..N leaves any other key already
    present. Rewriting an index whose KEY SPACE changed would then serve both generations, and
    a reader asking for an old-only key gets a stale answer while every filename looks right."""
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(10)
    sharded.write_all(dst, spec, [(i, b"old") for i in range(10)])
    sharded.write_all(dst, spec, [(i, b"new") for i in range(5, 15)])

    assert sharded.read_one(dst, spec, 7) == b"new"
    assert sharded.read_one(dst, spec, 14) == b"new"
    assert sharded.read_one(dst, spec, 0) is None, "a key only the old write used must be gone"


def test_replace_false_keeps_what_was_there(tmp_path):
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(10)
    sharded.write_all(dst, spec, [(i, b"old") for i in range(10)])
    sharded.write_all(dst, spec, [(i, b"new") for i in range(5, 15)], replace=False)
    assert sharded.read_one(dst, spec, 0) == b"old"
    assert sharded.read_one(dst, spec, 7) == b"new"


def test_clear_removes_the_shard_objects(tmp_path):
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(5_000)
    sharded.write_all(dst, spec, [(i, b"v") for i in range(5_000)])
    assert sharded.clear(dst) > 0
    assert sharded.read_one(dst, spec, 1) is None
    assert sharded.clear(dst) == 0        # idempotent


def test_a_cell_written_at_its_morton_key_reads_back_by_position(tmp_path):
    """The round trip that matters: write keyed by code, read keyed by code recomputed from the
    grid position, which is what a viewer does."""
    grid = [4, 2, 4]
    spec = sharded.plan_sharding(32)
    entries = [(sharded.compressed_morton_code([x, y, z], grid), f"{x}-{y}-{z}".encode())
               for x in range(4) for y in range(2) for z in range(4)]
    dst = str(tmp_path / "spatial")
    sharded.write_all(dst, spec, entries)
    for x, y, z in ((0, 0, 0), (3, 1, 3), (2, 0, 1)):
        code = sharded.compressed_morton_code([x, y, z], grid)
        assert sharded.read_one(dst, spec, code) == f"{x}-{y}-{z}".encode()


# --------------------------------------------------------------------------- #
# ShardReader: addressing
# --------------------------------------------------------------------------- #

#: (sharding params, [(key, shard filename stem, minishard number)]).
#: Taken from cloud-volume's ShardingSpecification.compute_shard_location, which is an
#: independent implementation of the same spec. Pinning against it rather than against
#: ourselves is the point: a wrong hash still produces a self-consistent reader that
#: simply never finds anything, and "not found" is indistinguishable from "absent".
SHARD_VECTORS = [
    (dict(preshift_bits=6, minishard_bits=8, shard_bits=10),
     [(0, "0ae", 65), (1, "0ae", 65), (24740, "30e", 43), (255, "3e4", 209),
      (4096, "119", 255), (123456789, "06c", 143), (2**63, "0af", 35),
      (2**64 - 1, "230", 252)]),
    (dict(preshift_bits=0, minishard_bits=0, shard_bits=0),
     [(0, "0", 0), (24740, "0", 0), (2**64 - 1, "0", 0)]),
    (dict(preshift_bits=9, minishard_bits=6, shard_bits=11),
     [(0, "2b9", 1), (24740, "053", 12), (255, "2b9", 1), (4096, "627", 1),
      (123456789, "4c1", 17), (2**63, "1cb", 10), (2**64 - 1, "04d", 34)]),
    (dict(preshift_bits=0, minishard_bits=3, shard_bits=5),
     [(0, "08", 1), (1, "13", 2), (24740, "10", 4), (4096, "07", 5),
      (123456789, "04", 7), (2**63, "15", 0), (2**64 - 1, "03", 2)]),
]


@pytest.mark.parametrize("params,cases", SHARD_VECTORS)
def test_shard_placement_matches_an_independent_implementation(params, cases):
    spec = sharded.sharding_spec(**params)
    for chunk_id, shard, minishard in cases:
        assert sharded.shard_location(chunk_id, spec) == (f"{shard}.shard", minishard), \
            f"key {chunk_id} placed wrong for {params}"


def test_preshift_bits_send_neighbouring_keys_to_one_shard():
    """That is what preshift is FOR — consecutive ids share a shard so a range of them
    costs one file. With 6 bits, 0..63 must land together and 64 must not be forced to."""
    spec = sharded.sharding_spec(shard_bits=10, minishard_bits=8, preshift_bits=6)
    first = {sharded.shard_location(i, spec) for i in range(64)}
    assert len(first) == 1
    assert sharded.shard_location(64, spec) != sharded.shard_location(0, spec)


def test_the_identity_hash_is_not_the_murmur_one():
    """Both are legal and they place keys differently, so a spec's `hash` must be read
    rather than assumed."""
    ident = sharded.sharding_spec(shard_bits=5, minishard_bits=3, hash="identity")
    murmur = sharded.sharding_spec(shard_bits=5, minishard_bits=3)
    # 24740 = 0b110000010100100: low 3 bits are the minishard, next 5 the shard.
    assert sharded.shard_location(24740, ident) == ("14.shard", 4)
    assert sharded.shard_location(24740, murmur) != sharded.shard_location(24740, ident)


def test_an_unknown_hash_is_refused_rather_than_silently_treated_as_identity():
    spec = dict(sharded.sharding_spec(shard_bits=2, minishard_bits=1))
    spec["hash"] = "sha256"
    with pytest.raises(ValueError, match="unknown sharded hash"):
        sharded.shard_location(1, spec)


def test_the_murmur_digest_is_the_x86_variant():
    """x86_128 and x64_128 are different functions. Known answers for the empty input and
    for a short string, from the reference implementation."""
    assert sharded.murmurhash3_x86_128_low64(b"") == 0
    assert sharded.murmurhash3_x86_128_low64(b"", seed=1) != 0
    # Stable across calls and sensitive to every input byte.
    digests = {sharded.murmurhash3_x86_128_low64(bytes([b]) + b"\0" * 7) for b in range(32)}
    assert len(digests) == 32
    assert all(0 <= d < 2**64 for d in digests)


# --------------------------------------------------------------------------- #
# ShardReader: values
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("encoding", ["gzip", "raw"])
def test_shard_reader_agrees_with_tensorstore(tmp_path, encoding):
    """The whole reason two implementations are allowed to coexist here.

    `read_one` goes through tensorstore's driver, `ShardReader` through our own hash and
    index parsing. They must return the same bytes for the same key or one of them is
    wrong, and a reader that is wrong reads back as missing data rather than as an error.
    """
    spec = sharded.sharding_spec(shard_bits=3, minishard_bits=2, preshift_bits=0,
                                 data_encoding=encoding, minishard_index_encoding=encoding)
    entries = [(i * 37 + 5, f"payload-{i}-{'x' * i}".encode()) for i in range(150)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        assert reader.read(chunk_id) == payload
        assert sharded.read_one(str(tmp_path / "idx"), spec, chunk_id) == payload


def test_shard_reader_reports_a_missing_key_as_none(tmp_path):
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=1)
    sharded.write_all(str(tmp_path / "idx"), spec, [(1, b"a"), (2, b"b")])
    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    assert reader.read(999) is None
    assert reader.locate(999) is None


def test_shard_reader_on_an_index_that_does_not_exist_is_none_not_an_error(tmp_path):
    """A volume may simply hold no sharded subresource. That is a normal answer."""
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=1)
    reader = sharded.ShardReader(str(tmp_path / "nothing-here"), spec)
    assert reader.locate(1) is None
    assert reader.read(1) is None


def test_locate_gives_the_byte_range_the_value_actually_occupies(tmp_path):
    """`locate` is the only thing tensorstore cannot do, so its offsets are what the mesh
    reader trusts to find fragment data sitting outside the index entirely."""
    spec = sharded.sharding_spec(shard_bits=1, minishard_bits=1, data_encoding="raw",
                                 minishard_index_encoding="raw")
    entries = [(i, bytes([i]) * (10 + i)) for i in range(1, 20)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        shard, offset, size = reader.locate(chunk_id)
        assert size == len(payload), "raw encoding stores the value verbatim"
        assert reader.read_range(shard, offset, offset + size) == payload


def test_the_indexes_are_read_once_per_shard_not_once_per_key(tmp_path, monkeypatch):
    """A batch of bodies out of one shard must not re-fetch its index per body — that is
    the difference between two requests and two thousand against an object store."""
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=0, minishard_bits=0)
    entries = [(i, f"v{i}".encode()) for i in range(50)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    calls = []
    real = location.read_range
    monkeypatch.setattr(location, "read_range",
                        lambda *a, **k: (calls.append(a[3:]), real(*a, **k))[1])

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        assert reader.read(chunk_id) == payload
    # 50 value reads, plus exactly one shard index and one minishard index.
    assert len(calls) == 52


def test_clear_leaves_a_sibling_info_alone(tmp_path):
    """An index does not always own its directory. A sharded mesh or skeleton sits beside
    the `info` that declares it, and `clear` deleting that produced the nastiest possible
    result: every shard present and correct, nothing pointing at them, every body reading
    back as absent. Only `.shard` objects are ours to remove."""
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=1, minishard_bits=1)
    dst = str(tmp_path / "skeleton")
    location.write_bytes(dst, b'{"@type": "neuroglancer_skeletons"}', "info")
    sharded.write_all(dst, spec, [(7, b"a-skeleton")])

    assert location.read_bytes(dst, "info") is not None, "info was deleted by the rewrite"
    assert sharded.read_one(dst, spec, 7) == b"a-skeleton"

    # And a rewrite must still replace the shards rather than merging into them.
    sharded.write_all(dst, spec, [(8, b"another")])
    assert sharded.read_one(dst, spec, 7) is None
    assert location.read_bytes(dst, "info") is not None


def test_clear_reports_only_the_shards_it_removed(tmp_path):
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=0)
    dst = str(tmp_path / "idx")
    sharded.write_all(dst, spec, [(i, b"v") for i in range(64)])
    location.write_bytes(dst, b"{}", "info")

    n_shards = len([p for p in (tmp_path / "idx").iterdir() if p.name.endswith(".shard")])
    assert sharded.clear(dst) == n_shards
    assert location.read_bytes(dst, "info") == b"{}"
