"""The sharded uint64 index, which tensorstore implements and this package only wraps.

What is worth testing is therefore not the format but the wrapper's contract: the key byte
order, that the metadata handed to the kvstore is the same object that goes into an `info`,
and that a whole batch lands in one transaction rather than rewriting a shard per key.
"""

import struct

import pytest

from em_volume_tools import sharded


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
