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
