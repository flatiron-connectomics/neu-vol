import pytest

from em_volume_tools import Block, block_map, idempotent, iter_blocks


def test_iter_blocks_exact_tiling():
    blocks = list(iter_blocks(shape=(4, 4), chunks=(2, 2)))
    assert len(blocks) == 4
    assert {b.index for b in blocks} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    b = next(b for b in blocks if b.index == (1, 1))
    assert b.region == (slice(2, 4), slice(2, 4))
    assert b.shape == (2, 2)


def test_iter_blocks_clips_edges():
    blocks = list(iter_blocks(shape=(5, 3), chunks=(2, 2)))
    # ceil(5/2)=3 x ceil(3/2)=2 = 6 blocks
    assert len(blocks) == 6
    edge = next(b for b in blocks if b.index == (2, 1))
    assert edge.region == (slice(4, 5), slice(2, 3))
    assert edge.shape == (1, 1)


def test_iter_blocks_covers_volume_exactly():
    shape, chunks = (10, 7, 13), (4, 4, 4)
    covered = 0
    for b in iter_blocks(shape, chunks):
        covered += b.shape[0] * b.shape[1] * b.shape[2]
    assert covered == 10 * 7 * 13


def test_iter_blocks_rank_mismatch_raises():
    with pytest.raises(ValueError):
        list(iter_blocks(shape=(4, 4), chunks=(2, 2, 2)))


def test_block_map_serial():
    blocks = list(iter_blocks(shape=(4, 4), chunks=(2, 2)))
    out = block_map(blocks, lambda b: b.index, client=None)
    assert sorted(out) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_idempotent_skips_done_blocks():
    blocks = list(iter_blocks(shape=(4,), chunks=(1,)))  # 4 blocks
    done = {(0,), (2,)}
    fn = idempotent(lambda b: "ran", is_done=lambda b: b.index in done)
    out = block_map(blocks, fn)
    assert out.count("skipped") == 2
    assert out.count("ran") == 2
