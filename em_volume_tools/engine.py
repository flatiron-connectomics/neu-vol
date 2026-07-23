"""Block-map engine.

Volume operations are expressed as a map over **output chunks**: tile the output
grid into :class:`Block`s, and run a per-block function that reads the input
region it needs, transforms it, and writes its output chunk. Blocks are the
``items`` for the dask.bag pattern in docs/dask-slurm-rusty.md.

Design rules (docs/DESIGN.md §4): tasks are idempotent (skip already-written
blocks -> resume-by-relaunch), workers write to the store and return small status
tuples (never large arrays through the scheduler).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterator, Sequence

# A per-axis tuple of slices bounding a block in a volume's index space.
Region = tuple[slice, ...]


@dataclass(frozen=True)
class Block:
    """One output chunk: its grid index and its region in the output volume."""

    index: tuple[int, ...]   # position in the chunk grid, per axis
    region: Region           # slices into the output volume

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.region)


def iter_blocks(shape: Sequence[int], chunks: Sequence[int]) -> Iterator[Block]:
    """Tile ``shape`` into blocks of ``chunks`` (edge blocks are clipped).

    Axis order is whatever ``shape``/``chunks`` use (canonical ``(c,) z, y, x``).
    """
    if len(shape) != len(chunks):
        raise ValueError(f"shape {tuple(shape)} and chunks {tuple(chunks)} differ in rank")
    if any(c <= 0 for c in chunks):
        raise ValueError(f"chunks must be positive, got {tuple(chunks)}")
    grid = [-(-s // c) for s, c in zip(shape, chunks)]  # ceil division per axis
    for index in product(*(range(g) for g in grid)):
        region = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(index, chunks, shape)
        )
        yield Block(index=index, region=region)


def block_map(
    blocks: Sequence[Block],
    fn: Callable[[Block], Any],
    *,
    client: Any | None = None,
    npartitions: int | None = None,
) -> list[Any]:
    """Apply ``fn`` to each block, returning the per-block results.

    ``fn`` must be top-level / picklable and should return a *small* status value
    (write big outputs to the store, not back through the scheduler).

    If ``client`` is ``None``, runs serially in-process (ideal for smoke tests).
    Otherwise dispatches via ``dask.bag`` under the active distributed client
    (see :func:`em_volume_tools.dask_runner.start_dask`).
    """
    blocks = list(blocks)
    if client is None:
        return [fn(b) for b in blocks]

    import dask.bag as db

    n = npartitions or len(blocks)
    n = max(1, min(n, len(blocks)))
    bag = db.from_sequence(blocks, npartitions=n).map(fn)
    return list(bag.compute())


def idempotent(
    fn: Callable[[Block], Any],
    is_done: Callable[[Block], bool],
    *,
    skipped: Any = "skipped",
) -> Callable[[Block], Any]:
    """Wrap ``fn`` so blocks that are already done are skipped.

    Enables resume-by-relaunch. ``is_done`` runs on the worker, so keep it cheap
    (e.g. a store existence/metadata check).
    """

    def wrapped(block: Block) -> Any:
        if is_done(block):
            return skipped
        return fn(block)

    return wrapped
