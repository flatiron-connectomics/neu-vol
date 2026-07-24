"""Driver-side, single-writer completion manifest.

The driver is the only writer, so there is no concurrency to coordinate: as
per-block results stream back it appends one JSON line per block recording its
status (``written`` | ``empty`` | ``skipped``). On resume it reads the manifest
and the ops filter out already-done blocks before dispatch — no per-object
existence checks, and empty (elided) chunks are remembered as done rather than
reprocessed. See docs/DESIGN.md.

Statuses:
  written  - a non-fill chunk was written
  empty    - the block equaled the fill value and was intentionally not written
  skipped  - a verify pass found the chunk already present in storage
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Sequence


class Manifest:
    """Append-only JSONL record of completed blocks, keyed by (level, index)."""

    def __init__(self, path: str | None):
        self.path = path
        self._done: dict[tuple[int, tuple[int, ...]], str] = {}
        self._fh = None

    # -- read -------------------------------------------------------------
    def load(self) -> "Manifest":
        self._done.clear()
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line from a hard crash
                    self._done[(rec["level"], tuple(rec["index"]))] = rec["status"]
        return self

    def done_indices(self, level: int) -> set[tuple[int, ...]]:
        return {idx for (lvl, idx) in self._done if lvl == level}

    def is_done(self, level: int, index: Sequence[int]) -> bool:
        return (level, tuple(index)) in self._done

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for status in self._done.values():
            out[status] = out.get(status, 0) + 1
        return out

    # -- write (driver only) ----------------------------------------------
    def reset(self) -> "Manifest":
        """Truncate the manifest (fresh run): clear in-memory + on-disk records."""
        self._done.clear()
        if self.path:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            open(self.path, "w").close()
        return self

    def _open(self) -> None:
        if self.path and self._fh is None:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "a")

    def record(self, level: int, results: Iterable[tuple[Sequence[int], str]]) -> None:
        """Record (index, status) results for a level; durable (flush+fsync)."""
        results = list(results)
        if self.path:
            self._open()
            for idx, status in results:
                self._fh.write(json.dumps({"level": level, "index": list(idx), "status": status}) + "\n")
        for idx, status in results:
            self._done[(level, tuple(idx))] = status
        if self._fh:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
