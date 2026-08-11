"""Put `tmp_path` on tmpfs when one is available.

These suites are dominated by **fsync latency, not computation** — wall clock ran
4x the CPU time. Every block a run completes gets one `Manifest.record`, and each
of those flushes and fsyncs to make progress durable, which is exactly right in
production and pure overhead against a throwaway volume in a temp directory.

The default temp root decides what that costs, and on a Flatiron workstation the
default is the slow choice: `/tmp` is RAID-backed xfs at ~8 ms per fsync, against
~0.004 ms on `/dev/shm`. At the few thousand blocks a full suite writes that is the
difference between minutes and seconds, and none of it is work anyone is testing.

`PYTEST_DEBUG_TEMPROOT` is the knob rather than `--basetemp` on purpose: pytest
reads it inside `getbasetemp()`, so numbered per-run directories, the retention
policy and the sweep of old runs all keep working exactly as before — the tree just
lands somewhere faster. `--basetemp`, by contrast, `rm -rf`s the path it is given at
session start and retains nothing, so a failed run's artifacts are gone.

Escape hatches, in the order they win: an explicit `--basetemp` (pytest ignores the
temproot entirely), an inherited `PYTEST_DEBUG_TEMPROOT`, and `EM_TESTS_TMPFS=0` to
force the platform default.

Duplicated verbatim in all three em-* repos, which are separate git repos and must
stay independently testable — a shared copy would mean a test-time import across the
layering. Keep the copies in step.
"""

import os
from pathlib import Path

# tmpfs spends real memory, so only use one with room to spare. A whole suite leaves
# well under a hundred MB, counting the runs pytest retains.
_MIN_FREE_BYTES = 2 * 1024**3

_CANDIDATES = ("/dev/shm", "/run/shm")


def _tmpfs_root() -> str | None:
    """A writable tmpfs directory with room to spare, or None to leave the default."""
    if os.environ.get("EM_TESTS_TMPFS") == "0":
        return None
    for candidate in _CANDIDATES:
        root = Path(candidate)
        if not root.is_dir() or not os.access(root, os.W_OK):
            continue
        try:
            st = os.statvfs(root)
        except OSError:
            continue
        if st.f_bavail * st.f_frsize < _MIN_FREE_BYTES:
            continue
        # Per-user, because tmpfs is node-wide and pytest insists on owning its root.
        mine = root / f"em-tests-{os.getuid()}"
        try:
            mine.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            continue
        return str(mine)
    return None


_root = _tmpfs_root()
if _root is not None:
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", _root)
