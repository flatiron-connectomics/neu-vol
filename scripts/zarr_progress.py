"""Point-in-time progress for a multiscale zarr v3 volume being written.

Counts written chunks per level (via the kvstore, no filesystem walk) against the
expected chunk count, so you can check a detached run from any terminal:

    pixi run -e dev python scripts/zarr_progress.py /mnt/ceph/.../05_yuri_v3.zarr

Assumes unsharded zarr v3 (chunk keys under "c/"). Levels are discovered by
probing 0, 1, 2, ... until one is missing (the group's OME metadata is only
written at the very end, so we don't rely on it). Re-run to refresh, or wrap in
`watch -n 30 ...`.
"""

from __future__ import annotations

import argparse
import math
import os

import tensorstore as ts

DEFAULT = "/path/to/data/eschomburg/sample3/05_yuri_v3.zarr"


def _open_level(group: str, i: int):
    try:
        return ts.open({"driver": "zarr3",
                        "kvstore": {"driver": "file", "path": os.path.join(group, str(i))}}).result()
    except Exception:
        return None


def _stored_chunks(group: str, i: int) -> int:
    kv = ts.KvStore.open({"driver": "file", "path": os.path.join(group, str(i)) + "/"}).result()
    return sum(1 for k in kv.list().result()
               if (k.decode() if isinstance(k, bytes) else str(k)).startswith("c/"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?", default=DEFAULT, help="path to the .zarr group")
    args = ap.parse_args()

    print(f"{args.group}\n{'level':>5} {'shape':>22} {'chunks':>16} {'done/total':>18} {'%':>7}")
    tot_done = tot_exp = 0
    i = 0
    while True:
        store = _open_level(args.group, i)
        if store is None:
            break
        shape = tuple(int(s) for s in store.shape)
        chunk = tuple(int(c) for c in store.chunk_layout.write_chunk.shape)
        expected = math.prod(math.ceil(s / c) for s, c in zip(shape, chunk))
        done = _stored_chunks(args.group, i)
        tot_done += done
        tot_exp += expected
        pct = 100.0 * done / expected if expected else 100.0
        print(f"{i:>5} {str(shape):>22} {str(chunk):>16} {f'{done}/{expected}':>18} {pct:>6.1f}%")
        i += 1

    if i == 0:
        print("no levels found yet (level 0 not created)")
    else:
        pct = 100.0 * tot_done / tot_exp if tot_exp else 100.0
        print(f"{'TOTAL':>5} {'':>22} {'':>16} {f'{tot_done}/{tot_exp}':>18} {pct:>6.1f}%")


if __name__ == "__main__":
    main()
