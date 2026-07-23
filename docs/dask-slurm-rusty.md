# Parallel maps on Flatiron Rusty with Dask + SLURM

A portable, self-contained recipe for running an **embarrassingly-parallel
"apply `fn` to each of N independent items"** workload on the Flatiron **Rusty**
cluster, using `dask-jobqueue` to launch the workers as SLURM jobs. Drop this
file into any repo; everything you need is below (helper code, config templates,
launch/monitor steps, and the hard-won gotchas).

> **For the agent reading this:** this is a cookbook, not a framework. The
> "tooling" is one ~40-line helper plus a YAML config. Copy `start_dask` into the
> target repo, write a `dask-config.yaml`, express the work as a `dask.bag` map,
> smoke-test locally, then scale to SLURM. Read **Gotchas** before launching —
> several are silent failure modes.

---

## 1. When to use this (and when not to)

Use it when: the work is **many independent tasks** (one per file / object / chunk),
each doing nontrivial CPU work, and you want them spread across SLURM workers with
results collected back in the driver.

**Don't reach for dask if** the tasks are pure fire-and-forget with no in-Python
result passing — a plain `sbatch` array or Flatiron's **disBatch**
(<https://github.com/flatironinstitute/disBatch>) is simpler and lighter on the
scheduler (it runs *inside* an existing allocation). Dask earns its keep when you
want: in-memory result passing, multi-stage task graphs with dependencies,
futures/adaptive scaling, **or the same code running locally and on SLURM** by
swapping one config.

---

## 2. Mental model

- **Scheduler** runs in *your driver process* (on the workstation). It holds the
  task graph and hands tasks to workers.
- **Workers** are `dask-worker` processes. `dask_jobqueue.SLURMCluster.scale(N)`
  **generates an sbatch script and submits it** — the launched workers connect
  back to the scheduler over TCP/ethernet. You never call `sbatch` yourself; dask
  does. On driver exit, the jobs are `scancel`led automatically.
- The driver runs on the **workstation**, workers on **SLURM nodes**, connected
  over **ethernet — no InfiniBand needed** for loosely-coupled maps. **The driver
  must stay alive for the whole run** (use `tmux`/`nohup`/a background process); if
  it dies, the cluster and its SLURM jobs are torn down.

---

## 3. Dependencies

```
dask
distributed
dask-jobqueue
pyyaml
```

(Plus whatever your `fn` needs.) Pin versions if reproducibility matters;
`dask-jobqueue` ≥ 0.8 is fine.

---

## 4. The helper (`dask_slurm.py`) — copy this verbatim

Self-contained; no project-specific imports. Switches between a real SLURM
cluster and a local cluster based on the config's first `jobqueue` key, so the
same call works for dev and production.

```python
import contextlib
import logging
import yaml
import dask
from dask.distributed import Client

# Pin every worker to a single thread for numerical libs, so N single-threaded
# dask workers on a node don't each spawn a full BLAS thread-pool and oversubscribe.
_THREAD_PIN = [
    f"export {v}=1"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
]


@contextlib.contextmanager
def start_dask(num_workers, config_path="dask-config.yaml", label="job", logger=None,
               wait_timeout=600):
    """Start a Dask cluster from a YAML config and yield a connected Client.

    The config's first `jobqueue` key selects the cluster type:
      - "local": a LocalCluster with `num_workers` single-threaded workers.
      - "slurm": a SLURMCluster; `scale(num_workers)` submits sbatch jobs.
    Tears the cluster (and SLURM jobs) down on exit.
    """
    logger = logger or logging.getLogger(__name__)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cluster_type = next(iter(config["jobqueue"]))
    dask.config.update(dask.config.config, config)  # load YAML into dask's global config

    if cluster_type == "local":
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    elif cluster_type == "slurm":
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster(job_script_prologue=_THREAD_PIN)  # reads jobqueue.slurm.* from config
        cluster.scale(num_workers)                               # <-- submits sbatch jobs
    else:
        raise ValueError(f"unsupported cluster type: {cluster_type!r}")

    client = Client(cluster)
    try:
        try:
            client.wait_for_workers(num_workers, timeout=wait_timeout)
        except TimeoutError:
            logger.warning("Not all workers up within %ss; proceeding with whatever is up "
                           "(more will join as SLURM schedules them).", wait_timeout)
        logger.info("[%s] %d workers requested. Dashboard: %s",
                    label, num_workers, client.dashboard_link)
        yield client
    finally:
        client.shutdown()
        client.close()
```

---

## 5. The config (`dask-config.yaml`)

This is the only SLURM-specific thing you author. The `jobqueue.slurm` block maps
~1:1 onto `#SBATCH` flags.

### 5a. Production — `gen` partition (genoa: 96 cores / ~1.5 TB)

```yaml
jobqueue:
  slurm:
    cores: 96               # CPUs per SLURM job (--cpus-per-task); whole genoa node
    processes: 24           # dask workers per job  ->  cores/processes = threads/worker
    memory: 1400GB          # RAM per job (--mem);  memory/processes = ~58 GB per worker
    walltime: "12:00:00"    # per-job wall limit (-t); see Gotchas re: long runs

    account: your-account            # -A  (your center's account)
    queue: gen              # --partition
    job-extra-directives: ["--constraint=genoa"]   # arbitrary extra #SBATCH lines

    name: my-job            # job name in squeue
    log-directory: dask-job-logs   # worker stdout/err land here

distributed:
  scheduler:
    work-stealing: true
  worker:
    memory:
      # DISABLE dask's memory manager. Use `false`, NOT 0.0 — see Gotchas #1.
      target: false
      spill: false
      pause: false
      terminate: false
  admin:
    tick:
      limit: 3h             # don't warn about long GIL-free C calls
```

Then `start_dask(num_workers=48, ...)` → 2 jobs (48/24) on 2 nodes.

### 5b. Memory-bound work — `mem` partition (3–6 TB nodes, ~31 GB/core)

```yaml
jobqueue:
  slurm:
    cores: 96
    processes: 24           # ~120 GB/worker on a 3 TB node
    memory: 2900GB
    walltime: "12:00:00"
    account: your-account
    queue: mem
    job-extra-directives: ["--exclude=workermem02"]  # if you need uniform node sizes
    name: my-job
    log-directory: dask-job-logs
distributed: { worker: { memory: { target: false, spill: false, pause: false, terminate: false } } }
```

### 5c. Local (dev / smoke test) — no SLURM

```yaml
jobqueue:
  local: {}
distributed: { worker: { memory: { target: false, spill: false, pause: false, terminate: false } } }
```

**Knobs that matter:** `cores` = CPUs/job; `processes` = workers/job;
`memory/processes` = RAM/worker; `cores/processes` = threads/worker (keep at 1,
paired with the thread-pin prologue). To give each worker more RAM, **lower
`processes`** (keep `cores`/`memory` = whole node) — fewer, fatter workers.

---

## 6. The application pattern

Express the work as a map over a list. `fn` must be **top-level / importable**
(picklable) and ideally **idempotent** (see Gotchas #6, resume).

```python
import logging
import dask.bag as db
from dask_slurm import start_dask

logging.basicConfig(level=logging.INFO)

def work_fn(item):
    # runs on a worker; return a small result (write big outputs to shared storage)
    ...
    return (item, "ok")

def main(items, num_workers=48, config="dask-config.yaml"):
    n = max(1, min(len(items), num_workers * 10))   # ~10 partitions/worker; don't make 1 task/tiny-item
    bag = db.from_sequence(items, npartitions=n).map(work_fn)
    with start_dask(num_workers, config, label="my-job"):
        results = bag.compute()
    # ... summarize results in the driver ...
```

Keep the **driver lightweight**: compute the work-list (and any shared read-only
context) in the driver, pass each worker only what it needs, and have workers
**write their outputs to shared storage** (ceph) and return small status tuples —
don't ship large arrays back through the scheduler.

---

## 7. Smoke test, then scale (do this every time)

1. **Local, tiny:** point at the `local` config, `num_workers=2`, 3–5 items. Confirm
   `fn` runs, outputs land, results return. This catches pickling errors, path
   bugs, and logic bugs without touching SLURM.
2. **SLURM, small:** `gen` config, `num_workers` = one node's worth, full but
   modest item list. Confirm jobs start (`squeue`) and throughput is sane.
3. **SLURM, full:** scale to your CPU budget (see Gotchas #4).

---

## 8. Launch and monitor

```bash
# Launch from the workstation in a session that will outlive your terminal:
nohup python -u run.py > run.log 2>&1 &        # or run inside tmux
# Dashboard URL is printed by start_dask (http://<host>:8787/status).
```

Monitoring (read-only, and **don't poll in tight loops** — the SLURM service is
shared):

```bash
squeue -u "$USER" -o "%.18i %.22j %.8T %.11M %R"   # your jobs + state
sacct -u "$USER" --starttime now-1day -o JobID,State,MaxRSS,Elapsed  # post-hoc / OOM check
# progress = count of output files produced so far, not by hammering squeue
```

---

## 9. Gotchas (symptom → fix) — read before launching

1. **Workers hang, never run; logs say "Worker is at 0% memory usage. Pausing
   worker."** — `distributed.worker.memory` thresholds were set to **`0.0`**, which
   means "trigger at 0% usage" → pause immediately → deadlock. **Use `false`** to
   disable memory management, not `0.0`.

2. **Each worker spawns many threads / node is oversubscribed / slower than
   expected.** — numpy/BLAS multithreading. Ensure the **thread-pin prologue**
   (`OMP_NUM_THREADS=1`, etc.) is applied (`job_script_prologue` in the helper) and
   `cores == processes` (1 thread/worker).

3. **`KilledWorker` kills the whole run after a few retries.** — A worker was
   **OS-OOM-killed** (or hit the SLURM `--mem` cgroup limit). This is **uncatchable
   in-process**: dask retries the task on other workers, they die too, and the
   phase fails. dask's *memory management* (target/spill) does **not** help — it
   only governs dask-tracked managed memory, not arbitrary allocations (e.g. a big
   array read). Fixes, in order of preference:
   - **Bound per-task memory**: cap the input size each task loads; for oversized
     items, fall back to a coarser resolution or chunk + merge.
   - **Run the heavy part in a forked child** with `resource.setrlimit(RLIMIT_AS,
     cap)` so a runaway dies *catchably* in the child and you skip that item.
   - Give workers more RAM (lower `processes`).
   Keep `per_worker_RAM × tasks_in_flight` under the node limit.

4. **Jobs sit PENDING forever / only some start.** — You exceeded the **QOS CPU cap
   per user** (for `ccn`/`gen` it's ~200 CPUs ≈ 2 full nodes). `scale()` past it and
   the extra jobs pend. Size `num_workers` so `jobs × cores ≤ cap`.

5. **A node crashes mid-run and you lose everything.** — All workers were on one
   node. **Spread across ≥2 nodes** (`NODE_FAIL` resilience): e.g. `cores`=whole
   node and request 2+ jobs.

6. **A run dies/times out partway and you have to redo it all.** — Make tasks
   **idempotent**: at the top of `fn`, skip items whose output already exists; in
   the driver, filter the work-list to not-yet-done items. Then a failed run
   *resumes* by re-launching the same command. (Also lets you grow `walltime`
   budget across restarts.)

7. **Run exceeds `walltime` and SLURM kills the jobs.** — `walltime` is **per job**,
   and each job is resubmitted per dask "phase". Estimate `N_items × per_item_time
   / num_workers`; if it's near the limit, raise `walltime` and/or rely on the
   idempotent-resume from #6.

8. **Transient read errors / timeouts under many parallel workers.** — Dozens of
   workers hammering shared storage (ceph/GPFS) can momentarily time out. Add a
   **retry-with-backoff** around the read in `fn`.

9. **`scale()` is real job submission.** It issues `sbatch` and counts against your
   QOS. If your environment requires manual/interactive allocation as a checkpoint,
   treat launching the driver as that submission step. Monitor only with read-only
   `squeue`/`sacct`/`seff`/`scontrol show`.

10. **Too many / too few partitions.** One task per tiny item → scheduler overhead
    dominates; one giant partition → no parallelism. Aim for ~a few × `num_workers`
    partitions (or chunk items so each task is seconds-to-minutes of work).

---

## 10. Flatiron Rusty specifics (quick reference)

- **Account:** your center's (e.g. `ccn`). **Driver host:** a workstation (e.g.
  `ccnlin0xx`); workers are SLURM jobs, reachable over ethernet.
- **Partitions:** `gen` — genoa 96c/~1.5 TB (~16 GB/core), rome 128c/1 TB, icelake
  64c/1 TB; `mem` — 3–6 TB nodes (~31 GB/core) for memory-bound work. Pin a CPU
  type with `--constraint=genoa` etc.
- **QOS:** ~200 CPUs/user concurrent on `gen` (≈2 full nodes). Size `scale()`
  accordingly.
- **Filesystems:** code + envs on `/mnt/home` (GPFS, backed up, good for small
  files); bulk data + outputs on `/mnt/ceph`. Put `log-directory` and large outputs
  on ceph.
- **Modules vs conda/pixi:** don't mix Lmod modules with a conda/pixi env (ABI
  mismatch). Build your env one way.
- **SLURM etiquette:** the scheduler is shared — scope queries (`squeue -u $USER`),
  never poll in loops, and don't submit beyond the QOS.

---

## 11. Minimal end-to-end skeleton

```
myproject/
├── dask_slurm.py          # section 4, verbatim
├── dask-config.yaml       # section 5a (or 5c for local smoke test)
└── run.py                 # section 6: build items, define work_fn, main()
```

```bash
# 1. smoke test (edit run.py to use a local config + few items), then:
python run.py
# 2. full run on the workstation, surviving logout:
nohup python -u run.py > run.log 2>&1 &
squeue -u "$USER"
```

That's the whole toolkit. The helper + config are reusable as-is; the only
per-application work is `work_fn` and the item list.
