"""Dask cluster startup for local or Rusty/SLURM execution.

Lifted from docs/dask-slurm-rusty.md §4 (the ``start_dask`` cookbook), verbatim
in behavior. The config's first ``jobqueue`` key selects the cluster type:
``local`` -> LocalCluster, ``slurm`` -> SLURMCluster (``scale`` submits sbatch).
The driver process must stay alive for the whole run; on exit the cluster (and
its SLURM jobs) are torn down.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

import dask
import yaml
from dask.distributed import Client

# Pin every worker to a single thread for numerical libs, so N single-threaded
# dask workers on a node don't each spawn a full BLAS thread-pool and oversubscribe.
_THREAD_PIN = [
    f"export {v}=1"
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
]


@contextlib.contextmanager
def start_dask(
    num_workers: int,
    config_path: str = "dask-config.yaml",
    label: str = "job",
    logger: logging.Logger | None = None,
    wait_timeout: int = 600,
) -> Iterator[Client]:
    """Start a Dask cluster from a YAML config and yield a connected Client.

    The config's first ``jobqueue`` key selects the cluster type:
      - ``"local"``: a LocalCluster with ``num_workers`` single-threaded workers.
      - ``"slurm"``: a SLURMCluster; ``scale(num_workers)`` submits sbatch jobs.
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
            logger.warning(
                "Not all workers up within %ss; proceeding with whatever is up "
                "(more will join as SLURM schedules them).",
                wait_timeout,
            )
        logger.info(
            "[%s] %d workers requested. Dashboard: %s",
            label,
            num_workers,
            client.dashboard_link,
        )
        yield client
    finally:
        client.shutdown()
        client.close()
