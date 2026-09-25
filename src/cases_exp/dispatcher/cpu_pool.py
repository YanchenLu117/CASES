"""CPU worker pool with the spec §2.4 discipline: BLAS single-thread, RAM-capped."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def worker_cap(total_cores: int, ram_gb: float, peak_gb_per_job: float) -> int:
    return min(192, int(0.70 * ram_gb / max(peak_gb_per_job, 1e-6)), total_cores)


class CPUPOOL:
    def __init__(self, n_workers: int, peak_gb_per_job: float = 1.0, ram_gb: float = 2015.0):
        self.n = min(n_workers, worker_cap(224, ram_gb, peak_gb_per_job))

    def map(self, fn: Callable, items: list[Any]) -> list[Any]:
        if self.n <= 1:
            return [fn(x) for x in items]
        with ProcessPoolExecutor(max_workers=self.n, initializer=_init_worker) as ex:
            return list(ex.map(fn, items, chunksize=max(1, len(items) // (self.n * 4))))
