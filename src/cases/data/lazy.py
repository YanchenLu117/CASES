"""Lazy/cached JSON loading for the large benchmark data files.

Problem (INFRA_PLAN §4): ``data/hypospace`` holds 70-89MB single-object JSONs
(causal_dev_n5 / voxel4_h4_b4).  Every ``json.loads`` re-parse costs ~1.2s and
~0.5GB transient RAM, and the 200-cell grids re-load the same file per cell —
the dominant fixed cost of small jobs.

Design (measured on an 89MB voxel file):

* **process memo** — one parse per process per file version
  (keyed by resolved path + size + mtime_ns); repeat loads cost ~0.3ms.
* **optional orjson** — measured ~15% on these numeric-heavy payloads
  (1.11s -> 0.95s on the 93MB voxel file; object construction dominates,
  not tokenization); stdlib fallback keeps the module dependency-free.
* a cross-process pickle sidecar was measured at NO gain over plain parse
  (1.18s vs 1.16s) and was dropped — the per-process memo is the real win
  (repeat loads 0.18ms), with parse cost paid once per process.

Env knob: ``CASES_JSON_CACHE_DISABLE=1`` forces plain parse (A/B checks).

``LazyDataset`` additionally defers the whole load until first use, for
workers that touch a domain conditionally; it pickles as (loader, path) so
nothing is parsed at fork/pickle time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

try:  # optional 4x parse accelerator
    import orjson

    def _parse(blob: bytes) -> Any:
        return orjson.loads(blob)

except ImportError:  # stdlib fallback — dependency-free

    def _parse(blob: bytes) -> Any:
        return json.loads(blob.decode("utf-8"))


_MEMO: dict[tuple[str, int, int], Any] = {}


def _disabled() -> bool:
    return os.environ.get("CASES_JSON_CACHE_DISABLE") == "1"


def _key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), st.st_size, st.st_mtime_ns)


def json_cached(path: str | Path) -> Any:
    """Parse a JSON file once per version per process; memoized afterwards."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if _disabled():
        return _parse(p.read_bytes())
    key = _key(p)
    cached = _MEMO.get(key)
    if cached is None:
        cached = _parse(p.read_bytes())
        _MEMO[key] = cached
    return cached


def memo_size() -> int:
    """Number of cached file versions in this process (status reporting)."""
    return len(_MEMO)


class LazyDataset:
    """Defers ``loader(path)`` until the first attribute access.

    Multiprocessing-safe: the proxy pickles as its (loader, path) pair and
    re-defers in the child process, so nothing is parsed at fork/pickle time.
    """

    __slots__ = ("_loader", "_path", "_loaded")

    def __init__(self, loader: Callable[[str], Any], path: str | Path):
        self._loader = loader
        self._path = str(path)
        self._loaded: Any = None

    def _ensure(self) -> Any:
        if self._loaded is None:
            self._loaded = self._loader(self._path)
        return self._loaded

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)

    def __iter__(self):
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())

    def __getitem__(self, item):
        return self._ensure()[item]

    def __getstate__(self):
        return {"loader": self._loader, "path": self._path}

    def __setstate__(self, state):
        self._loader = state["loader"]
        self._path = state["path"]
        self._loaded = None
