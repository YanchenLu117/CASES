"""Shared pytest configuration for the CASES test suite (T3 tech-debt).

Unifies the previously-per-file ``sys.path.insert`` hacks: this single
``conftest.py`` guarantees the repo ``src/`` is on ``sys.path`` for every test
module regardless of the working directory pytest is invoked from, so individual
test files no longer need their own ``sys.path`` mutation (which was fragile and
repeated in six files).  It has no fixtures of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
