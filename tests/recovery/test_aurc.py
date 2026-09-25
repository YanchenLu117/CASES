"""P1 — AURC (Area Under Recovery-vs-Budget Curve), trapezoidal integral."""

from __future__ import annotations

import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


def test_aurc_sequence_linear():
    from cases.metrics import aurc

    # linear recovery [0,0.25,0.5,0.75,1.0] over unit budgets 0..4
    assert aurc([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.5)


def test_aurc_pairs_and_mapping():
    from cases.metrics import aurc

    pairs = aurc([(0, 0), (1, 0.5), (2, 1.0)])
    mapped = aurc({0: 0, 1: 0.5, 2: 1.0})
    assert pairs == pytest.approx(0.5)
    assert mapped == pytest.approx(0.5)


def test_aurc_edges():
    from cases.metrics import aurc

    assert aurc([0.5]) == 0.0          # fewer than two points
    assert aurc([1.0, 1.0, 1.0]) == pytest.approx(1.0)  # full immediate recovery
    assert aurc([0.0, 0.0, 0.0]) == pytest.approx(0.0)  # no recovery


def test_aurc_exported():
    import cases.metrics as mm
    assert hasattr(mm, "aurc")
