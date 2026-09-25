"""this baseline registry tests (§5)."""

import pytest

import numpy as np

from cases.baselines import available_baselines, create_baseline, get_baseline


def test_baseline_registry_names():
    names = available_baselines()
    # controlled block
    for n in ("a0_history", "a1_summary", "a2_scalar_diversity", "a3_gp_lse", "a4_full_cases",
              "gp_bo", "dkl_bo", "alde", "random"):
        assert n in names


def test_a0_history_repeats_best():
    prop = create_baseline("a0_history", {"history": {"a": 0.9, "b": 0.2, "c": 0.5}})
    chosen, _ = prop.select(pool=list("abc"), ids=["a", "b", "c"], k=2)
    assert tuple(chosen) == ("a", "c")  # best two by history


def test_gp_lse_baseline_runs():
    prop = create_baseline("a3_gp_lse", {})
    prop.update([[-1.0], [1.0]], [-1.0, 1.0])
    dom = np.linspace(-1, 1, 21)
    chosen, scores = prop.select(dom.tolist(), ids=[f"p{i}" for i in range(21)], k=2)
    assert len(chosen) == 2
    assert all(s > 0 for s in scores)


def test_gp_bo_baseline_runs():
    prop = create_baseline("gp_bo", {})
    prop.update([[-1.0], [1.0]], [-1.0, 1.0])
    dom = np.linspace(-1, 1, 21)
    chosen, scores = prop.select(dom.tolist(), ids=[f"p{i}" for i in range(21)], k=3)
    assert len(chosen) == 3
    assert len(scores) == 3


def test_external_arms_raise_honestly():
    # A1 is now a REAL arm; only these remain honest non-runnable shims
    for name in ("a4_full_cases", "dkl_bo", "alde"):
        prop = create_baseline(name, {})
        with pytest.raises(RuntimeError):
            prop.select(pool=[0.0], k=1)


def test_a1_summary_is_real_proposer():
    prop = create_baseline("a1_summary", {"token_cap": 40})
    prop.set_history({"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.8})
    chosen, _ = prop.select(pool=list("abcd"), ids=["a", "b", "c", "d"], k=2)
    assert tuple(chosen) == ("a", "d")  # top observed values replay
    s = prop.summarize()
    assert "a: 0.9" in s


def test_a2_scalar_diversity_selects_high_and_spread():
    prop = create_baseline("a2_scalar_diversity", {"alpha": 1.0, "lam": 0.0})
    prop.set_scalars({"p0": 0.9, "p1": 0.8, "p2": 0.1, "p3": 0.2})
    prop.set_features({"p0": [0, 0], "p1": [0, 1], "p2": [5, 5], "p3": [5, 6]})
    chosen, _ = prop.select(pool=None if False else [[0.9], [0.8], [0.1], [0.2]],
                            ids=["p0", "p1", "p2", "p3"], k=2)
    assert len(chosen) == 2
    assert chosen[0] == "p0"  # highest scalar first (lam=0)


def test_a2_diversity_bonus_rounds_out():
    # lam>0 should not just take the top scalar; with equal-ish scalars it spreads
    prop = create_baseline("a2_scalar_diversity", {"alpha": 0.5, "lam": 1.0})
    prop.set_scalars({"x": 0.6, "y": 0.59, "z": 0.58})
    prop.set_features({"x": [0.0], "y": [0.0], "z": [10.0]})
    chosen, _ = prop.select(pool=None, ids=["x", "y", "z"], k=2)
    assert chosen[0] == "x"  # highest scalar first
    assert chosen[1] == "z"  # diversity beats y (far from x) despite slightly lower scalar
