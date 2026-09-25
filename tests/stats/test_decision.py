"""P0-b tests: statistical decision layer (Plan §3.1/§12, Detail §14)."""

import numpy as np
import pytest

from cases.stats import (
    DECISION_EQUIVALENT,
    DECISION_HARMFUL,
    DECISION_IMPROVES,
    DECISION_INCONCLUSIVE,
    bootstrap_ci,
    cluster_bootstrap_ci,
    decide,
    hierarchical_bootstrap_ci,
    holm_adjust,
    paired_summary,
)


def test_holm_adjust_known_values() -> None:
    raw = [0.9, 0.02, 0.001, 0.04]
    assert holm_adjust(raw) == pytest.approx([0.9, 0.06, 0.004, 0.08])


def test_holm_adjust_enforces_monotonicity() -> None:
    # 0.03*2 = 0.06 then 0.05*1 = 0.05 -> running max keeps 0.06
    assert holm_adjust([0.05, 0.03]) == pytest.approx([0.06, 0.06])


def test_holm_adjust_validates_input() -> None:
    with pytest.raises(ValueError):
        holm_adjust([0.5, 1.5])
    with pytest.raises(ValueError):
        holm_adjust([])


def test_decide_improves() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(0.3, 0.05, size=30)
    d = decide(
        [x + 5.0 for x in deltas],
        [5.0 for _ in deltas],
        delta_min=0.05,
        delta_eq=0.05,
        delta_harm=0.05,
        seed=11,
    )
    assert d.label == DECISION_IMPROVES
    assert d.summary.ci_low > 0.05
    assert d.holm_p < 0.05


def test_decide_materially_harmful() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(-0.3, 0.05, size=30)
    d = decide(
        [0.0 for _ in deltas],
        [-x for x in deltas],
        delta_min=0.05,
        delta_eq=0.05,
        delta_harm=0.05,
        seed=11,
    )
    assert d.label == DECISION_HARMFUL
    assert d.summary.ci_high < -0.05


def test_decide_equivalent() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.normal(0.001, 0.005, size=40)
    d = decide(
        [1.0 + x for x in deltas],
        [1.0 for _ in deltas],
        delta_min=0.05,
        delta_eq=0.05,
        delta_harm=0.05,
        seed=5,
    )
    assert d.label == DECISION_EQUIVALENT
    assert d.tost_low_p < 0.05 and d.tost_high_p < 0.05


def test_decide_inconclusive_wide_noise() -> None:
    rng = np.random.default_rng(21)
    deltas = rng.normal(0.05, 0.5, size=12)
    d = decide(
        [x for x in deltas],
        [0.0 for _ in deltas],
        delta_min=0.05,
        delta_eq=0.01,
        delta_harm=0.05,
        seed=3,
    )
    assert d.label == DECISION_INCONCLUSIVE


def test_family_holm_p_gates_improves() -> None:
    rng = np.random.default_rng(9)
    deltas = rng.normal(0.06, 0.1, size=30)
    # build a 40-test family where this contrast is the smallest raw p: Holm
    # multiplier 40 pushes holm_p above 0.05 while the CI alone would pass
    d_probe = decide([x for x in deltas], [0.0 for _ in deltas], delta_min=0.02, delta_eq=0.02, delta_harm=0.05, seed=13)
    family = [d_probe.raw_p] + [0.05 + 0.01 * k for k in range(39)]
    d = decide(
        [x for x in deltas],
        [0.0 for _ in deltas],
        delta_min=0.02,
        delta_eq=0.02,
        delta_harm=0.05,
        family_pvalues=family,
        seed=13,
    )
    assert d.raw_p < 0.05
    assert d.holm_p == pytest.approx(holm_adjust(family)[family.index(d_probe.raw_p)])
    assert d.holm_p > 0.05
    assert d.holm_p >= d.raw_p
    assert d.label != DECISION_IMPROVES


def test_harmful_takes_precedence_over_equivalent() -> None:
    # R4 Major-1: with delta_eq > delta_harm a strongly negative effect can
    # satisfy both TOST and the harm bound — harm MUST win the label
    deltas = [-0.3] * 12
    d = decide(
        [x for x in deltas],
        [0.0 for _ in deltas],
        delta_min=0.05,
        delta_eq=0.5,
        delta_harm=0.1,
        seed=4,
    )
    assert d.label == DECISION_HARMFUL


def test_permutation_p_floor() -> None:
    deltas = np.full(12, 0.5)
    from cases.stats.decision import paired_permutation_pvalue

    p = paired_permutation_pvalue(deltas, seed=1)
    # exact enumeration at n<=16: only the all-+ and all-− patterns reach |mean|
    assert p == 2 / (1 << 12)


def test_zero_variance_constant_improves() -> None:
    d = decide([5.3] * 15, [5.0] * 15, delta_min=0.1, delta_eq=0.1, delta_harm=0.1, seed=2)
    assert d.label == DECISION_IMPROVES
    assert d.summary.ci_low == pytest.approx(0.3)
    assert d.summary.ci_high == pytest.approx(0.3)


def test_cluster_bootstrap_equals_preaveraged_instances() -> None:
    values = [1.0, 1.2, 0.8, 2.0, 2.4, 3.0, 3.2]
    clusters = ["i1", "i1", "i1", "i2", "i2", "i3", "i3"]
    per_instance = [np.mean([1.0, 1.2, 0.8]), np.mean([2.0, 2.4]), np.mean([3.0, 3.2])]
    low_a, high_a = cluster_bootstrap_ci(values, clusters, n_boot=5000, seed=42)
    low_b, high_b = bootstrap_ci(per_instance, n_boot=5000, seed=42)
    assert (low_a, high_a) == pytest.approx((low_b, high_b))


def test_hierarchical_bootstrap_zero_variance_is_exact() -> None:
    values = [4.0] * 9
    groups = ["t1"] * 4 + ["t2"] * 3 + ["t3"] * 2
    units = [f"{g}s{k}" for g, k in zip(groups, [1, 2, 3, 4, 1, 2, 3, 1, 2])]
    low, high = hierarchical_bootstrap_ci(values, groups, units, n_boot=2000, seed=0)
    assert low == pytest.approx(4.0) and high == pytest.approx(4.0)


def test_summary_reports_all_detail_14_fields() -> None:
    rng = np.random.default_rng(1)
    cases = list(rng.normal(6.0, 0.2, size=25))
    base = list(rng.normal(5.0, 0.2, size=25))
    s = paired_summary(cases, base, seed=4)
    assert s.n == 25
    assert s.mean == pytest.approx(1.0, abs=0.15)
    assert s.ci_low < s.mean < s.ci_high
    assert 0.0 <= s.permutation_p <= 1.0
    assert s.smd > 0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        decide([0.1], [0.0], delta_min=0.05, delta_eq=0.05, delta_harm=0.05)
    with pytest.raises(ValueError):
        decide([0.1, 0.2], [0.0, 0.0], delta_min=0.0, delta_eq=0.05, delta_harm=0.05)
    with pytest.raises(ValueError):
        paired_summary([0.1, 0.2], [0.0, 0.0, 0.1])
    with pytest.raises(ValueError):
        paired_summary([0.1, float("nan")], [0.0, 0.0])
