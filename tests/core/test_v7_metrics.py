"""this metrics tests: Recovery-AURC family (§8.1)."""

from cases.metrics import from_state, recovery_aurc, recovery_curve


def test_recovery_aurc_truth_on_top_is_better():
    # true solutions are the 2 top-scored of 5 -> good recovery
    scores = {"a": 0.9, "b": 0.8, "c": 0.4, "d": 0.2, "e": 0.1}
    good = recovery_aurc(scores, {"a", "b"})
    # truths at the bottom -> poor recovery
    bad = recovery_aurc(scores, {"d", "e"})
    assert good["au_prc"] > bad["au_prc"]
    assert good["aurc_risk"] < bad["aurc_risk"]
    assert good["recall_at_0.5"] >= 1.0
    assert bad["recall_at_0.5"] < 1.0


def test_recovery_curve_shapes():
    covers, recalls, precisions, risks = recovery_curve({"a": 0.9, "b": 0.5}, {"a"})
    assert covers == [0.0, 0.5, 1.0]
    assert recalls == [0.0, 1.0, 1.0]
    assert precisions == [1.0, 1.0, 0.5]
    assert risks == [0.0, 0.0, 0.5]


def test_recovery_aurc_empty_reference():
    # degenerate: no known true solutions -> all covered objects are false
    # positives, so risk is positive; must not crash and must be in [0,1].
    r = recovery_aurc({"a": 0.9}, [])
    assert r["n_reference"] == 0
    assert 0.0 <= r["aurc_risk"] <= 1.0
