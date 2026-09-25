"""P0-f tests part 3: online admission (§3.1), evidence ledger (§3.2/§3.13),
narrow headline gate (§4.2), and §2.3 round-interface carriers."""

import pytest

from cases.certification import DeclaredRelations, EvaluatorUniverse, build_image_frame, certify_level2
from cases.protocol import (
    COMPLETENESS_UNVERIFIED,
    LOGIC_REPORT_ITEMS,
    EvidenceLedger,
    LogicReport,
    RemainingBudgets,
    RoundInputs,
    RecoveryPrediction,
    StoppingRule,
    TaskDossier,
    admit,
    FidelityGate,
)
from cases.protocol.admission import DeclaredOnlyChecker


# ------------------------------------------------------------- admission


def _logic_items(**overrides) -> dict:
    items = {name: True for name in LOGIC_REPORT_ITEMS if name not in ("completeness_status", "provisional_level")}
    items["completeness_status"] = COMPLETENESS_UNVERIFIED
    items["provisional_level"] = 1
    items.update(overrides)
    return items


def test_logic_report_requires_all_nine_items() -> None:
    with pytest.raises(ValueError):
        LogicReport({name: True for name in LOGIC_REPORT_ITEMS[:-1]})
    report = LogicReport(_logic_items())
    assert report.logic_valid
    assert report.completeness_status == "UNVERIFIED"


def test_admit_passes_with_clean_report_and_fidelity() -> None:
    gate = FidelityGate({"round_trip": 0.9, "semantic_contrast": 0.5})
    decision = admit(LogicReport(_logic_items()), {"round_trip": 0.95, "semantic_contrast": 0.6}, gate)
    assert decision.admitted and decision.reasons == ()


def test_admit_rejects_failed_logic_and_low_fidelity() -> None:
    gate = FidelityGate({"round_trip": 0.9})
    report = LogicReport(_logic_items(outcome_leakage=False))
    decision = admit(report, {"round_trip": 0.5}, gate)
    assert not decision.admitted
    assert any("LogicValid=0" in r and "outcome_leakage" in r for r in decision.reasons)
    assert any("fidelity below thresholds" in r for r in decision.reasons)


def test_m1_declared_only_checker_marks_unverified() -> None:
    checker = DeclaredOnlyChecker(
        {
            "typing_structure": lambda p: True,
            "executability": lambda p: True,
            "task_constraints": lambda p: True,
            "provenance": lambda p: True,
            "outcome_leakage": lambda p: True,
            "semantic_distinctions": lambda p: True,
            "relation_cover_overlap_validity": lambda p: True,
        }
    )
    report = checker.check({"generators": []}, {})
    assert report.logic_valid
    assert report.completeness_status == "UNVERIFIED"  # never claimed during campaign
    assert report.provisional_level == 1


# ---------------------------------------------------------------- ledger


def test_evidence_ledger_is_append_only_and_monotone(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence/ledger.jsonl")
    ledger.append([{"exp": 1, "y": 0.5}, {"exp": 2, "y": 0.7}])
    fp1 = ledger.fingerprints()
    ledger.append([{"exp": 3, "y": 0.9}])
    assert ledger.extends(fp1)  # D_t ⊆ D_{t+1} as fingerprint prefix
    assert len(ledger.records()) == 3
    # a ledger that rewrites history is NOT a valid extension
    forged = EvidenceLedger(tmp_path / "evidence/forged.jsonl")
    forged.append([{"exp": 999}])
    assert not forged.extends(fp1)


# -------------------------------------------------------- narrow headline


def _preds():
    return {"g0": lambda w: w in {"w0", "w1"}, "g1": lambda w: w in {"w1", "w2"}}


def test_level2_narrow_when_family_separation_below_headline() -> None:
    # coarse generators: signatures partition W into {w0,w1}, {w2}, {w3}
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    preds = {"g0": lambda w: w in {"w0", "w1"}, "g1": lambda w: w in {"w0", "w1", "w2"}}
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    assert frame.size == 4
    # a family exactly covering one cell -> FamilySep = 1.0 (not narrow);
    # the semantically-true refinement x0 ⊑ x1 must be declared for condition 1
    strong = certify_level2(
        universe, preds, DeclaredRelations(refinements=((0, 1),)), _Identity(frame), [("h1", 0, True)],
        official_families=[lambda w: w in {"w0", "w1"}],
    )
    assert strong.achieved_level == 2 and strong.narrow is False
    # a family containing NO full cell ({w0} is half of cell {w0,w1}) -> FamilySep = 0 -> narrow
    straddling = certify_level2(
        universe, preds, DeclaredRelations(refinements=((0, 1),)), _Identity(frame), [("h1", 0, True)],
        official_families=[lambda w: w in {"w0"}],
    )
    assert straddling.achieved_level == 2 and straddling.narrow is True
    assert straddling.stats["family_sep"] == 0.0


class _Identity:
    def __init__(self, frame):
        self._frame = frame

    def meet(self, a, b):
        return self._frame.meet(a, b)

    def join(self, a, b):
        return self._frame.join(a, b)

    def map_term_mask(self, m):
        return m


# --------------------------------------------------- §2.3 round carriers


def test_round_interface_carriers() -> None:
    dossier = TaskDossier(
        task_id="toy_main",
        public_description_ref="dossier://toy",
        validator_interface_ref="validators://toy",
        initial_evidence_ids=("t1", "t2"),
        legality_rule_id="legal_unqueried_only",
    )
    inputs = RoundInputs(
        task_dossier=dossier,
        current_evidence=({"obs": 1},),
        legal_candidate_ids=("c1", "c2"),
        remaining_budgets=RemainingBudgets(oracle=6, resource={"tokens": 1000.0}),
        stopping_rule=StoppingRule("budget_exhausted", "stopping://toy"),
    )
    assert inputs.remaining_budgets.oracle == 6
    # missing prediction carries the failed-checkpoint marker
    missing = RecoveryPrediction(missing=True)
    assert missing.missing and missing.scores is None
    scored = RecoveryPrediction(scores={"c1": 0.4, "c2": 0.9})
    assert scored.submitted_set is None and not scored.missing
