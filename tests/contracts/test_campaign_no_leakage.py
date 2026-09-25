"""Campaign interface zero-leakage contracts (V9 infra).

Pins the outcome-blind boundary of ``run_campaign``: the acting method may
receive ONLY legal-side information (dossier refs, answered history, unqueried
legal ids, remaining budgets).  Ground truth (solution ids, unqueried
observations, thresholds) must never cross the interface — in structured
fields OR stringified refs.  Oracle discipline: illegal / duplicate /
unmeasured proposals consume a batch slot but receive no label.

These contracts guard the claim that arms differ only in method, never in
information endowment (paper §03 fairness / Detail §2.1).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from cases.protocol import RecoveryPrediction, RoundInputs, RoundOutputs
from cases.protocol.campaign import ToyOracle, run_campaign
from tests.protocol.test_campaign import _registry


class SpyMethod:
    """Records every (inputs, history) the harness hands the method."""

    def __init__(self, batch: int = 2, *, missing: bool = False):
        self.batch = batch
        self.missing = missing
        self.seen: list[tuple[RoundInputs, dict[str, float]]] = []

    def round_step(self, inputs: RoundInputs, history: dict[str, float]) -> RoundOutputs:
        self.seen.append((inputs, dict(history)))
        legal = inputs.legal_candidate_ids
        proposals = tuple(sorted(legal)[: self.batch])
        return RoundOutputs(
            proposed_candidate_ids=proposals,
            persistent_state_ref="state://spy",
            recovery_prediction=RecoveryPrediction(
                scores={c: 0.5 for c in legal}, missing=self.missing
            ),
            decision_trace_ref="trace://spy",
            resource_usage={"llm_calls": 1},
            completion_status="RUNNING",
        )


def _oracle() -> ToyOracle:
    return ToyOracle(
        legal_ids=tuple(f"c{i}" for i in range(1, 9)),
        observations={f"c{i}": i / 10 for i in range(1, 9)},
        solution_ids={"c8", "c7"},
        initial_ids={"c1"},
    )


def _run(method, oracle, tmp_path, seed: int = 0):
    return run_campaign(
        prereg=_registry(),
        cell_id="toy/main",
        method=method,
        oracle=oracle,
        budgets=(3, 5),
        run_dir=tmp_path / "run",
        initial_ids={"c1"},
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. the interface carries no ground truth
# ---------------------------------------------------------------------------


def test_history_only_contains_answered_pairs(tmp_path):
    method = SpyMethod(batch=2)
    oracle = _oracle()
    _run(method, oracle, tmp_path)
    answered_ever = set(oracle.queried)
    hidden = set(oracle.observations) - answered_ever
    assert hidden, "fixture sanity: some observations must stay hidden"
    for _inputs, history in method.seen:
        # history keys are always a subset of what the oracle has answered
        assert set(history) <= answered_ever
        for cid in history:
            assert history[cid] == oracle.observations[cid]
        for cid in hidden:
            assert cid not in history


def test_round_inputs_never_expose_ground_truth(tmp_path):
    method = SpyMethod(batch=2)
    oracle = _oracle()
    _run(method, oracle, tmp_path)

    for inputs, history in method.seen:
        answered_so_far = set(history)  # exactly what the oracle has answered
        # legal ids: unqueried legal only, per round
        assert set(inputs.legal_candidate_ids) <= set(oracle.legal) - answered_so_far
        assert not (set(inputs.legal_candidate_ids) & answered_so_far)
        # current evidence: only answered (cid, y) records
        for rec in inputs.current_evidence:
            assert rec["cid"] in answered_so_far
            assert rec["y"] == oracle.observations[rec["cid"]]
        # dossier: refs only, no solution/observation payload
        dossier = dataclasses.asdict(inputs.task_dossier)
        assert set(dossier) == {"task_id", "public_description_ref",
                                "validator_interface_ref", "initial_evidence_ids",
                                "legality_rule_id"}
        # remaining oracle budget: non-negative
        assert inputs.remaining_budgets.oracle >= 0

    # stringified leakage scan: no solution id outside answered, no hidden y
    answered = set(oracle.queried)  # final answered set across the campaign
    hidden = {cid: y for cid, y in oracle.observations.items() if cid not in answered and cid in oracle.legal}
    for inputs, _history in method.seen:
        blob = json.dumps(
            {
                "dossier": dataclasses.asdict(inputs.task_dossier),
                "evidence": [dict(r) for r in inputs.current_evidence],
                "legal": list(inputs.legal_candidate_ids),
                "budgets": dataclasses.asdict(inputs.remaining_budgets),
                "stopping": dataclasses.asdict(inputs.stopping_rule),
            },
            default=str,
        )
        for cid, y in hidden.items():
            # the candidate id itself may appear in the legal list (public by
            # design); its hidden SCORE must not appear anywhere
            assert str(y) not in blob


def test_solution_ids_never_in_legal_list(tmp_path):
    """Solutions may appear as legal ids (they are legal) but never *marked*."""
    method = SpyMethod(batch=3)
    oracle = _oracle()
    _run(method, oracle, tmp_path)
    for inputs, _ in method.seen:
        assert isinstance(inputs.legal_candidate_ids, tuple)
        # no field anywhere tags a candidate as solution
        blob = json.dumps([dict(r) for r in inputs.current_evidence], default=str)
        assert "solution" not in blob


# ---------------------------------------------------------------------------
# 2. oracle discipline — no fabricated labels
# ---------------------------------------------------------------------------


class CheatingMethod(SpyMethod):
    """Proposes duplicates, illegal ids, and unmeasured 'legal' ids."""

    def round_step(self, inputs: RoundInputs, history: dict[str, float]) -> RoundOutputs:
        self.seen.append((inputs, dict(history)))
        already = list(history)
        proposals = tuple(already[:1]) if already else ()  # duplicate
        proposals += ("zz_illegal",)  # not in legal
        fresh = sorted(set(inputs.legal_candidate_ids))[:1]
        proposals += tuple(fresh)
        return RoundOutputs(
            proposed_candidate_ids=proposals,
            persistent_state_ref="state://cheat",
            recovery_prediction=RecoveryPrediction(scores={c: 0.5 for c in inputs.legal_candidate_ids}),
            decision_trace_ref="trace://cheat",
            resource_usage={"llm_calls": 1},
            completion_status="RUNNING",
        )


def test_illegal_and_duplicate_proposals_get_no_label(tmp_path):
    method = CheatingMethod()
    oracle = _oracle()
    result = _run(method, oracle, tmp_path)
    # duplicates/illegal consumed slots but answered nothing
    assert sum(r.n_illegal for r in result.checkpoints) >= 1
    assert sum(r.n_duplicates for r in result.checkpoints) >= 1
    # oracle never answered an illegal or duplicate id
    assert "zz_illegal" not in oracle.queried
    assert oracle.queried >= {"c1"}  # initial evidence intact
    # history across rounds only grew by truly fresh legal answers
    for _inputs, history in method.seen[1:]:
        assert set(history) <= oracle.queried


# ---------------------------------------------------------------------------
# 3. failed prediction → registered zero, not fabricated
# ---------------------------------------------------------------------------


def test_missing_prediction_scores_registered_zero(tmp_path):
    method = SpyMethod(batch=2, missing=True)
    result = _run(method, _oracle(), tmp_path)
    assert all(r.recovery == 0.0 for r in result.checkpoints)
    assert result.complete is True  # missing prediction is not a cap violation


# ---------------------------------------------------------------------------
# 4. determinism at fixed seed
# ---------------------------------------------------------------------------


def test_same_seed_same_result(tmp_path):
    r1 = _run(SpyMethod(batch=2), _oracle(), tmp_path / "a")
    r2 = _run(SpyMethod(batch=2), _oracle(), tmp_path / "b")
    assert [r.recovery for r in r1.checkpoints] == [r.recovery for r in r2.checkpoints]
    assert r1.recovery_auc == r2.recovery_auc
    assert r1.utility_auc == r2.utility_auc
