"""Unit tests for the P0 deterministic evaluator (synthetic snapshots).

Covers all 8 query-class families on synthetic bh/made/gb1 states, including
the hallucination double-check (state carrier + observation log) on both
paths: fabricated measurement (absent from log) and fabricated evidence span
(absent from snapshot).  Run with cases-core python:

    PYTHONPATH=src python3 -m pytest \
        src/cases/adapters/_shared/test_p0_evaluator.py -q
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from cases.adapters._shared.p0_evaluator import (
    GradeResult,
    ObservationLog,
    P0Evaluator,
    SnapshotView,
)


# ----------------------------------------------------------- synthetic oracle

class SyntheticOracle:
    """Fixed universe + solution set + legality, deterministic."""

    def __init__(self, task, universe, solution_set, gamma=0.8):
        self.task = task
        self._universe = list(universe)
        self._solution = set(solution_set)
        self._gamma = gamma

    def candidates(self):
        return list(self._universe)

    def solution_set(self):
        return list(self._solution)

    def solution_membership(self, key):
        return key in self._solution

    def gamma(self):
        return self._gamma

    def gold_values(self):
        return [0.0, 1.0]

    def is_legal(self, key):
        return key in self._universe


@pytest.fixture()
def evaluator(tmp_path):
    doc = {
        "meta": {"layer": "L1", "system": "s3_codescientist"},
        "rubric_notes": {"grading_scale": "0/0.5/1"},
        "pairs": {
            t: {"task_context": "", "oracle_ref": "",
                "instances": [
                    {"query_class": c, "task": t, "question": "",
                     "canonical_answer_derivation": "", "rubric_points": "",
                     "expected_state_carrier": ""}
                    for c in [
                        "tested_hypotheses", "tested_hypotheses",
                        "inferred_but_untested_alternatives", "inferred_but_untested_alternatives",
                        "semantic_equivalence", "semantic_equivalence",
                        "distinct_solution_families", "distinct_solution_families",
                        "coverage_gaps", "coverage_gaps",
                        "likely_boundaries", "likely_boundaries",
                        "uncertainty_in_untested_regions", "uncertainty_in_untested_regions",
                        "currently_unrepresentable_hypotheses",
                        "currently_unrepresentable_hypotheses"]]
                } for t in ("bh", "made", "gb1")}
    }
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(doc), encoding="utf-8")

    uni = {
        "bh": ["a0|l0|b0|d0", "a0|l0|b1|d0", "a1|l1|b0|d0", "a1|l1|b1|d0"],
        "gb1": ["V1", "V2", "V3", "V12"],
        "made": ["mechA::c1", "mechA::c2", "mechB::c1", "mechB::c2"],
    }
    sol = {"bh": {"a0|l0|b0|d0"}, "gb1": {"V1"}, "made": {"mechA::c1"}}
    gam = {"bh": 0.8, "gb1": 0.5, "made": 0.5}

    def factory(task):
        return SyntheticOracle(task, uni[task], sol[task], gam[task])

    return P0Evaluator(str(p), factory)


ANS_OK = json.dumps({"answer": "x", "evidence_span": "x", "abstain": False,
                     "abstain_reason": "", "currently_unrepresentable": False,
                     "flags": []})


def ans(text, span=""):
    return json.dumps({"answer": text, "evidence_span": span, "abstain": False,
                       "abstain_reason": "", "currently_unrepresentable": False,
                       "flags": []})


# ------------------------------------------------------------ family 0/1 (bh)

def test_bh_f0_complete_list(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    log.add("a1|l1|b0|d0", 42.0, "submitted")
    snap = SnapshotView(text="results", state_measured=set(log.measured))
    g = evaluator.grade("bh", 0, snap, log, ans("measured: a0|l0|b0|d0 = 91, a1|l1|b0|d0 = 42"))
    assert g.score == 1.0


def test_bh_f0_partial(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    log.add("a1|l1|b0|d0", 42.0, "submitted")
    snap = SnapshotView(text="results", state_measured=set(log.measured))
    g = evaluator.grade("bh", 0, snap, log, ans("measured: a0|l0|b0|d0 = 91"))
    assert g.score == 0.5


def test_bh_f0_hallucinated_not_in_log(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    snap = SnapshotView(text="results", state_measured={"a0|l0|b0|d0"})
    g = evaluator.grade("bh", 0, snap, log,
                        ans("measured: a1|l1|b1|d0 yield 77"))
    assert g.score == 0.0
    assert any("not_in_observation_log" in f for f in g.flags)


def test_bh_f0_evidence_span_not_in_snapshot(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    snap = SnapshotView(text="results", state_measured={"a0|l0|b0|d0"})
    g = evaluator.grade("bh", 0, snap, log,
                        ans("measured a0|l0|b0|d0", span="this span is not in the snapshot"))
    assert g.score == 0.0
    assert any("evidence_span_not_in_snapshot" in f for f in g.flags)


def test_bh_f1_best(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.5, "initial")
    log.add("a1|l1|b0|d0", 42.0, "submitted")
    snap = SnapshotView(text="results", state_measured=set(log.measured))
    g = evaluator.grade("bh", 1, snap, log, ans("best is a0|l0|b0|d0 with yield 91.5"))
    assert g.score == 1.0
    g = evaluator.grade("bh", 1, snap, log, ans("best is a0|l0|b0|d0"))
    assert g.score == 0.5
    g = evaluator.grade("bh", 1, snap, log, ans("best is a1|l1|b0|d0 with 91.5"))
    assert g.score == 0.5


# --------------------------------------------------- family 2/3 (untested)

def test_bh_f2_oracle_verified_hit(evaluator):
    log = ObservationLog()
    log.add("a1|l1|b1|d0", 10.0, "initial")  # NOT in S*
    snap = SnapshotView(text="ideas", proposals=["a0|l0|b0|d0"])  # in S*, untested
    g = evaluator.grade("bh", 2, snap, log,
                        ans("promising untested candidate: a0|l0|b0|d0"))
    assert g.score == 1.0


def test_bh_f2_measured_claimed_untested_zero(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    snap = SnapshotView(text="ideas", proposals=["a0|l0|b0|d0"],
                        state_measured={"a0|l0|b0|d0"})
    g = evaluator.grade("bh", 2, snap, log,
                        ans("promising untested candidate a0|l0|b0|d0"))
    assert g.score == 0.0
    assert any("measured_claimed_untested" in f for f in g.flags)


def test_bh_f3_rationale_faithful(evaluator):
    span = "we prefer phosphine ligands because they stabilize oxidative addition"
    snap = SnapshotView(text=span, rationale_spans=[span])
    g = evaluator.grade("bh", 3, snap, ObservationLog(),
                        ans(f"rationale present: {span}"))
    assert g.score == 1.0
    g = evaluator.grade("bh", 3, snap, ObservationLog(),
                        ans("there is no rationale in the state"))
    assert g.score == 0.0


# ------------------------------------------------ family 4/5 (semantics)

def test_bh_f4_duplicates_found(evaluator):
    snap = SnapshotView(text="ideas", proposals=["a0|l0|b0|d0", "a0|l0|b0|d0"])
    g = evaluator.grade("bh", 4, snap, ObservationLog(),
                        ans("duplicate identical tuples a0|l0|b0|d0 appear twice"))
    assert g.score == 1.0


def test_bh_f5_one_factor_pairs(evaluator):
    snap = SnapshotView(text="ideas", proposals=["a0|l0|b0|d0", "a0|l0|b1|d0"])
    g = evaluator.grade("bh", 5, snap, ObservationLog(),
                        ans("a0|l0|b0|d0 and a0|l0|b1|d0 differ in exactly one slot (one-factor)"))
    assert g.score == 1.0
    g = evaluator.grade("bh", 5, snap, ObservationLog(),
                        ans("a0|l0|b0|d0 and a1|l1|b0|d0 are one-factor"))
    assert g.score == 0.0


# ------------------------------------------------ families 6/7 (families)

def test_bh_f6_partition(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    log.add("a1|l1|b0|d0", 42.0, "submitted")
    snap = SnapshotView(text="state", state_measured=set(log.measured))
    g = evaluator.grade("bh", 6, snap, log,
                        ans("2 families: l0 -> a0|l0|b0|d0; l1 -> a1|l1|b0|d0"))
    assert g.score == 1.0
    g = evaluator.grade("bh", 6, snap, log, ans("some families exist"))
    assert g.score == 0.0


def test_bh_f7_dominance_ligand_slot(evaluator):
    log = ObservationLog()
    for i, key in enumerate(["a0|l0|b0|d0", "a0|l0|b1|d0", "a1|l1|b0|d0"]):
        log.add(key, float(50 + i), "initial")
    snap = SnapshotView(text="s", state_measured=set(log.measured))
    g = evaluator.grade("bh", 7, snap, log,
                        ans("dominant slot: a (2 distinct values) over l/b/d"))
    assert g.score == 1.0


# ------------------------------------------------ families 8/9 (coverage)

def test_bh_f8_marginal_near_total(evaluator):
    log = ObservationLog()
    log.add("a0|l0|b0|d0", 91.0, "initial")
    oracle = evaluator.oracle("bh")
    # synthetic oracle has no legal_slot_values -> near_total False path
    snap = SnapshotView(text="s")
    g = evaluator.grade("bh", 8, snap, log,
                        ans("near-total non-coverage: most of the legal grid is unmeasured"))
    # without slot metadata the evaluator cannot enumerate legal values:
    assert g.score in (0.0, 0.5)


def test_made_f9_unevaluated_proposals(evaluator):
    snap = SnapshotView(text="plans", proposals=["mechA::c1", "mechB::c2"])
    log = ObservationLog()
    log.add("mechA::c1", 0.7, "submitted")
    g = evaluator.grade("made", 9, snap, log, ans("unevaluated: mechB::c2"))
    assert g.score == 1.0
    g = evaluator.grade("made", 9, snap, log, ans("nothing"))
    assert g.score == 0.0


# ------------------------------------------- families 10/11 (boundaries)

def test_bh_f10_boundary_contradiction(evaluator):
    log = ObservationLog()
    log.add("a1|l1|b0|d0", 30.0, "initial")
    snap = SnapshotView(text="s", negative_claims=["a1|l1|b0|d0"])
    g = evaluator.grade("bh", 10, snap, log,
                        ans("low side includes a0|l0|b0|d0 which performs poorly"))
    assert g.score == 0.0
    assert any("contradicts_oracle" in f for f in g.flags)


def test_bh_f11_negative_partition(evaluator):
    log = ObservationLog()
    log.add("a1|l1|b0|d0", 30.0, "submitted")
    snap = SnapshotView(
        text="a1|l1|b0|d0 measured low; a0|l0|b1|d0 never tried",
        state_measured={"a1|l1|b0|d0"},
        negative_claims=["a1|l1|b0|d0", "a0|l0|b1|d0"])
    g = evaluator.grade("bh", 11, snap, log,
                        ans("a1|l1|b0|d0 measured low; a0|l0|b1|d0 never tried"))
    assert g.score == 1.0
    # fabricated measured-failure: claims an untried key as measured low
    g = evaluator.grade("bh", 11, snap, log,
                        ans("a0|l0|b1|d0 was measured and failed low"))
    assert g.score == 0.0
    assert any("hallucinated_state_claim" in f for f in g.flags)


# --------------------------------- families 12/13 (uncertainty, ranking)

def test_made_f13_ranking_faithful(evaluator):
    snap = SnapshotView(text="s", rankings=[("by predicted gain: chemA then chemB",
                                             ["chemA", "chemB"])])
    g = evaluator.grade("made", 13, snap, ObservationLog(),
                        ans("ranking: chemA > chemB; basis: by predicted gain"))
    assert g.score == 1.0
    # misdescribed basis -> 0.5
    g = evaluator.grade("made", 13, snap, ObservationLog(),
                        ans("ranking: chemA > chemB; basis: model posterior"))
    assert g.score == 0.5


def test_bh_f12_uncertainty_carriers(evaluator):
    snap = SnapshotView(text="s", uncertainty_regions=["base b1 region unexplored"])
    g = evaluator.grade("bh", 12, snap, ObservationLog(),
                        ans("acknowledges: base b1 region unexplored"))
    assert g.score == 1.0
    g = evaluator.grade("bh", 12, snap, ObservationLog(),
                        ans("claims confident coverage everywhere"))
    assert g.score == 0.0


# ------------------------------- families 14/15 (representable, capacity)

def test_bh_f14_none_declared(evaluator):
    snap = SnapshotView(text="s")
    g = evaluator.grade("bh", 14, snap, ObservationLog(),
                        ans("none declared"))
    assert g.score == 1.0


def test_bh_f15_capacity_vs_content(evaluator):
    snap = SnapshotView(text="s", proposals=[])
    g = evaluator.grade("bh", 15, snap, ObservationLog(),
                        ans("capacity exists (records can carry interaction hypotheses); no actual epistasis hypothesis present"))
    assert g.score == 1.0


def test_gb1_f1_best_with_count(evaluator):
    log = ObservationLog()
    log.add("V2", 0.9, "initial")
    log.add("V3", 0.4, "submitted")
    snap = SnapshotView(text="s", state_measured=set(log.measured))
    g = evaluator.grade("gb1", 1, snap, log, ans("best V2 fitness 0.9; 2 records"))
    assert g.score == 1.0
    g = evaluator.grade("gb1", 1, snap, log, ans("best V2 fitness 0.9"))
    assert g.score == 0.5
    g = evaluator.grade("gb1", 1, snap, log, ans("best V2 fitness 0.9; 96 records"))
    assert g.score == 0.5
