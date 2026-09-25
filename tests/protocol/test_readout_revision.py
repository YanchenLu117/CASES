"""P0-c tests: readout schema (Detail §3) and revision routing (Detail §3.2)."""

import pytest

from cases.protocol import (
    GroundingSubkind,
    MIGRATION_REENCODED,
    ContradictionPredicate,
    EmpiricalPredicate,
    ProbeThresholdPredicate,
    ReadoutValidationError,
    RevisionEvent,
    RevisionRouter,
    TriggerKind,
    UnfrozenTriggerError,
    build_readout,
    readout_to_json,
    validate_readout,
)
from cases.protocol.readout import READOUT_FIELDS
from cases.protocol.revision import FrozenPredicate, PredicateVerdict


# ---------------------------------------------------------------- readout


def test_build_readout_fills_all_fields() -> None:
    r = build_readout(observed_supported=[{"id": "h1"}])
    assert set(r) == set(READOUT_FIELDS) | {"schema_version"}
    assert r["observed_supported"] == [{"id": "h1"}]
    assert r["inferred_untested"] == []
    assert r["abstentions"] == []


def test_build_readout_rejects_unknown_field() -> None:
    with pytest.raises(ReadoutValidationError):
        build_readout(secret_hypotheses=[{"id": "x"}])


def test_validate_all_omitted_is_fatal() -> None:
    problems = validate_readout({"schema_version": 1})
    assert len(problems) == len(READOUT_FIELDS)
    assert all(p.startswith("missing field") for p in problems)


def test_validate_empty_fields_ok_when_no_claims() -> None:
    assert validate_readout(build_readout()) == []


def test_claim_without_evidence_or_abstention_rejected() -> None:
    r = build_readout(
        inferred_untested=[{"id": "h9", "family": "oscillators"}],
        provenance=[{"id": "p1", "claim_ids": ["other"], "evidence_ids": ["e1"]}],
    )
    problems = validate_readout(r)
    assert any("h9" in p and "no evidence link" in p for p in problems)


def test_claim_linked_via_provenance_passes() -> None:
    r = build_readout(
        observed_supported=[{"id": "h1"}],
        provenance=[{"id": "p1", "claim_ids": ["h1"], "evidence_ids": ["e1", "e2"]}],
    )
    assert validate_readout(r) == []


def test_claim_abstained_passes_and_missing_reason_fails() -> None:
    r = build_readout(
        uncertainties=[{"id": "u1"}],
        abstentions=[{"id": "a1", "claim_ids": ["u1"], "reason": "no legal grounding declared"}],
    )
    assert validate_readout(r) == []
    bad = build_readout(
        uncertainties=[{"id": "u1"}],
        abstentions=[{"id": "a1", "claim_ids": ["u1"], "reason": ""}],
    )
    assert any("reason" in p for p in validate_readout(bad))


def test_provenance_without_evidence_ids_fails() -> None:
    r = build_readout(
        observed_supported=[{"id": "h1"}],
        provenance=[{"id": "p1", "claim_ids": ["h1"], "evidence_ids": []}],
    )
    assert any("evidence_ids" in p for p in validate_readout(r))


def test_duplicate_claim_id_fails() -> None:
    r = build_readout(
        observed_supported=[{"id": "h1"}],
        boundaries=[{"id": "h1"}],
        abstentions=[{"id": "a1", "claim_ids": ["h1"], "reason": "x"}],
    )
    assert any("duplicate claim id" in p for p in validate_readout(r))


def test_readout_to_json_roundtrip() -> None:
    r = build_readout(
        observed_supported=[{"id": "h1"}],
        provenance=[{"id": "p1", "claim_ids": ["h1"], "evidence_ids": ["e1"]}],
    )
    text = readout_to_json(r)
    assert "observed_supported" in text


# --------------------------------------------------------------- revision


def _router_with(kind_predicate_pairs) -> RevisionRouter:
    router = RevisionRouter()
    for pred in kind_predicate_pairs:
        router.register(pred)
    return router


def test_empirical_fires_after_consecutive_breach_window() -> None:
    pred = EmpiricalPredicate("emp1", "recovery", threshold=0.5, consecutive_checkpoints=3, direction="below")
    history = [
        {"recovery": 0.6},
        {"recovery": 0.4},
        {"recovery": 0.45},
        {"recovery": 0.3},
    ]
    verdict = pred.evaluate({"checkpoint_history": history})
    assert verdict.triggered
    assert verdict.detail["consecutive_checkpoints"] == 3

    short = pred.evaluate({"checkpoint_history": history[:2]})
    assert not short.triggered and short.detail["reason"] == "insufficient_history"

    broken = pred.evaluate({"checkpoint_history": history[:3] + [{"recovery": 0.9}]})
    assert not broken.triggered and broken.detail["reason"] == "streak_broken"


def test_probe_threshold_predicate() -> None:
    pred = ProbeThresholdPredicate(
        TriggerKind.GROUNDING, "gap1", probe_key="semantic_coverage", threshold=0.5,
        direction="below", subkind=GroundingSubkind.SEMANTIC_GAP,
    )
    assert pred.evaluate({"probe_values": {"semantic_coverage": 0.2}}).triggered
    assert not pred.evaluate({"probe_values": {"semantic_coverage": 0.8}}).triggered
    absent = pred.evaluate({"probe_values": {}})
    assert not absent.triggered and absent.detail["reason"] == "probe_absent"


def test_contradiction_predicate_on_invariant_violation() -> None:
    pred = ContradictionPredicate(
        "contra1",
        invariant=lambda state: "inconsistent" if state.get("a") and state.get("a") >= state.get("b") else None,
        invariant_id="order_preserved",
    )
    assert pred.evaluate({"invariant_state": {"a": 3, "b": 1}}).triggered
    assert not pred.evaluate({"invariant_state": {"a": 1, "b": 3}}).triggered


def test_router_requires_frozen_predicates() -> None:
    router = RevisionRouter()
    with pytest.raises(UnfrozenTriggerError):
        router.require_frozen([TriggerKind.EMPIRICAL, TriggerKind.CONTRADICTION])


def test_unregistered_trigger_kind_never_fires() -> None:
    # only GROUNDING registered; EMPIRICAL conditions present in context
    router = _router_with(
        [
            ProbeThresholdPredicate(
                TriggerKind.GROUNDING, "g1", "semantic_coverage", 0.9, direction="below",
                subkind=GroundingSubkind.REALIZATION_GAP,
            )
        ]
    )
    context = {
        "probe_values": {"semantic_coverage": 0.1},
        "checkpoint_history": [{"recovery": 0.0}] * 5,
    }
    kind, predicate, verdict = router.route(context)
    assert kind is TriggerKind.GROUNDING  # the registered one fires first anyway
    # now register nothing and show EMPIRICAL cannot fire
    empty = RevisionRouter()
    kind2, _, _ = empty.route(context)
    assert kind2 is None


def test_router_precedence_grounding_before_empirical() -> None:
    router = _router_with(
        [
            ProbeThresholdPredicate(
                TriggerKind.GROUNDING, "g1", "semantic_coverage", 0.9, direction="below",
                subkind=GroundingSubkind.SEMANTIC_GAP,
            ),
            EmpiricalPredicate("e1", "recovery", 0.5, 2, direction="below"),
        ]
    )
    context = {
        "probe_values": {"semantic_coverage": 0.1},
        "checkpoint_history": [{"recovery": 0.1}, {"recovery": 0.1}],
    }
    kind, predicate, verdict = router.route(context)
    assert kind is TriggerKind.GROUNDING
    assert predicate.predicate_id == "g1"


def test_make_event_carries_grounding_subkind_and_full_record() -> None:
    router = _router_with(
        [
            ProbeThresholdPredicate(
                TriggerKind.GROUNDING, "g1", "realization_coverage", 0.5, direction="below",
                subkind=GroundingSubkind.REALIZATION_GAP,
            )
        ]
    )
    kind, predicate, verdict = router.route({"probe_values": {"realization_coverage": 0.2}})
    event = router.make_event(
        kind, predicate, verdict,
        triggering_evidence={"probe": "realization_coverage"},
        pre_state_ref="state_v3",
        proposed_diff={"add_generator": "g_new"},
        logic_verdict={"logic_valid": 1},
        fidelity_verdict={"round_trip": "ok"},
        event_id="rev_001",
    )
    assert event.trigger["subkind"] == GroundingSubkind.REALIZATION_GAP.value
    # routing-time fields are filled; downstream stages (acceptance, migration,
    # re-grounding, field rebuild, post metrics) fill the rest before archiving
    assert event.accepted is None and event.migration_status is None
    event.accepted = True
    event.migration_status = MIGRATION_REENCODED
    event.regrounding_coverage = {"regrounded": 12, "uncovered": 0}
    event.field_rebuild_ref = "field/posterior_round_9.npz"
    event.post_metrics = {"recovery": 0.71}
    assert event.is_complete() == []
    payload = event.to_dict()
    for key in RevisionEvent.REQUIRED:
        assert payload[key] is not None
    assert payload["migration_status"] == MIGRATION_REENCODED


def test_revision_event_incomplete_flags_missing() -> None:
    event = RevisionEvent(
        trigger={"kind": "FIDELITY"},
        triggering_evidence={},
        pre_state_ref="s",
        proposed_diff={},
        logic_verdict={},
        fidelity_verdict={},
    )
    missing = event.is_complete()
    assert "accepted" in missing and "migration_status" in missing and "post_metrics" in missing


def test_custom_predicate_via_base_class() -> None:
    class AlwaysFires(FrozenPredicate):
        kind = TriggerKind.FIDELITY
        predicate_id = "custom"

        def evaluate(self, context):
            return PredicateVerdict(True, {"why": "test"})

    router = _router_with([AlwaysFires()])
    kind, _, verdict = router.route({})
    assert kind is TriggerKind.FIDELITY and verdict.triggered
