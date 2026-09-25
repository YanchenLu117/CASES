"""P0-f tests part 4: R^run_t carrier (representation.py), solution field
(field.py), §13 fairness/Pareto (fairness.py), §18 checklist, and the
evidence-migration pipeline (§3.2)."""

import math

import pytest

from cases.protocol import (
    Cover,
    DeclaredRelation,
    EvidenceLedger,
    Generator,
    GammaEntry,
    GroundingRule,
    MigrationReport,
    RepresentationState,
    ResourceMeter,
    SolutionField,
    SolutionFieldError,
    SolutionFieldPoint,
)
from cases.protocol import (
    consistent_domination,
    expected_satisfaction,
    pareto_compare,
    run_completion_checklist,
)
from cases.protocol.revision import MIGRATION_REENCODED, MIGRATION_UNREPRESENTABLE, migrate_evidence


# ------------------------------------------------------ R^run_t carrier


def _state() -> RepresentationState:
    return RepresentationState(
        contract_id="contract_v2",
        contract_version=2,
        generators=(
            Generator("g0", "predicate", "reactant class A", "bh", "sha256:prov0", "dsl://g0"),
            Generator("g1", "predicate", "reactant class B", "bh", "sha256:prov1", "dsl://g1"),
        ),
        relations=(DeclaredRelation("r1", "refinement", ("g0", "g1")),),
        covers=(Cover("c1", "g0", ("g0", "g1"), certificate=None),),
        grounding_rules=(GroundingRule("g0", "region_A", "rule://g0"), GroundingRule("g1", "region_B", "rule://g1")),
    )


def test_representation_state_carries_six_tuple() -> None:
    st = _state()
    assert len(st.generators) == 2 and len(st.relations) == 1 and len(st.covers) == 1
    assert len(st.grounding_rules) == 2
    d = st.to_dict()
    for key in ("generators", "relations", "covers", "grounding_rules", "gamma", "contract_id", "contract_version"):
        assert key in d
    assert st.snapshot_hash().startswith("sha256:")
    # deterministic snapshot
    assert _state().snapshot_hash() == st.snapshot_hash()


def test_gamma_append_and_uncovered_report() -> None:
    st = _state()
    st.materialize_correspondence("h1", "region_A", ("e1",))
    st.materialize_correspondence("h2", "region_UNKNOWN")  # unknown generator → uncovered
    uncovered = st.uncovered(certified_regions={"region_A", "region_B"})
    assert [g.h_id for g in uncovered] == ["h2"]
    assert uncovered[0].covered is False
    # multivalued Γ: one h may ground to several z
    st.materialize_correspondence("h1", "region_B", ("e2",))
    h1_entries = [g for g in st.gamma if g.h_id == "h1"]
    assert len(h1_entries) == 2


def test_gap_classification() -> None:
    st = _state()
    assert st.classify_gap("h9", representable=False, realizable=False) == "SEMANTIC_GAP"
    assert st.classify_gap("h1", representable=True, realizable=False) == "REALIZATION_GAP"
    assert st.classify_gap("h1", representable=True, realizable=True) == "NONE"


# -------------------------------------------------------- solution field


def test_expected_satisfaction_is_expectation_not_posterior() -> None:
    # posterior P(y=1)=0.6, P(y=0)=0.4; Sat=1 iff y==1 → E[Sat]=0.6 (the two
    # coincide under a binary indicator; under graded Sat they diverge)
    point = expected_satisfaction(
        z_id="z1",
        h_id="h1",
        posterior={"1": 0.6, "0": 0.4},
        response={"yield": 0.7},
        satisfaction_fn=lambda h, r, y: 1.0 if y == "1" else 0.0,
    )
    assert point.p_sat == pytest.approx(0.6)
    graded = expected_satisfaction(
        z_id="z1",
        h_id="h1",
        posterior={"1": 0.6, "0": 0.4},
        response={"yield": 0.7},
        satisfaction_fn=lambda h, r, y: r["yield"] if y == "1" else 0.2,
    )
    # E[Sat] = 0.6*0.7 + 0.4*0.2 = 0.5 — NOT P(y=1)=0.6
    assert graded.p_sat == pytest.approx(0.6 * 0.7 + 0.4 * 0.2)


def test_expected_satisfaction_validation() -> None:
    with pytest.raises(SolutionFieldError):
        expected_satisfaction(
            z_id="z", h_id="h", posterior={"1": -0.5}, response=None, satisfaction_fn=lambda h, r, y: 1.0
        )
    with pytest.raises(SolutionFieldError):
        expected_satisfaction(
            z_id="z", h_id="h", posterior={"1": 0.5}, response=None, satisfaction_fn=lambda h, r, y: 2.0
        )


def test_solution_set_domain_and_readouts() -> None:
    p1 = SolutionFieldPoint(z_id="z1", h_id="h1", p_sat=0.9, posterior={})
    p2 = SolutionFieldPoint(z_id="z2", h_id="h2", p_sat=0.55, posterior={})
    p3 = SolutionFieldPoint(z_id="z3", h_id="h3", p_sat=0.2, posterior={})
    field = SolutionField({"z1": p1, "z2": p2, "z3": p3})
    assert field.solution_set(0.5) == frozenset({"z1", "z2"})
    assert field.solution_set(0.95) == frozenset()
    assert field.realization_gaps({"z1", "z_out"}) == frozenset({"z_out"})
    assert field.boundary(0.4, 0.95) == frozenset({"z1", "z2"})
    assert field.untested_alternatives({"z1"}) == frozenset({"z2"})


# ---------------------------------------------------------- §13 fairness


def test_resource_meter_caps_and_unknown_keys() -> None:
    meter = ResourceMeter({"llm_calls": 10, "oracle_budget": 4, "tool_calls": 5,
                           "generated_tokens": 100, "persistent_state_tokens": 50,
                           "wall_time_seconds": 60, "retries": 1})
    meter.consume("llm_calls", 9)
    assert not meter.exceeded["llm_calls"]
    meter.consume("llm_calls", 2)
    assert meter.exceeded["llm_calls"] is True
    meter.consume("oracle_budget", 4)
    assert not meter.exceeded["oracle_budget"]
    with pytest.raises(ValueError):
        meter.consume("gpu_hours", 1)
    with pytest.raises(ValueError):
        ResourceMeter({"gpu_hours": 1})


def test_pareto_single_axis_verdicts() -> None:
    assert pareto_compare(ci_low=0.05, delta_harm=0.02, delta_min=0.01) == "strictly_better"
    assert pareto_compare(ci_low=-0.01, delta_harm=0.02, delta_min=0.01) == "no_worse"
    assert pareto_compare(ci_low=-0.05, delta_harm=0.02, delta_min=0.01) == "dominated"


def test_consistent_domination_requires_two_campaigns() -> None:
    good = {"utility": "strictly_better", "tokens": "no_worse"}
    single = {"c1": good}
    assert consistent_domination(single, utility_axes=["utility"], resource_axes=["tokens"]) is False
    two = {"c1": good, "c2": good}
    assert consistent_domination(two, utility_axes=["utility"], resource_axes=["tokens"]) is True
    dominated = {"c1": good, "c2": {"utility": "dominated", "tokens": "no_worse"}}
    assert consistent_domination(dominated, utility_axes=["utility"], resource_axes=["tokens"]) is False
    no_strict = {"c1": {"utility": "no_worse", "tokens": "no_worse"}, "c2": {"utility": "no_worse", "tokens": "no_worse"}}
    assert consistent_domination(no_strict, utility_axes=["utility"], resource_axes=["tokens"]) is False


# ------------------------------------------------------ §18 checklist


def test_completion_checklist_all_or_flagged() -> None:
    context = {key: True for key, _ in [
        ("terminal_status_all_units", ""), ("failures_represented", ""),
        ("evaluator_isolation_hashes", ""), ("primary_contrasts_complete", ""),
        ("certification_language_matches", ""), ("no_post_freeze_changes", ""),
        ("identity_ledgers_pass", ""), ("claim_table_updated", ""),
    ]}
    report = run_completion_checklist(context)
    assert report.complete
    incomplete = run_completion_checklist({**context, "failures_represented": False})
    assert not incomplete.complete
    assert any("failures_represented" in p for p in incomplete.problems)
    partial = run_completion_checklist({"terminal_status_all_units": True})
    assert not partial.complete  # missing keys fail, never silently pass


# ------------------------------------------------- migration pipeline


def test_migrate_evidence_reencodes_and_keeps_unrepresentable() -> None:
    records = [
        {"exp_id": "e1", "y": 0.5, "legacy_field": "a"},
        {"exp_id": "e2", "y": 0.7, "legacy_field": "DROP_ME"},
        {"exp_id": "e3", "y": 0.9, "legacy_field": "b"},
    ]

    def encoder(record, new_version):
        if record.get("legacy_field") == "DROP_ME":
            return None  # not representable under v2
        return {"exp_id": record["exp_id"], "yield": record["y"]}

    report = migrate_evidence(records, from_version=1, to_version=2, encoder=encoder)
    assert report.excluded_count == 1
    assert [r["exp_id"] for r in report.reencoded] == ["e1", "e3"]
    assert report.reencoded[0]["migration_status"] == MIGRATION_REENCODED
    assert report.unrepresentable[0]["migration_status"] == MIGRATION_UNREPRESENTABLE
    # unencodable records are KEPT with full payload (never fabricated away)
    assert report.unrepresentable[0]["y"] == 0.7


def test_migrate_evidence_encoder_crash_keeps_record() -> None:
    def broken_encoder(record, version):
        raise RuntimeError("encoder bug")

    report = migrate_evidence([{"exp_id": "e1"}], from_version=1, to_version=2, encoder=broken_encoder)
    assert report.excluded_count == 1
    assert any("encoder bug" in r.get("migration_error", "") for r in report.unrepresentable)
