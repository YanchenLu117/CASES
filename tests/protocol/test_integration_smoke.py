"""P0-f integration smoke: the V8 modules compose end to end on synthetic data.

Exercises the §2.2 registry → §15 artifacts → §3 readout/revision/induction/
controller → Plan §3.1 statistics → §4 certification chain without any LLM.
"""

import pytest

from cases.certification import (
    DeclaredRelations,
    EvaluatorUniverse,
    build_image_frame,
    certify_level2,
)
from cases.prereg import PreregRegistry
from cases.stats import DECISION_IMPROVES, decide
from cases.protocol import (
    ControllerWeights,
    EmpiricalPredicate,
    EvaluatorSealedError,
    EvaluatorVault,
    FrozenController,
    InductionOutcome,
    InductionPrompt,
    RepresentationInductor,
    RevisionRouter,
    RunArtifactWriter,
    RunManifestBuilder,
    TriggerKind,
    ValidatorReport,
    build_readout,
    validate_readout,
)

PROMPT = InductionPrompt("induction_v1", "sha256:p")


def _registry() -> PreregRegistry:
    reg = PreregRegistry("integration smoke")
    reg.set("repository.commits", {"cases": "023ba37"})
    reg.set("repository.submodule_commits", {"made": "abc123"})
    reg.set("datasets.oracle_hashes", {"toy": "sha256:o"})
    reg.set("datasets.dataset_hashes", {"toy": "sha256:d"})
    reg.set("corpus.task_dossier_hashes", {"toy": "sha256:ds"})
    reg.set("corpus.frozen_corpus_hashes", {"toy": "sha256:c"})
    reg.set("model.model_id", "deepseek-v4-flash")
    reg.set("model.tokenizer_id", "deepseek-v4-flash")
    reg.set("model.temperature", 1.0)
    reg.set("model.tool_config", {"llm_base_url": "http://127.0.0.1:8002/v1"})
    reg.set("prompts.prompt_hashes", {"induction": "sha256:p"})
    reg.set("prompts.persistent_state_schemas", {"readout": "v1"})
    reg.set("legality.candidate_legality_rules", {"toy": "legal_unqueried_only"})
    reg.set("legality.replacement_rules", {"toy": "frozen_fill_v1"})
    reg.set("solutions.thresholds", {"toy": "threshold_v1"})
    reg.set("solutions.outcome_blind_geometry", {"toy": "geometry_v1"})
    reg.set("budgets.stopping_rules", {"toy": "budget_exhausted"})
    reg.set("budgets.retry_policy", {"toy": "one_infra_retry"})
    reg.set("budgets.timeout_policy", {"toy": "per_call_120s"})
    reg.set("budgets.failure_rules", {"toy": "zero_from_failure_checkpoint"})
    reg.set("seeds.initial_evidence_ids", {"toy": ["t1", "t2"]})
    reg.set("adapters.identity_reports", {"toy": "sha256:i"})
    reg.set("adapters.supported_cells", {"toy": ["cases_a4"]})
    reg.set_endpoint(
        "toy_recovery",
        {
            "metric_function": "recovery_aurc_v1",
            "curve_scalarization": "normalized_trapezoid",
            "favorable_direction": "maximize",
            "delta_min": 0.01,
            "delta_eq": 0.01,
            "delta_harm": 0.02,
            "failure_score": 0.0,
            "ci_method": "paired_bootstrap_95",
            "effect_size_formula": "paired_mean_diff",
            "resampling_scheme": "campaign_seed",
            "multiplicity_family": "toy_main",
        },
    )
    reg.set_track(
        "toy",
        {
            "task_list": ["toy_main"],
            "min_samples": 10,
            "max_samples": 10,
            "seed_list": list(range(10)),
            "budget_table": {"initial": 4, "batches": 2, "batch_size": 2},
            "checkpoints": [4, 6, 8],
            "batch_fill_algorithm": "frozen_fill_v1",
        },
    )
    reg.set_baseline(
        "gp_bo",
        {
            "official_identifier": "toy://gp_bo",
            "commit_hash": "deadbeef",
            "container_hash": "sha256:container",
            "entry_point": "gp_bo.run",
            "config": {"acq": "lse"},
            "permitted_adapter_diff": "oracle_boundary_only",
            "reproduction_tolerance": "ci_overlap",
            "gate_result": "PASS",
        },
    )
    reg.register_cell("toy/main/cases_a4_vs_gp_bo", endpoint_id="toy_recovery", track_id="toy", baseline_ids=("gp_bo",))
    return reg


class _FakeProposer:
    def __init__(self, first, repair):
        self.first, self.repair_result = first, repair
        self.repair_inputs = []

    def propose(self, ctx):
        return self.first

    def repair(self, report):
        self.repair_inputs.append(dict(report))
        return self.repair_result


class _FakeValidator:
    def __init__(self, ok_generators):
        self.ok_generators = ok_generators

    def check(self, proposal):
        ok = any(g in self.ok_generators for g in proposal.get("generators", []))
        return ValidatorReport(
            passed=ok,
            problems=() if ok else ("unknown_generator",),
            actor_visible={"problems": [] if ok else ["unknown_generator"]},
        )


class _IdentityFrame:
    def __init__(self, frame):
        self._frame = frame

    def meet(self, a, b):
        return self._frame.meet(a, b)

    def join(self, a, b):
        return self._frame.join(a, b)

    def map_term_mask(self, m):
        return m


def test_v8_chain_composes(tmp_path) -> None:
    # 1. P0 registry: finalized, cell scoreable
    reg = _registry()
    doc = reg.finalize()
    assert doc["registry_hash"].startswith("sha256:")
    reg.assert_scoreable("toy/main/cases_a4_vs_gp_bo")

    # 2. artifacts: writer + run manifest + evaluator vault isolation
    run_dir = tmp_path / "run"
    writer = RunArtifactWriter(run_dir)
    writer.write("metrics_round", {"round": 4, "primary_endpoint": "toy_recovery", "primary_value": 0.7}, round_t=4)
    writer.write("resources_usage", {"round": 4, "llm_calls": 3, "tokens_in": 120, "tokens_out": 40}, round_t=4)
    builder = RunManifestBuilder()
    for key, value in {
        "p0_registry_hash": doc["registry_hash"],
        "commits": {"cases": "023ba37"},
        "environment": "dev/cases-core",
        "model_config_ids": ["deepseek-v4-flash", "zai-org/GLM-5.3-Flash"],
        "prompts": {"induction": "sha256:p"},
        "seed": 0,
        "budgets": {"oracle": 8},
        "thresholds": {"toy": "threshold_v1"},
        "baseline_roster": ["gp_bo"],
        "result_visibility_time": "2099-01-01T00:00:00Z",
        "representation_checkpoint_map": {"4": 2},
        "achieved_certification_level": 2,
        "terminal_status": "COMPLETE",
        "na_reasons": {},
    }.items():
        builder.set(key, value)
    manifest = builder.build()
    writer.write_manifest(manifest)
    assert writer.verify() == []

    vault = EvaluatorVault(tmp_path / "vault", signing_key=b"integration")
    vault.write("evaluator_certification_level", {"version": 1, "achieved_level": 2, "basis": "smoke"}, version=1)
    with pytest.raises(EvaluatorSealedError):
        vault.join_to_run_manifest(manifest)  # join forbidden pre-seal
    vault.seal()
    joined = vault.join_to_run_manifest(manifest)
    assert "evaluator_digest_hash" in joined
    assert vault.verify_digest() == []

    # 3. readout: claims must be evidence-linked; written as an artifact
    readout = build_readout(
        observed_supported=[{"id": "h1", "note": "recovered family A"}],
        provenance=[{"id": "p1", "claim_ids": ["h1"], "evidence_ids": ["e1", "e2"]}],
    )
    assert validate_readout(readout) == []
    writer.write("readout_computable", readout, round_t=4)
    assert writer.verify() == []

    # 4. revision: empirical trigger fires on 3 consecutive breaches
    router = RevisionRouter()
    router.register(EmpiricalPredicate("emp1", "recovery", 0.5, 3, direction="below"))
    router.require_frozen([TriggerKind.EMPIRICAL])
    kind, predicate, verdict = router.route(
        {"checkpoint_history": [{"recovery": 0.4}, {"recovery": 0.45}, {"recovery": 0.3}]}
    )
    assert kind is TriggerKind.EMPIRICAL
    event = router.make_event(
        kind,
        predicate,
        verdict,
        triggering_evidence={"metric": "recovery"},
        pre_state_ref="state_v2",
        proposed_diff={"add_generator": "g_new"},
        logic_verdict={"logic_valid": 1},
        fidelity_verdict={"round_trip": "ok"},
        event_id="ev1",
    )
    # downstream stages fill acceptance/migration/re-grounding before archiving
    event.accepted = True
    event.migration_status = "REENCODED"
    event.regrounding_coverage = {"regrounded": 5, "uncovered": 0}
    event.field_rebuild_ref = "field/posterior_round_4.json"
    event.post_metrics = {"recovery": 0.65}
    assert event.is_complete() == []
    writer.write(
        "revision_trigger",
        {"round": 4, "event_id": "ev1", "trigger": event.to_dict()["trigger"]},
        round_t=4,
        event_id="ev1",
    )
    assert writer.verify() == []

    # 5. induction: repair path admitted; actor-visible-only repair input; budget 2
    proposer = _FakeProposer({"generators": "bad"}, {"generators": ["g_ok"], "relations": []})
    validator = _FakeValidator({"g_ok"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, {"required": {"generators": list, "relations": list}})
    result = inductor.run(dossier_ref="toy://main", visible_evidence={"obs": [1]})
    assert result.outcome is InductionOutcome.ADMITTED_AFTER_REPAIR
    assert result.call_budget.total_calls == 2
    # the first proposal died at the schema check, so the repair call received
    # exactly the actor-visible schema report (Detail §3.3 step 4)
    assert proposer.repair_inputs[0] == {
        "schema_problems": ["key generators: wrong type str", "missing required key: relations"]
    }
    writer.write("representation_generators", result.contract, version=2)
    assert writer.verify() == []

    # 6. controller: frozen weights rank candidates deterministically
    ctrl = FrozenController(ControllerWeights(lambda_s=0.5, lambda_u=0.3, lambda_c=0.1, lambda_o=0.1))
    scores = ctrl.score(
        {
            "z1": {"p_sat": 0.9, "utility": 0.6, "coverage_gap": 0.1, "open_gap": 0.0},
            "z2": {"p_sat": 0.8, "utility": 0.6, "coverage_gap": 0.2, "open_gap": 0.0},
        }
    )
    decision = ctrl.select_batch(scores, 1)
    assert decision.batch == ("z1",)
    writer.write("controller_decision", {"round": 4, "scores": scores, "batch": list(decision.batch)}, round_t=4)
    assert writer.verify() == []

    # 7. statistics: paired improvement decision
    d = decide(
        [0.5 + 0.01 * s for s in range(10)],
        [0.0 for _ in range(10)],
        delta_min=0.05,
        delta_eq=0.05,
        delta_harm=0.05,
        seed=3,
    )
    assert d.label == DECISION_IMPROVES

    # 8. certification: tiny universe reaches Level 2
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    preds = {"g0": lambda w: w in {"w0", "w1"}, "g1": lambda w: w in {"w1", "w2"}}
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    report = certify_level2(universe, preds, DeclaredRelations(), _IdentityFrame(frame), [("h1", 0, True)])
    assert report.achieved_level == 2

    # everything written survives a final content-address verification
    assert writer.verify() == []
