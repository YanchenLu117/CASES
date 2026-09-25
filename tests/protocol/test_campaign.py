"""P0-f tests part 5: campaign harness wiring (Detail §2.3/§3/§6 loop)."""

import json

import pytest

from cases.protocol.campaign import CampaignResult, ToyOracle, run_campaign
from cases.protocol import (
    RemainingBudgets,
    RoundInputs,
    RoundOutputs,
    RecoveryPrediction,
    RunArtifactWriter,
)


class GreedyMethod:
    """Toy acting process: proposes the highest-scored unqueried candidates."""

    def __init__(self, scores: dict[str, float], batch: int = 2):
        self.scores = scores
        self.batch = batch

    def round_step(self, inputs: RoundInputs, history: dict[str, float]) -> RoundOutputs:
        legal = inputs.legal_candidate_ids
        ranked = sorted(legal, key=lambda c: (-self.scores.get(c, 0.0), c))
        proposals = tuple(ranked[: self.batch])
        prediction = RecoveryPrediction(scores={c: self.scores.get(c, 0.0) for c in legal})
        return RoundOutputs(
            proposed_candidate_ids=proposals,
            persistent_state_ref="state://toy",
            recovery_prediction=prediction,
            decision_trace_ref="trace://toy",
            resource_usage={"llm_calls": 1},
            completion_status="RUNNING",
        )


def _registry():
    from cases.prereg import PreregRegistry

    reg = PreregRegistry("campaign smoke")
    reg.set("repository.commits", {"cases": "x"})
    reg.set("repository.submodule_commits", {"m": "y"})
    reg.set("datasets.oracle_hashes", {"toy": "sha256:o"})
    reg.set("datasets.dataset_hashes", {"toy": "sha256:d"})
    reg.set("corpus.task_dossier_hashes", {"toy": "sha256:ds"})
    reg.set("corpus.frozen_corpus_hashes", {"toy": "sha256:c"})
    reg.set("model.model_id", "toy")
    reg.set("model.tokenizer_id", "toy")
    reg.set("model.temperature", 1.0)
    reg.set("model.tool_config", {"u": "http://127.0.0.1:8002/v1"})
    reg.set("prompts.prompt_hashes", {"p": "sha256:p"})
    reg.set("prompts.persistent_state_schemas", {"readout": "v1"})
    reg.set("legality.candidate_legality_rules", {"toy": "legal_unqueried_only"})
    reg.set("legality.replacement_rules", {"toy": "frozen_fill_v1"})
    reg.set("solutions.thresholds", {"toy": "t"})
    reg.set("solutions.outcome_blind_geometry", {"toy": "g"})
    reg.set("budgets.stopping_rules", {"toy": "s"})
    reg.set("budgets.retry_policy", {"toy": "r"})
    reg.set("budgets.timeout_policy", {"toy": "t"})
    reg.set("budgets.failure_rules", {"toy": "f"})
    reg.set("seeds.initial_evidence_ids", {"toy": ["i1"]})
    reg.set("adapters.identity_reports", {"toy": "sha256:i"})
    reg.set("adapters.supported_cells", {"toy": ["cases_a4"]})
    reg.set_endpoint(
        "toy_recovery",
        {
            "metric_function": "recovery_aurc_v1", "curve_scalarization": "trap",
            "favorable_direction": "maximize", "delta_min": 0.01, "delta_eq": 0.01,
            "delta_harm": 0.02, "failure_score": 0.0, "ci_method": "b",
            "effect_size_formula": "d", "resampling_scheme": "seed",
            "multiplicity_family": "toy",
        },
    )
    reg.set_track(
        "toy",
        {
            "task_list": ["toy"], "min_samples": 1, "max_samples": 1,
            "seed_list": [0], "budget_table": {"initial": 1}, "checkpoints": [2, 4],
            "batch_fill_algorithm": "fill",
        },
    )
    reg.set_baseline(
        "gp_bo",
        {
            "official_identifier": "toy://gp", "commit_hash": "h", "container_hash": "sha256:c",
            "entry_point": "e", "config": {"a": 1}, "permitted_adapter_diff": "none",
            "reproduction_tolerance": "ci", "gate_result": "PASS",
        },
    )
    reg.register_cell("toy/main", endpoint_id="toy_recovery", track_id="toy", baseline_ids=("gp_bo",))
    reg.finalize()  # P0 must be published before any scored run
    return reg


def test_campaign_end_to_end(tmp_path) -> None:
    legal = (f"c{i}" for i in range(1, 9))
    oracle = ToyOracle(
        legal_ids=tuple(legal),
        observations={f"c{i}": i / 10 for i in range(1, 9)},
        solution_ids={"c8", "c7"},
        initial_ids={"c1"},
    )
    method = GreedyMethod({f"c{i}": i / 10 for i in range(1, 9)}, batch=2)
    result = run_campaign(
        prereg=_registry(),
        cell_id="toy/main",
        method=method,
        oracle=oracle,
        budgets=[2, 4, 6],
        run_dir=tmp_path / "run",
        initial_ids={"c1"},
    )
    assert isinstance(result, CampaignResult)
    assert result.complete
    assert 0.0 <= result.recovery_auc <= 1.0
    # greedy with perfect scores recovers the top solution members first
    assert result.checkpoints[0].recovery == pytest.approx(1.0)
    assert result.recovery_auc == pytest.approx(1.0)
    # oracle discipline: initial + greedy top-scored = queried; no silent replacement
    assert oracle.queried == {"c1", "c8", "c7", "c6", "c5", "c4"}
    ledger_records = json.loads((tmp_path / "run/evidence/ledger.jsonl").read_text().splitlines()[0])
    assert ledger_records["cid"] == "c8"  # greedy queries the top-scored candidate first


def test_campaign_illegal_and_duplicate_slots_consume_no_labels(tmp_path) -> None:
    oracle = ToyOracle(
        legal_ids=("a", "b", "c"),
        observations={"a": 0.1, "b": 0.2, "c": 0.3},
        solution_ids={"c"},
        initial_ids={"a"},
    )

    class BadMethod:
        def round_step(self, inputs, history):
            return RoundOutputs(
                proposed_candidate_ids=("a", "ghost", "c"),  # duplicate + illegal
                persistent_state_ref="s",
                recovery_prediction=RecoveryPrediction(scores={"b": 0.5, "c": 0.9, "ghost": 0.9}),
                decision_trace_ref="t",
                resource_usage={},
                completion_status="RUNNING",
            )

    result = run_campaign(
        prereg=_registry(),
        cell_id="toy/main",
        method=BadMethod(),
        oracle=oracle,
        budgets=[2, 3],
        run_dir=tmp_path / "run2",
        initial_ids={"a"},
    )
    # "ghost" (illegal) and duplicate "a" consumed slots; only c got a label
    assert result.checkpoints[0].n_illegal >= 1 or result.checkpoints[0].n_duplicates >= 1
    assert result.complete
    # the common universe excludes everything queried by ANY arm — here the
    # single arm queried a (initial), a(dup), c
    writer = RunArtifactWriter(tmp_path / "run2")
    assert writer.verify() == []


def test_campaign_blocked_without_scoreable_cell(tmp_path) -> None:
    reg = _registry()
    with pytest.raises(Exception):
        run_campaign(
            prereg=reg,
            cell_id="toy/UNREGISTERED",
            method=GreedyMethod({}),
            oracle=ToyOracle(("a",), {"a": 1.0}, {"a"}, {"a"}),
            budgets=[1, 2],
            run_dir=tmp_path / "run",
            initial_ids={"a"},
        )


def test_published_registry_document_reloads_and_blocks(tmp_path) -> None:
    reg = _registry()
    doc = reg.finalize()
    tampered = dict(doc)
    tampered["content"] = json.loads(json.dumps(doc["content"]))
    tampered["content"]["endpoints"]["toy_recovery"]["delta_min"] = 99.0
    oracle = ToyOracle(("a", "b"), {"a": 1.0, "b": 2.0}, {"b"}, {"a"})
    with pytest.raises(Exception):
        run_campaign(
            prereg=tampered,  # hash mismatch — tampered P0 cannot drive a run
            cell_id="toy/main",
            method=GreedyMethod({}),
            oracle=oracle,
            budgets=[1, 2],
            run_dir=tmp_path / "run",
            initial_ids={"a"},
        )
