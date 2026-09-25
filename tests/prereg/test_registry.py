"""P0-a tests: preregistration registry (CASES Experiment Detail V8 §2.2).

Contract under test:
- complete registry -> zero blocked cells, cells scoreable;
- unset mandatory endpoint/track/baseline field -> blocks ONLY cells bound to it;
- unset global mandatory section -> blocks every cell;
- post-finalize mutation raises FrozenRegistryError (P0 cannot be completed by a
  choice made after results are visible);
- content-addressing: identical content -> identical hash, any tampering is
  detected by verify_registry_hash.
"""

import copy
import json

import pytest

from cases.prereg import (
    BlockedCellError,
    FrozenRegistryError,
    PreregRegistry,
    canonical_registry_bytes,
    verify_registry_hash,
)
from cases.prereg.registry import RegistryHashMismatch

ENDPOINT_SPEC = {
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
    "multiplicity_family": "bh_utility",
}

TRACK_SPEC = {
    "task_list": ["bh_main"],
    "min_samples": 10,
    "max_samples": 10,
    "seed_list": list(range(10)),
    "budget_table": {"initial": 40, "batches": 10, "batch_size": 20},
    "checkpoints": [40, 80, 120, 160, 200, 240],
    "batch_fill_algorithm": "frozen_fill_v1",
}

BASELINE_SPEC = {
    "official_identifier": "https://github.com/example/gp-bo",
    "commit_hash": "deadbeef",
    "container_hash": "sha256:container",
    "entry_point": "gp_bo.run",
    "config": {"acq": "lse"},
    "permitted_adapter_diff": "oracle_boundary_only",
    "reproduction_tolerance": "ci_overlap",
    "gate_result": "PASS",
}


def _populate(reg: PreregRegistry) -> PreregRegistry:
    reg.set("repository.commits", {"cases_v7": "8eb0815"})
    reg.set("repository.submodule_commits", {"made": "abc123"})
    reg.set("datasets.oracle_hashes", {"bh": "sha256:oracle"})
    reg.set("datasets.dataset_hashes", {"bh": "sha256:dataset"})
    reg.set("corpus.task_dossier_hashes", {"bh": "sha256:dossier"})
    reg.set("corpus.frozen_corpus_hashes", {"bh": "sha256:corpus"})
    reg.set("model.model_id", "deepseek-v4-flash")
    reg.set("model.tokenizer_id", "deepseek-v4-flash")
    reg.set("model.temperature", 1.0)
    reg.set("model.tool_config", {"llm_base_url": "http://127.0.0.1:8002/v1"})
    reg.set("prompts.prompt_hashes", {"induction": "sha256:prompt"})
    reg.set("prompts.persistent_state_schemas", {"readout": "v1"})
    reg.set("legality.candidate_legality_rules", {"bh": "legal_unqueried_only"})
    reg.set("legality.replacement_rules", {"bh": "frozen_fill_v1"})
    reg.set("solutions.thresholds", {"bh": "top_5pct_measured_yield"})
    reg.set("solutions.outcome_blind_geometry", {"bh": "frozen_reaction_structure"})
    reg.set("budgets.stopping_rules", {"bh": "budget_exhausted"})
    reg.set("budgets.retry_policy", {"bh": "one_infra_retry"})
    reg.set("budgets.timeout_policy", {"bh": "per_call_120s"})
    reg.set("budgets.failure_rules", {"bh": "zero_from_failure_checkpoint"})
    reg.set("seeds.initial_evidence_ids", {"bh": ["b1", "b2"]})
    reg.set("adapters.identity_reports", {"bh": "sha256:identity"})
    reg.set("adapters.supported_cells", {"bh": ["cases_a4", "gp_bo"]})
    reg.set_endpoint("bh_top_region_recovery", ENDPOINT_SPEC)
    reg.set_track("bh", TRACK_SPEC)
    reg.set_baseline("gp_bo", BASELINE_SPEC)
    reg.register_cell(
        "bh/main/cases_a4_vs_gp_bo",
        endpoint_id="bh_top_region_recovery",
        track_id="bh",
        baseline_ids=("gp_bo",),
    )
    return reg


def _fresh_registry() -> PreregRegistry:
    return _populate(PreregRegistry("BH main campaign"))


def test_complete_registry_blocks_nothing() -> None:
    reg = _fresh_registry()
    reg.finalize()
    assert reg.blocked_cells() == {}
    reg.assert_scoreable("bh/main/cases_a4_vs_gp_bo")  # must not raise


def test_incomplete_endpoint_blocks_only_bound_cell() -> None:
    reg = _fresh_registry()
    # second, incomplete endpoint bound to a different cell
    reg.set_endpoint("bh_utility", {k: v for k, v in ENDPOINT_SPEC.items() if k != "delta_min"})
    reg.register_cell("bh/main/utility", endpoint_id="bh_utility", track_id="bh")
    reg.finalize()
    blocks = reg.blocked_cells()
    assert "bh/main/utility" in blocks
    assert any("endpoint:bh_utility:delta_min" in reason for reason in blocks["bh/main/utility"])
    # the complete cell stays scoreable
    reg.assert_scoreable("bh/main/cases_a4_vs_gp_bo")


def test_unregistered_track_blocks_cell() -> None:
    reg = _fresh_registry()
    reg.register_cell("gb1/main/cases_a4", endpoint_id="bh_top_region_recovery", track_id="gb1")
    reg.finalize()
    blocks = reg.blocked_cells()
    assert blocks["gb1/main/cases_a4"] == ["track:gb1:UNREGISTERED"]


def test_global_missing_section_blocks_every_cell() -> None:
    reg = PreregRegistry("BH main campaign")
    _populate(reg)
    # wipe one global mandatory path after population
    del reg._document["content"]["solutions"]["thresholds"]
    reg.register_cell("bh/main/second", endpoint_id="bh_top_region_recovery", track_id="bh")
    reg.finalize()
    blocks = reg.blocked_cells()
    assert set(blocks) == {"bh/main/cases_a4_vs_gp_bo", "bh/main/second"}
    for reasons in blocks.values():
        assert "global:solutions.thresholds" in reasons
    with pytest.raises(BlockedCellError):
        reg.assert_scoreable("bh/main/cases_a4_vs_gp_bo")


def test_not_finalized_rejects_scored_run() -> None:
    reg = _fresh_registry()
    with pytest.raises(FrozenRegistryError):
        reg.assert_scoreable("bh/main/cases_a4_vs_gp_bo")


def test_finalize_freezes_document() -> None:
    reg = _fresh_registry()
    reg.finalize()
    with pytest.raises(FrozenRegistryError):
        reg.set("model.temperature", 0.7)
    with pytest.raises(FrozenRegistryError):
        reg.set_endpoint("bh_top_region_recovery", ENDPOINT_SPEC)
    with pytest.raises(FrozenRegistryError):
        reg.register_cell("bh/main/new", endpoint_id="bh_top_region_recovery", track_id="bh")


def test_hash_is_content_addressed_and_tamper_evident() -> None:
    doc_a = _fresh_registry().finalize()
    doc_b = _fresh_registry().finalize()
    assert doc_a["registry_hash"] == doc_b["registry_hash"]

    verify_registry_hash(doc_a)  # must not raise

    tampered = copy.deepcopy(doc_a)
    tampered["content"]["endpoints"]["bh_top_region_recovery"]["delta_min"] = 0.99
    with pytest.raises(RegistryHashMismatch):
        verify_registry_hash(tampered)

    # dropping publish metadata must not change the content envelope
    envelope = {k: doc_a[k] for k in ("schema_version", "title", "created_at", "content")}
    expected = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert canonical_registry_bytes(doc_a) == expected


def test_temperature_rejects_bool() -> None:
    reg = PreregRegistry("type guard")
    _populate(reg)
    reg.set("model.temperature", True)  # bool is not a valid temperature
    reg.register_cell("bh/main/cases_a4_vs_gp_bo", endpoint_id="bh_top_region_recovery", track_id="bh")
    # remove the real temperature by overwrite: set(True) already replaced it
    reg.finalize()
    blocks = reg.blocked_cells()
    assert any("global:model.temperature" in reason for reasons in blocks.values() for reason in reasons)
