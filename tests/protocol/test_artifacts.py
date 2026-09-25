"""P0-e tests: artifact standard and evaluator isolation (Detail §15)."""

import json

import pytest

from cases.protocol.artifacts import (
    ACTOR_TREE,
    EVALUATOR_TREE,
    RUN_MANIFEST_MANDATORY,
    ArtifactSchemaError,
    EvaluatorSealedError,
    EvaluatorVault,
    RunArtifactWriter,
    RunManifestBuilder,
)


def _full_run_manifest() -> dict:
    builder = RunManifestBuilder()
    builder.set("p0_registry_hash", "sha256:abc")
    builder.set("commits", {"cases": "023ba37"})
    builder.set("environment", "cases-core")
    builder.set("model_config_ids", ["deepseek-v4-flash"])
    builder.set("prompts", {"induction": "sha256:p"})
    builder.set("seed", 7)
    builder.set("budgets", {"oracle": 240})
    builder.set("thresholds", {"solution": "top_5pct"})
    builder.set("baseline_roster", ["gp_bo", "gryffin_plain"])
    builder.set("result_visibility_time", "2099-01-01T00:00:00Z")
    builder.set("representation_checkpoint_map", {"3": 2})
    builder.set("achieved_certification_level", 1)
    builder.set("terminal_status", "COMPLETE")
    builder.set("na_reasons", {})
    return builder.build()


def test_actor_tree_covers_detail_15_paths() -> None:
    rels = [rel for _, rel in ACTOR_TREE]
    assert "run_manifest.json" in rels
    assert "artifact_manifest.sha256.json" in rels
    assert "evidence/ledger.jsonl" in rels
    assert "representation/certificate_v{version}.json" in rels
    assert "revision/migration_round_{round}_{event_id}.json" in rels
    assert len(rels) == 39


def test_evaluator_tree_six_kinds() -> None:
    assert len(EVALUATOR_TREE) == 6
    assert all(rel.startswith("evaluator/") for _, rel in EVALUATOR_TREE)


def test_writer_resolves_templates() -> None:
    writer = RunArtifactWriter("/tmp/cases_test_run_a")
    assert writer.resolve_path("metrics_round", round_t=3) == "metrics/round_3.json"
    assert writer.resolve_path("representation_contract", version=2) == "representation/contract_v2.json"
    assert writer.resolve_path("revision_diff", round_t=4, event_id="ev1") == "revision/diff_round_4_ev1.json"
    with pytest.raises(KeyError):
        writer.resolve_path("nonexistent_kind")


def test_writer_validates_payloads_against_schemas() -> None:
    writer = RunArtifactWriter("/tmp/cases_test_run_b")
    with pytest.raises(ArtifactSchemaError):
        writer.write("metrics_round", {"round": 1})  # missing primary_endpoint
    with pytest.raises(ArtifactSchemaError):
        writer.write("metrics_round", {"round": 1, "primary_endpoint": "x", "primary_value": 1})  # round_t required
    writer.write("metrics_round", {"round": 1, "primary_endpoint": "recovery_aurc", "primary_value": 0.42}, round_t=1)
    assert (writer.run_dir / "metrics/round_1.json").exists()


def test_writer_jsonl_and_md_kinds() -> None:
    writer = RunArtifactWriter("/tmp/cases_test_run_c")
    writer.write("evidence_ledger", [{"e": 1}, {"e": 2}])
    text = (writer.run_dir / "evidence/ledger.jsonl").read_text(encoding="utf-8")
    assert text.count("\n") == 2
    writer.write("readout_natural_language", "# readout\n", round_t=1)
    assert (writer.run_dir / "readout/natural_language_round_1.md").read_text().startswith("# readout")
    with pytest.raises(ArtifactSchemaError):
        writer.write("native_raw_output", {"not": "a list"}, round_t=1)


def test_manifest_content_addresses_and_verifies(tmp_path) -> None:
    writer = RunArtifactWriter(tmp_path)
    writer.write("metrics_round", {"round": 1, "primary_endpoint": "x", "primary_value": 1}, round_t=1)
    writer.write("resources_usage", {"round": 1, "llm_calls": 2, "tokens_in": 10, "tokens_out": 5}, round_t=1)
    rm = _full_run_manifest()
    writer.write_manifest(rm)
    assert writer.verify() == []
    # tampering is detected
    metrics = tmp_path / "metrics/round_1.json"
    metrics.write_text('{"round": 1, "primary_endpoint": "x", "primary_value": 999}')
    problems = writer.verify()
    assert any("hash mismatch" in p for p in problems)


def test_run_manifest_refuses_missing_fields() -> None:
    builder = RunManifestBuilder()
    builder.set("seed", 1)
    missing = builder.missing_fields()
    assert "p0_registry_hash" in missing and "terminal_status" in missing
    with pytest.raises(ArtifactSchemaError):
        builder.build()
    assert set(RUN_MANIFEST_MANDATORY) == set(_full_run_manifest()) - {"schema_version", "generated_at"}


def test_evaluator_vault_isolation(tmp_path) -> None:
    vault = EvaluatorVault(tmp_path, signing_key=b"k")
    vault.write("evaluator_certification_level", {"version": 2, "achieved_level": 2}, version=2)
    vault.log_access("acting_process", "blocked_read_attempt")
    with pytest.raises(EvaluatorSealedError):
        vault.join_to_run_manifest(_full_run_manifest())
    vault.seal()
    assert vault.sealed
    joined = vault.join_to_run_manifest(_full_run_manifest())
    assert "evaluator_digest_hash" in joined
    assert vault.verify_digest() == []
    # access log is append-only jsonl with reader + ts
    log = tmp_path / "evaluator/access_log.jsonl"
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["reader"] == "acting_process" and "ts" in entry


def test_evaluator_digest_detects_tampering(tmp_path) -> None:
    vault = EvaluatorVault(tmp_path, signing_key=b"k")
    vault.write("evaluator_hidden_relations", {"version": 1, "relations": []}, version=1)
    vault.seal()
    assert vault.verify_digest() == []
    # case 1: artifact file tampered -> content hash mismatch (signature intact)
    target = tmp_path / "evaluator/hidden_relations_v1.json"
    target.write_text('{"version": 1, "relations": ["forgotten"]}')
    assert vault.verify_digest() == ["hash mismatch: evaluator/hidden_relations_v1.json"]
    # case 2: digest manifest itself altered -> signature mismatch
    digest_path = tmp_path / "evaluator/digest_manifest.json"
    digest = json.loads(digest_path.read_text())
    digest["evaluator_artifacts"]["evaluator/hidden_relations_v1.json"] = "sha256:0" * 1
    digest_path.write_text(json.dumps(digest, sort_keys=True, separators=(",", ":")))
    assert vault.verify_digest() == ["evaluator digest signature mismatch"]
