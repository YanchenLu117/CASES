"""this infrastructure tests: runout artifact pipeline, kernel factory, contract diff."""

from __future__ import annotations

import json

import pytest

from cases.core.contract import contract_diff
from cases.recovery.kernel_factory import resolve_kernel
from cases.runout import RunArchiver


def _boolean_model():
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask
    from cases.core.model import EvidenceRecord, CASESModel
    from cases.core.types import LanguageHypothesis

    task = BooleanTask(
        dataset="boolean_test", observation_set_id="t", variables=("x", "y"),
        operators=frozenset({"AND", "OR", "NOT"}), max_depth=2,
        mechanistic_opts={"apply_commutativity": True, "apply_idempotence_and_or": True,
                          "flatten_associativity": True},
        observations=(), n_observations=0,
    )
    m = CASESModel(adapter=BooleanAdapter(task))
    m.update([
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t))
        for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
    ])
    m.recover()
    return m


def _spec(**kw):
    from cases.core.representation import (
        RelationDecl, RepresentationSpecification, TransformDecl, TypedVariable,
    )

    base = dict(spec_id="s0", name="t", description="t",
                typed_variables=(TypedVariable("a", "continuous", "state"),),
                relations=(RelationDecl("r", "obj-obj", "objects"),),
                transforms=(TransformDecl("e", "obj->obj", True),),
                semantic_commitments=("a",))
    base.update(kw)
    return RepresentationSpecification(**base)


def test_contract_diff_added_variable():
    from cases.core.representation import TypedVariable

    old = _spec()
    new = _spec(typed_variables=(TypedVariable("a", "continuous", "state"),
                                 TypedVariable("b", "continuous", "state")))
    d = contract_diff(old, new)
    assert d["typed_variables"]["added"] == ["b"]
    assert d["typed_variables"]["removed"] == []


def test_contract_diff_from_none():
    d = contract_diff(None, _spec())
    assert d["version"] == 1
    assert d["backend_changed"] is True or d["backend_changed"] is False


def test_kernel_resolve_fallback_laplacian():
    dec = resolve_kernel(None)
    assert dec.backend == "laplacian"
    assert dec.provenance["reason"] == "no contract -> fallback"


def test_kernel_resolve_rbf_geometry():
    spec = _spec(geometry={"kernel": "rbf", "features": "euclidean"})
    dec = resolve_kernel(spec, override="rbf_gp")
    assert dec.backend == "rbf_gp"
    assert dec.features_scheme == "euclidean"


def test_kernel_resolve_hint_governs():
    spec = _spec(geometry={"kernel": "rbf"}, backend_hint="graph_matern")
    dec = resolve_kernel(spec, override="graph_matern")
    assert dec.backend == "graph_matern"


def test_runout_writes_standard_layout(tmp_path):
    m = _boolean_model()
    with RunArchiver(tmp_path / "run", model_id="test", seed=0, regime="controlled") as arc:
        written = arc.write_round(m, round_idx=0)
    files = arc.list_artifacts()
    # core artifacts must be present
    for need in ("run_manifest.json", "evidence/ledger.jsonl", "field/posterior_round_0.npz",
                 "readout/natural_language_round_0.md", "readout/computable_round_0.json",
                 "controller/decision_round_0.json", "oracle/observations_round_0.jsonl",
                 "metrics/round_0.json", "resources/usage_round_0.json"):
        assert need in files, f"missing {need}"
    # parse a couple
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["regime"] == "controlled"
    ledger = (tmp_path / "run" / "evidence" / "ledger.jsonl").read_text().strip()
    assert ledger  # non-empty
    ctrl = json.loads((tmp_path / "run" / "controller" / "decision_round_0.json").read_text())
    assert "gear" in ctrl  # gear recorded dynamically, no hardcode required


def test_runout_ledger_idempotent(tmp_path):
    m = _boolean_model()
    arc = RunArchiver(tmp_path / "run", seed=0)
    arc.write_round(m, round_idx=0)
    first = (tmp_path / "run" / "evidence" / "ledger.jsonl").read_text()
    arc.write_round(m, round_idx=0)  # re-write same round: no new evidence
    second = (tmp_path / "run" / "evidence" / "ledger.jsonl").read_text()
    assert first == second  # idempotent


def test_contract_diff_revised_spec_no_crash():
    """infra-1 audit fix: diffing a revised '.r' spec must not crash, and the
    real version is carried (never parsed from spec_id)."""
    old = _spec(spec_id="p_t")
    new = _spec(spec_id="p_t.r")
    d = contract_diff(old, new, version=2)
    assert d["version"] == 2
    assert "typed_variables" in d and "geometry" in d


def test_runout_controller_decision_full_fields(tmp_path):
    """infra-1 audit fix: §12.2 decision record carries all mandated fields."""
    import json as _json

    m = _boolean_model()
    arc = RunArchiver(tmp_path / "r", seed=0)
    arc.write_round(m, round_idx=0)
    d = _json.loads((tmp_path / "r" / "controller" / "decision_round_0.json").read_text())
    for key in ("gear", "channel", "channel_probability", "candidate_mask",
                "selection_probability", "fallback"):
        assert key in d, f"missing controller decision field: {key}"


def test_runout_figure_slot_exists(tmp_path):
    """infra-1 audit fix: a figures/ artifact always exists for the round."""
    m = _boolean_model()
    arc = RunArchiver(tmp_path / "r", seed=0)
    arc.write_round(m, round_idx=0)
    figs = [a for a in arc.list_artifacts() if a.startswith("figures/")]
    assert figs, "no figures artifact written"


def test_kernel_sequence_mixed_scheme():
    """kernel factory reports sequence/mixed features_scheme from geometry."""
    assert resolve_kernel(_spec(geometry={"kernel": "string"})).features_scheme == "sequence"
    assert resolve_kernel(_spec(geometry={"kernel": "hamming"})).features_scheme == "sequence"
    assert resolve_kernel(_spec(geometry={"kernel": "mixed-typed"})).features_scheme == "mixed"


def test_runout_cross_instance_ledger_idempotent(tmp_path):
    """a NEW RunArchiver over the same run dir must keep the ledger unchanged."""
    m = _boolean_model()
    RunArchiver(tmp_path / "r", seed=0).write_round(m, round_idx=0)
    first = (tmp_path / "r" / "evidence" / "ledger.jsonl").read_text()
    RunArchiver(tmp_path / "r", seed=0).write_round(m, round_idx=0)  # fresh instance
    second = (tmp_path / "r" / "evidence" / "ledger.jsonl").read_text()
    assert first == second


def test_runout_manifest_token_budget_and_prompt_hash(tmp_path):
    m = _boolean_model()
    arc = RunArchiver(tmp_path / "r", seed=0, model_id="m1")
    arc.write_round(m, round_idx=0, token_budget=987)
    manifest = json.loads((tmp_path / "r" / "run_manifest.json").read_text())
    assert manifest.get("token_budget") == 987
    assert "prompt_hash" in manifest
    assert manifest.get("model_id") == "m1"


def test_backend_consumes_kernel_factory():
    """infra final-audit major#1: the field's substrate is driven by the adopted
    contract via resolve_kernel, and the decision is recorded & auditable."""
    m = _boolean_model()
    m.recover()
    kd = m.state.metadata.get("kernel_decision")
    assert kd is not None
    assert kd["backend"] in {"laplacian", "rbf_gp"}
    assert "provenance" in kd and kd["provenance"]
    assert m.backend is not None
