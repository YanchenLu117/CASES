"""this Core representation-layer tests (representation + revision + dual gate).

Covers the restored this Core objects (proposal §5, §7, §11-§12, §21-§25):
  - RepresentationSpecification / default induction (P_t)
  - LogicGate  L_tau  (Type/Structural/Executable/Semantic) + FidelityGate F_t
  - dual admission gate (LogicValid AND F_t >= eps_F), paradigm-respecting
  - bidirectional LanguageCorrespondence backward (Gamma^left)
  - revision triggers + evidence-driven representation revision

Pure-computation path only — no LLM gateway required.
"""

from __future__ import annotations

import pytest


def _boolean_adapter():
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask

    task = BooleanTask(
        dataset="boolean_test",
        observation_set_id="t",
        variables=("x", "y"),
        operators=frozenset({"AND", "OR", "NOT"}),
        max_depth=2,
        mechanistic_opts={
            "apply_commutativity": True,
            "apply_idempotence_and_or": True,
            "flatten_associativity": True,
        },
        observations=(),
        n_observations=0,
    )
    return BooleanAdapter(task)


def _model(**kw):
    from cases.core.model import CASESModel

    return CASESModel(adapter=_boolean_adapter(), **kw)


def _spec(**kw):
    from cases.core.representation import (
        RelationDecl,
        RepresentationSpecification,
        TransformDecl,
        TypedVariable,
    )

    base = dict(
        spec_id="s0",
        name="test-spec",
        description="test",
        typed_variables=(TypedVariable("v", "continuous", "state"),),
        relations=(RelationDecl("rel", "obj-obj", "objects", outcome_blind=True),),
        transforms=(TransformDecl("edit", "obj->obj", executable=True),),
        semantic_commitments=("n_ops", "depth"),
    )
    base.update(kw)
    return RepresentationSpecification(**base)


# ---------------------------------------------------------------------------
# representation specification + induction
# ---------------------------------------------------------------------------


def test_default_induction_admitted_full_v7():
    m = _model()
    spec = m.representation  # lazy induce + admit
    assert spec.source == "adapter_default"
    assert spec.admitted is True  # dual gate passes for the faithful adapter spec
    assert m.representational_state()["P_t"]["admitted"] is True
    # completed persistent state S^this has all five slots
    state = m.representational_state()
    assert set(state) == {"D_t", "P_t", "C_t", "Gamma_t", "B_t"}
    assert state["Gamma_t"]["bidirectional"] is True


def test_paradigm_default():
    from cases.core.representation import RepresentationParadigm

    m = _model()
    assert m.paradigm is RepresentationParadigm.FULL
    assert m.paradigm.runs_logic_gate and m.paradigm.runs_fidelity_gate


# ---------------------------------------------------------------------------
# logic gate  L_tau
# ---------------------------------------------------------------------------


def test_logic_gate_rejects_outcome_leak():
    from cases.core.representation import LogicGate

    gate = LogicGate(_boolean_adapter())
    leaky = _spec(semantic_commitments=("yield_of_ligand", "n_ops"))
    report = gate.check(leaky)
    assert report.valid is False
    assert any(v.check == "semantic" for v in report.violations)


def test_logic_gate_rejects_no_transforms():
    from cases.core.representation import LogicGate

    gate = LogicGate(_boolean_adapter())
    noskill = _spec(transforms=())
    report = gate.check(noskill)
    assert report.valid is False
    assert any(v.check == "executable" for v in report.violations)


def test_logic_gate_rejects_bad_type():
    from cases.core.representation import LogicGate, TypedVariable

    gate = LogicGate(_boolean_adapter())
    bad = _spec(typed_variables=(TypedVariable("v", "blob", "state"),))
    report = gate.check(bad)
    assert report.valid is False
    assert any(v.check == "type" for v in report.violations)


def test_logic_gate_accepts_clean_spec():
    from cases.core.representation import LogicGate

    gate = LogicGate(_boolean_adapter())
    assert gate.check(_spec()).valid is True


# ---------------------------------------------------------------------------
# fidelity gate  F_t
# ---------------------------------------------------------------------------


def test_fidelity_score_uses_adapter_domains():
    m = _model()
    assert m.fidelity_score() == pytest.approx(1.0, abs=1e-9)  # all commitments exposed


def test_fidelity_low_when_commitments_unexposed():
    from cases.core.representation import FidelityGate

    gate = FidelityGate(_boolean_adapter())
    # half the commitments are not real adapter dimensions => F_t = 0.5
    ok, f = gate.admissible(_spec(semantic_commitments=("not_a_real_dim", "depth")), epsilon_f=0.9)
    assert ok is False
    assert f == pytest.approx(0.5)  # 1 of 2 honored


# ---------------------------------------------------------------------------
# dual gate + paradigm arms
# ---------------------------------------------------------------------------


def test_v7_no_logic_bypasses_gate():
    from cases.core.representation import RepresentationParadigm

    m = _model(paradigm=RepresentationParadigm.INDUCED_NO_LOGIC)
    leaky = _spec(semantic_commitments=("yield",))
    admitted, report, f = m.admit_representation(leaky)
    assert report.valid is True  # logic gate bypassed
    assert admitted.admitted is True  # fidelity gate also bypassed


def test_admit_representation_dual_gate():
    m = _model()
    admitted, report, f = m.admit_representation(_spec())
    assert report.valid is True
    assert f >= m.epsilon_f
    assert admitted.admitted is True


# ---------------------------------------------------------------------------
# backward correspondence  Gamma^left
# ---------------------------------------------------------------------------


def test_backward_correspondence():
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    m = _model()
    m.update(
        [
            EvidenceRecord(
                kind="hypothesis",
                hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
            )
            for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
        ]
    )
    m.recover()
    text = m.interpret_solution_space()
    assert isinstance(text, str) and text.strip()
    assert "Recovered solution space" in text
    # after recover without observations the posterior is non-discriminative on
    # HypoSpace (no scalar field); S_t(eta) is empty, so we map the empty set.
    assert "no recovered solutions" in text


# ---------------------------------------------------------------------------
# revision triggers + evidence-driven representation revision
# ---------------------------------------------------------------------------


def test_empirical_trigger_before_recover():
    from cases.core.revision import RevisionTrigger

    m = _model()
    signal = m.check_revision()  # no recover yet -> posterior not discriminative
    assert signal.fires(RevisionTrigger.EMPIRICAL)


def test_ungroundable_trigger_from_metadata():
    from cases.core.revision import RevisionTrigger

    m = _model()
    m.state.metadata["ungroundable_hypotheses"] = 2
    signal = m.check_revision()
    assert signal.fires(RevisionTrigger.UNGROUNDABLE)


def test_non_adaptive_paradigms_never_fire():
    from cases.core.representation import RepresentationParadigm

    for p in (RepresentationParadigm.FIXED_EMBEDDING, RepresentationParadigm.FIXED_SYMBOLIC):
        m = _model(paradigm=p)
        signal = m.check_revision()
        assert signal.any is False


def test_revision_tightens_fidelity_and_readmits():
    m = _model(epsilon_f=1.0)  # strict fidelity threshold
    low_fid = _spec(semantic_commitments=("not_a_real_dim", "depth"))  # F=0.5 < eps=1.0
    m._representation = low_fid
    m._representation_resolved = True
    report = m.revise_representation()
    assert report.needed is True
    assert report.new_spec is not None
    # the revised spec drops the unexposed commitment, restoring fidelity
    assert m.representation.admitted is True
    assert m.fidelity_score() == pytest.approx(1.0, abs=1e-9)


def test_persistent_state_has_representation():
    m = _model()
    rep = m.representation  # explicitly resolve/induce P_t (snapshot stays lazy)
    snap = m.snapshot()
    assert snap["paradigm"] == "full"
    assert snap["representation"]["spec_id"] == rep.spec_id
    assert snap["representation"]["admitted"] is True
    assert snap["representation"]["backend"] == "laplacian"


def test_import_surface_representation():
    import cases

    assert hasattr(cases, "RepresentationParadigm")
    from cases.core import (
        LogicGate,
        FidelityGate,
        RepresentationRuntime,
        RepresentationSpecification,
        RepresentationParadigm,
        RepresentationRevisioner,
        RevisionTrigger,
    )  # noqa: F401


# ---------------------------------------------------------------------------
# this Core coherence: C_t drives B_t, explicit-backend honor, revision rebuilds
# ---------------------------------------------------------------------------


def test_representation_led_backend_selects_hint():
    from cases.core.representation import (
        RelationDecl,
        RepresentationParadigm,
        RepresentationSpecification,
        TransformDecl,
        TypedVariable,
    )

    m = _model()  # no explicit backend, Full
    # the default induced spec selects the laplacian graph substrate
    assert m._backend().name == "laplacian"
    # craft a Euclidean-hinted spec; the runtime should select rbf_gp for C_t
    euclid = RepresentationSpecification(
        spec_id="p_e", name="e", description="e",
        typed_variables=(TypedVariable("a", "continuous", "state"),),
        relations=(RelationDecl("r", "obj-obj", "objects"),),
        transforms=(TransformDecl("t", "obj->obj", True),),
        semantic_commitments=("a",),
        backend_hint="rbf_gp",
        provenance={"paradigm": RepresentationParadigm.FULL.value},
    )
    m._representation = euclid
    m._representation_resolved = False
    assert m._backend().name == "rbf_gp"


def test_explicit_backend_honored():
    from cases.recovery.registry import create_backend

    m = _model(backend=create_backend("graph_matern"))
    assert m._backend().name == "graph_matern"


def test_revision_rebuilds_field():
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    m = _model(epsilon_f=1.0)
    m.update(
        [
            EvidenceRecord(
                kind="hypothesis",
                hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
            )
            for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
        ]
    )
    m.recover()
    m._representation = _spec(semantic_commitments=("not_a_real_dim", "depth"))  # F=0.5 < 1
    m._representation_resolved = True
    report = m.revise_representation()
    assert report.needed is True
    # invariant: evidence unchanged, field rebuilt under the new C_t
    assert m.state.metadata.get("representation_field_rebuilt") is True
    assert m.state.posterior is not None


def test_inducer_default_no_llm():
    from cases.core.representation import RepresentationParadigm

    m = _model()
    s = m._inducer.induce(_boolean_adapter(), paradigm=RepresentationParadigm.FULL)
    assert s.source == "adapter_default"
    assert s.semantic_commitments


def test_logic_gate_duplicate_variable():
    from cases.core.representation import LogicGate, TypedVariable

    gate = LogicGate(_boolean_adapter())
    dup = _spec(typed_variables=(TypedVariable("v", "continuous", "state"),
                                 TypedVariable("v", "continuous", "state")))
    report = gate.check(dup)
    assert report.valid is False
    assert any(v.check == "type" and "more than once" in v.message for v in report.violations)


# ---------------------------------------------------------------------------
# Audit-fix tests: three-component fidelity certificate, contract G/K fields,
# LogicGate hard constraints, and the (now live) async LLM-induced path.
# ---------------------------------------------------------------------------


def test_fidelity_certificate_three_components():
    from cases.core.representation import FidelityGate

    gate = FidelityGate(_boolean_adapter())
    m = _model()
    cert = gate.certificate(_spec())  # both commitments are real domain dims
    assert cert.F_query == pytest.approx(1.0)
    assert cert.F_dist == pytest.approx(1.0)  # vacuous (no objects)
    assert cert.F_impl == pytest.approx(1.0)
    # scalar score is the conservative min over components
    assert m.fidelity_score() >= 0.0


def test_fidelity_componentwise_admission():
    from cases.core.representation import FidelityGate

    gate = FidelityGate(_boolean_adapter())
    ok, cert = gate.admitted_componentwise(
        _spec(semantic_commitments=("not_a_real_dim", "depth")), epsilon_f=0.9
    )
    assert ok is False                      # F_query = 0.5
    assert cert.F_query == pytest.approx(0.5)
    assert cert.as_tuple()[:3] == (0.5, 1.0, 1.0)


def test_contract_has_geometry_and_grounding_fields():
    from cases.core.representation import RepresentationInducer

    spec = RepresentationInducer().induce(_boolean_adapter())
    # paper G (grounding) and K (geometry) components are first-class
    assert hasattr(spec, "grounding") and hasattr(spec, "geometry")
    assert hasattr(spec, "entities")
    # backward compatible: fields default cleanly
    from cases.core.representation import RepresentationSpecification

    bare = RepresentationSpecification(spec_id="s", name="n", description="d")
    assert bare.geometry is None and bare.grounding is None and bare.entities == ()


def test_logicgate_hard_constraints():
    from cases.core.representation import LogicGate

    gate = LogicGate(_boolean_adapter(), hard_constraints=(lambda s: False,))
    report = gate.check(_spec())
    assert report.valid is False
    assert any(v.check == "semantic" and "hard constraint" in v.message for v in report.violations)


def test_llm_induced_path_is_live():
    """The async LLM-proposal path actually runs (was previously dead)."""

    class _FakeJSONLLM:
        provider_name = "fake"
        model_name = "fake-m"

        async def generate_json(self, request=None, **kwargs):
            return {
                "typed_variables": [{"name": "x", "vtype": "continuous", "role": "state"}],
                "relations": [{"name": "r", "polymorphic_type": "obj-obj", "scope": "objects"}],
                "transforms": [{"name": "edit", "arity": "obj->obj"}],
                "semantic_commitments": ["a", "b"],
            }

        async def generate(self, request=None, **kwargs):
            from cases.llm.base import LLMResponse

            return LLMResponse(text="", parsed=None, tool_calls=(), finish_reason="stop")

    m = _model(llm=_FakeJSONLLM())
    spec = m.representation
    assert spec.source == "llm_induced"  # the LLM proposal ran, no silent fallback


def test_default_spec_populates_g_k_sigma():
    """cycle-3: the adopted representation carries paper's Sigma/G/K (not None)."""
    from cases.core.representation import RepresentationInducer

    spec = RepresentationInducer().induce(_boolean_adapter())
    assert spec.entities  # Sigma: entity list non-empty
    assert spec.geometry is not None and spec.geometry.get("backend")  # K
    assert spec.grounding is not None and spec.grounding.get("forward")  # G
    # Runtime consumes geometry: recording appears in the instantiation
    from cases.core.representation import RepresentationRuntime

    reg = RepresentationRuntime().instantiate(spec)
    assert reg.metadata.get("geometry") is not None


def test_fidelity_componentwise_epsilon_default():
    from cases.core.representation import FidelityGate

    gate = FidelityGate(_boolean_adapter())
    assert gate.componentwise_epsilon == (0.5, 0.5, 0.5)


def test_revision_preserves_contract_g_k_sigma():
    """cycle-3 major fix: a revised spec keeps Sigma/G/K (not reverted to None)."""
    from cases.core.representation import RepresentationParadigm
    from cases.core.revision import RepresentationRevisioner

    m = _model(epsilon_f=1.0)
    m.state.metadata["posterior_discriminative"] = False  # forces EMPIRICAL trigger
    spec = m.representation  # populated entities/geometry/grounding
    assert spec.entities and spec.geometry and spec.grounding
    rev = RepresentationRevisioner(_boolean_adapter(), epsilon_f=1.0).decide_and_revise(
        m.state, spec, paradigm=RepresentationParadigm.FULL, epsilon_f=1.0
    )
    assert rev.needed is True
    assert rev.new_spec is not None
    assert rev.new_spec.entities == spec.entities
    assert rev.new_spec.geometry == spec.geometry
    assert rev.new_spec.grounding == spec.grounding


def test_fidelity_cert_exception_is_recorded():
    """cycle-3.1 major #2: a failing certificate is recorded, never swallowed."""
    import warnings

    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    m = _model()
    m.update([
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id="h0", text="x AND y"))
    ])  # give state.objects so the post-admission certificate passes non-empty objects

    real_cert = m.fidelity.certificate

    def _boom(spec, objects=(), **k):
        # only the post-admission certificate block passes objects -> raise there,
        # while admission's score() (no objects) still succeeds.
        if tuple(objects):
            raise ValueError("cert-boom")
        return real_cert(spec, objects=objects, **k)

    m.fidelity.certificate = _boom
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m._admit_reuse(_spec())
    meta = m.state.metadata.get("representation_fidelity_cert")
    assert isinstance(meta, dict) and "error" in meta  # error marker set, not swallowed
    assert any("fidelity certificate unavailable" in str(w.message) for w in caught)


def test_fidelity_cert_round_trip_reported():
    """round_trip_fidelity (Q_roundtrip, plan §2.2) is reported in the cert."""
    from cases.core.model import EvidenceRecord
    from cases.core.representation import FidelityGate
    from cases.core.types import LanguageHypothesis

    adapter = _boolean_adapter()
    gate = FidelityGate(adapter)
    spec = _spec(grounding={"forward": "descriptor", "backward": "target_instruction"})
    objs = []
    from cases.core.types import GroundedObject

    for i, t in enumerate(["x AND y", "x OR y", "NOT x"]):
        objs.append(GroundedObject(object_id=f"o{i}", task_id="bool",
                                   object_type="hypothesis", canonical_form=t,
                                   payload={"text": t}, display_text=t,
                                   source_hypothesis_ids=()))
    cert = gate.certificate(spec, objects=objs)
    assert cert.round_trip is None or (0.0 <= cert.round_trip <= 1.0)
    assert cert.as_tuple()[3] is cert.round_trip

    m = _model(epsilon_f=1.0)
    m.update([
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t))
        for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
    ])
    m.recover()
    stored = m.state.metadata.get("representation_fidelity_cert") or {}
    rt = stored.get("round_trip_fidelity")
    assert rt is None or (0.0 <= rt <= 1.0)


class _FakeReviseLLM:
    """Fake LLM that proposes a revision ADDING a typed variable (structural ΔP)."""
    def generate_json(self, prompt):
        return {
            "typed_variables": [
                {"name": "a", "vtype": "continuous", "role": "state"},
                {"name": "z", "vtype": "continuous", "role": "state"},
            ],
            "relations": [{"name": "r", "polymorphic_type": "", "scope": "objects"}],
            "transforms": [{"name": "e", "arity": "obj->obj"}],
            "semantic_commitments": ["a", "z"],
            "entities": ["bool"],
            "geometry": {"kernel": "rbf", "features": "euclidean"},
            "grounding": {"forward": "descriptor", "backward": "target_instruction"},
        }


def test_llm_revise_structural_expansion():
    """closes residual 3: LLM-Revise may ADD a variable (real ΔP), not prune."""
    from cases.core.representation import RepresentationInducer, RepresentationParadigm

    ind = RepresentationInducer()
    new = ind.revise(_boolean_adapter(), _spec(),
                     llm=_FakeReviseLLM(), paradigm=RepresentationParadigm.FULL)
    assert new is not None and new.source == "llm_revise"
    assert new.spec_id.endswith(".r")
    assert [v.name for v in new.typed_variables] == ["a", "z"]  # added z


def test_revision_uses_llm_inductor_when_given():
    """the revisioner takes the LLM candidate when an inductor is supplied."""
    from cases.core.representation import RepresentationInducer, RepresentationParadigm
    from cases.core.revision import RepresentationRevisioner

    m = _model(epsilon_f=1.0)
    m.state.metadata["posterior_discriminative"] = False  # fires EMPIRICAL
    rev = RepresentationRevisioner(_boolean_adapter(), epsilon_f=1.0)

    def inductor(spec, signals):
        return RepresentationInducer().revise(
            _boolean_adapter(), spec, signals, llm=_FakeReviseLLM(),
            paradigm=RepresentationParadigm.FULL)

    report = rev.decide_and_revise(m.state, m.representation,
                                   paradigm=RepresentationParadigm.FULL,
                                   epsilon_f=1.0, llm_inductor=inductor)
    assert report.needed is True
    assert report.new_spec is not None and report.new_spec.source == "llm_revise"


def test_rejected_llm_revision_keeps_old_contract():
    """an LLM revision with non-exposed commitments is gated OUT; old P kept."""
    from cases.core.representation import RepresentationInducer, RepresentationParadigm

    m = _model(epsilon_f=1.0)
    m.state.metadata["posterior_discriminative"] = False
    old_id = m.representation.spec_id

    def inductor(spec, signals):
        return RepresentationInducer().revise(
            _boolean_adapter(), spec, signals, llm=_FakeReviseLLM(),
            paradigm=RepresentationParadigm.FULL)

    report = m.revise_representation(llm_inductor=inductor)
    assert report.new_spec is not None and report.new_spec.admitted is False  # a/z not exposed
    assert m.representation.spec_id == old_id  # rejected revision does not replace P
