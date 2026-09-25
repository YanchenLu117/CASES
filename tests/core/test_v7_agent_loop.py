"""this Core — CASESAgentLoop (Algorithm 1) tests.

Exercises the orchestrator over the real BooleanAdapter facade with an injected
fake LLM proposer:

  - a full ``run_round`` under a non-OFF gear (HIGH) reaching all four channels
    SOL / BND / COV / OPEN
  - default implementer + injected synthetic evaluator growing the evidence
  - a revision round when a trigger (FIDELITY / UNGROUNDABLE) fires, then
    re-admission
  - the keep-old-spec branch when the revised spec is NOT admitted

Async methods are driven with ``asyncio.run`` (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio
import random

from cases.core.model import EvidenceRecord, CASESModel
from cases.core.types import LanguageHypothesis, Observation
from cases.core.acquisition import ExplorationGear, OPEN, SOL, BND, COV


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _boolean_model(exploration=ExplorationGear.HIGH, **kw):
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
    return CASESModel(adapter=BooleanAdapter(task), exploration=exploration, **kw)


def _seed_objects(m, texts):
    m.update(
        [
            EvidenceRecord(
                kind="hypothesis",
                hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
            )
            for i, t in enumerate(texts)
        ]
    )
    m.recover()


class _FakeProposer:
    """Returns hypotheses drawn from a pool, cycling per channel, and records
    every channel it was asked to propose for."""

    def __init__(self, pool):
        self.pool = list(pool)
        self.calls = []
        self._i = 0

    async def __call__(self, channel, readout_text, task_prompt, n):
        self.calls.append(channel)
        hyps = []
        for _ in range(max(n, 0)):
            text = self.pool[self._i % len(self.pool)]
            self._i += 1
            hyps.append(
                LanguageHypothesis(
                    hypothesis_id=f"h_{channel}_{self._i}",
                    text=text,
                    proposer="fake",
                )
            )
        return hyps


def _synthetic_evaluator(m):
    async def _ev(obj):
        value = float(len(obj.canonical_form) % 7)
        m.update(
            [
                EvidenceRecord(
                    kind="observation",
                    observation=Observation(
                        object_id=obj.object_id,
                        value=value,
                        round_index=m.state.round_index,
                        evaluator="fake_synthetic",
                    ),
                )
            ]
        )
        return obj

    return _ev


# ---------------------------------------------------------------------------
# full run_round under a non-OFF gear reaching all channels
# ---------------------------------------------------------------------------


def test_run_round_high_gear_full_integration():
    m = _boolean_model(exploration=ExplorationGear.HIGH)
    _seed_objects(m, ["x AND y", "x OR y"])
    pool = ["x AND y", "x OR y", "NOT x", "x AND NOT y", "NOT y", "x OR NOT y"]
    proposer = _FakeProposer(pool)
    from cases.core.agent import CASESAgentLoop

    loop = CASESAgentLoop(
        m,
        proposer,
        evaluator=_synthetic_evaluator(m),
        rng=random.Random(0),
    )
    summary = asyncio.run(loop.run_round("find a boolean rule", batch_size=10))

    assert summary["round"] == 0
    # HIGH gear + batch_size=10 gives positive counts on every channel.
    used = set(summary["channels_used"])
    assert {SOL, BND, COV, OPEN} <= used
    # the proposer was asked on the OPEN channel with the open directive.
    assert OPEN in proposer.calls
    # implementer grounded new hypotheses into the space.
    assert summary["n_new_objects"] >= 1
    # synthetic evaluator produced observations for every grounded object.
    assert summary["n_new_observations"] >= 1
    assert summary["evidence_count"] > 0
    # stable schema
    for key in (
        "round", "n_new_objects", "n_new_observations", "channels_used",
        "triggers", "representation_admitted", "field_rebuilt", "evidence_count",
    ):
        assert key in summary
    # evidence ledger actually grew for hypotheses + observations.
    n_hyp = sum(1 for r in m.evidence_log if r.kind == "hypothesis")
    n_obs = sum(1 for r in m.evidence_log if r.kind == "observation")
    assert n_hyp >= 1
    assert n_obs >= 1


def test_run_round_empty_space_uses_open_only():
    m = _boolean_model(exploration=ExplorationGear.HIGH)
    # no seeded objects -> SOL/BND/COV have no in-space targets, only OPEN runs.
    proposer = _FakeProposer(["NOT x"])
    from cases.core.agent import CASESAgentLoop

    loop = CASESAgentLoop(m, proposer, evaluator=_synthetic_evaluator(m))
    summary = asyncio.run(loop.run_round("find a rule", batch_size=10))
    assert summary["channels_used"] == [OPEN]
    assert summary["round"] == 0
    assert summary["n_new_observations"] >= 1


# ---------------------------------------------------------------------------
# revision round
# ---------------------------------------------------------------------------


def _low_fidelity_spec():
    from cases.core.representation import (
        RelationDecl,
        RepresentationSpecification,
        TransformDecl,
        TypedVariable,
    )

    return RepresentationSpecification(
        spec_id="s_lowfid",
        name="low-fid",
        description="low fidelity",
        typed_variables=(TypedVariable("v", "continuous", "state"),),
        relations=(RelationDecl("rel", "obj-obj", "objects"),),
        transforms=(TransformDecl("edit", "obj->obj", True),),
        semantic_commitments=("not_a_real_dim", "depth"),  # F=0.5 < 1.0
    )


def test_run_round_revises_on_trigger_and_readmits():
    m = _boolean_model(exploration=ExplorationGear.MEDIUM, epsilon_f=1.0)
    _seed_objects(m, ["x AND y", "x OR y", "NOT x"])
    # force FIDELITY: a low-fidelity current spec.
    m._representation = _low_fidelity_spec()
    m._representation_resolved = True

    proposer = _FakeProposer(["x AND NOT y", "NOT y"])
    from cases.core.agent import CASESAgentLoop

    loop = CASESAgentLoop(
        m,
        proposer,
        evaluator=_synthetic_evaluator(m),
        rng=random.Random(1),
    )
    summary = asyncio.run(loop.run_round("find a rule", batch_size=8))

    assert summary["revision_fired"] is True
    assert "fidelity" in summary["triggers"]
    # the tightened revised spec is re-admitted, restoring fidelity.
    assert summary["representation_admitted"] is True
    assert m.fidelity_score() == 1.0
    assert summary["field_rebuilt"] is True or m.state.metadata.get(
        "representation_field_rebuilt"
    ) in (True, None)

    log = loop.revision_log()
    assert any("revision_admitted" in e for e in log)


def test_run_round_keeps_old_spec_when_revision_not_admitted():
    from cases.core.representation import (
        RelationDecl,
        RepresentationSpecification,
        TransformDecl,
        TypedVariable,
    )

    m = _boolean_model(exploration=ExplorationGear.MEDIUM, epsilon_f=0.5)
    _seed_objects(m, ["x AND y"])
    # A spec whose commitments are entirely unexposed -> its tightened revision
    # still carries commitments with F=0 < epsilon_f -> NOT admitted.
    stale = RepresentationSpecification(
        spec_id="s_stale",
        name="stale",
        description="stale",
        typed_variables=(TypedVariable("v", "continuous", "state"),),
        relations=(RelationDecl("rel", "obj-obj", "objects"),),
        transforms=(TransformDecl("edit", "obj->obj", True),),
        semantic_commitments=("structural_domain",),  # not exposed by the adapter
    )
    m._representation = stale
    m._representation_resolved = True
    # force a trigger so a revision is attempted.
    m.state.metadata["ungroundable_hypotheses"] = 1

    proposer = _FakeProposer(["NOT x"])
    from cases.core.agent import CASESAgentLoop

    loop = CASESAgentLoop(m, proposer, evaluator=_synthetic_evaluator(m))
    summary = asyncio.run(loop.run_round("find a rule", batch_size=4))

    assert summary["revision_fired"] is True
    assert summary["representation_admitted"] is False
    assert summary["field_rebuilt"] is False
    # the old spec is kept, not the rejected revision.
    assert m.representation.spec_id == "s_stale"
    log = loop.revision_log()
    assert any(e.get("revision_not_admitted") is True for e in log)


def test_import_surface_agent():
    from cases.core.agent import (
        CASESAgentLoop,
        default_evaluator,
        default_implementer,
    )

    assert callable(CASESAgentLoop)
    assert callable(default_implementer)
    assert callable(default_evaluator)
    assert OPEN in ("SOL", "BND", "COV", "OPEN")
