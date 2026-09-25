"""this Core — ReGroundingPass tests.

Covers the missing-data semantics on representation revision:
  - present / unknown attribute classification against ``spec.typed_variables``
  - the ``usable_in_inference`` rule (all-required-present OR missing_data_ok)
  - the unknown branch: a record missing a required attr -> queried + UnknownAttribute
  - the excluded branch: a record that cannot be re-grounded under the contract
  - apply_missing_data_policy across retain / exclude / query

Uses a tiny in-test synthetic adapter (no existing files touched).
"""

from __future__ import annotations

import asyncio

from cases.core.model import EvidenceRecord, CASESModel, LanguageCorrespondence  # noqa: F401
from cases.core.state import SolutionSpaceState
from cases.core.types import (
    GroundedObject,
    LanguageHypothesis,
    RawScientificObject,
    RevisionGraph,
    ScientificGraph,
    VerificationResult,
)
from cases.core.representation import (
    RepresentationSpecification,
    TypedVariable,
)
from cases.core.reground import (
    MISSING_DATA_POLICY,
    ReGroundingPass,
    ReGroundingReport,
    UnknownAttribute,
    apply_missing_data_policy,
)


# ---------------------------------------------------------------------------
# Tiny synthetic adapter
# ---------------------------------------------------------------------------

_CONTRACT_ATTRS = ("alpha", "beta", "gamma")


class _FakeRegroundAdapter:
    """compile -> verify -> canonicalize over plain text, with controllable
    failure (``EXCLUDE`` = ungroundable) and attribute emission (``GAMMA`` adds
    the ``gamma`` attribute)."""

    task_id = "fake_reground"
    object_type = "fake_obj"

    async def compile(self, hypothesis, llm):
        if "EXCLUDE" in hypothesis.text:
            raise ValueError("compiler cannot recover this record under the new contract")
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={"expr": hypothesis.text},
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
        )

    def verify(self, raw):
        return VerificationResult(valid=True)

    def canonicalize(self, raw):
        expr = raw.payload["expr"]
        return GroundedObject(
            object_id="obj_" + expr.replace(" ", "_"),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form="f|" + expr,
            payload={"expr": expr},
            display_text=expr,
            source_hypothesis_ids=(raw.source_hypothesis_id,),
        )

    def descriptor(self, obj):
        attrs = {"alpha": 1, "beta": 2}
        if "GAMMA" in obj.payload["expr"]:
            attrs["gamma"] = 3
        return attrs

    def build_scientific_graph(self, objects):
        return ScientificGraph(node_ids=tuple(o.object_id for o in objects), edges=())

    def build_revision_graph(self, objects):
        return RevisionGraph(node_ids=tuple(o.object_id for o in objects), edges=())

    def render_object(self, obj):
        return obj.display_text


def _state(hypotheses: dict):
    return SolutionSpaceState(
        task_id="fake_reground",
        round_index=0,
        objects={},
        hypotheses=hypotheses,
        scientific_graph=ScientificGraph(node_ids=(), edges=()),
        revision_graph=RevisionGraph(node_ids=(), edges=()),
        observations={},
    )


def _spec(missing_data_ok=False, typed_names=_CONTRACT_ATTRS):
    return RepresentationSpecification(
        spec_id="p1",
        name="re-ground contract",
        description="new contract",
        typed_variables=tuple(
            TypedVariable(n, "continuous", "state") for n in typed_names
        ),
        semantic_commitments=tuple(typed_names),
        provenance={"missing_data_ok": missing_data_ok},
    )


def _hyp(id_, text):
    return LanguageHypothesis(hypothesis_id=id_, text=text)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _mixed_ledger():
    """a: missing gamma; b: all present; c: ungroundable(EXCLUDE)."""
    return {
        "obj_a": _hyp("h_a", "plain a"),
        "obj_b": _hyp("h_b", "GAMMA b"),      # descriptor emits gamma
        "obj_c": _hyp("h_c", "EXCLUDE c"),    # cannot re-ground
    }


def test_unknown_and_excluded_branches():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    report = asyncio.run(pass_obj.re_ground_ledger(_state(_mixed_ledger()), _spec()))

    # a and b re-ground fine; c cannot.
    assert set(report.regrounded) == {"obj_a", "obj_b"}
    assert set(report.excluded) == {"obj_c"}
    # a is missing gamma -> queried (targeted experiment), not usable.
    assert set(report.queried) == {"obj_a"}
    # b has all three attrs present -> usable.
    assert set(report.usable_ids) == {"obj_b"}
    # unknown-branch bookkeeping: obj_a gamma unrecoverable (+ c's required attrs).
    ua = {(u.record_id, u.attr) for u in report.unknown_attrs}
    assert ("obj_a", "gamma") in ua
    assert ("obj_c", "alpha") in ua
    assert all(u.reason == "unrecoverable" for u in report.unknown_attrs)


def test_usable_inference_requires_all_attrs_or_missing_data_ok():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    ledger = _state(_mixed_ledger())

    report = asyncio.run(pass_obj.re_ground_ledger(ledger, _spec(missing_data_ok=False)))
    assert report.usable_in_inference["obj_a"] is False
    assert report.usable_in_inference["obj_b"] is True
    assert report.usable_in_inference["obj_c"] is False

    # same ledger but the contract declares missing-data-valid: a is now usable.
    report_ok = asyncio.run(
        pass_obj.re_ground_ledger(ledger, _spec(missing_data_ok=True))
    )
    assert report_ok.usable_in_inference["obj_a"] is True
    assert set(report_ok.usable_ids) == {"obj_a", "obj_b"}
    assert set(report_ok.queried) == set()


def test_missing_data_policy_branches():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    report = asyncio.run(pass_obj.re_ground_ledger(_state(_mixed_ledger()), _spec()))

    # retain: every re-groundable record stays usable (missing included).
    assert apply_missing_data_policy(report, "retain") == {"obj_a", "obj_b"}
    # exclude: missing-attribute records are dropped from inference.
    assert apply_missing_data_policy(report, "exclude") == {"obj_b"}
    # query: same numeric usable set, but obj_a is flagged for a targeted experiment.
    assert "obj_a" in report.queried
    assert apply_missing_data_policy(report, "query") == {"obj_b"}


def test_invalid_policy_rejected():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    report = asyncio.run(pass_obj.re_ground_ledger(_state(_mixed_ledger()), _spec()))
    import pytest

    with pytest.raises(ValueError):
        apply_missing_data_policy(report, "bogus")


def test_empty_ledger_and_empty_contract():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    # empty ledger -> empty-but-valid report.
    report = asyncio.run(pass_obj.re_ground_ledger(_state({}), _spec()))
    assert report.regrounded == () and report.excluded == () and report.usable_ids == ()
    # a contract with no typed variables: every re-groundable record is usable.
    ledger = _state({"obj_x": _hyp("h_x", "GAMMA x")})
    report2 = asyncio.run(pass_obj.re_ground_ledger(ledger, _spec(typed_names=())))
    assert set(report2.regrounded) == {"obj_x"}
    assert set(report2.usable_ids) == {"obj_x"}


def test_sync_wrapper():
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    report = pass_obj.re_ground_ledger_sync(_state(_mixed_ledger()), _spec())
    assert set(report.regrounded) == {"obj_a", "obj_b"}
    assert set(report.excluded) == {"obj_c"}


def test_constants_and_types():
    assert MISSING_DATA_POLICY == {"retain", "exclude", "query"}
    ua = UnknownAttribute("r", "x")
    assert ua.reason == "unrecoverable"
    assert ReGroundingReport(regrounded=(), excluded=(), queried=(),
                             unknown_attrs=(), usable_ids=()).metadata == {}


def test_does_not_mutate_ledger():
    """Re-grounding must NOT delete records — evidence D_t is invariant."""
    pass_obj = ReGroundingPass(_FakeRegroundAdapter())
    state = _state(_mixed_ledger())
    before = dict(state.hypotheses)
    asyncio.run(pass_obj.re_ground_ledger(state, _spec()))
    assert state.hypotheses == before  # nothing deleted, only reported
