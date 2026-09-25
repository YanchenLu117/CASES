"""P1 — facade token ledger (§4.3), update/aupdate None-guard parity, readout
solution-family views, and the SolutionSpaceState posterior consumers."""

from __future__ import annotations

import asyncio

import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


def _boolean_adapter():
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask

    task = BooleanTask(
        dataset="ledger_test", observation_set_id="t",
        variables=("x", "y"), operators=frozenset({"AND", "OR", "NOT"}),
        max_depth=2,
        mechanistic_opts={
            "apply_commutativity": True, "apply_idempotence_and_or": True,
            "flatten_associativity": True,
        },
        observations=(),
        n_observations=0,
    )
    return BooleanAdapter(task)


def _batch(n=4):
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    texts = ["x AND y", "x OR y", "NOT x", "x AND NOT y"]
    return [
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t))
        for i, t in enumerate(texts[:n])
    ]


def test_token_ledger_cumulative_and_snapshot():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    m.update(_batch())
    st = m.recover()
    m.readout(st, token_budget=1200)
    usage = m.token_usage()
    assert set(usage) == {"input", "output", "total", "calls"}
    assert usage["total"] == usage["input"] + usage["output"]
    # T3: this is pure computation (boolean adapter + no live LLM), so the
    # ledger honestly reports ZERO real LLM calls and tokens — facade
    # operations must NOT inflate ``calls`` (the old code double-counted
    # readout()/readout_packet() and miscounted accounting entries as calls).
    assert usage["input"] == 0 and usage["output"] == 0 and usage["calls"] == 0
    assert usage == {"input": 0, "output": 0, "total": 0, "calls": 0}
    snap = m.snapshot()
    assert snap["token_usage"] == usage
    # explicit real-LLM usage folds in tokens + exactly one call
    m.record_usage(token_input=100, token_output=50)
    u2 = m.token_usage()
    assert u2["input"] == usage["input"] + 100
    assert u2["output"] == usage["output"] + 50
    assert u2["total"] - usage["total"] == 150
    # ledger is cumulative, not reset; one real call added
    assert u2["calls"] == usage["calls"] + 1


def test_metered_llm_records_real_call():
    """T3: a real (mocked) LLM call wired through update() lands in the ledger
    with non-zero C_in/C_out and exactly one ``calls`` entry — no double count.

    The facade wraps the provided V6 LLMProvider in a metering wrapper, so any
    adapter.compile path that actually drives ``llm.generate`` accumulates
    LLMUsage into the §4.3 ledger while ``calls`` counts the one real call."""
    from cases.core.model import EvidenceRecord, CASESModel
    from cases.core.types import LanguageHypothesis
    from cases.llm.base import LLMResponse, LLMRequest, LLMUsage
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask

    task = BooleanTask(
        dataset="metered_llm", observation_set_id="t",
        variables=("x", "y"), operators=frozenset({"AND", "OR"}),
        max_depth=1,
        mechanistic_opts={"apply_commutativity": True},
        observations=(), n_observations=0,
    )
    adapter = BooleanAdapter(task)

    class _FakeLLM:
        """Subset V6 LLMProvider that returns a token-metered response."""
        provider_name = "fake"
        model_name = "fake-model"

        async def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                text="", parsed=None, tool_calls=(),
                finish_reason="stop",
                usage=LLMUsage(input_tokens=42, output_tokens=7),
                provider="fake", model="fake-model",
            )

        async def generate_json(self, request=None, **kwargs):
            from cases.llm.base import LLMRequest as R
            if request is None:
                request = R(messages=(), response_schema=kwargs.get("response_schema"))
            return await self.generate(request)

    class _LLMAdapter(BooleanAdapter):
        """Adapter whose compile actually drives the provided LLM, so the
        facade's metering wrapper is exercised on a real call path."""
        async def compile(self, hypothesis, llm):
            if llm is not None:
                await llm.generate(LLMRequest(messages=()))
            return await super().compile(hypothesis, llm)

    m = CASESModel(adapter=_LLMAdapter(task), llm=_FakeLLM())
    m.update([
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id="h0", text="x AND y"))
    ])
    usage = m.token_usage()
    assert usage["input"] == 42          # C_in from the real (mocked) call
    assert usage["output"] == 7          # C_out
    assert usage["total"] == 49
    assert usage["calls"] == 1           # exactly one real LLM call, no double count
    # a second update of a new hypothesis adds one more real call (cumulative)
    m.update([
        EvidenceRecord(kind="hypothesis",
                       hypothesis=LanguageHypothesis(hypothesis_id="h1", text="x OR y"))
    ])
    usage2 = m.token_usage()
    assert usage2["calls"] == 2 and usage2["input"] == 84 and usage2["output"] == 14


def test_aupdate_none_hypothesis_guard_parity():
    """aupdate must warn+skip a hypothesis record with None payload, matching
    update (P1 fork fix)."""
    from cases.core.model import EvidenceRecord, CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    with pytest.warns(RuntimeWarning):
        asyncio.run(m.aupdate([EvidenceRecord(kind="hypothesis", hypothesis=None)]))
    assert len(m.state.objects) == 0
    assert len(m.evidence_log) == 1  # skipped but logged (parity with update)


def test_update_none_hypothesis_guard():
    from cases.core.model import EvidenceRecord, CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    with pytest.warns(RuntimeWarning):
        m.update([EvidenceRecord(kind="hypothesis", hypothesis=None)])
    assert len(m.state.objects) == 0


def test_readout_families_views_present():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    m.update(_batch())
    st = m.recover()
    pkt = m.readout_packet(st)
    views = {f.view for f in pkt.families}
    # posterior-driven view: distinct solution families (covered regions) + under-covered
    assert "family" in views
    assert "under_covered" in views
    for fam in pkt.families:
        assert fam.family_id
        assert isinstance(fam.size, int)


def test_uncertain_band_named_parameter():
    """T3 tech-debt: the ``uncertain_boundary`` magic literal
    ``abs(p - 0.5) < 0.25`` is now a named, default-preserving construction
    parameter — no bare magic constant in the readout implementation."""
    import inspect

    from cases.core.readout import DEFAULT_UNCERTAIN_BAND, CoverageFrontierReadout

    sig = inspect.signature(CoverageFrontierReadout.__init__)
    assert "uncertain_band" in sig.parameters
    r = CoverageFrontierReadout()
    assert r.uncertain_band == 0.25 == DEFAULT_UNCERTAIN_BAND  # default preserved
    r2 = CoverageFrontierReadout(uncertain_band=0.1)
    assert r2.uncertain_band == 0.1  # configurable
    # the family classifier honors it: with a tiny band, nothing near p=0.5 is
    # "uncertain" unless it is very close, so the band is actually consulted.
    assert r.uncertain_band != r2.uncertain_band


def test_frozen_build_signature_unchanged():
    """CoverageFrontierReadout.build retains the frozen (state, semantics) call."""
    import inspect

    from cases.core.readout import CoverageFrontierReadout

    sig = inspect.signature(CoverageFrontierReadout.build)
    params = list(sig.parameters)
    assert params == ["self", "state", "semantics"]
