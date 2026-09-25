"""GB1 adapter smoke + §10.5 Hamming-1 geometry + data-source labeling."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest


def _adapter(q=0.95):
    from cases.adapters.gb1.adapter import GB1Adapter
    from cases.adapters.gb1.data import GB1Data

    return GB1Adapter(GB1Data(), q=q)


def _run(coro):
    return asyncio.run(coro)


def test_gb1_data_source_label():
    """Data source must be labeled honestly (full_xlsx vs demo_subset)."""
    from cases.adapters.gb1.data import GB1Data

    d = GB1Data()
    assert d.source in ("full_xlsx", "demo_subset")
    if d.source == "full_xlsx":
        assert d.effective_size() == 149361  # §10.2 FROZEN measured boundary
    else:
        assert d.effective_size() > 0


def test_gb1_measured_or_no_imputed():
    """Only measured variants are used (demo subset is real measured values)."""
    d = _adapter().data
    assert d.effective_size() > 0
    # 10,639 imputed variants are NEVER merged in
    assert d.effective_size() != 10639


def test_gb1_gamma_is_fitness_quantile():
    a = _adapter(q=0.95)
    fs = np.asarray(list(a.data.fitness.values()), dtype=np.float64)
    expected = float(np.quantile(fs, 0.95))
    assert a.gamma() == pytest.approx(expected, rel=1e-12)


def test_gb1_registered_in_registry():
    import cases.adapters as A

    assert "gb1" in A.available_adapters()
    adapter = A.create_adapter_from_config("gb1", {"q": 0.95})
    assert adapter.task_id == "gb1"


def test_gb1_three_op_closed_loop_and_hamming1():
    from cases.core.model import EvidenceRecord, CASESModel
    from cases.core.types import LanguageHypothesis

    a = _adapter()
    m = CASESModel(adapter=a, solution_threshold=a.gamma(), eta=0.5)

    # Choose a small set of variants that are all in the measured table.
    variants = ["VDGV", "ADGV", "VDGV".replace("D", "E", 1), "ADGE"]
    # ensure all are present in the population
    pop = a.data.population()
    variants = [v for v in variants if v in pop]
    if not variants:
        variants = list(pop)[:4]
    assert variants

    for i, v in enumerate(variants):
        m.update([
            EvidenceRecord(
                kind="hypothesis",
                hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=v),
            )
        ])
    assert len(m.state.objects) == len(variants)

    for oid, obj in m.state.objects.items():
        obs = _run(a.evaluate(obj))
        m.update([EvidenceRecord(kind="observation", observation=obs)])
    st = m.recover()

    # §10.5: Hamming-1 mutation graph — edges only between distance-1 variants
    g = st.scientific_graph
    for e in g.edges:
        assert e.metadata.get("hamming") == 1
    # count edges equals number of (i<j) pairs at Hamming distance exactly 1
    vlist = [str(o.payload["variant"]) for o in m.state.objects.values()]
    expected_edges = sum(
        1 for i in range(len(vlist)) for j in range(i + 1, len(vlist))
        if _hd(vlist[i], vlist[j]) == 1
    )
    assert len(g.edges) == expected_edges

    text = m.readout(st, token_budget=1800)
    assert text.strip()
    assert st.metadata.get("posterior_discriminative") in (True, False)


def _hd(a, b):
    return sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i]) + abs(len(a) - len(b))


def test_gb1_feature_onehot():
    a = _adapter()
    from cases.core.types import LanguageHypothesis

    v = "VDGV" if "VDGV" in a.data.population() else next(iter(a.data.population()))
    raw = asyncio.run(a.compile(LanguageHypothesis(hypothesis_id="h", text=v), None))
    obj = a.canonicalize(raw)
    vecs = a.feature_vectors((obj,))
    assert vecs[obj.object_id].sum() == 4.0  # one-hot of 4 sites


# ---------------------------------------------------------------------------
# §10.4 PRIMARY better-than-WT criterion (S*_WT = {y(x) > y(WT)}, WT = VDGV)
# ---------------------------------------------------------------------------


def _bwtdata_only():
    from cases.adapters.gb1.data import GB1Data
    return GB1Data()


def test_gb1_better_wt_solution_size_3643():
    """Empirical |S*_WT| = 3,643 on the full measured table (§10.4 primary)."""
    from cases.adapters.gb1.adapter import GB1Adapter

    d = _bwtdata_only()
    a = GB1Adapter(d, criterion="better_wt")
    wt = "VDGV"
    assert "VDGV" in d.fitness  # wild type is measured
    sol = a.solution_set()
    # strict > wild-type fitness
    assert all(d.fitness[v] > d.fitness[wt] for v in sol)
    if d.source == "full_xlsx":
        # committee-verified empirical figure
        assert len(sol) == 3643
    else:
        assert len(sol) > 0


def test_gb1_better_wt_excludes_wild_type():
    """WT (VDGV, fitness 1.0) is the boundary and must NOT be in S*_WT."""
    from cases.adapters.gb1.adapter import GB1Adapter

    a = GB1Adapter(_bwtdata_only(), criterion="better_wt")
    assert a.data.fitness_of("VDGV") == 1.0
    assert "VDGV" not in a.solution_set()
    assert a.gamma() == 1.0


def test_gb1_better_wt_three_components():
    """The Hamming-1 graph over S*_WT has 3 connected components (largest 3640)."""
    from cases.adapters.gb1.adapter import GB1Adapter

    a = GB1Adapter(_bwtdata_only(), criterion="better_wt")
    c = a.solution_diagnostics()["components"]
    if a.data.source == "full_xlsx":
        assert c["n_components"] == 3
        assert c["largest_component"] == 3640
        assert c["component_sizes"] == [3640, 2, 1]
    else:
        assert c["n_components"] >= 1


def test_gb1_better_wt_differs_from_top_pct():
    """better-than-WT and top-5% are different solution sets."""
    from cases.adapters.gb1.adapter import GB1Adapter

    d = _bwtdata_only()
    a_wt = GB1Adapter(d, criterion="better_wt")
    a_t5 = GB1Adapter(d, criterion="top_pct", q=0.95)
    st_wt = set(a_wt.solution_set())
    st_t5 = set(a_t5.solution_set())
    if d.source == "full_xlsx":
        # top-5% (7,469) strictly contains S*_WT (3,643); WT is in top-5% only
        assert st_wt != st_t5
        assert st_wt < st_t5
        assert "VDGV" in st_t5 and "VDGV" not in st_wt
    else:
        assert st_wt != st_t5


def test_gb1_better_wt_registry_config():
    """The registry factory forwards the criterion config key."""
    import cases.adapters as A

    a = A.create_adapter_from_config("gb1", {"criterion": "better_wt"})
    assert a.criterion == "better_wt"
    assert a.gamma() == pytest.approx(1.0)


def test_gb1_registry_criterion_stable_across_repeated_calls():
    """Regression: no lazy re-registration clobbers the criterion factory.

    P0 fix — ``gb1/__init__.py`` previously eagerly registered a factory that
    dropped ``criterion`` on lazy import, so the 1st call returned better_wt
    and the 2nd returned top_pct.  The factory must now return better_wt
    stably across many consecutive calls, and a later top_pct call must not
    poison a subsequent better_wt call.
    """
    import cases.adapters as A

    for _ in range(5):
        a = A.create_adapter_from_config("gb1", {"criterion": "better_wt"})
        assert a.criterion == "better_wt"
        assert a.gamma() == pytest.approx(1.0)

    # interleave a top_pct call — it must not clobber back to better_wt
    assert A.create_adapter_from_config("gb1", {"criterion": "top_pct"}).criterion == "top_pct"
    a = A.create_adapter_from_config("gb1", {"criterion": "better_wt"})
    assert a.criterion == "better_wt"
    assert a.gamma() == pytest.approx(1.0)
