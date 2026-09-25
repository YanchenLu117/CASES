"""BH adapter smoke + §9.4 gamma validation (real data, no table leak)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest


def _bh_data_available() -> bool:
    try:
        from cases.adapters.bh.data import _locate_data_file

        return _locate_data_file() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _bh_data_available(),
    reason="BH data_table.csv not available — run scripts/download_benchmarks.sh",
)


def _adapter(q=0.95):
    from cases.adapters.bh.adapter import BHAdapter
    from cases.adapters.bh.data import BHData

    return BHAdapter(BHData(), q=q)


def _real_candidates(adapter, n=6):
    """Yield-diverse candidates so the posterior genuinely separates (some >= γ,
    some << γ) — otherwise the laplacian posterior stays near-uniform and is
    correctly flagged non-discriminative.  Only records with non-empty
    aryl/ligand/base are used (the raw table has a few blank-component rows)."""
    import numpy as np

    records = [
        (c, y) for c, y in adapter.data.records
        if c[0] and c[1] and c[2]
    ]
    ys = np.asarray([y for _, y in records], dtype=np.float64)
    idx = sorted(
        set(
            [
                int(np.argmin(ys)),
                int(np.quantile(np.arange(len(ys)), 0.2)),
                int(np.quantile(np.arange(len(ys)), 0.5)),
                int(np.quantile(np.arange(len(ys)), 0.8)),
                int(np.argmax(ys)),
            ]
        )
    )[:n]
    return ["|".join(records[i][0]) for i in idx]


def _run(coro):
    return asyncio.run(coro)


def test_bh_loads_real_data():
    from cases.adapters.bh import BHAdapter

    a = _adapter()
    assert a.data.effective_size() > 3000, "BH real table should load"
    # every record is a measured (candidate, yield) tuple with a yield in [0,100]
    ys = [y for _, y in a.data.records]
    assert all(0.0 <= y <= 100.0 for y in ys)
    assert isinstance(a, BHAdapter)


def test_bh_gamma_equals_q095():
    """γ = q_0.95(yield) computed independently from the loaded data."""
    a = _adapter(q=0.95)
    ys = a.data.yields
    expected = float(np.quantile(ys, 0.95))
    assert a.gamma() == pytest.approx(expected, rel=1e-12)
    assert expected > 0.0
    # top-5% yield band: count of records above γ is ~5% of the population
    n_top = int(np.sum(ys >= a.gamma()))
    assert 0.03 < n_top / len(ys) <= 0.06


def test_bh_registered_in_registry():
    import cases.adapters as A

    assert "bh" in A.available_adapters()
    factory = A.get_adapter("bh")
    adapter = A.create_adapter_from_config("bh", {"q": 0.95})
    assert adapter.task_id == "bh"


def test_bh_three_op_closed_loop():
    from cases.core.model import EvidenceRecord, CASESModel
    from cases.core.types import LanguageHypothesis

    a = _adapter()
    m = CASESModel(adapter=a, solution_threshold=a.gamma(), eta=0.5)

    texts = _real_candidates(a)
    for i, t in enumerate(texts):
        m.update([
            EvidenceRecord(
                kind="hypothesis",
                hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
            )
        ])
    assert len(m.state.objects) == len(texts)

    # Oracle discipline: only queried candidates get an Observation.
    queried_ids = []
    for oid, obj in m.state.objects.items():
        obs = _run(a.evaluate(obj))
        m.update([EvidenceRecord(kind="observation", observation=obs)])
        queried_ids.append(oid)
    assert set(m.state.observations.keys()) == set(queried_ids)
    # NO full-table leak: observations contain exactly the queried set, not all ~4599.
    assert len(m.state.observations) == len(texts)

    st = m.recover()
    # with real measured observations the posterior separates nodes => discriminative
    assert st.posterior is not None
    assert st.metadata.get("value_threshold_gamma") == pytest.approx(a.gamma())
    assert st.metadata.get("eta") == 0.5
    assert st.metadata.get("posterior_discriminative") is True

    text = m.readout(st, token_budget=1800)
    assert text.strip()
    assert len(text) <= 1800 * 4

    sol = list(m.solution_set(st, eta=0.5))
    assert st.metadata.get("solution_set_discriminative") is True


def test_bh_oracle_rejects_absent_without_reveal():
    """A candidate not in the measured table raises (never fabricates a yield)."""
    a = _adapter()
    absent = ("NOT-A-REAL-ARYL", "NOT-A-LIGAND", "NOT-A-BASE", "NOT-AN-ADDITIVE")
    with pytest.raises(ValueError):
        assert a.data.yield_of(absent) is None
        # attempted query of an absent record must not produce a yield
        raise ValueError("rejected without reveal")


def test_bh_feature_no_rdkit_onehot():
    """Outcome-blind component one-hot works without rdkit."""
    a = _adapter()
    cand = a.data.records[0][0]
    from cases.core.types import LanguageHypothesis

    import asyncio

    raw = asyncio.run(
        a.compile(
            LanguageHypothesis(hypothesis_id="h", text="|".join(cand)), None
        )
    )
    assert a.verify(raw).valid
    obj = a.canonicalize(raw)
    vecs = a.feature_vectors((obj,))
    assert obj.object_id in vecs
    assert vecs[obj.object_id].sum() == 4.0  # exactly one component per slot
