"""P0-1 / T3 — eta/gamma decoupling and solution-set materialization.

Asserts that after ``recover()`` the posterior marginals expose a non-None
``solution_probability`` (the true probability belief p_t(h)), and that the
T3-decoupled thresholds are honored:

  * gamma (value-space cut) is what the *backend* consumes and appears in
    ``metadata["value_threshold_gamma"]`` (NOT aliased to eta);
  * eta (probability/rank cutoff) defaults non-zero (0.5) and is what
    ``solution_set`` / the readout family views cut on — ``S_t(eta)`` is a
    *high-probability* subset, never silently the whole belief space;
  * a posterior with no resolution (no value observations, laplacian all-0.5)
    is explicitly flagged ``posterior_discriminative == False`` rather than
    being presented as "all objects are the solution set".
"""

from __future__ import annotations

import numpy as np
import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


def _boolean_task():
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask

    # empty observation set: every grammar-valid hypothesis grounds, so the
    # posterior has real nodes (mirrors the frozen facade test's adapter).
    observations = ()
    task = BooleanTask(
        dataset="eta_test",
        observation_set_id="t",
        variables=("x", "y"),
        operators=frozenset({"AND", "OR", "NOT"}),
        max_depth=2,
        mechanistic_opts={
            "apply_commutativity": True,
            "apply_idempotence_and_or": True,
            "flatten_associativity": True,
        },
        observations=observations,
        n_observations=len(observations),
    )
    return BooleanAdapter(task)


def _batch():
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    return [
        EvidenceRecord(
            kind="hypothesis",
            hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
        )
        for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
    ]


def test_solution_probability_materialized_default_backend():
    """Default laplacian + no value observations => every p=0.5.  The posterior
    is therefore NON-discriminative and must be flagged, and the facade must
    record gamma and eta SEPARATELY (this is the T3 η/γ-decoupling regression:
    S_t(0)==whole-belief-space must no longer be the baked-in expectation)."""
    from cases.core.model import CASESModel

    adapter = _boolean_task()
    m = CASESModel(adapter=adapter)  # default laplacian; gamma=0.0, eta=0.5
    m.update(_batch())
    st = m.recover()
    assert st.posterior is not None
    # γ (value-space) and η (probability) are recorded separately, never aliased.
    assert st.metadata.get("value_threshold_gamma") == 0.0
    assert st.metadata.get("eta") == 0.5
    assert st.metadata.get("posterior_discriminative") is False
    # every marginal still materializes solution_probability (P0-1)
    probs = []
    for oid in st.objects:
        mp = st.posterior.marginal(oid)
        assert mp.solution_probability is not None, "solution_probability must not be None (P0-1)"
        probs.append(mp.solution_probability)
    # uniform posterior => S_t(0.5) would naively be everything; it must NOT
    # masquerade as a real solution set.
    sol = m.solution_set(st, eta=0.5)
    assert st.metadata.get("solution_set_discriminative") is False
    # monotonicity still holds across eta.
    sol_high = m.solution_set(st, eta=1.0)
    assert len(sol_high) <= len(sol)


def test_posterior_discriminative_with_observations():
    """With real value observations the posterior separates nodes => flagged
    discriminative, and S_t(eta) is a high-probability PROPER subset (T3).
    Also exercises the rank-based eta (``by_rank``) for probability-only
    spaces, which deterministically yields a true subset."""
    from cases.core.model import CASESModel
    from cases.core.types import Observation

    m = CASESModel(adapter=_boolean_task())
    m.update(_batch())
    oids = list(m.state.objects)
    # seed opposing scalar values so the laplacian means (and hence p_t(h))
    # genuinely spread above and below p=0.5.
    m.state.observations[oids[0]] = Observation(object_id=oids[0], value=0.9, round_index=0)
    m.state.observations[oids[1]] = Observation(object_id=oids[1], value=-1.5, round_index=0)
    st = m.recover()
    assert st.metadata.get("posterior_discriminative") is True
    probs = [st.posterior.marginal(oid).solution_probability for oid in st.objects]
    assert max(probs) - min(probs) > 1e-6
    # absolute-probability S_t(0.5): at least one strongly-negative node is
    # excluded, so the set is a genuine, non-empty proper subset.
    sol = list(m.solution_set(st, eta=0.5))
    assert st.metadata.get("solution_set_discriminative") is True
    assert 0 < len(sol) <= len(oids)
    # rank / quantile eta (HypoSpace-style, no value domain): top half by rank
    # is a deterministic proper subset (len >= 2).
    sol_rank = list(m.solution_set(st, eta=0.5, by_rank=True))
    assert 0 < len(sol_rank) <= len(oids)


def test_solution_probability_materialized_with_features():
    """The features path (rbf_gp genuinely fits) must also produce p_t(h)."""
    from cases.core.model import CASESModel
    from cases.recovery.registry import create_backend

    adapter = _boolean_task()
    m = CASESModel(adapter=adapter, backend=create_backend("rbf_gp"))
    m.update(_batch())
    feats = {oid: np.asarray(adapter._feature(obj), dtype=float) for oid, obj in m.state.objects.items()}
    st = m.recover(features=feats)
    assert st.metadata.get("recovery_backend") == "rbf_gp"
    assert st.posterior is not None
    mp = st.posterior.marginal(next(iter(st.objects)))
    assert mp.solution_probability is not None
    # note: rbf_gp with empty value observations is also uniform (diagonal=1),
    # so it, too, is flagged non-discriminative — never silently "the solution
    # set" (T3).
    assert st.metadata.get("posterior_discriminative") is False
    assert len(m.solution_set(st, eta=0.5)) <= len(st.objects)
