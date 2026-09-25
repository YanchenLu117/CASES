"""GB1 campaign engine tests (§10).

Verifies:
  * the better-than-WT oracle (|S*|=3,643, prevalence 0.024, WT excluded),
  * the FROZEN Hamming-1 region geometry over S*_WT (components [3640, 2, 1]),
  * the Hamming-1 propagation mechanism recovers more S* members than uniform
    sampling (the §10.12 cross-domain recovery signature).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import numpy as np
import pytest

from cases.campaigns.gb1_oracle import GB1FiniteOracle
from cases.campaigns.gb1_predict import GB1HammingProp
from cases.adapters.gb1.data import WILD_TYPE


def _oracle():
    return GB1FiniteOracle(criterion="better_wt")


def test_oracle_wt_boundary_and_size():
    o = _oracle()
    assert o.gamma == pytest.approx(1.0)
    assert WILD_TYPE not in o.solution_set            # WT (fitness 1.0) excluded
    assert o.solution_membership(WILD_TYPE) is False
    # full measured table -> 3,643 better-than-WT variants (committee-verified)
    if o.data.is_full_measured():
        assert len(o.solution_set) == 3643
        assert o.solution_prevalence() == pytest.approx(3643 / 149361, abs=1e-6)


def test_hamming1_region_components():
    o = _oracle()
    h = GB1HammingProp(o)
    # frozen geometry: connected components of S*_WT under Hamming-1
    sol_idx = {h.index[v] for v in o.solution_set}
    nab = {i: set(h._neighbors.get(i, ())) for i in sol_idx}
    visited: set[int] = set()
    sizes = []
    for start in sol_idx:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for w in nab[u]:
                if w in sol_idx and w not in visited:
                    visited.add(w)
                    stack.append(w)
        sizes.append(size)
    sizes.sort(reverse=True)
    if o.data.is_full_measured():
        assert sizes == [3640, 2, 1]   # §10.4 committee-verified structure
    else:
        assert len(sizes) >= 1


@pytest.mark.skipif(
    not _oracle().data.is_full_measured(),
    reason="full GB1 measured table not available — run scripts/download_benchmarks.sh",
)
def test_hamming_recovery_beats_random():
    """After 96+2x96 queries, Hamming-1 propagation recovers far more S* members
    than the random-query expectation (mechanism signature)."""
    o = _oracle()
    rng = np.random.default_rng(0)
    pool = list(o.candidates)
    init = rng.choice(len(pool), size=96, replace=False)
    known = {pool[i]: o.evaluate(pool[i]) for i in init}
    queried = set(known)
    h = GB1HammingProp(o)
    for _ in range(2):
        secs = []
        h.fit(known, secs)
        avail = [c for c in pool if c not in queried]
        avail_idx = [h.index[c] for c in avail]
        probs = h.predict_probs_by_index(avail_idx)
        for i in np.argsort(-probs)[:96]:
            c = avail[i]
            if c not in known:
                known[c] = o.evaluate(c)
        queried = set(known)
    found = sum(1 for c in queried if o.solution_membership(c))
    random_exp = 288 * o.solution_prevalence()
    assert found > 3 * random_exp   # must clearly beat random expectation
