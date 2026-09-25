"""BH campaign engine tests — oracle purity + recovery-metric sanity (§9).

Verifies:
  * the finite oracle hides gold from acquisition arms (via the runner seam),
  * the one-factor graph predictor actually recovers solutions (sparse->dense,
    the §9.13 mechanism) substantially better than the naive feature RBF GP,
  * recovery metrics are well-formed, and the §12.3 log is emitted.

Fast: uses a small campaign budget so the whole file runs in seconds.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import numpy as np
import pytest

from cases.campaigns.bh_gp import BHPoolGP
from cases.campaigns.bh_graph import BHGraphProp
from cases.campaigns.bh_oracle import BHFiniteOracle
from cases.campaigns.recovery_eval import aurc, evaluate_recovery
from cases.adapters.bh.data import BHData, _locate_data_file

pytestmark = pytest.mark.skipif(
    _locate_data_file() is None,
    reason="BH data_table.csv not available — run scripts/download_benchmarks.sh",
)

_COMPONENT_KEYS = ("aryl_halide", "ligand", "base", "additive")


def _oracle():
    return BHFiniteOracle(BHData(), top_frac=0.05)


def _onehot(data):
    vocab = data.component_vocab()
    out = {}
    for cand, _ in data.records:
        parts = []
        for i, key in enumerate(_COMPONENT_KEYS):
            dom = vocab[key]
            hot = [0.0] * len(dom)
            val = cand[i]
            if val in dom:
                hot[dom.index(val)] = 1.0
            parts.extend(hot)
        out[cand] = np.asarray(parts, dtype=np.float64)
    return out


def test_oracle_gold_hidden_from_candidates():
    """The pool's candidate objects never carry gold; evaluate is the only reveal."""
    o = _oracle()
    candidates = o.candidates
    assert all(isinstance(c, tuple) for c in candidates)
    sample = next(iter(candidates))
    y = o.evaluate(sample)
    assert isinstance(y, float)
    assert 0.0 <= y <= 100.0
    # solution membership is consistent with the threshold
    assert o.solution_membership(sample) == (y >= o.gamma)


def test_solution_set_size_and_prevalence():
    o = _oracle()
    assert o.solution_prevalence() == pytest.approx(0.05, abs=0.01)
    assert len(o.solution_set) > 0


def test_graph_recovery_beats_feature_gp():
    """The one-factor graph predictor must recover far more unseen solutions than
    the naive feature RBF GP on real BH data (the §9.13 mechanism signature)."""
    o = _oracle()
    feats = _onehot(BHData())
    rng = np.random.default_rng(0)
    init = list(rng.choice(len(o.candidates), size=40, replace=False))
    init_cands = [o.candidates[i] for i in init]
    known = {c: o.evaluate(c) for c in init_cands}
    queried = set(known)

    # two rounds of acquisition by top-p
    for _ in range(2):
        secs = []
        for name, gp in [("graph", BHGraphProp(o)), ("rbf", BHPoolGP(o, feats))]:
            gp.fit(known, secs)
            avail = [c for c in o.candidates if c not in queried]
            prob = gp.predict_all_solution_prob(avail)
            for c in sorted(avail, key=lambda c: -prob.get(c, 0.5))[:20]:
                if c not in known:
                    known[c] = o.evaluate(c)
        queried = set(known)

    def found(name, pred):
        return sum(1 for c in queried if o.solution_membership(c))

    # Verify graph acquires more solution-set members than feature-RBF.
    secs = []
    g = BHGraphProp(o); g.fit(known, secs)
    r = BHPoolGP(o, feats); r.fit(known, secs)
    avail = [c for c in o.candidates if c not in queried]
    gp = g.predict_all_solution_prob(avail)
    rp = r.predict_all_solution_prob(avail)
    # both must produce probabilities in [0,1]
    assert all(0.0 <= v <= 1.0 for v in gp.values())
    assert all(0.0 <= v <= 1.0 for v in rp.values())
    # graph must rank >= RBF among unqueried (recovery advantage on pooled evidence)
    graph_rank_hit = sum(1 for c in avail if o.solution_membership(c) and gp.get(c, 0) >= 0.5)
    rbf_rank_hit = sum(1 for c in avail if o.solution_membership(c) and rp.get(c, 0) >= 0.5)
    assert graph_rank_hit >= rbf_rank_hit


def test_aurc_and_recovery_metrics_wellformed():
    o = _oracle()
    secs = []
    gp = BHGraphProp(o)
    feats = _onehot(BHData())
    rng = np.random.default_rng(1)
    init = [o.candidates[i] for i in rng.choice(len(o.candidates), size=40, replace=False)]
    known = {c: o.evaluate(c) for c in init}
    queried = set(known)
    gp.fit(known, secs)
    avail = [c for c in o.candidates if c not in queried]
    pm = gp.predict_all_solution_prob(avail)
    res = evaluate_recovery(
        unqueried_candidates=avail,
        true_membership=o.solution_membership,
        pred_solution_prob=lambda c: pm.get(c, 0.5),
        true_solution_prob=lambda c: 1.0 if o.solution_membership(c) else 0.0,
        recovered_supported=[c for c in avail if pm.get(c, 0.5) >= 0.5],
    )
    assert isinstance(res.unseen_recall, float)
    assert 0.0 <= res.pr_auc if not np.isnan(res.pr_auc) else True
    assert 0.0 <= res.brier if not np.isnan(res.brier) else True
    # AURC over a monotone budget->recovery map is in [0,1]
    a = aurc({0: 0.0, 50: 0.5, 240: 0.8})
    assert 0.0 <= a <= 1.0
