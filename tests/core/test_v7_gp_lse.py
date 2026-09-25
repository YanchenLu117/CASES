"""this baseline tests: GP-LSE sequential level-set classification (§5)."""

import numpy as np

from cases.baselines import GPLSE


def test_gp_lse_selects_informative():
    # 1D domain; seed two points at the far ends; truth crosses threshold at 0.
    dom = np.linspace(-2, 2, 101)
    truth = dom  # f(x)=x, threshold 0 -> level set x>=0
    gp = GPLSE(threshold=0.0, beta=2.0, length_scale=0.5)
    gp.update([[-2.0], [2.0]], [truth[0], truth[-1]])
    ids = [f"p{i}" for i in range(len(dom))]
    dec = gp.select(dom.tolist(), ids=ids, k=5)
    assert len(dec.ids) == 5  # batch of 5
    # every queried point is informative (its GP CI crosses the threshold)
    assert all(s > 0.0 for s in dec.scores)


def test_gp_lse_classify_level_set():
    gp = GPLSE(threshold=0.0, length_scale=0.5)
    gp.update([[-1.0], [1.0]], [-1.0, 1.0])
    mean, std, cls = gp.predict([[-1.0], [0.0], [1.0]])
    assert cls[0] == 0.0  # clearly below threshold (in-sample)
    assert cls[2] == 1.0  # clearly above threshold (in-sample)
    assert cls[1] == 0.5  # uncertain/boundary region


def test_gp_lse_query_accounting():
    gp = GPLSE(threshold=0.0, length_scale=0.5)
    dom = np.linspace(-1, 1, 21)
    gp.update([[-1.0], [1.0]], [-1.0, 1.0])
    gp.select(dom.tolist(), ids=[f"p{i}" for i in range(21)], k=2)
    assert gp.n_queries == 2
    assert gp.info()["name"].startswith("GP-LSE")
