"""this Core exploration-dial tests (acquisition layer, v0.1 gear + channels).

Covers:
  - five gears, monotonic exploration/OPEN mass, OFF = pure exploit
  - select_channel categorical sampling respects the gear distribution
  - within-space channel policies (SOL / BND / COV) over a fake posterior
  - masked-softmax normalized policy sums to 1 and masks forbidden states
  - batched allocation: counts conserve q; residuals carry across rounds
  - CASESModel integration: default OFF, acquire entries, recommended_gear
"""

from __future__ import annotations

import random

import pytest

from cases.core.acquisition import (
    BND,
    COV,
    OPEN,
    SOL,
    CHANNELS,
    ExplorationController,
    ExplorationGear,
)


def _boolean_model(exploration=ExplorationGear.OFF, **kw):
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask
    from cases.core.model import CASESModel

    task = BooleanTask(
        dataset="boolean_test", observation_set_id="t", variables=("x", "y"),
        operators=frozenset({"AND", "OR", "NOT"}), max_depth=2,
        mechanistic_opts={"apply_commutativity": True, "apply_idempotence_and_or": True,
                          "flatten_associativity": True},
        observations=(), n_observations=0,
    )
    return CASESModel(adapter=BooleanAdapter(task), exploration=exploration, **kw)


def _update_boolean(m, texts=("x AND y", "x OR y", "NOT x", "x AND NOT y")):
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    m.update([EvidenceRecord(kind="hypothesis",
                             hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t))
              for i, t in enumerate(texts)])


class _Marg:
    def __init__(self, p, u):
        self.solution_probability = p
        self.std = u
        self.mean = 0.0


class _FakePosterior:
    def __init__(self, objs):
        self._d = objs

    def marginal(self, oid):
        return self._d[oid]


# ---------------------------------------------------------------------------
# gear profiles
# ---------------------------------------------------------------------------


def test_five_gears_monotonic_exploration_mass():
    ctl = ExplorationController()
    mass = {g: ctl.profile(g).exploration_mass for g in ctl.gears()}
    assert mass[ExplorationGear.OFF] == pytest.approx(0.0)
    ordered = [ExplorationGear.OFF, ExplorationGear.LOW, ExplorationGear.MEDIUM,
               ExplorationGear.HIGH, ExplorationGear.ULTRA]
    vals = [mass[g] for g in ordered]
    assert vals == sorted(vals)
    assert vals == list(dict.fromkeys(vals))  # strictly increasing


def test_open_mass_monotonic():
    ctl = ExplorationController()
    ordered = [ExplorationGear.OFF, ExplorationGear.LOW, ExplorationGear.MEDIUM,
               ExplorationGear.HIGH, ExplorationGear.ULTRA]
    opens = [ctl.profile(g).w_open for g in ordered]
    assert opens == sorted(opens)


def test_off_is_pure_exploit():
    ctl = ExplorationController()
    prof = ctl.profile(ExplorationGear.OFF)
    assert prof.weights == (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# channel selection
# ---------------------------------------------------------------------------


def test_select_channel_off_always_sol():
    ctl = ExplorationController()
    rng = random.Random(0)
    for _ in range(200):
        assert ctl.select_channel(rng, ExplorationGear.OFF) == SOL


def test_select_channel_ultra_reaches_all():
    ctl = ExplorationController()
    rng = random.Random(1)
    seen = {ctl.select_channel(rng, ExplorationGear.ULTRA) for _ in range(4000)}
    assert seen == set(CHANNELS)


def test_select_channel_proportions_match_weights():
    ctl = ExplorationController()
    rng = random.Random(2)
    counts = {c: 0 for c in CHANNELS}
    n = 20000
    for _ in range(n):
        counts[ctl.select_channel(rng, ExplorationGear.MEDIUM)] += 1
    weights = dict(zip(CHANNELS, ctl.profile(ExplorationGear.MEDIUM).weights))
    for c in CHANNELS:
        assert abs(counts[c] / n - weights[c]) < 0.03


# ---------------------------------------------------------------------------
# within-space policies over a fake posterior
# ---------------------------------------------------------------------------

POST = _FakePosterior({
    "a": _Marg(p=0.9, u=0.1),
    "b": _Marg(p=0.5, u=1.0),
    "c": _Marg(p=0.1, u=0.5),
})


def test_sol_prefers_high_prob():
    ctl = ExplorationController()
    scores = ctl.channel_scores(SOL, ("a", "b", "c"), POST, eta=0.5)
    assert scores["a"] > scores["b"] > scores["c"]


def test_bnd_peaks_near_eta_with_uncertainty():
    ctl = ExplorationController()
    scores = ctl.channel_scores(BND, ("a", "b", "c"), POST, eta=0.5)
    # b sits on the boundary (p=eta) with high uncertainty => top BND target
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_cov_prefers_high_uncertainty_without_geometry():
    ctl = ExplorationController()
    scores = ctl.channel_scores(COV, ("a", "b", "c"), POST, eta=0.5)
    # default coverage gap = uncertainty; s = (1+beta) * u
    assert scores["b"] > scores["c"] > scores["a"]


def test_normalized_masks_and_sums_to_one():
    ctl = ExplorationController()
    pol = ctl.normalized(SOL, ("a", "b", "c"), POST, eta=0.5, mask={"a"})
    assert set(pol) == {"b", "c"}
    assert sum(pol.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# batched allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gear", list(ExplorationGear))
def test_batch_plan_conserves_q(gear):
    ctl = ExplorationController()
    for q in (1, 2, 5, 17):
        counts, res = ctl.batch_plan(q, gear)
        assert sum(counts.values()) == q
        assert all(c >= 0 for c in counts.values())


def test_batch_plan_residual_carries_and_converges():
    ctl = ExplorationController()
    gear = ExplorationGear.MEDIUM
    q = 10
    total = {c: 0 for c in CHANNELS}
    res = None
    rounds = 50
    for _ in range(rounds):
        counts, res = ctl.batch_plan(q, gear, res)
        for c in CHANNELS:
            total[c] += counts[c]
    weights = dict(zip(CHANNELS, ctl.profile(gear).weights))
    grand = rounds * q
    for c in CHANNELS:
        assert abs(total[c] / grand - weights[c]) < 0.01


# ---------------------------------------------------------------------------
# CASESModel integration
# ---------------------------------------------------------------------------


def test_model_default_gear_off():
    m = _boolean_model()
    assert m.exploration is ExplorationGear.OFF
    _update_boolean(m)
    picks = m.acquire(5, rng=random.Random(0))
    assert len(picks) == 5
    # OFF => every slot is SOL with an in-space object
    assert all(ch == SOL and oid is not None for ch, oid in picks)


def test_model_acquire_mixed_under_medium():
    m = _boolean_model(exploration=ExplorationGear.MEDIUM)
    _update_boolean(m)
    picks = m.acquire(80, rng=random.Random(3))
    channels = {ch for ch, _ in picks}
    assert SOL in channels


def test_model_open_never_crashes():
    m = _boolean_model(exploration=ExplorationGear.ULTRA)
    _update_boolean(m)
    picks = m.acquire(200, rng=random.Random(7))
    assert all(ch in CHANNELS for ch, _ in picks)


def test_open_directive_text():
    m = _boolean_model()
    assert "OPEN proposal" in m.open_directive()


def test_recommended_gear_on_representation_inadequate():
    m = _boolean_model()
    m.state.metadata["representation_admitted"] = False
    assert m.recommended_gear() is ExplorationGear.ULTRA


def test_recommended_gear_default_medium():
    m = _boolean_model()
    assert m.recommended_gear() is ExplorationGear.MEDIUM
