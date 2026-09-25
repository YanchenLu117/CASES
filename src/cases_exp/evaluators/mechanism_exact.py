"""Exact mechanism evaluators for E2a (Main A) and E3 (Main B). CPU-only, no LLM.

Implements:
- exact Bayes risk R*(e;W) over finite H x W with rational arithmetic (fractions.Fraction)
- boundary loss decomposition R_cmp - R0 = Delta_src + Delta_commit + Delta_ret
- forbidden-depth paired contrast T_H3
- minimal collapse supports K_t(p) via brute-force + pruning (certified worlds cross-check)
- HS-tree minimum-cost hitting set (Reiter-style, fairness: no CASES governance inside)
- GVR (Governance-Valid Recovery) checker for E3
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Iterable, Sequence

Prob = Fraction


def bayes_risk(pairs: Sequence[tuple[tuple, str, Prob]], channel) -> tuple[Fraction, dict]:
    """R*(e(H); W) for a finite episode.

    pairs: list of (h_key, w_key, joint_prob P(h,w)) — Y and loss enter via loss table
    channel: callable h_key -> observable encoding e (deterministic)
    Loss is provided by the caller through action-optimal cell losses; here we take
    the generic form: each (h,w,y) has loss per action; we abstract via cell_min losses.

    Simpler contract used by E2a worlds: caller supplies for each (h,w) the vector
    L[a] over actions; we group by (channel(h), w) and sum the cell-minimum.
    """
    raise NotImplementedError("use bayes_risk_cells for the concrete world format")


def bayes_risk_cells(cells: dict[tuple, dict[str, Prob]], action_losses: dict[tuple, dict[str, Prob]]) -> Fraction:
    """cells: (encoding, w) -> {h_key: P(h,w)}. action_losses: (h_key or cell, action) -> loss.

    Concrete E2a format: action_losses maps h_key -> {action: loss}, and the Bayes action
    per cell minimizes expected loss under the cell's h-distribution.
    """
    total = Fraction(0)
    for (enc, w), hdist in cells.items():
        actions = next(iter(hdist.values().__iter__().__next__().__class__().__dict__.get("x", {}) or {}), None) if False else None
        # collect actions from the first h_key
        first_h = next(iter(hdist))
        acts = action_losses[first_h].keys()
        best = None
        for a in acts:
            exp = sum(p * action_losses[h][a] for h, p in hdist.items())
            if best is None or exp < best:
                best = exp
        total += best
    return total


def boundary_decomposition(r0: Fraction, rsrc: Fraction, rsem: Fraction, rcmp: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    """R_cmp - R_0 = Delta_src + Delta_commit + Delta_ret (Standard Proposition)."""
    return rsrc - r0, rsem - rsrc, rcmp - rsem


def minimal_supports(equations_per_commitment: dict[str, list[tuple[str, str]]], target_pair: tuple[str, str]) -> list[frozenset[str]]:
    """K_t(p): inclusion-minimal C subset of A_t s.t. target_pair is joined by the
    transitive closure of union of rho(a), a in C.

    Brute force increasing by size with superset pruning (spec §28.1).
    """
    from collections import defaultdict

    a_ids = sorted(equations_per_commitment)

    def joins(subset: Iterable[str]) -> bool:
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in subset:
            for u, v in equations_per_commitment[a]:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
        return find(target_pair[0]) == find(target_pair[1])

    supports: list[frozenset[str]] = []
    for size in range(1, len(a_ids) + 1):
        for combo in combinations(a_ids, size):
            cs = frozenset(combo)
            if any(s <= cs for s in supports):
                continue  # prune supersets of known minimal supports
            if joins(cs):
                supports.append(cs)
        if supports and size >= max(len(s) for s in supports):
            # can early-exit when all smaller supersets already pruned
            pass
    return supports


def hs_tree_min_cost_hitting_set(supports: list[frozenset[str]], costs: dict[str, Fraction]) -> tuple[frozenset[str], Fraction] | None:
    """Minimum-cost hitting set over the complete support family (spec §28.2).

    Fairness rule: NO governance constraints inside — pure structural optimum.
    Exact branch-and-bound (sufficient for certification-scale worlds).
    """
    if not supports:
        return frozenset(), Fraction(0)
    elements = sorted({e for s in supports for e in s})
    best: tuple[frozenset[str], Fraction] | None = None

    def hits_all(chosen: frozenset[str]) -> bool:
        return all(chosen & s for s in supports)

    def bb(idx: int, chosen: frozenset[str], cost: Fraction) -> None:
        nonlocal best
        if best is not None and cost >= best[1]:
            return
        if hits_all(chosen):
            if best is None or cost < best[1]:
                best = (chosen, cost)
            return
        if idx >= len(elements):
            return
        # lower bound: must still hit every unhit support
        unhit = [s for s in supports if not (chosen & s)]
        min_extra = Fraction(0)
        if unhit:
            min_extra = min((costs.get(e, Fraction(1)) for e in unhit[0]), default=Fraction(0))
        if best is not None and cost + min_extra >= best[1]:
            return
        e = elements[idx]
        bb(idx + 1, chosen | {e}, cost + costs.get(e, Fraction(1)))
        bb(idx + 1, chosen, cost)

    bb(0, frozenset(), Fraction(0))
    return best


def gvr(distinction_restored: bool, authorization_valid: bool,
        protected_preserved: bool, dependency_closure_ok: bool) -> bool:
    """GVR = 1[DistinctionRestored AND AuthorizationValid AND ProtectedPreserved AND DependencyClosure] (spec §16.2)."""
    return distinction_restored and authorization_valid and protected_preserved and dependency_closure_ok


def t_h3(g_values: dict[tuple[str, str, int], Fraction], d_star: dict[str, int]) -> Fraction:
    """T_H3 = mean over (task, locus, d < d*) of G[i,l,d] - G[i,l,d*] (spec §14.9)."""
    contrasts: list[Fraction] = []
    for (task, locus, d), g in g_values.items():
        ds = d_star[locus]
        if d < ds and (task, locus, ds) in g_values:
            contrasts.append(g - g_values[(task, locus, ds)])
    if not contrasts:
        return Fraction(0)
    return sum(contrasts) / len(contrasts)
