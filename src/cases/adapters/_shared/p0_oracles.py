"""Production oracle wrappers for the P0 evaluator (E4-1, 2026-09-01).

Normalizes the frozen task oracles (evaluator-only gold accessors) to the
evaluator's canonical string-key space:
  bh   — "|".join((a, l, b, d))          (slot-wise Hamming = allowed-info distance)
  gb1  — variant string (e.g. "T35P")
  made — pending E2 recovery-backend wiring (raises NotImplementedError until
         the frozen rbf_gp posterior reconstruction lands; made instances are
         graded once that view is registered).
"""

from __future__ import annotations

from typing import Any, Sequence


class BHOracleView:
    """Pipe-key view over BHFiniteOracle (partition slot = ligand = index 1)."""

    def __init__(self, inner) -> None:
        self._inner = inner

    @staticmethod
    def key(cand) -> str:
        return "|".join(str(x) for x in cand)

    @property
    def candidates(self):
        return [self.key(c) for c in self._inner.candidates]

    def evaluate(self, key: str) -> float:
        return float(self._inner.evaluate(self._parse(key)))

    def gold_of(self, key: str) -> float:
        return float(self._inner.gold_of(self._parse(key)))

    def solution_set(self):
        return [self.key(c) for c in self._inner.solution_set()]

    def solution_membership(self, key: str) -> bool:
        return bool(self._inner.solution_membership(self._parse(key)))

    @property
    def gamma(self) -> float:
        return float(self._inner.gamma)

    def gold_values(self):
        return list(self._inner.gold_values())

    def is_legal(self, key: str) -> bool:
        return self._inner._data.contains(self._parse(key))

    @property
    def edit_distance(self):
        return _slot_hamming

    def legal_slot_values(self) -> dict[str, list[str]]:
        slots: dict[int, set] = {}
        for c in self._inner.candidates:
            for i, v in enumerate(c):
                slots.setdefault(i, set()).add(str(v))
        return {s: sorted(v) for s, v in zip("albd", slots.values())}

    def _parse(self, key: str):
        a, l, b, d = key.split("|")
        return (a, l, b, d)


class GB1OracleView:
    """String-key view over GB1FiniteOracle (position-set partition)."""

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def candidates(self):
        return [str(c) for c in self._inner.candidates]

    def evaluate(self, key: str) -> float:
        return float(self._inner.evaluate(key))

    def gold_of(self, key: str) -> float:
        return float(self._inner.gold_of(key))

    def solution_set(self):
        return [str(c) for c in self._inner.solution_set()]

    def solution_membership(self, key: str) -> bool:
        return bool(self._inner.solution_membership(key))

    @property
    def gamma(self) -> float:
        return float(self._inner.gamma)

    def gold_values(self):
        return list(self._inner.gold_values())

    def is_legal(self, key: str) -> bool:
        return key in set(self._inner.candidates)

    @staticmethod
    def edit_distance(a: str, b: str) -> int:
        """Position-set Hamming: distinct mutated positions between variants."""
        sa, sb = _positions(a), _positions(b)
        return len(sa ^ sb)

    @staticmethod
    def position_set(key: str) -> str:
        return ",".join(sorted(_positions(key)))

    def legal_position_sets(self):
        return sorted({self.position_set(k) for k in self._inner.candidates})


def _positions(variant: str) -> set[str]:
    """Mutation positions of a variant like 'T35P,A46G' -> {'35', 'A46'...}."""
    out = set()
    for part in variant.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            out.add(digits)
    return out


def _slot_hamming(a: str, b: str) -> int:
    pa, pb = a.split("|"), b.split("|")
    if len(pa) != len(pb):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(pa, pb) if x != y)


def oracle_factory(task: str) -> Any:
    if task == "bh":
        from cases.adapters.bh.data import BHData
        from cases.campaigns.bh_oracle import BHFiniteOracle
        return BHOracleView(BHFiniteOracle(BHData()))
    if task == "gb1":
        from cases.campaigns.gb1_oracle import GB1FiniteOracle
        return GB1OracleView(GB1FiniteOracle())
    raise NotImplementedError(
        "made oracle view pending E2 rbf_gp recovery-backend wiring")
