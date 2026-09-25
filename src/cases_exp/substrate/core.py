"""CASES v9.5 substrate: minimal executable core of the theory objects.

Chain (proposal §7): H -> e_src(H) -> e_sem(H) -> e_cmp(H)
- SourceFrame: registry of closed claims + source compilation delta^src
- Commitments: finite retractable identifications, quotient Theta_{A_t}
- Grounding: computational carrier map gamma: S_t -> C_t

The exact Bayes-risk machinery (Standard Proposition, Main A) lives in
evaluators/mechanism_exact.py; this module carries the state objects that
methods construct and mutate at the three access depths D1/D2/D3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AccessDepth(Enum):
    GROUNDING_ONLY = 1     # D1: may modify gamma / carrier / executable realization
    COMMITMENT_EDIT = 2    # D2: may modify A_t, Theta_A (+ downstream)
    SOURCE_EDIT = 3        # D3: may modify P^src, F, delta^src (+ everything)


class Undetermined(Enum):
    UNDETERMINED = "undetermined"


@dataclass
class Claim:
    claim_id: str
    version: int = 1
    text: str = ""
    provenance: list[str] = field(default_factory=list)
    evidence_bindings: list[str] = field(default_factory=list)


@dataclass
class Commitment:
    """A retractable scientific identification a in A_t contributing equations rho(a) on F_t."""
    commitment_id: str
    equations: list[tuple[str, str]] = field(default_factory=list)  # pairs of source terms identified
    witness: dict[str, Any] = field(default_factory=dict)           # witness-validity payload
    protected: bool = False                                         # governance: may not be retracted
    dependencies: list[str] = field(default_factory=list)           # cascade closure ids


@dataclass
class SourceState:
    """P^src: source presentation + compilation delta^src: Reg^cl -> Term(P^src)."""
    claims: dict[str, Claim] = field(default_factory=dict)
    source_compilation: dict[str, str | None] = field(default_factory=dict)  # claim_id -> source term or None (unrepresented)
    vocabulary: list[str] = field(default_factory=list)

    def compile(self, claim_id: str) -> str | None:
        return self.source_compilation.get(claim_id)


@dataclass
class SubstrateState:
    """The full X_t = (Sem, Op, FA) in serializable minimal form."""
    source: SourceState = field(default_factory=SourceState)
    active_commitments: dict[str, Commitment] = field(default_factory=dict)
    computational_grounding: dict[str, str] = field(default_factory=dict)  # semantic -> carrier terms
    frontier_certificates: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1

    def retract(self, removal_ids: set[str]) -> "SubstrateState":
        """Pure commitment retraction R: S_t^{-R} = F_t / Theta_{A_t \\ R} (theory §9.1).
        Returns a new state; dependency closure is applied by governance, not here."""
        new = SubstrateState(
            source=self.source,
            active_commitments={k: v for k, v in self.active_commitments.items() if k not in removal_ids},
            computational_grounding=dict(self.computational_grounding),
            frontier_certificates=list(self.frontier_certificates),
            version=self.version + 1,
        )
        return new

    def semantic_quotient_pairs(self) -> set[frozenset[str]]:
        """Collided claim pairs under Theta_{A_t}: pairs whose compiled source terms
        are joined by commitment equations (transitive closure)."""
        import itertools
        # union-find over source terms
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for c in self.active_commitments.values():
            for a, b in c.equations:
                union(a, b)
        groups: dict[str, list[str]] = {}
        for claim_id, term in self.source.source_compilation.items():
            if term is not None:
                groups.setdefault(find(term), []).append(claim_id)
        collided = set()
        for members in groups.values():
            if len(members) > 1:
                for a, b in itertools.combinations(sorted(members), 2):
                    collided.add(frozenset((a, b)))
        return collided
