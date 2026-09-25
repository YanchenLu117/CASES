"""CausalAdapter — HypoSpace causal-graph domain (Agent A).

Grounding chain (per the frozen CASES contract and the official HypoSpace
semantics):

    language -> official parse_llm_response (edge list, DAG checks)
             -> official validation (perturbation-effect equality)
             -> canonical sorted labeled edge set.

Compile / verify execute the *official* repo code through
``cases.adapters.hypospace.official``; this module only wires the official
semantics into the shared ``TaskAdapter`` contract and defines the CASES-layer
canonical form, G_sci and G_rev.

Canonical form (uniqueness == official ``CausalGraph.get_hash`` equivalence):

    "causal_dag|<sorted nodes>|<sorted edges>"   e.g. "causal_dag|A;B|A->B;B->C"

G_sci  : cosine similarity on outcome-blind directed-edge indicator vectors.
G_rev  : add_edge / remove_edge / reverse_edge, only when the resulting object
         passes the official validator (target objects are verified).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.errors import EvaluationNotApplicable, GroundingError
from ...core.ids import stable_edge_id, stable_object_id
from ...core.interfaces import TaskAdapter
from ...core.types import (
    GroundedObject,
    LanguageHypothesis,
    RawScientificObject,
    RevisionEdge,
    RevisionGraph,
    ScientificEdge,
    ScientificGraph,
    VerificationResult,
)
from . import official
from .tasks import CausalTask

OBJECT_TYPE = "causal_dag"


class CausalAdapter(TaskAdapter):
    task_id = "hypospace_causal"
    object_type = OBJECT_TYPE

    def __init__(self, task: CausalTask) -> None:
        self.task = task
        self._parser = official.causal_parser(task.nodes, task.max_edges)

    # -- compile: P_tau (official parsing) ---------------------------------

    async def compile(self, hypothesis: LanguageHypothesis, llm) -> RawScientificObject:
        """Official causal response parser applied to the hypothesis text."""
        graph = self._parser(hypothesis.text)
        if graph is None:
            raise GroundingError(
                f"causal parse failed (unparseable / unknown node / self-loop / "
                f">max_edges / cyclic) for hypothesis {hypothesis.hypothesis_id}"
            )
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={
                "edges": sorted(graph.edges),
                "nodes": sorted(self.task.nodes),
                "n_edges": len(graph.edges),
            },
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
            metadata={"evaluator": "hypospace_official_parse", "parser": "parse_llm_response"},
        )

    # -- verify: V_tau (official validation; deterministic, no repair) ------

    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        graph = official.CausalGraph(raw_object.payload["nodes"], raw_object.payload["edges"])
        errors: list[str] = []

        # Re-check DAG (official parse already guarantees it; belt & suspenders).
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(graph.nodes)
        g.add_edges_from(graph.edges)
        if not nx.is_directed_acyclic_graph(g):
            errors.append("not_a_dag")

        if not official.validate_causal(graph, self.task.observations):
            errors.append("effects_mismatch:not_consistent_with_observations")

        if raw_object.payload["n_edges"] > (self.task.max_edges or 10**9):
            errors.append("edges_exceed_max")

        if errors:
            return VerificationResult(valid=False, errors=tuple(errors))
        return VerificationResult(valid=True)

    # -- canonicalize: K_tau -------------------------------------------------

    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        edges = sorted(tuple(e) for e in raw_object.payload["edges"])
        nodes = tuple(sorted(raw_object.payload["nodes"]))
        edge_str = ";".join(f"{u}->{v}" for u, v in edges) or "no-edges"
        canonical_form = f"{OBJECT_TYPE}|{';'.join(nodes)}|{edge_str}"

        display = "no edges" if not edges else "; ".join(f"{u}->{v}" for u, v in edges)
        return GroundedObject(
            object_id=stable_object_id(self.task_id, canonical_form),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form=canonical_form,
            payload={"nodes": nodes, "edges": edges, "n_edges": len(edges)},
            display_text=display,
            source_hypothesis_ids=(raw_object.source_hypothesis_id,),
            metadata={
                "evaluator": "hypospace_official",
                "domain": "causal",
                "observation_set_id": self.task.observation_set_id,
            },
        )

    # -- G_sci: frozen, outcome-blind structural relations -------------------

    def _feature(self, obj: GroundedObject) -> np.ndarray:
        """Directed edge-indicator vector over all ordered node pairs."""
        nodes = obj.payload["nodes"]
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        vec = np.zeros(n * (n - 1), dtype=np.float64)
        for u, v in obj.payload["edges"]:
            vec[index[u] * (n - 1) + (index[v] if index[v] < index[u] else index[v] - 1)] = 1.0
        return vec

    def build_scientific_graph(self, objects: tuple[GroundedObject, ...]) -> ScientificGraph:
        feats = {o.object_id: self._feature(o) for o in objects}
        edges: list[ScientificEdge] = []
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                fa, fb = feats[a.object_id], feats[b.object_id]
                denom = (np.linalg.norm(fa) * np.linalg.norm(fb)) or 1.0
                cos = float(fa @ fb / denom)
                if cos > 0.0:
                    edges.append(
                        ScientificEdge(
                            edge_id=stable_edge_id(a.object_id, b.object_id, "causal_structure"),
                            source_id=a.object_id,
                            target_id=b.object_id,
                            weight=cos,
                            distance=1.0 - cos,
                            relation_type="causal_structure",
                            metadata={"shared_edges": int(fa @ fb), "outcome_blind": True},
                        )
                    )
        return ScientificGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "causal", "relation_type": "causal_structure", "outcome_blind": True},
        )

    # -- G_rev: interpretable, benchmark-valid executable revisions ----------

    def build_revision_graph(self, objects: tuple[GroundedObject, ...]) -> RevisionGraph:
        edges: list[RevisionEdge] = []
        for a in objects:
            for b in objects:
                if a.object_id == b.object_id:
                    continue
                e1, e2 = set(a.payload["edges"]), set(b.payload["edges"])
                # add_edge / remove_edge: symmetric difference of exactly one
                if len(e1 ^ e2) == 1:
                    (only,) = e1 ^ e2
                    action = "remove_edge" if only in e1 else "add_edge"
                    self._append_revision(edges, a, b, action, {"edge": list(only)})
                    continue
                # reverse_edge: all edges equal except one reversed pair
                common = e1 & e2
                if len(e1) == len(e2) and len(common) == len(e1) - 1:
                    diff1 = e1 - common
                    diff2 = e2 - common
                    if len(diff1) == 1 and len(diff2) == 1:
                        (u1, v1), = diff1
                        (u2, v2), = diff2
                        if u1 == v2 and v1 == u2:
                            self._append_revision(
                                edges, a, b, "reverse_edge", {"edge": [u1, v1]}
                            )
        return RevisionGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "causal", "outcome_blind": True},
        )

    def _append_revision(self, edges, a, b, action, payload) -> None:
        # Benchmark-validity guarantee: every constructed node passed the
        # official validator at insertion, so a G_rev target is valid by
        # construction.  Audit re-check: the edit result must remain a DAG
        # (syntax space) — deterministic, no observation re-evaluation.
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(a.payload["nodes"])
        g.add_edges_from(b.payload["edges"])
        if not nx.is_directed_acyclic_graph(g):
            return
        edges.append(
            RevisionEdge(
                edge_id=stable_edge_id(a.object_id, b.object_id, action),
                source_id=a.object_id,
                target_id=b.object_id,
                action_type=action,
                action_description=f"{action}: {payload['edge']}",
                edit_payload=payload,
                metadata={"domain": "causal", "outcome_blind": True},
            )
        )

    # -- evaluation ----------------------------------------------------------

    async def evaluate(self, obj: GroundedObject):
        raise EvaluationNotApplicable(
            "HypoSpace has no scalar empirical utility field; validity is a "
            "deterministic benchmark verdict, not an Observation."
        )

    def render_object(self, obj: GroundedObject) -> str:
        return obj.display_text

    # -- FrontierSemantics (V6.1 SpaceReadout) --------------------------------
    # Descriptor per Hypo问题诊断1: phi(G) = (|E|, degree profile,
    # #v-structures, longest path, source/sink pattern).  Outcome-blind; from
    # canonical object + public task grammar only.

    def descriptor(self, obj: GroundedObject) -> dict[str, Any]:
        from itertools import combinations as _combs

        edges = obj.payload["edges"]
        nodes = obj.payload["nodes"]
        n = len(nodes)
        indeg = {v: 0 for v in nodes}
        outdeg = {v: 0 for v in nodes}
        for u, v in edges:
            outdeg[u] += 1
            indeg[v] += 1
        # v-structures: u->w, v->w with no edge between u and v
        edge_set = set(edges)
        n_vstruct = 0
        for w in nodes:
            parents = [p for p in nodes if (p, w) in edge_set]
            for p1, p2 in _combs(parents, 2):
                if (p1, p2) not in edge_set and (p2, p1) not in edge_set:
                    n_vstruct += 1
        # longest path (edges) via topological DP on the DAG
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        dist = {v: 0 for v in nodes}
        try:
            for u in nx.topological_sort(g):
                for v in g.successors(u):
                    dist[v] = max(dist[v], dist[u] + 1)
        except Exception:  # not a DAG (defensive; verified objects are DAGs)
            pass
        sources = tuple(sorted(v for v in nodes if indeg[v] == 0))
        sinks = tuple(sorted(v for v in nodes if outdeg[v] == 0))
        degree_profile = tuple(sorted((indeg[v] + outdeg[v] for v in nodes), reverse=True))
        return {
            "n_edges": len(edges),
            "degree_profile": degree_profile,
            "n_vstructures": n_vstruct,
            "longest_path": max(dist.values()),
            "source_sink": (sources, sinks),
        }

    def attribute_domains(self) -> dict[str, tuple[Any, ...]]:
        from itertools import combinations as _combs, combinations_with_replacement as _cwr

        n = len(self.task.nodes)
        max_e = self.task.max_edges if self.task.max_edges is not None else n * (n - 1)
        profile_domain = sorted({tuple(sorted(seq, reverse=True)) for seq in _cwr(range(n + 1), n)})
        subsets = [tuple(sorted(c)) for r in range(n + 1) for c in _combs(self.task.nodes, r)]
        ss_domain = tuple((s, t) for s in subsets for t in subsets)
        return {
            "n_edges": tuple(range(0, max_e + 1)),
            "degree_profile": tuple(profile_domain),
            "n_vstructures": tuple(range(0, max(1, n * (n - 1) // 2) + 1)),
            "longest_path": tuple(range(0, n)),
            "source_sink": ss_domain,
        }

    def revision_hints(self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]) -> tuple[str, ...]:
        hints: list[str] = []
        if to_descriptor.get("n_edges", 0) > from_descriptor.get("n_edges", 0):
            hints.append("add_edge")
        if to_descriptor.get("n_edges", 0) < from_descriptor.get("n_edges", 0):
            hints.append("remove_edge")
        if to_descriptor.get("longest_path", 0) != from_descriptor.get("longest_path", 0):
            hints.append("reverse_edge")
        return tuple(dict.fromkeys(hints)) or ("add_edge",)

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """Readout-v2 feasibility: DAG-degree handshake and path bounds under
        the public causal grammar."""
        n = len(self.task.nodes)
        n_edges = int(descriptor["n_edges"])
        max_e = self.task.max_edges if self.task.max_edges is not None else n * (n - 1)
        profile = descriptor["degree_profile"]
        longest = int(descriptor["longest_path"])
        n_vs = int(descriptor["n_vstructures"])
        (sources, sinks) = descriptor["source_sink"]
        if not (0 <= n_edges <= max_e):
            return False
        if len(profile) != n or any(d < 0 or d > n - 1 for d in profile):
            return False
        if sum(profile) != 2 * n_edges:  # handshake
            return False
        if longest < 0 or longest >= n or longest > n_edges:
            return False
        if n_vs < 0 or n_vs > n_edges:
            return False
        if not sources or not sinks:
            return False
        if not set(sources).issubset(set(self.task.nodes)) or not set(sinks).issubset(set(self.task.nodes)):
            return False
        # a DAG has at least one source and one sink; if n_edges > 0, some
        # node must have positive degree
        if n_edges > 0 and all(d == 0 for d in profile):
            return False
        return True

    def _observation_must_reach(self) -> dict[str, frozenset[str]]:
        """Forced descendant sets derived from the PUBLIC perturbation
        observations.

        ``must_reach[p]`` = nodes that MUST be a descendant of ``p`` (effect=1
        in some perturbation of p).  Public task information (it is in the task
        prompt given to the model) — the adapter uses it to prune
        observation-impossible frontier cells, WITHOUT calling the validator or
        enumerating the admissible set (no-leakage contract preserved).
        """
        acc: dict[str, set[str]] = {}
        for obs in self.task.observations:
            p = obs["perturbed_node"]
            targets = {n for n, v in obs["effects"].items() if n != p and int(v) == 1}
            if p in targets:  # defensive; p's own effect is always 0
                targets.discard(p)
            acc[p] = acc.get(p, set()) | targets
        return {p: frozenset(t) for p, t in acc.items()}

    def observation_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """Observation-aware frontier-cell feasibility (hyperspace adapter
        layer; consumed by the CASES renderer).

        Same grammar rules as ``cell_feasible``, PLUS public-observation
        structural constraints so only cells realizable by a graph consistent
        with the observations are offered to the model:
          * forced descendants cannot be sources (in-degree 0 => unreachable);
          * a node that must reach others cannot be a sink;
          * the degree profile needs enough positive-degree nodes to host the
            most demanding forced-descendant fan (>= max|must_reach| + 1).

        No-leakage: uses only public task grammar + public observations.
        """
        if not self.cell_feasible(descriptor):
            return False
        must_reach = self._observation_must_reach()
        src_set, snk_set = set(descriptor["source_sink"][0]), set(descriptor["source_sink"][1])
        profile = descriptor["degree_profile"]
        max_fan = 0
        for p, targets in must_reach.items():
            max_fan = max(max_fan, len(targets))
            if not targets:
                continue
            # p must have at least one outgoing edge to reach its descendants
            if p in snk_set:
                return False
            # each forced descendant needs an incoming edge -> cannot be a source
            if targets & src_set:
                return False
        pos_deg = sum(1 for d in profile if d > 0)
        if max_fan > 0 and pos_deg < max_fan + 1:
            return False
        return True

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        """Domain-flavoured directive for a causal frontier target.

        Keep it ACTIONABLE and observation-respecting.  Concrete edge/path
        direction (add/remove ~1 edge, lengthen/shorten the path) is useful
        steering; dictating exact sources/sinks / degree profile / #v-structures
        is counter-productive (the local model drifts off the observations and
        returns invalid graphs).  The observation constraint is re-stated
        explicitly every time because it is the binding feasibility rule.
        """
        parts = []
        n_edges = int(descriptor["n_edges"])
        lp = int(descriptor["longest_path"])
        if n_edges:
            parts.append(f"with about {n_edges} edge(s)")
        if lp:
            parts.append(f"a longest causal path of about {lp} edge(s)")
        base = "Produce a valid causal DAG " + (", ".join(parts) if parts else "with a different structure")
        return (
            base
            + " that STILL matches ALL the listed observations EXACTLY "
              "(every descendant effect must hold as given), and whose edge set "
              "differs from every anchored DAG above"
        )
