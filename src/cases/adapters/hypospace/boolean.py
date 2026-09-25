"""BooleanAdapter — HypoSpace boolean-logic domain (Agent A).

Grounding chain:

    language -> official parse_llm_response (expression string)
             -> official validation (variables / operators / no constants /
                depth / observation consistency)
             -> deterministic simplification via the official mechanistic key
                (commutativity + AND/OR idempotence + associativity flattening)
             -> canonical expression rebuilt from the key.

IMPORTANT — canonicalization follows the *official* dedup semantics: the
official benchmark deliberately does NOT collapse truth-table-equivalent
expressions (its README: CNF is "for display only, not used for dedup").
Structural (mechanistic) equivalence is the official uniqueness notion, so
``x AND y`` and ``y AND x`` collapse, while structurally different expressions
with the same truth table do NOT.  This preserves the official Uniqueness /
Recovery metrics exactly.

Canonical form:  "boolean_expr|<expression rebuilt from mechanistic key>"
G_sci  : cosine similarity on outcome-blind operator/variable/depth features.
G_rev  : wrap_transform / replace_variable / replace_operator /
         add_literal / remove_literal, only when the target passes the
         official validator.
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
from .tasks import BooleanTask

OBJECT_TYPE = "boolean_expr"


class BooleanAdapter(TaskAdapter):
    task_id = "hypospace_boolean"
    object_type = OBJECT_TYPE

    def __init__(self, task: BooleanTask) -> None:
        self.task = task
        self._parser = official.boolean_parser()

    # -- compile -------------------------------------------------------------

    async def compile(self, hypothesis: LanguageHypothesis, llm) -> RawScientificObject:
        expr_str = self._parser(hypothesis.text)
        if expr_str is None:
            raise GroundingError(
                f"boolean parse failed for hypothesis {hypothesis.hypothesis_id}"
            )
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={"expr": expr_str},
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
            metadata={"evaluator": "hypospace_official_parse", "parser": "parse_llm_response"},
        )

    # -- verify (official validate_expression semantics) ----------------------

    def candidate_space(self, *, limit: int = 64) -> list[str]:
        """Legal candidate boolean expressions (campaign §4 candidate space).

        Outcome-blind grammar over the task variables/operators; used by
        cases.api.ScientificCampaign as the shared candidate boundary.  Never
        includes fitted outcomes or evaluator truths.
        """
        vs = tuple(getattr(self.task, "variables", ("x", "y")))
        ops = set(getattr(self.task, "operators", frozenset({"AND", "OR", "NOT"})) or frozenset({"AND", "OR", "NOT"}))
        out = list(vs)
        out += [f"NOT {a}" for a in vs] if "NOT" in ops else []
        if ("AND" in ops or "OR" in ops):
            for a in vs:
                for op in ops:
                    if op == "NOT":
                        continue
                    for b in vs:
                        out.append(f"{a} {op} {b}")
        seen = []
        for x in out:
            if x not in seen:
                seen.append(x)
        return seen[: max(1, int(limit))]

    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        expr_str = str(raw_object.payload["expr"])
        valid, _ = official.validate_boolean(
            expr_str,
            self.task.variables,
            self.task.operators,
            self.task.max_depth,
            self.task.observations,
        )
        if not valid:
            return VerificationResult(
                valid=False,
                errors=("invalid_boolean:variables_operators_constants_depth_or_observations",),
            )
        return VerificationResult(valid=True)

    # -- canonicalize ----------------------------------------------------------

    def _mechanistic_key(self, expr_str: str) -> tuple | None:
        return official.boolean_mechanistic_key(
            expr_str, self.task.variables, self.task.operators, dict(self.task.mechanistic_opts)
        )

    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        expr_str = str(raw_object.payload["expr"])
        key = self._mechanistic_key(expr_str)
        if key is None:
            from ...core.errors import CanonicalizationError

            raise CanonicalizationError(f"cannot compute mechanistic key for {expr_str!r}")
        canonical_expr = self._rebuild(key)
        canonical_form = f"{OBJECT_TYPE}|{canonical_expr}"
        return GroundedObject(
            object_id=stable_object_id(self.task_id, canonical_form),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form=canonical_form,
            payload={
                "expr": canonical_expr,
                "mechanistic_key": _key_to_serializable(key),
            },
            display_text=canonical_expr,
            source_hypothesis_ids=(raw_object.source_hypothesis_id,),
            metadata={
                "evaluator": "hypospace_official",
                "domain": "boolean",
                "observation_set_id": self.task.observation_set_id,
                "mechanistic_opts": dict(self.task.mechanistic_opts),
            },
        )

    # -- deterministic rebuild of a canonical expression from the key --------

    def _rebuild(self, key: tuple) -> str:
        """Rebuild an expression string whose mechanistic key is ``key``.

        AND/OR children are joined infix (associativity flattening makes
        parenthesization irrelevant); a single surviving AND/OR child is
        duplicated so re-parsing reproduces the same key after idempotence.
        Non-atomic children are parenthesized so re-parsing preserves the
        tree shape (``(x OR y) AND z`` must not re-parse as ``x OR (y AND z)``).
        """
        tag = key[0]
        if tag == "VAR":
            return str(key[1])
        if tag == "CONST":  # never valid in-space; defensive
            return "TRUE" if key[1] else "FALSE"
        if tag == "NOT":
            inner = key[1]
            # NOR is stored as its own tag by the official keyer.
            if isinstance(inner, tuple) and inner and inner[0] == "NOR":
                return f"NOR({', '.join(self._rebuild(k) for k in inner[1])})"
            return f"NOT ({self._rebuild(inner)})"
        if tag == "NOR":
            return f"NOR({', '.join(self._rebuild(k) for k in key[1])})"
        kids = [self._rebuild(k) for k in key[1]]
        kids = [k if _is_atomic(k) else f"({k})" for k in kids]
        if tag in ("AND", "OR") and len(kids) == 1:
            return f"{kids[0]} {tag} {kids[0]}"
        if tag in ("AND", "OR", "XOR"):
            return f" {tag} ".join(kids)
        # Unknown op (defensive): serialize the tuple itself.
        return f"{tag}({', '.join(kids)})"

    # -- G_sci ----------------------------------------------------------------

    _OP_COUNTS = ("AND", "OR", "NOT", "XOR", "NOR")

    def _feature(self, obj: GroundedObject) -> np.ndarray:
        key = _key_from_serializable(obj.payload["mechanistic_key"])
        counts = {op: 0 for op in self._OP_COUNTS}
        var_incidence = {v: 0 for v in self.task.variables}
        depth = 0

        def walk(k: tuple, d: int) -> None:
            nonlocal depth
            tag = k[0]
            if tag == "VAR":
                var_incidence[str(k[1])] = var_incidence.get(str(k[1]), 0) + 1
                depth = max(depth, d)
                return
            if tag in counts:
                counts[tag] += 1
            for kid in _children(k):
                walk(kid, d + 1)

        walk(key, 0)
        vec = (
            [float(counts[op]) for op in self._OP_COUNTS]
            + [float(var_incidence[v]) for v in self.task.variables]
            + [float(depth)]
        )
        return np.asarray(vec, dtype=np.float64)

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
                            edge_id=stable_edge_id(a.object_id, b.object_id, "boolean_structure"),
                            source_id=a.object_id,
                            target_id=b.object_id,
                            weight=cos,
                            distance=1.0 - cos,
                            relation_type="boolean_structure",
                            metadata={"outcome_blind": True},
                        )
                    )
        return ScientificGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "boolean", "relation_type": "boolean_structure", "outcome_blind": True},
        )

    # -- G_rev -----------------------------------------------------------------

    def build_revision_graph(self, objects: tuple[GroundedObject, ...]) -> RevisionGraph:
        edges: list[RevisionEdge] = []
        for a in objects:
            for b in objects:
                if a.object_id == b.object_id:
                    continue
                ka = _key_from_serializable(a.payload["mechanistic_key"])
                kb = _key_from_serializable(b.payload["mechanistic_key"])
                edit = _boolean_edit(ka, kb)
                if edit is None:
                    continue
                action, payload = edit
                # Benchmark-validity guarantee: constructed nodes passed the
                # official validator at insertion; audit re-check = in-space
                # (syntax space only; no observation re-evaluation).
                if not official.boolean_in_space(
                    b.payload["expr"], self.task.variables, self.task.operators, self.task.max_depth
                ):
                    continue
                edges.append(
                    RevisionEdge(
                        edge_id=stable_edge_id(a.object_id, b.object_id, action),
                        source_id=a.object_id,
                        target_id=b.object_id,
                        action_type=action,
                        action_description=f"{action}: {payload}",
                        edit_payload=payload,
                        metadata={"domain": "boolean", "outcome_blind": True},
                    )
                )
        return RevisionGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "boolean", "outcome_blind": True},
        )

    # -- evaluation ------------------------------------------------------------

    async def evaluate(self, obj: GroundedObject):
        raise EvaluationNotApplicable(
            "HypoSpace has no scalar empirical utility field; validity is a "
            "deterministic benchmark verdict, not an Observation."
        )

    def render_object(self, obj: GroundedObject) -> str:
        return obj.display_text

    # -- FrontierSemantics (V6.1 SpaceReadout) --------------------------------
    # Descriptor per Hypo问题诊断1: phi(h) = (#ops, depth, variable support,
    # operator family, interaction order, #negations).  Outcome-blind; from
    # canonical key + public task grammar only.

    def descriptor(self, obj: GroundedObject) -> dict[str, Any]:
        key = _key_from_serializable(obj.payload["mechanistic_key"])

        def walk(k: tuple, d: int) -> tuple[int, int, int, set[str], set[str], int]:
            """(n_ops, depth, interaction_order, variables, families, n_neg)."""
            tag = k[0]
            if tag == "VAR":
                return 0, d, 0, {str(k[1])}, set(), 0
            if tag == "CONST":
                return 0, d, 0, set(), set(), 0
            kid_results = [walk(kid, d + 1) for kid in _children(k)]
            n_ops = 1 + sum(r[0] for r in kid_results)
            depth = max((r[1] for r in kid_results), default=d)
            inter = max((r[2] for r in kid_results), default=0)
            vars_ = set().union(*(r[3] for r in kid_results))
            fams = set().union(*(r[4] for r in kid_results))
            n_neg = sum(r[5] for r in kid_results)
            if tag == "NOT":
                n_neg += 1
            elif tag in ("AND", "OR", "XOR"):
                fams.add(tag)
                inter = max(inter, len(_children(k)))
            elif tag == "NOR":
                fams.add("NOR")
            return n_ops, depth, inter, vars_, fams, n_neg

        n_ops, depth, inter, vars_, fams, n_neg = walk(key, 0)
        return {
            "n_ops": n_ops,
            "depth": depth,
            "variable_support": tuple(sorted(vars_)),
            "operator_family": tuple(sorted(fams)),
            "interaction_order": inter,
            "n_negations": n_neg,
        }

    def attribute_domains(self) -> dict[str, tuple[Any, ...]]:
        from itertools import combinations as _combs

        vars_ = self.task.variables
        ops = tuple(sorted(self.task.operators))
        n = len(vars_)
        max_ops = 2 ** (self.task.max_depth + 1)  # loose grammar bound
        support_domain = tuple(sorted((tuple(sorted(c)) for r in range(n + 1) for c in _combs(vars_, r))))
        family_domain = tuple(sorted((tuple(sorted(c)) for r in range(1, len(ops) + 1) for c in _combs(ops, r))))
        return {
            "n_ops": tuple(range(0, max_ops + 1)),
            "depth": tuple(range(0, self.task.max_depth + 1)),
            "variable_support": support_domain,
            "operator_family": family_domain,
            "interaction_order": tuple(range(0, self.task.max_depth + 1)),
            "n_negations": tuple(range(0, max_ops + 1)),
        }

    def revision_hints(self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]) -> tuple[str, ...]:
        hints: list[str] = []
        if to_descriptor.get("operator_family") != from_descriptor.get("operator_family"):
            hints.append("replace_operator")
        if to_descriptor.get("variable_support") != from_descriptor.get("variable_support"):
            hints.append("replace_variable")
        if to_descriptor.get("n_ops", 0) > from_descriptor.get("n_ops", 0):
            hints.append("add_literal")
        if to_descriptor.get("n_ops", 0) < from_descriptor.get("n_ops", 0):
            hints.append("remove_literal")
        if to_descriptor.get("n_negations", 0) != from_descriptor.get("n_negations", 0):
            hints.append("wrap_transform")
        return tuple(dict.fromkeys(hints)) or ("replace_operator",)

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """Readout-v2 feasibility: descriptor cells must be consistent under
        the PUBLIC boolean grammar (tree of ops over vars, depth-limited)."""
        n_ops = int(descriptor["n_ops"])
        depth = int(descriptor["depth"])
        support = descriptor["variable_support"]
        family = descriptor["operator_family"]
        inter = int(descriptor["interaction_order"])
        n_neg = int(descriptor["n_negations"])
        if not support or not set(support).issubset(set(self.task.variables)):
            return False
        if not set(family).issubset(set(self.task.operators)):
            return False
        if depth < 0 or depth > self.task.max_depth:
            return False
        if depth == 0:
            # a bare variable: exactly one variable, no ops, no negations
            return n_ops == 0 and len(support) == 1 and not family and n_neg == 0 and inter == 0
        # depth >= 1 requires at least one operator
        if n_ops < 1 or not family:
            return False
        # tree bounds: n_ops >= depth; n_ops <= 2**(depth+1); negations <= n_ops
        if n_ops < depth or n_ops > 2 ** (depth + 1):
            return False
        if n_neg > n_ops:
            return False
        # interaction order (max operand count of a binary op) bounded by tree
        if inter > max(2, n_ops):
            return False
        return True

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        """Domain-flavoured directive (Hypo问题诊断1 §23 example style)."""
        parts = []
        support = descriptor["variable_support"]
        family = descriptor["operator_family"]
        if support:
            parts.append(f"using variables {', '.join(support)}")
        if family:
            parts.append(f"with operator(s) {', '.join(family)}")
        if descriptor["n_negations"]:
            parts.append(f"with {descriptor['n_negations']} negation(s)")
        if descriptor["depth"]:
            parts.append(f"depth about {descriptor['depth']}")
        if descriptor["interaction_order"]:
            parts.append(f"interaction order about {descriptor['interaction_order']}")
        base = "Produce a Boolean hypothesis " + (", ".join(parts) if parts else "with a different structure")
        return base + ", satisfying the observations, with a canonical form distinct from all anchors"


# ---------------------------------------------------------------------------
# Key (de)serialization helpers
#
# The official mechanistic key has an irregular shape: ``VAR``/``CONST`` are
# leaves; ``NOT`` wraps a SINGLE child (``("NOT", child)``); ``NOR`` and
# ``AND``/``OR``/``XOR`` wrap a TUPLE of children.  ``_children`` normalizes
# traversal across those shapes.
# ---------------------------------------------------------------------------


def _is_leaf(key: tuple) -> bool:
    return key[0] in ("VAR", "CONST")


def _children(key: tuple) -> tuple:
    tag = key[0]
    if _is_leaf(key):
        return ()
    if tag == "NOT":
        return (key[1],)
    return tuple(key[1])


def _key_to_serializable(key: tuple) -> list:
    tag = key[0]
    if tag == "VAR":
        return [tag, str(key[1])]
    if tag == "CONST":
        return [tag, bool(key[1])]
    if tag == "NOT":
        return [tag, _key_to_serializable(key[1])]
    return [tag, [_key_to_serializable(k) for k in key[1]]]


def _key_from_serializable(data) -> tuple:
    tag = data[0]
    if tag == "VAR":
        return (tag, data[1])
    if tag == "CONST":
        return (tag, data[1])
    if tag == "NOT":
        return (tag, _key_from_serializable(data[1]))
    return (tag, tuple(_key_from_serializable(k) for k in data[1]))


# ---------------------------------------------------------------------------
# Interpretable structural edits between two mechanistic-key trees.
# Returns (action_type, payload) or None.  Deterministic and conservative.
# ---------------------------------------------------------------------------


def _boolean_edit(ka: tuple, kb: tuple) -> tuple[str, dict] | None:
    if ka == kb:
        return None
    # wrap_transform: b = NOT(a) or a = NOT(b)
    if kb[0] == "NOT" and kb[1] == ka:
        return ("wrap_transform", {"op": "add_NOT"})
    if ka[0] == "NOT" and ka[1] == kb:
        return ("wrap_transform", {"op": "remove_NOT"})

    # replace_variable: substitute a single variable for another, everywhere
    sub = _variable_substitution(ka, kb)
    if sub is not None:
        return ("replace_variable", {"from": sub[0], "to": sub[1]})

    # replace_operator: same shape, one operator node differs
    op_edit = _operator_edit(ka, kb)
    if op_edit is not None:
        return ("replace_operator", op_edit)

    # add_literal / remove_literal: same AND/OR/XOR root, kid set differs by one
    lit = _literal_edit(ka, kb)
    if lit is not None:
        return lit
    return None


def _variable_substitution(ka: tuple, kb: tuple) -> tuple[str, str] | None:
    """Find (x, y) such that applying x->y to ka yields kb."""
    vars_a = _collect_vars(ka)
    vars_b = _collect_vars(kb)
    if not vars_a or not vars_b:
        return None
    # All occurrence substitution must map exactly one variable.
    for x in sorted(vars_a):
        for y in sorted(vars_b):
            if x == y:
                continue
            if _subst(ka, x, y) == kb:
                return (x, y)
    return None


def _subst(k: tuple, x: str, y: str) -> tuple:
    tag = k[0]
    if tag == "VAR":
        return ("VAR", y if k[1] == x else k[1])
    if tag == "NOT":
        return (tag, _subst(k[1], x, y))
    return (tag, tuple(_subst(kid, x, y) for kid in k[1]))


def _collect_vars(k: tuple) -> set[str]:
    tag = k[0]
    if tag == "VAR":
        return {str(k[1])}
    out: set[str] = set()
    for kid in _children(k):
        out |= _collect_vars(kid)
    return out


def _is_atomic(k: str) -> bool:
    """A rebuild that needs no parentheses (bare variable name)."""
    return bool(__import__("re").fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k))


def _operator_edit(ka: tuple, kb: tuple) -> dict | None:
    """Exactly one operator node differs; leaf variables identical.

    Uses an OP-placeholder structural signature: if the two trees have the
    same shape and the same variable leaves, and exactly one operator tag
    differs, the edit is a single ``replace_operator``.
    """
    if _signature(ka) != _signature(kb):
        return None
    diffs: list[tuple[str, str]] = []

    def walk(a: tuple, b: tuple) -> None:
        if _is_leaf(a) or _is_leaf(b):
            return
        if a[0] != b[0]:
            diffs.append((a[0], b[0]))
        for x, y in zip(_children(a), _children(b)):
            walk(x, y)

    walk(ka, kb)
    if len(diffs) == 1:
        return {"from_op": diffs[0][0], "to_op": diffs[0][1]}
    return None


def _signature(k: tuple) -> tuple:
    tag = k[0]
    if tag == "VAR":
        return ("VAR", k[1])
    return ("OP", tuple(_signature(kid) for kid in _children(k)))


def _literal_edit(ka: tuple, kb: tuple) -> tuple[str, dict] | None:
    """Same associative op root; kid multiset differs by exactly one element."""
    if ka[0] != kb[0] or ka[0] not in ("AND", "OR", "XOR"):
        return None
    if ka[0] in ("AND", "OR") and len(ka[1]) == len(kb[1]):
        return None
    if ka[0] == "XOR" and len(ka[1]) != len(kb[1]) - 1 and len(ka[1]) != len(kb[1]) + 1:
        return None
    from collections import Counter

    ca, cb = Counter(ka[1]), Counter(kb[1])
    added = list((cb - ca).elements())
    removed = list((ca - cb).elements())
    if len(added) == 1 and not removed:
        return ("add_literal", {"op": ka[0], "literal": _key_to_serializable(added[0])})
    if len(removed) == 1 and not added:
        return ("remove_literal", {"op": ka[0], "literal": _key_to_serializable(removed[0])})
    return None
