"""HypoSpace metrics — official-parity Validity / Uniqueness / Recovery plus
CASES structural coverage (Agent A).

Counting follows the official runners exactly:

  * causal  : valid_rate = n_valid/n_queries; novelty = unique parsed graphs
              (official hash) / n_queries; recovery = unique *valid* graph
              hashes inside the GT hash set / |GT|.
  * boolean : valid_rate = n_valid/n_queries; novelty = unique *in-space*
              expressions (mechanistic key) / n_queries; recovery = GT
              mechanistic keys matched by unique valid expression keys / |GT|.
  * voxel   : valid_rate = n_valid/n_queries; novelty = unique parsed
              structures (official hash) / n_queries; recovery = valid
              structure hashes inside GT hash set / |GT|.

CASES-specific additions (documented in the adapter README):
  * Recovery@Budget     : cumulative recovery after each query (checkpoints at
                          25/50/75/100% of the generation budget).
  * structural coverage : fraction of connected components of the *full-space*
                          G_sci (built over the enumerated admissible set,
                          evaluation-only) that contain at least one
                          constructed object; plus region-size profile.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import exact_space, official
from .tasks import BooleanTask, CausalTask, VoxelTask


def gt_keys(domain: str, gt_objects: Sequence[Any], task=None) -> list[str]:
    """Official keys of the ground-truth admissible set (evaluation-only).

    GT objects come from the official dataset JSONs in three formats:
    causal -> official ``CausalGraph`` (from_dict); boolean -> dicts carrying
    ``mechanistic_key`` (official ``str(tuple)``, parsed like the official
    runner); voxel3d -> dicts carrying ``layers`` (official to_dict format).
    """
    if domain == "causal":
        return [official.causal_hash(g) for g in gt_objects]
    if domain == "boolean":
        import ast

        keys = []
        for g in gt_objects:
            raw = g["mechanistic_key"] if isinstance(g, dict) else g.mechanistic_key()
            if isinstance(raw, str):
                raw = ast.literal_eval(raw)
            keys.append(repr(raw))
        return keys
    if domain == "voxel3d":
        return [
            official.voxel_hash(
                official.make_structure3d(g["layers"])
            )
            for g in gt_objects
        ]
    raise ValueError(f"unknown domain: {domain}")


def _constructed_key(domain: str, grounded: Any, task) -> str:
    """Official key of a constructed CASES GroundedObject (evaluation-only)."""
    if domain == "causal":
        graph = official.CausalGraph(grounded.payload["nodes"], grounded.payload["edges"])
        return official.causal_hash(graph)
    if domain == "voxel3d":
        layers = [np.asarray(layer, dtype=np.int64) for layer in grounded.payload["tensor"]]
        struct = official.make_structure3d(layers)
        return official.voxel_hash(struct)
    if domain == "boolean":
        from .boolean import _key_from_serializable

        return _key_str(_key_from_serializable(grounded.payload["mechanistic_key"]))
    raise ValueError(f"unknown domain: {domain}")


def _key_str(key: tuple) -> str:
    return repr(key)


def compute_instance_metrics(
    *,
    domain: str,
    task,
    gt_keys_list: Sequence[str],
    queries: Sequence[Mapping[str, Any]],
    budget_checkpoints: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Metrics for one task instance.

    ``queries``: one record per generation query with keys
        ``parse_ok`` (bool), ``in_space`` (bool|None), ``valid`` (bool),
        ``official_key`` (str|None), ``object_id`` (str|None).
    """
    n_queries = len(queries)
    gt_set = set(gt_keys_list)

    n_parse = sum(1 for q in queries if q["parse_ok"])
    n_in_space = sum(1 for q in queries if q.get("in_space"))
    n_valid = sum(1 for q in queries if q["valid"])

    # official uniqueness sets
    unique_all: set[str] = set()
    unique_valid: set[str] = set()
    for q in queries:
        k = q["official_key"]
        if k is None:
            continue
        if domain == "boolean":
            if q.get("in_space"):
                unique_all.add(k)
        else:
            unique_all.add(k)
        if q["valid"]:
            unique_valid.add(k)

    recovered = gt_set & unique_valid
    recovery_rate = len(recovered) / len(gt_set) if gt_set else 0.0

    # Recovery@Budget: cumulative recovery after each query
    cumulative: list[float] = []
    seen_valid: set[str] = set()
    for q in queries:
        k = q["official_key"]
        if q["valid"] and k is not None:
            seen_valid.add(k)
        cumulative.append(len(gt_set & seen_valid) / len(gt_set) if gt_set else 0.0)

    # RR-AUC: area under recovery vs normalized query index (trapezoid),
    # in [0, 1] — the preregistered primary endpoint (Hypo问题诊断1 §statistics).
    rr_auc = 0.0
    if len(cumulative) > 1:
        xs = np.linspace(0.0, 1.0, len(cumulative))
        rr_auc = float(np.trapezoid(cumulative, xs) if hasattr(np, "trapezoid") else np.trapz(cumulative, xs))

    checkpoints = {}
    for frac in budget_checkpoints:
        idx = min(n_queries - 1, max(0, int(round(frac * n_queries)) - 1)) if n_queries else 0
        checkpoints[f"recovery_at_{int(frac * 100)}pct"] = (
            cumulative[idx] if n_queries else 0.0
        )

    return {
        "n_queries": n_queries,
        "n_ground_truths": len(gt_set),
        "n_parse_success": n_parse,
        "parse_success_rate": n_parse / n_queries if n_queries else 0.0,
        "in_space_rate": n_in_space / n_queries if n_queries else 0.0,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_queries if n_queries else 0.0,
        "n_unique_all": len(unique_all),
        "uniqueness_rate": len(unique_all) / n_queries if n_queries else 0.0,
        "n_unique_valid": len(unique_valid),
        "n_recovered": len(recovered),
        "recovery_rate": recovery_rate,
        "rr_auc": rr_auc,
        "recovery_curve": cumulative,
        **checkpoints,
    }


def structural_coverage(
    *,
    domain: str,
    task,
    adapter,
    constructed: Sequence[Any],
) -> dict[str, Any]:
    """Fraction of full-space G_sci components hit by constructed objects.

    Evaluation-only: the full space is the enumerated admissible set.  For
    large admissible sets this is expensive; callers gate it via config
    (``metrics.structural_coverage``).
    """
    from .voxel3d import _parse_top_view

    if domain == "causal":
        space = exact_space.causal_solution_space(task)
        raws = [
            {"edges": sorted(g.edges), "nodes": sorted(task.nodes), "n_edges": len(g.edges)}
            for g in space
        ]
    elif domain == "boolean":
        space = exact_space.boolean_solution_space(task)
        raws = [{"expr": e.formula} for e in space]
    elif domain == "voxel3d":
        space = exact_space.voxel_solution_space(task)
        raws = [{"layers": [np.asarray(layer, dtype=np.int64) for layer in s.layers]} for s in space]
    else:
        raise ValueError(domain)

    from ...core.types import RawScientificObject

    grounded = []
    for i, raw in enumerate(raws):
        obj = adapter.canonicalize(
            RawScientificObject(
                task_id=adapter.task_id,
                object_type=adapter.object_type,
                payload=raw,
                source_hypothesis_id=f"space-{domain}-{i}",
            )
        )
        grounded.append(obj)

    if not grounded:
        return {"n_components": 0, "components_hit": 0, "component_coverage": 0.0,
                "space_size": 0, "component_sizes": []}

    sci = adapter.build_scientific_graph(tuple(grounded))
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(sci.node_ids)
    for e in sci.edges:
        g.add_edge(e.source_id, e.target_id)
    components = list(nx.connected_components(g))

    constructed_ids = {c.object_id for c in constructed}
    hit = sum(1 for comp in components if comp & constructed_ids)

    return {
        "n_components": len(components),
        "components_hit": hit,
        "component_coverage": hit / len(components) if components else 0.0,
        "space_size": len(grounded),
        "component_sizes": sorted((len(c) for c in components), reverse=True)[:50],
    }
