"""Evaluation-only exact solution-space enumeration (Agent A).

The full admissible set (all solutions consistent with the observations) is
*evaluation-only ground truth*: it is used exclusively by the metrics path and
tests — never by prompts, never by state, never by graph construction
(agent-prompt rule 6, leakage gate G4).

Enumeration executes the official repo generators through
``cases.adapters.hypospace.official`` so the admissible set is byte-identical to
the official datasets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import official
from .tasks import BooleanTask, CausalTask, VoxelTask

__all__ = [
    "causal_solution_space",
    "boolean_solution_space",
    "voxel_solution_space",
    "solution_space_sizes",
    "official_key",
]


def causal_solution_space(task: CausalTask) -> tuple[Any, ...]:
    """All DAGs consistent with the task observations (official enumeration)."""
    return official.enumerate_causal_space(task.nodes, task.max_edges, task.observations)


def boolean_solution_space(task: BooleanTask) -> tuple[Any, ...]:
    """All in-space expressions consistent with the observations (official
    enumeration, deduplicated by the official mechanistic key)."""
    return official.enumerate_boolean_space(
        task.variables,
        task.operators,
        task.max_depth,
        dict(task.mechanistic_opts),
        task.observations,
    )


def voxel_solution_space(task: VoxelTask) -> tuple[Any, ...]:
    """All structures matching the observed top view (official enumeration)."""
    from .voxel3d import _parse_top_view

    top_view = _parse_top_view(task.observation, task.grid_size)
    return official.enumerate_voxel_space(top_view, task.max_height)


def solution_space_sizes(task) -> int:
    """Admissible-set size (diagnostic; evaluation-only)."""
    if isinstance(task, CausalTask):
        return len(causal_solution_space(task))
    if isinstance(task, BooleanTask):
        return len(boolean_solution_space(task))
    if isinstance(task, VoxelTask):
        return len(voxel_solution_space(task))
    raise TypeError(f"unknown task type: {type(task)}")


def official_key(domain: str, obj: Any, task=None) -> str:
    """Official uniqueness key of a solution-space object:
    causal/voxel -> official hash; boolean -> mechanistic-key serialization."""
    if domain == "causal":
        return official.causal_hash(obj)
    if domain == "voxel3d":
        # Enumeration yields the generator's Structure3D; wrap into the
        # benchmark-runner Structure3D (official get_hash semantics).
        wrapped = official.make_structure3d(obj.layers)
        return official.voxel_hash(wrapped)
    if domain == "boolean":
        expr = official.BooleanExpression(
            obj.formula, task.variables, task.operators
        )
        key = expr.mechanistic_key(**dict(task.mechanistic_opts))
        return _key_str(key)
    raise ValueError(f"unknown domain: {domain}")


def _key_str(key: tuple) -> str:
    return repr(key)
