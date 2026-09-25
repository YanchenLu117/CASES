"""HypoSpace task specifications and GT-stripping dataset loaders (Agent A).

The official dataset JSONs store observations *next to* their ground truth
(``ground_truth_graphs`` / ``ground_truth_expressions`` /
``ground_truth_structures``).  These loaders strip every ground-truth field at
load time: a :class:`CausalTask` / :class:`BooleanTask` / :class:`VoxelTask`
contains observations and task constraints ONLY.  Ground truth is returned as a
separate, explicitly-requested value and is consumed exclusively by the
evaluation-only metrics path (``cases.adapters.hypospace.metrics``) and tests —
never by prompts, state, or graph construction (leakage audit, gate G4).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Task specifications (observations + constraints only — NO ground truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CausalTask:
    dataset: str
    observation_set_id: str
    nodes: tuple[str, ...]
    max_edges: int | None
    observations: tuple[dict[str, Any], ...]  # {"perturbed_node", "effects"}
    n_observations: int
    n_compatible_graphs: int | None = None  # info only (from official metadata)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"hypospace_causal:{self.dataset}:{self.observation_set_id}"


@dataclass(frozen=True, slots=True)
class BooleanTask:
    dataset: str
    observation_set_id: str
    variables: tuple[str, ...]
    operators: frozenset[str]
    max_depth: int
    mechanistic_opts: Mapping[str, Any]
    observations: tuple[dict[str, Any], ...]  # {"inputs": {...}, "output": int}
    n_observations: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"hypospace_boolean:{self.dataset}:{self.observation_set_id}"


@dataclass(frozen=True, slots=True)
class VoxelTask:
    dataset: str
    observation_id: str
    grid_size: int
    max_height: int
    observation: str  # compact top-view string, e.g. "101000101"
    n_compatible_structures: int | None = None  # info only
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"hypospace_voxel3d:{self.dataset}:{self.observation_id}"


# ---------------------------------------------------------------------------
# GT-stripping loaders
# ---------------------------------------------------------------------------

_GT_KEYS = {
    "causal": ("ground_truth_graphs",),
    "boolean": ("ground_truth_expressions",),
    "voxel3d": ("ground_truth_structures",),
}


def _strip_gt(obs_set: dict[str, Any], domain: str) -> dict[str, Any]:
    out = dict(obs_set)
    for key in _GT_KEYS[domain]:
        out.pop(key, None)
    return out


def load_causal_dataset(path: str | Path) -> tuple[tuple[CausalTask, ...], dict[str, list[Any]]]:
    """Load an official causal dataset JSON.

    Returns ``(tasks, gt)`` where ``gt[observation_set_id]`` holds the official
    ``CausalGraph`` ground-truth objects.  Tasks never contain GT.
    """
    from cases.data.lazy import json_cached  # V9 infra: parse-once cache for 70-89MB files

    data = json_cached(path)
    metadata = data.get("metadata", {})
    nodes = tuple(metadata.get("nodes", []))
    max_edges = metadata.get("max_edges")
    datasets = _flatten_datasets(data)
    tasks: list[CausalTask] = []
    gt: dict[str, list[Any]] = {}
    for obs_set in datasets:
        oid = obs_set.get("observation_set_id", obs_set.get("id", "unknown"))
        observations = tuple(obs_set.get("observations", []))
        from . import official

        gt_objs = [official.causal_graph_from_dict(g) for g in obs_set.get("ground_truth_graphs", [])]
        gt[oid] = gt_objs
        tasks.append(
            CausalTask(
                dataset=Path(path).stem,
                observation_set_id=str(oid),
                nodes=nodes,
                max_edges=max_edges,
                observations=observations,
                n_observations=len(observations),
                n_compatible_graphs=obs_set.get("n_compatible_graphs"),
                metadata={"source": str(path), "n_compatible_graphs": obs_set.get("n_compatible_graphs")},
            )
        )
    return tuple(tasks), gt


def load_boolean_dataset(path: str | Path) -> tuple[tuple[BooleanTask, ...], dict[str, list[Any]]]:
    """Load an official boolean dataset JSON -> (tasks, gt) with GT stripped."""
    from cases.data.lazy import json_cached  # V9 infra: parse-once cache for 70-89MB files

    data = json_cached(path)
    metadata = data.get("metadata", {})
    variables = tuple(metadata.get("variables", []))
    operators = frozenset(metadata.get("operators", []))
    max_depth = int(metadata.get("max_depth", 2))
    mech = dict(metadata.get("mechanistic_opts", {}))
    datasets = _flatten_datasets(data)
    tasks: list[BooleanTask] = []
    gt: dict[str, list[Any]] = {}
    for obs_set in datasets:
        oid = obs_set.get("observation_set_id", obs_set.get("id", "unknown"))
        gt[oid] = list(obs_set.get("ground_truth_expressions", []))
        tasks.append(
            BooleanTask(
                dataset=Path(path).stem,
                observation_set_id=str(oid),
                variables=variables,
                operators=operators,
                max_depth=max_depth,
                mechanistic_opts=mech,
                observations=tuple(obs_set.get("observations", [])),
                n_observations=len(obs_set.get("observations", [])),
                metadata={"source": str(path)},
            )
        )
    return tuple(tasks), gt


def load_voxel_dataset(path: str | Path) -> tuple[tuple[VoxelTask, ...], dict[str, list[Any]]]:
    """Load an official 3D dataset JSON -> (tasks, gt) with GT stripped."""
    from cases.data.lazy import json_cached  # V9 infra: parse-once cache for 70-89MB files

    data = json_cached(path)
    metadata = data.get("metadata", {})
    grid_size = int(metadata.get("grid_size", 3))
    max_height = int(metadata.get("max_height", 3))
    tasks: list[VoxelTask] = []
    gt: dict[str, list[Any]] = {}
    for obs_set in data.get("observation_sets", []):
        oid = str(obs_set.get("observation_id", "unknown"))
        gt[oid] = list(obs_set.get("ground_truth_structures", []))
        tasks.append(
            VoxelTask(
                dataset=Path(path).stem,
                observation_id=oid,
                grid_size=grid_size,
                max_height=max_height,
                observation=str(obs_set.get("observation", "")),
                n_compatible_structures=obs_set.get("n_compatible_structures"),
                metadata={"source": str(path)},
            )
        )
    return tuple(tasks), gt


def load_dataset(path: str | Path, domain: str):
    if domain == "causal":
        return load_causal_dataset(path)
    if domain == "boolean":
        return load_boolean_dataset(path)
    if domain == "voxel3d":
        return load_voxel_dataset(path)
    raise ValueError(f"unknown domain: {domain}")


def _flatten_datasets(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "datasets" in data:
        return list(data["datasets"])
    if "datasets_by_n_observations" in data:
        out: list[dict[str, Any]] = []
        for datasets in data["datasets_by_n_observations"].values():
            out.extend(datasets)
        return out
    if "sampled_datasets" in data:
        return list(data["sampled_datasets"])
    raise ValueError("dataset JSON has no recognized datasets key")


def sample_tasks(tasks: tuple[Any, ...], n: int, seed: int) -> tuple[Any, ...]:
    """Deterministic subset of task instances (dev/full separation helper)."""
    rng = random.Random(seed)
    if n >= len(tasks):
        return tasks
    idx = rng.sample(range(len(tasks)), n)
    return tuple(tasks[i] for i in sorted(idx))
