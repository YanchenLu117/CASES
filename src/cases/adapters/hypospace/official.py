"""Pinned official HypoSpace repository bridge (Agent A).

HypoSpace's *evaluator* semantics (LLM-response parsing, validation, and
exact-space enumeration) are defined by the official repository pinned in
``external/repos.lock.yaml``.  This module is the ONLY place in the CASES tree
that imports from the official checkout, and adapters reach official code
exclusively through here — the adapters never re-implement evaluator logic.

Pinning (merge gate G1):
  * ``scripts/external_lock.py --verify`` compares the locked commit with
    the checkout HEAD.
  * ``pinned_commit()`` re-reads ``external/repos.lock.yaml`` at import time so
    the running process always reports the commit it actually executed.

Leakage rule: nothing in this module (or anywhere in the adapter) ever exposes
``ground_truth_*`` fields to prompts or state — ground truth is loaded by
``tasks.py`` only through a GT-stripping loader and handed to metrics directly.
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable

_HYPO_ENV_VAR = "CASES_HYPOSPACE_REPO"
_LOCK_PATH = "external/repos.lock.yaml"
_LOCK_KEY = "hypospace"

# Domain subdirectories of the official repo, in import order (3d must come
# last: it imports ``modules.llm_interface`` which needs requests).
_DOMAIN_DIRS = ("causal", "boolean", "3d")


class OfficialRepoError(RuntimeError):
    """The pinned official HypoSpace checkout is missing or unreadable."""


def repo_root() -> Path:
    """Resolve the official HypoSpace checkout root.

    Defaults to ``<cases-package>/../../external/repos/hypospace`` relative to
    this file (i.e. the workspace layout ``<root>/external/repos/hypospace``).
    ``CASES_HYPOSPACE_REPO`` overrides it (used by tests / CI sandboxes).
    """
    import os

    override = os.environ.get(_HYPO_ENV_VAR)
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    # here = <root>/src/cases/adapters/hypospace/official.py
    root = here.parents[4]
    return root / "external" / "repos" / "hypospace"


def pinned_commit() -> str:
    """Pin from external/repos.lock.yaml: git commit when the checkout has its
    own .git, else the content fingerprint (snapshot restored from archive)."""
    here = Path(__file__).resolve()
    root = here.parents[4]
    lock = root / _LOCK_PATH
    if not lock.exists():
        raise OfficialRepoError(f"lock manifest not found: {lock}")
    import yaml

    data = yaml.safe_load(lock.read_text(encoding="utf-8"))
    entry = data.get("repos", {}).get(_LOCK_KEY) or data.get(_LOCK_KEY, {})
    commit = entry.get("commit") or entry.get("fingerprint")
    if not commit:
        raise OfficialRepoError(f"no commit pinned for {_LOCK_KEY!r} in {lock}")
    return str(commit)


def checkout_fingerprint() -> str:
    """Content fingerprint for snapshot-only checkouts (no .git): sha256 over
    sorted relative paths + sizes, truncated to 16 hex (external_lock.py)."""
    root = repo_root()
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts:
            h.update(f"{p.relative_to(root)}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def checkout_commit() -> str:
    """Actual HEAD of the official checkout (best-effort)."""
    root = repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        git_dir = Path(out.stdout.strip())
        if root.resolve() not in git_dir.parents and git_dir.parent != root.resolve():
            return checkout_fingerprint()  # nested dir without its own .git
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return head.stdout.strip()
    except Exception:  # pragma: no cover - git unavailable
        return checkout_fingerprint()


def verify_pin() -> dict[str, str]:
    """Return {locked, checkout} and raise if they disagree (G1 gate)."""
    locked = pinned_commit()
    checkout = checkout_commit()
    if checkout != "unknown" and locked != checkout:
        raise OfficialRepoError(
            f"HypoSpace pin mismatch: lock={locked} checkout={checkout}. "
            "Run scripts/external_lock.py --verify / scripts/setup_external.py."
        )
    return {"locked": locked, "checkout": checkout}


def _import_domains() -> None:
    """Insert official domain dirs on sys.path and import the modules used by
    the adapters.  Cached; idempotent."""
    root = repo_root()
    if not root.is_dir():
        raise OfficialRepoError(
            f"official HypoSpace checkout not found at {root}. "
            "Clone it with scripts/setup_external.py (pinned commit)."
        )
    import sys

    for d in _DOMAIN_DIRS:
        domain_dir = root / d
        if not domain_dir.is_dir():
            raise OfficialRepoError(f"official repo missing domain dir: {domain_dir}")
        if str(domain_dir) not in sys.path:
            sys.path.insert(0, str(domain_dir))


@functools.lru_cache(maxsize=1)
def _modules() -> dict[str, Any]:
    _import_domains()
    import boolean_dataset  # noqa: F401
    import boolean_benchmark  # noqa: F401
    import generate_causal_dataset  # noqa: F401
    import generate_3d_dataset_complete  # noqa: F401
    import modules.models  # noqa: F401
    import run_3d_benchmark  # noqa: F401
    import run_causal_benchmark  # noqa: F401

    return {
        "causal_models": modules.models,
        "causal_gen": generate_causal_dataset,
        "causal_bench": run_causal_benchmark,
        "boolean_dataset": boolean_dataset,
        "boolean_bench": boolean_benchmark,
        "voxel_gen": generate_3d_dataset_complete,
        "voxel_bench": run_3d_benchmark,
    }


def pin_metadata() -> dict[str, str]:
    """{locked, checkout} recorded into every result/artifact (audit)."""
    return verify_pin()


# --------------------------------------------------------------------------
# Official classes (for payloads and evaluation)
# --------------------------------------------------------------------------


def CausalGraph(nodes, edges):  # noqa: N802 - official class name
    return _modules()["causal_models"].CausalGraph(nodes=nodes, edges=edges)


def causal_graph_from_dict(data: dict[str, Any]) -> Any:
    """Official ``CausalGraph.from_dict`` (dataset JSON -> object)."""
    return _modules()["causal_models"].CausalGraph.from_dict(data)


def PerturbationObservation(perturbed_node, effects):  # noqa: N802
    return _modules()["causal_gen"].PerturbationObservation(perturbed_node, effects)


def BooleanExpression(formula, variables, operators):  # noqa: N802
    return _modules()["boolean_dataset"].BooleanExpression(formula, variables, operators)


def BooleanObservation(inputs, output):  # noqa: N802
    return _modules()["boolean_dataset"].BooleanObservation(inputs, output)


def Structure3D(layers):  # noqa: N802 - official class name
    return _modules()["voxel_bench"].Structure3D(layers)


def make_structure3d(layers) -> Any:
    """Official ``Structure3D`` from numpy/list/compact-string layers.

    Accepts the two official dataset formats:
      * list of 2D arrays / nested lists (one per layer);
      * list of compact strings (official ``to_dict`` format, one string per
        layer, e.g. ``"100000000"``).
    The official constructor iterates layers with ``if layer:``, which raises
    on numpy arrays — convert to plain lists first.
    """
    import numpy as np

    converted = []
    for layer in layers:
        if isinstance(layer, str):
            n = len(layer)
            grid = int(n ** 0.5)
            rows = [[int(c) for c in layer[i * grid : (i + 1) * grid]] for i in range(grid)]
            converted.append(rows)
        else:
            converted.append(np.asarray(layer, dtype=np.int64).tolist())
    return Structure3D(converted)


# --------------------------------------------------------------------------
# Official evaluator entry points (parse + validate), bridged via __new__ so
# that no benchmark class constructor (which expects dataset files) runs.
# --------------------------------------------------------------------------


def causal_parser(nodes: tuple[str, ...], max_edges: int | None) -> Callable[[str], Any | None]:
    """Official causal ``parse_llm_response`` bound to task node/max_edges."""
    bench = _modules()["causal_bench"].CausalBenchmarkEnhanced.__new__(
        _modules()["causal_bench"].CausalBenchmarkEnhanced
    )
    bench.nodes = list(nodes)
    bench.max_edges = max_edges
    return bench.parse_llm_response


def boolean_parser() -> Callable[[str], str | None]:
    """Official boolean ``parse_llm_response`` (stateless)."""
    bench = _modules()["boolean_bench"].BooleanBenchmarkRefined.__new__(
        _modules()["boolean_bench"].BooleanBenchmarkRefined
    )
    return bench.parse_llm_response


def voxel_parser() -> Callable[[str], Any | None]:
    """Official 3D ``parse_llm_response`` (stateless)."""
    bench = _modules()["voxel_bench"].Benchmark3D.__new__(_modules()["voxel_bench"].Benchmark3D)
    return bench.parse_llm_response


def validate_causal(
    graph: Any,
    observations: tuple[dict[str, Any], ...],
) -> bool:
    """Official causal validation: exact perturbation-effect equality for every
    observation (descendants -> 1, perturbed -> 0, others -> 0)."""
    gen = _modules()["causal_gen"].CausalDatasetGenerator
    for obs in observations:
        official_obs = PerturbationObservation(
            obs["perturbed_node"], obs["effects"]
        )
        predicted = gen.get_perturbation_effects(graph, obs["perturbed_node"])
        if predicted.effects != official_obs.effects:
            return False
    return True


def validate_boolean(
    expr_str: str,
    variables: tuple[str, ...],
    operators: frozenset[str],
    max_depth: int,
    observations: tuple[dict[str, Any], ...],
) -> tuple[bool, dict[str, Any] | None]:
    """Official boolean validation (space constraints + observation
    consistency).  Returns (valid, truth_table_or_None)."""
    bench = _modules()["boolean_bench"].BooleanBenchmarkRefined.__new__(
        _modules()["boolean_bench"].BooleanBenchmarkRefined
    )
    bench.variables = list(variables)
    bench.operators = set(operators)
    bench.max_depth = max_depth
    obs = [BooleanObservation(o["inputs"], o["output"]) for o in observations]
    return bench.validate_expression(expr_str, obs)


def boolean_in_space(
    expr_str: str,
    variables: tuple[str, ...],
    operators: frozenset[str],
    max_depth: int,
) -> bool:
    """Official ``_in_space`` (variables / operators / no constants / depth)."""
    bench = _modules()["boolean_bench"].BooleanBenchmarkRefined.__new__(
        _modules()["boolean_bench"].BooleanBenchmarkRefined
    )
    bench.variables = list(variables)
    bench.operators = set(operators)
    bench.max_depth = max_depth
    return bench._in_space(expr_str)


def validate_voxel(structure: Any, observation: str) -> bool:
    """Official 3D validation: top-view equality + per-layer support."""
    bench = _modules()["voxel_bench"].Benchmark3D.__new__(_modules()["voxel_bench"].Benchmark3D)
    return bench.validate_structure_matches_observations(structure, observation)


# --------------------------------------------------------------------------
# Official exact-space enumeration (EVALUATION-ONLY; never prompt/state).
# --------------------------------------------------------------------------


def enumerate_causal_space(
    nodes: tuple[str, ...],
    max_edges: int | None,
    observations: tuple[dict[str, Any], ...],
) -> tuple[Any, ...]:
    """All DAGs over ``nodes`` (≤max_edges) consistent with the observations.

    Mirrors the official dataset generator's admissible-set construction.
    """
    gen = _modules()["causal_gen"].CausalDatasetGenerator
    all_dags = gen.generate_all_dags(list(nodes), max_edges=max_edges)
    return tuple(g for g in all_dags if validate_causal(g, observations))


def enumerate_boolean_space(
    variables: tuple[str, ...],
    operators: frozenset[str],
    max_depth: int,
    mechanistic_opts: dict[str, Any],
    observations: tuple[dict[str, Any], ...],
) -> tuple[Any, ...]:
    """All in-space expressions consistent with the observations, deduplicated
    by the official mechanistic key (== official admissible set)."""
    game = _modules()["boolean_dataset"].BooleanDiscoveryGame
    all_exprs = game.generate_all_expressions(
        list(variables), set(operators), max_depth, mechanistic_opts
    )
    obs = [BooleanObservation(o["inputs"], o["output"]) for o in observations]
    return tuple(game.find_all_compatible_expressions(obs, list(variables), set(operators), max_depth, mechanistic_opts, all_exprs))


def enumerate_voxel_space(
    top_view: Any,
    max_height: int,
) -> tuple[Any, ...]:
    """All structures matching the top view with heights 1..max_height."""
    from itertools import product  # noqa: F401 - official module needs none

    gen = _modules()["voxel_gen"]
    return tuple(gen.enumerate_all_structures_by_heights(top_view, max_height))


def causal_hash(graph: Any) -> str:
    """Official uniqueness hash for a causal graph."""
    return graph.get_hash()


def boolean_mechanistic_key(expr_str: str, variables, operators, mechanistic_opts) -> tuple | None:
    try:
        expr = BooleanExpression(expr_str, variables, operators)
        return expr.mechanistic_key(**mechanistic_opts)
    except Exception:
        return None


def voxel_hash(structure: Any) -> str:
    """Official uniqueness hash for a voxel structure (normalized)."""
    return structure.get_hash()
