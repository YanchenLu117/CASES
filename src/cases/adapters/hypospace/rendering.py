"""HypoSpace mechanism state renderers (Agent A).

Prompt construction mirrors the official ``create_prompt`` for each domain
(observations + constraints + output format), which every mechanism shares.
The *state block* (prior-information) differs per mechanism and implements the
five HypoSpace mechanism conditions:

  independent            : no state block (each query independent)
  history                : prior proposals in language (trajectory history;
                           HypoSpace has no scalar observations, so the shared
                           HistoryState — which renders observation history —
                           is empty here; this renderer mirrors the official
                           ``prior_hypotheses`` behaviour instead)
  dedup_archive          : unique constructed objects (shared PopulationState)
  complexity_stratified  : unique constructed objects grouped by complexity
                           (causal: #edges; boolean: key depth; voxel: height)
  cases                   : structural coverage state — V_t, G_sci regions
                           (connected components), G_rev revision
                           neighborhoods, coverage statistics.

The CASES condition is deliberately *structural only*: HypoSpace has no
empirical utility field (A2: S_t = (V_t, G_t, L_t)), so no posterior numbers
are ever fabricated.  All blocks are capped at ``token_budget`` tokens using
the shared 4-chars-per-token convention.
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from ...core.state import SolutionSpaceState
from ...core.types import GroundedObject

_CHARS_PER_TOKEN = 4


def _cap(text: str, token_budget: int) -> str:
    return text[: token_budget * _CHARS_PER_TOKEN] if token_budget >= 0 else text


# ---------------------------------------------------------------------------
# Task prompts (mirror the official create_prompt; identical across mechanisms)
# ---------------------------------------------------------------------------


def render_task_prompt(domain: str, task) -> str:
    if domain == "causal":
        return _causal_prompt(task)
    if domain == "boolean":
        return _boolean_prompt(task)
    if domain == "voxel3d":
        return _voxel_prompt(task)
    raise ValueError(f"unknown domain: {domain}")


def _causal_prompt(task) -> str:
    nodes_str = ", ".join(task.nodes)
    obs_block = "\n".join(
        f"Perturb({o['perturbed_node']}) -> "
        + ", ".join(f"{n}:{v}" for n, v in sorted(o["effects"].items()))
        for o in task.observations
    )
    constraint = (
        f"\nConstraint: The graph should have at most {task.max_edges} edges."
        if task.max_edges is not None
        else ""
    )
    return (
        "You are given observations from perturbation experiments on a causal system.\n"
        "Semantics:\n"
        "- When a node is perturbed, the perturbed node is 0.\n"
        "- A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.\n"
        "- All other nodes are 0.\n"
        f"\nNodes: {nodes_str}{constraint}\n"
        f"\nObservations:\n{obs_block}\n"
        "\nTask:\n"
        "Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.\n"
        "\nFormatting rules:\n"
        "1) Use only the listed nodes. No self-loops. No cycles.\n"
        "2) Respond with exactly one line:\n"
        "- If there are edges: Graph: A->B, B->C\n"
        "- If there are no edges: Graph: No edges\n"
    )


def _boolean_prompt(task) -> str:
    obs_block = "\n".join(
        "(" + ", ".join(f"{k}={v}" for k, v in sorted(o["inputs"].items())) + f") -> {o['output']}"
        for o in task.observations
    )
    return (
        f"You are given partial observations of a Boolean function with variables: "
        f"{', '.join(task.variables)}\n"
        f"Allowed operators: {', '.join(sorted(task.operators))}\n"
        f"\nObservations (input -> output):\n{obs_block}\n"
        "\nTask: Generate a single Boolean expression that is consistent with ALL observations.\n"
        "Requirements:\n"
        f"1. Use ONLY the variables: {', '.join(task.variables)}\n"
        f"2. Use ONLY these operators: {', '.join(sorted(task.operators))}\n"
        "3. The expression must match all given observations\n"
        "4. Do not use boolean constants True or False anywhere in the expression.\n"
        f"5. Expression depth should be at most {task.max_depth} levels of nesting\n"
        "Output format:\n"
        '- Return ONLY the Boolean expression on a single line\n'
        "- Use uppercase for operators (AND, OR, NOT, XOR, NOR)\n"
        "- Use lowercase for variables\n"
        '- Start your response with "Expression: " followed by the expression\n'
    )


def _voxel_prompt(task) -> str:
    grid = task.grid_size
    obs = "\n".join(
        " ".join(task.observation[r * grid : r * grid + grid]) for r in range(grid)
    )
    return (
        f"You are given observations of a 3D structure made of unit blocks on a {grid}x{grid} grid.\n"
        f"Each observation shows a view of the structure from a specific angle.\n"
        f"The maximum height of the structure is {task.max_height} layers.\n"
        f"\nTop view:\n{obs}\n"
        "\nTask: Infer the complete 3D structure that could produce these observations.\n"
        "Structure specifications:\n"
        f"- Grid size: {grid}x{grid}\n"
        f"- Maximum height: {task.max_height} layers\n"
        "Important constraints:\n"
        "1. Blocks must be supported from below (a block at height h requires a block at height h-1 in the same position)\n"
        "2. A column whose cell is 0 in the top view must have height 0\n"
        "3. Every column shown as 1 in the top view must have height AT LEAST 1\n"
        "4. No column may exceed the maximum height\n"
        "\nOutput a compact HEIGHTS MATRIX describing the structure. Row i, column j is the height of blocks stacked at that grid position (0 = empty).\n"
        "Output format (exactly this):\n"
        "Heights:\n"
        "[row 1: heights separated by spaces]\n"
        "[row 2: heights separated by spaces]\n"
        "example for a 3x3 grid:\n"
        "Heights:\n"
        "0 1 2\n"
        "0 0 1\n"
        "0 1 0\n"
        "(layer-by-layer 'Structure:' format is also accepted as a fallback)\n"
    )


# ---------------------------------------------------------------------------
# State blocks per mechanism
# ---------------------------------------------------------------------------


async def render_state_block(
    mechanism: str,
    state: SolutionSpaceState,
    adapter,
    *,
    valid_ids: frozenset[str] = frozenset(),
    token_budget: int = 1800,
    readout=None,
    foreign_packet=None,
    target_complexity: int | None = None,
    explore: str = "medium",
) -> str:
    """The prior-information block for a mechanism (empty for independent).

    ``readout``         : core CoverageFrontierReadout (cases / shuffled_frontier)
    ``foreign_packet``  : another task's ReadoutPacket (shuffled_frontier)
    ``target_complexity``: complexity level to target (stratified_gt)
    """
    if mechanism == "independent":
        return ""
    if mechanism == "history":
        return _cap(_history_block(state, adapter, valid_ids), token_budget)
    if mechanism == "dedup_archive":
        return _cap(_dedup_block(state, adapter, valid_ids), token_budget)
    if mechanism in ("complexity_stratified", "stratified_uniform"):
        return _cap(_stratified_block(state, adapter, valid_ids), token_budget)
    if mechanism == "stratified_gt":
        return _cap(
            _stratified_gt_block(state, adapter, valid_ids, target_complexity),
            token_budget,
        )
    if mechanism == "cases":
        return _cap(
            _frontier_block(state, adapter, readout, token_budget, explore=explore),
            token_budget,
        )
    if mechanism == "shuffled_frontier":
        return _cap(
            _frontier_block(state, adapter, readout, token_budget, foreign_packet=foreign_packet),
            token_budget,
        )
    raise ValueError(f"unknown mechanism: {mechanism}")


def _history_block(state: SolutionSpaceState, adapter, valid_ids) -> str:
    lines = ["## Prior proposals (trajectory history)"]
    for h in sorted(state.hypotheses.values(), key=lambda h: h.round_index):
        obj = _first_object_for_hypothesis(state, h.hypothesis_id)
        verdict = "valid" if obj and obj.object_id in valid_ids else "invalid/unparsed"
        rep = adapter.render_object(obj) if obj else h.text[:200]
        lines.append(f"- r{h.round_index}: {rep} [{verdict}]")
    return "\n".join(lines)


def _dedup_block(state: SolutionSpaceState, adapter, valid_ids) -> str:
    """Canonical archive + explicit non-repeat constraint (per Hypo问题诊断1
    §26: memory-based reject-duplicate decoding, not a generic population
    renderer)."""
    lines = [
        "## Canonical archive (reject-duplicate decoding)",
        "Do NOT repeat any hypothesis whose canonical form matches an archived "
        "form below.  Produce a hypothesis whose canonical form differs from "
        "ALL archived forms.",
    ]
    for oid, obj in sorted(state.objects.items(), key=lambda kv: kv[0]):
        marker = "*" if oid in valid_ids else " "
        lines.append(f"- [{marker}] {adapter.render_object(obj)}  (canonical: {obj.canonical_form})")
    return "\n".join(lines)


def _stratified_block(state: SolutionSpaceState, adapter, valid_ids) -> str:
    """Stratified-Uniform: uniform emphasis across complexity levels; the
    complexity distribution is NOT taken from ground truth (realistic
    baseline; reported separately from Stratified-GT)."""
    groups: dict[int, list[GroundedObject]] = {}
    for oid, obj in state.objects.items():
        c = _complexity(obj)
        groups.setdefault(c, []).append(obj)
    lines = ["## Unique constructed objects by complexity (Stratified-Uniform)"]
    for c in sorted(groups):
        objs = sorted(groups[c], key=lambda o: o.object_id)
        reps = ", ".join(adapter.render_object(o) for o in objs)
        lines.append(f"- complexity {c} ({len(objs)}): {reps}")
    return "\n".join(lines)


def _stratified_gt_block(state, adapter, valid_ids, target_complexity: int | None) -> str:
    """Stratified-GT (oracle-assisted diagnostic control, reported separately):
    the harness instructs a complexity level drawn from the GROUND-TRUTH
    complexity distribution.  Legal because HypoSpace is a diagnostic and this
    baseline is reported as Stratified-GT, not as a realistic competitor."""
    lines = ["## Stratified-GT (oracle-assisted control)"]
    lines.append(
        f"- target complexity level: {target_complexity} — produce a hypothesis "
        f"whose complexity equals this level (observations still apply)."
    )
    for oid, obj in sorted(state.objects.items(), key=lambda kv: kv[0]):
        marker = "*" if oid in valid_ids else " "
        lines.append(f"- [{marker}] {adapter.render_object(obj)}")
    return "\n".join(lines)


def _frontier_block(
    state: SolutionSpaceState,
    adapter,
    readout,
    token_budget: int,
    foreign_packet=None,
    explore: str = "medium",
) -> str:
    """V6.1 CoverageFrontierReadout block (S_t -> rho(F_t) -> LLM).

    CASES:             current covered regions + current uncovered frontiers.
    Shuffled Frontier: same format/regions, but the frontier targets come from
                      another task's readout (relation-content control).
    Readout v2: directive phrasing + canonical non-repeat clause.
    """
    if readout is None:
        return "## (readout unavailable)"
    packet = readout.build(state, adapter)
    if foreign_packet is not None:
        packet = _replace_targets(packet, foreign_packet.frontier_targets)
    # Clean adapter-local frontier renderer.  We deliberately do NOT use the
    # shared core `render_readout` here: its per-target dump of raw descriptor
    # tuples ("degree_profile=(...)", "source_sink=(('A','D'),('B','C','D'))",
    # "differs from anchor ...: attr->attr; revision classes: ...") overloads
    # and misleads the model on HypoSpace (it drifted off the observations and
    # returned invalid graphs; see H1 audit).  A concise directive through the
    # adapter's observation-aware `target_instruction` + clean covered-anchor
    # list is what makes the local model follow the observations (validity 1.0
    # in a controlled probe).
    body = _render_frontier_clean(packet, state, adapter, explore=explore)
    # canonical non-repeat clause (dedup-pressure parity); reserve budget so
    # the clause is never truncated away
    anchors = ", ".join(adapter.render_object(o) for o in state.objects.values())
    clause = (
        "\n\nNon-repeat constraint: do NOT repeat any canonical form already "
        f"constructed ({anchors or 'none yet'})."
    )
    return _cap(body + clause, token_budget)


_EXPLORE_LEVELS = ("slow", "medium", "high", "ultra")


def _render_frontier_clean(packet, state: SolutionSpaceState, adapter, explore: str = "medium") -> str:
    """HypoSpace-local, token-budgeted frontier block, gated by the
    exploration dial:

    * ``explore="slow"`` — recovery-first: NEVER emit "explore"/"different"
      divergence phrasing.  Even when readout produced frontier targets, the
      block only lists covered anchors and asks to recover still-uncovered
      legal structures.  This is how we ask "can CASES recover without
      sacrificing validity?" on hard-constraint domains (H2 root-cause).
    * ``explore="medium"`` (default) — current behavior: cold-start seed-first
      + seeded "explore structurally different but VALID structures".
    * ``explore="high"`` / ``"ultra"`` — progressively stronger divergence
      phrasing (for permissive domains / saturation studies).
    """
    if explore not in _EXPLORE_LEVELS:
        explore = "medium"

    lines: list[str] = [
        "## Covered structural regions",
        f"- {packet.constructed_count} constructed object(s), "
        f"{packet.covered_cell_count} distinct structure(s) already found:",
    ]
    for oid in sorted(state.objects):
        lines.append(f"  - {adapter.render_object(state.objects[oid])}")

    if not state.objects:
        # ---- cold start / no seed yet --------------------------------------
        lines.append("## Uncovered frontiers")
        lines.append(
            "- Objective RIGHT NOW: produce ANY single VALID structure that "
            "satisfies ALL the listed observations EXACTLY.  Correctness is the "
            "only thing that matters at this step; do not chase novelty yet.  "
            "Once a valid structure is found, further exploration may begin."
        )
        return "\n".join(lines)

    if explore == "slow":
        # ---- recovery-first (no divergence phrasing) -----------------------
        lines.append("## Uncovered frontiers")
        lines.append(
            "- Recover as many VALID structures as possible that still satisfy "
            "ALL the listed observations EXACTLY.  Prefer recovering an "
            "uncovered legal structure over changing one just to be different."
        )
        return "\n".join(lines)

    # ---- seeded: explore (medium/high/ultra) --------------------------------
    heading = {
        "medium": "## Uncovered frontiers — explore structurally different but VALID structures",
        "high": "## Uncovered frontiers — actively explore DIFFERENT, valid structures to broaden coverage",
        "ultra": "## Uncovered frontiers — aggressively explore the widest range of DIFFERENT valid structures",
    }[explore]
    lines.append(heading)
    feas_filter = getattr(adapter, "observation_feasible", None)
    if feas_filter is None:
        feas_filter = getattr(adapter, "cell_feasible", None)
    targets = []
    if feas_filter is not None:
        for t in packet.frontier_targets:
            try:
                if feas_filter(t.descriptor):
                    targets.append(t)
            except Exception:
                targets.append(t)
    else:
        targets = list(packet.frontier_targets)
    for i, t in enumerate(targets, start=1):
        directive = ""
        if hasattr(adapter, "target_instruction"):
            try:
                directive = adapter.target_instruction(t.descriptor)
            except Exception:
                directive = ""
        if directive:
            lines.append(f"- Do next ({i}): {directive}")
        else:
            lines.append(f"- Do next ({i}): produce a structurally different valid structure")
    if not targets:
        lines.append("- (frontier exhausted for now — consider any remaining structure)")
    return "\n".join(lines)


def _replace_targets(packet, foreign_targets):
    from ...core.readout import ReadoutPacket

    return ReadoutPacket(
        task_id=packet.task_id,
        round_index=packet.round_index,
        covered_regions=packet.covered_regions,
        frontier_targets=tuple(foreign_targets),
        constructed_count=packet.constructed_count,
        covered_cell_count=packet.covered_cell_count,
        metadata={**packet.metadata, "frontier_shuffled": True},
    )


def _first_object_for_hypothesis(state, hypothesis_id: str) -> GroundedObject | None:
    for obj in state.objects.values():
        if hypothesis_id in obj.source_hypothesis_ids:
            return obj
    return None


def _complexity(obj: GroundedObject) -> int:
    if obj.object_type == "causal_dag":
        return int(obj.payload.get("n_edges", 0))
    if obj.object_type == "boolean_expr":
        key = obj.payload.get("mechanistic_key")
        if key is None:
            return 0

        def children(data):
            if data[0] in ("VAR", "CONST"):
                return ()
            if data[0] == "NOT":
                return (data[1],)
            return tuple(data[1])

        def depth(data) -> int:
            if data[0] in ("VAR", "CONST"):
                return 0
            return 1 + max((depth(k) for k in children(data)), default=0)

        return depth(key)
    if obj.object_type == "voxel3d":
        return int(obj.payload.get("height", 0))
    return 0
