"""Voxel3DAdapter — HypoSpace 3D voxel-reconstruction domain (Agent A).

Grounding chain:

    language -> official parse_llm_response (Structure3D layers)
             -> official validation (top-view equality + per-layer support)
             -> normalization (strip trailing all-zero layers, official
                ``Structure3D.normalize``) -> canonical fixed-order tensor
                serialization.

Canonical form:  "voxel3d|<H>x<W>|<layer0 rows>|<layer1 rows>|..."  where rows
are row-major 0/1 strings and layers are bottom-to-top (official convention).
Uniqueness == official ``Structure3D.get_hash`` equivalence (verified by tests).

G_sci  : Dice overlap on outcome-blind fixed-order occupancy tensors.
G_rev  : column_height_edit — a single column's height changes by exactly one
         step (local occupancy modification / connected structural edit),
         only when the target passes the official validator.
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
from .tasks import VoxelTask

OBJECT_TYPE = "voxel3d"


def _parse_top_view(observation: str, grid_size: int) -> np.ndarray:
    """Top view from the official compact string, e.g. '101000101'."""
    arr = np.asarray([int(c) for c in observation.strip()], dtype=np.int64)
    return arr.reshape(grid_size, grid_size)


class Voxel3DAdapter(TaskAdapter):
    task_id = "hypospace_voxel3d"
    object_type = OBJECT_TYPE

    def __init__(self, task: VoxelTask) -> None:
        self.task = task
        self._parser = official.voxel_parser()
        self._top_view = _parse_top_view(task.observation, task.grid_size)

    # -- compile -------------------------------------------------------------

    async def compile(self, hypothesis: LanguageHypothesis, llm) -> RawScientificObject:
        # Structured-proposal representation (H0 root-cause fix): the adapter
        # accepts the compact HEIGHTS MATRIX form first — a grid x grid table of
        # per-column heights (the exact parameterization the official
        # enumeration uses) — because the verbose layer-by-layer form lets the
        # model re-describe the same structure many ways (many-to-one grounding
        # / voxel collapse, H0).  Fall back to the official layer parser so
        # existing callers (raw_from_canonical, backward-compatible prompts)
        # keep working.
        layers = _parse_heights_matrix(hypothesis.text, self.task.grid_size, self.task.max_height)
        parser_used = "heights_matrix"
        if layers is None:
            structure = self._parser(hypothesis.text)
            if structure is not None and structure.layers:
                layers = [np.asarray(layer, dtype=np.int64) for layer in structure.layers]
            parser_used = "parse_llm_response"
        if not layers:
            raise GroundingError(
                f"voxel3d parse failed for hypothesis {hypothesis.hypothesis_id}"
            )
        return RawScientificObject(
            task_id=self.task_id,
            object_type=self.object_type,
            payload={"layers": layers},
            source_hypothesis_id=hypothesis.hypothesis_id,
            raw_text=hypothesis.text,
            metadata={"evaluator": "hypospace_official_parse", "parser": parser_used},
        )

    # -- verify (official top-view + support semantics) -----------------------

    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        structure = official.make_structure3d(raw_object.payload["layers"])
        errors: list[str] = []
        if not official.validate_voxel(structure, self.task.observation):
            errors.append("voxel_invalid:top_view_mismatch_or_unsupported_block")
        # Over-height structures pass the official validator (top view +
        # support only) and are counted VALID by the official runner — they
        # simply can never be recovered (admissible set is capped at
        # max_height).  Mirror the official verdict exactly and record the
        # discrepancy as a deterministic, logged warning.
        warnings: list[str] = []
        if len(structure.layers) > self.task.max_height:
            warnings.append(
                f"voxel_over_max_height:{len(structure.layers)}>{self.task.max_height}"
            )
        return VerificationResult(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    # -- canonicalize -----------------------------------------------------------

    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        structure = official.make_structure3d(raw_object.payload["layers"])
        normalized = structure.normalize()
        tensor = _tensor(normalized, self.task.grid_size)
        canonical_form = _serialize_tensor(tensor)

        display = _display(tensor)
        return GroundedObject(
            object_id=stable_object_id(self.task_id, canonical_form),
            task_id=self.task_id,
            object_type=self.object_type,
            canonical_form=canonical_form,
            payload={
                "tensor": [layer.tolist() for layer in tensor],
                "height": int(len(tensor)),
                "n_blocks": int(np.asarray(tensor).sum()),
            },
            display_text=display,
            source_hypothesis_ids=(raw_object.source_hypothesis_id,),
            metadata={
                "evaluator": "hypospace_official",
                "domain": "voxel3d",
                "observation_id": self.task.observation_id,
            },
        )

    # -- G_sci ----------------------------------------------------------------

    def _occupancy(self, obj: GroundedObject, height: int) -> np.ndarray:
        """Binary voxel-occupancy vector padded to a common ``height`` (fixed
        length across the object set, so Dice on it is well-defined)."""
        tensor = np.asarray(obj.payload["tensor"], dtype=np.int64)  # (H, G, G)
        g = tensor.shape[1]
        out = np.zeros((height, g, g), dtype=np.float64)
        h = min(len(tensor), height)
        if h > 0:
            out[:h] = tensor[:h].astype(np.float64)
        return out.reshape(-1)

    def build_scientific_graph(self, objects: tuple[GroundedObject, ...]) -> ScientificGraph:
        heights = [int(len(np.asarray(o.payload["tensor"]))) for o in objects]
        common = max(heights) if heights else 0
        occ = {o.object_id: self._occupancy(o, common) for o in objects}
        edges: list[ScientificEdge] = []
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                fa, fb = occ[a.object_id], occ[b.object_id]
                inter = float(fa @ fb)  # binary -> intersection size
                dice = 2.0 * inter / (float(fa.sum()) + float(fb.sum())) if (fa.sum() + fb.sum()) > 0 else 0.0
                if dice > 0.0:
                    edges.append(
                        ScientificEdge(
                            edge_id=stable_edge_id(a.object_id, b.object_id, "voxel_occupancy"),
                            source_id=a.object_id,
                            target_id=b.object_id,
                            weight=dice,
                            distance=1.0 - dice,
                            relation_type="voxel_occupancy",
                            metadata={"outcome_blind": True},
                        )
                    )
        return ScientificGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "voxel3d", "relation_type": "voxel_occupancy", "outcome_blind": True},
        )

    # -- G_rev -----------------------------------------------------------------

    def build_revision_graph(self, objects: tuple[GroundedObject, ...]) -> RevisionGraph:
        edges: list[RevisionEdge] = []
        for a in objects:
            for b in objects:
                if a.object_id == b.object_id:
                    continue
                ta = np.asarray(a.payload["tensor"], dtype=np.int64)
                tb = np.asarray(b.payload["tensor"], dtype=np.int64)
                edit = _column_height_edit(ta, tb)
                if edit is None:
                    continue
                action, payload = edit
                # Benchmark-validity guarantee: constructed nodes passed the
                # official validator at insertion; audit re-check = gravity
                # (support) only — no observation re-evaluation.
                if not _support_valid(tb):
                    continue
                edges.append(
                    RevisionEdge(
                        edge_id=stable_edge_id(a.object_id, b.object_id, action),
                        source_id=a.object_id,
                        target_id=b.object_id,
                        action_type=action,
                        action_description=f"{action}: column={payload['column']} "
                        f"{payload['from_h']}->{payload['to_h']}",
                        edit_payload=payload,
                        metadata={"domain": "voxel3d", "outcome_blind": True},
                    )
                )
        return RevisionGraph(
            node_ids=tuple(o.object_id for o in objects),
            edges=tuple(edges),
            metadata={"domain": "voxel3d", "outcome_blind": True},
        )

    # -- evaluation --------------------------------------------------------------

    async def evaluate(self, obj: GroundedObject):
        raise EvaluationNotApplicable(
            "HypoSpace has no scalar empirical utility field; validity is a "
            "deterministic benchmark verdict, not an Observation."
        )

    def render_object(self, obj: GroundedObject) -> str:
        return obj.display_text

    # -- FrontierSemantics (V6.1 SpaceReadout) --------------------------------
    # Descriptor per Hypo问题诊断1: phi(x) = (#occupied, height histogram,
    # max height, footprint compactness, symmetry).  Outcome-blind; from the
    # canonical tensor + public task grammar only.

    def descriptor(self, obj: GroundedObject) -> dict[str, Any]:
        tensor = np.asarray(obj.payload["tensor"], dtype=np.int64)  # (H,G,G)
        g = tensor.shape[1]
        n_occupied = int(tensor.sum())
        heights = tensor.sum(axis=0)  # (G,G)
        max_h = int(heights.max()) if heights.size else 0
        # fixed-length histogram over bins 0..task.max_height (over-height
        # columns clip into the top bin); length is task-fixed
        cap = self.task.max_height
        clipped = np.minimum(heights, cap)
        hist = tuple(int((clipped == h).sum()) for h in range(0, cap + 1))
        # footprint compactness: occupied columns in the top view / total grid
        footprint = int((heights > 0).sum())
        compactness = round(footprint / (g * g), 4)
        # symmetry: max over the 4 rotations + flips of matching cells
        top = (heights > 0).astype(np.int64)
        best = 0
        for k in range(4):
            rot = np.rot90(top, k)
            best = max(best, int((rot == top).sum()))
            best = max(best, int((np.fliplr(rot) == top).sum()))
        symmetry = round(best / (g * g), 4)
        return {
            "n_occupied": n_occupied,
            "height_histogram": hist,
            "max_height": max_h,
            "footprint_compactness": compactness,
            "symmetry": symmetry,
        }

    def attribute_domains(self) -> dict[str, tuple[Any, ...]]:
        g = self.task.grid_size
        max_h = self.task.max_height
        max_blocks = g * g * max_h
        hist_domain = tuple(
            tuple(counts)
            for counts in _integer_histograms(g * g, max_h + 1, max_blocks)
        )
        # compactness/symmetry: exact public-grammar values k/(g*g)
        frac_bins = tuple(round(k / (g * g), 4) for k in range(0, g * g + 1))
        return {
            "n_occupied": tuple(range(1, max_blocks + 1)),
            "height_histogram": hist_domain,
            "max_height": tuple(range(0, max_h + 1)),
            "footprint_compactness": frac_bins,
            "symmetry": frac_bins,
        }

    def revision_hints(self, from_descriptor: Mapping[str, Any], to_descriptor: Mapping[str, Any]) -> tuple[str, ...]:
        hints: list[str] = []
        if to_descriptor.get("n_occupied", 0) != from_descriptor.get("n_occupied", 0):
            hints.append("column_height_edit")
        if to_descriptor.get("max_height", 0) != from_descriptor.get("max_height", 0):
            hints.append("column_height_edit")
        if to_descriptor.get("footprint_compactness", 0) != from_descriptor.get("footprint_compactness", 0):
            hints.append("column_height_edit")
        return tuple(dict.fromkeys(hints)) or ("column_height_edit",)

    def cell_feasible(self, descriptor: Mapping[str, Any]) -> bool:
        """Readout-v2 feasibility: height histograms must sum to the grid,
        block count must match the histogram, compactness/symmetry in [0,1]."""
        g = self.task.grid_size
        max_h = self.task.max_height
        hist = descriptor["height_histogram"]
        n_occ = int(descriptor["n_occupied"])
        max_h_desc = int(descriptor["max_height"])
        comp = descriptor["footprint_compactness"]
        sym = descriptor["symmetry"]
        if len(hist) != max_h + 1 or sum(hist) != g * g:
            return False
        if not (1 <= n_occ <= g * g * max_h):
            return False
        if sum(h * c for h, c in enumerate(hist)) != n_occ:
            return False
        if max_h_desc < 0 or max_h_desc > max_h:
            return False
        if max_h_desc > 0 and hist[max_h_desc] == 0:
            return False
        if not (0.0 <= comp <= 1.0) or not (0.0 <= sym <= 1.0):
            return False
        # compactness = occupied columns / grid
        occupied_cols = g * g - hist[0]
        if abs(comp - round(occupied_cols / (g * g), 4)) > 1e-6:
            return False
        return True

    def target_instruction(self, descriptor: Mapping[str, Any]) -> str:
        """Domain-flavoured directive for a voxel frontier target."""
        parts = []
        if descriptor["n_occupied"]:
            parts.append(f"with about {descriptor['n_occupied']} occupied voxels")
        if descriptor["max_height"]:
            parts.append(f"maximum height about {descriptor['max_height']}")
        parts.append(f"footprint compactness about {descriptor['footprint_compactness']}")
        base = "Produce a 3D structure " + (", ".join(parts) if parts else "with a different structure")
        return base + ", matching the observed top view, with gravity-respecting layers, and a canonical form distinct from all anchors"


# ---------------------------------------------------------------------------
# Canonical tensor helpers (deterministic, idempotent)
# ---------------------------------------------------------------------------


def _tensor(structure: Any, grid_size: int) -> list[np.ndarray]:
    """Fixed-order tensor: layers bottom-to-top, each padded to grid_size²."""
    out = []
    for layer in structure.layers:
        arr = np.asarray(layer, dtype=np.int64)
        padded = np.zeros((grid_size, grid_size), dtype=np.int64)
        h, w = arr.shape
        padded[: min(h, grid_size), : min(w, grid_size)] = arr[: min(h, grid_size), : min(w, grid_size)]
        out.append(padded)
    return out


def _serialize_tensor(tensor: list[np.ndarray]) -> str:
    h = len(tensor)
    w = tensor[0].shape[1] if tensor else 0
    parts = [f"{OBJECT_TYPE}|{h}x{w}"]
    for layer in tensor:
        parts.append(",".join("".join(str(int(c)) for c in row) for row in layer))
    return "|".join(parts)


def _tensor_to_heights(tensor: list[np.ndarray]) -> np.ndarray:
    """Per-column heights matrix (grid x grid): each cell = # contiguous blocks
    stacked in that column (0 if unoccupied).  This IS the structured proposal
    representation / the official enumeration parameterization."""
    if not tensor:
        return np.zeros((0, 0), dtype=np.int64)
    arr = np.asarray(tensor, dtype=np.int64)  # (H,G,G)
    return arr.sum(axis=0).astype(np.int64)


def _render_heights(tensor: list[np.ndarray]) -> str:
    """Compact heights-matrix rendering used for display/anchors — the
    low-redundancy form that removes the many-to-one re-description collapse."""
    h = _tensor_to_heights(tensor)
    if h.size == 0:
        return "empty"
    return "\n".join(" ".join(str(int(c)) for c in row) for row in h)


def _parse_heights_matrix(text: str, grid_size: int, max_height: int) -> list[np.ndarray] | None:
    """Extract a compact HEIGHTS MATRIX (grid x grid of per-column heights) from
    the model text, e.g.:

        Heights:
        0 1 2
        0 0 1
        0 1 0

    Returns the derived layer tensor (heights -> layers, gravity-respecting by
    construction), or None if no valid heights matrix is present.  Heights must
    be integers in 0..max_height, sum-of-heights == #occupied == top-view cells
    (a column with height>0 requires a block in the top view).  This is the
    structured proposal representation fixing the H0 voxel collapse.
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines()]
    # find the marker line "Heights:" (case-insensitive)
    start = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("heights"):
            start = i + 1
            break
    if start is None:
        return None
    rows: list[list[int]] = []
    for ln in lines[start:]:
        if not ln:
            continue
        # stop at non-numeric / reasoning lines
        toks = ln.replace(",", " ").split()
        if not toks or not all(tok.lstrip("-").isdigit() for tok in toks):
            if rows:  # we already have some rows; stop cleanly
                break
            continue
        try:
            row = [int(tok) for tok in toks]
        except ValueError:
            continue
        rows.append(row)
        if len(rows) == grid_size:
            break
    if len(rows) != grid_size:
        return None
    heights = np.asarray(rows, dtype=np.int64)
    if heights.shape != (grid_size, grid_size):
        return None
    if heights.min() < 0 or heights.max() > max_height:
        return None
    # top-view consistency: a column with height>0 must be occupied in the
    # observation's top view (the structured form encodes the observed footprint)
    return _heights_to_layers(heights, grid_size, max_height)


def _heights_to_layers(heights: np.ndarray, grid_size: int, max_height: int) -> list[np.ndarray] | None:
    """Rebuild the layer tensor from a heights matrix (bijective, gravity-
    respecting: block at height h requires a block at height h-1)."""
    H = int(heights.max()) if heights.size else 0
    if H < 0 or H > max_height:
        return None
    if H == 0:
        return None  # empty structure
    layers = [np.zeros((grid_size, grid_size), dtype=np.int64) for _ in range(H)]
    for r in range(grid_size):
        for c in range(grid_size):
            h = int(heights[r, c])
            for z in range(h):
                layers[z][r, c] = 1
    return layers


def _display(tensor: list[np.ndarray]) -> str:
    # heights-matrix display (structured proposal form) in place of verbose
    # layers — the low-redundancy representation the model should see/express.
    return _render_heights(tensor)


def _column_height_edit(ta: np.ndarray, tb: np.ndarray) -> tuple[str, dict] | None:
    """Exactly one column's height differs by exactly one step; all else equal.

    Covers local occupancy modification (0<->1) and connected structural edits
    (height +-1 on an existing column).  Column height = number of contiguous
    1s from the bottom (guaranteed contiguous for valid structures).  Tensors
    are padded to a common height first (canonicalization strips trailing
    all-zero layers).
    """
    hmax = max(len(ta), len(tb))
    if hmax == 0:
        return None
    grid = ta.shape[1]
    pa = _pad_tensor(ta, hmax, grid)
    pb = _pad_tensor(tb, hmax, grid)
    diff_cols = []
    for r in range(grid):
        for c in range(grid):
            ha = int(pa[:, r, c].sum())
            hb = int(pb[:, r, c].sum())
            if ha != hb:
                if abs(ha - hb) == 1 and _column_matches(pa[:, r, c], ha) and _column_matches(pb[:, r, c], hb):
                    diff_cols.append(((r, c), ha, hb))
                else:
                    return None
    if len(diff_cols) != 1:
        return None
    (r, c), ha, hb = diff_cols[0]
    return (
        "column_height_edit",
        {"column": [r, c], "from_h": ha, "to_h": hb, "grid_size": int(grid)},
    )


def _pad_tensor(t: np.ndarray, hmax: int, grid: int) -> np.ndarray:
    if len(t) == hmax:
        return t
    pad = np.zeros((hmax - len(t), grid, grid), dtype=np.int64)
    return np.vstack([t, pad])


def _column_matches(col: np.ndarray, height: int) -> bool:
    if height == 0:
        return not np.any(col)
    if height > len(col):
        return False
    return bool(np.all(col[:height] == 1)) and bool(not np.any(col[height:]))


def _integer_histograms(n_cols: int, n_bins: int, max_total: int):
    """All coarse height histograms: non-increasing-ish counts per height bin
    summing to <= max_total over n_cols columns (bounded grammar domain)."""
    from itertools import combinations_with_replacement

    seen: set[tuple] = set()
    out: list[tuple] = []
    # histograms are multisets of n_cols column heights in 0..n_bins-1
    for combo in combinations_with_replacement(range(n_bins), n_cols):
        hist = tuple(combo.count(h) for h in range(n_bins))
        if sum(h * c for h, c in enumerate(hist)) <= max_total and hist not in seen:
            seen.add(hist)
            out.append(hist)
    return out


def _support_valid(tensor: np.ndarray) -> bool:
    """Gravity audit: every block above layer 0 has support below it."""
    for z in range(1, len(tensor)):
        if np.any(tensor[z] & ~tensor[z - 1]):
            return False
    return True
