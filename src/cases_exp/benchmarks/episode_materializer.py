"""Episode materializer for E2b/E3 (spec §14.2, §15.3, §16).

Compiles a DB-SYNTH task into a method-independent canonical SubstrateState,
then applies exactly ONE corruption at a chosen locus (SRC / COMMIT / RET).
The corruption identity is stored in a PRIVATE ground-truth file that methods
must never see; only the evaluator reads it.

Locus semantics (theory chain H -> e_src -> e_sem -> e_cmp):
- SRC:     a decision-relevant distinction is removed from the source compilation
           (claim's source term becomes None: unrepresented in P^src).
- COMMIT:  a spurious active commitment equation collapses two distinct source
           terms (representation-loss by over-identification in Theta_{A_t}).
- RET:     a computational grounding map entry is dropped (semantic distinction
           exists but has no carrier -> undetermined at execution).

Deterministic under (task_id, seed). Exact deficit computation is delegated to
evaluators/mechanism_exact.py at scoring time, not here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..substrate.core import Claim, Commitment, SourceState, SubstrateState

LOCUS_CHOICES = ("SRC", "COMMIT", "RET")


def compile_episode(task, query_idx: int = 0) -> tuple[SubstrateState, dict[str, Any]]:
    """Build the canonical uncorrupted substrate for one DB-SYNTH query."""
    q = task.queries[query_idx]
    if isinstance(q, list):  # query group: use its first (seed) question
        q = q[0] if q else {}
    cols: list[str] = [c for c in (q.get("relevant_cols") or []) if isinstance(c, str)]
    target = q.get("target_col") or ""
    if target and target not in cols:
        cols.append(target)
    if not cols:
        # dataset columns are {name, description} dicts in synth/test metadata
        for d in task.datasets[:1]:
            raw = d.get("columns") or []
            cols = [c["name"] for c in raw if isinstance(c, dict) and c.get("name")][:8]
            break
    cols = [c for c in cols if isinstance(c, str)]

    claims: dict[str, Claim] = {}
    source_compilation: dict[str, str | None] = {}
    for i, c in enumerate(sorted(set(cols))):
        cid = f"c{i}"
        claims[cid] = Claim(claim_id=cid, text=f"column: {c}",
                            provenance=[f"{task.task_id}#col:{c}"])
        source_compilation[cid] = f"term_{c}"

    # hypothesis claim: the decision statement to be recovered
    hyp_id = "h_gold"
    hyp_text = q.get("true_hypothesis") or ""
    if not hyp_text:
        try:
            from .discoverybench import gold_hypothesis as _gold
            hyp_text = _gold(task, query_idx) or ""
        except Exception:
            hyp_text = ""
    claims[hyp_id] = Claim(claim_id=hyp_id, text=hyp_text,
                           provenance=[f"{task.task_id}#qid:{q.get('qid')}"])
    source_compilation[hyp_id] = f"term_hyp_{q.get('qid', 0)}"

    commitments: dict[str, Commitment] = {}
    grounding: dict[str, str] = {}
    # identity commitments linking hypothesis terms to column terms (witness: co-occurrence)
    for i, c in enumerate(sorted(set(cols))):
        aid = f"a{i}"
        commitments[aid] = Commitment(
            commitment_id=aid, equations=[(f"term_hyp_{q.get('qid', 0)}", f"term_{c}")],
            witness={"type": "usage", "col": c})
        grounding[f"term_{c}"] = f"carrier::{c}"
    grounding[f"term_hyp_{q.get('qid', 0)}"] = f"carrier::hyp_{q.get('qid', 0)}"

    state = SubstrateState(
        source=SourceState(claims=claims, source_compilation=source_compilation,
                           vocabulary=sorted(set(cols))),
        active_commitments=commitments,
        computational_grounding=grounding,
        frontier_certificates=[{"frontier": "values not in metadata"}],
    )
    meta = {
        "task_id": task.task_id, "query_idx": query_idx, "qid": q.get("qid"),
        "question": q.get("question", ""),
        "gold_hypothesis": q.get("true_hypothesis", ""),
        "gold_expr": q.get("true_hypothesis_expr", ""),
        "target_col": target, "relevant_cols": cols,
        "decision_terms": [f"term_{c}" for c in sorted(set(cols))] + [f"term_hyp_{q.get('qid', 0)}"],
    }
    return state, meta


def corrupt(state: SubstrateState, meta: dict[str, Any], locus: str, rng: random.Random) -> tuple[SubstrateState, dict[str, Any]]:
    """Apply ONE corruption at `locus`; returns (corrupted_state, private_truth)."""
    terms = [t for t in state.source.source_compilation.values() if t]
    truth: dict[str, Any] = {"locus": locus}

    if locus == "SRC":
        # remove one decision-relevant source term from compilation
        victim_claim = rng.choice([cid for cid, t in state.source.source_compilation.items() if t and cid != "h_gold"])
        term = state.source.source_compilation[victim_claim]
        new_state = SubstrateState(
            source=SourceState(
                claims=dict(state.source.claims),
                source_compilation={**state.source.source_compilation, victim_claim: None},
                vocabulary=list(state.source.vocabulary)),
            active_commitments={k: Commitment(commitment_id=v.commitment_id,
                                              equations=list(v.equations),
                                              witness=dict(v.witness),
                                              protected=v.protected,
                                              dependencies=list(v.dependencies))
                                for k, v in state.active_commitments.items()},
            computational_grounding=dict(state.computational_grounding),
            frontier_certificates=list(state.frontier_certificates),
            version=state.version + 1)
        truth.update({"removed_claim": victim_claim, "removed_term": term,
                      "min_repair_depth": "SOURCE_EDIT", "d_star": 3})
        return new_state, truth

    if locus == "COMMIT":
        # add spurious commitment collapsing two distinct terms
        pair = rng.sample([t for t in terms if not t.startswith("term_hyp")], 2)
        spurious = Commitment(commitment_id="a_spurious", equations=[(pair[0], pair[1])],
                              witness={"type": "spurious"})
        new_state = SubstrateState(
            source=state.source,
            active_commitments={**state.active_commitments, "a_spurious": spurious},
            computational_grounding=dict(state.computational_grounding),
            frontier_certificates=list(state.frontier_certificates),
            version=state.version + 1)
        truth.update({"spurious_commitment": "a_spurious", "collapsed_pair": pair,
                      "min_repair_depth": "COMMITMENT_EDIT", "d_star": 2})
        return new_state, truth

    if locus == "RET":
        # drop one grounding entry (distinction alive but not executable)
        victim_term = rng.choice([t for t in state.computational_grounding if not t.startswith("term_hyp")])
        new_state = SubstrateState(
            source=state.source,
            active_commitments={k: Commitment(commitment_id=v.commitment_id,
                                              equations=list(v.equations),
                                              witness=dict(v.witness),
                                              protected=v.protected,
                                              dependencies=list(v.dependencies))
                                for k, v in state.active_commitments.items()},
            computational_grounding={k: v for k, v in state.computational_grounding.items() if k != victim_term},
            frontier_certificates=list(state.frontier_certificates),
            version=state.version + 1)
        truth.update({"dropped_term": victim_term,
                      "min_repair_depth": "GROUNDING_ONLY", "d_star": 1})
        return new_state, truth

    raise ValueError(f"unknown locus {locus}")


def serialize(state: SubstrateState) -> dict[str, Any]:
    return {
        "source": {"claims": {k: vars(c) for k, c in state.source.claims.items()},
                   "source_compilation": state.source.source_compilation,
                   "vocabulary": state.source.vocabulary},
        "active_commitments": {k: vars(v) for k, v in state.active_commitments.items()},
        "computational_grounding": state.computational_grounding,
        "frontier_certificates": state.frontier_certificates,
        "version": state.version,
    }


def materialize_batch(tasks, out_dir: Path, seed: int = 92711,
                      loci: tuple[str, ...] = LOCUS_CHOICES) -> list[dict[str, Any]]:
    """Materialize one corrupted episode per task; write public episode file +
    PRIVATE truth file (evaluator-only). Returns index records."""
    out_dir = Path(out_dir)
    (out_dir / "public").mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    index = []
    for i, task in enumerate(tasks):
        rng = random.Random(f"{seed}:{task.task_id}")
        locus = loci[i % len(loci)]
        state, meta = compile_episode(task)
        corrupted, truth = corrupt(state, meta, locus, rng)
        rec = {"episode_id": f"{task.task_id}::{meta['qid']}", "locus_slot": i % len(loci), **meta}
        (out_dir / "public" / f"{i:04d}.json").write_text(json.dumps(
            {"episode": rec, "corrupted_state": serialize(corrupted)}, ensure_ascii=False, indent=1))
        (out_dir / "private" / f"{i:04d}.json").write_text(json.dumps(
            {**rec, "truth": truth}, ensure_ascii=False, indent=1))
        index.append(rec)
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return index
