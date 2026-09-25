"""P0 deterministic evaluator for the E4-1 readout audit (§12.3, 2026-09-01).

Grades the result-blind extractor's answers against canonical answers derived
from the coordinator-approved L1 instance set
(work/G_generalists/readout_audit/instances/s3_codescientist_L1.json,
sha256 7a24b31a219d956e8b4edbcfacb76c69c760072808427098affd690c0bcca743).

Faithful to each instance's ``canonical_answer_derivation`` field; every grade
is deterministic (no LLM, no randomness).  Hallucination double-check: a
"measured/tested" assertion must (a) have a state carrier in the snapshot and
(b) exist in the evaluator-held observation log; failure of either = 0 with a
``hallucinated_state_claim`` flag.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# --------------------------------------------------------------- data holders


@dataclass
class ObservationLog:
    """Evaluator-held ground truth of what was actually measured."""

    measured: dict[str, float] = field(default_factory=dict)   # key -> outcome
    source: dict[str, str] = field(default_factory=dict)       # key -> initial|submitted

    def add(self, key: str, outcome: float, src: str = "submitted") -> None:
        self.measured[key] = float(outcome)
        self.source[key] = src

    @property
    def n_initial(self) -> int:
        return sum(1 for v in self.source.values() if v == "initial")

    def best(self) -> tuple[str, float]:
        if not self.measured:
            raise ValueError("empty observation log")
        key = max(sorted(self.measured), key=lambda k: self.measured[k])
        return key, self.measured[key]


@dataclass
class SnapshotView:
    """Deterministic parse of one system's round-t state snapshot."""

    text: str = ""
    state_measured: set[str] = field(default_factory=set)   # results echoed in state
    proposals: list[str] = field(default_factory=list)      # named promising candidates
    rationale_spans: list[str] = field(default_factory=list)
    negative_claims: list[str] = field(default_factory=list)
    uncertainty_regions: list[str] = field(default_factory=list)
    rankings: list[tuple[str, list[str]]] = field(default_factory=list)
    limit_declarations: list[str] = field(default_factory=list)
    declared_groups: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class GradeResult:
    score: float
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------ utilities

_MEASURED_RE = re.compile(
    r"\b(?:measured|tested|evaluated|queried|recorded|observed|outcome|yield|fitness)\b",
    re.IGNORECASE)
_NEGATED_RE = re.compile(
    r"(never tried|not measured|untested|unevaluated|unqueried|no outcome|no result)",
    re.IGNORECASE)


def _asserted_measured(text: str, key: str) -> bool:
    """True iff some local context window around ``key`` asserts a measurement."""
    for m in re.finditer(re.escape(key), text):
        lo, hi = max(0, m.start() - 60), min(len(text), m.end() + 60)
        window = text[lo:hi]
        if _NEGATED_RE.search(window):
            continue
        if _MEASURED_RE.search(window):
            return True
    return False


def _basis_faithful(basis: str, text: str) -> bool:
    if basis in text:
        return True
    words = basis.split()
    grams = [" ".join(words[i:i + 2]) for i in range(max(1, len(words) - 1))]
    return any(g in text for g in grams if len(g) >= 10)
_UNTESTED_RE = re.compile(
    r"(promising|untested|unevaluated|unqueried|propose|candidate|suggest)", re.IGNORECASE)


def _parse_json_answer(answer: str) -> dict:
    try:
        return json.loads(answer)
    except Exception:
        m = re.search(r"\{.*\}", answer, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"answer": answer, "evidence_span": "", "abstain": False,
            "abstain_reason": "", "currently_unrepresentable": False, "flags": []}


def _mentioned(text: str, keys: list[str]) -> list[str]:
    return [k for k in keys if k and k in text]


def _hamming(a: str, b: str) -> int:
    pa, pb = a.split("|"), b.split("|")
    if len(pa) == len(pb) and len(pa) > 1:
        return sum(1 for x, y in zip(pa, pb) if x != y)
    return 0 if a == b else max(len(a), len(b))


def _dist_fn(task: str, oracle) -> Callable[[str, str], int]:
    fn = getattr(oracle, "edit_distance", None)
    return fn if fn is not None else _hamming


def _family_key(task: str, key: str, oracle) -> str:
    if task == "bh":  # partition by ligand slot (frozen review fix: ligand identity)
        parts = key.split("|")
        return parts[1] if len(parts) > 1 else key
    if task == "gb1":  # partition by mutated-position set (normalized variant)
        fn = getattr(oracle, "position_set", None)
        return fn(key) if fn else key
    # made: mechanism identity per frozen config encoding ("mech::rest")
    return key.split("::")[0] if "::" in key else key


def _slot_distinct_counts(task: str, keys: list[str]):
    counts: dict[str, set] = {}
    for k in keys:
        parts = k.split("|")
        for si, slot in enumerate("albd"):
            if si < len(parts):
                counts.setdefault(slot, set()).add(parts[si])
    order = sorted(counts.items(), key=lambda kv: (-len(kv[1]), "albd".index(kv[0])))
    return order


# ------------------------------------------------------------------- grader


class P0Evaluator:
    """Deterministic grader for the 48 S3 L1 instances (3 tasks x 8 classes x 2)."""

    def __init__(self, instances_path: str,
                 oracle_factory: Callable[[str], Any]) -> None:
        with open(instances_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.meta = doc["meta"]
        self.rubric_notes = doc["rubric_notes"]
        self.pairs = doc["pairs"]
        self._oracle_factory = oracle_factory
        self._oracles: dict[str, Any] = {}

    # ------------------------------------------------------------- plumbing

    def oracle(self, task: str):
        if task not in self._oracles:
            self._oracles[task] = self._oracle_factory(task)
        return self._oracles[task]

    def _candidate_keys(self, task: str) -> list[str]:
        c = self.oracle(task).candidates
        seq = c() if callable(c) else c
        return [str(x) for x in seq]

    def _top_set(self, task: str) -> set[str]:
        return {str(c) for c in self.oracle(task).solution_set()}

    def _threshold(self, task: str) -> float:
        o = self.oracle(task)
        try:
            return float(o.gamma())
        except Exception:
            vals = sorted(o.gold_values())
            return vals[int(0.95 * (len(vals) - 1))]

    def _all_keys(self, snap: SnapshotView, log: ObservationLog) -> list[str]:
        return sorted(set(snap.proposals) | set(log.measured))

    # --------------------------------------------------- hallucination gate

    def _hallucination_flags(self, task: str, ans: dict, snap: SnapshotView,
                             log: ObservationLog) -> list[str]:
        """Double-verified fabrication check (state carrier + observation log)."""
        flags: list[str] = []
        text = str(ans.get("answer", ""))
        span = str(ans.get("evidence_span", ""))
        if span and span not in snap.text:
            flags.append("hallucinated_state_claim:evidence_span_not_in_snapshot")
        universe = set(self._candidate_keys(task))
        universe |= set(snap.proposals) | set(snap.negative_claims) | set(log.measured)
        for k in sorted(universe):
            if k and k in text and _asserted_measured(text, k):
                if k not in log.measured:
                    flags.append(f"hallucinated_state_claim:not_in_observation_log:{k}")
                elif not self._has_state_carrier(k, snap):
                    flags.append(f"hallucinated_state_claim:no_state_carrier:{k}")
        return sorted(set(flags))

    @staticmethod
    def _has_state_carrier(key: str, snap: SnapshotView) -> bool:
        return (key in snap.state_measured or key in snap.proposals
                or key in snap.negative_claims or key in snap.text)

    # -------------------------------------------------------------- dispatch

    def grade(self, task: str, idx: int, snap: SnapshotView, log: ObservationLog,
              answer: str) -> GradeResult:
        ans = _parse_json_answer(str(answer))
        flags = self._hallucination_flags(task, ans, snap, log)
        if any(f.startswith("hallucinated_state_claim") for f in flags):
            return GradeResult(0.0, flags, {"reason": "hallucination double-check failed"})
        table = {
            "tested_hypotheses": (self._fam0_list, self._fam1_best),
            "inferred_but_untested_alternatives": (self._fam2_named, self._fam3_rationale),
            "semantic_equivalence": (self._fam4_identical, self._fam5_one_factor),
            "distinct_solution_families": (self._fam6_partition, self._fam7_dominance),
            "coverage_gaps": (self._fam8_marginal, self._fam9_deep),
            "likely_boundaries": (self._fam10_boundary, self._fam11_negatives),
            "uncertainty_in_untested_regions": (self._fam12_uncertainty, self._fam13_ranking),
            "currently_unrepresentable_hypotheses": (self._fam14_declared, self._fam15_capacity),
        }
        inst = self.pairs[task]["instances"][idx]
        fam = table[inst["query_class"]]
        return (fam[0] if idx % 2 == 0 else fam[1])(task, snap, log, ans, flags)

    # --------------------------------------------- family 0: tested list ----

    def _fam0_list(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        canonical = sorted(log.measured)
        if not canonical:
            # rubric: correct statement that only initial records exist = 1
            good = re.search(rf"\bonly\b.*\b{log.n_initial}\b|\binitial\b", text, re.IGNORECASE)
            return GradeResult(1.0 if good else 0.5, flags, {"canonical": "initial-only"})
        mentioned = [k for k in canonical if k in text]
        if len(mentioned) == len(canonical):
            return GradeResult(1.0, flags, {"canonical_n": len(canonical)})
        # rubric: "correct statement that only the initial N records exist" = 1
        if (log.n_initial == len(canonical)
                and re.search(rf"\bonly\b[^.]{{0,60}}\b{len(canonical)}\b", text, re.IGNORECASE)):
            return GradeResult(1.0, flags, {"initial_only_statement": True})
        if mentioned:
            return GradeResult(0.5, flags, {"omitted": [k for k in canonical if k not in mentioned]})
        return GradeResult(0.0, flags, {"canonical_n": len(canonical)})

    # --------------------------------------------- family 1: best-so-far ----

    def _fam1_best(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        key, val = log.best()
        has_key = key in text
        has_val = re.search(rf"{val:.6g}", text) is not None or re.search(rf"{val:.4g}", text) is not None
        if task == "gb1":
            count_ok = re.search(rf"\b{len(log.measured)}\b", text) is not None
            parts = sum((has_key, has_val, count_ok))
            score = 1.0 if parts == 3 else (0.5 if parts >= 2 else 0.0)
        else:
            score = 1.0 if (has_key and has_val) else (0.5 if (has_key or has_val) else 0.0)
        return GradeResult(score, flags, {"canonical_best": key, "value": val})

    # ----------------------------------------- family 2: untested named -----

    def _fam2_named(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        top = self._top_set(task)
        named = [p for p in snap.proposals if p in text]
        if snap.proposals and not named:
            return GradeResult(0.0, flags, {"reason": "carrier exists but answer names none"})
        if not snap.proposals:
            if str(ans.get("abstain", "")).lower() == "true" or ans.get("abstain") is True:
                return GradeResult(1.0, flags, {"legitimate_abstain": "no carrier (D2c)"})
            return GradeResult(0.5, flags, {"no_carrier": True})
        already = [k for k in named if k in log.measured]
        if already:
            flags.append("hallucinated_state_claim:measured_claimed_untested")
            return GradeResult(0.0, flags, {"already_measured": already})
        illegal = [k for k in named if not self._legal(task, k)]
        if illegal:
            flags.append("fabricated_candidate:illegal_under_legality_mask")
            return GradeResult(0.0, flags, {"illegal": illegal})
        hits = [k for k in named if k in top or self._one_step_near(task, k, top)]
        if hits:
            return GradeResult(1.0, flags, {"oracle_verified_hits": hits})
        return GradeResult(0.5, flags, {"direction_only": True})

    def _legal(self, task: str, key: str) -> bool:
        o = self.oracle(task)
        fn = getattr(o, "is_legal", None)
        return fn(key) if fn else True

    def _one_step_near(self, task: str, cand: str, top: set[str]) -> bool:
        return any(_hamming(cand, s) == 1 for s in top)

    # ---------------------------------------- family 3: rationale carrier ---

    def _fam3_rationale(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        present = bool(snap.rationale_spans)
        says_absent = bool(re.search(r"(no rationale|no reasoning|not present|absent)",
                                     text, re.IGNORECASE))
        if present != (not says_absent):
            return GradeResult(0.0, flags, {"presence": present, "reported_absent": says_absent})
        if not present:
            return GradeResult(1.0, flags, {"canonical": "no rationale in state"})
        faithful = any((span in text) or (span[:60] in text) for span in snap.rationale_spans if span)
        return GradeResult(1.0 if faithful else 0.5, flags, {"faithful": faithful})

    # -------------------------------- family 4: identical normalized forms --

    def _fam4_identical(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        norm = getattr(self.oracle(task), "normalize", None) or (lambda k: k)
        forms = [norm(p) for p in snap.proposals]
        dups = sorted({(a, b) for i, a in enumerate(forms) for b in forms[i + 1:] if a == b})
        claims_none = bool(re.search(r"(no duplicate|none|no equivalent)", text, re.IGNORECASE))
        if not dups:
            return GradeResult(1.0 if claims_none or not text else 0.5, flags,
                               {"canonical": "none"})
        found = sum(1 for a, b in dups if a in text and b in text)
        if found == len(dups):
            return GradeResult(1.0, flags, {"pairs": dups})
        if found:
            return GradeResult(0.5, flags, {"canonical": dups})
        return GradeResult(0.0, flags, {"canonical": dups})

    # -------------------------------- family 5: one-factor related pairs ----

    def _fam5_one_factor(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        props = sorted(set(snap.proposals))
        pairs = [(a, b) for i, a in enumerate(props) for b in props[i + 1:]
                 if _hamming(a, b) == 1]
        found = [(a, b) for a, b in pairs if a in text and b in text]
        declared_ok = True
        if snap.declared_groups:
            for g, members in snap.declared_groups.items():
                fams = {_family_key(task, m, self.oracle(task)) for m in members}
                if len(fams) > 1:
                    declared_ok = False
                    flags.append(f"declared_group_inconsistent:{g}")
        if not pairs:
            return GradeResult(1.0 if not found else 0.0, flags, {"canonical": "none"})
        if len(found) == len(pairs) and declared_ok:
            return GradeResult(1.0, flags, {"pairs": pairs})
        if found:
            return GradeResult(0.5, flags, {"found": found, "canonical": pairs})
        return GradeResult(0.0, flags, {"canonical": pairs})

    # ---------------------------------------- family 6: family partition ----

    def _fam6_partition(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        keys = sorted(set(snap.proposals) | set(log.measured))
        o = self.oracle(task)
        families: dict[str, list[str]] = {}
        for k in keys:
            families.setdefault(_family_key(task, k, o), []).append(k)
        count_ok = re.search(rf"\b{len(families)}\b", text) is not None
        members = [m for fam in families.values() for m in fam]
        found = sum(1 for m in members if m in text)
        details = {"canonical_family_count": len(families),
                   "families": families,
                   "declared_groups": snap.declared_groups}
        if found == len(members) and count_ok:
            return GradeResult(1.0, flags, details)
        if found:
            return GradeResult(0.5, flags, details)
        return GradeResult(0.0, flags, details)

    # -------------------------------- family 7: dominance (count argmax) ----

    def _fam7_dominance(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        keys = sorted(set(snap.proposals) | set(log.measured))
        if task == "bh":
            order = _slot_distinct_counts(task, keys)
            canonical, secondary = (order[0][0] if order else None), (order[1][0] if len(order) > 1 else None)
            counts = {k: len(v) for k, v in order}
        elif task == "gb1":
            best = log.best()[0]
            counts = {"local": 0, "intermediate": 0, "distant": 0}
            for k in keys:
                d = _hamming(k, best)
                counts["local" if d == 1 else "intermediate" if 2 <= d <= 3 else "distant"] += 1
            canonical = max(counts, key=lambda c: (counts[c], c))
            secondary = None
        else:  # made: provenance families
            counts = {}
            for k in keys:
                fam = snap.provenance.get(k, "unattributed")
                counts[fam] = counts.get(fam, 0) + 1
            canonical = max(counts, key=lambda c: (counts[c], c))
            secondary = None
        ok = canonical is not None and re.search(
            rf"\b{re.escape(str(canonical))}\b", text) is not None
        return GradeResult(
            1.0 if ok else 0.5 if secondary and str(secondary) in text else 0.0,
            flags, {"canonical_dominant": canonical, "counts": counts})

    # -------------------------------- family 8: marginal coverage gaps ------

    def _fam8_marginal(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        uncovered = self._uncovered_marginals(task, log)
        total_vals = sum(len(v) for v in uncovered.values())
        near_total = self._near_total_coverage(task, log)
        named = sum(1 for vals in uncovered.values() for v in vals if v in text)
        full_slot = any(all(v in text for v in vals) for vals in uncovered.values() if vals)
        if near_total:
            good = bool(re.search(r"(near-total|most|nearly all|mostly un)",
                                  text, re.IGNORECASE))
            return GradeResult(1.0 if good else 0.5, flags,
                               {"near_total_coverage": near_total,
                                "uncovered_counts": {k: len(v) for k, v in uncovered.items()}})
        if full_slot or (uncovered and named >= max(1, total_vals // 4)):
            return GradeResult(1.0, flags, {"uncovered_counts": {k: len(v) for k, v in uncovered.items()}})
        if named:
            return GradeResult(0.5, flags, {})
        return GradeResult(0.0, flags, {"reason": "no verifiable gap identified"})

    def _uncovered_marginals(self, task, log):
        o = self.oracle(task)
        fn = getattr(o, "legal_slot_values", None)
        if fn is None:
            return {}
        slots = fn()
        measured_keys = list(log.measured)
        out = {}
        for slot, legal in slots.items():
            si = "albd".index(slot) if task == "bh" and slot in "albd" else None
            used = set()
            for k in measured_keys:
                parts = k.split("|")
                if si is not None and si < len(parts):
                    used.add(parts[si])
                else:
                    used.add(k)
            out[slot] = sorted(v for v in legal if v not in used)
        return out

    def _near_total_coverage(self, task, log) -> bool:
        uncovered = self._uncovered_marginals(task, log)
        legal_total = sum(len(v) for s in self._legal_slots(task).values() for v in s) \
            if hasattr(self.oracle(task), "legal_slot_values") else 0
        if not legal_total:
            return False
        uncovered_total = sum(len(v) for v in uncovered.values())
        return uncovered_total >= 0.8 * legal_total

    def _legal_slots(self, task):
        fn = getattr(self.oracle(task), "legal_slot_values", None)
        return fn() if fn else {}

    # -------------------------------- family 9: deep gaps -------------------

    def _fam9_deep(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        if task == "made":
            uneval = [p for p in snap.proposals if p not in log.measured]
            named = [p for p in uneval if p in text]
            reasons_faithful = all(r[:40] in text or r in text
                                   for r in (snap.uncertainty_regions or [])) or \
                not snap.uncertainty_regions
            if uneval and len(named) == len(uneval):
                return GradeResult(1.0, flags, {"canonical_unevaluated": uneval})
            if named:
                return GradeResult(0.5, flags, {"found": named, "canonical": uneval})
            return GradeResult(0.0, flags, {"canonical_unevaluated": uneval})
        regions = self._deep_gap_regions(task, log)
        named = [r for r in regions if r in text]
        if regions and named:
            return GradeResult(1.0, flags, {"regions": regions[:20]})
        if named:
            return GradeResult(0.5, flags, {"found": named})
        return GradeResult(0.0, flags, {"canonical_regions_sample": regions[:10]})

    def _deep_gap_regions(self, task, log):
        keys = list(log.measured)
        if task == "bh":
            pairs = set()
            for k in keys:
                parts = k.split("|")
                if len(parts) >= 4:
                    pairs.add((parts[1], parts[2]))  # ligand-base cells
            ligs = self._legal_slots(task).get("l", [])
            bases = self._legal_slots(task).get("b", [])
            return sorted({f"{l}|{b}" for l in ligs for b in bases} - pairs)
        # gb1: position-sets of queried variants
        o = self.oracle(task)
        fn = getattr(o, "position_set", None)
        seen = {fn(k) for k in keys} if fn else set()
        all_sets = getattr(o, "legal_position_sets", lambda: [])()
        return sorted(set(all_sets) - seen)

    # -------------------------------- family 10: boundary direction ---------

    def _fam10_boundary(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        top = self._top_set(task)
        # claimed-low keys that are actually in S* -> contradiction
        contra = [k for k in top if k in text and re.search(
            rf"{re.escape(k)}[^.|\n]{{0,80}}\b(?:low|poor\w*|bad|never|does not work)\b",
            text, re.IGNORECASE)]
        if contra:
            flags.append("fabricated_boundary_contradicts_oracle")
            return GradeResult(0.0, flags, {"contradictions": contra})
        low_side_named = bool(re.search(r"\b(low|poor|bad|avoid)\b", text, re.IGNORECASE))
        high_ok = any(s in text for s in top) or bool(re.search(
            r"\b(high|top|favorable|good)\b", text, re.IGNORECASE))
        if high_ok or low_side_named:
            return GradeResult(0.5, flags, {"direction_consistent": True})
        return GradeResult(0.0, flags, {"reason": "no grounded boundary claim"})

    # -------------------------------- family 11: negative claims partition --

    def _fam11_negatives(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        thr = self._threshold(task)
        measured_low, never_tried = [], []
        for k in snap.negative_claims:
            if k in log.measured:
                (measured_low if log.measured[k] < thr else never_tried).append(k)
            else:
                never_tried.append(k)
        # fabricated measured-failure claims in the answer
        fab = [k for k in re.findall(r"[\w\-]+(?:\|[\w\-]+)*", text)
               if k in snap.negative_claims and k not in log.measured
               and re.search(rf"{re.escape(k)}[^.]{{0,60}}(measured|failed|low)", text, re.IGNORECASE)]
        if fab:
            flags.append("hallucinated_state_claim:measured_failure_without_record")
            return GradeResult(0.0, flags, {"fabricated": fab})
        ok_measured = any(k in text for k in measured_low) if measured_low else True
        ok_never = any(k in text for k in never_tried) if never_tried else True
        score = 1.0 if (ok_measured and ok_never) else 0.5
        return GradeResult(score, flags, {"measured_low": measured_low, "never_tried": never_tried})

    # -------------------------------- family 12: uncertainty carriers -------

    def _fam12_uncertainty(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        if not snap.uncertainty_regions:
            good = bool(re.search(r"(no (uncertainty )?acknowledg|no unknown|does not express)",
                                  text, re.IGNORECASE))
            return GradeResult(1.0 if good else 0.5, flags, {"canonical": "none declared"})
        named = [r for r in snap.uncertainty_regions if r[:40] in text or r in text]
        unmeasured = [r for r in snap.uncertainty_regions
                      if not any(k in r for k in log.measured)]
        if named and unmeasured:
            return GradeResult(1.0, flags, {"verified_unmeasured_regions": unmeasured})
        if named:
            return GradeResult(0.5, flags, {"vague_or_measured_regions": True})
        return GradeResult(0.0, flags, {"reason": "fabricated or missing uncertainty carriers"})

    # -------------------------------- family 13: ranking faithfulness -------

    def _fam13_ranking(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        if not snap.rankings:
            good = bool(re.search(r"(no ranking|no priorit|no ordering|unranked)",
                                  text, re.IGNORECASE))
            return GradeResult(1.0 if good else 0.5, flags, {"canonical": "no ranking"})
        basis, ranked = snap.rankings[0]
        mentions = sum(1 for r in ranked if r in text)
        basis_ok = _basis_faithful(basis, text)
        if mentions == len(ranked) and basis_ok:
            return GradeResult(1.0, flags, {})
        if mentions:
            return GradeResult(0.5, flags, {"basis_faithful": basis_ok})
        return GradeResult(0.0, flags, {"canonical_basis": basis})

    # -------------------------------- family 14: declared limits ------------

    def _fam14_declared(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        declared = snap.limit_declarations
        conflates = bool(re.search(r"untested[^\.]{0,30}unrepresentable", text, re.IGNORECASE))
        if not declared:
            good = bool(re.search(r"(none declared|no (declared )?limit|no representational)",
                                  text, re.IGNORECASE))
            return GradeResult(1.0 if good else 0.5, flags, {"canonical": "none declared"})
        cited = any(d[:40] in text or d in text for d in declared)
        if cited and conflates:
            return GradeResult(0.0, flags, {"reason": "conflates untested with unrepresentable"})
        if cited:
            return GradeResult(1.0, flags, {})
        return GradeResult(0.5, flags, {"real_limit_mischaracterized": True})

    # -------------------------------- family 15: capacity vs content --------

    def _fam15_capacity(self, task, snap, log, ans, flags):
        text = str(ans.get("answer", ""))
        says_capacity = bool(re.search(r"capacity|capable|can (carry|represent|encode)",
                                       text, re.IGNORECASE))
        says_content = bool(re.search(r"(no (actual|such) |none (actually )?(declared|present))|"
                                      r"(actual|instance|pairwise|epistasis|interaction)",
                                       text, re.IGNORECASE))
        if says_capacity and says_content:
            return GradeResult(1.0, flags,
                               {"capacity": "exists", "content_present": bool(snap.proposals)})
        if says_capacity or says_content:
            return GradeResult(0.5, flags, {})
        return GradeResult(0.0, flags,
                           {"reason": "no capacity/content distinction; possible fabrication"})
