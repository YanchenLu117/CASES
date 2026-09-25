"""E1 ResearchBench hypothesis-composition methods (spec §12).

Five critical-path methods sharing the same consumer and budgets:
  direct / fixed_schema / editable_structured (B* candidates)
  cases_static / cases_full
Each follows its 4-call allocation (spec §9.2).

DEVIATIONS 2026-09-15 (implementation-defect repairs, canonical Eq.(1):
NLS^3Construct takes evidence D_t as input):
  - construct/audit stages now receive the background survey (D_t) and the
    construct schema elicits evidence_bindings per claim (frozen substrate
    schema already carries Claim.evidence_bindings);
  - fence-tolerant JSON extraction + schema key reconciliation;
  - normalize content-loss rejection guard;
  - audit input truncation is structural, not a mid-JSON character cut;
  - audit edit fallback: edit_claim miss degrades to add_claim;
  - parse/telemetry counters land in the representation artifact.
The shared consumer prompt (same-consumer rule) is unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import Method, MethodResult

COMPOSER_SYSTEM = (
    "You are a research hypothesis composer. You receive a research question, background, "
    "gold inspirations, and a method representation state. Compose ONE final research hypothesis. "
    "You are judged on coverage of the gold hypothesis key points (0-5)."
)


SYNTHESIS_CONTRACT = (
    "Synthesis rules for the final hypothesis (this method's representation contract): "
    "(1) The hypothesis is a causal chain, not a factor list: state which factor acts on which, "
    "in which direction, and under what condition, following the substrate's active_commitments. "
    "(2) Preserve every distinction/contrast the substrate carries (e.g. condition-A vs condition-B "
    "outcomes) explicitly and with the substrate's specific terms. "
    "(3) A flat enumeration such as \"A and B both influence C\" is a synthesis failure: if the "
    "substrate encodes a link between A and B, the hypothesis must verbalize that link."
)

def consumer_messages(task: dict[str, Any], representation: dict[str, Any]) -> list[dict[str, str]]:
    """Same consumer rule (spec §12.4): identical prompt except the typed state wrapper."""
    state_blob = json.dumps(representation, ensure_ascii=False, indent=1)
    question = task.get("research_question", "")
    background = task.get("background", "")
    inspirations = task.get("gold_inspirations", [])
    if isinstance(inspirations, list):
        ins_blob = "\n".join(f"- {i}" for i in inspirations)
    else:
        ins_blob = str(inspirations)
    return [
        {"role": "system", "content": COMPOSER_SYSTEM},
        {"role": "user", "content": (
            "Research question: " + question + "\n\n"
            "Background: " + background + "\n\n"
            "Gold inspirations:\n" + ins_blob + "\n\n"
            "Method representation state (typed wrapper):\n" + state_blob + "\n\n"
            "Compose the final research hypothesis now. Output only the hypothesis."
        )},
    ]


class DirectRefine(Method):
    """Draft -> critique -> revise -> final (4 calls). No persistent representation."""

    method_id = "direct_refine"

    def run(self, task: dict[str, Any]) -> MethodResult:
        msgs = consumer_messages(task, {"method": "direct", "step": "draft"})
        draft = self.client.content(msgs, max_tokens=1024, budget=self.budget)
        critique = self.client.content(
            [m for m in msgs[:1]] + [{"role": "user", "content": "Critique this hypothesis draft for coverage, specificity, testability:\n" + draft}],
            max_tokens=512, budget=self.budget)
        revised = self.client.content(
            msgs[:1] + [{"role": "user", "content": "Revise per critique. Original:\n" + draft + "\n\nCritique:\n" + critique}],
            max_tokens=1024, budget=self.budget)
        rep = {"method": self.method_id, "draft": draft, "critique": critique, "revised": revised}
        final = self.client.content(consumer_messages(task, rep), max_tokens=1024, budget=self.budget)
        return MethodResult(
            task_id=task.get("sample_id", task.get("task_id", "")),
            method_id=self.method_id, final_answer=final, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


class FixedSchema(Method):
    """Instantiate fixed schema -> refine -> compose -> verify (4 calls)."""

    method_id = "fixed_schema"
    SCHEMA_FIELDS = ["entities", "relations", "mechanism", "moderators", "outcome", "novelty"]

    def run(self, task: dict[str, Any]) -> MethodResult:
        schema_prompt = [
            {"role": "system", "content": "Fill a fixed scientific schema as JSON with keys: " + ", ".join(self.SCHEMA_FIELDS)},
            {"role": "user", "content": "Question: " + task.get("research_question", "") + "\nInspirations: " + json.dumps(task.get("gold_inspirations", []), ensure_ascii=False)},
        ]
        raw = self.client.content(schema_prompt, max_tokens=1024, budget=self.budget)
        _p = _extract_json_obj(raw)
        schema = _p if _p is not None else {"raw": raw}
        rep = {"method": self.method_id, "schema": schema}
        refined = self.client.content(
            [{"role": "system", "content": "Refine the schema; sharpen mechanism and outcome. JSON only."},
             {"role": "user", "content": json.dumps(schema, ensure_ascii=False)}],
            max_tokens=1024, budget=self.budget)
        _p2 = _extract_json_obj(refined)
        schema2 = _reconcile_schema_keys(_p2) if _p2 is not None else schema
        rep["schema"] = schema2
        final = self.client.content(consumer_messages(task, rep), max_tokens=1024, budget=self.budget)
        rep["final_call"] = "compose"
        return MethodResult(
            task_id=task.get("sample_id", task.get("task_id", "")),
            method_id=self.method_id, final_answer=final, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


def _extract_json_obj(text: str):
    """Fence-tolerant JSON extraction: qwen36 wraps JSON in ```json fences on a
    large fraction of calls; bare json.loads hollows the substrate
    (DEVIATIONS 2026-09-15)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def _reconcile_schema_keys(obj: dict) -> dict:
    """Normalize passes rename the frozen schema (claims -> normalized_claims);
    reconcile back so the consumer sees the frozen keys (DEVIATIONS 2026-09-15)."""
    if not isinstance(obj, dict):
        return obj
    if not obj.get("claims") and isinstance(obj.get("normalized_claims"), list):
        obj["claims"] = obj["normalized_claims"]
    if not obj.get("active_commitments") and isinstance(obj.get("commitments"), list):
        obj["active_commitments"] = obj["commitments"]
    for bad in ("source_compilation ", "source compilation", "sourceCompilation"):
        if bad in obj:
            v = obj.pop(bad)
            if not obj.get("source_compilation") and isinstance(v, dict):
                obj["source_compilation"] = v
    return obj


def _truncate_substrate(sub_json: dict, max_claims: int = 12, max_comms: int = 8) -> str:
    """Structural truncation for the audit input: keeps the JSON valid instead of
    cutting the serialized string at an arbitrary character (DEVIATIONS 2026-09-15)."""
    claims = list(sub_json.get("claims") or [])[:max_claims]
    comms = list(sub_json.get("active_commitments") or [])[:max_comms]
    view = {"claims": claims, "active_commitments": comms}
    for k in ("source_compilation", "grounding"):
        v = sub_json.get(k)
        if isinstance(v, dict):
            view[k] = dict(list(v.items())[:16])
    return json.dumps(view, ensure_ascii=False)


class EditableStructured(Method):
    """Structured representation the method may edit between calls (B* candidate)."""

    method_id = "editable_structured"

    def run(self, task: dict[str, Any]) -> MethodResult:
        init = [
            {"role": "system", "content": "Build an editable structured scientific representation as JSON: claims[], evidence_refs[], open_questions[], proposed_edits[]."},
            {"role": "user", "content": "Question: " + task.get("research_question", "")},
        ]
        raw = self.client.content(init, max_tokens=1024, budget=self.budget)
        _p = _extract_json_obj(raw)
        state = _p if _p is not None else {"raw": raw}
        edit = self.client.content(
            [{"role": "system", "content": "Edit the representation to maximize downstream hypothesis quality. Output the full updated JSON."},
             {"role": "user", "content": json.dumps(state, ensure_ascii=False)}],
            max_tokens=1024, budget=self.budget)
        _e = _extract_json_obj(edit)
        if _e is not None:
            state = _reconcile_schema_keys(_e)
        rep = {"method": self.method_id, "state": state}
        final = self.client.content(consumer_messages(task, rep), max_tokens=1024, budget=self.budget)
        return MethodResult(
            task_id=task.get("sample_id", task.get("task_id", "")),
            method_id=self.method_id, final_answer=final, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


class CASESStatic(Method):
    """Construct -> normalize -> compose (3 calls). Governance checks but no revision loop."""

    method_id = "cases_static"
    # H5 S-factor class switch (DEV-20260915-h5arms): the synthesis contract
    # must be absent from the rep BEFORE the consumer call, so the switch is
    # consulted at rep-build time, not popped afterwards.
    payload_contract = True

    CONSTRUCT_SYSTEM = (
        "Construct an CASES source representation from the research question AND the "
        "background survey evidence as JSON with keys: "
        "claims (list of {claim_id, text, evidence_bindings: [short verbatim quotes from the "
        "background survey that support the claim]}), "
        "source_compilation (claim_id -> source term), "
        "active_commitments (list of {commitment_id, equations: [[term_a, term_b]], witness}), "
        "grounding (term -> computational expression). "
        "Cover 4-8 claims spanning: the phenomena and causal factors, the methodologies and "
        "experimental/analysis designs present in the evidence, the measures or datasets used, "
        "and distinctions the question turns on. Every claim must bind to at least one evidence quote. "
        "When the evidence supports a causal or conditional link between two factors (X limits/activates/"
        "modulates Y, or Y holds only under condition C), encode that link explicitly: state it inside the "
        "claims and bind the two factors with an active_commitment equation pair whose witness states the "
        "direction/condition (e.g. term_a=initial depositional texture, term_b=diagenetic pathway). "
        "Never leave two interacting factors as unrelated parallel claims."
    )

    def _construct_normalize(self, task: dict[str, Any]):
        question = task.get("research_question", "")
        background = task.get("background", "") or task.get("background_survey", "")

        construct_prompt = [
            {"role": "system", "content": self.CONSTRUCT_SYSTEM},
            {"role": "user", "content": (
                "Research question: " + question + "\n\n"
                "Background survey (evidence):\n" + background + "\n\n"
                "Inspirations: " + json.dumps(task.get("gold_inspirations", []), ensure_ascii=False)
            )},
        ]
        raw = self.client.content(construct_prompt, max_tokens=2048, budget=self.budget)
        _p = _extract_json_obj(raw)
        construct_parse_failed = _p is None
        sub_json = _reconcile_schema_keys(_p) if _p is not None else {"raw": raw}
        # governed activation: only well-formed commitments enter
        activated = []
        for c in sub_json.get("active_commitments", []):
            eqs = c.get("equations", [])
            if all(isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) and x for x in e) for e in eqs):
                activated.append(c)
        normalize_system = (
            "Normalize the representation: deduplicate claims, canonicalize terms, verify each "
            "commitment has a witness. Keep every claim's evidence_bindings. Output full JSON."
        )
        if construct_parse_failed:
            normalize_system += (
                " The previous construct output failed to parse as JSON; recover the substrate "
                " faithfully from the RAW text field instead of inventing new content."
            )
        normalize = self.client.content(
            [{"role": "system", "content": normalize_system},
             {"role": "user", "content": json.dumps({**sub_json, "active_commitments": activated}, ensure_ascii=False)}],
            max_tokens=2048, budget=self.budget)
        normalize_parsed = False
        candidate = _extract_json_obj(normalize)
        if candidate is not None:
            candidate = _reconcile_schema_keys(candidate)
            _had_c, _had_k = len(sub_json.get("claims") or []), len(sub_json.get("active_commitments") or [])
            _got_c, _got_k = len(candidate.get("claims") or []), len(candidate.get("active_commitments") or [])
            if (_had_c and not _got_c) or (_had_k and not _got_k):
                pass  # normalize rewrote the tree and lost content: reject, keep construct substrate
            else:
                sub_json = candidate
                normalize_parsed = True
        if normalize_parsed:
            # W9: re-apply the deterministic well-formedness filter after the
            # LLM normalize pass — normalization must not smuggle in
            # ill-formed commitments — and recompute the activated list.
            re_gated = []
            for c in sub_json.get("active_commitments", []):
                eqs = c.get("equations", [])
                if all(isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) and x for x in e) for e in eqs):
                    re_gated.append(c)
            activated = re_gated
            sub_json["active_commitments"] = re_gated
        self._telemetry = {"construct_parse_failed": construct_parse_failed,
                           "normalize_parsed": normalize_parsed}
        return sub_json, activated

    def run(self, task: dict[str, Any]) -> MethodResult:
        sub_json, activated = self._construct_normalize(task)
        rep = {"method": self.method_id, "substrate": sub_json, "governed_activations": len(activated),
               "normalize_parsed": getattr(self, "_telemetry", {}).get("normalize_parsed", True)}
        if self.payload_contract:
            rep["synthesis_contract"] = SYNTHESIS_CONTRACT
        final = self.client.content(consumer_messages(task, rep), max_tokens=2048, budget=self.budget)
        return MethodResult(
            task_id=task.get("sample_id", task.get("task_id", "")),
            method_id=self.method_id, final_answer=final, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


class CASESFull(CASESStatic):
    """Construct (evidence-grounded) -> normalize -> diagnose + ONE surgical
    governed edit (applied deterministically in code) -> compose (4 calls, all
    functional).

    The audit stage sees the research question and a background digest so
    "decision-relevant information loss" is diagnosed against the evidence,
    per canonical Eq.(1) Construct(tau, D_t, ...). The edit is a patch spec,
    not a regeneration: the auditor returns a single add/edit instruction and
    the substrate is patched in code, so no original content can be lost by
    LLM rewriting."""

    method_id = "cases_full"

    def run(self, task: dict[str, Any]) -> MethodResult:
        sub_json, activated = self._construct_normalize(task)
        total_out = self.budget.max_total_output_tokens or 8192
        diag_max = 1024 if total_out <= 8192 else 8192
        question = task.get("research_question", "")
        background = task.get("background", "") or task.get("background_survey", "")
        audit_prompt = [
            {"role": "system", "content": (
                "Audit this scientific solution-space representation for decision-relevant "
                "information loss at three boundaries: source (missing distinction), commitment "
                "(collapsed distinction), grounding (lost in computation). The representation is "
                "insufficient if the evidence supports a distinction the substrate does not carry. "
                "Pick the single highest-impact loss and specify ONE surgical edit. Output ONLY JSON: "
                "{\"locus\": \"SRC|COMMIT|RET\", \"evidence\": str (<=200 chars), \"edit\": {"
                "\"action\": \"add_claim|edit_claim|add_commitment\", "
                "\"claim_id\": str, \"text\": str, "
                "\"equations\": [[term, term]], \"witness\": str}}"
            )},
            {"role": "user", "content": (
                "Research question: " + question + "\n\n"
                "Background (evidence digest):\n" + background[:2000] + "\n\n"
                "Substrate:\n" + _truncate_substrate(sub_json)
            )},
        ]
        raw = self.client.content(audit_prompt, max_tokens=diag_max, budget=self.budget)
        audit = _extract_json_obj(raw) or {"raw": raw[:500], "edit": None}
        edit = audit.get("edit") if isinstance(audit, dict) else None
        applied = "none"
        edit_error = None
        if isinstance(edit, dict):
            action = edit.get("action")
            try:
                if action == "add_claim" and edit.get("text"):
                    cid = str(edit.get("claim_id") or f"audit_claim_{len(sub_json.get('claims', [])) + 1}")
                    claims = sub_json.setdefault("claims", [])
                    if not any(c.get("claim_id") == cid for c in claims):
                        claims.append({"claim_id": cid, "text": str(edit["text"])})
                        applied = "add_claim:" + cid
                elif action == "edit_claim" and edit.get("text"):
                    matched = False
                    for c in sub_json.get("claims", []):
                        if c.get("claim_id") == edit.get("claim_id"):
                            c["text"] = str(edit["text"])
                            applied = "edit_claim:" + str(c.get("claim_id"))
                            matched = True
                            break
                    if not matched:
                        # auditor hallucinated the claim_id: keep the content as an
                        # added claim instead of silently dropping the edit
                        claims = sub_json.setdefault("claims", [])
                        cid = str(edit.get("claim_id") or f"audit_claim_{len(claims) + 1}")
                        claims.append({"claim_id": cid, "text": str(edit["text"])})
                        applied = "edit_miss->add:" + cid
                elif action == "add_commitment" and edit.get("equations"):
                    cid = str(edit.get("claim_id") or "audit_commit_1")
                    coms = sub_json.setdefault("active_commitments", [])
                    eqs = [[str(a), str(b)] for a, b in edit["equations"]
                           if isinstance(a, (str, int)) and isinstance(b, (str, int))][:4]
                    if eqs and not any(x.get("commitment_id") == cid for x in coms):
                        coms.append({"commitment_id": cid, "equations": eqs,
                                     "witness": {"type": "declaration", "note": str(edit.get("witness", "audit"))[:80]}})
                        activated.append({"commitment_id": cid, "equations": eqs})
                        applied = "add_commitment:" + cid
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                applied = "failed"
                edit_error = repr(e)[:200]
        # W9 verification: audit patches are already filtered deterministically
        # (equations coerced via [[str, str]] with len<=4); re-apply the
        # well-formedness filter defensively over the final substrate.
        final_gated = []
        for c in sub_json.get("active_commitments", []):
            eqs = c.get("equations", [])
            if all(isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) and x for x in e) for e in eqs):
                final_gated.append(c)
        sub_json["active_commitments"] = final_gated
        activated = [c for c in final_gated]
        telemetry = getattr(self, "_telemetry", {})
        rep = {"method": self.method_id, "substrate": sub_json,
               "governed_activations": len(activated), "diagnosis": audit, "edit_applied": applied,
               "construct_parse_failed": telemetry.get("construct_parse_failed", False),
               "normalize_parsed": telemetry.get("normalize_parsed", True)}
        if self.payload_contract:
            rep["synthesis_contract"] = SYNTHESIS_CONTRACT
        if edit_error:
            rep["edit_error"] = edit_error
        final = self.client.content(consumer_messages(task, rep), max_tokens=2048, budget=self.budget)
        result = MethodResult(
            task_id=task.get("sample_id", task.get("task_id", "")),
            method_id=self.method_id, final_answer=final, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )
        result.diagnosis = audit
        return result


class CASESFullNC(CASESFull):
    """H5 factorial arm (C=ON, S=OFF): full pipeline, v5 causal encoding,
    synthesis contract absent from the consumer payload. Only the S factor
    differs from cases_full (DEVIATIONS 2026-09-15, user sign-off)."""

    method_id = "h5_nc"
    payload_contract = False


class CASESFullNCE(CASESFull):
    """H5 factorial arm (C=OFF, S=ON): full pipeline, v4 construct text
    (verbatim, pre-04d1834f), synthesis contract present. Only the C factor
    differs from cases_full. Shared code path guarantees arm parity."""

    method_id = "h5_nce"

    CONSTRUCT_SYSTEM = (
            "Construct an CASES source representation from the research question AND the "
            "background survey evidence as JSON with keys: "
            "claims (list of {claim_id, text, evidence_bindings: [short verbatim quotes from the "
            "background survey that support the claim]}), "
            "source_compilation (claim_id -> source term), "
            "active_commitments (list of {commitment_id, equations: [[term_a, term_b]], witness}), "
            "grounding (term -> computational expression). "
            "Cover 4-8 claims spanning: the phenomena and causal factors, the methodologies and "
            "experimental/analysis designs present in the evidence, the measures or datasets used, "
            "and distinctions the question turns on. Every claim must bind to at least one evidence quote."
        )




E1_METHODS = {
    "direct_refine": DirectRefine,
    "fixed_schema": FixedSchema,
    "editable_structured": EditableStructured,
    "cases_static": CASESStatic,
    "cases_full": CASESFull,
    # H5 factorial arms (hard tercile only; DEVIATIONS 2026-09-15, user sign-off)
    "h5_nc": CASESFullNC,
    "h5_nce": CASESFullNCE,
}
