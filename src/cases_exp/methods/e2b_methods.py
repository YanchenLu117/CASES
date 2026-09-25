"""E2b blind diagnosis methods (spec §15): predict fault locus + min repair depth."""
from __future__ import annotations

import json
from typing import Any

from .base import Method, MethodResult

DIAG_SYSTEM = (
    "You are a representation-failure diagnostician. You see a scientific task and a "
    "corrupted representation state (you do NOT see the true fault). Predict where the "
    "representation lost decision-relevant information and the minimum repair depth needed. "
    "Output JSON: {predicted_loci: [SRC|COMMIT|RET|MIXED], recommended_min_depth: "
    "GROUNDING_ONLY|COMMITMENT_EDIT|SOURCE_EDIT, evidence_refs: [], certificate_status: str}"
)


class BlindDiagnosisBase(Method):
    method_id = "blind_base"

    def _diagnose(self, task: dict[str, Any], state_blob: str, system: str) -> MethodResult:
        out = self.client.content(
            [{"role": "system", "content": system},
             {"role": "user", "content": "Task: " + json.dumps(task.get("goal", ""), ensure_ascii=False) + "\n\nState:\n" + state_blob[:8000]}],
            max_tokens=1024, budget=self.budget)
        try:
            diag = json.loads(out)
        except json.JSONDecodeError:
            diag = {"raw": out[:500], "predicted_loci": [], "recommended_min_depth": None}
        return MethodResult(
            task_id=task.get("task_id", ""), method_id=self.method_id,
            final_answer=out, diagnosis=diag,
            usage=self.client.usage, budget_snapshot=self.budget.remaining())


class RawFlatDiagnosis(BlindDiagnosisBase):
    method_id = "raw_flat"
    DIAG_SYSTEM = "Look at the task and state and guess what is wrong. " + DIAG_SYSTEM

    def run(self, task: dict[str, Any]) -> MethodResult:
        return self._diagnose(task, json.dumps(task.get("corrupted_state", {}), ensure_ascii=False), self.DIAG_SYSTEM)


class StaticAdequacyDiagnosis(BlindDiagnosisBase):
    method_id = "static_adequacy"

    def run(self, task: dict[str, Any]) -> MethodResult:
        sysmsg = (
            "First compute an aliasing analysis: which claim pairs collide under the active "
            "commitments, and which distinctions are simply absent from the source terms. "
            + DIAG_SYSTEM
        )
        return self._diagnose(task, json.dumps(task.get("corrupted_state", {}), ensure_ascii=False), sysmsg)


class CASESDiagnosis(BlindDiagnosisBase):
    method_id = "cases"

    def run(self, task: dict[str, Any]) -> MethodResult:
        sysmsg = (
            "Use the CASES layered diagnosis: (1) does the source representation contain the "
            "distinction? (2) do active commitments collapse it? (3) does grounding lose it? "
            "Cite support certificates where possible. " + DIAG_SYSTEM
        )
        return self._diagnose(task, json.dumps(task.get("corrupted_state", {}), ensure_ascii=False), sysmsg)


E2B_METHODS = {"raw_flat": RawFlatDiagnosis, "static_adequacy": StaticAdequacyDiagnosis, "cases": CASESDiagnosis}
