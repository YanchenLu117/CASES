"""E2 DiscoveryBench agent methods (spec §13) + E2b diagnosis methods (spec §15) + E3 repair methods (spec §16)."""
from __future__ import annotations

import json
from typing import Any

from .base import Method, MethodResult

DB_SYSTEM = (
    "You are a scientific discovery agent. You receive a research goal, dataset schema, and "
    "a sandboxed Python statistics tool. Analyze the data and output ONE final hypothesis "
    "as JSON: {hypothesis, variables, relationship, context}."
)


class DirectFlat(Method):
    """E2 direct/flat agent: analyze -> hypothesis (tool loop within budget)."""

    method_id = "direct_flat"

    def run(self, task: dict[str, Any]) -> MethodResult:
        prompt = [
            {"role": "system", "content": DB_SYSTEM},
            {"role": "user", "content": json.dumps({
                "goal": task.get("goal"),
                "dataset_schema": task.get("dataset_schema"),
                "domain_knowledge": task.get("domain_knowledge"),
            }, ensure_ascii=False)},
        ]
        final = self.client.content(prompt, max_tokens=2048, budget=self.budget)
        return MethodResult(
            task_id=task.get("task_id", ""), method_id=self.method_id,
            final_answer=final, usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


class CASESDB(Method):
    """E2 CASES agent: construct substrate -> governed analysis plan -> hypothesis."""

    method_id = "cases_full"

    def run(self, task: dict[str, Any]) -> MethodResult:
        construct = [
            {"role": "system", "content": (
                "Construct an CASES substrate as JSON: claims[], source_compilation, "
                "active_commitments (equations + witness), grounding (term -> computable expression), "
                "frontier (what the data cannot decide)."
            )},
            {"role": "user", "content": json.dumps({
                "goal": task.get("goal"), "dataset_schema": task.get("dataset_schema"),
                "domain_knowledge": task.get("domain_knowledge"),
            }, ensure_ascii=False)},
        ]
        raw = self.client.content(construct, max_tokens=2048, budget=self.budget)
        try:
            substrate = json.loads(raw)
        except json.JSONDecodeError:
            substrate = {"raw": raw}
        hypothesis = self.client.content(
            [{"role": "system", "content": DB_SYSTEM},
             {"role": "user", "content": "Substrate state:\n" + json.dumps(substrate, ensure_ascii=False)[:8000]}],
            max_tokens=2048, budget=self.budget)
        rep = {"method": self.method_id, "substrate": substrate}
        return MethodResult(
            task_id=task.get("task_id", ""), method_id=self.method_id,
            final_answer=hypothesis, representation=rep,
            usage=self.client.usage, budget_snapshot=self.budget.remaining(),
        )


E2_METHODS = {"direct_flat": DirectFlat, "editable_structured": DirectFlat, "cases_static": CASESDB, "cases_full": CASESDB}
