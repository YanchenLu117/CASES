"""Method base class: every method produces representation.json under identical budgets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..models.openai_compat import Budget, LLMClient, Usage


@dataclass
class MethodResult:
    task_id: str
    method_id: str
    final_answer: str = ""
    representation: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None
    usage: Usage = field(default_factory=Usage)
    budget_snapshot: dict[str, Any] = field(default_factory=dict)


class Method(ABC):
    method_id: str = "base"

    def __init__(self, client: LLMClient, budget: Budget, config: dict[str, Any] | None = None):
        self.client = client
        self.budget = budget
        self.config = config or {}

    @abstractmethod
    def run(self, task: dict[str, Any]) -> MethodResult:
        """Execute the method on one task. Must respect self.budget; raises BudgetExceededError."""
