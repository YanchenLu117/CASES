"""Unified OpenAI-compatible LLM client with budget accounting.

Works against: any OpenAI-compatible chat-completions server (vLLM, llama-server,
and any OpenAI-compatible endpoint. Deterministic accounting of calls/tokens per run.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    request_failures: int = 0
    retries: int = 0

    def add(self, other: "Usage") -> None:
        self.llm_calls += other.llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.tool_calls += other.tool_calls
        self.request_failures += other.request_failures
        self.retries += other.retries

    def to_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "request_failures": self.request_failures,
            "retries": self.retries,
        }


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    max_llm_calls: int
    max_total_input_tokens: int
    max_total_output_tokens: int
    max_tool_calls: int = 0
    max_wall_seconds: float = 1e9
    max_solver_cpu_seconds: float = 1e9
    usage: Usage = field(default_factory=Usage)
    _t0: float = field(default_factory=time.time)

    def check(self) -> None:
        u = self.usage
        if u.llm_calls > self.max_llm_calls:
            raise BudgetExceededError(f"llm_calls {u.llm_calls} > {self.max_llm_calls}")
        if u.input_tokens > self.max_total_input_tokens:
            raise BudgetExceededError(f"input_tokens {u.input_tokens} > {self.max_total_input_tokens}")
        if u.output_tokens > self.max_total_output_tokens:
            raise BudgetExceededError(f"output_tokens {u.output_tokens} > {self.max_total_output_tokens}")
        if u.tool_calls > self.max_tool_calls:
            raise BudgetExceededError(f"tool_calls {u.tool_calls} > {self.max_tool_calls}")
        if time.time() - self._t0 > self.max_wall_seconds:
            raise BudgetExceededError("wall seconds exceeded")

    def remaining(self) -> dict[str, float]:
        return {
            "llm_calls": self.max_llm_calls - self.usage.llm_calls,
            "input_tokens": self.max_total_input_tokens - self.usage.input_tokens,
            "output_tokens": self.max_total_output_tokens - self.usage.output_tokens,
        }


class LLMClient:
    """Synchronous OpenAI-compatible chat client with retry + accounting hooks.

    reasoning models (GLM-5.3-Flash) return content plus reasoning_content;
    we always read .content and tolerate empty reasoning.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        max_retry: int = 5,
        backoff_base: float = 2.0,
        concurrency: int = 12,
        usage: Usage | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retry = max_retry
        self.backoff_base = backoff_base
        self.usage = usage or Usage()

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
        budget: Budget | None = None,
    ) -> dict[str, Any]:
        if budget is not None:
            budget.check()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            body.update(extra)
        data = json.dumps(body).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retry + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode())
                u = out.get("usage", {})
                self.usage.llm_calls += 1
                self.usage.input_tokens += u.get("prompt_tokens", 0) or 0
                self.usage.output_tokens += u.get("completion_tokens", 0) or 0
                if budget is not None:
                    budget.usage.add(self.usage_snapshot(out, u))
                    budget.check()
                out["_latency_s"] = round(time.time() - t0, 3)
                return out
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                self.usage.request_failures += 1
                if attempt < self.max_retry:
                    self.usage.retries += 1
                    time.sleep(self.backoff_base**attempt)
        raise RuntimeError(f"LLM request failed after {self.max_retry + 1} attempts: {last_err}")

    @staticmethod
    def usage_snapshot(out: dict, u: dict) -> Usage:
        return Usage(
            llm_calls=1,
            input_tokens=u.get("prompt_tokens", 0) or 0,
            output_tokens=u.get("completion_tokens", 0) or 0,
        )

    def content(self, messages: list[dict[str, str]], **kw: Any) -> str:
        out = self.chat(messages, **kw)
        c = out["choices"][0]["message"].get("content")
        if c is None:
            # finish_reason==length with only reasoning emitted; retry with larger budget
            raise RuntimeError("empty content (reasoning model hit max_tokens before content)")
        return c
