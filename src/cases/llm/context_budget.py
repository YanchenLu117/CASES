"""Context-window budget precheck — the qwen35 lesson, productized.

2026-09-01 feasibility probe (E-line, qwen3.5-35b): 48% of real final-session
prompts exceeded 28k tokens against a 32768 window, which would have turned
into non-transient vLLM 400s on half the grid.  The tripwire caught it *after*
launch; this module moves the check *before* launch.

Protocol (dispatch gate 1b):

1. collect real session transcripts from the layer's smoke runs
   (``all_*.json`` / prompt logs — the same artifacts the E-line probe used);
2. estimate tokens with the chars/4 heuristic (calibrated against the
   2026-09-01 probe; swap in a real tokenizer when one is available);
3. compare p90 / max against ``max_model_len - max_tokens``;
4. REFUSE the launch (exit 42, matching the disk-gate refuse convention)
   when the projected p90 does not fit, printing the numbers that decided it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TokenStats:
    """Prompt-size distribution over measured sessions."""

    n: int
    p50: int
    p90: int
    max: int

    def fits(self, max_model_len: int, max_tokens: int, *, headroom: float = 0.05) -> bool:
        """True when p90 + generation budget fits the window with headroom."""
        budget = max_model_len * (1.0 - headroom) - max_tokens
        return self.p90 <= budget

    def refusal_reason(self, max_model_len: int, max_tokens: int, *, headroom: float = 0.05) -> str:
        budget = max_model_len * (1.0 - headroom) - max_tokens
        over = self.p90 - budget
        return (
            f"context budget REFUSED: p90 prompt {self.p90} tokens exceeds "
            f"usable window {budget:.0f} (max_model_len={max_model_len}, "
            f"max_tokens={max_tokens}, headroom={headroom:.0%}) by {over:.0f} tokens; "
            f"fix by raising max_model_len, trimming prompts, or reducing max_tokens"
        )


def estimate_tokens(text: str) -> int:
    """chars/4 heuristic (2026-09-01 probe calibration)."""
    return math.ceil(len(text) / 4) if text else 0


def measure_texts(texts: list[str]) -> TokenStats:
    """Token-stats over a list of prompt texts."""
    if not texts:
        raise ValueError("no sessions to measure — run the smoke batch first")
    toks = sorted(estimate_tokens(t) for t in texts)

    def pct(p: float) -> int:
        idx = min(len(toks) - 1, max(0, math.ceil(p * len(toks)) - 1))
        return toks[idx]

    return TokenStats(n=len(toks), p50=pct(0.50), p90=pct(0.90), max=toks[-1])


def collect_prompts(paths: list[str | Path]) -> list[str]:
    """Extract prompt texts from smoke-run session artifacts.

    Accepts ``all_*.json``-style session dumps where the message history lives
    under ``messages`` / ``history`` / ``prompt`` keys (the shapes the E-line
    drivers write); plain-text files are read whole.
    """
    import json

    out: list[str] = []
    for p in map(Path, paths):
        if not p.exists():
            continue
        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            out.extend(_prompts_from(data))
        else:
            out.append(p.read_text(encoding="utf-8", errors="replace"))
    return out


def _prompts_from(data) -> list[str]:
    if isinstance(data, dict):
        for key in ("messages", "history"):
            msgs = data.get(key)
            if isinstance(msgs, list):
                return [
                    m.get("content", "") if isinstance(m, dict) else str(m)
                    for m in msgs
                ]
        if isinstance(data.get("prompt"), str):
            return [data["prompt"]]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "content" in v[0]:
                return [m.get("content", "") for m in v]
    if isinstance(data, list):  # a list of sessions
        return [t for s in data for t in _prompts_from(s)]
    return []


def check_layer(
    prompts: list[str],
    max_model_len: int,
    max_tokens: int,
    *,
    headroom: float = 0.05,
) -> tuple[bool, TokenStats, str]:
    """Dispatch gate 1b: (fits, stats, reason). reason is human-readable."""
    stats = measure_texts(prompts)
    if stats.fits(max_model_len, max_tokens, headroom=headroom):
        return True, stats, (
            f"context budget OK: p50={stats.p50} p90={stats.p90} max={stats.max} "
            f"vs usable {max_model_len * (1 - headroom) - max_tokens:.0f}"
        )
    return False, stats, stats.refusal_reason(max_model_len, max_tokens, headroom=headroom)
