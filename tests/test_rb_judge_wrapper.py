"""scripts/run_rb_judge.py — command assembly only (no network, no vendor import)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "run_rb_judge", _HERE.parent / "scripts" / "run_rb_judge.py"
)


def _mod():
    assert _SPEC and _SPEC.loader
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def test_build_command_defaults():
    mod = _mod()
    cmd = mod.build_command("rb", "p.jsonl", "d.jsonl", "qwen38-27b", None, None, 15, None)
    assert cmd == [
        "rb", "score-generate",
        "--pred", "p.jsonl",
        "--data", "d.jsonl",
        "--judge-model", "qwen38-27b",
    ]


def test_build_command_full():
    mod = _mod()
    cmd = mod.build_command(
        "rb", "p.jsonl", "d.jsonl", "m1", "http://x/v1", "k", 8, "o.json"
    )
    assert cmd == [
        "rb", "score-generate",
        "--pred", "p.jsonl",
        "--data", "d.jsonl",
        "--judge-model", "m1",
        "--base-url", "http://x/v1",
        "--api-key", "k",
        "--out", "o.json",
        "--concurrency", "8",
    ]


def test_default_judge_model():
    mod = _mod()
    assert mod.DEFAULT_JUDGE_MODEL == "qwen38-27b"


if __name__ == "__main__":
    sys.exit(0)
