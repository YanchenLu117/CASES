#!/usr/bin/env python
"""Thin wrapper around the official ResearchBench ``score-generate`` judge CLI.

The official scorer (github.com/ankitala/ResearchBench, MIT) is the single
authority for the ``matched_score`` (0-5) prompt and its parsing; this wrapper
only wires file paths and endpoint configuration from the environment.
Temperature is fixed at 0.0 inside the official worker, and predictions that
already carry ``matched_score`` are skipped (resumable).

Install the upstream package first::

    pip install git+https://github.com/ankitala/ResearchBench

Usage::

    python scripts/run_rb_judge.py --pred generations.jsonl --data tasks.jsonl --out scores.json

Inputs:
  --pred  JSONL, one row per generation: {"sample_id": str, "final_hypothesis": str}
  --data  JSONL, one row per task:       {"sample_id": str, "gold_hypothesis": str,
                                          "gold_key_points": [str, ...]}
  --out   output JSON (default: <pred>.score.json, written by the official CLI)

Judge configuration is never hardcoded (the paper setting used qwen38-27b):
  CASES_JUDGE_BASE_URL      OpenAI-compatible judge endpoint (or --base-url)
  CASES_JUDGE_API_KEY       judge API key (or --api-key)
  CASES_JUDGE_MODEL         judge model id (or --judge-model; default qwen38-27b)
  CASES_RESEARCHBENCH_BIN   override the researchbench executable (default: PATH lookup)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_JUDGE_MODEL = "qwen38-27b"


def build_command(
    program: str,
    pred: str,
    data: str,
    judge_model: str,
    base_url: str | None,
    api_key: str | None,
    concurrency: int,
    out: str | None,
) -> list[str]:
    """Assemble the official-CLI argv (no shell interpolation)."""
    cmd = [
        program, "score-generate",
        "--pred", pred,
        "--data", data,
        "--judge-model", judge_model,
    ]
    if base_url:
        cmd += ["--base-url", base_url]
    if api_key:
        cmd += ["--api-key", api_key]
    if out:
        cmd += ["--out", out]
    if concurrency and concurrency != 15:
        cmd += ["--concurrency", str(int(concurrency))]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred", required=True, help="predictions JSONL (sample_id, final_hypothesis)")
    ap.add_argument("--data", required=True, help="task JSONL (sample_id, gold_hypothesis, gold_key_points)")
    ap.add_argument("--out", help="output JSON (default: <pred>.score.json)")
    ap.add_argument("--judge-model", default=os.environ.get("CASES_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    ap.add_argument("--base-url", default=os.environ.get("CASES_JUDGE_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("CASES_JUDGE_API_KEY", ""))
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the resolved command and exit"
    )
    args = ap.parse_args()

    program = os.environ.get("CASES_RESEARCHBENCH_BIN") or shutil.which("researchbench")
    if not program:
        print(
            "the official 'researchbench' CLI was not found on PATH.\n"
            "Install it with:  pip install git+https://github.com/ankitala/ResearchBench\n"
            "(or point CASES_RESEARCHBENCH_BIN at the executable)",
            file=sys.stderr,
        )
        return 2

    cmd = build_command(
        program,
        args.pred,
        args.data,
        args.judge_model,
        args.base_url or None,
        args.api_key or None,
        args.concurrency,
        args.out,
    )
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
