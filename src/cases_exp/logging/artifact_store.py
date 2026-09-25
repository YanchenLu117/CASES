"""Deterministic run IDs and immutable artifact store (spec §4)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def run_id(
    experiment_id: str,
    benchmark_commit: str,
    dataset_hash: str,
    task_id: str,
    method_config_hash: str,
    model_config_hash: str,
    judge_config_hash: str,
    seed: int,
) -> str:
    payload = "".join(
        [experiment_id, benchmark_commit, dataset_hash, task_id, method_config_hash, model_config_hash, judge_config_hash, str(seed)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


RUN_REQUIRED_FILES = [
    "request.json", "response.json", "method_state.json", "budget.json",
    "timing.json", "tool_trace.jsonl", "evaluator_input.json", "evaluator_output.json",
    "stdout.log", "stderr.log", "checksums.json",
]

CASES_EXTRA_FILES = [
    "source_state.json", "active_commitments.json", "computational_state.json",
    "frontier_certificates.json", "collapse_supports.json", "repair_certificate.json",
    "migration_audit.json",
]


class RunStore:
    """Atomic write-once run directories under runs/<experiment>/<model>/<method>/<task_id>/<run_id>/."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, experiment: str, model: str, method: str, task_id: str, rid: str) -> Path:
        p = self.root / experiment / model / method / task_id / rid
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, run_dir: Path, name: str, obj: Any) -> Path:
        tmp = run_dir / f".{name}.tmp"
        final = run_dir / name
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp, final)  # atomic
        return final

    def write_text(self, run_dir: Path, name: str, text: str) -> Path:
        tmp = run_dir / f".{name}.tmp"
        final = run_dir / name
        tmp.write_text(text)
        os.replace(tmp, final)
        return final

    def checksums(self, run_dir: Path) -> dict[str, str]:
        out = {}
        for f in sorted(run_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        (run_dir / "checksums.json").write_text(json.dumps(out, indent=2))
        return out

    def is_complete(self, run_dir: Path, cases: bool = False) -> list[str]:
        required = RUN_REQUIRED_FILES + (CASES_EXTRA_FILES if cases else [])
        return [f for f in required if not (run_dir / f).exists()]
