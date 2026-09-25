"""DiscoveryBench adapter: task loading (real + synth), query-level enumeration,
gold extraction, and evaluator invocation wrapper.

Evaluator is the pinned repo's discovery_eval.py (LLM-based dimension scorer).
The judge LLM for evaluation is qwen38-27b via the local gateway
(JUDGE_EVALUATOR_CONFIG.yaml, REJUDGED namespace for DB judge calls).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "discoverybench"


@dataclass
class DBTask:
    task_id: str
    split: str
    real_or_synth: str
    domain: str
    metadata_path: Path
    queries: list  # list of query-groups; each group is a list of {qid, question, question_type}
    datasets: list
    domain_knowledge: str = ""
    workflow_tags: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def dataset_paths(self) -> list[str]:
        out = []
        for d in self.datasets:
            p = Path(d.get("path", ""))
            if not p.is_absolute():
                p = self.metadata_path.parent / p
            out.append(str(p))
        return out


def _read_json_robust(mp: Path) -> dict:
    """Some synth metadata files contain non-UTF8 bytes (upstream artifact);
    fall back to latin-1 rather than crash the census."""
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(mp.read_text(encoding="latin-1"))


def _load_metadata(mp: Path, split: str, rors: str) -> DBTask:
    d = _read_json_robust(mp)
    return DBTask(
        task_id=f"{rors}/{mp.parent.name}/{mp.stem}",
        split=split,
        real_or_synth=rors,
        domain=d.get("domain", ""),
        metadata_path=mp,
        queries=(
            # synth: flat single-question dicts -> normalize to 1-question groups
            [[q] for q in d.get("queries", []) if isinstance(q, dict)]
            if d.get("queries") and isinstance(d.get("queries")[0], dict)
            else d.get("queries", [])
        ),
        datasets=d.get("datasets", []),
        domain_knowledge=d.get("domain_knowledge", ""),
        workflow_tags=d.get("workflow_tags", ""),
        raw=d,
    )


def load_tasks(rors: str = "real", split: str = "test") -> list[DBTask]:
    root = VENDOR / "discoverybench" / rors / split
    return sorted(
        (_load_metadata(mp, split, rors) for mp in root.rglob("metadata*.json")),
        key=lambda t: t.task_id,
    )


_ANSWER_KEY: dict[str, dict[str, str]] = {}


def _load_answer_key(rors: str) -> dict[str, dict[str, str]]:
    """Join key: (dataset_dir, metadata_id, query_group_idx) -> gold_hypo."""
    if rors in _ANSWER_KEY:
        return _ANSWER_KEY[rors]
    import csv
    key: dict[str, dict[str, str]] = {}
    path = VENDOR / "eval" / f"answer_key_{rors}.csv"
    if path.exists():
        for row in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            k = (row["dataset"], row["metadataid"])
            key.setdefault(k, {})[row["query_id"]] = row["gold_hypo"]
    _ANSWER_KEY[rors] = key
    return key


def gold_hypothesis(task: DBTask, query_group_idx: int = 0) -> str:
    """Gold hypothesis from the official answer key (evaluator-private).

    real: key rows are (dataset_dir, metadata_id, query_group_idx).
    synth: queries is a flat list of single-question dicts with global qid;
    key rows are (dataset_dir, metadata_id, qid).
    """
    key = _load_answer_key(task.real_or_synth)
    meta_id = task.metadata_path.stem.replace("metadata_", "")
    entry = key.get((task.metadata_path.parent.name, meta_id), {})
    if not entry:
        return ""
    if str(query_group_idx) in entry:
        return entry[str(query_group_idx)]
    if task.real_or_synth == "synth":
        # join by qid of the query at flat index query_group_idx
        flat = [q for qg in task.queries if isinstance(qg, list) for q in qg] if task.queries and isinstance(task.queries[0], list) else task.queries
        if query_group_idx < len(flat):
            qid = flat[query_group_idx].get("qid")
            if str(qid) in entry:
                return entry[str(qid)]
        # single-gold fallback
        for v in entry.values():
            return v
    if "0" in entry:
        return entry["0"]
    return ""


def gold_workflow(task: DBTask, query_group_idx: int = 0) -> str:
    hyp = task.raw.get("hypotheses", {})
    mains = hyp.get("main", [])
    if query_group_idx < len(mains) and isinstance(mains[query_group_idx], dict):
        wf = mains[query_group_idx].get("workflow", "")
        if isinstance(wf, str):
            return wf
    return ""


def question_text(task: DBTask, query_group_idx: int = 0) -> str:
    """The final (decision) question of a query group is the task question."""
    qg = task.queries[query_group_idx] if query_group_idx < len(task.queries) else []
    texts = [q.get("question", "") for q in qg if q.get("question")]
    return texts[-1] if texts else ""


def agent_input_payload(task: DBTask, query_group_idx: int = 0) -> dict:
    """PUBLIC payload for agent methods (no gold)."""
    qg = task.queries[query_group_idx] if query_group_idx < len(task.queries) else []
    return {
        "task_id": task.task_id,
        "query_group_idx": query_group_idx,
        "question": question_text(task, query_group_idx),
        "all_questions": [q.get("question", "") for q in qg],
        "domain": task.domain,
        "domain_knowledge": task.domain_knowledge,
        "workflow_tags": task.workflow_tags,
        "dataset_descriptions": [
            {k: d.get(k) for k in ("name", "description", "columns") if k in d}
            for d in task.datasets
        ],
        "dataset_paths": task.dataset_paths,
        "metadata_path": str(task.metadata_path),
    }


_EVAL_TMP_JSON = Path("/tmp/cases_db_eval_last.json")


def run_evaluator(task: DBTask, pred_hypo: str, pred_workflow: str,
                  query_group_idx: int = 0, judge_model: str = "qwen38-27b",
                  timeout: float = 300.0) -> dict:
    """Invoke pinned discovery_eval.py. Returns parsed eval json.

    The evaluator reads OPENAI_API_KEY / OPENAI_BASE_URL env for its LLM judge;
    callers must point these at the local gateway before calling.
    """
    cmd = [
        sys.executable, str(VENDOR / "discovery_eval.py"),
        "--gold_hypo", gold_hypothesis(task, query_group_idx),
        "--gold_workflow", gold_workflow(task, query_group_idx),
        "--pred_hypo", pred_hypo,
        "--pred_workflow", pred_workflow,
        "--metadata_path", str(task.metadata_path),
        "--metadata_type", "real" if task.real_or_synth == "real" else "synth",
        "--eval_output_path", str(_EVAL_TMP_JSON),
        question_text(task, query_group_idx),
    ]
    # unique output file per invocation: concurrent evaluator calls must not
    # race on a shared tmp path (the evaluator APPENDS to its output file)
    eval_out = Path(f"/tmp/cases_db_eval_{os.getpid()}_{id(task):x}.json")
    cmd[cmd.index("--eval_output_path") + 1] = str(eval_out)
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "sk-cases-v9-gateway")
    env.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9300/v1")
    if eval_out.exists():
        eval_out.unlink()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(VENDOR), env=env)
    out = proc.stdout.strip()
    # Prefer the dedicated output file: stdout also carries evaluator debug prints
    # (judge answers etc.) that break naive brace-spanning json extraction.
    if eval_out.exists():
        try:
            result = {"eval": json.loads(eval_out.read_text()),
                    "returncode": proc.returncode,
                    "stderr_tail": proc.stderr[-500:]}
            eval_out.unlink(missing_ok=True)
            return result
        except json.JSONDecodeError:
            pass
    eval_out.unlink(missing_ok=True)
    start = out.find("{")
    end = out.rfind("}")
    if start >= 0 and end > start:
        try:
            return {"eval": json.loads(out[start:end + 1]),
                    "returncode": proc.returncode,
                    "stderr_tail": proc.stderr[-500:]}
            eval_out.unlink(missing_ok=True)
            return result
        except json.JSONDecodeError:
            pass
    eval_out.unlink(missing_ok=True)
    return {"eval": None, "returncode": proc.returncode,
            "stdout_tail": out[-500:], "stderr_tail": proc.stderr[-500:]}
