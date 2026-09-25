"""E2 pilot runner: DB hypothesis generation methods scored by official evaluator (spec §13/§14).

Methods: direct_flat (Raw/Flat negative control) + cases_full (CASESDB).
Equal budget per spec E2-DB starting values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cases_exp.benchmarks.discoverybench import (
    load_tasks, gold_hypothesis, question_text, agent_input_payload, run_evaluator)
from cases_exp.models.openai_compat import LLMClient, Budget
from cases_exp.methods.e2_methods import DirectFlat, CASESDB

GATEWAY_URL = "http://127.0.0.1:9300/v1"
GATEWAY_KEY = "sk-cases-v9-gateway"
E2_BUDGET = dict(max_llm_calls=10, max_total_input_tokens=100000,
                 max_total_output_tokens=12000, max_tool_calls=12)

METHODS = {"direct_flat": DirectFlat, "cases_full": CASESDB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rors", default="synth")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="runs/phase0/e2")
    ap.add_argument("--model", default="qwen36-35b")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.rors, args.split)[: args.limit or None]

    rows = []
    for method_id in args.methods:
        cls = METHODS[method_id]
        for t in tasks:
            payload = agent_input_payload(t)
            task_in = {
                "task_id": payload["task_id"],
                "goal": payload["question"],
                "dataset_schema": {"descriptions": payload["dataset_descriptions"],
                                   "paths": payload["dataset_paths"]},
                "domain_knowledge": payload["domain_knowledge"],
            }
            client = LLMClient(GATEWAY_URL, GATEWAY_KEY, args.model, timeout=300.0)
            budget = Budget(**E2_BUDGET)
            method = cls(client, budget)
            t0 = time.time()
            try:
                res = method.run(task_in)
                pred = res.final_answer
                status = "OK"
            except Exception as e:
                pred, status = "", f"ERR:{type(e).__name__}:{str(e)[:150]}"
            row = {"method_id": method_id, "task_id": t.task_id,
                   "qid": payload.get("query_group_idx", 0), "status": status,
                   "seconds": round(time.time() - t0, 1), "pred": pred,
                   "budget_remaining": budget.remaining()}
            if status == "OK" and not args.skip_eval:
                try:
                    ev = run_evaluator(t, pred_hypo=pred, pred_workflow="",
                                       query_group_idx=0)
                    row["final_score"] = (ev.get("eval") or {}).get("final_score")
                except Exception as e:
                    row["final_score"] = None
                    row["eval_error"] = f"{type(e).__name__}:{str(e)[:120]}"
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "pred"},
                             ensure_ascii=False), flush=True)
            (out_dir / "e2_results.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    summary = {}
    for m in args.methods:
        rs = [r for r in rows if r["method_id"] == m]
        ok = [r for r in rs if r["status"] == "OK"]
        sc = [r["final_score"] for r in ok if r.get("final_score") is not None]
        summary[m] = {"n": len(rs), "gen_ok": len(ok), "scored": len(sc),
                      "mean_final_score": round(sum(sc) / len(sc), 4) if sc else None,
                      "median_seconds": sorted(r["seconds"] for r in ok)[len(ok)//2] if ok else None}
    (out_dir / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
