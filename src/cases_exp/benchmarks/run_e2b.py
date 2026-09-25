"""E2b pilot runner: blind diagnosis on corrupted episodes (spec §15).

Reads materialized episodes (public only), runs each diagnosis method under the
E2b budget (3 calls / 4096 out), scores against private truth (evaluator side).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cases_exp.models.openai_compat import LLMClient, Budget
from cases_exp.methods.e2b_methods import E2B_METHODS

GATEWAY_URL = "http://127.0.0.1:9300/v1"
GATEWAY_KEY = "sk-cases-v9-gateway"
E2B_BUDGET = dict(max_llm_calls=3, max_total_input_tokens=48000, max_total_output_tokens=4096)

LOCUS_MAP = {"SRC": "SOURCE_EDIT", "COMMIT": "COMMITMENT_EDIT", "RET": "GROUNDING_ONLY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="runs/phase0/e2b_episodes")
    ap.add_argument("--out", default="runs/phase0/e2b")
    ap.add_argument("--model", default="qwen36-35b")
    ap.add_argument("--methods", nargs="+", default=list(E2B_METHODS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ep_dir = Path(args.episodes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pubs = sorted((ep_dir / "public").glob("*.json"))
    if args.limit:
        pubs = pubs[:args.limit]

    rows = []
    for method_id in args.methods:
        for pf in pubs:
            rec = json.loads(pf.read_text())
            priv = json.loads((ep_dir / "private" / pf.name).read_text())
            truth = priv["truth"]
            task = {
                "task_id": rec["episode"]["episode_id"],
                "goal": rec["episode"]["question"],
                "corrupted_state": rec["corrupted_state"],
            }
            client = LLMClient(GATEWAY_URL, GATEWAY_KEY, args.model, timeout=300.0)
            budget = Budget(**E2B_BUDGET)
            method = E2B_METHODS[method_id](client, budget)
            t0 = time.time()
            try:
                res = method.run(task)
                diag = res.diagnosis or {}
                pred_loci = [str(x).upper() for x in (diag.get("predicted_loci") or [])]
                canon = {"SRC": {"SRC", "SOURCE", "SOURCE_TERM", "P_SRC"},
                         "COMMIT": {"COMMIT", "COMMITMENT", "COMMITMENT_EDIT", "THETA", "MIXED"},
                         "RET": {"RET", "GROUNDING", "GROUNDING_ONLY", "GAMMA", "RETRACTION"}}
                target = canon.get(truth["locus"], {truth["locus"]})
                locus_hit = any(p in target or any(p in c or c in p for c in target) for p in pred_loci)
                depth_hit = diag.get("recommended_min_depth") == LOCUS_MAP.get(truth["locus"])
                status = "OK"
            except Exception as e:
                locus_hit, depth_hit, status = False, False, f"ERR:{type(e).__name__}:{str(e)[:120]}"
                diag = {}
            row = {
                "method_id": method_id, "episode": rec["episode"]["episode_id"],
                "truth_locus": truth["locus"], "truth_depth": truth["min_repair_depth"],
                "pred_loci": diag.get("predicted_loci"), "pred_depth": diag.get("recommended_min_depth"),
                "locus_hit": bool(locus_hit), "depth_hit": bool(depth_hit),
                "status": status, "seconds": round(time.time() - t0, 1),
                "budget_remaining": budget.remaining(),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    (out_dir / "e2b_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    # summary
    import collections
    summary = {}
    for m in args.methods:
        rs = [r for r in rows if r["method_id"] == m and r["status"] == "OK"]
        n = len(rs)
        summary[m] = {
            "n": n,
            "locus_acc": round(sum(r["locus_hit"] for r in rs) / n, 3) if n else None,
            "depth_acc": round(sum(r["depth_hit"] for r in rs) / n, 3) if n else None,
            "errors": sum(1 for r in rows if r["method_id"] == m and r["status"] != "OK"),
        }
    (out_dir / "e2b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
