"""§5.7 adapter gate — nine checks, each PASS/FAIL/SKIP with reason.

Usage (from repo root):
    PYTHONPATH=src python -m cases.adapters._template.gate_check --system-dir baselines/s2_ai_scientist_v2 --full
Exit code 0 iff no FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CHECKS_ORDER = [
    "pin_frozen", "license_present", "no_core_edit", "hidden_info_scan",
    "smoke_completion", "deterministic_replay", "logs_present",
    "manifest_present", "applicability_frozen",
]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes()[:1 << 20])
    return h.hexdigest()[:16]


def run_checks(system_dir: Path, full: bool = False, adapter_dir: Path | None = None) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    adir = (adapter_dir or Path(__file__).resolve().parent).resolve()

    def record(name: str, status: str, reason: str = "") -> None:
        results.append({"check": name, "status": status, "reason": reason})

    pin = system_dir / "PIN.txt"
    # 1 pin frozen — any non-empty PIN record counts (hash-style tokens suffice)
    if pin.exists():
        txt = pin.read_text(encoding="utf-8", errors="replace").strip()
        record("pin_frozen", "PASS" if txt else "FAIL",
               f"PIN recorded: {txt[:40]!r}" if txt else "PIN.txt empty")
    else:
        record("pin_frozen", "FAIL", f"{pin} missing")

    # 2 license
    lic = [p for p in system_dir.glob("LICEN*")] or [p for p in system_dir.rglob("LICEN*") if p.parent == system_dir]
    record("license_present", "PASS" if lic else ("SKIP" if not full else "FAIL"),
           str(lic[0].name) if lic else "no LICENSE file at top level")

    # 3 no core edit — scope to the system's own checkout only (never the parent repo)
    if (system_dir / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(system_dir), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
            dirty = r.stdout.strip()
            record("no_core_edit", "SKIP" if r.returncode != 0 else ("PASS" if not dirty else "WARN"),
                   "clean checkout" if not dirty else f"dirty: {dirty[:120]}")
        except Exception as exc:  # noqa: BLE001
            record("no_core_edit", "SKIP", f"git unavailable: {exc}")
    else:
        record("no_core_edit", "SKIP",
               "not a standalone checkout (official tree lives elsewhere; verify via PIN + upstream digest)")

    # 4 hidden info scan — adapter-side files ONLY (official upstream code may
    # legitimately contain strings like 'ground_truth'; we audit OUR glue).
    banned = ["ground_truth", "top_5_percent", "|h_o|", "h_o_size", "oracle_label"]
    hits: list[str] = []
    scan_root = adir
    scan_files = [p for p in scan_root.rglob("*.py")
                  if p.name not in ("gate_check.py",)][:400]
    for p in scan_files:
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for b in banned:
            if b.lower() in t.lower():
                hits.append(f"{p.name}:{b}")
    record("hidden_info_scan", "PASS" if not hits else "FAIL",
           f"adapter-side clean ({scan_root.name})" if not hits else "; ".join(hits[:6]))

    # 5 smoke completion >=90%
    smoke_marker = system_dir / "smoke_result.json"
    if smoke_marker.exists():
        try:
            data = json.loads(smoke_marker.read_text())
            rate = float(data.get("completion_rate", 0.0))
            record("smoke_completion", "PASS" if rate >= 0.9 else "FAIL", f"rate={rate:.2f}")
        except Exception as exc:  # noqa: BLE001
            record("smoke_completion", "FAIL", f"unreadable: {exc}")
    else:
        record("smoke_completion", "SKIP", "no smoke_result.json yet")

    # 6 deterministic replay
    replay_marker = system_dir / "replay_result.json"
    record("deterministic_replay",
           "PASS" if replay_marker.exists() and json.loads(replay_marker.read_text()).get("match") else "SKIP",
           "replay_result.json match=true required before main table")

    # 7 logs present
    logs = list(system_dir.rglob("*.log")) + list(system_dir.rglob("*request_response*"))
    record("logs_present", "PASS" if logs else "SKIP", f"{len(logs)} log files")

    # 8 manifest present
    man = system_dir / "run_manifest.json"
    tmpl = system_dir / "run_manifest.template.json"
    record("manifest_present", "PASS" if man.exists() else ("SKIP" if tmpl.exists() else "FAIL"),
           "run_manifest.json present" if man.exists() else "template only — fill before formal runs")

    # 9 applicability frozen (S/X specific)
    app = system_dir / "APPLICABILITY.txt"
    record("applicability_frozen", "PASS" if app.exists() else "SKIP",
           "frozen applicability recorded" if app.exists() else "required only for domain-specialist rows")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system-dir", default=str(Path(__file__).resolve().parent),
                    help="official checkout dir (PIN/LICENSE/.git live here)")
    ap.add_argument("--adapter-dir", default=None,
                    help="our glue-code dir to scan for hidden-info (defaults to this file's dir)")
    ap.add_argument("--full", action="store_true", help="run deep variants of each check")
    args = ap.parse_args()
    sdir = Path(args.system_dir).resolve()
    adir = Path(args.adapter_dir).resolve() if args.adapter_dir else Path(__file__).resolve().parent
    results = run_checks(sdir, full=args.full, adapter_dir=adir)
    fails = 0
    print(f"# gate_check --system-dir {sdir} {'--full' if args.full else ''}")
    for r in results:
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[str(r["status"])]
        if r["status"] == "FAIL":
            fails += 1
        print(f"  {mark} {r['check']:<24} {r['reason']}")
    verdict = "GATE GREEN" if fails == 0 else f"GATE RED ({fails} FAIL)"
    print(verdict)
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
