"""this external-system adapter gate (EXPERIMENT_BENCHMARKS_BASELINES_FINAL §5.3).

Every external AI-scientist system (S1-S4) and the MADE/PiEvo native hosts must
pass this gate before any of its numbers count toward evaluation.  It is a
code-level, deterministic checkpoint over the SYSTEM'S CHECKOUT + a smoke report:

  1. official immutable commit pinned + license recorded;
  2. contract & hidden-information tests pass;
  3. >=90% preregistered smoke episodes complete;
  4. official core source untouched (adapter lives OUTSIDE the pinned checkout);
  5. full action/observation/resource trace logged;
  6. fixed-state deterministic tool replay passes.

A system that fails stays NAMED with `adapter gate failed` in the appendix and is
NOT replaced after outcomes are seen (§5.3).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ..sciexplorer.arms import smoke_gate_ok as _se_smoke


class AdapterGateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"  # official repo not present yet — not evaluable


def _repo_commit(repo_dir) -> str | None:
    try:
        import subprocess
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def adapter_gate(
    host: str,
    *,
    repo_dir: str | Path | None = None,
    adapter_src_dir: str | Path | None = None,
    smoke_report: Mapping[str, Any] | None = None,
    license_recorded: bool | None = None,
    hidden_info_clean: bool | None = None,
    trace_logged: bool | None = None,
    deterministic_replay_pass: bool | None = None,
    commit: str | None = None,
    repo_present: bool | None = None,
) -> tuple[AdapterGateStatus, dict[str, Any]]:
    """Return (status, checks).  ``status`` is PENDING when the official checkout
    is not present; otherwise PASS only if every required gate passes."""
    repo_dir = Path(repo_dir) if repo_dir else None
    checks: dict[str, Any] = {"host": host}

    _inline_present = repo_present is not None
    repo_present = repo_present if _inline_present else (repo_dir is not None and (repo_dir / ".git").exists())
    commit = commit if commit is not None else (_repo_commit(repo_dir) if repo_present and not _inline_present else None)
    checks.update({
        "repo_present": bool(repo_present),
        "commit_pinned": commit is not None,
        "commit": commit,
        "license_recorded": bool(license_recorded) if license_recorded is not None else False,
    })

    # official core untouched: adapter code must NOT live inside the pinned checkout
    if adapter_src_dir is not None and repo_dir is not None:
        try:
            a = Path(adapter_src_dir).resolve()
            r = repo_dir.resolve()
            checks["core_untouched"] = not (str(a) == str(r) or str(r) in str(a))
        except Exception:
            checks["core_untouched"] = False
    else:
        checks["core_untouched"] = False

    lk = _smoke_report(smoke_report)
    completion = float(lk.get("completion_rate", 0.0))
    checks.update({
        "smoke_completion_ge_90": completion >= 0.90,
        "smoke_completion": completion,
        "trace_logged": bool(trace_logged) if trace_logged is not None else bool(lk.get("tool_calls_logged", 0) > 0),
        "hidden_info_clean": bool(hidden_info_clean) if hidden_info_clean is not None else False,
        "deterministic_replay_pass": bool(deterministic_replay_pass) if deterministic_replay_pass is not None else False,
    })

    if not repo_present:
        return AdapterGateStatus.PENDING, checks
    required = ("commit_pinned", "license_recorded", "core_untouched",
                "smoke_completion_ge_90", "trace_logged",
                "hidden_info_clean", "deterministic_replay_pass")
    passed = all(checks[k] for k in required)
    checks["status"] = AdapterGateStatus.PASS.value if passed else AdapterGateStatus.FAIL.value
    return (AdapterGateStatus.PASS if passed else AdapterGateStatus.FAIL), checks


def _smoke_report(r: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(r or {})


def sciexplorer_adapter_gate(**kw) -> tuple[AdapterGateStatus, dict[str, Any]]:
    """SciExplorer gate: the §5.3 gate + the §3.1 smoke-gate self-check."""
    status, checks = adapter_gate("sciexplorer", **{k: v for k, v in kw.items() if k != "smoke_report"})
    se_ok = None
    if kw.get("smoke_report") is not None:
        se_ok, se_checks = _se_smoke(kw["smoke_report"])
        checks["sciexplorer_smoke_gate"] = se_ok
        checks["sciexplorer_smoke_checks"] = se_checks
    return status, checks
