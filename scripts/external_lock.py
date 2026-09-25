#!/usr/bin/env python
"""Generate/verify external/repos.lock.yaml (G1 pin manifest, Detail §2.2).

Repos with their own .git are pinned by HEAD commit; snapshot-only repos
(no .git, restored from archives) are pinned by a content fingerprint
(sha256 over sorted relative paths + sizes).  verify mode re-derives the
fingerprint/HEAD and exits nonzero on any drift.

Usage: python scripts/external_lock.py [--verify]
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "external" / "repos.lock.yaml"


def _content_fingerprint(repo: dict) -> str:
    h = hashlib.sha256()
    for p in sorted(repo.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts:
            h.update(f"{p.relative_to(repo)}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def _independent_git(repo: dict) -> str | None:
    """HEAD only when the checkout has its OWN .git (a nested dir inside the
    CASES repo otherwise resolves to the parent worktree and poisons the pin)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        git_dir = Path(out.stdout.strip()).resolve()
        if repo.resolve() in git_dir.parents or git_dir.parent == repo.resolve():
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            return head.stdout.strip() or None if head.returncode == 0 else None
        return None
    except Exception:
        return None


def _git_head(repo: dict) -> str | None:
    return _independent_git(repo)


def build() -> dict:
    repos_root = ROOT / "external" / "repos"
    entries = {}
    if not repos_root.is_dir():
        print(f"no repos dir at {repos_root}", file=sys.stderr)
        sys.exit(2)
    for repo in sorted(p for p in repos_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        head = _git_head(repo)
        if head:
            entries[repo.name] = {"pin_type": "git_commit", "commit": head}
        else:
            entries[repo.name] = {"pin_type": "content_fingerprint", "fingerprint": _content_fingerprint(repo)}
    return {"schema": "cases_external_lock_v1", "generated_under": "CASES rerun prep", "repos": entries}


def verify(lock: dict) -> list[str]:
    problems = []
    for name, entry in lock.get("repos", {}).items():
        repo = ROOT / "external" / "repos" / name
        if not repo.is_dir():
            problems.append(f"{name}: MISSING checkout")
            continue
        if entry.get("pin_type") == "git_commit":
            head = _git_head(repo)
            if head != entry.get("commit"):
                problems.append(f"{name}: pin mismatch lock={entry.get('commit')} checkout={head}")
        else:
            fp = _content_fingerprint(repo)
            if fp != entry.get("fingerprint"):
                problems.append(f"{name}: content drift lock={entry.get('fingerprint')} now={fp}")
    on_disk = {p.name for p in (ROOT / "external" / "repos").iterdir() if p.is_dir() and not p.name.startswith("_")}
    for extra in sorted(on_disk - set(lock.get("repos", {}))):
        problems.append(f"{extra}: on disk but NOT in lock")
    return problems


def main() -> int:
    import yaml

    if "--verify" in sys.argv:
        if not LOCK.exists():
            print("lock missing; run without --verify to generate", file=sys.stderr)
            return 2
        data = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
        problems = verify(data)
        if problems:
            print("\n".join(problems))
            return 1
        print(f"LOCK_OK ({len(data.get('repos', {}))} repos verified)")
        return 0
    lock = build()
    import yaml

    LOCK.write_text(yaml.safe_dump(lock, sort_keys=True), encoding="utf-8")
    print(f"wrote {LOCK} with {len(lock['repos'])} repos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
