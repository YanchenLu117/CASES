#!/usr/bin/env python
"""Fetch the pinned third-party checkouts declared in external/repos.lock.yaml.

Clones each repository at its pinned commit into ``external/repos/<name>``
(the layout expected by ``cases.adapters.hypospace.official``).  Existing
non-empty checkouts are left untouched unless ``--force`` is given.

Usage:
  python scripts/setup_external.py               # fetch every lock entry
  python scripts/setup_external.py --repo hypospace
  python scripts/setup_external.py --force       # re-clone even if present
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "external" / "repos.lock.yaml"
REPOS = ROOT / "external" / "repos"


def fetch(name: str, entry: dict, force: bool) -> bool:
    target = REPOS / name
    if target.is_dir() and any(target.iterdir()):
        if not force:
            print(f"{name}: already present at {target} (use --force to re-clone)")
            return True
        shutil.rmtree(target)
    url = entry.get("url")
    commit = entry.get("commit") or entry.get("fingerprint")
    if not url or not commit:
        print(f"{name}: lock entry has no clonable url/commit", file=sys.stderr)
        return False
    if entry.get("pin_type") not in (None, "git_commit"):
        print(f"{name}: pin_type={entry.get('pin_type')!r} is not clonable; "
              "restore the snapshot archive manually", file=sys.stderr)
        return False
    REPOS.mkdir(parents=True, exist_ok=True)
    print(f"{name}: cloning {url} @ {str(commit)[:12]} ...")
    subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", str(commit)], check=True)
    return True


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", help="fetch a single named repo instead of all")
    ap.add_argument("--force", action="store_true", help="re-clone existing checkouts")
    args = ap.parse_args()

    if not LOCK.exists():
        print(f"lock manifest not found: {LOCK}", file=sys.stderr)
        return 2
    data = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    repos = data.get("repos", {})
    if args.repo:
        if args.repo not in repos:
            print(f"{args.repo}: not pinned in {LOCK}", file=sys.stderr)
            return 2
        names = [args.repo]
    else:
        names = sorted(repos)

    failures = [n for n in names if not fetch(n, repos[n], args.force)]
    verifier = ROOT / "scripts" / "external_lock.py"
    if verifier.exists() and not failures:
        subprocess.run([sys.executable, str(verifier), "--verify"], check=False)
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
