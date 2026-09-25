#!/usr/bin/env bash
# Download the official benchmark assets required by the BH and GB1 lanes.
# Everything lands in ./external/repos/ (git-ignored).  Multi-connection via
# aria2c when available, plain curl otherwise.  Re-running is safe: existing
# checkouts are skipped.
#
# Usage:  bash scripts/download_benchmarks.sh
set -euo pipefail
OUT=external/repos
mkdir -p "$OUT" /tmp/cases_dl

dl() { # dl <url> <out-dir>
  local url="$1" out="$2"
  echo "== $out =="
  if [ -d "$OUT/$out" ] && [ -n "$(ls -A "$OUT/$out" 2>/dev/null)" ]; then
    echo "   already present — skipping"
    return 0
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 --max-tries=40 --retry-wait=3 --connect-timeout=20 --timeout=90 \
      -d /tmp/cases_dl -o "$out.tgz" "$url" >/dev/null 2>&1 || true
  else
    curl -sL --retry 5 --connect-timeout 20 -o /tmp/cases_dl/"$out".tgz "$url" || true
  fi
  if gzip -t /tmp/cases_dl/"$out".tgz 2>/dev/null; then
    mkdir -p "$OUT/$out" && tar xzf /tmp/cases_dl/"$out".tgz -C "$OUT/$out" --strip-components=1
    echo "   OK"
  else
    echo "   INCOMPLETE — rerun this script (check network access to codeload.github.com)"
    return 1
  fi
}

# BH: official reaction-yield table + descriptors (doylelab/rxnpredict), MIT
dl "https://codeload.github.com/doylelab/rxnpredict/tar.gz/refs/heads/master" bh_rxnpredict
# GB1: ALDE official sims/encodings/models, MIT
dl "https://codeload.github.com/jsunn-y/ALDE/tar.gz/refs/heads/master" gb1_alde
# GB1: CLADE clustering+MLDE, MIT — its Input/GB1.xlsx is the full 149,361-variant
# measured table consumed by cases.adapters.gb1 (env override: CASES_GB1_XLSX)
dl "https://codeload.github.com/WeilabMSU/CLADE/tar.gz/refs/heads/main" gb1_clade
