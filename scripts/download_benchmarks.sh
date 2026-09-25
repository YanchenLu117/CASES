#!/usr/bin/env bash
# Download official benchmark code/data (BH, GB1).
#
# 说明: Download official benchmark assets from their upstream sources.
# Uses aria2c multi-connection download of the official tarballs; reproducible entry point.
#
# 用法: 在能连 codeload.github.com 的机器上运行, 产物落到 ./external/repos/
# Depends on: aria2c (macOS: brew install aria2), gzip
set -euo pipefail
OUT=external/repos
mkdir -p "$OUT"

dl() { # dl <url> <out.tar.gz.tar>
  local url="$1" out="$2"
  echo "== $out =="
  aria2c -x 16 -s 16 --max-tries=40 --retry-wait=3 --connect-timeout=20 --timeout=90 \
    -d /tmp/cases_dl -o "$out.tgz" "$url" >/dev/null 2>&1 || true
  # validate + extract
  if gzip -t /tmp/cases_dl/"$out".tgz 2>/dev/null; then
    mkdir -p "$OUT/$out" && tar xzf /tmp/cases_dl/"$out".tgz -C "$OUT/$out" --strip-components=1
    echo "   OK"
  else
    echo "   INCOMPLETE after max-tries — rerun this script; validate gzip -t"
    return 1
  fi
}

mkdir -p /tmp/cases_dl
# BH: doylelab/rxnpredict (数据 + descriptors + data_table.csv), MIT
dl "https://codeload.github.com/doylelab/rxnpredict/tar.gz/refs/heads/master" bh_rxnpredict
# GB1: ALDE (official sims/encodings/models) , MIT
dl "https://codeload.github.com/jsunn-y/ALDE/tar.gz/refs/heads/main" gb1_alde
# GB1: CLADE (clustering+MLDE), MIT
dl "https://codeload.github.com/WeilabMSU/CLADE/tar.gz/refs/heads/main" gb1_clade
echo "== GB1 measured oracle: eLife supp File 1 (149,361 variants) 见报告 §10.2, 需另取 (eLife 403 需 UA) =="
