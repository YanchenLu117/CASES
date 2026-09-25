#!/usr/bin/env bash
# No-LLM end-to-end smoke test: construct → audit → scoped exploration →
# recover → readout on the toy Boolean domain (official HypoSpace evaluator).
# No data downloads, no API keys.  Exit 0 = healthy.
#
# Usage: ./smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

if [ ! -d external/repos/hypospace ]; then
  echo "[smoke] official HypoSpace checkout missing — fetching pinned commit..."
  "$PY" scripts/setup_external.py --repo hypospace
fi

exec "$PY" - <<'EOF'
from cases.adapters.hypospace.boolean import BooleanAdapter
from cases.adapters.hypospace.tasks import BooleanTask
from cases.core.model import CASESModel, EvidenceRecord
from cases.core.types import LanguageHypothesis

task = BooleanTask(
    dataset="boolean_smoke",
    observation_set_id="smoke",
    variables=("x", "y"),
    operators=frozenset({"AND", "OR", "NOT"}),
    max_depth=2,
    mechanistic_opts={
        "apply_commutativity": True,
        "apply_idempotence_and_or": True,
        "flatten_associativity": True,
    },
    observations=(),
    n_observations=0,
)
m = CASESModel(adapter=BooleanAdapter(task))
batch = [
    EvidenceRecord(
        kind="hypothesis",
        hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
    )
    for i, t in enumerate(["x AND y", "x OR y", "NOT x", "x AND NOT y"])
]
snap = m.update(batch)
st = m.recover()
summary = m.readout(st, token_budget=1800)
print("objects:", snap["n_objects"], "| recovery backend:", st.metadata.get("recovery_backend"))
print("readout bytes:", len(summary))
print("SMOKE_OK")
EOF
