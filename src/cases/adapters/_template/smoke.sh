#!/usr/bin/env bash
# Smoke template (>=90% preregistered completion required by §5.7 gate).
# Copy into src/cases/adapters/<system>/ and replace RUN_CMD with the official
# system's own entry point routed through the campaign-API boundary adapter.
set -u
cd "$(dirname "$0")/../../.."

SYSTEM="${1:-TEMPLATE}"
TASKS="${2:-1}"          # number of preregistered smoke tasks
DONE=0

# TODO(system): RUN_CMD must invoke the OFFICIAL entry point only, with
# observations/candidates supplied via the boundary adapter — never GT.
RUN_CMD="echo 'TODO: official entry point'"

for t in $(seq 1 "$TASKS"); do
  if eval "$RUN_CMD" > "runs/smoke_${SYSTEM}_task${t}.log" 2>&1; then
    DONE=$((DONE + 1))
  fi
done

RATE=$(python3 -c "print(round($DONE/max($TASKS,1),3))")
printf '{"system":"%s","completion_rate":%s,"done":%d,"tasks":%d}\n' \
  "$SYSTEM" "$RATE" "$DONE" "$TASKS" > "baselines/${SYSTEM}/smoke_result.json"
echo "smoke completion_rate=$RATE (need >=0.90)"
[ "$(python3 -c "print(1 if $RATE>=0.9 else 0)")" = "1" ]
