#!/usr/bin/env bash
# Background launcher of the forward-validation study (deadline + log; never waits blindly).
#   bash experiments/v6_2/forward/launch.sh simulate   # replay of the shadow loop in a temporary folder
#   bash experiments/v6_2/forward/launch.sh run        # current-state run + registry/cost summary
set -u
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python}
LOGS=experiments/v6_2/forward/cache
mkdir -p "$LOGS"
case "${1:-}" in
  simulate)
    nohup timeout 60m "$PY" -m experiments.v6_2.forward.simulate > "$LOGS/simulate.log" 2>&1 &
    ;;
  run)
    nohup timeout 30m "$PY" -m experiments.v6_2.forward.run > "$LOGS/run.log" 2>&1 &
    ;;
  *)
    echo "usage: launch.sh simulate|run"; exit 2 ;;
esac
