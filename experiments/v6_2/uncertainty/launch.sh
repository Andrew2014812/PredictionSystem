#!/usr/bin/env bash
# Background launcher of the uncertainty study (deadline + log per step; never waits blindly).
#   bash experiments/v6_2/uncertainty/launch.sh ensemble   # three ensemble families in parallel (n_jobs = 2 each)
#   bash experiments/v6_2/uncertainty/launch.sh run        # analysis (after the ensembles are done)
set -u
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python}
LOGS=experiments/v6_2/uncertainty/cache/logs
mkdir -p "$LOGS"
case "${1:-}" in
  ensemble)
    for fam in seed boot sub; do
      nohup timeout 4h "$PY" -m experiments.v6_2.uncertainty.ensemble "$fam" > "$LOGS/ens_$fam.log" 2>&1 &
    done
    ;;
  run)
    ( timeout 30m "$PY" -m experiments.v6_2.uncertainty.prepare 2026 && timeout 90m "$PY" -m experiments.v6_2.uncertainty.run ) > "$LOGS/run.log" 2>&1 &
    ;;
  *)
    echo "usage: launch.sh ensemble|run"; exit 2 ;;
esac
