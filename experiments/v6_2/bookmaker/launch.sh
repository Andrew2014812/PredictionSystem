#!/usr/bin/env bash
# Runs one step of the bookmaker study in the background with a deadline, a log and a done-marker.
#   bash experiments/v6_2/bookmaker/launch.sh hybrids --reuse-encompassing
# Log: experiments/v6_2/bookmaker/cache/<step>.log, marker: cache/<step>.done ("EXIT <code>").
set -u
cd "$(dirname "$0")/../../.."
step="$1"; shift
PY=/home/mlitvinenko/repositories/PredictionSystem/.venv/bin/python
mkdir -p experiments/v6_2/bookmaker/cache
rm -f "experiments/v6_2/bookmaker/cache/$step.done"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
timeout 3600 "$PY" -u -m "experiments.v6_2.bookmaker.$step" "$@" > "experiments/v6_2/bookmaker/cache/$step.log" 2>&1
echo "EXIT $?" > "experiments/v6_2/bookmaker/cache/$step.done"
