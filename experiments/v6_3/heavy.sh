#!/usr/bin/env bash
# Run a heavy research job (model fitting) in one of two shared slots, so that parallel studies never fit more than
# two models at once on this shared machine. Waits for a free slot up to HEAVY_WAIT seconds (default 3 h).
#
#   experiments/v6_3/heavy.sh <timeout-seconds> <logfile> <command...>
#
# The command runs under `timeout`, its output goes to <logfile>; the exit code is the command's.
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
limit="$1"; logfile="$2"; shift 2
dir="$(cd "$(dirname "$0")" && pwd)"
wait_max="${HEAVY_WAIT:-10800}"
start=$(date +%s)
mkdir -p "$(dirname "$logfile")"
while :; do
  for slot in 1 2; do
    exec {fd}>"$dir/.heavy$slot.lock"
    if flock -n "$fd"; then
      echo "[heavy] slot $slot taken at $(date -Is): $*" >> "$logfile"
      timeout "$limit" "$@" >> "$logfile" 2>&1
      code=$?
      echo "[heavy] exit $code at $(date -Is)" >> "$logfile"
      flock -u "$fd"
      exit "$code"
    fi
    exec {fd}>&-
  done
  if [ $(( $(date +%s) - start )) -gt "$wait_max" ]; then
    echo "[heavy] no free slot after ${wait_max}s" >> "$logfile"
    exit 75
  fi
  sleep 20
done
