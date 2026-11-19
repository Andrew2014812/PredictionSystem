#!/usr/bin/env bash
# Runs the three market studies of v6.3 one after another (light jobs, ~5 minutes in total) and builds the PDFs.
set -e
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
for s in market_aware timing consensus; do
  timeout 1800 $PY -m experiments.v6_3.$s.run
done
for s in market_aware timing consensus; do
  timeout 900 $PY -m experiments.v6_3.$s.build_pdf
done
