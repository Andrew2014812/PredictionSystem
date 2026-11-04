"""E6: cost of each explanation method per match, measured on the final implementation.

    .venv/bin/python experiments/v6_1/explain/timing.py

Cold start = building the TreeExplainers of the four models (once per server process); warm = one
match page (Main + Risk + one other market) once the explainers exist. n_jobs = 3.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Data, chain, log, old_explain, panel, save_json, setup_logging  # noqa: E402

from src.explain import shap_explainer  # noqa: E402

N = 30


def main() -> None:
    setup_logging("timing")
    data = Data()
    pool = data.pool()
    sm = pool.loc[pool["model_version"] == "prod_20261012"].sample(N, random_state=7)
    bundles = data.bundles["prod_20261012"]
    cold = {}
    for name, b in bundles.items():
        t = time.time()
        shap_explainer._explainer(b)
        cold[name] = time.time() - t
    page_new, page_old, attr, old_sel = [], [], [], []
    for snap in sm.itertuples(index=False):
        row = data.row(snap.match_id)
        sels = panel(snap._asdict(), data.match_markets(snap.match_id))[:3]       # Main, Risk, one other
        t = time.time()
        for m, s, _ in sels:
            chain.explain_market(row, m, s, bundles, key=f"timing|{snap.match_id}")
        page_new.append(time.time() - t)
        t = time.time()
        for m, s, _ in sels:
            t1 = time.time()
            old_explain(row, m, s, bundles)
            old_sel.append(time.time() - t1)
        page_old.append(time.time() - t)
        inputs = chain.chain_inputs(row, bundles, chain.GOAL_OUTPUTS)
        for m, s, _ in sels:
            if not m.startswith("CORNERS"):
                t = time.time()
                chain.attribute(inputs, m, s)
                attr.append(time.time() - t)
    ms = lambda xs: {"median": float(np.median(xs) * 1000), "p90": float(np.quantile(xs, 0.9) * 1000)}  # noqa: E731
    out = {"matches": N, "cold_start_explainer_s": cold, "cold_start_total_s": float(sum(cold.values())),
           "page_three_cards_new_ms": ms(page_new), "page_three_cards_old_ms": ms(page_old),
           "attribution_only_ms": ms(attr), "old_per_selection_ms": ms(old_sel)}
    save_json("e6_timing.json", out)
    log.info("E6 %s", out)


if __name__ == "__main__":
    main()
