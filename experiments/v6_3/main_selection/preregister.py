"""Freeze the B candidates of the Main-selection study for a forward check (results/preregistration.json).

    python -m experiments.v6_3.main_selection.preregister

The minimum forward sample is the number of matches at which the 95 % interval of the paired ROI difference,
at the width observed on 2026/27 (close), would exclude zero if the true effect equalled the observed one:
n = n_2026/27 · (half-width / |ΔROI|)². Matches settled after ``frozen_at`` only.
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

from . import data as D

RES = D.DIRS["results"]

B_CANDIDATES = {
    "bt:reliability": {
        "hypothesis": "Totals are over-weighted in the Main ranking: lowering the reliability of TOTAL from 1.00 "
                      "to 0.95 moves part of the Main to Asian handicaps / 1X2 and raises Main ROI.",
        "rule": "production Main rule with config.MARKET_RELIABILITY['TOTAL'] = 0.95 (all other parameters as "
                "prod 2026-11-18); chosen on the 2025/26 backtest",
        "production_change": "src/config.py MARKET_RELIABILITY['TOTAL']: 1.00 -> 0.95"},
    "bt:ev_weight": {
        "hypothesis": "A stronger EV term (MAIN_EV_WEIGHT 0.5 -> 1.0) raises Main ROI slightly; the only family "
                      "whose 2025/26 choice is positive on 2025/26, close and T.",
        "rule": "production Main rule with config.MAIN_EV_WEIGHT = 1.0",
        "production_change": "src/config.py MAIN_EV_WEIGHT: 0.5 -> 1.0"},
    "floor_1.40": {
        "hypothesis": "A hard minimum real price of 1.40 for Main (statistical Main kept but marked 'not a bet' "
                      "when no selection is priced >= 1.40) raises Main ROI.",
        "rule": "eligible = production band and group rules AND real price >= 1.40; rank by the production "
                "score; no eligible selection -> production's statistical Main, shown, settled, NOT counted in ROI",
        "production_change": "src/markets/recommendation.py main_prediction: filter candidates by "
                             "has_odds and odds >= MAIN_MIN_ODDS (new config 1.40); fallback Recommendation "
                             "kind 'model' (not a bet) with a reason"},
}


def main() -> None:
    res = C.read_json(RES / "main_selection.json")
    rows = {(r["candidate"], r["universe"]): r for r in res["candidates"]}
    frozen = pd.Timestamp.now(tz="Europe/London").isoformat(timespec="seconds")
    out = []
    for name, spec in B_CANDIDATES.items():
        cl, t = rows[(name, "close")], rows[(name, "t09")]
        half = (cl["diff_ci"][1] - cl["diff_ci"][0]) / 2
        n_obs = cl["bets"]
        n_req = int(round(n_obs * (half / abs(cl["roi_diff"])) ** 2)) if cl["roi_diff"] else None
        out.append({"id": f"v63_main_selection:{name}", "decision": "B", **spec,
                    "primary_metric": "paired ROI difference vs the production Main on the same matches "
                                      "(unit stake, real price at decision time; close and T reported)",
                    "evidence_2026_27": {"close": {k: cl[k] for k in ("bets", "roi", "roi_diff", "diff_ci", "p",
                                                                      "p_holm")},
                                         "t09": {k: t[k] for k in ("bets", "roi", "roi_diff", "diff_ci", "p",
                                                                   "p_holm", "common_diff")}},
                    "minimum_forward_matches": n_req,
                    "success": "paired ROI difference > 0 with 95 % interval above 0 on matches settled after "
                               "frozen_at, at close AND at T; no month with a difference below -5 p.p.",
                    "frozen_at": frozen})
    C.write_json(RES / "preregistration.json", out)
    print(f"{len(out)} candidates frozen at {frozen}")


if __name__ == "__main__":
    main()
