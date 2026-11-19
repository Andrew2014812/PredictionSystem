"""Freeze the B candidate of the market-profitability study (results/preregistration.json).

    python -m experiments.v6_3.market_profitability.preregister
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

RES = C.study_dirs("market_profitability")["results"]

B_CANDIDATES = {
    "bt:boost": {
        "hypothesis": "Asian handicaps are under-weighted in the Main ranking (their flat candidate ROI is the best "
                      "of all market groups in 2025/26 and 2026/27): raising the HANDICAP reliability by 15 % "
                      "(0.90 -> 1.035) moves Main from totals to handicaps and raises Main ROI.",
        "rule": "production Main rule with config.MARKET_RELIABILITY['HANDICAP'] = 1.035 (= 0.90 x 1.15); "
                "chosen on the 2025/26 backtest and by the expanding window in every month",
        "production_change": "src/config.py MARKET_RELIABILITY['HANDICAP']: 0.90 -> 1.035 (same direction as "
                             "study 1 B candidate TOTAL 1.00 -> 0.95; test one of them, not both at once)"},
}


def main() -> None:
    res = C.read_json(RES / "market_profitability.json")
    rows = {(r["candidate"], r["universe"]): r for r in res["candidates"]}
    frozen = pd.Timestamp.now(tz="Europe/London").isoformat(timespec="seconds")
    out = []
    for name, spec in B_CANDIDATES.items():
        cl, t = rows[(name, "close")], rows[(name, "t09")]
        half = (cl["diff_ci"][1] - cl["diff_ci"][0]) / 2
        n_req = int(round(cl["bets"] * (half / abs(cl["roi_diff"])) ** 2)) if cl["roi_diff"] else None
        out.append({"id": f"v63_market_profitability:{name}", "decision": "B", **spec,
                    "primary_metric": "paired ROI difference vs the production Main on the same matches "
                                      "(unit stake, real price at decision time; close and T reported)",
                    "evidence_2026_27": {"close": {k: cl[k] for k in ("bets", "roi", "roi_diff", "diff_ci", "p",
                                                                      "p_holm")},
                                         "t09": {k: t[k] for k in ("bets", "roi", "roi_diff", "diff_ci", "p",
                                                                   "p_holm", "common_diff")}},
                    "minimum_forward_matches": n_req,
                    "success": "paired ROI difference > 0 with 95 % interval above 0 on matches settled after "
                               "frozen_at, at close AND at T",
                    "frozen_at": frozen})
    C.write_json(RES / "preregistration.json", out)
    print(f"{len(out)} candidate(s) frozen at {frozen}")


if __name__ == "__main__":
    main()
