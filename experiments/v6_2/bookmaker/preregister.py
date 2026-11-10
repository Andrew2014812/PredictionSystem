"""Freezes the one bookmaker-benchmark hypothesis worth a forward check (python -m experiments.v6_2.bookmaker.preregister).

Selection seasons (2023/24, 2024/25) show no gain of any hybrid over the de-margined Bet365 opening price, and the
market encompasses the model there. In 2026/27 (information only, used by v6.1) the model's weight next to the
Bet365 opening price is positive for 1X2 and total 2.5, and the log-linear pool H2 is better than the market for
total 2.5. That is a low-prior hypothesis: it is frozen here (parameters fitted on 2022/23–2025/26) and judged only
on matches played after this file is committed (see experiments/v6_2/forward/). Production is not changed.
"""
from __future__ import annotations

import time

from . import common as C


def main() -> None:
    hy = C.load_json("hybrids.json")
    enc = C.load_json("encompassing.json")
    res = hy["results"]
    doc = {
        "study": "experiments/v6_2/bookmaker",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "hypothesis B (low prior): not adopted; forward check only",
        "evidence_at_freeze": {
            "selection_seasons": {k: {"H2_minus_H0_logloss": res[k]["H2"]["d_ll_vs_H0"],
                                      "model_weight_b2": enc["tests"].get(f"{k.split('|')[0]}|select|b365_open",
                                                                          {}).get("b_model")}
                                  for k in ("1X2|2023", "1X2|2024", "OU25|2023", "OU25|2024")},
            "info_2025_26": {k: res[k]["H2"]["d_ll_vs_H0"] for k in ("1X2|2025", "OU25|2025")},
            "info_2026_27_not_holdout": {k: res[k]["H2"]["d_ll_vs_H0"] for k in ("1X2|2026", "OU25|2026")},
            "encompassing_2026_27": {k: enc["tests"][k] for k in ("1X2|2026|b365_open", "OU25|2026|b365_open")},
        },
        "frozen_variant": {
            "name": "H2 log-linear pool of the model and the Bet365 opening price",
            "formula": "P(k) = softmax_k(a_k + b1 * log q_k + b2 * log p_k), a_1 = 0",
            "q": "Bet365 opening price (Football-Data B365H/D/A, B365>2.5/<2.5) de-margined by the power method",
            "p": "the production model probability stored for the match (1X2 classifier; total 2.5 from the score "
                 "distribution), latest version before kick-off",
            "theta_1x2": res["1X2|2026"]["params"]["H2_theta"],
            "theta_ou25": res["OU25|2026"]["params"]["H2_theta"],
            "theta_order": "[a_2..a_k, b1 (market), b2 (model)]",
            "fitted_on": "walk-forward model predictions 2022/23–2025/26 (experiments/v6_1/main/cache)",
        },
        "forward_sample": {
            "matches": "league matches of the 21 production leagues with kick-off AFTER the commit of this file, "
                       "with a stored pre-kick-off model prediction and both Football-Data Bet365 opening prices",
            "minimum": "1500 matches per market before any conclusion; one evaluation, at the first data update "
                       "after the minimum is reached",
        },
        "primary_criterion": {
            "metric": "mean log loss of H2 minus mean log loss of the de-margined Bet365 opening price (H0), "
                      "per market (1X2, total 2.5), paired by match",
            "pass": "upper bound of the 95 % paired bootstrap interval (2000 resamples by match) < 0 for total 2.5; "
                    "1X2 reported as secondary with Holm correction over the two markets",
        },
        "secondary": ["encompassing weight b2 of the model with its 95 % bootstrap interval (> 0 expected)",
                      "value bets EV = p_H2 x Bet365 opening - 1 > 0: ROI and CLV against the Betfair-exchange "
                      "close — reported, never a pass criterion"],
        "decision_rule": "pass -> B: propose H2 probabilities as the EV input for Main / Risk and run a separate "
                         "forward test of the betting rule; fail -> C (drop)",
    }
    C.save_json("preregistration.json", doc)
    print("wrote results/preregistration.json")


if __name__ == "__main__":
    main()
