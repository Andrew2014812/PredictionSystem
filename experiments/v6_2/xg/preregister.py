"""Freezes the promising xG variant(s) and the forward-test criterion in results/preregistration.json.

The file must be committed BEFORE the forward matches are played; the forward study (experiments/v6_2/forward)
evaluates it on matches with a kick-off after ``frozen_at``. Every number below is read from the result files.
"""
from __future__ import annotations

import time

from . import common as C
from . import models as M
from . import xg_features as F


def run() -> dict:
    dec = C.load_json("decisions.json")
    met = C.load_json("metrics.json")
    main = C.load_json("main.json")
    info = C.load_json("variants_info.json")
    b_variants = [v for v, d in dec.items() if d["decision"] == "B"]
    x5 = met["vs_B0"]["X5"]
    t1 = met["vs_B0"]["T1"]
    sd_x5 = x5["goals_dev"]["sd"]
    doc = {
        "study": "experiments/v6_2/xg",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_until": met["eval_period"][1],
        "git_sha_of_code": C.git_sha(),
        "b_variants": b_variants,
        "primary": {
            "name": "X5",
            "description": "one-parameter geometric blend of the v6.1 goal rates with the opponent-adjusted xG rating",
            "goals": "log mu_side = (1 - w) * log lambda_B0_side + w * log E[xG_side]",
            "w": "refitted every 7 days (7-day weeks starting on Tuesday 2026-09-01) by Poisson maximum likelihood on all "
                 "2026/27 matches of the 18 xG leagues played before the week, w in [0, 1], one w for home and away",
            "w_history": info["in_season"]["coefs"]["X5_w"],
            "rating_model": {"form": "E[xG_home] = exp(mu + hfa + att_home - dfn_away), E[xG_away] = exp(mu + att_away - dfn_home)",
                             "fit": "per league, refitted for every match day on the matches of that league with xG played on an "
                                    "earlier day (lag 0), Poisson quasi-likelihood", "ridge": F.RIDGE,
                             "min_league_matches": F.MIN_LEAGUE_MATCHES},
            "one_x_two": "p_k = p_B0,k * P_k(mu) / P_k(lambda_B0), renormalised (P_k = independent-Poisson outcome probability)",
            "markets": "production score matrix: reconcile_with_1x2(score_matrix(mu_home, mu_away), p); corners unchanged",
            "baseline": f"B0 = production v6.1 model set {C.PRODUCTION_VERSION} (unchanged)",
            "fallback": "B0 where the league has no xG or fewer than min_league_matches earlier matches with xG",
            "in_sample_2026_27": {"matches": met["n_matches"], "goals_dev_diff": x5["goals_dev"]["mean"],
                                  "goals_dev_ci95": x5["goals_dev"]["ci95"], "logloss_1x2_diff": x5["logloss_1x2"]["mean"],
                                  "logloss_1x2_ci95": x5["logloss_1x2"]["ci95"],
                                  "roi_main_diff": main["vs_B0"]["X5"]["roi"]["diff"],
                                  "roi_main_ci95": main["vs_B0"]["X5"]["roi"]["ci95"],
                                  "main_gap_diff": main["vs_B0"]["X5"]["gap"]["diff"],
                                  "sd_per_match_goals_dev_diff": sd_x5,
                                  "n_required_80pct_power": x5["goals_dev"]["n_required"]},
        },
        "secondary": [{
            "name": "T1",
            "description": "v6.1 XGBoost settings refitted on 2021/22-2025/26 with shot-based xG-proxy features "
                           "(proxy = a + b * shots + c * shots on target, fitted on August 2026 xG matches)",
            "proxy_mapping": info["proxy_mapping"], "features": M.PROXY_COLS,
            "in_sample_2026_27": {"logloss_1x2_diff": t1["logloss_1x2"]["mean"], "logloss_1x2_ci95": t1["logloss_1x2"]["ci95"],
                                  "goals_dev_diff": t1["goals_dev"]["mean"]},
        }],
        "forward_test": {
            "matches": "played 2026/27 (and later) matches of the 18 xG leagues with kick-off after frozen_at, every "
                       "prediction made with data available before kick-off; same matches for X5 and B0",
            "minimum_matches": 3000,
            "planned_power_note": "with the in-sample effect and spread, 3000 matches give roughly 60 % one-sided power; "
                                  "the in-sample estimate of the matches needed for 80 % two-sided power is n_required_80pct_power",
            "primary_metric": "mean per-match Poisson deviance of home + away goals, X5 - B0",
            "adopt_if": ["primary difference < 0 with one-sided bootstrap p < 0.05 (2000 resamples by match)",
                         "1X2 log loss difference: upper 95 % bound < +0.002",
                         "Main ROI at real pre-match prices, X5 - B0: lower 95 % bound > -0.02",
                         "the over-confidence of the selected Main (stated probability - hit rate) not larger than B0's"],
            "otherwise": "stays B if the point estimate is < 0, else C",
            "secondary_rule": "T1 is reported with the same metrics (1X2 log loss as its primary) for information; "
                              "Holm correction over the two forward tests",
        },
    }
    C.save_json("preregistration.json", doc)
    return doc


if __name__ == "__main__":
    C.setup_logging("prereg")
    run()
