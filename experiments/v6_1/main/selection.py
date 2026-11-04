"""Choice of the strategy to pre-register (folds 2023/24 + 2024/25 only) -> results/selection.json.

Primary candidate: the variant (any family except A) with the highest worst-fold ROI over 2023/24 and 2024/25
among variants with >= 300 bets on each fold (bettable universe, identical matches). The best variant of every
other family is pre-registered as secondary (reported, not used for adoption). The adoption rules 1–3 of
preregistration.json are evaluated here on the folds; rule 4 needs 2026/27 (independent.py).
"""
from __future__ import annotations

import numpy as np

from . import common as C

DEFINITIONS = {
    "A": "main_prediction() with the production constants (src/config.py MAIN_*, MARKET_RELIABILITY)",
    "B1 isotonic": "per-key isotonic map (group|side) of the no-refund probability fitted on the inner season; rule A on the calibrated probability",
    "B2 Platt": "per-key logistic map of logit p fitted on the inner season; rule A on it",
    "B3 selection-conditional": "isotonic map fitted on the inner season's Main picks of A only; rule A on it",
    "C1 EV weight 1": "rule A with MAIN_EV_WEIGHT = 1", "C1 EV weight 2": "rule A with MAIN_EV_WEIGHT = 2",
    "C1 EV weight 4": "rule A with MAIN_EV_WEIGHT = 4",
    "C2 value first EV>=0": "rule A, first among priced candidates with EV >= 0",
    "C2 value first EV>=0.03": "rule A, first among priced candidates with EV >= 3 %",
    "C3 max EV": "priced candidate in the band 0.40–0.85 with the highest EV (raw p)",
    "C4 calibrated + EV weight 2": "B1 probability, EV weight 2",
    "C5 calibrated max EV": "highest EV with the B1 probability",
    "D1 no totals": "rule A without total-goals candidates", "D2 only 1X2": "rule A on 1X2 only",
    "D3 drop over-confident groups": "rule A without groups whose inner-season A picks were > 2 pp over-confident (>= 100 picks)",
    "D4 totals reliability 0.9": "rule A with MARKET_RELIABILITY['TOTAL'] = 0.9",
    "E1 lower confidence bound": "rule A on (B1 probability - standard error of the inner hit rate of its key and 0.05 bin)",
    "E2 candidate-count penalty": "rule A on p - lambda x log(number of candidates), lambda fitted on the inner season",
    "F1 blend Shin (inner weight)": "rule A on w p + (1 - w) q_Shin(price of the selection's market), w per group by inner log loss",
    "F2 blend proportional (inner weight)": "as F1 with proportional de-margining",
    "F3 blend 50/50": "rule A on 0.5 p + 0.5 q_Shin", "F4 market only": "rule A on q_Shin",
    "F5 blend Pinnacle": "as F1 with Pinnacle prices (folds only)",
    "G1 EV>0 (raw)": "Main = A; stake if raw EV > 0", "G2 EV>5% (raw)": "Main = A; stake if raw EV > 5 %",
    "G3 EV>0 (calibrated)": "Main = A; stake if EV with the B1 probability > 0",
    "G4 EV>3% (calibrated)": "Main = A; stake if EV with the B1 probability > 3 %",
    "G5 EV>0 (blend)": "Main = A; stake if EV with the F1 probability > 0",
    "G6 EV>2% (blend)": "Main = A; stake if EV with the F1 probability > 2 %",
    "G7 EV>0 (meta-model)": "Main = A; stake if EV with the inner logistic meta-model probability > 0 (groups seen in training)",
    "G8 EV>3% (meta-model)": "as G7 with EV > 3 %",
    "G9 agrees with market (|p-q|<3pp)": "Main = A; stake if |p - q_Shin| < 3 pp",
    "G10 odds 1.60–2.50": "Main = A; stake if the price is in 1.60–2.50",
    "G11 F1 Main, EV>0 (blend)": "Main = F1; stake if EV with the F1 probability > 0",
}


def run() -> dict:
    st = C.load_json("strategies.json")
    sel = st["modes"]["bettable"]["selection"]
    table = sel["table"]
    pooled = st["pooled_vs_A"]
    folds = st["modes"]["bettable"]["folds"]
    eligible = [n for n, t in table.items() if n != "A" and t["eligible"]]
    primary = max(eligible, key=lambda n: (table[n]["worst_roi"], table[n]["pooled_roi"]))

    def rules(n):
        t = table[n]
        return {"rule1_better_both": bool(t["better_than_A_both"]),
                "rule2_pooled_p05_gt_0": bool(pooled[n]["p05"] > 0) if n in pooled else False,
                "rule3_calibration_not_worse": bool(t["calibration_not_worse"]),
                "pooled_diff": pooled.get(n, {}).get("diff"), "pooled_p05": pooled.get(n, {}).get("p05"),
                "pooled_ci95": pooled.get(n, {}).get("ci95")}
    fam_best = sel["family_best"]
    secondary = [n for f, n in fam_best.items() if f != "A" and n != primary]
    passing = [n for n in eligible if all(v for k, v in rules(n).items() if k.startswith("rule"))]
    out = {
        "rule": "primary = highest worst-fold ROI (2023/24, 2024/25) among variants with >= 300 bets on each fold",
        "primary": primary, "secondary": secondary,
        "fold_evidence": {n: {"roi": table[n]["roi"], "bets": table[n]["bets"], "worst_roi": table[n]["worst_roi"],
                              "roi_A": [folds[str(s)]["A"]["roi"] for s in C.SELECT], **rules(n)}
                          for n in [primary] + secondary},
        "variants_passing_rules_1_to_3": passing,
        "preregistered": {
            "primary": {"name": primary, "definition": DEFINITIONS.get(primary, ""), "passes_fold_rules_1_3":
                        primary in passing},
            "secondary": [{"name": n, "definition": DEFINITIONS.get(n, "")} for n in secondary],
            "baseline": {"name": "A", "definition": DEFINITIONS["A"]},
            "fitting_for_2026_27": "every fitted component (calibration maps, blend weights, meta-model, lambda, "
                                   "dropped groups) is fitted on 2025/26 predicted by the models trained on < 2025/26 "
                                   "(walk-forward), bettable universe with Football-Data Bet365 prices; nothing is "
                                   "fitted on 2026/27",
            "universes_2026_27": {
                "V1 (primary)": "production store: latest pre-kick-off version of every 2026/27 match of the unified "
                                "history; candidates = every selection with a recorded price or, if none, the last "
                                "OddsPapi archive quote strictly before kick-off; matches with priced 1X2 and total "
                                "2.5; q = de-margined price of the same market (Shin) where all its selections are priced",
                "V2 (like for like)": "the same 2026/27 matches and model probabilities, but only the Football-Data "
                                      "markets (1X2, total 2.5, the Asian line) at Football-Data Bet365 opening prices",
                "blend weights outside the fitted groups": "double chance uses the 1X2 weight; BTTS and corners the "
                                                           "totals weight (transforms.BLEND_GROUPS)"},
            "adoption": "production changes only if the primary passes rules 1–3 on the folds AND rule 4 on 2026/27 V1",
        },
        "definitions": DEFINITIONS,
    }
    C.save_json("selection.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("selection")
    r = run()
    print(r["primary"], r["secondary"], r["variants_passing_rules_1_to_3"])
    _ = np
