"""Freezes the B candidates of the meta-selection study (results/preregistration.json) for a forward check.

    python -m experiments.v6_3.meta_selection.preregister
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

from . import features as F
from .conclusions import decisions

RULES = {
    "meta_lr_ev": "StandardScaler + LogisticRegression(C=0.1) on the study features; Main = max P_meta (1 - P(push)) "
                  "odds - (1 - P(push)) over priced Main-eligible candidates",
    "meta_gbm_ev": "XGBoost classifier depth 3, 300 trees, lr 0.03, subsample 0.8, colsample 0.8, min_child_weight 20, "
                   "reg_lambda 5; Main = max EV_meta over priced Main-eligible candidates",
    "meta_gbm_ret": "XGBoost regressor of the unit profit (same settings, min_child_weight 50); Main = max predicted "
                    "return over priced Main-eligible candidates",
    "meta_lr_rule": "production Main rule with the model probability replaced by P_meta (1 - P(push)) of meta_lr_ev",
    "rule_priced": "production Main rule (raw model probabilities) restricted to priced Main-eligible candidates",
    "max_p": "Main = highest model probability among priced Main-eligible candidates",
    "max_ev": "Main = highest EV (model probability) among priced Main-eligible candidates",
    "max_q": "Main = highest de-margined market probability among priced Main-eligible candidates",
    "rule_calibrated": "production Main rule on probabilities calibrated by the study 2 primary map, priced candidates",
}


def main() -> None:
    r = C.read_json(F.DIRS["results"] / "meta_selection.json")
    dec = decisions(r)
    now = pd.Timestamp.now(tz="Europe/London").isoformat(timespec="seconds")
    entries = []
    for name, d in dec.items():
        if d["decision"] != "B":
            continue
        entries.append({
            "id": f"v63_meta_selection_{name}@v1",
            "hypothesis": "Choosing the Main among the priced candidates with this rule gives a higher Main ROI than "
                          "the production rule (Baseline B) on the same matches.",
            "exact_rule": {"rule": RULES[name], "pool": "selections the production rule may choose (not European "
                           "handicap / exact score, model p in [0.40, 0.85]) with a real price at the decision moment",
                           "training": "2025/26 backtest pool + every 2026/27 match settled before the 1st of the "
                                       "match's month; refit monthly" if name.startswith("meta") else "no training",
                           "fallback": "production Main when the match has no priced candidate"},
            "primary_metric": "paired ROI difference vs Baseline B, unit stake, close prices; secondary: t09 prices",
            "evidence_2026_27": {"close": d["close"], "t09": d["t09"]},
            "minimum_forward_sample": "1500 settled Main bets after frozen_at (about 3 months) before any decision",
            "frozen_at": now})
    C.write_json(F.DIRS["results"] / "preregistration.json", {"frozen_at": now, "entries": entries, "decisions": dec})
    print(len(entries), "entries")


if __name__ == "__main__":
    main()
