"""Freezes the B candidates of the calibration study (results/preregistration.json) for a forward check.

    python -m experiments.v6_3.calibration.preregister
"""
from __future__ import annotations

import pandas as pd

from experiments.v6_3 import common as C

from . import data as D
from .conclusions import decisions


def main() -> None:
    r = C.read_json(D.DIRS["results"] / "calibration.json")
    dec = decisions(r)
    now = pd.Timestamp.now(tz="Europe/London").isoformat(timespec="seconds")
    entries = []
    for name, d in dec.items():
        if d["decision"] != "B":
            continue
        m, s, src = name.split("|")
        entries.append({
            "id": f"v63_calibration_{m}_{s}_{src}@v1",
            "hypothesis": "The production Main rule applied to probabilities calibrated by this map (fitted only on "
                          "matches settled before the predicted match's month) gives a higher Main ROI than the "
                          "production rule on raw probabilities, with a smaller stated-probability gap.",
            "exact_rule": {"map": m, "scope": s, "fit_source": src,
                           "fit": "2025/26 backtest predictions plus every 2026/27 match settled before the 1st of the "
                                  "match's month" if src == "pooled" else src,
                           "scale": "pc = p / (1 - P(push)); p' = pc' (1 - P(push))",
                           "main_rule": "src/markets/recommendation.main_prediction with unchanged parameters, "
                                        "band and EV on the calibrated probability"},
            "primary_metric": "paired ROI difference vs Baseline B (production rule, raw probabilities), unit stake, "
                              "close prices; secondary: t09 prices, gap stated p - hit rate of the Main",
            "evidence_2026_27": {"close": d["close"], "t09": d["t09"]},
            "minimum_forward_sample": "1500 settled Main bets after frozen_at (about 3 months) before any decision",
            "frozen_at": now})
    C.write_json(D.DIRS["results"] / "preregistration.json", {"frozen_at": now, "entries": entries,
                                                              "decisions": dec})
    print(len(entries), "entries")


if __name__ == "__main__":
    main()
