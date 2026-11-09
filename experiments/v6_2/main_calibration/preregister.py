"""Writes results/preregistration.json: the frozen post-selection calibration of the shown Main and its forward test.

Run after run.py and commit the file BEFORE any match of the forward window is played. The forward window starts
on the day after the commit; evaluation uses only matches played in it.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from src import config

from . import common as C
from . import methods as M


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_monthly_fit() -> dict:
    """The H1 map that would be applied to matches of the next month: Platt on all settled shown 2026/27 Mains."""
    from experiments.v6_1.main import candidates as CD
    st = CD.stored_live_mains()
    mk = pd.read_parquet(config.PREDICTIONS_DIR / "markets.parquet",
                         columns=["match_id", "market", "selection", "push_probability"]).drop_duplicates(
        ["match_id", "market", "selection"], keep="last")
    st = st.merge(mk, how="left", on=["match_id", "market", "selection"])
    dec = st.loc[st["status"].isin(["WON", "LOST"])]
    pc = dec["probability"].astype(float) / (1 - dec["push_probability"].fillna(0.0))
    m = M.Platt().fit(pc.to_numpy(), (dec["status"] == "WON").astype(float).to_numpy())
    return {"fitted_on_settled_shown_mains": int(len(dec)), "last_match_date": str(pd.to_datetime(dec["date"]).max().date()),
            "a": m.a, "b": m.b}


def main() -> None:
    C.setup_logging()
    res = C.load_json("variants.json")
    sel = C.load_json("selection.json")
    info26 = res["info"]["site"][str(C.CURRENT)]
    v26 = res["variants"]["site"][str(C.CURRENT)]
    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit_at_writing": C.git_sha(),
        "study": "v6.2 Main Calibration 2.0",
        "status_of_seasons": {"selection": ["2023/24", "2024/25"], "information": ["2025/26"],
                              "current_already_looked_at": "2026/27 up to the data snapshot below (NOT a holdout)"},
        "data_snapshot": {"markets.parquet": _sha(config.PREDICTIONS_DIR / "markets.parquet"),
                          "picks.parquet": _sha(config.PREDICTIONS_DIR / "picks.parquet"),
                          "data_version": C.data_version()},
        "selection_by_frozen_rule": {"rule": sel["rule"]["calibration"], "chosen": sel["calibration"]["chosen"],
                                     "promising": sel["calibration"]["promising"]},
        "why_these_hypotheses": (
            "On 2023/24-2025/26 every post-selection map removes the over-estimation of the chosen Main; on 2026/27 "
            "the maps fitted on walk-forward history do not transfer to production (gap of the shown Main "
            f"{v26['A как показано']['gap']:+.4f}; after PS-Platt {v26['показано + PS-Platt']['gap']:+.4f}), while the "
            f"same map refitted month by month on production's own shown Mains gives {v26['показано + PS-Platt по месяцам']['gap']:+.4f} "
            f"and the group-wise history map gives {v26['показано + PS-Group']['gap']:+.4f}. Both observations were made "
            "on 2026/27 after looking at it; this file turns them into forward hypotheses."),
        "hypotheses": {
            "H1_primary": {
                "name": "PS-Platt по месяцам (на показанных Main продакшна)",
                "rule": "For a Main shown for a match in calendar month m: stated probability p' = sigmoid(a_m + b_m * "
                        "logit(pc)), pc = probability / (1 - push_probability). (a_m, b_m) = logistic regression "
                        "(sklearn, C = 1e4) of WON (1) / LOST (0) on logit(pc) over ALL settled shown Mains of the "
                        "unified history of season 2026/27 (src.analytics.performance.unified_history, role main) whose "
                        "match date is before the first day of month m. The pick itself is unchanged.",
                "current_fit_for_information": current_monthly_fit()},
            "H2_challenger": {
                "name": "PS-Group (карты по группам рынков, история 2022/23-2025/26)",
                "rule": "p' = sigmoid(a_g + b_g * logit(pc)) with the frozen per-group parameters below (group of the "
                        "Main; 'pool' for any other group). The pick itself is unchanged.",
                "frozen_parameters": info26["PS-Group"]},
        },
        "forward_test": {
            "window": "matches played from the day after the commit of this file until the earlier of (a) 600 settled "
                      "(WON/LOST) shown Mains or (b) the end of season 2026/27",
            "population": "shown Main of every match of the unified history in the window (src.analytics.performance."
                          "unified_history, role main, status WON/LOST); refunds excluded as in production",
            "metric": "gap = mean(p') - hit rate; raw gap = mean(pc) - hit rate; 2000 bootstrap resamples by match, "
                      "seed 20261109",
            "criteria": ["C1: the 95 % CI of gap(p') contains 0",
                         "C2: the paired 95 % CI of |gap(p')| - |gap(pc)| lies entirely below 0",
                         "C3: Holm correction over the two hypotheses (H1, H2) for C2 (p < 0.05 after correction)"],
            "decision": "A (adopt the stated-probability calibration of the shown Main, display only, picks unchanged) "
                        "if C1-C3 hold for H1, else for H2; otherwise C. No change to the Main pick or to betting is "
                        "implied: the ROI of the Main is not affected by either map.",
        },
        "not_registered": "No ROI hypothesis: no Main rule improved ROI on both selection seasons with a Holm-corrected "
                          "pooled test (selection.json, roi.promising = false).",
    }
    C.save_json("preregistration.json", doc)


if __name__ == "__main__":
    main()
