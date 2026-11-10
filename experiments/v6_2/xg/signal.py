"""Univariate signal of the pre-match xG features (step 2-3 of the study).

On the evaluation matches (played, xG leagues, from EVAL_START):

* correlation of every pre-match feature with the B0 residuals (goals - lambda_B0 per side, and the 1X2
  residual (1[H] - pH) - (1[A] - pA)) — what the baseline does NOT already know;
* the same for goal / shot based analogues that the baseline already has, as a reference;
* "is xG more predictive than goals?": correlation of the pre-match xG difference and of the pre-match goal
  difference (same window, same matches) with the goal difference of the match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from . import models as M

XG_FEATS = {"zh1": "рейтинг xG (лог, хозяева)", "zh2": "xG за, сжатое (хозяева)", "zh3": "xG против соперника, сжатое",
            "zh4": "голы − xG (реализация хозяев)", "h_xgf10": "xG за, 10 матчей", "h_xga10": "xG против, 10 матчей",
            "h_xgf_v5": "xG за дома, 5 матчей", "h_xgf_ewm": "xG за, EWM", "h_gmx10": "голы − xG, 10 матчей",
            "d1": "разница рейтингов xG", "d2": "разница сжатых xG", "d4": "разница реализации"}
REF_FEATS = {"h_form10_gf": "голы за, 10 (есть в v6.1)", "h_form10_sot_for": "удары в створ, 10 (есть в v6.1)",
             "h_form10_shots_for": "удары, 10 (есть в v6.1)", "h_form10_adj_gf": "голы за с поправкой (есть)",
             "elo_diff": "разница Эло (есть)"}


def _corr(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return {"r": None, "n": int(m.sum())}
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    se = 1 / np.sqrt(m.sum() - 3)
    z = np.arctanh(r)
    return {"r": r, "lo": float(np.tanh(z - 1.96 * se)), "hi": float(np.tanh(z + 1.96 * se)), "n": int(m.sum())}


def run(df: pd.DataFrame) -> dict:
    ref = D.season_features(columns=["match_id"] + [c for c in REF_FEATS] + ["a_form10_gd", "h_form10_gd",
                                                                                "a_form10_gf", "a_form10_sot_for"])
    e = df.loc[df["played"].astype(bool) & (df["date"] >= C.EVAL_START) & df["xg_ok"]].merge(ref, on="match_id", how="left")
    res_h = e["home_goals"] - e["lh"]
    res_1x2 = ((e["y"] == 0).astype(float) - e["pH"]) - ((e["y"] == 2).astype(float) - e["pA"])
    gd = e["home_goals"] - e["away_goals"]
    rows = []
    for f, label in {**XG_FEATS, **REF_FEATS}.items():
        target = res_1x2 if f.startswith("d") or f == "elo_diff" else res_h
        rows.append({"feature": f, "label": label, "kind": "xG" if f in XG_FEATS else "v6.1",
                     "target": "1X2" if target is res_1x2 else "голы хозяев",
                     "resid": _corr(e[f], target), "raw": _corr(e[f], gd if target is res_1x2 else e["home_goals"])})
    # xG difference vs goal difference over the same window (last 10, matches with >= 5 xG matches each)
    s = e.loc[e["min_xg_n"] >= 5]
    pred = {"xgd10_diff": s["h_xgd10"] - s["a_xgd10"],
            "gd10_diff": s["h_form10_gd"] - s["a_form10_gd"]}
    head = {k: _corr(v, s["home_goals"] - s["away_goals"]) for k, v in pred.items()}
    head["n"] = int(len(s))
    # bootstrap the difference of the two correlations (paired by match)
    x1, x2, yv = pred["xgd10_diff"].to_numpy(float), pred["gd10_diff"].to_numpy(float), (s["home_goals"] - s["away_goals"]).to_numpy(float)
    ok = np.isfinite(x1) & np.isfinite(x2)
    x1, x2, yv = x1[ok], x2[ok], yv[ok]
    diffs = []
    for idx in C.boot_index(len(yv), 1000):
        for r in idx:
            diffs.append(np.corrcoef(x1[r], yv[r])[0, 1] - np.corrcoef(x2[r], yv[r])[0, 1])
    head["diff_r"] = float(np.corrcoef(x1, yv)[0, 1] - np.corrcoef(x2, yv)[0, 1])
    head["diff_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    # does the xG difference add to the goal difference? partial correlation via residuals
    A = np.c_[np.ones_like(x2), x2]
    r_y = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
    r_x = x1 - A @ np.linalg.lstsq(A, x1, rcond=None)[0]
    head["partial_xg_given_gd"] = float(np.corrcoef(r_x, r_y)[0, 1])
    out = {"n": int(len(e)), "features": rows, "xg_vs_goals": head,
           "finishing_regression": _corr(e["zh4"], res_h)}
    C.save_json("signal.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("signal")
    run(M.frame(0))
