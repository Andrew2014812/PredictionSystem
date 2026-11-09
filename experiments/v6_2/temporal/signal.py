"""Does a candidate feature carry information that the model (or the market) has not already used?

On the walk-forward predictions of the production model B0 (seed 42) for the selection seasons 2023/24 + 2024/25
(2025/26 for information), the residuals

    r_model  = (3·1[H] + 1[D]) − (3 p_H + p_D)        home points minus the model's expected home points
    r_market = (3·1[H] + 1[D]) − (3 q_H + q_D)        the same against the de-margined Bet365 opening price
    r_close  = the same against the de-margined Bet365 CLOSING price (for information)
    r_goals  = (home goals − λ_home) − (away goals − λ_away)

are correlated (Pearson, feature winsorised at 1 % / 99 %) with every candidate temporal feature. A feature that the model already contains has
correlation ≈ 0 with r_model; a correlation with r_market would mean the market misses it as well.
p-values: Fisher z with n − 3 degrees of freedom; Holm across all features × residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from experiments.v6_2.leagues import wf

from . import common as C
from . import variants as V


def rows(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        fr = wf.frame(s)
        fr = fr.loc[fr["in_eval"]].reset_index(drop=True)
        p = fr[["match_id"]].merge(wf.base_predictions(s, C.PROD_SEED), on="match_id", how="left")
        pts = np.where(fr["y"] == 0, 3.0, np.where(fr["y"] == 1, 1.0, 0.0))
        out.append(pd.DataFrame({"match_id": fr["match_id"], "season": s, "league": fr["league"],
                                 "r_model": pts - (3 * p["pH"] + p["pD"]),
                                 "r_market": pts - (3 * fr["qH"] + fr["qD"]),
                                 "r_close": pts - (3 * fr["cH"] + fr["cD"]),
                                 "r_goals": (fr["home_goals"] - p["lh"]) - (fr["away_goals"] - p["la"])}))
    r = pd.concat(out, ignore_index=True)
    prod = wf.features()[["match_id", *PROD_CONTROLS]]
    return r.merge(V.temporal_features(), on="match_id", how="left").merge(prod, on="match_id", how="left")


# production features shown as controls (already in the model)
PROD_CONTROLS = ["diff_form5_ppg", "diff_form10_ppg", "h_form5_ppg", "a_form5_ppg", "diff_season_ppg", "elo_diff",
                 "h_trend_scoring", "a_trend_scoring", "h_rest_days", "a_rest_days"]


DECILE_FEATURES = ["diff_ew8_ppg", "diff_form5_ppg", "diff_elo_d10"]


def corr(x, y) -> dict:
    """Pearson correlation of the residual with the feature (winsorised at the 1st / 99th percentile).

    Pearson, not Spearman: for calibrated probabilities E[r | x] = 0, so the covariance is 0 whatever x is, while a
    rank correlation of a three-valued outcome minus its expectation is not 0 even for perfectly calibrated
    forecasts (the residual's skew depends on the favourite's probability)."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 50:
        return {"n": n, "rho": None, "p": None}
    xv = x[ok]
    lo, hi = np.quantile(xv, [0.01, 0.99])
    xv = np.clip(xv, lo, hi)
    if xv.std() == 0:
        return {"n": n, "rho": None, "p": None}
    rho = float(np.corrcoef(xv, y[ok])[0, 1])
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(np.clip(rho, -0.999, 0.999))
    return {"n": n, "rho": rho, "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
            "p": float(2 * (1 - stats.norm.cdf(abs(z) / se))),
            "spearman": float(stats.spearmanr(x[ok], y[ok]).statistic)}   # shown only to document the artefact


def run() -> dict:
    res = {}
    feats = V.new_columns() + PROD_CONTROLS
    for label, seasons in (("select", C.SELECT), ("2025", (C.INFO,))):
        r = rows(seasons)
        table, pv = {}, {}
        for f in feats:
            x = r[f].to_numpy(float)
            table[f] = {t: corr(x, r[t].to_numpy(float)) for t in ("r_model", "r_market", "r_close", "r_goals")}
            for t in ("r_model", "r_market", "r_close", "r_goals"):
                pv[f"{f}|{t}"] = table[f][t]["p"]
        adj = C.holm(pv)
        for f in feats:
            for t in ("r_model", "r_market", "r_close", "r_goals"):
                table[f][t]["p_holm"] = adj.get(f"{f}|{t}")
        dec = {}
        for f in DECILE_FEATURES:
            q = pd.qcut(r[f].rank(method="first"), 10, labels=False)
            dec[f] = [{"decile": int(k), "x": float(r.loc[q == k, f].mean()),
                       **{t: C.mean_ci(r.loc[q == k, t].to_numpy(float)) for t in ("r_model", "r_market")}}
                      for k in range(10)]
        res_dec = dec
        res[label] = {"n": int(len(r)), "features": table, "deciles": res_dec, "n_tests": len([v for v in pv.values() if v is not None])}
    C.save_json("signal.json", res)
    return res
