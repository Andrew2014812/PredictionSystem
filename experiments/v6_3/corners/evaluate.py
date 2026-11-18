"""Evaluation of the corner models and distributions (light: no tree fitting; reads cache/preds_*.parquet).

Selection protocol (nothing is chosen on 2026/27):
* mean model — lowest Poisson deviance on 2025/26 (models fitted on 2021/22–2024/25);
* distribution — lowest log score on 2025/26 with two-fold cross-fitting inside the season (parameters fitted on
  Jul–Dec, scored on Jan–Jun and vice versa); the final parameters are then fitted on the whole of 2025/26 and
  applied unchanged to 2026/27;
* calibration of the over / under lines — kept only if it improves the cross-fitted log loss on 2025/26.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.corners import data as D
from experiments.v6_3.corners import dists as Q
from experiments.v6_3.corners import models as M

PROD_R = None          # production nb_size, filled from the bundle
TREE_MODELS = ["xgb_prod", "xgb_stats", "xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha"]
LABELS = {"xgb_prod": "XGBoost продакшн", "xgb_prod_refit": "XGBoost продакшн (переобучен)",
          "xgb_stats": "XGBoost + новые признаки", "xgb_mkt": "XGBoost + новые + рынок",
          "lgbm_mkt": "LightGBM", "cat_mkt": "CatBoost", "xgb_ha": "XGBoost хозяева + гости",
          "rating": "Динамические рейтинги", "glm": "Poisson GLM", "naive": "Среднее лиги",
          "ens_w": "Ансамбль (веса 2025/26)", "ens_eq": "Ансамбль (равные веса)"}
DISTS = ["poisson", "nb_prod", "nb_mom", "nb", "nb_league", "nbp", "cmp", "conv_ind", "conv_frank"]
DLABELS = {"poisson": "Пуассон", "nb_prod": "NB, r продакшна", "nb_mom": "NB, r по моментам",
           "nb": "NB, r по ML", "nb_league": "NB, r по лигам", "nbp": "NB-P (r по матчу)",
           "cmp": "COM-Пуассон", "conv_ind": "Свёртка NB, независимо", "conv_frank": "Свёртка NB, копула Франка"}
ENS_PARTS = ["xgb_mkt", "lgbm_mkt", "cat_mkt", "xgb_ha", "rating", "glm"]


def prod_r() -> float:
    from experiments.v6_3.replay.engine import production_bundles
    return float(production_bundles()["corners"].extra["nb_size"])


def leagues() -> list[str]:
    from experiments.v6_3.replay.engine import production_bundles
    return list(production_bundles()["corners"].extra["leagues"])


# ---------------------------------------------------------------------------------------------- tables
def stage_table(stage: str, frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Rows of the predicted season with the target, the side counts and every model's mean."""
    p = pd.read_parquet(D.CACHE / f"preds_{stage}.parquet")
    cols = ["match_id", "date", "season", "league", "home_team", "away_team", "played", "total_corners",
            "home_corners", "away_corners", "rc_mu_h", "rc_mu_a", "league_avg_corners", "h_x_nhist", "a_x_nhist"]
    t = frame[cols].merge(p, on="match_id", how="inner")
    t = t.loc[t["league"].isin(leagues())].reset_index(drop=True)
    t["mu_rating"] = t["rc_mu_h"] + t["rc_mu_a"]
    t["mu_h_rating"], t["mu_a_rating"] = t["rc_mu_h"], t["rc_mu_a"]
    t["mu_naive"] = t["league_avg_corners"].astype(float)
    tgt = frame.set_index("match_id").loc[t["match_id"]].reset_index()
    t["mu_glm"] = M.glm_predictions(frame, leagues(), cut, tgt)
    return t


def add_ensembles(t: pd.DataFrame, weights: dict) -> pd.DataFrame:
    P = t[[f"mu_{k}" for k in ENS_PARTS]].to_numpy(float)
    t["mu_ens_w"] = P @ np.array([weights[k] for k in ENS_PARTS])
    t["mu_ens_eq"] = P.mean(1)
    return t


def scored(t: pd.DataFrame) -> pd.DataFrame:
    return t.loc[t["played"].astype(bool) & t["total_corners"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- mean metrics
def point_metrics(y, mu) -> dict:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    r = y - mu
    return {"n": int(len(y)), "mae": float(np.abs(r).mean()), "rmse": float(np.sqrt((r ** 2).mean())),
            "r2": float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()), "poisson_dev": D.poisson_dev(y, mu),
            "mean_pred": float(mu.mean()), "mean_true": float(y.mean())}


def unit_poisson_dev(y, mu) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (t - (y - mu))


# ---------------------------------------------------------------------------------------------- distributions
def fit_dist(name: str, y, mu, mu_h=None, mu_a=None, league=None, r_prod=None) -> dict:
    if name == "poisson":
        return {}
    if name == "nb_prod":
        return {"r": r_prod}
    if name == "nb_mom":
        return {"r": Q.nb_size_mom(y, mu) or 1e5}
    if name == "nb":
        return {"r": Q.fit_nb(y, mu)}
    if name == "nb_league":
        r = Q.fit_nb(y, mu)
        return {"r": r, "by_league": Q.fit_nb_league(y, mu, league, r)}
    if name == "nbp":
        a, p = Q.fit_nbp(y, mu)
        return {"alpha": a, "p": p}
    if name == "cmp":
        return {"nu": Q.fit_cmp(y, mu)}
    if name == "conv_ind":
        return Q.fit_conv(y, mu_h, mu_a, independent=True)
    if name == "conv_frank":
        return Q.fit_conv(y, mu_h, mu_a)
    raise KeyError(name)


def make_pmf(name: str, prm: dict, mu, mu_h=None, mu_a=None, league=None) -> np.ndarray:
    if name == "poisson":
        return Q.pmf_poisson(mu)
    if name in ("nb_prod", "nb_mom", "nb"):
        return Q.pmf_nb(mu, prm["r"])
    if name == "nb_league":
        r = np.array([prm["by_league"].get(str(L), prm["r"]) for L in league])
        return Q.pmf_nb(mu, r)
    if name == "nbp":
        return Q.pmf_nbp(mu, prm["alpha"], prm["p"])
    if name == "cmp":
        return Q.pmf_cmp(mu, prm["nu"])
    if name in ("conv_ind", "conv_frank"):
        return Q.pmf_conv(mu_h, mu_a, prm["r_h"], prm["r_a"], prm["theta"])
    raise KeyError(name)


def has_sides(model: str) -> bool:
    return model in ("xgb_ha", "rating")


def dist_args(t: pd.DataFrame, model: str) -> dict:
    out = {"mu": t[f"mu_{model}"].to_numpy(float), "league": t["league"].to_numpy()}
    if has_sides(model):
        out["mu_h"], out["mu_a"] = t[f"mu_h_{model}"].to_numpy(float), t[f"mu_a_{model}"].to_numpy(float)
    return out


def dist_names(model: str) -> list[str]:
    return [d for d in DISTS if has_sides(model) or not d.startswith("conv")]


def line_block(y, pmf) -> dict:
    out = {}
    for line in D.LINES:
        p = Q.p_over(pmf, line)
        o = (np.asarray(y) > line).astype(float)
        out[str(line)] = {"ll": float(Q.binary_ll(o, p).mean()), "brier": float(((p - o) ** 2).mean()),
                          "ece": Q.ece(o, p), "mean_p": float(p.mean()), "freq": float(o.mean())}
    out["ll_mean"] = float(np.mean([out[str(x)]["ll"] for x in D.LINES]))
    return out


def dist_metrics(y, pmf) -> dict:
    return {"log_score": float(Q.log_score(pmf, y).mean()), "rps": float(Q.rps(pmf, y).mean()),
            "lines": line_block(y, pmf)}


def crossfit_dist(t: pd.DataFrame, model: str, name: str, r_prod: float) -> dict:
    """Two-fold cross-fitted log score inside the selection season (halves by date)."""
    half = t["date"] < pd.Timestamp(f"{D.SELECT + 1}-01-01")
    y = t["total_corners"].to_numpy(float)
    a = dist_args(t, model)
    ls, ll = np.zeros(len(t)), {line: np.zeros(len(t)) for line in D.LINES}
    for fit_mask in (half.to_numpy(), ~half.to_numpy()):
        sub = {k: v[fit_mask] for k, v in a.items()}
        prm = fit_dist(name, y[fit_mask], r_prod=r_prod, **sub)
        other = {k: v[~fit_mask] for k, v in a.items()}
        pmf = make_pmf(name, prm, **other)
        ls[~fit_mask] = Q.log_score(pmf, y[~fit_mask])
        for line in D.LINES:
            ll[line][~fit_mask] = Q.binary_ll((y[~fit_mask] > line).astype(float), Q.p_over(pmf, line))
    return {"log_score_cf": float(ls.mean()), "ll_mean_cf": float(np.mean([ll[x].mean() for x in D.LINES]))}


# ---------------------------------------------------------------------------------------------- calibration
def platt_fit(o, p):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m = LogisticRegression(C=1e4).fit(z[:, None], o)
    return lambda q: m.predict_proba(np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))[:, None])[:, 1]


def isotonic_fit(o, p):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, o)
    return lambda q: m.predict(q)


CALIB = {"platt": platt_fit, "isotonic": isotonic_fit}


def calibrate_lines(p_fit: dict, y_fit, p_apply: dict, method: str) -> dict:
    out = {}
    for line in D.LINES:
        f = CALIB[method]((np.asarray(y_fit) > line).astype(float), p_fit[line])
        out[line] = np.clip(f(p_apply[line]), 1e-4, 1 - 1e-4)
    return out


def crossfit_calibration(t: pd.DataFrame, pover: dict, method: str) -> dict:
    half = (t["date"] < pd.Timestamp(f"{D.SELECT + 1}-01-01")).to_numpy()
    y = t["total_corners"].to_numpy(float)
    ll = {line: np.zeros(len(t)) for line in D.LINES}
    for fm in (half, ~half):
        cal = calibrate_lines({k: v[fm] for k, v in pover.items()}, y[fm], {k: v[~fm] for k, v in pover.items()},
                              method)
        for line in D.LINES:
            ll[line][~fm] = Q.binary_ll((y[~fm] > line).astype(float), cal[line])
    raw = {line: Q.binary_ll((y > line).astype(float), pover[line]).mean() for line in D.LINES}
    return {"ll_cf": {str(k): float(v.mean()) for k, v in ll.items()}, "ll_raw": {str(k): float(v) for k, v in raw.items()},
            "ll_mean_cf": float(np.mean([v.mean() for v in ll.values()])),
            "ll_mean_raw": float(np.mean(list(raw.values())))}


def expanding_calibration(sel: pd.DataFrame, p_sel: dict, ev: pd.DataFrame, p_ev: dict, method: str) -> dict:
    """Monthly expanding map inside 2026/27: a match of month M uses a map fitted on 2025/26 and the
    settled 2026/27 matches before the 1st of M."""
    out = {line: np.full(len(ev), np.nan) for line in D.LINES}
    months = ev["date"].dt.to_period("M")
    y_sel = sel["total_corners"].to_numpy(float)
    for mth in sorted(months.unique()):
        start = mth.to_timestamp()
        tgt = (months == mth).to_numpy()
        past = (ev["date"] < start).to_numpy() & ev["played"].astype(bool).to_numpy() & ev["total_corners"].notna().to_numpy()
        y_fit = np.concatenate([y_sel, ev["total_corners"].to_numpy(float)[past]])
        p_fit = {line: np.concatenate([p_sel[line], p_ev[line][past]]) for line in D.LINES}
        cal = calibrate_lines(p_fit, y_fit, {line: p_ev[line][tgt] for line in D.LINES}, method)
        for line in D.LINES:
            out[line][tgt] = cal[line]
    return out


def lines_from_pover(y, pover: dict) -> dict:
    out = {}
    for line in D.LINES:
        p = pover[line]
        o = (np.asarray(y) > line).astype(float)
        out[str(line)] = {"ll": float(Q.binary_ll(o, p).mean()), "brier": float(((p - o) ** 2).mean()),
                          "ece": Q.ece(o, p), "mean_p": float(p.mean()), "freq": float(o.mean())}
    out["ll_mean"] = float(np.mean([out[str(x)]["ll"] for x in D.LINES]))
    return out


def paired_lines(y, p_new: dict, p_old: dict) -> dict:
    """Paired bootstrap of the mean line log loss difference (new − old), per line and averaged."""
    out = {}
    d_all = np.zeros(len(y))
    for line in D.LINES:
        o = (np.asarray(y) > line).astype(float)
        d = Q.binary_ll(o, p_new[line]) - Q.binary_ll(o, p_old[line])
        d_all += d / len(D.LINES)
        out[str(line)] = C.paired_mean_diff(Q.binary_ll(o, p_new[line]), Q.binary_ll(o, p_old[line]))
    out["mean"] = C.paired_mean_diff(d_all, np.zeros(len(y)))
    return out
