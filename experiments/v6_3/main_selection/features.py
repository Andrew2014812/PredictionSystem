"""Walk-forward inputs of the selection rules (every fitted number uses only matches settled earlier).

* ``p_cal``   — Platt calibration of the candidate probabilities by market group, logit(p_cal) = a_g + b_g·logit(p)
                on the refund-free scale. Fitted on settled candidates of matches BEFORE the first day of the month
                of the match: 2025/26 backtest (+ earlier months of 2026/27). Inside 2025/26 itself (used only to
                choose parameters) the fit uses earlier months of 2025/26; August 2025 stays uncalibrated.
* ``p_gap``   — p minus the historical over-estimation of production's chosen Main of the same group
                (mean p − won of the Main, shrunk to the all-group value with 50 pseudo-matches), same windows.
* ``ncand``   — number of Main-eligible selections of the match (band, non-excluded) — winner's-curse penalty.
* ``p_bl{w}`` — w·p_model + (1 − w)·p_market (de-margined), model p where the market is incomplete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rules as R

BAND = (R.PRODUCTION.min_p, R.PRODUCTION.max_p)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def platt_fit(x: np.ndarray, y: np.ndarray, iters: int = 30) -> tuple[float, float]:
    """Logistic regression y ~ a + b·x by Newton steps (tiny ridge for stability)."""
    a, b = 0.0, 1.0
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        p = _sig(a + b * x)
        w = p * (1 - p)
        g = X.T @ (y - p) - 1e-3 * np.array([a, b - 1])
        H = (X * w[:, None]).T @ X + 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-7:
            break
    return float(a), float(b)


def _train_rows(c: pd.DataFrame) -> pd.DataFrame:
    p = c["probability"].to_numpy(float)
    ok = (~c["excluded"]) & c["status"].isin(["WON", "LOST"]) & (p > 0.05) & (p < 0.97)
    return c.loc[ok, ["group", "probability", "push_probability", "status"]]


def platt_maps(train: pd.DataFrame) -> dict:
    out = {}
    if train.empty:
        return out
    pc = train["probability"] / np.clip(1 - train["push_probability"], 1e-6, None)
    x, y = _logit(pc.to_numpy(float)), (train["status"] == "WON").to_numpy(float)
    g = train["group"].to_numpy()
    for grp in np.unique(g):
        m = g == grp
        if m.sum() >= 300:
            out[grp] = platt_fit(x[m], y[m])
    return out


def apply_platt(c: pd.DataFrame, maps: dict) -> np.ndarray:
    p = c["probability"].to_numpy(float)
    push = c["push_probability"].to_numpy(float)
    pc = p / np.clip(1 - push, 1e-6, None)
    out = p.copy()
    for grp, (a, b) in maps.items():
        m = (c["group"] == grp).to_numpy()
        out[m] = _sig(a + b * _logit(pc[m])) * (1 - push[m])
    return out


def selected_gap(c: pd.DataFrame, prior_n: float = 50.0) -> dict:
    """Over-estimation (mean p − won) of production's Main by group, shrunk to the all-group gap."""
    if c.empty:
        return {}
    main = R.select(c, R.PRODUCTION)
    main = main.loc[main["status"].isin(["WON", "LOST"])]
    if main.empty:
        return {}
    d = main["probability"] / np.clip(1 - main["push_probability"], 1e-6, None) - (main["status"] == "WON")
    allg = float(d.mean())
    s = d.groupby(main["group"]).agg(["sum", "size"])
    return {g: float((r["sum"] + prior_n * allg) / (r["size"] + prior_n)) for g, r in s.iterrows()} | {"_all": allg}


def walk_forward_columns(c: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    """Add p_cal / p_gap to ``c`` month by month; ``history`` (earlier season) is always in the training window."""
    c = c.copy()
    c["p_cal"] = c["probability"]
    c["p_gap"] = c["probability"]
    hist_train = _train_rows(history) if history is not None else None
    months = sorted(c["month"].unique())
    info = {}
    for m in months:
        before = c.loc[c["month"] < m]
        parts = [x for x in (hist_train, _train_rows(before)) if x is not None and len(x)]
        train = pd.concat(parts) if parts else pd.DataFrame()
        maps = platt_maps(train)
        rows = (c["month"] == m).to_numpy()
        target = c.loc[rows]
        c.loc[rows, "p_cal"] = apply_platt(target, maps)
        hist_c = pd.concat([x for x in (history, before) if x is not None and len(x)]) if (
            history is not None or len(before)) else pd.DataFrame()
        if len(hist_c):
            hist_c = hist_c.reset_index(drop=True).assign(mi=lambda d: pd.factorize(d["match_id"])[0])
        gaps = selected_gap(hist_c) if len(hist_c) else {}
        if gaps:
            gap = target["group"].map(gaps).fillna(gaps["_all"]).to_numpy(float)
            c.loc[rows, "p_gap"] = np.clip(target["probability"].to_numpy(float)
                                           - gap * (1 - target["push_probability"].to_numpy(float)), 1e-4, 1)
        info[m] = {"train_rows": int(len(train)), "platt": maps, "gap": gaps}
    c.attrs["walk_forward"] = info
    return c


def add_static(c: pd.DataFrame, weights=(0.75, 0.5, 0.25, 0.0)) -> pd.DataFrame:
    p = c["probability"].to_numpy(float)
    pm = c["p_mkt"].to_numpy(float)
    for w in weights:
        c[f"p_bl{int(w * 100)}"] = np.where(np.isfinite(pm), w * p + (1 - w) * pm, p)
    band = (~c["excluded"]) & c["probability"].between(*BAND)
    c["ncand"] = band.groupby(c["mi"]).transform("sum").astype(float)
    c["ncand_priced"] = (band & c["has_odds"]).groupby(c["mi"]).transform("sum").astype(float)
    return c
