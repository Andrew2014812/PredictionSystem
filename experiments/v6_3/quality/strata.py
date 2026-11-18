"""How data completeness and quality relate to model quality and Main — strata of the production replay.

2026/27: production outputs (Baseline B's models) and Baseline B Main in both universes. 2024/25 + 2025/26:
out-of-sample refits of the production models (oos.py) against the Football-Data market, pooled ("history") —
more matches, and the same strata, so an effect seen only in 2026/27 can be told from a stable one.

Nothing is excluded: every played match belongs to exactly one bin of each dimension; Main ROI by stratum is a
description, not a selection rule (the study keeps one Main per match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C

from . import data as Q

DQ_EDGES = [-0.001, 0.0005, 0.05, 0.2, 1.0]
DQ_LABELS = ["0 %", "0–5 %", "5–20 %", "> 20 %"]
SMALL = 100           # fewer bets than this -> flagged as a small sample


def mean_ci(v) -> dict:
    v = np.asarray(pd.Series(v).dropna(), float)
    if len(v) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    vals = np.concatenate([v[idx].mean(1) for idx in C.boot_idx(len(v))])
    return {"mean": float(v.mean()), "ci95": C.ci(vals), "n": int(len(v))}


def diff_ci(a, b) -> dict:
    """mean(b) − mean(a), independent bootstrap of the two groups; two-sided p of 'difference = 0'."""
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) == 0 or len(b) == 0:
        return {"diff": None, "ci95": [None, None], "p": None}
    rng = np.random.default_rng(C.SEED)
    va = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(C.N_BOOT)])
    vb = np.array([b[rng.integers(0, len(b), len(b))].mean() for _ in range(C.N_BOOT)])
    d = vb - va
    return {"diff": float(b.mean() - a.mean()), "ci95": C.ci(d),
            "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}


def main_roi(g: pd.DataFrame, u: str) -> dict:
    bets = g.loc[g[f"main_{u}_bet"] == 1]
    r = C.roi_ci(g[f"main_{u}_profit"].fillna(0).to_numpy(), g[f"main_{u}_bet"].to_numpy()) if len(bets) else \
        {"roi": None, "ci95": [None, None], "bets": 0}
    r["hit"] = float(g[f"main_{u}_won"].mean()) if g[f"main_{u}_won"].notna().any() else None
    r["small"] = r["bets"] < SMALL
    return r


def metrics(g: pd.DataFrame, current: bool) -> dict:
    out = {"matches": int(len(g)), "log_loss": mean_ci(g["ll"]), "dev_home": float(g["dev_home"].mean()),
           "dev_away": float(g["dev_away"].mean()), "dev_corners": float(g["dev_corners"].mean())
           if g["dev_corners"].notna().any() else None}
    if current:
        out["excess_close"] = mean_ci(g["excess_close"])
        out["excess_t09"] = mean_ci(g["excess_t09"])
        for u in ("close", "t09"):
            out[f"main_{u}"] = main_roi(g, u)
    else:
        out["excess_fd"] = mean_ci(g["excess_fd"])
        if "main_fd_bet" in g:
            out["main_fd"] = main_roi(g, "fd")
    return out


def add_bins(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["dq_bin"] = pd.cut(x["dq_missing"], DQ_EDGES, labels=DQ_LABELS).astype(str)
    x["stats_bin"] = np.where(x["stats_missing"] > 0, "partial", "full")
    x["newcomer_like"] = np.where(x["hist_min"] < 10, "history < 10", "history 10")
    x["club_early"] = x["club_group"] + np.where(x["n_min"] <= 5, " | 0–5", " | 6+")
    return x


DIMENSIONS = ["dq_bin", "stats_bin", "club_group", "club_early", "n_bucket", "newcomer_like", "tier", "league"]
REFERENCE = {"dq_bin": "0 %", "stats_bin": "full", "club_group": "established", "club_early": "established | 6+",
             "n_bucket": "10+", "newcomer_like": "history 10", "tier": 1}


def by_dimension(x: pd.DataFrame, current: bool) -> dict:
    out = {}
    for dim in DIMENSIONS:
        res = {}
        for key, g in x.groupby(dim):
            res[str(key)] = metrics(g, current)
        ref = REFERENCE.get(dim)
        if ref is not None:
            base = x.loc[x[dim] == ref]
            col = "excess_close" if current else "excess_fd"
            for key, g in x.groupby(dim):
                if key == ref:
                    continue
                d = {"excess_ll_diff": diff_ci(base[col], g[col]), "ll_diff": diff_ci(base["ll"], g["ll"])}
                if current:
                    for u in ("close", "t09"):
                        pa = base.loc[base[f"main_{u}_bet"] == 1, f"main_{u}_profit"]
                        pb = g.loc[g[f"main_{u}_bet"] == 1, f"main_{u}_profit"]
                        d[f"roi_diff_{u}"] = diff_ci(pa, pb)
                res[str(key)]["vs_reference"] = d
        out[dim] = res
    return out


def price_availability(x: pd.DataFrame) -> dict:
    """t09: matches whose Main had no price at T vs priced, judged at closing prices (same matches, close)."""
    out = {}
    for flag, g in x.groupby("main_t09_priced"):
        out["t09_main_priced" if flag else "t09_main_unpriced"] = {
            "matches": int(len(g)), "close": main_roi(g, "close"), "t09": main_roi(g, "t09"),
            "excess_close": mean_ci(g["excess_close"]),
            "t09_main_groups": g["main_t09_group"].value_counts().to_dict()}
    for flag, g in x.groupby("oddspapi_at_t09"):
        out["oddspapi_at_t09" if flag else "football_data_only_at_t09"] = {
            "matches": int(len(g)), "close": main_roi(g, "close"), "t09": main_roi(g, "t09"),
            "n_priced_median": float(g["n_priced_t09"].median())}
    x = x.assign(np_bin=pd.cut(x["n_priced_t09"], [-1, 10, 30, 60, 1000], labels=["≤10", "11–30", "31–60", "> 60"]))
    out["by_priced_count"] = {str(k): {"matches": int(len(g)), "t09": main_roi(g, "t09"), "close": main_roi(g, "close")}
                              for k, g in x.groupby("np_bin", observed=True)}
    return out


def run(x_cur: pd.DataFrame, x_hist: pd.DataFrame) -> dict:
    xc, xh = add_bins(x_cur), add_bins(x_hist)
    res = {"current": by_dimension(xc, True), "history": by_dimension(xh, False),
           "price_availability": price_availability(xc),
           "counts": {"current_matches": int(len(xc)), "history_matches": int(len(xh)),
                      "history_seasons": sorted(int(s) for s in pd.to_datetime(xh["date"]).dt.year.unique())},
           "spearman": {"dq_vs_excess_close": float(xc["dq_missing"].corr(xc["excess_close"], method="spearman")),
                        "dq_vs_excess_fd_hist": float(xh["dq_missing"].corr(xh["excess_fd"], method="spearman")),
                        "nmin_vs_excess_close": float(xc["n_min"].corr(xc["excess_close"], method="spearman")),
                        "nmin_vs_excess_fd_hist": float(xh["n_min"].corr(xh["excess_fd"], method="spearman"))}}
    return res
