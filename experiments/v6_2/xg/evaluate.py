"""Scoring every variant on the SAME matches (step 5-6 of the study).

Evaluation set: played 2026/27 matches of the 18 xG leagues from EVAL_START on, for which every variant has a
prediction. Per match the code keeps the loss of every metric so that differences against B0 are bootstrapped
by match (paired), corrected with Holm across the variants, and turned into a sample-size estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config
from src.markets.distributions import corners_distribution, reconcile_with_1x2, score_matrix
from src.markets.markets import corner_markets, goal_markets, settle_selection
from src.markets.odds import price
from src.markets.recommendation import main_prediction
from src.modelling.decision import decide

from . import common as C
from . import models as M

TOTAL_LINES = (1.5, 2.5, 3.5)
AH_LINES = (-2.5, -1.5, 1.5, 2.5)          # half lines (no refund), home perspective
PRIMARY = ("goals_dev", "logloss_1x2", "roi_main")


# ------------------------------------------------------------------------------------------------ matrices
def matrices(lh, la, p) -> np.ndarray:
    """Production score matrices: Poisson(lh) x Poisson(la) rescaled to the 1X2 probabilities."""
    return np.stack([reconcile_with_1x2(score_matrix(a, b), q) for a, b, q in zip(lh, la, p)])


def per_match(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-match losses of one variant (pred and truth aligned by row)."""
    hg, ag = truth["home_goals"].to_numpy(float), truth["away_goals"].to_numpy(float)
    y = truth["y"].to_numpy(int)
    lh, la = pred["lh"].to_numpy(float), pred["la"].to_numpy(float)
    p = pred[["pH", "pD", "pA"]].to_numpy(float)
    p = p / p.sum(1, keepdims=True)
    out = pd.DataFrame({"match_id": truth["match_id"].to_numpy()})
    out["dev_h"], out["dev_a"] = C.poisson_dev(hg, lh), C.poisson_dev(ag, la)
    out["goals_dev"] = out["dev_h"] + out["dev_a"]
    out["se_h"], out["se_a"] = (hg - lh) ** 2, (ag - la) ** 2
    out["ae_h"], out["ae_a"] = np.abs(hg - lh), np.abs(ag - la)
    out["se_total"] = (hg + ag - lh - la) ** 2
    out["logloss_1x2"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    out["brier_1x2"] = ((p - np.eye(3)[y]) ** 2).sum(1)
    rule = decide(p, 1.0, config.DRAW_THRESHOLD, config.DRAW_MAX_GAP)
    out["pred_rule"], out["pred_argmax"], out["y"] = rule, p.argmax(1), y
    out["acc_rule"], out["acc_argmax"] = (rule == y).astype(float), (p.argmax(1) == y).astype(float)
    for k, c in enumerate("HDA"):
        out[f"p{c}"] = p[:, k]
    mats = matrices(lh, la, p)
    n = mats.shape[1]
    i, j = np.indices((n, n))
    tot = i + j
    for line in TOTAL_LINES:
        pr = mats[:, tot > line].sum(1)
        o = (hg + ag > line).astype(float)
        out[f"ll_over{line}"], out[f"br_over{line}"] = C.binary_loss(o, pr)
        out[f"p_over{line}"], out[f"o_over{line}"] = pr, o
    pr = mats[:, (i > 0) & (j > 0)].sum(1)
    o = ((hg > 0) & (ag > 0)).astype(float)
    out["ll_btts"], out["br_btts"] = C.binary_loss(o, pr)
    out["p_btts"], out["o_btts"] = pr, o
    for line in AH_LINES:
        pr = mats[:, (i + line - j) > 0].sum(1)
        o = ((hg + line - ag) > 0).astype(float)
        out[f"ll_ah{line:+g}"], out[f"br_ah{line:+g}"] = C.binary_loss(o, pr)
    hi, ai = np.clip(hg.astype(int), 0, n - 1), np.clip(ag.astype(int), 0, n - 1)
    cell = mats[np.arange(len(hg)), hi, ai]
    out["ll_exact"] = -np.log(np.clip(cell, 1e-12, 1))
    flat = mats.reshape(len(hg), -1)
    top5 = np.argsort(flat, 1)[:, ::-1][:, :5]
    out["exact_top5"] = (top5 == (hi * n + ai)[:, None]).any(1).astype(float)
    out["exact_top1"] = (top5[:, 0] == hi * n + ai).astype(float)
    return out


# ------------------------------------------------------------------------------------------------ Main
def main_picks(pred: pd.DataFrame, truth: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Production Main rule on the variant's probabilities and the real pre-match prices."""
    rows = []
    lh, la = pred["lh"].to_numpy(float), pred["la"].to_numpy(float)
    p = pred[["pH", "pD", "pA"]].to_numpy(float)
    p = p / p.sum(1, keepdims=True)
    for k, t in enumerate(truth.itertuples(index=False)):
        mat = reconcile_with_1x2(score_matrix(lh[k], la[k]), p[k])
        sels = goal_markets(mat)
        if np.isfinite(t.cmu):
            sels += corner_markets(corners_distribution(t.cmu, t.nb_size))
        priced = price(sels, lookup.get(t.match_id))
        rec = main_prediction(priced)
        s = rec.pick
        st = settle_selection(s.market, s.selection, t.home_goals, t.away_goals, t.total_corners)
        priced_ok = bool(s.has_odds)
        decided = st in ("WON", "LOST")
        rows.append({"match_id": t.match_id, "market": s.market, "selection": s.selection, "group": s.group,
                     "p": s.probability, "odds": s.odds if priced_ok else np.nan, "status": st,
                     "priced": priced_ok, "decided": decided, "won": float(st == "WON") if decided else np.nan,
                     "profit": ((s.odds - 1) if st == "WON" else (-1.0 if st == "LOST" else 0.0)) if priced_ok and st else np.nan,
                     "ev": s.ev if priced_ok else np.nan})
    return pd.DataFrame(rows)


def main_summary(mp: pd.DataFrame) -> dict:
    bet = mp["priced"] & mp["decided"]
    b = mp.loc[bet]
    dec = mp.loc[mp["decided"]]
    return {"matches": int(len(mp)), "decided": int(len(dec)), "bets": int(len(b)),
            "hit_rate": float(dec["won"].mean()), "mean_p": float(dec["p"].mean()),
            "gap": float(dec["p"].mean() - dec["won"].mean()),
            "gap_ci95": C.ratio_ci((mp["p"] - mp["won"]).where(mp["decided"], 0.0), mp["decided"].astype(float)),
            "bet_hit_rate": float(b["won"].mean()), "bet_mean_p": float(b["p"].mean()),
            "bet_gap": float(b["p"].mean() - b["won"].mean()),
            "bet_gap_ci95": C.ratio_ci((mp["p"] - mp["won"]).where(bet, 0.0), bet.astype(float)),
            "avg_odds": float(b["odds"].mean()), "breakeven": float(np.mean(1 / b["odds"])),
            "stated_ev": float(b["ev"].mean()), "roi": float(b["profit"].mean()),
            "roi_ci95": C.ratio_ci(mp["profit"].where(bet, 0.0), bet.astype(float)),
            "profit": float(b["profit"].sum()),
            "group_share": b["group"].value_counts(normalize=True).to_dict(),
            "ece": C.ece(dec["won"], dec["p"]), "reliability": C.reliability(dec["won"], dec["p"], 10, 20)}


# ------------------------------------------------------------------------------------------------ assembly
def assemble(df: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Common evaluation set and the aligned predictions of every variant."""
    ev = df.loc[df["played"].astype(bool) & (df["date"] >= C.EVAL_START) & df["xg_ok"]].copy()
    ids = set(ev["match_id"])
    for v, p in preds.items():
        ids &= set(p["match_id"])
    ev = ev.loc[ev["match_id"].isin(ids)].sort_values(["date", "match_id"]).reset_index(drop=True)
    aligned = {"B0": ev[["match_id", "lh", "la", "pH", "pD", "pA"]].copy()}
    for v, p in preds.items():
        aligned[v] = p.drop_duplicates("match_id").set_index("match_id").loc[ev["match_id"]].reset_index()
    return ev, aligned


METRICS = {   # name: (column(s), aggregation, lower is better)
    "goals_dev": ("goals_dev", "mean"), "dev_h": ("dev_h", "mean"), "dev_a": ("dev_a", "mean"),
    "rmse_h": ("se_h", "rmse"), "rmse_a": ("se_a", "rmse"), "mae_h": ("ae_h", "mean"), "mae_a": ("ae_a", "mean"),
    "rmse_total": ("se_total", "rmse"), "logloss_1x2": ("logloss_1x2", "mean"), "brier_1x2": ("brier_1x2", "mean"),
    "acc_rule": ("acc_rule", "mean"), "acc_argmax": ("acc_argmax", "mean"),
    "ll_over2.5": ("ll_over2.5", "mean"), "br_over2.5": ("br_over2.5", "mean"), "ll_over1.5": ("ll_over1.5", "mean"),
    "ll_over3.5": ("ll_over3.5", "mean"), "ll_btts": ("ll_btts", "mean"), "br_btts": ("br_btts", "mean"),
    "ll_ah-1.5": ("ll_ah-1.5", "mean"), "ll_ah+1.5": ("ll_ah+1.5", "mean"), "ll_ah-2.5": ("ll_ah-2.5", "mean"),
    "ll_ah+2.5": ("ll_ah+2.5", "mean"), "ll_exact": ("ll_exact", "mean"), "exact_top5": ("exact_top5", "mean"),
}
HIGHER_BETTER = {"acc_rule", "acc_argmax", "exact_top5", "macro_f1"}


def summary(pm: pd.DataFrame) -> dict:
    out = {}
    for name, (col, agg) in METRICS.items():
        v = pm[col].to_numpy(float)
        out[name] = float(np.sqrt(v.mean())) if agg == "rmse" else float(v.mean())
    out["macro_f1"] = float(f1_score(pm["y"], pm["pred_rule"], average="macro", labels=[0, 1, 2], zero_division=0))
    out["macro_f1_argmax"] = float(f1_score(pm["y"], pm["pred_argmax"], average="macro", labels=[0, 1, 2], zero_division=0))
    out["ece_1x2"] = float(np.mean([C.ece(pm["y"] == k, pm[f"p{c}"]) for k, c in enumerate("HDA")]))
    out["ece_over2.5"] = C.ece(pm["o_over2.5"], pm["p_over2.5"])
    out["ece_btts"] = C.ece(pm["o_btts"], pm["p_btts"])
    out["n"] = int(len(pm))
    return out


def paired(pm_a: pd.DataFrame, pm_b: pd.DataFrame, name: str) -> dict:
    """b - a for one metric, bootstrapped by match; RMSE via the mean squared error."""
    col, agg = METRICS[name]
    d = pm_b[col].to_numpy(float) - pm_a[col].to_numpy(float)
    r = C.boot_mean_ci(d)
    if agg == "rmse":
        a, b = np.sqrt(pm_a[col].mean()), np.sqrt(pm_b[col].mean())
        r["rmse_diff"] = float(b - a)
    r["n_required"] = C.n_required(r.get("sd"), r.get("mean"))
    return r


def f1_diff(pm_a: pd.DataFrame, pm_b: pd.DataFrame) -> dict:
    y, a, b = pm_a["y"].to_numpy(), pm_a["pred_rule"].to_numpy(), pm_b["pred_rule"].to_numpy()
    f = lambda yy, pp: f1_score(yy, pp, average="macro", labels=[0, 1, 2], zero_division=0)  # noqa: E731
    base = f(y, b) - f(y, a)
    diffs = np.concatenate([[f(y[i], b[i]) - f(y[i], a[i]) for i in idx] for idx in C.boot_index(len(y), 1000)])
    return {"diff": float(base), "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}


def main_paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """ROI and over-confidence differences of the Main (b - a), bootstrapped by match."""
    ba, bb = (a["priced"] & a["decided"]), (b["priced"] & b["decided"])
    roi = C.boot_ratio_diff(a["profit"].where(ba, 0.0), ba.astype(float), b["profit"].where(bb, 0.0), bb.astype(float))
    gap = C.boot_ratio_diff((a["p"] - a["won"]).where(a["decided"], 0.0), a["decided"].astype(float),
                            (b["p"] - b["won"]).where(b["decided"], 0.0), b["decided"].astype(float))
    hit = C.boot_ratio_diff(a["won"].where(a["decided"], 0.0), a["decided"].astype(float),
                            b["won"].where(b["decided"], 0.0), b["decided"].astype(float))
    # sample size for the ROI difference: per-match profit difference (0 when no bet)
    d = b["profit"].where(bb, 0.0).to_numpy(float) - a["profit"].where(ba, 0.0).to_numpy(float)
    return {"roi": roi, "gap": gap, "hit": hit,
            "same_main": float(((a["market"] == b["market"]) & (a["selection"] == b["selection"])).mean()),
            "n_required_roi": C.n_required(float(np.std(d, ddof=1)), roi["diff"] * bb.mean() if roi["diff"] else None)}
