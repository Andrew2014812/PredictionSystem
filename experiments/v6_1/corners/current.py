"""Current production corners model, recomputed with the stored bundles (read only).

    backtest_2526    fitted on 2021/22–2024/25, scored on 2025/26 (the metadata's "test" season)
    prod_20261012    the active model (fitted on 2021/22–2025/26), scored on 2026/27 to date

Also the production prediction history of 2026/27 (``unified_history(load_picks())``): every
corner pick that was settled at a real bookmaker price, by role.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.analytics.performance import unified_history
from src.markets.distributions import corners_distribution, prob_over
from src.modelling.registry import load_bundles, version_metadata
from src.prediction.store import load_picks

from . import data as D
from . import dist as Q
from .models import pdev


def score_block(y, mu, r, league=None) -> dict:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    pmf = Q.pmf_nb(mu, r)
    res = y - mu
    out = {"n": int(len(y)), "mae": float(np.abs(res).mean()), "rmse": float(np.sqrt(np.mean(res ** 2))),
           "r2": float(r2_score(y, mu)), "poisson_deviance": pdev(y, mu), "mean_actual": float(y.mean()),
           "mean_predicted": float(mu.mean()), "log_score": float(Q.log_score(pmf, y).mean()),
           "crps": float(Q.crps(pmf, y).mean()), "pit": Q.pit_summary(Q.pit(pmf, y)), "nb_size": r, "lines": {}}
    for line in D.LINES:
        p, o = Q.p_over(pmf, line), y > line
        out["lines"][str(line)] = {"log_loss": float(Q.binary_ll(o, p).mean()), "brier": float(np.mean((p - o) ** 2)),
                                   "ece": Q.ece(o, p), "observed": float(o.mean()), "mean_p": float(p.mean()),
                                   "accuracy": float(((p > 0.5) == o).mean())}
    return out


def production_check(mu, r) -> float:
    """Max |difference| between this study's line probabilities and src.markets (first 300 matches)."""
    pmf = Q.pmf_nb(mu[:300], r)
    diff = 0.0
    for i, m in enumerate(mu[:300]):
        ref = corners_distribution(float(m), r)
        for line in D.LINES:
            diff = max(diff, abs(prob_over(ref, line) - float(Q.p_over(pmf[[i]], line)[0])))
    return diff


def history() -> dict:
    picks = unified_history(load_picks())
    c = picks.loc[(picks["group"] == "CORNERS") & (picks["season"] == D.CHECK)].copy()
    settled = c.loc[c["status"].isin(["WON", "LOST"])]
    out = {"rows": int(len(c)), "settled": int(len(settled)), "by_source": c.groupby("source").size().to_dict(),
           "versions": c["model_version"].value_counts().to_dict(), "roles": {}}
    priced = settled.loc[settled["real_odds"].fillna(False).astype(bool) & settled["odds"].notna()]
    rng = np.random.default_rng(7)
    for role, g in [("all", priced)] + list(priced.groupby("role")):
        prof = g["profit"].to_numpy(float)
        boots = [prof[rng.integers(0, len(prof), len(prof))].mean() for _ in range(2000)] if len(prof) else [np.nan]
        out["roles"][role] = {"n": int(len(g)), "hit_rate": float((g["status"] == "WON").mean()) if len(g) else None,
                              "mean_odds": float(g["odds"].mean()) if len(g) else None,
                              "mean_probability": float(g["probability"].mean()) if len(g) else None,
                              "roi": float(prof.mean()) if len(g) else None,
                              "roi_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                              "profit": float(prof.sum()), "by_market": g.groupby("market").agg(
                                  n=("profit", "size"), roi=("profit", "mean")).to_dict(orient="index")}
    mk = priced.loc[priced["role"] == "market"]
    out["market_role_calibration"] = Q.reliability((mk["status"] == "WON").to_numpy(), mk["probability"].to_numpy(),
                                                   bins=10, min_n=15)
    out["unpriced_settled"] = int(len(settled) - len(priced))
    return out


def run() -> dict:
    df = D.with_corners(D.frame())
    out = {}
    for version, season in (("backtest_2526", 2025), ("prod_20261012", 2026)):
        b = load_bundles(version)["corners"]
        meta = version_metadata(version)["tasks"]["corners"]
        rows = df.loc[(df["season"] == season) & df["league"].isin(b.extra["leagues"])]
        mu = b.predict(rows).astype(float)
        r = b.extra.get("nb_size")
        out[version] = {"season": season, "metadata_test_metrics": meta["test_metrics"],
                        "params": meta["params"], "n_estimators": meta["n_estimators"], "n_features": meta["n_features"],
                        "nb_size": r, "metrics": score_block(rows["total_corners"], mu, r),
                        "check_vs_src_markets_max_abs": production_check(mu, r)}
        pd.DataFrame({"match_id": rows["match_id"].to_numpy(), "mu": mu}).to_parquet(D.CACHE / f"bundle_{version}.parquet")
    out["history_2026"] = history()
    D.save_result("current.json", out)
    return out
