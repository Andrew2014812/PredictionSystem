"""Metrics of one strategy on one season, with a match-level bootstrap shared by all strategies.

A bet = a settled (WON / LOST) pick with a real price, stake 1: profit odds - 1 or -1; a refund
(PUSH) and a pick without a price are not bets (as src/analytics/performance.py). The bootstrap
resamples MATCHES with the same multinomial weights for every strategy of a season, so differences
between strategies are paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

B = 2000
ODDS_BANDS = [(1.0, 2.5, "<2.50"), (2.5, 3.0, "2.50–3.00"), (3.0, 3.5, "3.00–3.50"), (3.5, 5.0, "3.50–5.00"),
              (5.0, 6.0, "5.00–6.00"), (6.0, np.inf, ">6.00")]


class Season:
    """The matches of one season (chronological) and the bootstrap weights."""

    def __init__(self, df: pd.DataFrame, seed: int):
        m = df.drop_duplicates("match_id")[["match_id", "date"]].sort_values(["date", "match_id"])
        self.ids = m["match_id"].to_numpy()
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(len(self.ids), np.full(len(self.ids), 1 / len(self.ids)), size=B).astype(np.float32)
        self.with_main = int(df.loc[df["is_main"], "match_id"].nunique())

    def vectors(self, picks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prof = np.zeros(len(self.ids), np.float32)
        stake = np.zeros(len(self.ids), np.float32)
        bets = picks.loc[picks["decided"] & picks["priced"]]
        i = self.pos.loc[bets["match_id"]].to_numpy()
        prof[i] = bets["profit"].to_numpy()
        stake[i] = 1.0
        return prof, stake

    def boot_roi(self, prof, stake) -> np.ndarray:
        den = self.w @ stake
        return np.where(den > 0, (self.w @ prof) / np.maximum(den, 1), np.nan)


def max_drawdown(profits: np.ndarray) -> float:
    if not len(profits):
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(profits)])
    return float(np.max(np.maximum.accumulate(cum) - cum))


def longest_losing(won: np.ndarray) -> int:
    best = run = 0
    for w in won:
        run = 0 if w else run + 1
        best = max(best, run)
    return best


def metrics(picks: pd.DataFrame, season: Season, base_boot: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    settled = picks.loc[picks["decided"]]
    bets = settled.loc[settled["priced"]].sort_values(["date", "match_id"])
    prof, stake = season.vectors(picks)
    boot = season.boot_roi(prof, stake)
    n_b = len(bets)
    out = {
        "matches": int(len(season.ids)), "matches_with_main": season.with_main,
        "picks": int(len(picks)), "coverage": len(picks) / len(season.ids),
        "predictions": int(len(settled)), "bets": int(n_b),
        "priced_share": n_b / len(settled) if len(settled) else None,
        "hit_rate": float(settled["won"].mean()) if len(settled) else None,
        "hit_rate_bets": float(bets["won"].mean()) if n_b else None,
        "avg_odds": float(bets["odds"].mean()) if n_b else None,
        "median_odds": float(bets["odds"].median()) if n_b else None,
        "avg_p_bets": float(bets["p"].mean()) if n_b else None,
        "profit": float(bets["profit"].sum()), "roi": float(bets["profit"].mean()) if n_b else None,
        "roi_ci95": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))] if n_b else None,
        "avg_ev": float(bets["ev"].mean()) if n_b else None,
        "ev_minus_roi": float(bets["ev"].mean() - bets["profit"].mean()) if n_b else None,
        "calibration_gap": float(bets["won"].mean() - bets["p"].mean()) if n_b else None,
        "calibration_z": float((bets["won"].sum() - bets["p"].sum()) / np.sqrt((bets["p"] * (1 - bets["p"])).sum()))
        if n_b else None,
        "profit_sd": float(bets["profit"].std()) if n_b > 1 else None,
        "max_drawdown": max_drawdown(bets["profit"].to_numpy()),
        "longest_losing_streak": longest_losing(bets["won"].to_numpy() > 0),
        "tier_shares": {str(k): float(v) for k, v in picks["tier"].value_counts(normalize=True).items()}
        if "tier" in picks else {},
        "group_shares": {k: float(v) for k, v in bets["group"].value_counts(normalize=True).items()},
        "odds_bands": band_breakdown(bets),
    }
    if base_boot is not None and n_b:
        d = boot - base_boot
        out["diff_vs_current_ci95"] = [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]
        out["p_better_than_current"] = float(np.nanmean(d > 0))
    return out, boot


def band_breakdown(bets: pd.DataFrame) -> dict:
    out = {}
    for lo, hi, label in ODDS_BANDS:
        b = bets.loc[(bets["odds"] >= lo) & (bets["odds"] < hi)]
        if len(b):
            out[label] = {"bets": int(len(b)), "roi": float(b["profit"].mean()), "hit": float(b["won"].mean()),
                          "avg_p": float(b["p"].mean()), "avg_ev": float(b["ev"].mean()),
                          "avg_odds": float(b["odds"].mean())}
    return out


def pooled(results: dict, seasons: list[str], name: str, boots: dict) -> dict:
    """ROI over several seasons (stake-weighted) with a stratified bootstrap (sum of season replicates)."""
    rows = [results[s][name] for s in seasons]
    bets = sum(r["bets"] for r in rows)
    profit = sum(r["profit"] for r in rows)
    out = {"bets": bets, "profit": profit, "roi": profit / bets if bets else None,
           "worst_roi": min(r["roi"] for r in rows if r["roi"] is not None),
           "min_avg_odds": min(r["avg_odds"] for r in rows if r["avg_odds"] is not None),
           "min_priced_share": min(r["priced_share"] for r in rows if r["priced_share"] is not None),
           "min_coverage_of_main": min(r["picks"] / max(r["matches_with_main"], 1) for r in rows)}
    if all((s, name) in boots for s in seasons):
        num = sum(boots[(s, name)][1] for s in seasons)
        den = sum(boots[(s, name)][2] for s in seasons)
        roi = num / np.maximum(den, 1)
        out["roi_ci95"] = [float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975))]
        if all((s, "Текущий Risk") in boots for s in seasons):
            cnum = sum(boots[(s, "Текущий Risk")][1] for s in seasons)
            cden = sum(boots[(s, "Текущий Risk")][2] for s in seasons)
            d = roi - cnum / np.maximum(cden, 1)
            out["diff_vs_current_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            out["p_better_than_current"] = float(np.mean(d > 0))
    return out
