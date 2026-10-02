"""Profit, ROI and hit-rate statistics of settled predictions.

Every prediction is one bet with a unit stake:
    win  -> profit = odds - 1
    loss -> profit = -1
    void -> profit = 0 (stake returned, not counted in the hit rate)
    ROI  = total profit / number of staked bets * 100
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS", "Yearly": "YS"}


def settled(picks: pd.DataFrame) -> pd.DataFrame:
    """Decided bets (WON / LOST) that have odds."""
    if picks.empty:
        return picks
    return picks.loc[picks["status"].isin(["WON", "LOST"]) & picks["profit"].notna()]


def _streaks(statuses: list[str]) -> tuple[int, int]:
    best_win = best_loss = run_win = run_loss = 0
    for s in statuses:
        if s == "WON":
            run_win, run_loss = run_win + 1, 0
        else:
            run_win, run_loss = 0, run_loss + 1
        best_win, best_loss = max(best_win, run_win), max(best_loss, run_loss)
    return best_win, best_loss


def summarize(picks: pd.DataFrame) -> dict:
    done = settled(picks)
    n = len(done)
    won = int((done["status"] == "WON").sum()) if n else 0
    profit = float(done["profit"].sum()) if n else 0.0
    ordered = done.sort_values(["date", "match_id"])["status"].tolist() if n else []
    win_streak, loss_streak = _streaks(ordered)
    return {
        "predictions": int(len(picks)),
        "settled": n,
        "won": won,
        "lost": n - won,
        "pending": int((picks["status"] == "UPCOMING").sum()) if len(picks) else 0,
        "void": int((picks["status"] == "VOID").sum()) if len(picks) else 0,
        "hit_rate": won / n if n else np.nan,
        "profit": profit,
        "roi": profit / n * 100 if n else np.nan,
        "avg_odds": float(done["odds"].mean()) if n else np.nan,
        "avg_probability": float(done["probability"].mean()) if n else np.nan,
        "longest_win_streak": win_streak,
        "longest_loss_streak": loss_streak,
    }


def group_summary(picks: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    done = settled(picks)
    if done.empty:
        return pd.DataFrame()
    grouped = done.groupby(by, observed=True).agg(
        predictions=("profit", "size"),
        won=("status", lambda s: int((s == "WON").sum())),
        profit=("profit", "sum"),
        avg_odds=("odds", "mean"),
    )
    grouped["lost"] = grouped["predictions"] - grouped["won"]
    grouped["hit_rate"] = grouped["won"] / grouped["predictions"]
    grouped["roi"] = grouped["profit"] / grouped["predictions"] * 100
    return grouped.reset_index().sort_values("predictions", ascending=False)


def timeline(picks: pd.DataFrame, period: str = "Weekly") -> pd.DataFrame:
    """Per-period volume, hit rate, profit and ROI plus cumulative profit."""
    done = settled(picks)
    if done.empty:
        return pd.DataFrame()
    freq = PERIODS[period]
    grouped = done.set_index("date").groupby(pd.Grouper(freq=freq)).agg(
        predictions=("profit", "size"),
        won=("status", lambda s: int((s == "WON").sum())),
        profit=("profit", "sum"),
        avg_odds=("odds", "mean"),
    )
    grouped = grouped.loc[grouped["predictions"] > 0]
    grouped["hit_rate"] = grouped["won"] / grouped["predictions"]
    grouped["roi"] = grouped["profit"] / grouped["predictions"] * 100
    grouped["cumulative_profit"] = grouped["profit"].cumsum()
    grouped["cumulative_roi"] = grouped["cumulative_profit"] / grouped["predictions"].cumsum() * 100
    return grouped.reset_index().rename(columns={"date": "period"})


def cumulative_by_bet(picks: pd.DataFrame) -> pd.DataFrame:
    done = settled(picks).sort_values(["date", "match_id"])
    if done.empty:
        return done
    out = done[["date", "profit", "status"]].copy()
    out["bet"] = np.arange(1, len(out) + 1)
    out["cumulative_profit"] = out["profit"].cumsum()
    out["rolling_hit_rate"] = (out["status"] == "WON").rolling(200, min_periods=20).mean()
    out["rolling_roi"] = out["profit"].rolling(200, min_periods=20).mean() * 100
    return out
