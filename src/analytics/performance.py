"""Profit, ROI and hit-rate statistics of settled predictions.

Every prediction with a REAL bookmaker price is one bet with a fixed stake of
1 unit:
    win  -> profit = odds - 1
    loss -> profit = -1
    void -> profit = 0, stake returned (not part of the staked total)

    ROI = net profit / total amount staked × 100
        = sum(profit) / number of won or lost bets with real odds × 100

The hit rate counts every settled prediction (won / lost), with or without
a price; profit and ROI use only predictions with real odds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS", "Yearly": "YS"}
MIN_MARKET_SAMPLE = 100          # settled bets needed before a market can be "best" / "weakest"


def settled(picks: pd.DataFrame) -> pd.DataFrame:
    """Decided predictions (WON / LOST), with or without odds."""
    if picks.empty:
        return picks
    return picks.loc[picks["status"].isin(["WON", "LOST"])]


def staked(picks: pd.DataFrame) -> pd.DataFrame:
    """Decided predictions that count as bets: real odds and a profit."""
    done = settled(picks)
    if done.empty:
        return done
    return done.loc[done["odds"].notna() & done["profit"].notna()]


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
    done, bets = settled(picks), staked(picks)
    n, n_bets = len(done), len(bets)
    won = int((done["status"] == "WON").sum()) if n else 0
    profit = float(bets["profit"].sum()) if n_bets else 0.0
    ordered = bets.sort_values(["date", "match_id"])["status"].tolist() if n_bets else []
    win_streak, loss_streak = _streaks(ordered)
    return {
        "predictions": int(len(picks)),
        "settled": n,
        "won": won,
        "lost": n - won,
        "bets": n_bets,
        "pending": int((picks["status"] == "UPCOMING").sum()) if len(picks) else 0,
        "void": int((picks["status"] == "VOID").sum()) if len(picks) else 0,
        "hit_rate": won / n if n else np.nan,
        "profit": profit if n_bets else np.nan,
        "roi": profit / n_bets * 100 if n_bets else np.nan,
        "avg_odds": float(bets["odds"].mean()) if n_bets else np.nan,
        "avg_probability": float(done["probability"].mean()) if n else np.nan,
        "longest_win_streak": win_streak,
        "longest_loss_streak": loss_streak,
    }


def group_summary(picks: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    done = settled(picks)
    if done.empty:
        return pd.DataFrame()
    done = done.assign(_bet=done["odds"].notna() & done["profit"].notna(),
                       _won=done["status"] == "WON")
    grouped = done.groupby(by, observed=True).agg(
        predictions=("_won", "size"),
        won=("_won", "sum"),
        bets=("_bet", "sum"),
        profit=("profit", lambda s: s.sum(min_count=1)),
        avg_odds=("odds", "mean"),
    )
    grouped["lost"] = grouped["predictions"] - grouped["won"]
    grouped["hit_rate"] = grouped["won"] / grouped["predictions"]
    grouped["roi"] = np.where(grouped["bets"] > 0, grouped["profit"] / grouped["bets"].clip(lower=1) * 100, np.nan)
    return grouped.reset_index().sort_values("predictions", ascending=False)


def best_and_worst(picks: pd.DataFrame, by: str = "group",
                   min_sample: int = MIN_MARKET_SAMPLE) -> tuple[dict | None, dict | None]:
    """Best and weakest group by ROI among groups with enough real-odds bets."""
    g = group_summary(picks, by)
    if g.empty:
        return None, None
    g = g.loc[g["bets"] >= min_sample]
    if g.empty:
        return None, None
    g = g.sort_values("roi")
    return g.iloc[-1].to_dict(), (g.iloc[0].to_dict() if len(g) > 1 else None)


def timeline(picks: pd.DataFrame, period: str = "Weekly") -> pd.DataFrame:
    """Per-period volume, hit rate, profit and ROI plus cumulative profit."""
    done = settled(picks)
    if done.empty:
        return pd.DataFrame()
    done = done.assign(_bet=done["odds"].notna() & done["profit"].notna(), _won=done["status"] == "WON")
    grouped = done.set_index("date").groupby(pd.Grouper(freq=PERIODS[period])).agg(
        predictions=("_won", "size"),
        won=("_won", "sum"),
        bets=("_bet", "sum"),
        profit=("profit", lambda s: s.sum(min_count=1)),
        avg_odds=("odds", "mean"),
    )
    grouped = grouped.loc[grouped["predictions"] > 0]
    grouped["hit_rate"] = grouped["won"] / grouped["predictions"]
    grouped["roi"] = np.where(grouped["bets"] > 0, grouped["profit"] / grouped["bets"].clip(lower=1) * 100, np.nan)
    grouped["cumulative_profit"] = grouped["profit"].fillna(0).cumsum()
    grouped["cumulative_roi"] = (grouped["cumulative_profit"]
                                 / grouped["bets"].cumsum().replace(0, np.nan) * 100)
    return grouped.reset_index().rename(columns={"date": "period"})


def cumulative_by_date(picks: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Bet-by-bet cumulative profit, rolling ROI and hit rate, indexed by match date."""
    bets = staked(picks).sort_values(["date", "match_id"])
    if bets.empty:
        return bets
    out = bets[["date", "profit", "status"]].copy()
    out["cumulative_profit"] = out["profit"].cumsum()
    out["rolling_hit_rate"] = (out["status"] == "WON").rolling(window, min_periods=20).mean()
    out["rolling_roi"] = out["profit"].rolling(window, min_periods=20).mean() * 100
    return out


DATE_PRESETS = ("Last 7 days", "Last 30 days", "This month", "This season", "This year", "All time", "Custom")


def preset_range(preset: str, today, first_date, custom: tuple | None = None, season_start=None):
    """(start, end) dates of a date-range preset (inclusive)."""
    today = pd.Timestamp(today).normalize()
    first = pd.Timestamp(first_date).normalize()
    if preset == "Last 7 days":
        return today - pd.Timedelta(days=6), today
    if preset == "Last 30 days":
        return today - pd.Timedelta(days=29), today
    if preset == "This month":
        return today.replace(day=1), today
    if preset == "This season":
        return pd.Timestamp(season_start) if season_start is not None else first, today
    if preset == "This year":
        return today.replace(month=1, day=1), today
    if preset == "Custom" and custom and len(custom) == 2:
        return pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
    return first, max(today, first)
