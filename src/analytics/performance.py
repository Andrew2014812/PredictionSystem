"""Profit, ROI and hit-rate statistics of settled predictions.

Every prediction with a REAL bookmaker price is one bet with a fixed stake of
1 unit:
    win  -> profit = odds - 1
    loss -> profit = -1
    void -> profit = 0, stake returned (not part of the staked total)
    push -> Asian handicap refund: profit = 0, stake returned (not staked either)

    ROI = net profit / total amount staked × 100
        = sum(profit) / number of won or lost bets with real odds × 100

The hit rate counts every settled prediction (won / lost), with or without
a price; profit and ROI use only predictions with real odds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}     # weeks run Monday–Sunday
MIN_MARKET_SAMPLE = 15           # real-odds bets before a market can be "highest" / "lowest" ROI (display only)


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
        "push": int((picks["status"] == "PUSH").sum()) if len(picks) else 0,
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
    """Highest- and lowest-ROI group by real-odds bets.

    Groups with at least ``min_sample`` bets are compared first; only when none
    reaches it are smaller groups used, and the result is flagged
    ``small_sample`` so the interface can say so instead of hiding it."""
    g = group_summary(picks, by)
    if g.empty:
        return None, None
    g = g.loc[g["bets"] > 0]
    if g.empty:
        return None, None
    big = g.loc[g["bets"] >= min_sample]
    pool = big if len(big) >= 2 else g          # fewer than two solid groups: compare all, flag small ones
    pool = pool.sort_values("roi").assign(small_sample=lambda d: d["bets"] < min_sample)
    return pool.iloc[-1].to_dict(), (pool.iloc[0].to_dict() if len(pool) > 1 else None)


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


def cumulative_by_date(picks: pd.DataFrame, window: int | None = 50) -> pd.DataFrame:
    """Bet-by-bet cumulative profit plus ROI and hit rate over the last ``window`` bets — or, with
    ``window=None``, over all bets so far (a cumulative figure, not a rolling one)."""
    bets = staked(picks).sort_values(["date", "match_id"])
    if bets.empty:
        return bets
    out = bets[["date", "profit", "status"]].copy()
    out["cumulative_profit"] = out["profit"].cumsum()
    won = (out["status"] == "WON").astype(float)
    if window:
        first = min(window, 10)
        out["rolling_hit_rate"] = won.rolling(window, min_periods=first).mean()
        out["rolling_roi"] = out["profit"].rolling(window, min_periods=first).mean() * 100
    else:
        out["rolling_hit_rate"] = won.expanding().mean()
        out["rolling_roi"] = out["profit"].expanding().mean() * 100
    return out


DATE_PRESETS = ("Last 7 days", "Last 30 days", "Current month", "Previous month", "Current season", "All time",
                "Specific month", "Custom dates")


def preset_range(preset: str, today, first_date, custom: tuple | None = None, season_start=None):
    """(start, end) dates of a date-range preset (inclusive).

    ``custom`` is (start, end) for "Custom dates" and (year, month) for "Specific month"."""
    today = pd.Timestamp(today).normalize()
    first = pd.Timestamp(first_date).normalize()
    if preset == "Last 7 days":
        return today - pd.Timedelta(days=6), today
    if preset == "Last 30 days":
        return today - pd.Timedelta(days=29), today
    if preset == "Current month":
        return today.replace(day=1), today
    if preset == "Previous month":
        end = today.replace(day=1) - pd.Timedelta(days=1)
        return end.replace(day=1), end
    if preset == "Current season":
        return pd.Timestamp(season_start) if season_start is not None else first, today
    if preset == "Specific month" and custom and len(custom) == 2:
        start = pd.Timestamp(year=int(custom[0]), month=int(custom[1]), day=1)
        return start, start + pd.offsets.MonthEnd(0)
    if preset == "Custom dates" and custom and len(custom) == 2:
        return pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
    return first, max(today, first)


# Which stored predictions belong to which view:
#   recorded   — saved BEFORE kick-off by the production model ("live")
#   evaluation — after-the-fact model evaluation: the test-season backtest and the
#                current-season backfill (features are still pre-match, but nothing
#                was published before the match)
DATASETS = {"recorded": ("live",), "evaluation": ("backtest", "backfill"),
            "season_backtest": ("backfill",)}
# How each origin is called in the interface.
ORIGIN_LABELS = {"live": "Recorded", "backfill": "Backtest", "backtest": "Backtest"}


def period_label(period_end: pd.Timestamp, period: str) -> str:
    """Readable label of a timeline row: a day, "12.10 – 18.10.2026" for a week, "October 2026"."""
    ts = pd.Timestamp(period_end)
    if period == "Weekly":
        start = ts - pd.Timedelta(days=6)
        return f"{start:%d.%m} – {ts:%d.%m.%Y}"
    if period == "Monthly":
        return ts.strftime("%B %Y")
    return ts.strftime("%d.%m.%Y")


def select_dataset(picks: pd.DataFrame, key: str) -> pd.DataFrame:
    if picks.empty or "source" not in picks:
        return picks
    if key == "history":
        return unified_history(picks)
    return picks.loc[picks["source"].isin(DATASETS[key])]


RESCHEDULE_DAYS = 60          # the same league pairing within this many days is the same (moved) match


def rescheduled_duplicates(picks: pd.DataFrame) -> pd.DataFrame:
    """Match ids that are a second prediction of a match already predicted under another date.

    A fixture that moves (a postponement, or a provisional date fixed later) can be predicted twice: once
    under the first date — that prediction is later settled with the moved match's result — and again
    under the new date. One real match must count once, so per league pairing within RESCHEDULE_DAYS the
    prediction made for the date the match was really played is kept — the one the site showed on match
    day, made from the newest pre-match data — and a prediction for a provisional or postponed date is
    superseded (bookmakers void such bets too). A recorded prediction wins over a reconstruction, then a
    prediction whose date got a result over one whose date never did (a match moved to an earlier day),
    then the latest date, then the latest ``generated_at``. Whether a prediction won is never looked at;
    two settled predictions with different scores are different matches and are both kept. Returns the
    dropped ids with the id kept instead."""
    cols = ["match_id", "league", "home_team", "away_team", "date", "source", "generated_at",
            "status", "home_goals", "away_goals"]
    if picks.empty or not set(cols) <= set(picks.columns):
        return pd.DataFrame(columns=["match_id", "kept_match_id"])
    main = (picks.loc[picks["role"] == "main", cols].drop_duplicates("match_id")
            .assign(date=lambda d: pd.to_datetime(d["date"]),
                    order=lambda d: (d["source"] != "live").astype(int)))
    dropped = []
    for _, g in main.groupby(["league", "home_team", "away_team"]):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        clusters, current = [], [g.iloc[0]]
        for _, row in g.iloc[1:].iterrows():
            if (row["date"] - current[0]["date"]).days <= RESCHEDULE_DAYS:
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
        clusters.append(current)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            frame = pd.DataFrame(cluster)
            settled = frame.loc[frame["status"].isin(["WON", "LOST", "PUSH"])]
            if settled[["home_goals", "away_goals"]].drop_duplicates().shape[0] > 1:
                continue                                          # different results: different matches
            # a date that got a result beats a date that never did (a match moved to an EARLIER day leaves
            # the later provisional date without a result); whether the pick won is never looked at
            frame["no_result"] = (~frame["status"].isin(["WON", "LOST", "PUSH"])).astype(int)
            keep = frame.sort_values(["order", "no_result", "date", "generated_at"],
                                     ascending=[True, True, False, False]).iloc[0]["match_id"]
            dropped += [(mid, keep) for mid in frame["match_id"] if mid != keep]
    return pd.DataFrame(dropped, columns=["match_id", "kept_match_id"])


def same_day_duplicates(picks: pd.DataFrame) -> pd.DataFrame:
    """Match ids that book a club twice in one league on one day — the same match under another spelling
    of a club (e.g. Football-Data's fixtures file once wrote "KV Mechelen" for "Mechelen"), since a club
    never plays two league matches on one day. Kept: a settled prediction, then the one whose clubs both
    have a settled record in the league (the real clubs), then the earliest ``generated_at``."""
    cols = ["match_id", "league", "home_team", "away_team", "date", "generated_at", "status"]
    if picks.empty or not set(cols) <= set(picks.columns):
        return pd.DataFrame(columns=["match_id", "kept_match_id"])
    main = picks.loc[picks["role"] == "main", cols].drop_duplicates("match_id").copy()
    main["day"] = pd.to_datetime(main["date"]).dt.normalize()
    done = main.loc[main["status"].isin(["WON", "LOST", "PUSH"])]
    known = {lg: set(g["home_team"]) | set(g["away_team"]) for lg, g in done.groupby("league")}
    main["settled_first"] = (~main["status"].isin(["WON", "LOST", "PUSH"])).astype(int)
    main["unknown_club"] = [int(not ({h, a} <= known.get(lg, set())))
                            for lg, h, a in zip(main["league"], main["home_team"], main["away_team"])]
    dropped = []
    for _, g in main.groupby(["league", "day"]):
        if len(g) < 2:
            continue
        teams = pd.concat([g[["match_id", "home_team"]].rename(columns={"home_team": "team"}),
                           g[["match_id", "away_team"]].rename(columns={"away_team": "team"})])
        twice = teams.groupby("team")["match_id"].nunique()
        for team in twice[twice > 1].index:
            clash = g.loc[g["match_id"].isin(teams.loc[teams["team"] == team, "match_id"])]
            keep = clash.sort_values(["settled_first", "unknown_club", "generated_at"]).iloc[0]["match_id"]
            dropped += [(mid, keep) for mid in clash["match_id"] if mid != keep]
    return pd.DataFrame(sorted(set(dropped)), columns=["match_id", "kept_match_id"])


def unified_history(picks: pd.DataFrame) -> pd.DataFrame:
    """The one prediction history the website shows: per match, the predictions recorded before kick-off
    when they exist, otherwise the current-season reconstruction from pre-match data (never both, so no
    bet is counted twice), and one prediction per real match when a fixture moved to another date
    (``rescheduled_duplicates``) or a club was spelt two ways (``same_day_duplicates``). The origin stays in the data (``source``, ``origin``, ``odds_origin``)."""
    if picks.empty or "source" not in picks:
        return picks
    live = picks.loc[picks["source"] == "live"]
    rebuilt = picks.loc[(picks["source"] == "backfill") & ~picks["match_id"].isin(set(live["match_id"]))]
    history = pd.concat([live, rebuilt], ignore_index=True)
    moved = rescheduled_duplicates(history)
    history = history.loc[~history["match_id"].isin(set(moved["match_id"]))]
    spelt = same_day_duplicates(history)
    return history.loc[~history["match_id"].isin(set(spelt["match_id"]))].reset_index(drop=True)
