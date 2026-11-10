"""API-Football: what it delivers, and how much earlier than Football-Data a result arrives.

Method. Every pipeline run commits ``local_data/`` (CI runs to ``main``, merged into this branch), so the git
history is a log of what each source had delivered by each run:

* t_AF  = time of the first commit in which ``results_live/fixtures.parquet`` holds the match as played with
          ``provider = api_football``;
* t_FD  = time of the first commit in which a Football-Data season file ``results/2627/<league>.csv`` holds the
          match with a score.

Both sources are asked in the same run (providers first, then Football-Data), so a lead of 0 means "same run".
Times are commit times (UTC); a result can only be seen at a run, so the lead is measured in run intervals.
Universe: 2026/27 matches of the 21 leagues that kicked off after the first committed API-Football row.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C
from .football_data import LEAGUES

LIVE_COLS = ["match_id", "league", "date", "kickoff", "played", "provider", "home_goals", "away_goals",
             "home_team", "away_team"]


def live_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = C.commits(C.LIVE_FILE)
    first: dict[str, dict] = {}
    seen_rows = []
    for row in cm.itertuples():
        df = C.parquet_at(row.sha, C.LIVE_FILE)
        if df is None or "provider" not in df:
            continue
        df = df.reindex(columns=LIVE_COLS)
        af = df.loc[(df["provider"] == "api_football") & df["played"].astype(bool)]
        seen_rows.append({"sha": row.sha, "at_utc": row.at_utc, "rows": len(df),
                          "af_played": len(af), "af_rows": int((df["provider"] == "api_football").sum())})
        for r in af.itertuples(index=False):
            if r.match_id not in first:
                first[r.match_id] = {"match_id": r.match_id, "league": r.league, "kickoff": r.kickoff,
                                     "t_af": row.at_utc, "af_sha": row.sha, "af_home": r.home_goals,
                                     "af_away": r.away_goals, "home_team": r.home_team, "away_team": r.away_team}
    return pd.DataFrame(first.values()), pd.DataFrame(seen_rows)


def fd_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.data.cleaning import normalise
    path = f"local_data/results/{C.SEASON_CODE}"
    cm = C.commits(path)
    first: dict[str, dict] = {}
    per = []
    for row in cm.itertuples():
        n = 0
        for lg in LEAGUES:
            raw = C.csv_at(row.sha, f"{path}/{lg}.csv")
            if raw is None or raw.empty:
                continue
            raw.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in raw.columns]
            df = normalise(raw)
            df = df.loc[df["played"]]
            n += len(df)
            for r in df.itertuples(index=False):
                if r.match_id not in first:
                    first[r.match_id] = {"match_id": r.match_id, "fd_league": r.league, "fd_kickoff": r.kickoff,
                                         "t_fd": row.at_utc, "fd_sha": row.sha, "fd_home": r.home_goals,
                                         "fd_away": r.away_goals, "fd_home_team": r.home_team,
                                         "fd_away_team": r.away_team}
        per.append({"sha": row.sha, "at_utc": row.at_utc, "fd_played": n})
    return pd.DataFrame(first.values()), pd.DataFrame(per)


# Runs between these commits asked API-Football (3 requests each in api_football/quota.log) but saved no new
# result: 17.10 16:26 -> 28.10 17:46 UTC (the cause is not in the committed data; diagnostics start 04.11).
DEAD = (pd.Timestamp("2026-10-17 16:26:54"), pd.Timestamp("2026-10-28 17:46:27"))
UNMATCHED_FD = {"OFI": "OFI Crete", "Vicenza Virtus": "Vicenza", "Real Sociedad II": "Sociedad B"}


def fd_only_reason(r, runs: pd.DataFrame, unmatched: pd.DataFrame) -> str:
    """Why API-Football never delivered a match Football-Data has (checks that can be made locally)."""
    teams = {getattr(r, "fd_home_team", None), getattr(r, "fd_away_team", None)}
    if teams & set(UNMATCHED_FD.values()):
        return "имя клуба API-Football не сопоставлено"
    ko = r.kickoff_utc
    day = pd.Timestamp(r.kickoff).normalize()
    window = runs.loc[(runs["at_utc"] >= ko + pd.Timedelta(hours=2))
                      & (C.uk_to_utc(day + pd.Timedelta(days=2)).iloc[0] > runs["at_utc"])]
    if window.empty:
        return "не было запуска в окне API-Football (вчера/сегодня)"
    if window["at_utc"].between(DEAD[0] + pd.Timedelta(seconds=1), DEAD[1] - pd.Timedelta(seconds=1)).all():
        return "запуски 17–28.10 без данных API-Football"
    return "другое"


def picks_of(match_ids) -> pd.DataFrame:
    from src.analytics.performance import unified_history
    p = pd.read_parquet(C.PICKS_FILE, columns=["pick_id", "match_id", "league", "season", "date", "role", "market",
                                               "group", "selection", "status", "odds", "profit", "source",
                                               "generated_at", "home_team", "away_team", "home_goals",
                                               "away_goals", "origin", "odds_origin", "provider", "probability",
                                               "ev", "bookmaker"])
    h = unified_history(p)
    return h.loc[h["match_id"].isin(set(match_ids))]


def logos() -> dict:
    d = json.loads((C.DATA / "teams" / "logos.json").read_text())
    return {"clubs": len(d), "by_source": pd.Series([v.get("source") for v in d.values()]).value_counts().to_dict()}


def quota_log() -> dict:
    q = pd.read_json(C.API / "api_football" / "quota.log", lines=True)
    q["run"] = q["at"].astype(str).str[:13]
    per = q.groupby("run").size()
    return {"requests": int(len(q)), "runs": int(per.size), "per_run_min": int(per.min()), "per_run_max": int(per.max()),
            "first": str(q["at"].min()), "last": str(q["at"].max())}


def run() -> dict:
    af, live_runs = live_history()
    fd, fd_runs = fd_history()
    C.save_csv("af_live_commits.csv", live_runs)
    C.save_csv("fd_result_commits.csv", fd_runs)
    start = live_runs.loc[live_runs["af_played"] > 0, "at_utc"].min()
    last = max(live_runs["at_utc"].max(), fd_runs["at_utc"].max())

    m = fd.merge(af, on="match_id", how="outer")
    m["league"] = m["league"].fillna(m["fd_league"])
    m["kickoff"] = pd.to_datetime(m["kickoff"].fillna(m["fd_kickoff"]))
    m["kickoff_utc"] = C.uk_to_utc(m["kickoff"]).to_numpy()
    m = m.loc[m["league"].isin(LEAGUES) & (m["kickoff_utc"] >= start - pd.Timedelta(hours=3))].copy()
    # a match whose kick-off time is unknown (00:00 in Football-Data) is kept; its delays count from midnight
    m["af_after_ko_h"] = (m["t_af"] - m["kickoff_utc"]).dt.total_seconds() / 3600
    m["fd_after_ko_h"] = (m["t_fd"] - m["kickoff_utc"]).dt.total_seconds() / 3600
    m["lead_h"] = (m["t_fd"] - m["t_af"]).dt.total_seconds() / 3600

    def cat(r):
        if pd.notna(r.t_af) and pd.notna(r.t_fd):
            if r.t_af < r.t_fd:
                return "API-Football раньше"
            if r.t_af == r.t_fd:
                return "в одном запуске"
            return "Football-Data раньше"
        if pd.notna(r.t_af):
            return "только API-Football (Football-Data ещё нет)"
        return "только Football-Data"
    m["category"] = [cat(r) for r in m.itertuples()]
    m["score_differs"] = (m["af_home"].notna() & m["fd_home"].notna()
                          & ((m["af_home"] != m["fd_home"]) | (m["af_away"] != m["fd_away"])))
    C.save_csv("af_vs_fd_matches.csv", m.drop(columns=["af_sha", "fd_sha"]).sort_values("kickoff"))

    led = m.loc[m["category"] == "API-Football раньше"]
    pend = m.loc[m["category"].str.startswith("только API")]
    lead_days = (led["lead_h"] / 24)
    quant = lambda s, q: float(s.quantile(q)) if len(s) else None   # noqa: E731
    dist = {"n": int(len(led)), "mean_h": float(led["lead_h"].mean()) if len(led) else None,
            "median_h": quant(led["lead_h"], .5), "p25_h": quant(led["lead_h"], .25),
            "p75_h": quant(led["lead_h"], .75), "p90_h": quant(led["lead_h"], .9),
            "max_h": float(led["lead_h"].max()) if len(led) else None,
            "days_hist": lead_days.round().astype(int).value_counts().sort_index().to_dict()}
    delay = {}
    for src, col in (("api_football", "af_after_ko_h"), ("football_data", "fd_after_ko_h")):
        s = m[col].dropna()
        delay[src] = {"n": int(len(s)), "median_h": quant(s, .5), "p25_h": quant(s, .25), "p75_h": quant(s, .75),
                      "p90_h": quant(s, .9)}

    # results delivered first by API-Football per run (commit) and per Kyiv day
    led_runs = (m.loc[m["category"].isin(["API-Football раньше", "только API-Football (Football-Data ещё нет)"])]
                .groupby("t_af").size().rename("first_by_af").reset_index())
    led_runs["kyiv_day"] = C.utc_to_kyiv(led_runs["t_af"]).dt.date.astype(str).to_numpy()
    per_day = led_runs.groupby("kyiv_day")["first_by_af"].sum()
    all_af_runs = live_runs.loc[live_runs["at_utc"] >= start]
    per_run_stats = {"runs_with_live_file": int(len(all_af_runs)),
                     "runs_with_af_first_results": int(len(led_runs)),
                     "mean_per_run_with_results": float(led_runs["first_by_af"].mean()) if len(led_runs) else None,
                     "median_per_run": quant(led_runs["first_by_af"], .5),
                     "max_per_run": int(led_runs["first_by_af"].max()) if len(led_runs) else None,
                     "days": int(per_day.size), "mean_per_day": float(per_day.mean()) if len(per_day) else None}
    C.save_csv("af_first_results_per_run.csv", led_runs)

    # value: predictions settled from API-Football goals before Football-Data had the match, and later
    # predictions whose features already contained the API-Football result
    early = m.loc[m["category"].isin(["API-Football раньше", "только API-Football (Football-Data ещё нет)"])]
    picks = picks_of(early["match_id"])
    roles = picks.loc[picks["role"].isin(["main", "risk"])]
    corner = roles["group"].eq("CORNERS")
    settle = {"matches": int(early["match_id"].nunique()),
              "main_risk_predictions": int(len(roles)),
              "settleable_with_goals": int((~corner).sum()), "need_corner_statistics": int(corner.sum()),
              "market_forecasts": int((picks["role"] == "market").sum())}
    feats = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                            columns=["match_id", "league", "season", "kickoff", "home_team", "away_team", "played"])
    feats = feats.loc[feats["season"] == C.SEASON]
    feats["kickoff_utc"] = C.uk_to_utc(feats["kickoff"]).to_numpy()
    windows = led[["match_id", "league", "home_team", "away_team", "t_af", "t_fd"]]
    later = set()
    for r in windows.itertuples(index=False):
        g = feats.loc[(feats["league"] == r.league) & (feats["kickoff_utc"] > r.t_af) & (feats["kickoff_utc"] <= r.t_fd)
                      & (feats["home_team"].isin([r.home_team, r.away_team]) | feats["away_team"].isin([r.home_team, r.away_team]))]
        later |= set(g["match_id"])
    settle["next_matches_with_af_result_in_features"] = len(later)

    # what API-Football does not deliver: Football-Data-only matches, by reason that can be checked locally
    fd_only = m.loc[m["category"] == "только Football-Data"].copy()
    unmatched = pd.read_csv(C.API / "unmatched_teams.csv") if (C.API / "unmatched_teams.csv").exists() else pd.DataFrame()
    fd_only["reason"] = [fd_only_reason(r, live_runs, unmatched) for r in fd_only.itertuples()]
    C.save_csv("af_missing_matches.csv", fd_only[["match_id", "league", "kickoff", "t_fd", "reason"]].sort_values("kickoff"))
    not_played = pd.read_csv(C.API / "api_football" / "not_played.csv") if (C.API / "api_football" / "not_played.csv").exists() else pd.DataFrame()

    out = {"method": __doc__.strip(), "start_utc": str(start), "last_commit_utc": str(last),
           "commits_live_file": int(len(live_runs)), "commits_fd_files": int(len(fd_runs)),
           "matches": int(len(m)), "categories": m["category"].value_counts().to_dict(),
           "categories_by_league": m.groupby(["league", "category"]).size().unstack(fill_value=0).reset_index().to_dict("records"),
           "lead": dist, "delay_after_kickoff": delay, "per_run": per_run_stats,
           "per_day": per_day.to_dict(), "pending_now": int(len(pend)),
           "pending_oldest_h": float(((last - pend["t_af"]).dt.total_seconds() / 3600).max()) if len(pend) else None,
           "score_differs": m.loc[m["score_differs"], ["match_id", "af_home", "af_away", "fd_home", "fd_away"]].to_dict("records"),
           "value": settle,
           "fd_only": {"n": int(len(fd_only)), "by_league": fd_only["league"].value_counts().to_dict(),
                       "by_reason": fd_only["reason"].value_counts().to_dict(),
                       "dead_period": [str(DEAD[0]), str(DEAD[1])],
                       "examples": fd_only.sort_values("kickoff").tail(12)[["match_id", "kickoff", "t_fd"]].astype(str).to_dict("records")},
           "unmatched_names": unmatched.to_dict("records"),
           "not_played_rows": int(len(not_played)),
           "logos": logos(),
           "quota_log": quota_log()}
    C.save_json("api_football.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("start_utc", "matches", "categories", "lead", "delay_after_kickoff",
                                        "per_run", "pending_now", "value", "fd_only", "unmatched_names",
                                        "score_differs")}, indent=1, default=str))
