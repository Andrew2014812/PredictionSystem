"""Audit of the raw inputs: Football-Data season files, the Football-Data fixtures file, the API providers' live
fixtures / results, and how they changed over the committed daily versions (git history, read only).

Writes results/raw_*.csv / raw_audit.json.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning import normalise
from src.data.dataset import _read_raw_csv
from src.leagues import league_codes
from src.providers import aliases

from . import common as C

STAT_GROUPS = {"goals": ["home_goals", "away_goals"], "shots": ["home_shots", "away_shots"],
               "sot": ["home_sot", "away_sot"], "corners": ["home_corners", "away_corners"],
               "fouls": ["home_fouls", "away_fouls"], "cards": ["home_yellows", "away_yellows", "home_reds", "away_reds"],
               "price_1x2": ["odds_home", "odds_draw", "odds_away"], "price_ou25": ["odds_over25", "odds_under25"],
               "price_ah": ["odds_ah_line", "odds_ah_home", "odds_ah_away"]}
SINGLE_ROUND_ROBIN = {"E0", "E1", "E2", "E3", "D1", "D2", "SP1", "SP2", "I1", "I2", "F1", "F2", "N1", "P1", "T1"}
RESULT_FILES = "local_data/results"


def load_raw(enabled_only: bool = True) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(C.DATA / "results" / "*" / "*.csv"))):
        path = Path(p)
        if enabled_only and path.stem not in league_codes():
            continue
        raw = _read_raw_csv(path)
        n = normalise(raw)
        n["file_season"], n["file_league"] = path.parent.name, path.stem
        n["raw_rows"] = len(raw)
        n["low_odds_raw"] = int(sum((pd.to_numeric(raw[c], errors="coerce") <= 1.0).sum()
                                    for c in ("B365H", "B365D", "B365A") if c in raw))
        frames.append(n)
    return pd.concat(frames, ignore_index=True)


def _long(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["match_id", "league", "country", "season", "date", "played"]
    return pd.concat([df[cols + ["home_team"]].rename(columns={"home_team": "team"}).assign(venue="home"),
                      df[cols + ["away_team"]].rename(columns={"away_team": "team"}).assign(venue="away")],
                     ignore_index=True)


def inventory(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lg, s), g in m.groupby(["league", "season"]):
        lt = _long(g.loc[g["played"]])
        per_team = lt.groupby("team").size()
        teams = int(pd.concat([g["home_team"], g["away_team"]]).nunique())
        rr = teams * (teams - 1)
        rows.append({"league": lg, "season": int(s), "rows": len(g), "played": int(g["played"].sum()),
                     "unplayed_in_results": int((~g["played"]).sum()), "teams": teams,
                     "double_round_robin": rr, "extra_rows": int(len(g) - rr),
                     "games_per_team_min": int(per_team.min()) if len(per_team) else 0,
                     "games_per_team_max": int(per_team.max()) if len(per_team) else 0,
                     "first_date": str(g["date"].min().date()), "last_date": str(g["date"].max().date())})
    return pd.DataFrame(rows)


def missing_table(m: pd.DataFrame) -> pd.DataFrame:
    played = m.loc[m["played"]]
    rows = []
    for (lg, s), g in played.groupby(["league", "season"]):
        rec = {"league": lg, "season": int(s), "played": len(g)}
        for name, cols in STAT_GROUPS.items():
            rec[name] = float(g[cols].isna().any(axis=1).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def sanity(m: pd.DataFrame) -> dict:
    p = m.loc[m["played"]]
    out = {}
    out["sot_gt_shots"] = p.loc[(p["home_sot"] > p["home_shots"]) | (p["away_sot"] > p["away_shots"]),
                                ["match_id", "home_shots", "home_sot", "away_shots", "away_sot"]]
    out["goals_gt_sot"] = p.loc[(p["home_goals"] > p["home_sot"]) | (p["away_goals"] > p["away_sot"]),
                                ["match_id", "home_goals", "home_sot", "away_goals", "away_sot"]]
    num = [c for cols in STAT_GROUPS.values() for c in cols if c not in ("odds_ah_line",)]
    out["negative"] = p.loc[(p[num] < 0).any(axis=1), ["match_id"]]
    out["unplayed_rows"] = m.loc[~m["played"], ["match_id", "file_season", "file_league"]]
    lt = _long(p).sort_values(["league", "team", "date"])
    lt["gap_days"] = lt.groupby(["league", "season", "team"])["date"].diff().dt.days
    out["gap_le_1_day"] = lt.loc[lt["gap_days"] <= 1, ["match_id", "team", "gap_days"]]
    nostats = p.loc[p["home_shots"].isna()].copy()
    nostats["awarded_like"] = (((nostats["home_goals"] == 3) & (nostats["away_goals"] == 0))
                               | ((nostats["home_goals"] == 0) & (nostats["away_goals"] == 3)))
    out["no_stats"] = nostats[["match_id", "league", "season", "home_goals", "away_goals", "awarded_like"]]
    return out


def duplicates(m: pd.DataFrame) -> dict:
    single = m.loc[m["league"].isin(SINGLE_ROUND_ROBIN)]
    pair = single.loc[single.duplicated(["league", "season", "home_team", "away_team"], keep=False)]
    lt = _long(m)
    booked = lt.loc[lt.duplicated(["league", "date", "team"], keep=False)]
    return {"duplicate_match_id": int(m["match_id"].duplicated().sum()),
            "same_pairing_twice_single_rr": int(len(pair)), "team_twice_same_day": int(len(booked)),
            "season_folder_mismatch": int((m["season"].map(lambda s: f"{s % 100:02d}{(s + 1) % 100:02d}")
                                           != m["file_season"]).sum()),
            "division_mismatch": int((m["league"] != m["file_league"]).sum())}


# ------------------------------------------------------------------------------------------------ git history
def fixture_name_drift(m: pd.DataFrame) -> pd.DataFrame:
    """Every committed version of the fixtures file: club names unknown to the league's results of the
    current and previous season, and what the production rule (aliases.resolve) maps them to."""
    recent = m.loc[m["season"] >= m["season"].max() - 1]
    known = {lg: set(g["home_team"]) | set(g["away_team"]) for lg, g in recent.groupby("league")}
    rows = []
    versions = C.file_versions("local_data/fixtures/fixtures.csv")
    for sha, ts, _ in versions:
        fx = normalise(C.read_version_csv(sha, "local_data/fixtures/fixtures.csv"))
        fx = fx.loc[fx["league"].isin(league_codes())]
        for r in fx.itertuples():
            for side in ("home_team", "away_team"):
                name = getattr(r, side)
                if name not in known.get(r.league, set()):
                    rows.append({"commit": sha[:7], "committed_utc": str(ts), "league": r.league,
                                 "date": str(r.date.date()), "home": r.home_team, "away": r.away_team,
                                 "name": name, "resolves_to": aliases.resolve(name, known.get(r.league)),
                                 "match_id": r.match_id})
    out = pd.DataFrame(rows).drop_duplicates(["league", "date", "home", "away"]) if rows else pd.DataFrame(
        columns=["commit", "committed_utc", "league", "date", "home", "away", "name", "resolves_to", "match_id"])
    return out, len(versions)


def results_history() -> tuple[pd.DataFrame, int]:
    """Changes between consecutive committed versions of every 2026/27 results file: a played match that
    disappears (renamed id), a changed score or statistic, a newly appearing club name."""
    rows, n_versions = [], 0
    for p in sorted(glob.glob(str(C.DATA / "results" / "2627" / "*.csv"))):
        rel = str(Path(p).relative_to(C.ROOT))
        if Path(p).stem not in league_codes():
            continue
        prev, prev_names = None, None
        for sha, ts, _ in C.file_versions(rel):
            n_versions += 1
            cur = normalise(C.read_version_csv(sha, rel))
            cur = cur.loc[cur["played"]].set_index("match_id")
            names = set(cur["home_team"]) | set(cur["away_team"])
            if prev is not None:
                for mid in sorted(set(prev.index) - set(cur.index)):
                    rows.append({"file": Path(p).name, "commit": sha[:7], "committed_utc": str(ts),
                                 "change": "played match disappeared", "match_id": mid, "detail": ""})
                for name in sorted(names - prev_names):
                    if len(prev_names) >= 6:                     # a name new to a running season
                        n_games = int(((cur["home_team"] == name) | (cur["away_team"] == name)).sum())
                        rows.append({"file": Path(p).name, "commit": sha[:7], "committed_utc": str(ts),
                                     "change": "new club name in a running season", "match_id": "",
                                     "detail": f"{name} ({n_games} played)"})
                both = prev.index.intersection(cur.index)
                for col in ("home_goals", "away_goals", "home_shots", "away_shots", "home_corners", "away_corners",
                            "odds_home", "odds_draw", "odds_away"):
                    a, b = prev.loc[both, col], cur.loc[both, col]
                    changed = both[(a.fillna(-99) != b.fillna(-99))]
                    for mid in changed:
                        rows.append({"file": Path(p).name, "commit": sha[:7], "committed_utc": str(ts),
                                     "change": f"{col} changed", "match_id": mid,
                                     "detail": f"{prev.at[mid, col]} -> {cur.at[mid, col]}"})
            prev, prev_names = cur, names
    return pd.DataFrame(rows, columns=["file", "commit", "committed_utc", "change", "match_id", "detail"]), n_versions


def live_history() -> tuple[pd.DataFrame, int]:
    """Committed versions of the API providers' fixture / result file: results that turn back into
    fixtures, changed scores, removed rows."""
    rel = "local_data/results_live/fixtures.parquet"
    cols = ["match_id", "played", "home_goals", "away_goals", "provider", "date"]
    rows, prev = [], None
    versions = C.file_versions(rel)
    for sha, ts, _ in versions:
        cur = C.read_version_parquet(sha, rel, columns=cols).drop_duplicates("match_id", keep="last").set_index("match_id")
        cur["played"] = cur["played"].astype(bool)
        if prev is not None:
            both = prev.index.intersection(cur.index)
            a, b = prev.loc[both], cur.loc[both]
            for mid in both[a["played"] & ~b["played"]]:
                rows.append({"commit": sha[:7], "committed_utc": str(ts), "change": "result turned back into a fixture",
                             "match_id": mid, "detail": f"{a.at[mid, 'home_goals']:.0f}-{a.at[mid, 'away_goals']:.0f} "
                                                        f"({a.at[mid, 'provider']} -> {b.at[mid, 'provider']})"})
            sc = both[a["played"] & b["played"] & ((a["home_goals"] != b["home_goals"])
                                                   | (a["away_goals"] != b["away_goals"]))]
            for mid in sc:
                rows.append({"commit": sha[:7], "committed_utc": str(ts), "change": "score changed", "match_id": mid,
                             "detail": f"{a.at[mid, 'home_goals']}-{a.at[mid, 'away_goals']} -> "
                                       f"{b.at[mid, 'home_goals']}-{b.at[mid, 'away_goals']}"})
            for mid in sorted(set(prev.index) - set(cur.index)):
                rows.append({"commit": sha[:7], "committed_utc": str(ts), "change": "row removed", "match_id": mid,
                             "detail": f"played={bool(prev.at[mid, 'played'])} ({prev.at[mid, 'provider']})"})
        prev = cur
    return pd.DataFrame(rows, columns=["commit", "committed_utc", "change", "match_id", "detail"]), len(versions)


# ------------------------------------------------------------------------------------------------ live vs Football-Data
def live_vs_fd(m: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    live = pd.read_parquet(C.LIVE_FILE)
    live["date"] = pd.to_datetime(live["date"])
    live["played"] = live["played"].astype(bool)
    fd = m.loc[m["season"] >= m["season"].max() - 1]
    ids = set(fd["match_id"])
    key = set(zip(fd["league"], fd["date"], fd["home_team"], fd["away_team"]))
    swapped = [r.match_id for r in live.itertuples() if (r.league, r.date, r.away_team, r.home_team) in key]
    by_pair = {k: g["date"].tolist() for k, g in fd.groupby(["league", "home_team", "away_team"])}
    moved = []
    for r in live.itertuples():
        if r.match_id in ids:
            continue
        dates = [d for d in by_pair.get((r.league, r.home_team, r.away_team), []) if abs((d - r.date).days) <= 60]
        if dates:
            moved.append({"match_id": r.match_id, "provider": r.provider, "provider_date": str(r.date.date()),
                          "fd_date": str(dates[0].date()), "days": int((dates[0] - r.date).days),
                          "provider_played": bool(r.played)})
    j = live.loc[live["played"] & live["match_id"].isin(ids)].merge(
        fd[["match_id", "home_goals", "away_goals", "time"]], on="match_id", suffixes=("_api", "_fd"))
    score_mismatch = j.loc[(j["home_goals_api"] != j["home_goals_fd"]) | (j["away_goals_api"] != j["away_goals_fd"])]
    jt = live.loc[live["match_id"].isin(ids)].merge(fd[["match_id", "time"]], on="match_id", suffixes=("_api", "_fd"))
    api_only = live.loc[live["played"] & ~live["match_id"].isin(ids)]
    last_fd = fd.groupby("league")["date"].max()
    api_only_before_fd_end = api_only.loc[api_only["date"] <= api_only["league"].map(last_fd)]
    summary = {"live_rows": int(len(live)), "by_provider": live["provider"].value_counts().to_dict(),
               "played_rows": int(live["played"].sum()), "in_fd": int(live["match_id"].isin(ids).sum()),
               "swapped_home_away": len(swapped), "swapped_ids": swapped,
               "moved_vs_fd": len(moved), "score_compared": int(len(j)), "score_mismatch": int(len(score_mismatch)),
               "time_compared": int(len(jt)), "time_mismatch": int((jt["time_api"] != jt["time_fd"]).sum()),
               "api_only_played": int(len(api_only)),
               "api_only_played_before_fd_last_date": int(len(api_only_before_fd_end)),
               "api_only_played_before_fd_last_date_ids": api_only_before_fd_end["match_id"].tolist()}
    return summary, pd.DataFrame(moved)


def stale_fixtures(today: pd.Timestamp) -> pd.DataFrame:
    f = C.features(["match_id", "league", "date", "kickoff", "home_team", "away_team", "played"])
    up = f.loc[~f["played"].astype(bool)]
    last_played = f.loc[f["played"].astype(bool)].groupby("league")["date"].max()
    stale = up.loc[up["date"] < up["league"].map(last_played)].copy()
    stale["days_overdue_vs_league_data"] = (stale["league"].map(last_played) - stale["date"]).dt.days
    live = pd.read_parquet(C.LIVE_FILE, columns=["match_id", "provider"]).drop_duplicates("match_id", keep="last")
    stale["source"] = stale["match_id"].map(live.set_index("match_id")["provider"]).fillna("football_data fixtures")
    return stale


def provider_names(m: pd.DataFrame) -> pd.DataFrame:
    un = pd.read_csv(C.DATA / "api" / "unmatched_teams.csv")
    cur = m.loc[m["season"] == m["season"].max()]
    rows = []
    for r in un.itertuples():
        g = cur.loc[cur["league"] == r.league]
        known = set(g["home_team"]) | set(g["away_team"])
        norm = aliases.normalise(r.name)
        cands = sorted(k for k in known if set(aliases.normalise(k).split()) & set(norm.split())
                       or aliases.normalise(k).startswith(norm))
        games = {c: int(((g["home_team"] == c) | (g["away_team"] == c)).sum()) for c in cands}
        rows.append({"provider": r.provider, "league": r.league, "provider_name": r.name,
                     "normalised": norm, "resolves_to": aliases.resolve(r.name, known),
                     "fd_candidates": "; ".join(f"{c} ({n})" for c, n in games.items())})
    return pd.DataFrame(rows)


def run() -> dict:
    m = load_raw()
    inv = inventory(m)
    C.save_csv("raw_inventory.csv", inv)
    miss = missing_table(m)
    C.save_csv("raw_missing.csv", miss)
    san = sanity(m)
    dup = duplicates(m)
    drift, n_fx_versions = fixture_name_drift(m)
    C.save_csv("raw_fixture_name_drift.csv", drift)
    res_hist, n_res_versions = results_history()
    C.save_csv("raw_results_history.csv", res_hist)
    live_hist, n_live_versions = live_history()
    C.save_csv("raw_live_history.csv", live_hist)
    lv, moved = live_vs_fd(m)
    C.save_csv("raw_moved_matches.csv", moved)
    today = pd.Timestamp(C.FEATURES_FILE.stat().st_mtime, unit="s").normalize()
    stale = stale_fixtures(today)
    C.save_csv("raw_stale_fixtures.csv", stale.assign(date=stale["date"].dt.date.astype(str),
                                                      kickoff=stale["kickoff"].astype(str)))
    names = provider_names(m)
    C.save_csv("raw_unmatched_names.csv", names)
    conflicts = pd.read_csv(C.DATA / "api" / "team_id_conflicts.csv")
    ec = load_raw(enabled_only=False)
    ec = ec.loc[ec["league"] == "EC"]
    played = m.loc[m["played"]]
    out = {
        "files": int(m.groupby(["file_season", "file_league"]).ngroups), "rows": int(len(m)),
        "played": int(m["played"].sum()), "leagues": sorted(m["league"].unique()),
        "seasons": sorted(int(s) for s in m["season"].unique()),
        "duplicates": dup,
        "sanity": {"sot_gt_shots": int(len(san["sot_gt_shots"])), "sot_gt_shots_ids": san["sot_gt_shots"]["match_id"].tolist(),
                   "goals_gt_sot": int(len(san["goals_gt_sot"])),
                   "goals_gt_sot_share": float(len(san["goals_gt_sot"]) / max(len(played), 1)),
                   "negative_values": int(len(san["negative"])),
                   "unplayed_rows_in_results": san["unplayed_rows"]["match_id"].tolist(),
                   "same_team_two_games_within_1_day": int(len(san["gap_le_1_day"])),
                   "same_team_two_games_within_1_day_ids": san["gap_le_1_day"]["match_id"].tolist(),
                   "played_without_stats": int(len(san["no_stats"])),
                   "played_without_stats_awarded_like": int(san["no_stats"]["awarded_like"].sum()),
                   "played_without_stats_by_league_season": {f"{lg} {s}": int(n) for (lg, s), n in
                                                             san["no_stats"].groupby(["league", "season"]).size().items()},
                   "low_odds_raw_values": int(m.drop_duplicates(["file_season", "file_league"])["low_odds_raw"].sum())},
        "missing_any_stat_overall": {k: float(played[v].isna().any(axis=1).mean()) for k, v in STAT_GROUPS.items()},
        "missing_by_season": {int(s): {k: float(g[v].isna().any(axis=1).mean()) for k, v in STAT_GROUPS.items()}
                              for s, g in played.groupby("season")},
        "ec_disabled": {"rows": int(len(ec)), "corners_missing": float(ec.loc[ec["played"], "home_corners"].isna().mean())
                        if len(ec) else None},
        "fixture_versions": n_fx_versions, "fixture_name_drift": int(len(drift)),
        "fixture_name_drift_rows": drift.to_dict("records"),
        "results_versions": n_res_versions,
        "results_changes": res_hist["change"].value_counts().to_dict() if len(res_hist) else {},
        "live_versions": n_live_versions,
        "live_changes": live_hist["change"].value_counts().to_dict() if len(live_hist) else {},
        "live_vs_fd": lv, "stale_fixtures": int(len(stale)),
        "stale_fixture_ids": stale["match_id"].tolist(),
        "unmatched_provider_names": names.to_dict("records"), "team_id_conflicts": conflicts.to_dict("records"),
        "league_size_changes": _size_changes(inv),
    }
    C.save_json("raw_audit.json", out)
    return out


def _size_changes(inv: pd.DataFrame) -> list[dict]:
    out = []
    for lg, g in inv.sort_values("season").groupby("league"):
        prev = None
        for r in g.itertuples():
            if prev is not None and r.teams != prev and r.season < g["season"].max() + 1:
                out.append({"league": lg, "season": int(r.season), "teams_before": int(prev), "teams": int(r.teams)})
            prev = r.teams
    return out


if __name__ == "__main__":
    C.setup_logging("raw_audit")
    run()
