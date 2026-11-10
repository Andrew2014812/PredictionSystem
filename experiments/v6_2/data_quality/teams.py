"""Team histories: where every club of every league-season comes from, how much history the system has for it
at each match, clubs that appear under two names, and what the pre-match features of newcomers look like.

Status of a club in league L, season s (the system's own key is (country, Football-Data name)):

    established   played in L in season s-1
    moved_within  played in another of our leagues of the same country in s-1 (promoted / relegated inside our data)
    returner      known from an earlier season but absent in s-1 (came back from a division we do not load)
    outside       never seen before in our data (promoted from a division we do not load, or a new name)
    first_season  season 2021/22, the first season of the data (no status can be derived)

Writes results/teams_*.csv and teams_audit.json; cache/team_status.parquet is used by the model experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sets import model_features
from src.providers import aliases

from . import common as C

ID = ["match_id", "league", "country", "season", "date", "home_team", "away_team", "played"]


def _sides(f: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([f[ID].rename(columns={"home_team": "team", "away_team": "opp"}).assign(side="h"),
                      f[ID].rename(columns={"away_team": "team", "home_team": "opp"}).assign(side="a")],
                     ignore_index=True)


def team_seasons(f: pd.DataFrame) -> pd.DataFrame:
    """One row per (league, season, team) with its status and where it came from."""
    long = _sides(f.loc[f["played"].astype(bool)])
    ts = long.groupby(["league", "country", "season", "team"]).size().rename("games").reset_index()
    first = int(ts["season"].min())
    seen = {(r.country, r.team, r.season): r.league for r in ts.itertuples()}
    seasons_of = ts.groupby(["country", "team"])["season"].apply(set).to_dict()
    status, origin = [], []
    for r in ts.itertuples():
        prev = seen.get((r.country, r.team, r.season - 1))
        earlier = [s for s in seasons_of[(r.country, r.team)] if s < r.season - 1]
        if r.season == first:
            status.append("first_season"), origin.append("")
        elif prev == r.league:
            status.append("established"), origin.append(r.league)
        elif prev is not None:
            status.append("moved_within"), origin.append(prev)
        elif earlier:
            status.append("returner"), origin.append(f"last seen {max(earlier)}/{(max(earlier) + 1) % 100:02d}")
        else:
            status.append("outside"), origin.append("")
    ts["status"], ts["origin"] = status, origin
    return ts


def two_name_candidates(ts: pd.DataFrame) -> pd.DataFrame:
    """A club that first appears ('outside') the season another club of the same country and league level
    disappears, with a shared distinctive word in the normalised names."""
    rows = []
    for r in ts.loc[ts["status"].isin(["outside", "returner"])].itertuples():
        gone = ts.loc[(ts["country"] == r.country) & (ts["season"] == r.season - 1)]
        here = set(ts.loc[(ts["country"] == r.country) & (ts["season"] == r.season), "team"])
        words = set(aliases.normalise(r.team).split())
        for g in gone.itertuples():
            if g.team in here:
                continue
            common = words & set(aliases.normalise(g.team).split())
            if any(len(w) >= 5 for w in common):
                rows.append({"country": r.country, "season": int(r.season), "new_name": r.team, "new_league": r.league,
                             "old_name": g.team, "old_league": g.league, "shared_words": " ".join(sorted(common))})
    for r in ts.loc[ts["status"] == "outside"].itertuples():           # renamed back and forth
        back = ts.loc[(ts["country"] == r.country) & (ts["season"] == r.season + 1)]
        gone_next = r.team not in set(back["team"])
        words = set(aliases.normalise(r.team).split())
        for b in back.loc[back["status"] == "returner"].itertuples():
            common = words & set(aliases.normalise(b.team).split())
            if gone_next and any(len(w) >= 5 for w in common):
                rows.append({"country": r.country, "season": int(r.season), "new_name": r.team, "new_league": r.league,
                             "old_name": b.team, "old_league": b.league, "shared_words": " ".join(sorted(common))})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(
        columns=["country", "season", "new_name", "new_league", "old_name", "old_league", "shared_words"])


def match_status(f: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """Per match and side: club status this season, prior matches of the club in our data (any league of the
    country), its match number in the season, days since its previous match in our data."""
    long = _sides(f).sort_values(["country", "team", "date"])
    played = long["played"].astype(bool).to_numpy()
    long["prior_matches"] = long.groupby(["country", "team"])["played"].transform(
        lambda s: s.astype(int).cumsum().shift(1, fill_value=0))
    long["season_match_no"] = long.groupby(["league", "season", "team"])["played"].transform(
        lambda s: s.astype(int).cumsum().shift(1, fill_value=0)) + 1
    last = long["date"].where(played)
    long["prev_date"] = last.groupby([long["country"], long["team"]]).transform(lambda s: s.ffill().shift(1))
    long["days_since_prev"] = (long["date"] - long["prev_date"]).dt.days
    st = ts.set_index(["league", "season", "team"])["status"]
    long["status"] = [st.get((lg, s, t), "outside" if p == 0 else "unknown")
                      for lg, s, t, p in zip(long["league"], long["season"], long["team"], long["prior_matches"])]
    wide = long.pivot_table(index="match_id", columns="side",
                            values=["prior_matches", "season_match_no", "days_since_prev"], aggfunc="first")
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    stat = long.pivot(index="match_id", columns="side", values="status")
    wide["h_status"], wide["a_status"] = stat["h"], stat["a"]
    return wide.reset_index()


def feature_profile(f: pd.DataFrame, ms: pd.DataFrame) -> pd.DataFrame:
    """Share of missing result-model features for a side, by its status and season match number."""
    num, _ = model_features("result", list(f.columns))
    side_cols = {s: [c for c in num if c.startswith(f"{s}_")] for s in ("h", "a")}
    x = f[["match_id", "season"]].merge(ms, on="match_id")
    rows = []
    for s in ("h", "a"):
        miss = f[side_cols[s]].isna().mean(axis=1).to_numpy()
        part = pd.DataFrame({"status": x[f"{s}_status"].to_numpy(), "no": x[f"{s}_season_match_no"].to_numpy(),
                             "missing": miss, "elo": f[f"{s}_elo"].to_numpy(),
                             "form10_n": f[f"{s}_form10_n"].to_numpy(), "season": x["season"].to_numpy()})
        rows.append(part)
    p = pd.concat(rows)
    p = p.loc[p["season"] >= 2022]
    p["bucket"] = pd.cut(p["no"], [0, 1, 3, 5, 10, 60], labels=["1", "2-3", "4-5", "6-10", "11+"])
    return (p.groupby(["status", "bucket"], observed=True)
            .agg(sides=("missing", "size"), missing_share=("missing", "mean"), form10_n=("form10_n", "mean"),
                 elo=("elo", "mean")).reset_index())


def run() -> dict:
    f = C.features()
    f = f.loc[f["season"] >= 2021]
    ts = team_seasons(f)
    C.save_csv("teams_seasons.csv", ts)
    names = two_name_candidates(ts)
    C.save_csv("teams_two_names.csv", names)
    ms = match_status(f, ts)
    ms.to_parquet(C.CACHE / "team_status.parquet")
    prof = feature_profile(f, ms)
    C.save_csv("teams_feature_profile.csv", prof)
    played = ts.loc[ts["season"] >= 2022]
    counts = played.groupby(["season", "status"]).size().unstack(fill_value=0)
    by_league = played.loc[played["status"].isin(["outside", "returner"])].groupby("league").size()
    # what a newcomer's first match looks like (all seasons): the features the model gets
    first = f.merge(ms[["match_id", "h_status", "a_status", "h_season_match_no", "a_season_match_no"]], on="match_id")
    first = first.loc[(first["season"] >= 2022)]
    ex = []
    for s in ("h", "a"):
        sel = first.loc[(first[f"{s}_status"] == "outside") & (first[f"{s}_season_match_no"] == 1)]
        ex.append(sel[[f"{s}_elo", f"{s}_form10_n", f"{s}_season_games", f"{s}_rest_days"]].rename(
            columns=lambda c: c[2:]))
    ex = pd.concat(ex)
    tier_base = {}
    from src.leagues import get_league
    for lg in sorted(f["league"].unique()):
        tier_base[lg] = 1500.0 - 100.0 * (get_league(lg).tier - 1)
    # Elo of outside newcomers at their first match vs the league's average Elo that day
    elo_gap = []
    for s in ("h", "a"):
        sel = first.loc[(first[f"{s}_status"] == "outside") & (first[f"{s}_season_match_no"] == 1)]
        elo_gap.append(sel[f"{s}_elo"] - sel["league"].map(
            f.loc[f["season"] >= 2022].groupby("league")[["h_elo", "a_elo"]].mean().mean(axis=1)))
    # end-of-season outcome of outside newcomers: points per game vs league average (1.37 for a 3-1-0 table)
    out_ppg = []
    long = _sides(f.loc[f["played"].astype(bool) & (f["season"] >= 2022)])
    long = long.merge(f[["match_id", "home_goals", "away_goals"]], on="match_id")
    gf = np.where(long["side"] == "h", long["home_goals"], long["away_goals"])
    ga = np.where(long["side"] == "h", long["away_goals"], long["home_goals"])
    long["pts"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    tsx = ts.set_index(["league", "season", "team"])["status"]
    long["status"] = [tsx.get(k, "?") for k in zip(long["league"], long["season"], long["team"])]
    ppg = long.groupby("status")["pts"].mean().to_dict()
    out = {"team_seasons": int(len(ts)), "status_by_season": {int(k): v for k, v in counts.to_dict("index").items()},
           "outside_or_returner_by_league": by_league.to_dict(),
           "two_name_candidates": names.to_dict("records"),
           "first_match_of_outside_newcomers": {"n": int(len(ex)), "form10_n_mean": float(ex["form10_n"].mean()),
                                                "season_games_mean": float(ex["season_games"].mean()),
                                                "rest_days_missing": float(ex["rest_days"].isna().mean()),
                                                "elo_mean": float(ex["elo"].mean()),
                                                "elo_minus_league_mean": float(pd.concat(elo_gap).mean())},
           "points_per_game_by_status": ppg,
           "returner_gap_days": _returner_gap(ms, f),
           "tier_base_elo": tier_base}
    C.save_json("teams_audit.json", out)
    return out


def _returner_gap(ms: pd.DataFrame, f: pd.DataFrame) -> dict:
    x = f[["match_id", "season"]].merge(ms, on="match_id")
    vals = []
    for s in ("h", "a"):
        sel = x.loc[(x[f"{s}_status"] == "returner") & (x[f"{s}_season_match_no"] == 1)]
        vals.append(sel[f"{s}_days_since_prev"])
    v = pd.concat(vals).dropna()
    return {"n": int(len(v)), "median_days": float(v.median()) if len(v) else None,
            "min_days": float(v.min()) if len(v) else None, "max_days": float(v.max()) if len(v) else None}


if __name__ == "__main__":
    C.setup_logging("teams")
    run()
