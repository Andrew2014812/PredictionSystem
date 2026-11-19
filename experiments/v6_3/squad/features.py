"""Squad-strength features from the player data, using only information dated before the match.

Two families:

* **dynamic** (``sq_*``) — for a match of club c on date d: the last 10 club games of c (every competition) played
  strictly before d give the typical starting XI (most starts, then minutes), the bench (next 7), positions (modal
  position in those line-ups), stability; every player is rated with values dated before d: market value (latest
  valuation before d), goals + assists per 90 over the 365 days before d (shrunk to the position-group mean with
  900 prior minutes), form over his last 5 appearances, on-pitch goal-difference impact over 365 days. Needs line-ups
  of the games before the match, so it exists only where the player source has them (2021/22–2025/26).
* **pre-season** (``pre_*``) — the same squad seen on 1 July of the season (end of the previous season's line-ups,
  minus players transferred away by 1 July, plus players transferred in by 1 July), constant during the season.
  Computable for 2026/27 from data frozen in July 2026, and with the same definition for every earlier season, so a
  model trained on earlier seasons sees the same feature it is applied with.

Market values are the provider's rating at that date (Transfermarkt keeps their history) — today's values are never
used for a past match. Positions come only from earlier line-ups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3.squad import tmdata as T

WINDOW = 10
XI, BENCH = 11, 7
PRIOR_MIN = 900.0
IMPACT_PRIOR = 10.0
YEAR = pd.Timedelta(days=365)

DYN_COLS = ["squad_value", "xi_value", "att_value", "mid_value", "def_value", "bench_value", "stability",
            "xi_ga90", "form", "impact", "wide_value", "wide_n", "last_xi_share", "xi_age", "value_cov"]
PRE_COLS = ["squad_value", "xi_value", "att_value", "mid_value", "def_value", "bench_value", "xi_ga90", "impact",
            "wide_value", "continuity", "arrivals_value", "arrivals_n", "value_cov"]


# ------------------------------------------------------------------------------------------------ inputs
class PlayerData:
    """Club games, per-game player rows and per-player histories, restricted to the mapped clubs' games."""

    def __init__(self, clubs: set[int]):
        g = T.load("games")
        comp = T.load("competitions").set_index("competition_id")["type"]
        g = g.loc[~g["competition_id"].isin(T.NATIONAL) & (g["competition_id"].map(comp) != "national_team_competition")]
        g = g.dropna(subset=["home_club_id", "away_club_id", "home_club_goals", "away_club_goals"])
        home = g.assign(club_id=g["home_club_id"], gf=g["home_club_goals"], ga=g["away_club_goals"])
        away = g.assign(club_id=g["away_club_id"], gf=g["away_club_goals"], ga=g["home_club_goals"])
        cg = pd.concat([home, away])[["game_id", "date", "competition_id", "club_id", "gf", "ga"]]
        cg = cg.astype({"game_id": "int64", "club_id": "int64", "gf": float, "ga": float})
        self.club_games = cg.loc[cg["club_id"].isin(clubs)].sort_values(["club_id", "date", "game_id"])
        game_ids = set(self.club_games["game_id"])

        a = T.load("appearances")
        a = a.loc[~a["competition_id"].isin(T.NATIONAL)].dropna(subset=["player_id", "player_club_id", "game_id"])
        a = a.astype({"game_id": "int64", "player_id": "int64", "player_club_id": "int64"})
        a["minutes_played"] = a["minutes_played"].fillna(0).astype(float)
        a["ga"] = a["goals"].fillna(0).astype(float) + a["assists"].fillna(0).astype(float)
        lu = T.load("lineups").dropna(subset=["player_id", "club_id", "game_id"])
        lu = lu.astype({"game_id": "int64", "player_id": "int64", "club_id": "int64"})
        lu = lu.loc[lu["game_id"].isin(game_ids)]
        lu["started"] = (lu["type"].astype(str) == "starting_lineup").astype(float)
        lu["position"] = lu["position"].astype(str)
        # per (game, club) player rows: line-up (starter / substitute, position) + appearance (minutes, G+A)
        ap = a.loc[a["game_id"].isin(game_ids), ["game_id", "player_id", "player_club_id", "minutes_played", "ga"]]
        ap = ap.rename(columns={"player_club_id": "club_id"})
        rows = lu[["game_id", "player_id", "club_id", "started", "position"]].merge(
            ap, on=["game_id", "player_id", "club_id"], how="outer")
        rows["minutes_played"] = rows["minutes_played"].fillna(0.0)
        has_lineup = rows.groupby(["game_id", "club_id"])["started"].transform(lambda s: s.notna().any())
        rows.loc[rows["started"].isna() & ~has_lineup, "started"] = (rows["minutes_played"] >= 60).astype(float)
        rows["started"] = rows["started"].fillna(0.0)
        rows["position"] = rows["position"].fillna("UNK")
        rows = rows.sort_values(["club_id", "game_id"]).reset_index(drop=True)
        self.rows = rows
        kk = rows[["club_id", "game_id"]].to_numpy()
        cut = np.flatnonzero((kk[1:] != kk[:-1]).any(1)) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, len(rows)]
        self.slices = {(int(kk[a, 0]), int(kk[a, 1])): (a, b) for a, b in zip(starts, ends)}

        # per-player histories (club competitions, every club) for per-90 output, form and impact
        gd = self.club_games.set_index(["game_id", "club_id"])
        a = a.merge(lu[["game_id", "player_id", "started"]], on=["game_id", "player_id"], how="left")
        a["started"] = a["started"].fillna((a["minutes_played"] >= 60).astype(float))
        key = pd.MultiIndex.from_arrays([a["game_id"], a["player_club_id"]])
        a["team_gd"] = (gd["gf"] - gd["ga"]).reindex(key).to_numpy()
        a = a.sort_values(["player_id", "date"])
        self.hist = {}
        for pid, h in a.groupby("player_id", sort=False):
            d = h["date"].to_numpy("datetime64[ns]")
            mins = h["minutes_played"].to_numpy(float)
            ga = h["ga"].to_numpy(float)
            st = h["started"].to_numpy(float)
            tgd = np.nan_to_num(h["team_gd"].to_numpy(float)) * st
            self.hist[pid] = (d, np.cumsum(mins), np.cumsum(ga), np.cumsum(st), np.cumsum(tgd), mins, ga)
        # club goal difference per game (cumulative) for the impact baseline
        self.club_hist = {}
        for cid, h in self.club_games.groupby("club_id", sort=False):
            self.club_hist[cid] = (h["date"].to_numpy("datetime64[ns]"), h["game_id"].to_numpy(),
                                   np.cumsum((h["gf"] - h["ga"]).to_numpy(float)))
        # market values
        v = T.load("valuations").dropna().sort_values(["player_id", "date"])
        self.values = {pid: (h["date"].to_numpy("datetime64[ns]"), np.log10(h["value"].clip(lower=10_000).to_numpy(float)))
                       for pid, h in v.groupby("player_id", sort=False)}
        p = T.load("players", ["player_id", "date_of_birth"]).dropna()
        self.dob = dict(zip(p["player_id"].astype("int64"), p["date_of_birth"].to_numpy("datetime64[ns]")))
        # transfers (for the pre-season squad)
        t = T.load("transfers").dropna(subset=["player_id", "transfer_date"])
        t = t.astype({"player_id": "int64"})
        self.transfers = t
        # position-group priors of goals + assists per 90, from 2019/20–2020/21 only (before every studied season)
        early = rows.merge(self.club_games[["game_id", "club_id", "date"]], on=["game_id", "club_id"])
        early = early.loc[early["date"] < pd.Timestamp("2021-07-01")]
        early["group"] = early["position"].map(T.POS_GROUP).fillna("MID")
        s = early.groupby("group")[["ga", "minutes_played"]].sum()
        self.prior = (s["ga"] / s["minutes_played"] * 90).to_dict()

    # -------------------------------------------------------------------------------------------- player ratings
    def value(self, pid: int, d: np.datetime64) -> float:
        h = self.values.get(pid)
        if h is None:
            return np.nan
        i = np.searchsorted(h[0], d, side="left")
        return float(h[1][i - 1]) if i > 0 else np.nan

    def per90(self, pid: int, d: np.datetime64, group: str) -> tuple[float, float, float]:
        """(shrunk G+A per 90 over 365 days, form deviation over the last 5 appearances,
        (sum of team goal difference in his starts, number of starts) over 365 days)."""
        prior = self.prior.get(group, self.prior.get("MID", 0.2))
        h = self.hist.get(pid)
        if h is None:
            return prior, 0.0, (0.0, 0.0)
        dates, cmin, cga, cst, cgd, mins, ga = h
        i = np.searchsorted(dates, d, side="left")
        j = np.searchsorted(dates, d - np.timedelta64(365, "D"), side="left")
        if i == 0:
            return prior, 0.0, (0.0, 0.0)
        m_ = cmin[i - 1] - (cmin[j - 1] if j > 0 else 0.0)
        g_ = cga[i - 1] - (cga[j - 1] if j > 0 else 0.0)
        rate = (g_ + prior * PRIOR_MIN / 90) / (m_ + PRIOR_MIN) * 90
        k = max(0, i - 5)
        m5, g5 = mins[k:i].sum(), ga[k:i].sum()
        form = (g5 / m5 * 90 - rate) * (m5 / (m5 + 270)) if m5 > 0 else 0.0
        n_st = cst[i - 1] - (cst[j - 1] if j > 0 else 0.0)
        gd_st = cgd[i - 1] - (cgd[j - 1] if j > 0 else 0.0)
        return rate, form, (gd_st, n_st)

    def club_gd_rate(self, cid: int, d: np.datetime64) -> float:
        h = self.club_hist.get(cid)
        if h is None:
            return 0.0
        i = np.searchsorted(h[0], d, side="left")
        j = np.searchsorted(h[0], d - np.timedelta64(365, "D"), side="left")
        n = i - j
        if n <= 0:
            return 0.0
        return float((h[2][i - 1] - (h[2][j - 1] if j > 0 else 0.0)) / n)

    def window(self, cid: int, d: np.datetime64, n: int = WINDOW) -> list[int]:
        h = self.club_hist.get(cid)
        if h is None:
            return []
        i = np.searchsorted(h[0], d, side="left")
        return list(h[1][max(0, i - n):i])


# ------------------------------------------------------------------------------------------------ team features
def _mean(x):
    x = np.asarray([v for v in x if v == v], float)
    return float(x.mean()) if len(x) else np.nan


def squad_table(P: PlayerData, cid: int, games: list[int]) -> pd.DataFrame:
    """Per player over the window: starts, minutes, modal position, starts in the last game."""
    idx, order = [], []
    for i, g in enumerate(games):
        sl = P.slices.get((cid, int(g)))
        if sl is not None:
            idx.append(np.arange(*sl))
            order.append(np.full(sl[1] - sl[0], i))
    if not idx:
        return pd.DataFrame()
    w = P.rows.iloc[np.concatenate(idx)].assign(order=np.concatenate(order))
    last = w.loc[w["order"] == w["order"].max()]
    agg = w.groupby("player_id").agg(starts=("started", "sum"), minutes=("minutes_played", "sum"))
    pos = (w.loc[w["position"] != "UNK"].groupby(["player_id", "position"]).size().rename("n").reset_index()
           .sort_values(["player_id", "n"], ascending=[True, False]).drop_duplicates("player_id")
           .set_index("player_id")["position"])
    agg["position"] = pos.reindex(agg.index).fillna("UNK")
    agg["last_start"] = agg.index.isin(set(last.loc[last["started"] > 0, "player_id"]))
    agg["score"] = agg["starts"] + agg["minutes"] / (90.0 * len(games) * 10)
    return agg.sort_values(["score"], ascending=False)


def rate_players(P: PlayerData, cid: int, tab: pd.DataFrame, d: np.datetime64) -> pd.DataFrame:
    tab = tab.copy()
    tab["group"] = tab["position"].map(T.POS_GROUP).fillna("UNK")
    tab["value"] = [P.value(int(p), d) for p in tab.index]
    base = P.club_gd_rate(cid, d)
    r = [P.per90(int(p), d, g) for p, g in zip(tab.index, tab["group"])]
    tab["ga90"] = [x[0] for x in r]
    tab["form"] = [x[1] for x in r]
    # on-pitch impact: goal difference per start above the club's own rate, shrunk with 10 prior starts
    tab["impact"] = [(x[2][0] - x[2][1] * base) / (x[2][1] + IMPACT_PRIOR) for x in r]
    age = [(d - P.dob[p]) / np.timedelta64(365, "D") if p in P.dob else np.nan for p in tab.index]
    tab["age"] = np.asarray(age, float)
    tab["wide"] = tab["position"].isin(T.WIDE)
    tab["club_gd"] = base
    return tab


def team_features(tab: pd.DataFrame, stability: float | None = None) -> dict:
    xi, bench = tab.iloc[:XI], tab.iloc[XI:XI + BENCH]
    top = tab.iloc[:XI + BENCH]
    w = top["minutes"].to_numpy(float)
    v = top["value"].to_numpy(float)
    ok = np.isfinite(v) & (w > 0)
    out = {
        "squad_value": float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan,
        "xi_value": _mean(xi["value"]),
        "att_value": _mean(xi.loc[xi["group"] == "ATT", "value"]),
        "mid_value": _mean(xi.loc[xi["group"] == "MID", "value"]),
        "def_value": _mean(xi.loc[xi["group"].isin(["DEF", "GK"]), "value"]),
        "bench_value": _mean(bench["value"]),
        "xi_ga90": float(xi["ga90"].sum()),
        "form": float(xi["form"].sum()),
        "impact": np.nan,
        "wide_value": _mean(xi.loc[xi["wide"], "value"]),
        "wide_n": float(xi["wide"].sum()),
        "last_xi_share": float(xi["last_start"].mean()) if len(xi) else np.nan,
        "xi_age": _mean(xi["age"]),
        "value_cov": float(np.isfinite(xi["value"].to_numpy(float)).mean()) if len(xi) else 0.0,
    }
    if len(xi):
        out["impact"] = float(xi["impact"].mean())
    if stability is not None:
        out["stability"] = stability
    return out


def stability(P: PlayerData, cid: int, games: list[int]) -> float:
    sets = []
    for g in games[-6:]:
        sl = P.slices.get((cid, int(g)))
        if sl is not None:
            r = P.rows.iloc[sl[0]:sl[1]]
            sets.append(set(r.loc[r["started"] > 0, "player_id"]))
    pairs = [len(a & b) / XI for a, b in zip(sets[:-1], sets[1:]) if a and b]
    return float(np.mean(pairs)) if pairs else np.nan


def dynamic_features(P: PlayerData, cid: int, d: pd.Timestamp) -> dict | None:
    dd = np.datetime64(d, "ns")
    games = P.window(cid, dd)
    if len(games) < 5:
        return None
    tab = squad_table(P, cid, games)
    if len(tab) < XI:
        return None
    tab = rate_players(P, cid, tab, dd)
    return team_features(tab, stability(P, cid, games))


def preseason_features(P: PlayerData, cid: int, s: pd.Timestamp) -> dict | None:
    """Squad on 1 July: last 10 club games before it, minus departures, plus arrivals (by 1 July)."""
    dd = np.datetime64(s, "ns")
    games = P.window(cid, dd)
    if len(games) < 5:
        return None
    tab = squad_table(P, cid, games)
    if len(tab) < XI:
        return None
    first = P.club_hist[cid][0][np.searchsorted(P.club_hist[cid][0], dd) - len(games)]
    t = P.transfers
    recent = t.loc[(t["transfer_date"] > pd.Timestamp(first)) & (t["transfer_date"] <= s)]
    last_move = recent.sort_values("transfer_date").drop_duplicates("player_id", keep="last").set_index("player_id")
    gone = [p for p in tab.index if p in last_move.index and last_move.at[p, "to_club_id"] != cid]
    total_min = tab["minutes"].sum()
    stay = tab.drop(index=gone)
    arrivals = [int(p) for p in last_move.index[(last_move["to_club_id"] == cid)] if p not in tab.index]
    stay = rate_players(P, cid, stay, dd)
    out = team_features(stay)
    out["continuity"] = float(stay["minutes"].sum() / total_min) if total_min > 0 else np.nan
    av = [P.value(p, dd) for p in arrivals]
    out["arrivals_value"] = _mean(av)
    out["arrivals_n"] = float(len(arrivals))
    # squad value at 1 July: top 18 by market value of the players still there plus arrivals
    vals = np.asarray([v for v in list(stay["value"]) + av if v == v], float)
    out["squad_value"] = float(np.sort(vals)[::-1][:XI + BENCH].mean()) if len(vals) else np.nan
    return {k: out.get(k, np.nan) for k in PRE_COLS}


def season_start(season: int) -> pd.Timestamp:
    return pd.Timestamp(f"{season}-07-01")


def build(fd: pd.DataFrame, mapping: pd.DataFrame, log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dynamic per match side, pre-season per club season) for the covered leagues."""
    clubs = set(mapping["club_id"].astype("int64"))
    P = PlayerData(clubs)
    if log:
        log.info("player data: %d club games, %d player rows, %d players with values", len(P.club_games),
                 len(P.rows), len(P.values))
    key = mapping.set_index(["league", "team"])["club_id"].astype("int64")
    m = fd.loc[fd["league"].isin(T.LEAGUE_MAP)]
    dyn, cache = [], {}
    for r in m.itertuples(index=False):
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            cid = key.get((r.league, team))
            row = {"match_id": r.match_id, "side": side, "club_id": cid}
            if cid is not None and r.season <= 2025:
                k = (cid, r.date)
                if k not in cache:
                    # information up to two days before the match date: a provider may date the same game a day
                    # earlier (time zones), and that game must never be in its own window
                    cache[k] = dynamic_features(P, int(cid), r.date - pd.Timedelta(days=1))
                if cache[k] is not None:
                    row.update(cache[k])
            dyn.append(row)
    dyn = pd.DataFrame(dyn)
    pre = []
    for season in sorted(m["season"].unique()):
        teams = m.loc[m["season"] == season]
        for lg, team in set(zip(teams["league"], teams["home_team"])) | set(zip(teams["league"], teams["away_team"])):
            cid = key.get((lg, team))
            f = preseason_features(P, int(cid), season_start(season)) if cid is not None else None
            pre.append({"league": lg, "team": team, "season": season, "club_id": cid, **(f or {})})
    pre = pd.DataFrame(pre)
    return dyn, pre


def wide(fd: pd.DataFrame, dyn: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """One row per match: h_sq_* / a_sq_* / diff_sq_* (dynamic) and h_pre_* / a_pre_* / diff_pre_* (pre-season)."""
    out = fd[["match_id", "league", "season", "home_team", "away_team"]].copy()
    for side in ("h", "a"):
        d = dyn.loc[dyn["side"] == side].set_index("match_id").reindex(columns=DYN_COLS)
        d.columns = [f"{side}_sq_{c}" for c in d.columns]
        out = out.join(d, on="match_id")
        team = "home_team" if side == "h" else "away_team"
        p = pre.set_index(["league", "team", "season"]).reindex(columns=PRE_COLS)
        p.columns = [f"{side}_pre_{c}" for c in p.columns]
        out = out.join(p, on=["league", team, "season"])
    for c in DYN_COLS:
        out[f"diff_sq_{c}"] = out[f"h_sq_{c}"] - out[f"a_sq_{c}"]
    for c in PRE_COLS:
        out[f"diff_pre_{c}"] = out[f"h_pre_{c}"] - out[f"a_pre_{c}"]
    return out.drop(columns=["league", "season", "home_team", "away_team"])
