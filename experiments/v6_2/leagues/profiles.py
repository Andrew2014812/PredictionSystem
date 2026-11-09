"""League profiles and groups of similar leagues.

Profile of a league for fold s = averages over its matches of seasons < s (results and match statistics only,
no prices): goals per match, home share of goals, draw rate, home advantage (home-win rate − away-win rate),
over 2.5 rate, both-teams-score rate, corners, shots and shots on target per match (tempo proxies), goals per
shot on target (finishing), yellow cards per match, competitive balance (s.d. of the teams' points per game within a
league-season). Profiles are standardised across leagues and grouped by Ward hierarchical clustering; the number
of groups k ∈ {3..6} maximises the silhouette. Groups are recomputed for every fold from its own past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import common as C
from . import wf

VARS = {"goals": "голов за матч", "home_share": "доля голов хозяев", "draw": "доля ничьих",
        "home_adv": "перевес хозяев (П1 − П2)", "over25": "ТБ 2.5", "btts": "обе забьют", "corners": "угловых за матч",
        "shots": "ударов за матч", "sot": "ударов в створ за матч", "finishing": "голов на удар в створ",
        "yellows": "жёлтых за матч", "balance": "разброс очков за матч (с.о.)"}
K_RANGE = range(3, 7)


def profile(rows: pd.DataFrame) -> pd.DataFrame:
    hg, ag = rows["home_goals"].astype(float), rows["away_goals"].astype(float)
    d = pd.DataFrame({"league": rows["league"].to_numpy(), "season": rows["season"].to_numpy(),
                      "goals": (hg + ag).to_numpy(), "hg": hg.to_numpy(), "draw": (hg == ag).astype(float).to_numpy(),
                      "hw": (hg > ag).astype(float).to_numpy(), "aw": (hg < ag).astype(float).to_numpy(),
                      "over25": ((hg + ag) > 2.5).astype(float).to_numpy(),
                      "btts": ((hg > 0) & (ag > 0)).astype(float).to_numpy(),
                      "corners": (rows["home_corners"] + rows["away_corners"]).to_numpy(float),
                      "shots": (rows["home_shots"] + rows["away_shots"]).to_numpy(float),
                      "sot": (rows["home_sot"] + rows["away_sot"]).to_numpy(float),
                      "yellows": (rows["home_yellows"] + rows["away_yellows"]).to_numpy(float)})
    g = d.groupby("league")
    out = pd.DataFrame({"goals": g["goals"].mean(), "home_share": g["hg"].sum() / g["goals"].sum(),
                        "draw": g["draw"].mean(), "home_adv": g["hw"].mean() - g["aw"].mean(),
                        "over25": g["over25"].mean(), "btts": g["btts"].mean(), "corners": g["corners"].mean(),
                        "shots": g["shots"].mean(), "sot": g["sot"].mean(),
                        "finishing": g["goals"].sum() / g["sot"].sum(), "yellows": g["yellows"].mean(),
                        "n": g.size()})
    # competitive balance: s.d. of final points per game of the teams, averaged over league-seasons
    pts = []
    for (lg, s), r in rows.groupby(["league", "season"]):
        h = np.where(r["home_goals"] > r["away_goals"], 3, np.where(r["home_goals"] == r["away_goals"], 1, 0))
        a = np.where(r["home_goals"] < r["away_goals"], 3, np.where(r["home_goals"] == r["away_goals"], 1, 0))
        t = pd.concat([pd.DataFrame({"t": r["home_team"].to_numpy(), "p": h}),
                       pd.DataFrame({"t": r["away_team"].to_numpy(), "p": a})]).groupby("t")["p"].mean()
        pts.append({"league": lg, "sd": float(t.std())})
    out["balance"] = pd.DataFrame(pts).groupby("league")["sd"].mean()
    return out


def cluster(prof: pd.DataFrame) -> tuple[dict, dict]:
    X = prof[list(VARS)].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    Z = linkage(X, "ward")
    sil = {}
    for k in K_RANGE:
        lab = fcluster(Z, k, "maxclust")
        sil[k] = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else -1
    k = max(sil, key=sil.get)
    lab = fcluster(Z, k, "maxclust")
    # stable names: groups numbered by decreasing goals per match
    order = pd.Series(prof["goals"].to_numpy()).groupby(lab).mean().sort_values(ascending=False).index
    rename = {old: f"G{i + 1}" for i, old in enumerate(order)}
    groups = {lg: rename[l] for lg, l in zip(prof.index, lab)}
    return groups, {"k": int(k), "silhouette": sil}


_GROUPS: dict[int, dict] = {}


def groups_for(s: int) -> dict:
    """{league: group} for fold s (profiles from seasons < s)."""
    if s not in _GROUPS:
        df = wf.features()
        prof = profile(df.loc[df["season"] < s])
        _GROUPS[s] = cluster(prof)[0]
    return _GROUPS[s]


def run() -> dict:
    df = wf.features()
    out = {"folds": {}, "vars": VARS}
    labels = {}
    for s in C.SEASONS:
        prof = profile(df.loc[df["season"] < s])
        groups, info = cluster(prof)
        labels[s] = groups
        out["folds"][str(s)] = {"groups": groups, **info,
                                "profile": prof.round(4).reset_index().to_dict("records")}
    # stability of the grouping between consecutive folds
    leagues = sorted(labels[C.SEASONS[0]])
    out["ari"] = {f"{a}-{b}": float(adjusted_rand_score([labels[a][x] for x in leagues], [labels[b][x] for x in leagues]))
                  for a, b in zip(C.SEASONS[:-1], C.SEASONS[1:])}
    # full-history profile (all seasons up to now, for the report's description only)
    prof = profile(df)
    groups, info = cluster(prof)
    out["all_history"] = {"groups": groups, **info, "profile": prof.round(4).reset_index().to_dict("records")}
    C.save_json("profiles.json", out)
    return out
