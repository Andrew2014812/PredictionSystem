"""Dynamic attack / defence ratings, updated after every match day from earlier matches only.

Per league L and team t (a team that moves to another league starts a new rating there):

    log λ_home = μ^H_L + att_h − def_a  (+ δatt^H_h − δdef^A_a  in the home/away variant)
    log λ_away = μ^A_L + att_a − def_h  (+ δatt^A_a − δdef^H_h)

After a match day every match of the day updates the ratings with the Poisson score (y − λ):
att_h += η·(hg − λ_home), def_a −= η·(hg − λ_home), and the same for the away goals; the league levels move with
η_L; the home/away deviations δ with η_HA (only the side the team played). The pre-match values of a match use
only the match days BEFORE its date (matches of the same day never see each other). At the start of a new season
of the league the ratings shrink towards 0 by the factor (1 − r); a team new in the league starts at the mean
rating of the teams that left the league after the previous season (promoted ≈ relegated), 0 in the first season.

η, r and η_HA are chosen on the seasons 2023/24–2025/26 (Poisson deviance of home + away goals); 2021/22–2022/23
are the warm-up. The season 2026/27 is never used to choose them.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from experiments.v6_3.recent_weighting import lib as L

COLS = ["match_id", "league", "season", "date", "home_team", "away_team", "played", "home_goals", "away_goals"]
SELECT_SEASONS = (2023, 2024, 2025)


def compute(eta: float, r: float, eta_ha: float = 0.0, eta_l: float = 0.002, data: pd.DataFrame | None = None
            ) -> pd.DataFrame:
    """Pre-match ratings and rating-implied goal means of every match (played or not) of the feature table."""
    d = (data if data is not None else L.features()[COLS]).sort_values(["date", "match_id"]).reset_index(drop=True)
    att, dfn, dah, ddh, daa, dda = {}, {}, {}, {}, {}, {}
    mu_h, mu_a = {}, {}
    season_of = {}
    # rosters (fixture lists are known before a season starts)
    roster = {}
    for (L_, S_), g in d.groupby(["league", "season"]):
        roster[(L_, S_)] = set(g["home_team"]) | set(g["away_team"])
    out = np.full((len(d), 6), np.nan)
    lg = d["league"].to_numpy()
    se = d["season"].to_numpy()
    ht, at = d["home_team"].to_numpy(), d["away_team"].to_numpy()
    hg, ag = d["home_goals"].to_numpy(float), d["away_goals"].to_numpy(float)
    played = d["played"].astype(bool).to_numpy() & np.isfinite(hg) & np.isfinite(ag)
    dates = d["date"].to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], len(d)]
    for s0, s1 in zip(starts, ends):
        # season transitions (before predicting the day): shrink, then rate the newcomers
        for i in range(s0, s1):
            L_, S_ = lg[i], se[i]
            if season_of.get(L_) == S_:
                continue
            old = roster.get((L_, season_of.get(L_)), set())
            for t in old:
                k = (L_, t)
                if k in att:
                    att[k] *= (1 - r); dfn[k] *= (1 - r)
                    for dd in (dah, ddh, daa, dda):
                        if k in dd:
                            dd[k] *= (1 - r)
            new_roster = roster[(L_, S_)]
            left = [(L_, t) for t in old - new_roster if (L_, t) in att]
            pa = float(np.mean([att[k] for k in left])) if left else 0.0
            pd_ = float(np.mean([dfn[k] for k in left])) if left else 0.0
            for t in new_roster:
                if (L_, t) not in att:
                    att[(L_, t)], dfn[(L_, t)] = pa, pd_
            season_of[L_] = S_
            mu_h.setdefault(L_, math.log(1.5)); mu_a.setdefault(L_, math.log(1.15))
        # predictions of the day
        for i in range(s0, s1):
            kh, ka = (lg[i], ht[i]), (lg[i], at[i])
            eh = mu_h[lg[i]] + att[kh] - dfn[ka] + dah.get(kh, 0.0) - dda.get(ka, 0.0)
            ea = mu_a[lg[i]] + att[ka] - dfn[kh] + daa.get(ka, 0.0) - ddh.get(kh, 0.0)
            out[i] = (math.exp(eh), math.exp(ea), att[kh], dfn[kh], att[ka], dfn[ka])
        # updates after the day
        for i in range(s0, s1):
            if not played[i]:
                continue
            kh, ka, L_ = (lg[i], ht[i]), (lg[i], at[i]), lg[i]
            rh, ra = hg[i] - out[i, 0], ag[i] - out[i, 1]
            rh, ra = max(-3.0, min(3.0, rh)), max(-3.0, min(3.0, ra))
            att[kh] += eta * rh; dfn[ka] -= eta * rh
            att[ka] += eta * ra; dfn[kh] -= eta * ra
            mu_h[L_] += eta_l * rh; mu_a[L_] += eta_l * ra
            if eta_ha:
                dah[kh] = dah.get(kh, 0.0) + eta_ha * rh; dda[ka] = dda.get(ka, 0.0) - eta_ha * rh
                daa[ka] = daa.get(ka, 0.0) + eta_ha * ra; ddh[kh] = ddh.get(kh, 0.0) - eta_ha * ra
    res = pd.DataFrame(out, columns=["r_lh", "r_la", "r_h_att", "r_h_def", "r_a_att", "r_a_def"])
    res.insert(0, "match_id", d["match_id"].to_numpy())
    return res


def deviance(r: pd.DataFrame, seasons=SELECT_SEASONS) -> float:
    f = L.features()[["match_id", "season", "played", "home_goals", "away_goals"]]
    j = f.merge(r, on="match_id")
    j = j.loc[j["season"].isin(seasons) & j["played"].astype(bool) & j["home_goals"].notna()]
    return float((L.poisson_dev(j["home_goals"], j["r_lh"]).mean() + L.poisson_dev(j["away_goals"], j["r_la"]).mean())
                 / 2)


def select(log=None) -> dict:
    """Grid search of η, r (and η_HA for the home/away variant) on 2023/24–2025/26."""
    data = L.features()[COLS]
    grid = []
    for eta, r in itertools.product((0.01, 0.02, 0.03, 0.05, 0.08), (0.0, 0.25, 0.5)):
        dv = deviance(compute(eta, r, data=data))
        grid.append({"eta": eta, "r": r, "eta_ha": 0.0, "deviance": dv})
        if log:
            log.info("ratings eta=%s r=%s: %.5f", eta, r, dv)
    best = min(grid, key=lambda g: g["deviance"])
    for eta_ha in (0.005, 0.01, 0.02):
        dv = deviance(compute(best["eta"], best["r"], eta_ha, data=data))
        grid.append({"eta": best["eta"], "r": best["r"], "eta_ha": eta_ha, "deviance": dv})
        if log:
            log.info("ratings HA eta_ha=%s: %.5f", eta_ha, dv)
    best_ha = min([g for g in grid if g["eta_ha"] > 0], key=lambda g: g["deviance"])
    return {"grid": grid, "best": best, "best_ha": best_ha, "selection_seasons": list(SELECT_SEASONS)}
