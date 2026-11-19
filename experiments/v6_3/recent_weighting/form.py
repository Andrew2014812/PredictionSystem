"""Variant 8 — match age combined with the current form of the teams.

The model part carries the age weighting (a half-life scheme refitted monthly); the current form enters as a
shrunk correction of the model's own out-of-sample errors on the team's earlier matches of the SAME season:

    A_t = (goals scored by t + k) / (goals the model expected t to score + k)          (attack)
    D_t = (goals conceded by t + k) / (goals the model expected t to concede + k)      (defence)
    λ_home' = λ_home · A_home · D_away,     λ_away' = λ_away · A_away · D_home
    p_1X2' ∝ p_1X2 · P_Poisson(1X2 | λ') / P_Poisson(1X2 | λ)
    C_t = (corners in t's matches + k_c) / (model's expected corners + k_c),   μ' = μ · sqrt(C_home · C_away)

Only matches of the season dated BEFORE the predicted match (earlier days) enter; their model values are the
walk-forward predictions (made by models fitted before their own cut-off), so nothing is in-sample. k and k_c
(prior strength in goals / corners) are chosen on 2025/26 and applied unchanged to 2026/27.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import lib as L

K_GRID = (3, 6, 12, 25, 50, 100)
KC_GRID = (15, 30, 60, 120, 240, 480)


def team_sums(season: int, out: pd.DataFrame) -> pd.DataFrame:
    """Per match: the home / away teams' season-to-date sums (earlier days only) of actual and expected values."""
    m = L.season_rows(season)[["match_id", "date", "home_team", "away_team", "played", "home_goals", "away_goals",
                                "total_corners"]]
    m = m.merge(out[["match_id", "lam_home", "lam_away", "corner_mu"]], on="match_id", how="left")
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)
    keys = ["gf", "xgf", "ga", "xga", "c", "xc"]
    acc: dict[str, dict] = {}
    rows = np.zeros((len(m), 12))
    dates = m["date"].to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], len(m)]
    for s0, s1 in zip(starts, ends):
        for i in range(s0, s1):
            h, a = acc.get(m.at[i, "home_team"], {}), acc.get(m.at[i, "away_team"], {})
            rows[i, :6] = [h.get(k, 0.0) for k in keys]
            rows[i, 6:] = [a.get(k, 0.0) for k in keys]
        for i in range(s0, s1):
            if not m.at[i, "played"] or m.at[i, "home_goals"] != m.at[i, "home_goals"]:
                continue
            hg, ag, lh, la = m.at[i, "home_goals"], m.at[i, "away_goals"], m.at[i, "lam_home"], m.at[i, "lam_away"]
            c, mu = m.at[i, "total_corners"], m.at[i, "corner_mu"]
            for team, gf, xgf, ga, xga in ((m.at[i, "home_team"], hg, lh, ag, la), (m.at[i, "away_team"], ag, la, hg, lh)):
                t = acc.setdefault(team, {k: 0.0 for k in keys})
                t["gf"] += gf; t["xgf"] += xgf; t["ga"] += ga; t["xga"] += xga
                if c == c and mu == mu:
                    t["c"] += c; t["xc"] += mu
    cols = [f"h_{k}" for k in keys] + [f"a_{k}" for k in keys]
    return pd.concat([m[["match_id"]], pd.DataFrame(rows, columns=cols)], axis=1)


def poisson_1x2(lh, la) -> np.ndarray:
    mats = L.poisson_matrices(lh, la)
    from src.markets.distributions import outcome_masks
    return np.stack([mats[:, msk].sum(1) for msk in outcome_masks(L.MAXG + 1)], 1)


def apply(out: pd.DataFrame, sums: pd.DataFrame, k: float, kc: float) -> pd.DataFrame:
    o = out.merge(sums, on="match_id", how="left").reset_index(drop=True)
    att_h = (o["h_gf"] + k) / (o["h_xgf"] + k)
    att_a = (o["a_gf"] + k) / (o["a_xgf"] + k)
    def_h = (o["h_ga"] + k) / (o["h_xga"] + k)
    def_a = (o["a_ga"] + k) / (o["a_xga"] + k)
    lh2 = o["lam_home"] * att_h * def_a
    la2 = o["lam_away"] * att_a * def_h
    p = o[["p_home", "p_draw", "p_away"]].to_numpy(float)
    ok = np.isfinite(lh2) & np.isfinite(la2)
    ratio = np.ones_like(p)
    if ok.any():
        ratio[ok] = poisson_1x2(lh2[ok], la2[ok]) / np.clip(poisson_1x2(o.loc[ok, "lam_home"], o.loc[ok, "lam_away"]),
                                                          1e-9, None)
    p2 = p * ratio
    p2 = p2 / p2.sum(1, keepdims=True)
    ch = (o["h_c"] + kc) / (o["h_xc"] + kc)
    ca = (o["a_c"] + kc) / (o["a_xc"] + kc)
    res = out[["match_id"]].copy().reset_index(drop=True)
    res["p_home"], res["p_draw"], res["p_away"] = p2[:, 0], p2[:, 1], p2[:, 2]
    res["lam_home"], res["lam_away"] = lh2.to_numpy(), la2.to_numpy()
    res["corner_mu"] = (o["corner_mu"] * np.sqrt(ch * ca)).to_numpy()
    return res
