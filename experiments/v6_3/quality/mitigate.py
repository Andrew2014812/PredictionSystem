"""Mitigations for sparse data that keep exactly one Main per match (no match is excluded).

Sparsity of a match (known before kick-off, from the system's own features):

    s_season = clip((10 − n_min) / 10, 0, 1)      n_min    = min(h_season_games, a_season_games)
    s_hist   = clip((10 − hist_min) / 10, 0, 1)   hist_min = min(h_form10_n, a_form10_n)   (newcomer / phantom)

Variants (applied to the model outputs, then production's engine and Main rule):

    mkt_season(w)   p' = (1 − w·s_season)·p + w·s_season·q          q = de-margined 1X2 price at the decision moment
    mkt_hist(w)     p' = (1 − w·s_hist)·p + w·s_hist·q
    mkt_both        weight max(w₁·s_season, w₂·s_hist) with the chosen w₁, w₂
    mkt_uniform     the same average weight for every match (control: is targeting the sparse matches what helps?)
    prior(w)        λ' = (1 − w·s)·λ + w·s·λ_league (home / away goals, corners), s = max(s_season, s_hist)
    lowdata_rank    the Main rule without the EV term (EV weight 0) where max(s_season, s_hist) ≥ 0.5: an EV computed
                    from a poorly informed model is mostly noise

Selection (pre-declared): the weight of each family is chosen on the out-of-sample seasons 2024/25 + 2025/26
(pooled) by 1X2 log loss (market families) or by the summed Poisson deviance of home + away goals (prior), never
on 2026/27. Main ROI of the selection seasons (Football-Data Bet365 prices) is reported next to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3.alt_ml import mainroi as R
from experiments.v6_3.replay import evaluate as EV
from src.markets.recommendation import main_params

from . import data as Q

GRID = {"mkt_season": [0.25, 0.5, 0.75], "mkt_hist": [0.5, 1.0], "prior": [0.25, 0.5]}
LOWDATA = 0.5
PRIOR_COLS = ["league_avg_home_goals", "league_avg_away_goals", "league_avg_corners"]


def sparsity(fr: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=fr["match_id"])
    s["s_season"] = np.clip((10 - fr["n_min"].to_numpy(float)) / 10, 0, 1)
    s["s_hist"] = np.clip((10 - fr["hist_min"].to_numpy(float)) / 10, 0, 1)
    return s


def attach_priors(fr: pd.DataFrame) -> pd.DataFrame:
    f = Q.features().set_index("match_id")[PRIOR_COLS]
    return fr.drop(columns=[c for c in PRIOR_COLS if c in fr]).join(f, on="match_id")


def weight(fr: pd.DataFrame, name: str, w) -> np.ndarray:
    s = sparsity(fr)
    if name == "mkt_season":
        return w * s["s_season"].to_numpy()
    if name == "mkt_hist":
        return w * s["s_hist"].to_numpy()
    if name == "mkt_both":
        return np.maximum(w[0] * s["s_season"].to_numpy(), w[1] * s["s_hist"].to_numpy())
    if name == "mkt_uniform":
        return np.full(len(fr), float(w))
    if name == "prior":
        return w * np.maximum(s["s_season"].to_numpy(), s["s_hist"].to_numpy())
    raise ValueError(name)


def apply(outputs: pd.DataFrame, fr: pd.DataFrame, qcols: list[str], name: str, w) -> pd.DataFrame:
    """New outputs (same match order as ``outputs``) of variant ``name`` with parameter ``w``."""
    o = outputs.set_index("match_id").copy()
    x = fr.set_index("match_id").reindex(o.index)
    xr = x.reset_index()
    if name.startswith("mkt_"):
        wt = weight(xr, name, w)
        q = x[qcols].to_numpy(float)
        ok = np.isfinite(q).all(axis=1)
        wt = np.where(ok, wt, 0.0)[:, None]
        P = o[["p_home", "p_draw", "p_away"]].to_numpy(float)
        newp = (1 - wt) * P + wt * np.nan_to_num(q)
        newp = newp / newp.sum(axis=1, keepdims=True)
        o[["p_home", "p_draw", "p_away"]] = newp
    elif name == "prior":
        wt = weight(xr, name, w)
        for col, pc in (("lam_home", "league_avg_home_goals"), ("lam_away", "league_avg_away_goals"),
                        ("corner_mu", "league_avg_corners")):
            prior = x[pc].to_numpy(float)
            cur = o[col].to_numpy(float)
            ok = np.isfinite(prior) & np.isfinite(cur)
            o[col] = np.where(ok, (1 - wt) * cur + wt * np.nan_to_num(prior), cur)
    return o.reset_index()


def lowdata_ids(fr: pd.DataFrame) -> set:
    s = sparsity(fr)
    return set(s.index[np.maximum(s["s_season"], s["s_hist"]) >= LOWDATA])


def lowdata_main(base_main: pd.DataFrame, cands: pd.DataFrame, fr: pd.DataFrame, label: str) -> pd.DataFrame:
    """Baseline Main, except in low-data matches where the production rule runs with EV weight 0."""
    ids = lowdata_ids(fr)
    sub = cands.loc[cands["match_id"].isin(ids)]
    alt = EV.main_by_rule(sub, main_params().with_(ev_weight=0.0), method=label) if len(sub) else sub
    keep = base_main.loc[~base_main["match_id"].isin(ids)]
    out = pd.concat([keep, alt], ignore_index=True)
    out = out.loc[out["match_id"].isin(base_main["match_id"])]
    out["method"] = label
    return out.reset_index(drop=True)


def ll_1x2(outputs: pd.DataFrame, fr: pd.DataFrame) -> pd.Series:
    x = fr.set_index("match_id")
    o = outputs.set_index("match_id").reindex(x.index)
    y = x["result"].map({"H": 0, "D": 1, "A": 2})
    P = o[["p_home", "p_draw", "p_away"]].to_numpy(float)
    ok = y.notna().to_numpy()
    yi = y.fillna(0).astype(int).to_numpy()
    return pd.Series(np.where(ok, -np.log(np.clip(P[np.arange(len(x)), yi], 1e-9, 1)), np.nan), index=x.index)


def goal_dev(outputs: pd.DataFrame, fr: pd.DataFrame) -> pd.Series:
    x = fr.set_index("match_id")
    o = outputs.set_index("match_id").reindex(x.index)
    return pd.Series(Q.poisson_unit_deviance(x["home_goals"], o["lam_home"])
                     + Q.poisson_unit_deviance(x["away_goals"], o["lam_away"]), index=x.index)


def corner_dev(outputs: pd.DataFrame, fr: pd.DataFrame) -> pd.Series:
    x = fr.set_index("match_id")
    o = outputs.set_index("match_id").reindex(x.index)
    ok = x["total_corners"].notna() & o["corner_mu"].notna()
    d = Q.poisson_unit_deviance(x["total_corners"].fillna(0), o["corner_mu"].fillna(1))
    return pd.Series(np.where(ok, d, np.nan), index=x.index)


def variant_list(chosen: dict | None = None) -> list[tuple[str, str, object]]:
    """(label, family, parameter). With ``chosen`` the combined / control variants are appended."""
    out = [(f"{fam}_{w}", fam, w) for fam, ws in GRID.items() for w in ws]
    out.append(("lowdata_rank", "lowdata_rank", None))
    if chosen:
        out.append(("mkt_both", "mkt_both", (chosen["mkt_season"], chosen["mkt_hist"])))
        out.append(("mkt_uniform", "mkt_uniform", chosen["uniform_w"]))
    return out
