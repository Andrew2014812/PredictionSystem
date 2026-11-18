"""Candidate features of the meta-selection model (one row per priced Main-eligible candidate).

Pool of a match: every selection the production rule may choose (not European handicap, not exact score, model
probability in [0.40, 0.85]) that has a real price at the decision moment. Features (all known before the match):

* model: logit pc (production model), logit pc_cal (study 2 primary map: for 2026/27 fitted on 2025/26 + months
  before the match's month; for the 2025/26 training rows leave-one-month-out), EV at the price (raw / calibrated)
* price: log odds, de-margined market probability q (full book of the market: q = (1/o) / Σ(1/o) · k, k = 2 for
  double chance; otherwise 1/o divided by the group's median overround) and its flag, edge = logit pc − logit q
* market: group, side (home / away / draw / no draw / over / under / yes / no), line, P(push)
* match: model 1X2 probabilities, P(over 2.5), P(corners over 9.5), number of candidates (all / priced),
  production rule score and whether production would pick this row, month of the season
* statistics availability: min(season games of both teams), head-to-head count, min(last-10 count)
* market reliability before the month: hit-rate gap (mean pc − hit), ROI and log count of the group's pool rows
  settled in earlier months (2025/26 earlier months, then 2025/26 + 2026/27 earlier months)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.calibration import data as D
from experiments.v6_3.calibration import rule as R
from experiments.v6_3.calibration.maps import logit

STUDY = "meta_selection"
DIRS = C.study_dirs(STUDY)
BAND = (0.40, 0.85)
CAL_CACHE = C.V63 / "calibration" / "cache"
SIDE = {"H": "home", "1X": "home", "A": "away", "X2": "away", "D": "draw", "12": "nodraw", "OVER": "over",
        "UNDER": "under", "YES": "yes", "NO": "no"}
GROUP_FEATS = [f"g_{g}" for g in D.MAIN_GROUPS]
SIDE_FEATS = [f"s_{s}" for s in sorted(set(SIDE.values()))]
NUM_FEATS = ["lp", "lp_cal", "lodds", "lq", "q_full", "edge", "edge_cal", "ev_raw", "ev_cal", "line", "push",
             "pH", "pD", "pA", "p_over25", "p_c95", "ncand", "ncand_priced", "rule_score", "is_rule_main",
             "season_month", "min_games", "h2h_n", "min_form10", "rel_gap", "rel_roi", "rel_logn"]
FEATURES = NUM_FEATS + GROUP_FEATS + SIDE_FEATS


def _line(market: pd.Series) -> pd.Series:
    return pd.to_numeric(market.str.extract(r"_([+-]?\d+(?:\.\d+)?)$")[0], errors="coerce").fillna(0.0)


def market_prob(c: pd.DataFrame) -> pd.DataFrame:
    inv = 1 / c["odds"].astype(float)
    g = c.groupby(["match_id", "market"])
    n_sel = g["selection"].transform("size")
    n_pr = g["odds"].transform("count")
    s = inv.groupby([c["match_id"], c["market"]]).transform("sum")
    k = np.where(c["group"] == "DOUBLE_CHANCE", 2.0, 1.0)
    full = (n_sel == n_pr) & c["odds"].notna()
    over = (s / k).where(full)
    med = over.groupby(c["group"]).transform("median").fillna(1.05)
    q = np.where(full, inv / s * k, inv / med)
    return pd.DataFrame({"q": np.clip(q, 1e-3, 0.999), "q_full": full.astype(float)}, index=c.index)


def match_feats(c: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    def pick(market, sel):
        x = c.loc[(c["market"] == market) & (c["selection"] == sel)].set_index("match_id")["probability"]
        return x
    f = pd.DataFrame({"pH": pick("1X2", "H"), "pD": pick("1X2", "D"), "pA": pick("1X2", "A"),
                      "p_over25": pick("TOTAL_2.5", "OVER"), "p_c95": pick("CORNERS_9.5", "OVER")})
    e = c.loc[c["group"].isin(D.MAIN_GROUPS) & c["probability"].between(*BAND)]
    f["ncand"] = e.groupby("match_id").size()
    f["ncand_priced"] = e.loc[e["odds"].notna()].groupby("match_id").size()
    f = f.join(stats, how="left")
    return f.fillna({"ncand": 0, "ncand_priced": 0})


def stats_table(match_ids: set) -> pd.DataFrame:
    f = pd.read_parquet(C.DATA_FILES["features"], columns=["match_id", "h_season_games", "a_season_games", "h2h_n",
                                                           "h_form10_n", "a_form10_n"])
    f = f.loc[f["match_id"].isin(match_ids)].drop_duplicates("match_id").set_index("match_id")
    return pd.DataFrame({"min_games": f[["h_season_games", "a_season_games"]].min(1), "h2h_n": f["h2h_n"],
                         "min_form10": f[["h_form10_n", "a_form10_n"]].min(1)})


def build(c: pd.DataFrame, pc_cal: np.ndarray, season_start: str) -> pd.DataFrame:
    """Pool rows with features (without the reliability block, see ``add_reliability``)."""
    c = c.copy()
    c["pc_cal"] = pc_cal
    c = c.join(market_prob(c))
    sc = R.rule_scores(c)
    c["rule_score"] = sc["score"].to_numpy()
    main_idx = R.choose(c)
    c["is_rule_main"] = c.index.isin(main_idx).astype(float)
    mf = match_feats(c, stats_table(set(c["match_id"])))
    pool = c.loc[c["group"].isin(D.MAIN_GROUPS) & c["probability"].between(*BAND) & c["odds"].notna()].copy()
    pool = pool.join(mf, on="match_id")
    push = pool["push_probability"].to_numpy(float)
    odds = pool["odds"].to_numpy(float)
    pool["lp"] = logit(pool["pc"])
    pool["lp_cal"] = logit(pool["pc_cal"])
    pool["lodds"] = np.log(odds)
    pool["lq"] = logit(pool["q"])
    pool["edge"] = pool["lp"] - pool["lq"]
    pool["edge_cal"] = pool["lp_cal"] - pool["lq"]
    pool["ev_raw"] = np.clip(pool["probability"] * odds - (1 - push), -0.5, 0.5)
    pool["ev_cal"] = np.clip(pool["pc_cal"] * (1 - push) * odds - (1 - push), -0.5, 0.5)
    pool["line"] = _line(pool["market"])
    pool["push"] = push
    start = pd.Period(season_start, "M")
    pool["season_month"] = [(pd.Period(mm, "M") - start).n for mm in pool["month"]]
    for g in D.MAIN_GROUPS:
        pool[f"g_{g}"] = (pool["group"] == g).astype(float)
    side = pool["selection"].map(SIDE).fillna("other")
    for s in sorted(set(SIDE.values())):
        pool[f"s_{s}"] = (side == s).astype(float)
    pool["profit_u"] = np.where(pool["status"] == "WON", odds - 1, np.where(pool["status"] == "LOST", -1.0, 0.0))
    pool.loc[~pool["status"].isin(["WON", "LOST", "PUSH"]), "profit_u"] = np.nan
    return pool


def add_reliability(pool: pd.DataFrame) -> pd.DataFrame:
    """Group reliability known before each row's month: computed on the rows of earlier months of ``pool``
    (``pool`` is the chronological concatenation of what may be known: 2025/26, then 2026/27 of the universe)."""
    stats = []
    for m in sorted(pool["month"].unique()):
        prev = pool.loc[(pool["month"] < m) & pool["y"].notna()]
        g = prev.groupby("group").agg(rel_n=("y", "size"), rel_pc=("pc", "mean"), rel_hit=("y", "mean"),
                                      rel_roi=("profit_u", "mean")).reset_index()
        stats.append(g.assign(month=m))
    st = pd.concat(stats, ignore_index=True)
    out = pool.merge(st, on=["month", "group"], how="left")
    out["rel_gap"] = (out["rel_pc"] - out["rel_hit"]).fillna(0.0)
    out["rel_roi"] = out["rel_roi"].fillna(0.0)
    out["rel_logn"] = np.log1p(out["rel_n"].fillna(0.0))
    return out.drop(columns=["rel_pc", "rel_hit", "rel_n"])
