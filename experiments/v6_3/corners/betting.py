"""Corner markets at real prices and the effect on Main, in both decision universes of the v6.3 replay.

Prices: the candidate tables of the shared replay (``replay/cache/candidates_{close,t09}.parquet``): for every
2026/27 match the production market mathematics, the real price of each selection known at the decision moment
(OddsPapi bet365 corner lines 8.5 / 9.5 / 10.5; close = last quote before kick-off, t09 = quotes known at T) and
the settlement.

A variant only changes the probabilities of the CORNERS_* rows (P(over) from its distribution, P(under) = 1 − P(over),
EV = p · odds − 1 as production computes it for half lines). Then
* corner strategies — flat stake on corner selections with EV ≥ t, or the single best-EV corner selection of a match;
* Main — production's rule (``evaluate.main_by_rule``, unchanged parameters) on the modified candidate table.
Why not ``engine.batch_from_outputs``: the production engine takes one μ per match and ONE global NB size, so a
per-league / per-match dispersion, a COM-Poisson, a copula convolution or a calibration map cannot be expressed
through it. For the variants that it can express (μ + one global NB size) the two routes are checked to give the
same Main (``self_check_engine``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

UNIVERSES = ("close", "t09")


def load_cands(u: str) -> pd.DataFrame:
    return pd.read_parquet(C.SHARED_CACHE / f"candidates_{u}.parquet")


def baseline_b(u: str) -> pd.DataFrame:
    return pd.read_parquet(C.SHARED_CACHE / f"baseline_b_{u}.parquet")


def baseline_a() -> pd.DataFrame:
    return pd.read_parquet(C.SHARED_CACHE / "baseline_a.parquet")


def with_probs(cands: pd.DataFrame, pover: pd.DataFrame | None) -> pd.DataFrame:
    """Candidate table with the corner probabilities of a variant. ``pover``: match_id x line columns (8.5, …).
    None removes the corner selections (Main without corners)."""
    c = cands.copy()
    isc = c["group"] == "CORNERS"
    if pover is None:
        return c.loc[~isc].reset_index(drop=True)
    line = c.loc[isc, "market"].str.replace("CORNERS_", "").astype(float)
    p = pover.set_index("match_id")
    vals = np.array([p.at[m, l] if m in p.index else np.nan for m, l in zip(c.loc[isc, "match_id"], line)])
    vals = np.where(c.loc[isc, "selection"] == "OVER", vals, 1 - vals)
    keep_old = np.isnan(vals)
    newp = np.where(keep_old, c.loc[isc, "probability"].to_numpy(float), vals)
    c.loc[isc, "probability"] = newp
    odds = c.loc[isc, "odds"].to_numpy(float)
    c.loc[isc, "ev"] = np.where(np.isnan(odds), np.nan, newp * odds - 1.0)
    return c


def corner_rows(c: pd.DataFrame) -> pd.DataFrame:
    r = c.loc[(c["group"] == "CORNERS") & c["odds"].notna() & c["status"].isin(["WON", "LOST", "PUSH"])].copy()
    r["line"] = r["market"].str.replace("CORNERS_", "").astype(float)
    return r


STRATEGIES = {"ev0": ("all", 0.0), "ev5": ("all", 0.05), "best0": ("best", 0.0)}


def strategy(rows: pd.DataFrame, kind: str, t: float) -> pd.DataFrame:
    s = rows.loc[rows["ev"] >= t]
    if kind == "best":
        s = s.sort_values(["match_id", "ev"], ascending=[True, False]).drop_duplicates("match_id")
    return s


def per_match(sel: pd.DataFrame, ids) -> tuple[np.ndarray, np.ndarray]:
    g = sel.groupby("match_id").agg(profit=("profit", "sum"), stake=("bet", "sum"))
    g = g.reindex(ids).fillna(0.0)
    return g["profit"].to_numpy(), g["stake"].to_numpy()


def corner_roi(cands_by_variant: dict[str, pd.DataFrame], ref: str) -> dict:
    """ROI of the corner strategies for every variant, paired with the same strategy on the reference variant."""
    out = {}
    ids = sorted(set(corner_rows(cands_by_variant[ref])["match_id"]))
    for name, c in cands_by_variant.items():
        rows = corner_rows(c)
        res = {}
        for sk, (kind, t) in STRATEGIES.items():
            sel = strategy(rows, kind, t)
            pr, st = per_match(sel, ids)
            bets = sel.loc[sel["bet"] == 1]
            r = C.roi_ci(pr, st)
            sel_ref = strategy(corner_rows(cands_by_variant[ref]), kind, t)
            pr0, st0 = per_match(sel_ref, ids)
            res[sk] = {**r, "hit": float((bets["status"] == "WON").mean()) if len(bets) else None,
                       "avg_odds": float(bets["odds"].mean()) if len(bets) else None,
                       "profit": float(bets["profit"].sum()),
                       "vs_ref": C.paired_roi_diff(pr, st, pr0, st0) if name != ref else None,
                       "by_line": {str(l): (float(g.loc[g["bet"] == 1, "profit"].sum() / max(g["bet"].sum(), 1)),
                                            int(g["bet"].sum())) for l, g in bets.groupby("line")}}
        out[name] = res
    return out


def market_quality(cands: pd.DataFrame, y: pd.Series, pover_by_variant: dict[str, pd.DataFrame]) -> dict:
    """Log loss of the margin-free bookmaker probability vs each variant on matches where both sides are priced."""
    r = cands.loc[(cands["group"] == "CORNERS") & cands["odds"].notna()]
    w = r.pivot_table(index=["match_id", "market"], columns="selection", values="odds", aggfunc="first").dropna()
    w = w.reset_index()
    w["line"] = w["market"].str.replace("CORNERS_", "").astype(float)
    w["q"] = (1 / w["OVER"]) / (1 / w["OVER"] + 1 / w["UNDER"])
    w["margin"] = 1 / w["OVER"] + 1 / w["UNDER"] - 1
    w["y"] = w["match_id"].map(y)
    w = w.loc[w["y"].notna()]
    out = {}
    for line, g in w.groupby("line"):
        o = (g["y"] > line).astype(float).to_numpy()
        res = {"n": int(len(g)), "margin": float(g["margin"].mean()),
               "market_ll": float(_ll(o, g["q"].to_numpy()))}
        for name, pv in pover_by_variant.items():
            p = g["match_id"].map(pv.set_index("match_id")[line]).to_numpy(float)
            ok = np.isfinite(p)
            res[f"{name}_ll"] = float(_ll(o[ok], p[ok]))
        out[str(line)] = res
    return out


def _ll(o, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.mean(-(o * np.log(p) + (1 - o) * np.log(1 - p)))


def main_table(c: pd.DataFrame, name: str) -> pd.DataFrame:
    return EV.main_by_rule(c, method=name)


def played(main: pd.DataFrame) -> pd.DataFrame:
    return main.loc[main["status"].isin(["WON", "LOST", "PUSH", "UNKNOWN"])]


def main_summary(main: pd.DataFrame, base: pd.DataFrame, a: pd.DataFrame) -> dict:
    m = played(main)
    return {"summary": EV.summary(m), "roi_ci": EV.roi_with_ci(m), "vs_b": EV.compare(m, base),
            "vs_a": EV.compare(m, a), "corner_mains": int((m["group"] == "CORNERS").sum()),
            "corner_main_bets": int(((m["group"] == "CORNERS") & (m["bet"] == 1)).sum()),
            "corner_main_profit": float(m.loc[(m["group"] == "CORNERS") & (m["bet"] == 1), "profit"].sum())}


def robustness(main: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Dependence on single big wins and sensitivity to the odds."""
    m, b = played(main), played(base)
    out = {}
    bets = m.loc[m["bet"] == 1].sort_values("profit", ascending=False)
    for k in (1, 5, 10):
        rest = bets.iloc[k:]
        out[f"roi_without_top{k}"] = float(rest["profit"].sum() / len(rest)) if len(rest) else None
    for cut in (0.98, 0.95):
        def roi(t):
            bb = t.loc[t["bet"] == 1]
            prof = np.where(bb["status"] == "WON", bb["odds"] * cut - 1, -1.0)
            return float(prof.sum() / len(bb)) if len(bb) else None
        out[f"roi_odds_x{cut}"] = roi(m)
        out[f"base_roi_odds_x{cut}"] = roi(b)
    return out


def self_check_engine(matches: pd.DataFrame, mu: pd.DataFrame, r: float, universe_prices: str,
                      cands_variant: pd.DataFrame) -> dict:
    """Main from engine.batch_from_outputs (new μ + global NB size) vs Main from the modified candidate table."""
    from experiments.v6_3.replay import data as RD
    from experiments.v6_3.replay import engine as EN
    out = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    out = out.drop(columns="corner_mu").merge(mu.rename(columns={"mu": "corner_mu"}), on="match_id", how="left")
    sample = matches.iloc[::8]
    prices = pd.read_parquet(C.SHARED_CACHE / f"prices_{universe_prices}.parquet")
    batch = EN.batch_from_outputs(sample, out, RD.lookup(prices), "corners_check", extras={"corners": {"nb_size": r}})
    eng = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
    mine = EV.main_by_rule(cands_variant.loc[cands_variant["match_id"].isin(set(sample["match_id"]))])
    j = eng.merge(mine[["match_id", "market", "selection"]], on="match_id", suffixes=("_engine", "_table"))
    same = (j["market_engine"] == j["market_table"]) & (j["selection_engine"] == j["selection_table"])
    cm = batch.markets.loc[batch.markets["market"].str.startswith("CORNERS")]
    cv = cands_variant.loc[cands_variant["group"] == "CORNERS", ["match_id", "market", "selection", "probability"]]
    k = cm.merge(cv, on=["match_id", "market", "selection"], suffixes=("_engine", "_table"))
    return {"matches": int(len(j)), "same_main_share": float(same.mean()),
            "max_abs_corner_prob_diff": float((k["probability_engine"] - k["probability_table"]).abs().max())}
