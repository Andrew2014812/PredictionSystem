"""Candidate tables, settlement, Main selection and the comparison of a method with the baselines.

Status / profit follow production (``src/analytics/performance``): a bet is a WON / LOST prediction with a real
price; profit = odds − 1 or −1; PUSH (refund) and VOID return the stake and are not counted as bets; a prediction
without a real price is settled but is not a bet. ROI = Σ profit / bets.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src.markets.markets import settle_selection
from src.markets.odds import PricedSelection
from src.markets.recommendation import RuleParams, main_params, main_prediction

PER_MATCH_COLS = ["match_id", "date", "league", "home_team", "away_team", "home_goals", "away_goals",
                  "total_corners", "model_version", "method", "market", "selection", "group", "probability",
                  "odds", "bookmaker", "provider", "priced_at", "ev", "status", "profit", "data_type", "note"]


@lru_cache(maxsize=None)
def _settle(market, selection, hg, ag, corners):
    return settle_selection(market, selection, hg, ag, corners)


def settle_rows(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Add status (WON / LOST / PUSH / UPCOMING / UNKNOWN) and profit to rows with match_id, market, selection, odds."""
    out = df.copy()
    r = results.reindex(out["match_id"])
    hg, ag, co = r["home_goals"].to_numpy(), r["away_goals"].to_numpy(), r["total_corners"].to_numpy()
    status = []
    for m, s, h, a, c in zip(out["market"], out["selection"], hg, ag, co):
        if h != h:
            status.append("UPCOMING")
            continue
        st = _settle(m, s, float(h), float(a), None if c != c else float(c))
        status.append(st or "UNKNOWN")
    out["status"] = status
    odds = pd.to_numeric(out["odds"], errors="coerce")
    out["profit"] = np.where(out["status"] == "WON", odds - 1, np.where(out["status"] == "LOST", -1.0, 0.0))
    out.loc[odds.isna() | ~out["status"].isin(["WON", "LOST", "PUSH"]), "profit"] = np.nan
    out["bet"] = (odds.notna() & out["status"].isin(["WON", "LOST"])).astype(int)
    return out


def candidates(markets: pd.DataFrame, matches: pd.DataFrame, results: pd.DataFrame,
               extra_prices: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Long candidate table (one row per match x selection) with settlement and match information.
    ``extra_prices`` adds e.g. closing / early prices as ``odds_<name>`` columns (diagnostics only)."""
    c = markets.copy()
    c["odds"] = pd.to_numeric(c["odds"], errors="coerce")
    c["probability"] = pd.to_numeric(c["probability"], errors="coerce")
    c["push_probability"] = pd.to_numeric(c["push_probability"], errors="coerce").fillna(0.0)
    info = matches.set_index("match_id")[["date", "league", "home_team", "away_team"]]
    c = c.join(info, on="match_id")
    for name, prices in (extra_prices or {}).items():
        p = prices.set_index(["match_id", "market", "selection"])["odds"].rename(f"odds_{name}")
        c = c.join(p, on=["match_id", "market", "selection"])
    return settle_rows(c, results)


def to_selections(group: pd.DataFrame, prob_col: str = "probability") -> list[PricedSelection]:
    return [PricedSelection(r.market, r.selection, float(getattr(r, prob_col)),
                            None if r.odds != r.odds else float(r.odds), r.bookmaker, r.provider,
                            float(r.push_probability), r.priced_at)
            for r in group.itertuples(index=False)]


def main_by_rule(cands: pd.DataFrame, params: RuleParams | None = None, prob_col: str = "probability",
                 method: str = "rule") -> pd.DataFrame:
    """One Main per match with production's Main rule (``main_prediction``) and the given parameters,
    optionally on another probability column (e.g. calibrated). Returns the chosen candidate rows."""
    params = params or main_params()
    rows = []
    for mid, g in cands.groupby("match_id", sort=False):
        rec = main_prediction(to_selections(g, prob_col), params)
        if rec is None:
            continue
        hit = g.loc[(g["market"] == rec.pick.market) & (g["selection"] == rec.pick.selection)].iloc[0]
        rows.append(hit)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["method"] = method
    return out


def main_by_score(cands: pd.DataFrame, score: pd.Series, eligible: pd.Series | None = None,
                  fallback: pd.DataFrame | None = None, method: str = "score") -> pd.DataFrame:
    """One Main per match = the eligible candidate with the highest ``score``. A match with no eligible candidate
    takes its row from ``fallback`` (e.g. the production Main) and is marked ``note = fallback``."""
    c = cands.assign(_s=score.astype(float))
    if eligible is not None:
        c = c.loc[eligible.reindex(c.index).fillna(False).astype(bool)]
    c = c.loc[c["_s"].notna()]
    best = c.sort_values(["match_id", "_s", "probability"], ascending=[True, False, False]).drop_duplicates("match_id")
    best = best.drop(columns="_s").assign(note="")
    if fallback is not None:
        miss = fallback.loc[~fallback["match_id"].isin(set(best["match_id"]))].assign(note="fallback")
        best = pd.concat([best, miss], ignore_index=True)
    best["method"] = method
    return best.reset_index(drop=True)


def summary(main: pd.DataFrame) -> dict:
    """Production-style figures of a one-Main-per-match table."""
    done = main.loc[main["status"].isin(["WON", "LOST"])]
    bets = main.loc[main["bet"] == 1]
    n_bets = len(bets)
    profit = float(bets["profit"].sum()) if n_bets else 0.0
    ev = pd.to_numeric(bets.get("ev"), errors="coerce") if n_bets else pd.Series(dtype=float)
    return {"matches": int(main["match_id"].nunique()), "settled": int(len(done)),
            "won": int((done["status"] == "WON").sum()), "push": int((main["status"] == "PUSH").sum()),
            "bets": n_bets, "priced_share": float(main["odds"].notna().mean()) if len(main) else None,
            "hit_rate": float((done["status"] == "WON").mean()) if len(done) else None,
            "profit": profit, "roi": profit / n_bets if n_bets else None,
            "avg_odds": float(bets["odds"].mean()) if n_bets else None,
            "avg_probability": float(done["probability"].mean()) if len(done) else None,
            "neg_ev_share": float((ev < 0).mean()) if n_bets else None,
            "mean_ev": float(ev.mean()) if n_bets else None}


def roi_with_ci(main: pd.DataFrame) -> dict:
    m = main.loc[main["status"].isin(["WON", "LOST", "PUSH"])]
    return C.roi_ci(m["profit"].fillna(0).to_numpy(), m["bet"].to_numpy())


def compare(new: pd.DataFrame, base: pd.DataFrame, on: str = "match_id") -> dict:
    """Paired comparison on the matches both have settled: ROI difference (paired bootstrap), hit-rate difference,
    share of matches where the Main differs."""
    a = new.set_index(on)
    b = base.set_index(on)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    settled = a["status"].isin(["WON", "LOST", "PUSH"]) & b["status"].isin(["WON", "LOST", "PUSH"])
    a, b = a.loc[settled], b.loc[settled]
    roi = C.paired_roi_diff(a["profit"].fillna(0).to_numpy(), a["bet"].to_numpy(),
                            b["profit"].fillna(0).to_numpy(), b["bet"].to_numpy())
    hit = C.paired_mean_diff((a["status"] == "WON").astype(float).to_numpy(),
                             (b["status"] == "WON").astype(float).to_numpy())
    same = (a["market"] == b["market"]) & (a["selection"] == b["selection"])
    return {"matches": int(len(a)), "roi_diff": roi, "hit_rate_diff": hit, "same_main_share": float(same.mean())}


def breakdown(main: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in main.groupby(by):
        s = summary(g)
        rows.append({by: key, **{k: s[k] for k in ("matches", "bets", "hit_rate", "roi", "profit", "avg_odds")}})
    return pd.DataFrame(rows)


def per_match_table(main: pd.DataFrame, results: pd.DataFrame, method: str, model_version: str,
                    data_type: str = "replay") -> pd.DataFrame:
    """The per-match table every study publishes (fields of Prompt 9, §5)."""
    t = main.copy()
    r = results.reindex(t["match_id"])
    for c in ("home_goals", "away_goals", "total_corners"):
        t[c] = r[c].to_numpy()
    t["method"], t["model_version"], t["data_type"] = method, model_version, data_type
    if "note" not in t:
        t["note"] = ""
    t.loc[t["odds"].isna() & (t["note"] == ""), "note"] = "no real price at decision time"
    t.loc[t["status"] == "UNKNOWN", "note"] = "result of the market unknown (e.g. corners not reported)"
    return t.reindex(columns=PER_MATCH_COLS)
