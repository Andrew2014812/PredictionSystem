"""Evaluation helpers shared by the Main-selection and market-profitability studies.

* ``Grid`` — runs a list of rules on the universes (bt / close / t09), keeps the Main tables and a per-month
  profit table, compares every rule with Baseline B (= production rule on the same universe, paired bootstrap).
* ``expanding_choice`` — the honest way to pick a parameter: for every month M of 2026/27 the family member with
  the best ROI on (2025/26 backtest, real prices only) + (2026/27 months before M) is applied to month M.
* ``describe`` — everything Prompt 9 asks for one Main table: ROI / hit / profit / odds / bets, CI, negative-EV
  share, market mix, months, leagues, dependence on big wins, sensitivity to the price, share changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

from . import rules as R

SMALL = 30           # a league / month with fewer bets is marked as a small sample


def month_table(main: pd.DataFrame) -> pd.DataFrame:
    b = main.loc[main["bet"] == 1]
    return b.groupby("month").agg(profit=("profit", "sum"), bets=("bet", "sum"))


def roi_of(tab: pd.DataFrame) -> float | None:
    n = tab["bets"].sum()
    return float(tab["profit"].sum() / n) if n else None


def sensitivity(main: pd.DataFrame) -> dict:
    b = main.loc[main["bet"] == 1].copy()
    if b.empty:
        return {}
    won = b["status"] == "WON"
    out = {}
    for h in (0.02, 0.05):
        prof = np.where(won, b["odds"] * (1 - h) - 1, -1.0)
        out[f"roi_odds_minus_{int(h * 100)}pct"] = float(prof.sum() / len(b))
    top = b.sort_values("profit", ascending=False)
    for k in (5, 10):
        rest = top.iloc[k:]
        out[f"roi_without_top{k}_wins"] = float(rest["profit"].sum() / len(b))
    out["largest_win"] = float(top["profit"].iloc[0])
    out["share_profit_odds_ge_2"] = float((b["odds"] >= 2).mean())
    return out


def describe(main: pd.DataFrame, base: pd.DataFrame | None, extra_base: dict | None = None) -> dict:
    s = EV.summary(main)
    out = {"summary": s, "roi_ci": EV.roi_with_ci(main), "sensitivity": sensitivity(main),
           "fallback": int(main["note"].fillna("").str.startswith("fallback").sum()),
           "fallback_not_bet": int((main["note"].fillna("").str.contains("not a bet")).sum()),
           "unpriced_main": int(main["odds"].isna().sum()),
           "groups": main["group"].value_counts(normalize=True).round(4).to_dict(),
           "groups_bets": main.loc[main["bet"] == 1, "group"].value_counts().to_dict()}
    if base is not None:
        out["vs_b"] = EV.compare(main, base)
        out["changed_share"] = 1 - out["vs_b"]["same_main_share"]
        both = set(main.loc[main["bet"] == 1, "match_id"]) & set(base.loc[base["bet"] == 1, "match_id"])
        out["vs_b_common_bets"] = EV.compare(main.loc[main["match_id"].isin(both)],
                                             base.loc[base["match_id"].isin(both)])
        out["gap"] = (s["avg_probability"] - s["hit_rate"]) if s["hit_rate"] is not None else None
        mb = month_table(base)
        mm = month_table(main)
        months = []
        for m in sorted(set(mb.index) | set(mm.index)):
            r_new = roi_of(mm.loc[[m]]) if m in mm.index else None
            r_old = roi_of(mb.loc[[m]]) if m in mb.index else None
            months.append({"month": m, "bets": int(mm.loc[m, "bets"]) if m in mm.index else 0,
                           "roi": r_new, "roi_b": r_old,
                           "diff": None if r_new is None or r_old is None else r_new - r_old})
        out["months"] = months
        d = [x["diff"] for x in months if x["diff"] is not None]
        out["months_better"] = int(sum(x > 0 for x in d))
        out["months_n"] = len(d)
    for k, v in (extra_base or {}).items():
        out[f"vs_{k}"] = EV.compare(main, v)
    lg = EV.breakdown(main, "league")
    lg["small"] = lg["bets"] < SMALL
    out["leagues"] = lg.to_dict("records")
    return out


def neg_ev(main: pd.DataFrame) -> float | None:
    b = main.loc[main["bet"] == 1]
    return float((b["ev"] < 0).mean()) if len(b) else None


class Grid:
    """Rules x universes. ``data`` maps universe -> candidate table (bt rows force a real price)."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data
        self.stat = {u: R.statistical_main(c) for u, c in data.items()}
        self.base = {u: self.main(R.PRODUCTION, u) for u in data}
        self.mains: dict[tuple[str, str], pd.DataFrame] = {}

    def main(self, rule: R.Rule, universe: str) -> pd.DataFrame:
        c = self.data[universe]
        if universe == "bt":                 # 2025/26: only real prices are candidates (Football-Data markets)
            rule = rule.with_(require_price=True)
        return R.select(c, rule, None if universe == "bt" else self.stat[universe])

    def run(self, rule: R.Rule, universes=("bt", "close", "t09")) -> dict[str, pd.DataFrame]:
        out = {}
        for u in universes:
            if u not in self.data:
                continue
            key = (rule.name, u)
            if key not in self.mains:
                self.mains[key] = self.main(rule, u)
            out[u] = self.mains[key]
        return out

    def row(self, rule: R.Rule, u: str, family: str = "") -> dict:
        m = self.run(rule, (u,))[u]
        s = EV.summary(m)
        cmp = EV.compare(m, self.base[u]) if u != "bt" else None
        return {"family": family, "rule": rule.name, "universe": u, "bets": s["bets"], "roi": s["roi"],
                "hit_rate": s["hit_rate"], "profit": s["profit"], "avg_odds": s["avg_odds"],
                "neg_ev_share": s["neg_ev_share"],
                "fallback": int(m["note"].fillna("").str.startswith("fallback").sum()),
                "roi_diff": cmp["roi_diff"]["diff"] if cmp else None,
                "ci_lo": cmp["roi_diff"]["ci95"][0] if cmp else None,
                "ci_hi": cmp["roi_diff"]["ci95"][1] if cmp else None,
                "p": cmp["roi_diff"]["p"] if cmp else None,
                "changed_share": 1 - cmp["same_main_share"] if cmp else None}

    def expanding_choice(self, family: list[R.Rule], u: str, name: str) -> tuple[pd.DataFrame, list[dict]]:
        """Month M of 2026/27 gets the member with the best ROI on bt + months of ``u`` before M
        (ties -> earlier member of the list, so put production first)."""
        bt = {r.name: month_table(self.run(r, ("bt",))["bt"]) for r in family}
        cur = {r.name: self.run(r, (u,))[u] for r in family}
        cur_m = {k: month_table(v) for k, v in cur.items()}
        months = sorted(self.data[u]["month"].unique())
        rows, choices = [], []
        for m in months:
            best, best_roi = None, -np.inf
            for r in family:
                hist = pd.concat([bt[r.name], cur_m[r.name].loc[cur_m[r.name].index < m]])
                roi = roi_of(hist)
                roi = -np.inf if roi is None else roi
                if roi > best_roi + 1e-12:
                    best, best_roi = r, roi
            choices.append({"month": m, "chosen": best.name, "history_roi": best_roi})
            sel = cur[best.name]
            rows.append(sel.loc[sel["month"] == m])
        out = pd.concat(rows, ignore_index=True).assign(method=name)
        return out, choices

    def bt_choice(self, family: list[R.Rule]) -> tuple[R.Rule, list[dict]]:
        """The member with the best ROI on the 2025/26 backtest (real prices only) — frozen before 2026/27."""
        res = []
        for r in family:
            s = EV.summary(self.run(r, ("bt",))["bt"])
            res.append({"rule": r.name, "roi": s["roi"], "bets": s["bets"]})
        best = max(range(len(family)), key=lambda i: (res[i]["roi"] if res[i]["roi"] is not None else -9, -i))
        return family[best], res


def holm_table(rows: list[dict], key: str = "p") -> list[dict]:
    ps = {i: r[key] for i, r in enumerate(rows) if r.get(key) is not None}
    adj = C.holm(ps)
    for i, r in enumerate(rows):
        r["p_holm"] = adj.get(i)
    return rows
