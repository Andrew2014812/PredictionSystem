"""ИССЛЕДОВАНИЕ 4 — PROFITABILITY BY MARKET: which markets does FootPredict predict profitably, and should the
market priority of the Main change?

    python -m experiments.v6_3.main_selection.prepare      # shared enriched candidate tables (once)
    python -m experiments.v6_3.market_profitability.run

Part 1 (descriptive): ROI / calibration of every market group — flat over all Main-eligible priced candidates and
over production's Main — in 2025/26 (backtest) and 2026/27 (close, t09), with bootstrap intervals.
Part 2 (replay): the season replayed under market-selection rules — forced priority of one group, a soft boost,
excluding a group, and rules driven by the HISTORICAL profitability of markets (expanding window by day, seeded
with 2025/26; empirical-Bayes shrinkage; lower confidence bound; exclusion of significantly losing cells).
Every variant is one Main per match; paired with Baseline B of the same universe; Holm over the variants.
Parameters (β, k, cell, source) are chosen on 2025/26 or by expanding window over 2026/27 months.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.main_selection import evaluation as E
from experiments.v6_3.main_selection import prepare as P
from experiments.v6_3.main_selection import rules as R
from experiments.v6_3.main_selection import run as S1

from . import history as H

NAME = "market_profitability"
DIRS = C.study_dirs(NAME)
RES, FIG = DIRS["results"], DIRS["figures"]
PROD = R.PRODUCTION
GROUPS = ["1X2", "TOTAL", "HANDICAP", "DOUBLE_CHANCE", "BTTS", "CORNERS"]
MAIN_GROUPS = frozenset(GROUPS)


# ------------------------------------------------------------------------------------------------ history columns
HIST_SPECS = []          # (id, source, cell, k_fixed, kind, beta, seed)
for src in ("cand", "main"):
    for cell in ("group", "market", "group_odds"):
        for k in (None, 200.0):
            for beta in (3.0, 10.0):
                HIST_SPECS.append((f"eb_{src}_{cell}_{'kEB' if k is None else 'k200'}_b{beta:g}", src, cell, k,
                                   "eb", beta, True))
for cell in ("group", "market"):
    for beta in (3.0, 10.0):
        HIST_SPECS.append((f"lcb_cand_{cell}_b{beta:g}", "cand", cell, None, "lcb", beta, True))
    for src in ("cand", "main"):
        HIST_SPECS.append((f"exclude_losing_{src}_{cell}", src, cell, None, "exclude", None, True))
HIST_SPECS.append(("eb_cand_group_kEB_b3_noseed", "cand", "group", None, "eb", 3.0, False))


def multiplier(t: pd.DataFrame, kind: str, beta: float | None) -> np.ndarray:
    m = t["m"].to_numpy(float)
    if kind == "eb":
        return np.exp(beta * (t["roi_shr"].to_numpy(float) - m))
    raw, se, n = t["roi_raw"].to_numpy(float), t["se"].to_numpy(float), t["n"].to_numpy(float)
    if kind == "lcb":
        lcb = np.where(n > 0, raw - 1.645 * se, m - 0.3)
        return np.exp(beta * np.clip(np.nan_to_num(lcb - m, nan=-0.3), -0.3, 0.3))
    if kind == "exclude":                      # significantly worse than the pool on >= 100 bets
        bad = (n >= 100) & (raw + 1.96 * se < m)
        return np.where(bad, np.nan, 1.0)
    raise ValueError(kind)


def add_history(data: dict[str, pd.DataFrame], base: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """mult_<id> columns in every universe; bt uses its own expanding history (no seed)."""
    snap = {}
    for sid, src, cell, k, kind, beta, seed in HIST_SPECS:
        ev_bt = H.events(data["bt"], src, cell, base["bt"])
        for u, c in data.items():
            if u == "bt":
                tab = H.expanding(c, ev_bt, None, cell, k)
            else:
                tab = H.expanding(c, H.events(c, src, cell, base[u]), ev_bt if seed else None, cell, k)
            if sid in ("eb_cand_group_kEB_b3", "eb_main_group_kEB_b3") and u != "bt":
                tab.assign(spec=sid, universe=u).to_csv(RES / f"history_path_{sid}_{u}.csv", index=False)
            t = H.attach(c, tab, cell)
            c[f"mult_{sid}"] = multiplier(t, kind, beta)
            if kind == "eb" and beta == 3.0 and u != "bt":
                last = tab.loc[tab["date"] == tab["date"].max()]
                first = tab.loc[tab["date"] == tab["date"].min()]
                snap[f"{sid}|{u}"] = {"first_day": first[["cell", "roi_shr", "roi_raw", "n"]].to_dict("records"),
                                      "last_day": last[["cell", "roi_shr", "roi_raw", "n"]].to_dict("records"),
                                      "k_first": float(first["k"].iloc[0]), "k_last": float(last["k"].iloc[0])}
    return data, snap


# ------------------------------------------------------------------------------------------------ families
def families() -> dict[str, list[R.Rule]]:
    f = {}
    f["priority"] = [PROD] + [PROD.with_(name=f"priority_{g}", priority=(g,)) for g in GROUPS]
    f["boost"] = [PROD] + [R.with_reliability(PROD, f"boost_{g}", **{g: R.PROD_RELIABILITY[g] * 1.15})
                           for g in GROUPS]
    f["exclude"] = [PROD] + [PROD.with_(name=f"without_{g}", groups=MAIN_GROUPS - {g}) for g in GROUPS] + [
        PROD.with_(name="only_1X2_TOTAL_HANDICAP", groups=frozenset({"1X2", "TOTAL", "HANDICAP"})),
        PROD.with_(name="only_1X2", groups=frozenset({"1X2"}))]
    hist = {"eb": [PROD], "lcb": [PROD], "exclude_losing": [PROD]}
    for sid, src, cell, k, kind, beta, seed in HIST_SPECS:
        r = PROD.with_(name=sid, multiplier=lambda c, col=f"mult_{sid}": c[col].to_numpy(float))
        fam = "eb" if kind == "eb" else ("lcb" if kind == "lcb" else "exclude_losing")
        hist[fam].append(r)
    f["history_eb"] = hist["eb"]
    f["history_lcb"] = hist["lcb"]
    f["history_exclude"] = hist["exclude_losing"]
    return f


# ------------------------------------------------------------------------------------------------ descriptive
def group_profile(data: dict[str, pd.DataFrame], base: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for u, c in data.items():
        ok = (~c["excluded"]) & c["has_odds"] & c["status"].isin(["WON", "LOST"]) & c["probability"].between(0.4, 0.85)
        cand = c.loc[ok]
        main = base[u].loc[base[u]["bet"] == 1]
        for g in GROUPS + ["ALL"]:
            for src, t in (("cand", cand), ("main", main)):
                x = t if g == "ALL" else t.loc[t["group"] == g]
                if x.empty:
                    rows.append({"universe": u, "group": g, "source": src, "bets": 0})
                    continue
                ci = C.roi_ci(x["profit"].to_numpy(float), np.ones(len(x)))
                won = (x["status"] == "WON").to_numpy(float)
                # probability on the no-refund scale: PUSH rows are not in x, so p / (1 - P(push)) is comparable
                pc = x["probability"] / np.clip(1 - x["push_probability"], 1e-6, None)
                qc = x["p_mkt"] / np.clip(1 - x["push_probability"], 1e-6, None)
                rows.append({"universe": u, "group": g, "source": src, "bets": int(len(x)), "roi": ci["roi"],
                             "ci_lo": ci["ci95"][0], "ci_hi": ci["ci95"][1], "hit_rate": float(won.mean()),
                             "avg_p": float(pc.mean()), "gap": float((pc - won).mean()),
                             "avg_odds": float(x["odds"].mean()), "mean_ev": float(x["ev"].mean()),
                             "margin": float((1 / x["odds"] - qc).mean())})
    return rows


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    log = C.setup_logging(NAME, DIRS["logs"] / "run.log")
    data = P.load()
    g0 = E.Grid(data)
    data, snap = add_history(data, g0.base)
    log.info("history columns: %d specs", len(HIST_SPECS))
    g = E.Grid(data)
    base_a = pd.read_parquet(C.SHARED_CACHE / "baseline_a.parquet")
    base_a = base_a.loc[base_a["status"].isin(["WON", "LOST", "PUSH"])]
    profile = group_profile(data, g.base)

    fams = families()
    idx = {r.name: r for rules in fams.values() for r in rules}
    grid = []
    for fam, rules in fams.items():
        for r in rules:
            if r.name == "production" and fam != "priority":
                continue
            for u in ("bt", "close", "t09"):
                grid.append(g.row(r, u, fam))
    gdf = pd.DataFrame(grid)
    for u in ("close", "t09"):
        sub = gdf.loc[(gdf["universe"] == u) & (gdf["rule"] != "production")]
        gdf.loc[sub.index, "p_holm_grid"] = pd.Series(C.holm(sub["p"].to_dict()))
    gdf.to_csv(RES / "grid.csv", index=False)
    n_variants = int(gdf.loc[(gdf["universe"] == "close") & (gdf["rule"] != "production"), "rule"].nunique())
    log.info("grid: %d variants", n_variants)

    cands, bt_choices, ew_choices = {}, {}, {}
    for fam, rules in fams.items():
        best, res = g.bt_choice(rules)
        bt_choices[fam] = {"chosen": best.name, "members": res}
        if best.name != "production":
            cands[f"bt:{fam}"] = {"kind": f"chosen on 2025/26 ({best.name})", "rule": best.name,
                                  "mains": {u: g.run(best, (u,))[u] for u in ("close", "t09")}}
        ew = {u: g.expanding_choice(rules, u, f"ew:{fam}") for u in ("close", "t09")}
        ew_choices[fam] = {u: ew[u][1] for u in ew}
        cands[f"ew:{fam}"] = {"kind": "expanding window", "rule": None, "mains": {u: ew[u][0] for u in ew}}
    # a priori: the specification's default history rule (group cells, flat candidate ROI, EB k, beta 3, seeded)
    for name in ("eb_cand_group_kEB_b3", "lcb_cand_group_b3", "exclude_losing_cand_group"):
        cands[name] = {"kind": "a priori", "rule": name, "mains": {u: g.run(idx[name], (u,))[u] for u in ("close", "t09")}}

    details, table = {}, []
    for name, cd in cands.items():
        details[name] = {"kind": cd["kind"], "rule": cd["rule"]}
        for u, m in cd["mains"].items():
            d = E.describe(m, g.base[u], {"a": base_a})
            details[name][u] = d
            table.append(S1.compact(name, cd["kind"], u, d))
    for u in ("close", "t09"):
        sub = [r for r in table if r["universe"] == u]
        adj = C.holm({i: r["p"] for i, r in enumerate(sub)})
        for i, r in enumerate(sub):
            r["p_holm"] = adj.get(i)

    # Main ROI by market under every simple rule (effect of the rule on the market mix and on each market)
    mix = []
    for fam in ("priority", "exclude", "boost"):
        for r in fams[fam]:
            for u in ("close", "t09", "bt"):
                m = g.run(r, (u,))[u]
                b = m.loc[m["bet"] == 1]
                for grp, x in b.groupby("group"):
                    mix.append({"rule": r.name, "universe": u, "group": grp, "bets": int(len(x)),
                                "roi": float(x["profit"].mean()), "share": float(len(x) / max(len(b), 1))})

    # per-match tables + coverage of the a-priori history rule and the best pre-chosen candidate
    pm = {"baseline_b": {u: g.base[u] for u in ("close", "t09")},
          "eb_cand_group_kEB_b3": cands["eb_cand_group_kEB_b3"]["mains"]}
    rank = sorted([r for r in table if r["universe"] == "close" and r["candidate"] not in pm],
                  key=lambda r: -(r["roi_diff"] or -9))
    best_name = rank[0]["candidate"] if rank else None
    if best_name:
        pm[best_name] = cands[best_name]["mains"]
    cov = []
    for name, mm in pm.items():
        for u, m in mm.items():
            S1.per_match(m, f"{name}_{u}").to_csv(RES / f"per_match_{name.replace(':', '_')}_{u}.csv", index=False)
            cov += S1.coverage(g, {name: m}, u)
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)

    C.write_json(RES / "market_profitability.json", {
        "n_variants_grid": n_variants, "n_candidates": len(cands), "families": {k: [r.name for r in v]
                                                                                   for k, v in fams.items()},
        "profile": profile, "history_snapshots": snap, "bt_choices": bt_choices, "ew_choices": ew_choices,
        "candidates": table, "best_on_close_among_candidates": best_name, "mix": mix,
        "baselines": {u: E.describe(g.base[u], None) for u in ("close", "t09", "bt")}})
    C.write_json(RES / "candidates_detail.json", details)
    C.write_json(RES / "run_info.json", C.run_info(NAME, started, {
        "history": "expanding by day (bets settled on earlier days), seeded with the 2025/26 backtest",
        "cells": ["group", "market", "group_odds"], "odds_bands": H.ODDS_BANDS,
        "shrinkage": "empirical Bayes k = s2 / tau2 or fixed k = 200", "betas": [3, 10],
        "exclusion": "n >= 100 and ROI + 1.96 se < pooled ROI", "n_boot": C.N_BOOT, "seed": C.SEED}))
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
