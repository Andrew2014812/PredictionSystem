"""ИССЛЕДОВАНИЕ 1 — MAIN SELECTION 3.0: one Main per match, which rule?

    python -m experiments.v6_3.main_selection.prepare     # enriched candidate tables (once)
    python -m experiments.v6_3.main_selection.run         # every experiment -> results/, figures/

Protocol
* Universes: ``close`` (primary, 2026/27, last quote before kick-off), ``t09`` (robustness, prices known at T),
  ``bt`` (2025/26 production backtest, real Football-Data prices only — parameter choice only).
* Families of rules (Prompt 9, study 1, directions 1-10 + real price mandatory); production is the first member
  of every family. A parameter is chosen WITHOUT the evaluated period in two ways:
  - ``bt``  — the member with the best ROI on the 2025/26 backtest (frozen before 2026/27);
  - ``ew``  — expanding window: month M of 2026/27 uses the member with the best ROI on the backtest plus the
              2026/27 months before M (same universe).
  These pre-chosen candidates (+ the two rules named by the specification: hard minimum price 1.40 with the
  not-a-bet fallback, real price mandatory) are the study's answer; Holm over them.
* The full grid is reported separately (every member on the whole 2026/27, Holm over all of them) — it shows
  what is possible in hindsight, not what would have been chosen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

from . import data as D
from . import evaluation as E
from . import prepare as P
from . import rules as R

DIRS = D.DIRS
RES, FIG = DIRS["results"], DIRS["figures"]
PROD = R.PRODUCTION
MODEL_VERSION = "prod_20261012"
FLOORS = (1.30, 1.40, 1.50, 1.60)


# ------------------------------------------------------------------------------------------------ families
def _max(col_fn, name):
    return PROD.with_(name=name, require_price=True, score=col_fn)


def families() -> dict[str, list[R.Rule]]:
    f: dict[str, list[R.Rule]] = {}
    f["floor"] = [PROD] + [PROD.with_(name=f"floor_{m:.2f}", min_odds=m) for m in FLOORS]
    f["price"] = [PROD, PROD.with_(name="real_price_mandatory", require_price=True)]
    f["preference"] = [PROD] + [PROD.with_(name=n, preferred=pr, softness=sf) for n, pr, sf in (
        ("pref_1.50-2.50", (1.50, 2.50), 0.15), ("pref_1.60-2.60", (1.60, 2.60), 0.15),
        ("pref_1.40-3.00", (1.40, 3.00), 0.15), ("pref_soft0.05", (1.40, 2.50), 0.05),
        ("pref_soft0.30", (1.40, 2.50), 0.30))]
    f["power"] = [PROD] + [PROD.with_(name=f"power_{w:g}", power=w) for w in (0, 0.25, 0.5, 0.75, 1.5, 2, 3)]
    f["ev_weight"] = [PROD] + [PROD.with_(name=f"evw_{w:g}", ev_weight=w) for w in (0, 0.25, 1, 2, 4)]
    f["calibrated"] = [PROD,
                       PROD.with_(name="cal_rule", prob="p_cal"),
                       PROD.with_(name="cal_evw2", prob="p_cal", ev_weight=2.0),
                       PROD.with_(name="cal_max_ev", prob="p_cal", require_price=True,
                                  score=lambda c, p, ev: ev)]
    rel = [("rel_flat", dict(TOTAL=1.0, BTTS=1.0, DOUBLE_CHANCE=1.0, HANDICAP=1.0, CORNERS=1.0)),
           ("rel_total0.90", dict(TOTAL=0.90)), ("rel_total0.95", dict(TOTAL=0.95)),
           ("rel_1x2_1.10", {"1X2": 1.10}), ("rel_handicap1.0", dict(HANDICAP=1.0)),
           ("rel_dc1.0", dict(DOUBLE_CHANCE=1.0)), ("rel_btts1.0", dict(BTTS=1.0)),
           ("rel_corners1.0", dict(CORNERS=1.0)), ("rel_corners0.70", dict(CORNERS=0.70))]
    f["reliability"] = [PROD] + [R.with_reliability(PROD, n, **g) for n, g in rel]
    f["market_prob"] = [PROD] + [PROD.with_(name=f"blend_{w}", prob=f"p_bl{w}") for w in (75, 50, 25, 0)] + [
        PROD.with_(name="rel_x_market_p", score=lambda c, p, ev: (
            c["group"].map(dict(PROD.reliability)).fillna(0.8).to_numpy(float)
            * np.where(np.isfinite(c["p_mkt"].to_numpy(float)), c["p_mkt"].to_numpy(float), p)
            * R.odds_pref(np.where(c["has_odds"], c["odds"], 1 / np.maximum(p, 1e-9)), PROD.preferred,
                          PROD.softness)
            * np.where(c["has_odds"], 1 + PROD.ev_weight * np.clip(np.nan_to_num(ev), -0.3, 0.3),
                       PROD.unpriced)))]
    f["overestimation"] = [PROD, PROD.with_(name="gap_corrected", prob="p_gap")] + [
        PROD.with_(name=f"winners_curse_{g:g}", prob=f"p_wc{int(g * 10)}") for g in P.WC_GAMMAS]
    f["ranking"] = [PROD,
                    _max(lambda c, p, ev: ev, "max_ev"),
                    _max(lambda c, p, ev: ev / np.maximum(c["odds"].to_numpy(float) - 1, 1e-6), "max_kelly"),
                    _max(lambda c, p, ev: p, "max_p_priced"),
                    _max(lambda c, p, ev: c["p_mkt"].to_numpy(float), "max_market_p"),
                    _max(lambda c, p, ev: p - c["p_mkt"].to_numpy(float), "max_edge"),
                    PROD.with_(name="value_first_ev0", value_first_min_ev=0.0),
                    PROD.with_(name="value_first_ev3", value_first_min_ev=0.03)]
    return f


DIAGNOSTIC = [PROD.with_(name="floor_1.40_silent", min_odds=1.40, fallback_bet=True),
              PROD.with_(name="floor_1.40_effective", min_eff_odds=1.40)]
A_PRIORI = ["floor_1.40", "real_price_mandatory"]


# ------------------------------------------------------------------------------------------------ helpers
def rule_index(fams) -> dict[str, R.Rule]:
    idx = {}
    for rules in fams.values():
        for r in rules:
            idx[r.name] = r
    for r in DIAGNOSTIC:
        idx[r.name] = r
    return idx


def evaluate_main(g: E.Grid, main: pd.DataFrame, u: str, base_a: pd.DataFrame) -> dict:
    return E.describe(main, g.base[u], {"a": base_a})


def compact(name: str, kind: str, u: str, d: dict) -> dict:
    s, v = d["summary"], d.get("vs_b", {})
    rd = v.get("roi_diff", {})
    return {"candidate": name, "kind": kind, "universe": u, "bets": s["bets"], "roi": s["roi"],
            "roi_ci": d["roi_ci"]["ci95"], "hit_rate": s["hit_rate"], "profit": s["profit"],
            "avg_odds": s["avg_odds"], "neg_ev_share": s["neg_ev_share"], "roi_diff": rd.get("diff"),
            "diff_ci": rd.get("ci95"), "p": rd.get("p"), "changed_share": d.get("changed_share"),
            "fallback": d["fallback"], "fallback_not_bet": d["fallback_not_bet"],
            "months_better": d.get("months_better"), "months_n": d.get("months_n"),
            "hit_diff": v.get("hit_rate_diff", {}).get("diff"), "gap": d.get("gap"),
            "common_matches": d.get("vs_b_common_bets", {}).get("matches"),
            "common_diff": d.get("vs_b_common_bets", {}).get("roi_diff", {}).get("diff"),
            "common_p": d.get("vs_b_common_bets", {}).get("roi_diff", {}).get("p")}


def coverage(g: E.Grid, mains: dict[str, pd.DataFrame], u: str) -> list[dict]:
    m = pd.read_parquet(C.SHARED_CACHE / "matches.parquet", columns=["match_id", "played"])
    rows = []
    for name, t in mains.items():
        st = t["status"]
        rows.append({"method": name, "universe": u, "season_matches": int(len(m)),
                     "played": int(m["played"].astype(bool).sum()), "processed": int(len(t)),
                     "main_with_real_price": int(t["odds"].notna().sum()), "bets": int(t["bet"].sum()),
                     "not_bet_no_price": int((t["odds"].isna() & st.isin(["WON", "LOST"])).sum()),
                     "not_bet_fallback": int(t["note"].fillna("").str.contains("not a bet").sum()),
                     "push": int((st == "PUSH").sum()), "unknown_result": int((st == "UNKNOWN").sum()),
                     "without_enough_statistics": 0})
    return rows


def per_match(main: pd.DataFrame, name: str) -> pd.DataFrame:
    res = pd.read_parquet(C.SHARED_CACHE / "matches.parquet",
                          columns=["match_id", "played", "home_goals", "away_goals", "total_corners"])
    res = res.loc[res["played"].astype(bool)].set_index("match_id")
    t = main.copy()
    t["note"] = t["note"].fillna("")
    return EV.per_match_table(t, res, name, MODEL_VERSION)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    started = time.time()
    log = C.setup_logging("main_selection", DIRS["logs"] / "run.log")
    data = P.load()
    g = E.Grid(data)
    base_a = pd.read_parquet(C.SHARED_CACHE / "baseline_a.parquet")
    base_a = base_a.loc[base_a["status"].isin(["WON", "LOST", "PUSH"])]
    fams = families()
    idx = rule_index(fams)
    log.info("%d families, %d distinct rules (+%d diagnostic)", len(fams), len(idx) - len(DIAGNOSTIC),
             len(DIAGNOSTIC))

    # self-check: the vectorised production rule == Baseline B / the stored backtest Main
    checks = {}
    for u in ("close", "t09"):
        b = pd.read_parquet(C.SHARED_CACHE / f"baseline_b_{u}.parquet")
        j = R.select(data[u], PROD).merge(b[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_b"))
        checks[u] = float(((j["market"] == j["market_b"]) & (j["selection"] == j["selection_b"])).mean())
    pk = pd.read_parquet(C.DATA_FILES["picks"], columns=["match_id", "source", "role", "market", "selection"])
    pk = pk.loc[(pk["source"] == "backtest") & (pk["role"] == "main")]
    j = R.select(data["bt"], PROD).merge(pk, on="match_id", suffixes=("", "_b"))
    checks["bt"] = float(((j["market"] == j["market_b"]) & (j["selection"] == j["selection_b"])).mean())
    log.info("self-check %s", checks)

    # ---- full grid (hindsight) on every universe
    grid = []
    for fam, rules in fams.items():
        for r in rules:
            if r.name == "production" and fam != "floor":
                continue
            for u in ("bt", "close", "t09"):
                grid.append(g.row(r, u, fam))
    for r in DIAGNOSTIC:
        for u in ("bt", "close", "t09"):
            grid.append(g.row(r, u, "diagnostic"))
    gdf = pd.DataFrame(grid)
    for u in ("close", "t09"):
        sub = gdf.loc[(gdf["universe"] == u) & (gdf["rule"] != "production") & (gdf["family"] != "diagnostic")]
        adj = C.holm(sub["p"].to_dict())
        gdf.loc[sub.index, "p_holm_grid"] = pd.Series(adj)
    gdf.to_csv(RES / "grid.csv", index=False)
    n_variants = int(gdf.loc[(gdf["universe"] == "close") & (gdf["rule"] != "production")
                             & (gdf["family"] != "diagnostic"), "rule"].nunique())
    log.info("grid: %d variants", n_variants)

    # ---- pre-chosen candidates
    cands: dict[str, dict] = {}       # name -> {kind, rule (or None), mains per universe, choices}
    for name in A_PRIORI:
        cands[name] = {"kind": "a priori (specification)", "rule": name,
                       "mains": {u: g.run(idx[name], (u,))[u] for u in ("close", "t09")}}
    bt_choices, ew_choices = {}, {}
    for fam, rules in fams.items():
        if fam == "price":
            continue
        best, res = g.bt_choice(rules)
        bt_choices[fam] = {"chosen": best.name, "members": res}
        key = f"bt:{fam}"
        if best.name != "production":
            cands[key] = {"kind": f"chosen on 2025/26 ({best.name})", "rule": best.name,
                          "mains": {u: g.run(best, (u,))[u] for u in ("close", "t09")}}
        ew = {}
        for u in ("close", "t09"):
            m, ch = g.expanding_choice(rules, u, f"ew:{fam}")
            ew[u] = (m, ch)
        ew_choices[fam] = {u: ew[u][1] for u in ew}
        cands[f"ew:{fam}"] = {"kind": "expanding window", "rule": None, "mains": {u: ew[u][0] for u in ew}}
    # combination of the members chosen on 2025/26 (Prompt 9 §17: does a combination keep the advantage?)
    combo = PROD.with_(name="combo_bt", min_odds=1.40, ev_weight=1.0,
                       reliability=R.with_reliability(PROD, "x", TOTAL=0.95).reliability)
    idx["combo_bt"] = combo
    cands["combo_bt"] = {"kind": "combination of the 2025/26 choices (floor 1.40 + EV weight 1 + TOTAL 0.95)",
                         "rule": "combo_bt", "mains": {u: g.run(combo, (u,))[u] for u in ("close", "t09")}}
    bt_combo = EV.summary(g.run(combo, ("bt",))["bt"])
    # one procedure over the whole grid (every distinct rule, production first)
    allrules = [PROD] + [r for n, r in idx.items() if n not in ("production", "combo_bt") and r not in DIAGNOSTIC]
    ew_all = {u: g.expanding_choice(allrules, u, "ew:all") for u in ("close", "t09")}
    cands["ew:all"] = {"kind": "expanding window over the whole grid", "rule": None,
                       "mains": {u: ew_all[u][0] for u in ew_all}}
    ew_choices["all"] = {u: ew_all[u][1] for u in ew_all}

    details, table = {}, []
    for name, cd in cands.items():
        details[name] = {"kind": cd["kind"], "rule": cd["rule"]}
        for u, m in cd["mains"].items():
            d = evaluate_main(g, m, u, base_a)
            details[name][u] = d
            table.append(compact(name, cd["kind"], u, d))
    for u in ("close", "t09"):
        sub = [r for r in table if r["universe"] == u]
        adj = C.holm({i: r["p"] for i, r in enumerate(sub)})
        for i, r in enumerate(sub):
            r["p_holm"] = adj.get(i)
    base_desc = {u: evaluate_main(g, g.base[u], u, base_a) for u in ("close", "t09")}
    base_desc["a"] = E.describe(base_a.assign(note="", month=pd.to_datetime(base_a["date"]).dt.strftime("%Y-%m")
                                              .replace({"2026-07": "2026-08"})), None)

    # ---- diagnostics: odds floors in detail, silent vs correct fallback
    floors = []
    for r in fams["floor"] + DIAGNOSTIC:
        for u in ("close", "t09", "bt"):
            m = g.run(r, (u,))[u]
            s = EV.summary(m)
            fb = m.loc[m["note"].fillna("").str.startswith("fallback")]
            floors.append({"rule": r.name, "universe": u, "bets": s["bets"], "roi": s["roi"],
                           "hit_rate": s["hit_rate"], "avg_odds": s["avg_odds"], "fallback": int(len(fb)),
                           "fallback_won": int((fb["status"] == "WON").sum()),
                           "fallback_avg_price": float(fb["odds"].mean()) if fb["odds"].notna().any() else None,
                           "fallback_unpriced": int(fb["odds"].isna().sum()),
                           "fallback_profit_if_bet": float(np.where(fb["status"] == "WON", fb["odds"] - 1,
                                                                    np.where(fb["status"] == "LOST", -1, 0))[
                                                               fb["odds"].notna().to_numpy()].sum())})

    # ---- odds-band profile of production's Main (why a floor could matter)
    bands = []
    for u in ("close", "t09", "bt"):
        b = g.base[u].loc[g.base[u]["bet"] == 1].copy()
        b["band"] = pd.cut(b["odds"], [1.0, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 100], right=False).astype(str)
        for k, x in b.groupby("band", observed=True):
            bands.append({"universe": u, "band": k, "bets": int(len(x)), "roi": float(x["profit"].mean()),
                          "hit_rate": float((x["status"] == "WON").mean()), "avg_p": float(x["probability"].mean()),
                          "avg_ev": float(x["ev"].mean())})

    # ---- per-match tables + coverage
    pm_methods = {"baseline_b": {u: g.base[u] for u in ("close", "t09")}}
    for name in ("floor_1.40", "real_price_mandatory", "ew:all"):
        pm_methods[name] = cands[name]["mains"]
    close_rank = sorted([r for r in table if r["universe"] == "close" and r["candidate"] not in pm_methods],
                        key=lambda r: -(r["roi_diff"] or -9))
    best_name = close_rank[0]["candidate"] if close_rank else None
    if best_name:
        pm_methods[best_name] = cands[best_name]["mains"]
    cov = []
    for name, mm in pm_methods.items():
        for u, m in mm.items():
            fname = name.replace(":", "_").replace(".", "")
            per_match(m, f"{name}_{u}").to_csv(RES / f"per_match_{fname}_{u}.csv", index=False)
        for u, m in mm.items():
            cov += coverage(g, {name: m}, u)
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)

    # ---- monthly cumulative profit (figures)
    curves = []
    for name in ["baseline_b"] + [n for n in pm_methods if n != "baseline_b"]:
        for u in ("close", "t09"):
            m = (g.base[u] if name == "baseline_b" else pm_methods[name][u])
            b = m.loc[m["bet"] == 1].sort_values(["date", "match_id"])
            curves.append(pd.DataFrame({"method": name, "universe": u, "date": b["date"].to_numpy(),
                                        "cum_profit": b["profit"].cumsum().to_numpy(),
                                        "n": np.arange(1, len(b) + 1)}))
    pd.concat(curves).to_csv(RES / "cumulative_profit.csv", index=False)

    sizes = {u: {"matches": int(c["match_id"].nunique()), "rows": int(len(c)),
                 "rows_per_match": float(len(c) / c["match_id"].nunique()),
                 "eligible_per_match": float(c["ncand"].groupby(c["mi"]).first().mean()),
                 "priced_eligible_per_match": float(c["ncand_priced"].groupby(c["mi"]).first().mean())}
             for u, c in data.items()}
    out = {"self_check_rule_equals_baseline": checks, "universe_sizes": sizes, "n_variants_grid": n_variants,
           "n_candidates": len(cands), "families": {k: [r.name for r in v] for k, v in fams.items()},
           "a_priori": A_PRIORI, "bt_choices": bt_choices, "combo_bt_on_bt": bt_combo, "ew_choices": ew_choices, "candidates": table,
           "best_on_close_among_candidates": best_name,
           "baselines": base_desc, "floors": floors, "odds_bands": bands}
    C.write_json(RES / "main_selection.json", out)
    C.write_json(RES / "candidates_detail.json", details)
    C.write_json(RES / "run_info.json", C.run_info("main_selection", started, {
        "universes": {"close": "2026/27, last quote before kick-off (primary)",
                      "t09": "2026/27, prices known at T = min(09:00 UK, kick-off - 30 min)",
                      "bt": "2025/26 production backtest (backtest_2526, trained to 2025-06-01), Football-Data "
                            "Bet365 prices for 1X2 / total 2.5 / one Asian line; only priced candidates"},
        "floors": FLOORS, "winners_curse_sigma": P.WC_SIGMA, "winners_curse_gammas": P.WC_GAMMAS,
        "calibration": "Platt by market group, monthly expanding window (bt + earlier 2026/27 months)",
        "n_boot": C.N_BOOT, "seed": C.SEED}))
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
