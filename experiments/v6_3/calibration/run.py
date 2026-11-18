"""Study 2 (v6.3) — model probability calibration and its effect on the Main choice.

    experiments/v6_3/heavy.sh 3600 experiments/v6_3/calibration/logs/run.log \
        .venv/bin/python -m experiments.v6_3.calibration.run

Steps (every map is fitted on matches settled before the matches it is applied to):
 0. self-checks: the vectorised rule equals ``evaluate.main_by_rule`` on every match of both universes.
 1. choice of the primary map on 2025/26 only (walk-forward inside the backtest season, months Sep–May).
 2. probability quality on 2026/27 of every method x scope x fit source (backtest frozen / expanding by month
    inside 2026/27 / pooled = backtest + earlier 2026/27 months), per market group.
 3. Main chosen by the production rule on calibrated probabilities (pre-selection), both universes, paired with
    Baseline B; Holm over the variants.
 4. winner's curse: over-estimation of the chosen Main versus the number of candidates (observed, and a
    controlled experiment drawing k random candidates per match).
 5. robustness of the primary variant, per-match tables, coverage, preregistration.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

from . import data as D
from . import maps as MP
from . import metrics as MT
from . import rule as R

RES, CACHE = D.DIRS["results"], D.DIRS["cache"]
SOURCES = ["backtest", "expanding", "pooled"]
BAND = (0.40, 0.85)
K_SUBSETS = [1, 2, 3, 4, 6, 8, 12, 16, 99]
REPS = 20


def vname(method, scope, source):
    return f"{method}|{scope}|{source}"


# ------------------------------------------------------------------------------------------------ fitting
def fit_season(c26: pd.DataFrame, bt: pd.DataFrame, method: str, scope: str, source: str) -> np.ndarray:
    """Calibrated pc of every 2026/27 row; maps for month M use only matches settled before the 1st of M."""
    if method == "raw":
        return MP.Raw().predict(c26["pc"])
    if source == "backtest":
        return MP.ScopedMap(method, scope).fit(bt).predict(c26)
    out = np.full(len(c26), np.nan)
    for month in sorted(c26["month"].unique()):
        rows = (c26["month"] == month).to_numpy()
        train = c26.loc[(c26["month"] < month) & c26["y"].notna()]
        if source == "pooled":
            train = pd.concat([bt, train], ignore_index=True)
        if len(train) < MP.MIN_FIT:
            out[rows] = MP.Raw().predict(c26.loc[rows, "pc"])
            continue
        out[rows] = MP.ScopedMap(method, scope).fit(train).predict(c26.loc[rows])
    return out


def band_rows(d: pd.DataFrame) -> np.ndarray:
    return (d["group"].isin(D.MAIN_GROUPS) & d["probability"].between(*BAND)).to_numpy()


def select_on_backtest(bt: pd.DataFrame, log) -> dict:
    """Walk-forward inside 2025/26: fit on months < M, score month M (Main-eligible rows in the band)."""
    months = sorted(bt["month"].unique())[1:]
    ev = bt.loc[band_rows(bt) & bt["y"].notna()]
    rows = []
    for method in MP.METHODS:
        for scope in (["global"] if method == "raw" else MP.SCOPES):
            ll, br, n = 0.0, 0.0, 0
            for month in months:
                test = ev.loc[ev["month"] == month]
                train = bt.loc[(bt["month"] < month) & bt["y"].notna()]
                pc = MP.Raw().predict(test["pc"]) if method == "raw" else \
                    MP.ScopedMap(method, scope).fit(train).predict(test)
                ll += MT.log_loss(pc, test["y"]) * len(test)
                br += MT.brier(pc, test["y"]) * len(test)
                n += len(test)
            rows.append({"method": method, "scope": scope, "log_loss": ll / n, "brier": br / n, "n": n})
            log.info("selection 2025/26 %s/%s: log loss %.5f", method, scope, ll / n)
    t = pd.DataFrame(rows).sort_values("log_loss")
    best = t.loc[t["method"] != "raw"].iloc[0]
    return {"table": t.to_dict("records"), "months": months, "primary": {"method": best["method"],
                                                                         "scope": best["scope"]}}


def lomo_backtest(bt: pd.DataFrame, method: str, scope: str) -> np.ndarray:
    """Leave-one-month-out calibrated pc of the backtest rows (training features of the meta-model only)."""
    out = np.full(len(bt), np.nan)
    for month in bt["month"].unique():
        rows = (bt["month"] == month).to_numpy()
        train = bt.loc[~rows & bt["y"].notna().to_numpy()]
        out[rows] = MP.ScopedMap(method, scope).fit(train).predict(bt.loc[rows])
    return out


# ------------------------------------------------------------------------------------------------ quality
def quality(c26: pd.DataFrame, cal: dict[str, np.ndarray]) -> dict:
    played = c26["y"].notna().to_numpy()
    d = c26.loc[played]
    band = band_rows(d)
    out = {}
    for name, pc in cal.items():
        pc = pc[played]
        rec = {"all_main_groups": MT.prob_metrics(pc[d["group"].isin(D.MAIN_GROUPS).to_numpy()],
                                                  d.loc[d["group"].isin(D.MAIN_GROUPS), "y"]),
               "band": MT.prob_metrics(pc[band], d.loc[band, "y"])}
        for g in D.GROUPS_ALL:
            sel = (d["group"] == g).to_numpy()
            rec[g] = MT.prob_metrics(pc[sel], d.loc[sel, "y"])
        rec["1x2_multiclass_log_loss"] = multiclass_1x2(d, pc)
        out[name] = rec
    return out


def multiclass_1x2(d: pd.DataFrame, pc: np.ndarray) -> float:
    x = d.assign(_p=pc).loc[d["market"] == "1X2"]
    w = x.pivot_table(index="match_id", columns="selection", values="_p")
    y = x.pivot_table(index="match_id", columns="selection", values="y")
    w = w.div(w.sum(1), axis=0)
    return float(-np.log(np.clip((w * y).sum(1), 1e-9, 1)).mean())


def reliability_tables(c26, bt, cal, primary) -> dict:
    out = {}
    d = c26.loc[c26["y"].notna()]
    played = c26["y"].notna().to_numpy()
    for g in D.GROUPS_ALL:
        sel = (d["group"] == g).to_numpy()
        out[g] = {"raw": MT.reliability(d.loc[sel, "pc"], d.loc[sel, "y"]).to_dict("records"),
                  "primary": MT.reliability(cal[primary][played][sel], d.loc[sel, "y"]).to_dict("records")}
        b = bt.loc[(bt["group"] == g) & bt["y"].notna()]
        out[g]["backtest_raw"] = MT.reliability(b["pc"], b["y"]).to_dict("records")
    sel = band_rows(d)
    out["band"] = {"raw": MT.reliability(d.loc[sel, "pc"], d.loc[sel, "y"]).to_dict("records"),
                   "primary": MT.reliability(cal[primary][played][sel], d.loc[sel, "y"]).to_dict("records")}
    return out


# ------------------------------------------------------------------------------------------------ Main
def with_cal(c: pd.DataFrame, c26: pd.DataFrame, pc: np.ndarray, col: str) -> pd.DataFrame:
    """Attach a calibrated probability (production scale) of the close table to a universe table."""
    key = ["match_id", "market", "selection"]
    p = c26[key].assign(**{col: MP.to_p(c26, pc)})
    return c.merge(p, on=key, how="left")


def ncand(c: pd.DataFrame) -> pd.DataFrame:
    e = c.loc[band_rows(c)]
    return pd.DataFrame({"ncand": e.groupby("match_id").size(),
                         "ncand_priced": e.loc[e["odds"].notna()].groupby("match_id").size()}).fillna(0)


def winners_curse(c: pd.DataFrame, pcal_col: str, log) -> dict:
    """k random Main-eligible candidates per match -> production rule -> stated probability − hit rate."""
    c = c.loc[c["y"].notna() | (c["status"] == "PUSH")].reset_index(drop=True)
    elig = band_rows(c)
    rng = np.random.default_rng(C.SEED)
    rows = []
    for k in K_SUBSETS:
        for rep in range(REPS if k < 99 else 1):
            r = rng.random(len(c))
            rank = pd.Series(np.where(elig, r, np.inf)).groupby(c["match_id"].to_numpy()).rank(method="first")
            mask = elig & (rank.to_numpy() <= k)
            for prob_col in ("probability", pcal_col):
                m = R.main_table(c, prob_col, mask=mask)
                m = m.loc[m["status"].isin(["WON", "LOST"])]
                pc = m["stated_p"] / (1 - m["push_probability"])
                pcal = m[pcal_col] / (1 - m["push_probability"])
                y = (m["status"] == "WON").astype(float)
                bets = m.loc[m["bet"] == 1]
                rows.append({"k": k, "rep": rep, "rule_on": "raw" if prob_col == "probability" else "calibrated",
                             "gap_stated": float(pc.mean() - y.mean()), "gap_cal": float(pcal.mean() - y.mean()),
                             "mean_p": float(pc.mean()), "hit": float(y.mean()),
                             "roi": float(bets["profit"].sum() / len(bets)) if len(bets) else None,
                             "avg_odds": float(bets["odds"].mean()) if len(bets) else None, "n": int(len(m))})
        log.info("winner's curse k=%s done", k)
    t = pd.DataFrame(rows)
    agg = t.groupby(["rule_on", "k"]).agg(gap_stated=("gap_stated", "mean"), gap_stated_sd=("gap_stated", "std"),
                                         gap_cal=("gap_cal", "mean"), gap_cal_sd=("gap_cal", "std"),
                                         hit=("hit", "mean"), mean_p=("mean_p", "mean"), roi=("roi", "mean"),
                                         roi_sd=("roi", "std"), avg_odds=("avg_odds", "mean"), n=("n", "mean"))
    return {"reps": REPS, "k": K_SUBSETS, "table": agg.reset_index().to_dict("records")}


def observed_ncand(main: pd.DataFrame, nc: pd.DataFrame, col: str, bins) -> list[dict]:
    m = main.join(nc, on="match_id")
    m = m.loc[m["status"].isin(["WON", "LOST"])]
    m["bin"] = pd.cut(m[col], bins=bins)
    rows = []
    for b, g in m.groupby("bin", observed=True):
        r = MT.gap_ci(g, "stated_p")
        rows.append({"bin": str(b), "n": r["n"], "gap": r["gap"], "gap_ci": r["gap_ci"], "mean_p": r["mean_p"],
                     "hit": r["hit"]})
    return rows


def main() -> None:
    started = time.time()
    log = C.setup_logging("calibration", D.DIRS["logs"] / "run_python.log")
    bt = D.backtest()
    c26 = D.season("close")
    univ = {"close": c26, "t09": D.season("t09")}
    base_b = {u: D.baseline(f"baseline_b_{u}") for u in univ}
    base_a = D.baseline("baseline_a")
    m = D.matches()
    res = D.results(m)
    out = {"data": {"backtest_matches": int(bt["match_id"].nunique()), "backtest_rows": int(len(bt)),
                    "backtest_settled_rows": int(bt["y"].notna().sum()),
                    "backtest_priced_share": float(bt["odds"].notna().mean()),
                    "season_matches": int(c26["match_id"].nunique()), "season_rows": int(len(c26)),
                    "season_settled_rows": int(c26["y"].notna().sum()),
                    "months_2026": sorted(c26["month"].unique()), "months_2025": sorted(bt["month"].unique())}}

    # 0. self-checks
    chk = {}
    for u, c in univ.items():
        ref = EV.main_by_rule(c)
        mine = R.main_table(c)
        j = mine.merge(ref[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_ref"))
        eng = pd.read_parquet(C.SHARED_CACHE / f"baseline_b_{u}.parquet", columns=["match_id", "market", "selection"])
        k = mine.merge(eng, on="match_id", suffixes=("", "_eng"))
        chk[u] = {"matches": int(len(j)),
                  "equal_to_main_by_rule": float(((j["market"] == j["market_ref"]) & (j["selection"] == j["selection_ref"])).mean()),
                  "equal_to_engine_baseline_b": float(((k["market"] == k["market_eng"]) & (k["selection"] == k["selection_eng"])).mean()),
                  "note": "differences with the engine are exact score ties broken in another candidate order"}
    out["self_check"] = chk
    log.info("self-check %s", chk)

    # 1. primary map chosen on 2025/26 only
    sel = select_on_backtest(bt, log)
    out["selection"] = sel
    pm, ps = sel["primary"]["method"], sel["primary"]["scope"]
    primary = vname(pm, ps, "pooled")
    log.info("primary map: %s", primary)

    # 2. every method x scope x source on 2026/27
    cal = {"raw|-|-": MP.Raw().predict(c26["pc"])}
    maps_info = {}
    for method in MP.METHODS[1:]:
        for scope in MP.SCOPES:
            for source in SOURCES:
                name = vname(method, scope, source)
                cal[name] = fit_season(c26, bt, method, scope, source)
                log.info("fitted %s", name)
    # parameters of the frozen backtest maps (for the report)
    for method in ("platt", "temperature", "beta"):
        mp = MP.ScopedMap(method, "group").fit(bt)
        maps_info[method] = {f"{lv}:{k}": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                            for kk, vv in vars(v).items()} for (lv, k), v in mp.maps.items()}
    out["frozen_backtest_maps"] = maps_info
    q = quality(c26, cal)
    out["quality"] = q
    out["quality_backtest_raw"] = {g: MT.prob_metrics(bt.loc[(bt["group"] == g) & bt["y"].notna(), "pc"],
                                                      bt.loc[(bt["group"] == g) & bt["y"].notna(), "y"])
                                   for g in D.GROUPS_ALL}
    d = c26.loc[c26["y"].notna()]
    played = c26["y"].notna().to_numpy()
    deltas = {}
    for name in (primary, vname(pm, ps, "backtest"), vname(pm, ps, "expanding")):
        deltas[name] = {}
        for g in D.MAIN_GROUPS + ["band"]:
            s = band_rows(d) if g == "band" else (d["group"] == g).to_numpy()
            deltas[name][g] = MT.per_match_delta(d.loc[s], cal[name][played][s], d.loc[s, "pc"].to_numpy())
    out["paired_log_loss_vs_raw"] = deltas
    out["reliability"] = reliability_tables(c26, bt, cal, primary)
    pd.DataFrame({k: v.astype(np.float32) for k, v in cal.items()}).assign(
        match_id=c26["match_id"], market=c26["market"], selection=c26["selection"]).to_parquet(
        CACHE / "calibrated_2026.parquet")
    bt_oof = lomo_backtest(bt, pm, ps)
    bt[["match_id", "market", "selection"]].assign(pc_cal_lomo=bt_oof).to_parquet(CACHE / "backtest_cal_lomo.parquet")
    C.write_json(CACHE / "primary.json", {"primary": primary, "method": pm, "scope": ps})

    # 3. Main on calibrated probabilities
    family = [vname(mm, "group", "pooled") for mm in ("platt", "temperature", "beta", "isotonic", "histbin")]
    family += [vname("beta", "global", "pooled"), vname("beta", "key", "pooled"), vname(pm, ps, "backtest"),
               vname(pm, ps, "expanding"), primary]
    family = list(dict.fromkeys(family))
    out["family"] = family
    main_res, mains = {}, {}
    nc = ncand(c26)
    for u, c in univ.items():
        main_res[u] = {}
        b = base_b[u].assign(stated_p=lambda x: x["probability"].astype(float),
                             ev_stated=lambda x: pd.to_numeric(x["ev"], errors="coerce"))
        main_res[u]["baseline_b"] = MT.main_eval(b, base_b[u], base_a)
        mains[(u, "baseline_b")] = b
        for name in family:
            cc = with_cal(c, c26, cal[name], "p_cal")
            mm = R.main_table(cc, "p_cal", method=name)
            mm = mm.loc[mm["match_id"].isin(set(b["match_id"]))]
            mains[(u, name)] = mm
            main_res[u][name] = MT.main_eval(mm, base_b[u], base_a)
            log.info("%s %s: %s", u, name, MT.short(main_res[u][name]))
        pv = {n: main_res[u][n]["vs_b"]["roi_diff"]["p"] for n in family}
        main_res[u]["holm"] = C.holm(pv)
    out["main"] = main_res
    out["baseline_a"] = MT.main_eval(base_a.assign(stated_p=base_a["probability"].astype(float),
                                                   ev_stated=pd.to_numeric(base_a["ev"], errors="coerce")),
                                     base_b["close"])
    # Main gap of B judged by the primary map (which part of the over-estimation is marginal miscalibration)
    cc = with_cal(c26, c26, cal[primary], "p_cal")
    bb = base_b["close"].merge(cc[["match_id", "market", "selection", "p_cal"]], on=["match_id", "market", "selection"])
    out["b_gap_decomposition"] = {"stated": MT.gap_ci(bb.assign(stated_p=bb["probability"].astype(float))),
                                  "primary_map": MT.gap_ci(bb.assign(stated_p=bb["p_cal"]))}

    # 4. winner's curse
    out["ncand_observed"] = {
        "close_ncand": observed_ncand(mains[("close", "baseline_b")], nc, "ncand", [0, 12, 15, 17, 19, 30]),
        "close_ncand_priced": observed_ncand(mains[("close", "baseline_b")], nc, "ncand_priced", [-1, 12, 15, 17, 19, 30]),
        "primary_close_ncand": observed_ncand(mains[("close", primary)], nc, "ncand", [0, 12, 15, 17, 19, 30])}
    btm = R.main_table(bt)
    ncb = ncand(bt)
    out["ncand_observed"]["backtest_ncand_priced"] = observed_ncand(btm, ncb, "ncand_priced", [-1, 0, 1, 2, 3, 30])
    out["ncand_observed"]["backtest_all"] = MT.gap_ci(btm)
    out["ncand_summary"] = {"close": nc.describe().to_dict(), "backtest": ncb.describe().to_dict()}
    out["winners_curse"] = winners_curse(cc, "p_cal", log)

    # 5. robustness, per-match tables, coverage
    out["robustness"] = {u: {"baseline_b": MT.robustness(mains[(u, "baseline_b")], base_b[u]),
                             primary: MT.robustness(mains[(u, primary)], base_b[u])} for u in univ}
    for u in univ:
        for name, tag in (("baseline_b", "baseline_b"), (primary, "primary")):
            t = mains[(u, name)].copy()
            raw_p = t["probability"].astype(float)
            if name != "baseline_b":
                t["probability"], t["ev"] = t["stated_p"], t["ev_stated"]
            pm_t = EV.per_match_table(MT.with_teams(t, m), res, f"{tag}:{name}" if name != "baseline_b" else name, "prod_20261012")
            pm_t["raw_probability"] = raw_p.to_numpy()
            pm_t.to_csv(RES / f"per_match_{tag}_{u}.csv", index=False)
    out["coverage"] = {u: MT.coverage(m, {"baseline_b": mains[(u, "baseline_b")], primary: mains[(u, primary)]})
                       for u in univ}
    out["primary"] = primary
    C.write_json(RES / "calibration.json", out)
    C.write_json(RES / "run_info.json", C.run_info("v6_3/calibration", started, {
        "band": BAND, "min_fit": MP.MIN_FIT, "methods": MP.METHODS, "scopes": MP.SCOPES, "sources": SOURCES,
        "k_subsets": K_SUBSETS, "reps": REPS, "primary_chosen_on": "2025/26 backtest walk-forward (months Sep-May)",
        "fit_rule": "maps for month M of 2026/27 use backtest 2025/26 and/or 2026/27 matches before the 1st of M"}))
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
