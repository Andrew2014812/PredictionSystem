"""The evaluation stage: every variant on the same matches, paired against B0 and against its own control.

Writes results/metrics.json (summary + paired differences + Holm + sample size), results/main.json (Main
selection, over-confidence and ROI at real prices), results/breakdowns.json (league, month, xG-history
band), results/sensitivity.json, results/corners.json and results/decisions.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from . import evaluate as E
from . import models as M

PAIRS = [("X2a", "R0a"), ("X3a", "R0a"), ("X5", "B0"), ("M1a", "R0a"), ("X2Ma", "R0a"), ("X2", "R0"), ("M1", "R0"),
         ("X1", "G1"), ("S1", "S0"), ("T1", "T0")]
SHOW_METRICS = ["goals_dev", "rmse_h", "rmse_a", "mae_h", "mae_a", "rmse_total", "logloss_1x2", "brier_1x2",
                "acc_rule", "acc_argmax", "ll_over1.5", "ll_over2.5", "ll_over3.5", "br_over2.5", "ll_btts",
                "br_btts", "ll_ah-2.5", "ll_ah-1.5", "ll_ah+1.5", "ll_ah+2.5", "ll_exact", "exact_top5"]


def load_predictions(refresh: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    df = M.frame(0)
    path = C.CACHE / "preds_in_season.parquet"
    info = C.load_json("variants_info.json", {}) or {}
    if path.exists() and not refresh:
        allp = pd.read_parquet(path)
    else:
        preds, inf = M.in_season(df)
        allp = pd.concat([p.assign(variant=v) for v, p in preds.items()], ignore_index=True)
        allp.to_parquet(path)
        info["in_season"] = inf
    sp = C.CACHE / "preds_short.parquet"
    if sp.exists() and not refresh:
        short = pd.read_parquet(sp)
    else:
        sh = M.short_history(df)
        short = pd.concat([p.assign(variant=v) for v, p in sh.items()], ignore_index=True)
        short.to_parquet(sp)
    lp = C.CACHE / "preds_long.parquet"
    if lp.exists() and not refresh:
        long_ = pd.read_parquet(lp)
    else:
        lo, mp = M.long_history(df)
        long_ = pd.concat([p.assign(variant=v) for v, p in lo.items()], ignore_index=True)
        long_.to_parquet(lp)
        info["proxy_mapping"] = mp
    C.save_json("variants_info.json", info)
    allp = pd.concat([allp, short, long_], ignore_index=True)
    preds = {v: g.drop(columns="variant") for v, g in allp.groupby("variant")}
    return df, preds, info


def _bet365_reference(ev: pd.DataFrame) -> dict:
    """Log loss of the Bet365 pre-match 1X2 / over 2.5 prices (proportional de-margining) on the subset of the
    evaluation matches that have them — a market benchmark, not a variant."""
    f = D.season_features(columns=["match_id", "odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25"])
    e = ev.merge(f, on="match_id", how="left")
    ok = e[["odds_home", "odds_draw", "odds_away"]].notna().all(1)
    q = 1 / e.loc[ok, ["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    q = q / q.sum(1, keepdims=True)
    y = e.loc[ok, "y"].to_numpy(int)
    ll = -np.log(q[np.arange(len(y)), y])
    o = e[["odds_over25", "odds_under25"]].notna().all(1)
    qo = (1 / e.loc[o, "odds_over25"]) / (1 / e.loc[o, "odds_over25"] + 1 / e.loc[o, "odds_under25"])
    tot = (e.loc[o, "home_goals"] + e.loc[o, "away_goals"] > 2.5).astype(float)
    llo, _ = C.binary_loss(tot, qo)
    return {"n_1x2": int(ok.sum()), "logloss_1x2": float(ll.mean()), "ids_1x2": e.loc[ok, "match_id"].tolist(),
            "n_over": int(o.sum()), "ll_over2.5": float(llo.mean()), "ids_over": e.loc[o, "match_id"].tolist()}


def run(refresh: bool = False) -> dict:
    t = C.Timer()
    df, preds, info = load_predictions(refresh)
    ev, al = E.assemble(df, preds)
    variants = [v for v in M.VARIANTS if v in al]
    pms = {v: E.per_match(al[v], ev) for v in variants}
    pd.concat([pm.assign(variant=v) for v, pm in pms.items()]).to_parquet(C.CACHE / "per_match.parquet")
    summ = {v: E.summary(pms[v]) for v in variants}
    vs_b0 = {v: {m: E.paired(pms["B0"], pms[v], m) for m in SHOW_METRICS} for v in variants if v != "B0"}
    for v in vs_b0:
        vs_b0[v]["macro_f1"] = E.f1_diff(pms["B0"], pms[v])
    pairs = {f"{a}-{b}": {m: E.paired(pms[b], pms[a], m) for m in ("goals_dev", "logloss_1x2", "ll_over2.5", "ll_btts")}
             for a, b in PAIRS if a in pms and b in pms}

    # ---- Main at real prices
    px = D.prices(ev["match_id"])
    lookup = D.price_lookup(px)
    truth = ev[["match_id", "home_goals", "away_goals", "total_corners", "cmu", "nb_size", "league", "date"]]
    mains = {v: E.main_picks(al[v], truth, lookup) for v in variants}
    pd.concat([m.assign(variant=v) for v, m in mains.items()]).to_parquet(C.CACHE / "main_picks.parquet")
    main = {"summary": {v: E.main_summary(mains[v]) for v in variants},
            "vs_B0": {v: E.main_paired(mains["B0"], mains[v]) for v in variants if v != "B0"}}
    # B0 against the Main production actually stored for the same matches
    st = D.stored_snapshots().set_index("match_id")
    b0 = mains["B0"].set_index("match_id").join(st[["main_market", "main_selection", "model_version"]], how="left")
    main["b0_same_as_stored"] = float(((b0["market"] == b0["main_market"]) & (b0["selection"] == b0["main_selection"])).mean())
    main["b0_same_as_stored_prod_20261012"] = float(((b0["market"] == b0["main_market"]) & (b0["selection"] == b0["main_selection"]))
                                                    [b0["model_version"] == C.PRODUCTION_VERSION].mean())
    main["prices"] = C.load_json("prices_info.json")

    # ---- Holm on the three primary metrics across all variants vs B0
    holm = {}
    for m in ("goals_dev", "logloss_1x2"):
        holm[m] = C.holm({v: vs_b0[v][m]["p"] for v in vs_b0})
    holm["roi_main"] = C.holm({v: main["vs_B0"][v]["roi"]["p"] for v in main["vs_B0"]})

    metrics = {"n_matches": int(len(ev)), "n_variants": len(variants), "variants": variants,
               "labels": M.LABELS, "summary": summ, "vs_B0": vs_b0, "pairs": pairs, "holm": holm,
               "bet365": {k: v for k, v in _bet365_reference(ev).items() if not k.startswith("ids")},
               "eval_period": [str(ev["date"].min().date()), str(ev["date"].max().date())],
               "leagues": sorted(ev["league"].unique()), "minutes": t.minutes()}
    # B0 and best variants on the Bet365 subset (same matches as the benchmark)
    ref = _bet365_reference(ev)
    sub = ev["match_id"].isin(ref["ids_1x2"]).to_numpy()
    metrics["bet365"]["models_on_subset"] = {v: float(pms[v].loc[sub, "logloss_1x2"].mean()) for v in variants}
    subo = ev["match_id"].isin(ref["ids_over"]).to_numpy()
    metrics["bet365"]["models_on_subset_over"] = {v: float(pms[v].loc[subo, "ll_over2.5"].mean()) for v in variants}
    C.save_json("metrics.json", metrics)
    C.save_json("main.json", main)
    breakdowns(ev, pms, mains)
    reliability_data(ev, pms, mains)
    corners = M.corners_check(df)
    C.save_json("corners.json", corners)
    return metrics


def breakdowns(ev: pd.DataFrame, pms: dict, mains: dict) -> dict:
    keys = {"league": ev["league"].to_numpy(), "month": ev["date"].dt.strftime("%Y-%m").to_numpy(),
            "band": pd.cut(ev["min_xg_n"], [-1, 4, 7, 100], labels=["1-4", "5-7", "8+"]).astype(str).to_numpy()}
    show = [v for v in ("R0a", "X1", "X2a", "X3a", "X5", "M1a", "X2Ma", "S1", "T1") if v in pms]
    out = {}
    for kname, kv in keys.items():
        rows = []
        for g in sorted(set(kv)):
            m = kv == g
            row = {"group": g, "n": int(m.sum()), "B0_goals_dev": float(pms["B0"].loc[m, "goals_dev"].mean()),
                   "B0_logloss": float(pms["B0"].loc[m, "logloss_1x2"].mean())}
            for v in show:
                dg = pms[v].loc[m, "goals_dev"].to_numpy() - pms["B0"].loc[m, "goals_dev"].to_numpy()
                dl = pms[v].loc[m, "logloss_1x2"].to_numpy() - pms["B0"].loc[m, "logloss_1x2"].to_numpy()
                row[f"{v}_dgoals"] = C.boot_mean_ci(dg, 1000)
                row[f"{v}_dll"] = C.boot_mean_ci(dl, 1000)
                mb, mv = mains["B0"].loc[m], mains[v].loc[m]
                bb, bv = mb["priced"] & mb["decided"], mv["priced"] & mv["decided"]
                row[f"{v}_droi"] = float(mv.loc[bv, "profit"].mean() - mb.loc[bb, "profit"].mean()) if bv.any() and bb.any() else None
            bb = mains["B0"].loc[m]
            bb = bb.loc[bb["priced"] & bb["decided"]]
            row["B0_roi"] = float(bb["profit"].mean()) if len(bb) else None
            row["B0_bets"] = int(len(bb))
            rows.append(row)
        out[kname] = rows
    C.save_json("breakdowns.json", out)
    return out


def reliability_data(ev: pd.DataFrame, pms: dict, mains: dict) -> None:
    out = {}
    for v in ("B0", "R0a", "X2a", "X3a", "X5", "M1a", "X2Ma", "X1"):
        if v not in pms:
            continue
        pm = pms[v]
        y = pm["y"].to_numpy()
        pooled_o = np.r_[(y == 0), (y == 1), (y == 2)].astype(float)
        pooled_p = np.r_[pm["pH"], pm["pD"], pm["pA"]]
        mp = mains[v]
        dec = mp.loc[mp["decided"]]
        out[v] = {"1x2": C.reliability(pooled_o, pooled_p, 10, 30), "over2.5": C.reliability(pm["o_over2.5"], pm["p_over2.5"], 10, 30),
                  "main": C.reliability(dec["won"], dec["p"], 10, 25)}
    C.save_json("reliability.json", out)


def sensitivity() -> dict:
    """lag 4 days (publication delay) and ridge 3 / 30 for the anchored candidates (not used for selection)."""
    out = {}
    base = M.frame(0)
    variants = ("R0a", "X2a", "X3a", "X5", "M1a", "X2Ma")
    settings = {"lag4": dict(lag=4, ridge=None), "ridge3": dict(lag=0, ridge=3.0), "ridge30": dict(lag=0, ridge=30.0)}
    for name, s in settings.items():
        old = (M.GLM_RIDGE, M.MNL_RIDGE)
        if s["ridge"]:
            M.GLM_RIDGE = M.MNL_RIDGE = s["ridge"]
        df = M.frame(s["lag"]) if s["lag"] else base
        preds, _ = M.in_season(df, only=variants)
        M.GLM_RIDGE, M.MNL_RIDGE = old
        ev, al = E.assemble(df, preds)
        pms = {v: E.per_match(al[v], ev) for v in al}
        out[name] = {"n": int(len(ev)), **{v: {m: E.paired(pms["B0"], pms[v], m) for m in ("goals_dev", "logloss_1x2")}
                                           for v in variants}}
        if s["lag"]:
            out[name]["vs_R0a"] = {v: {m: E.paired(pms["R0a"], pms[v], m) for m in ("goals_dev", "logloss_1x2")}
                                   for v in variants if v != "R0a"}
    C.save_json("sensitivity.json", out)
    return out


def decisions() -> dict:
    """A / B / C per variant from the result files (rules written before the decision was taken):

    A  needs a significant improvement (Holm p < 0.05) of goals deviance or 1X2 log loss vs B0, no significant
       deterioration of the other primary metrics, AND forward evidence; 2026/27 is not independent and no
       forward data exist yet, so no variant can be A in this study.
    B  point estimate better than B0 on goals deviance or 1X2 log loss AND better than its own no-xG control,
       with no significant deterioration (95 % CI) of the other primary metric or of the Main ROI.
    C  otherwise.
    """
    met, main = C.load_json("metrics.json"), C.load_json("main.json")
    ctrl = {"X2a": "R0a", "X3a": "R0a", "M1a": "R0a", "X2Ma": "R0a", "X2": "R0", "X3": "R0", "M1": "R0",
            "X2M": "R0", "X4": "R0a", "X5": None, "X1": "G1", "S1": "S0", "T1": "T0"}
    out = {}
    for v, d in met["vs_B0"].items():
        g, ll = d["goals_dev"], d["logloss_1x2"]
        roi = main["vs_B0"][v]["roi"]
        better = (g["mean"] < 0) or (ll["mean"] < 0)
        c = ctrl.get(v)
        vs_ctrl = None
        if c and c in met["summary"]:
            vs_ctrl = {m: met["summary"][v][m] - met["summary"][c][m] for m in ("goals_dev", "logloss_1x2")}
        better_ctrl = vs_ctrl is None or (vs_ctrl["goals_dev"] < 0 or vs_ctrl["logloss_1x2"] < 0)
        worse_sig = ((g["ci95"][0] is not None and g["ci95"][0] > 0) or (ll["ci95"][0] is not None and ll["ci95"][0] > 0)
                     or (roi["ci95"][1] is not None and roi["ci95"][1] < 0))
        holm_sig = ((met["holm"]["goals_dev"].get(v, 1) < 0.05 and g["mean"] < 0)
                    or (met["holm"]["logloss_1x2"].get(v, 1) < 0.05 and ll["mean"] < 0))
        if v in M.CONTROLS:
            cat, why = "—", "контроль (не кандидат)"
        elif holm_sig and not worse_sig:
            cat, why = "B", "значимое улучшение на 2026/27, но сезон не независим и форвард-данных нет"
        elif better and better_ctrl and not worse_sig:
            cat, why = "B", "точечная оценка лучше v6.1 и своего контроля, значимого ухудшения нет"
        else:
            reasons = []
            if not better:
                reasons.append("не лучше v6.1 ни по отклонению голов, ни по log loss 1X2")
            if better and not better_ctrl:
                reasons.append("не лучше своего контроля без xG")
            if worse_sig:
                reasons.append("значимое ухудшение одной из основных метрик")
            cat, why = "C", "; ".join(reasons)
        out[v] = {"decision": cat, "why": why, "goals_dev": g["mean"], "goals_dev_ci95": g["ci95"],
                  "logloss_1x2": ll["mean"], "logloss_1x2_ci95": ll["ci95"], "roi_diff": roi["diff"],
                  "roi_ci95": roi["ci95"], "vs_control": vs_ctrl, "control": c}
    C.save_json("decisions.json", out)
    return out
