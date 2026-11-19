"""Stage 6 of v6.3 — do the B candidates that act on the Main choice combine well?

    .venv/bin/python -m experiments.v6_3.summary.combine          (light job: no model fitting, ~10–20 min)

Evaluates exactly the combinations frozen in ``results/combinations_plan.json`` (``plan.py``; refuses to run
without it), plus the single parts (to judge whether a combination keeps their gains and to check that this
harness reproduces the owning studies). Everything runs on the shared candidate tables of the replay
(``replay/cache/candidates_{close,t09}``) with the vectorised production rule of the Main-selection study
(``main_selection/rules.py``), which is first checked against Baseline B. Probability-level parts are rebuilt
with the owning studies' own code (calibration cache, ``market_aware.run.prepared`` + fixed 50/50 pool,
``consensus.run`` consensus + walk-forward pool); model-level parts go through production's engine
(``replay.engine.batch_from_outputs``) with cached per-match outputs.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.main_selection import rules as R
from experiments.v6_3.replay import evaluate as EV

from .plan import PLAN

DIRS = C.study_dirs("summary")
RES, CACHE = DIRS["results"], DIRS["cache"]
UNIVERSES = ("close", "t09")
EXCLUDED = {"EHANDICAP", "EXACT_SCORE"}
SETTLED = ("WON", "LOST", "PUSH")
PROD = R.PRODUCTION
RULE_PARTS = {"REL": dict(reliability=R.with_reliability(PROD, "x", TOTAL=0.95).reliability),
              "EVW": dict(ev_weight=1.0), "FLOOR": dict(min_odds=1.40)}
PROB_PARTS = ("CAL", "V6", "C2")
MODEL_PARTS = ("GOALS", "CORNERS")


# ------------------------------------------------------------------------------------------------ tables
def prepare(c: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Derived columns of ``main_selection.data.load`` on any candidate table (played matches only)."""
    c = c.copy()
    c["odds"] = pd.to_numeric(c["odds"], errors="coerce").astype(float)
    c["probability"] = c["probability"].astype(float)
    c["push_probability"] = pd.to_numeric(c["push_probability"], errors="coerce").astype(float).fillna(0.0)
    c.loc[c["odds"] <= 1.0, "odds"] = np.nan
    done = c.groupby("match_id")["status"].transform(lambda s: s.isin(SETTLED + ("UNKNOWN",)).any())
    c = c.loc[done]
    c = c.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    c["has_odds"] = c["odds"].notna()
    c["excluded"] = c["group"].isin(EXCLUDED)
    c["month"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m").replace({"2026-07": "2026-08"})
    c["mi"] = pd.factorize(c["match_id"])[0]
    c["universe"] = universe
    c["p_model"] = c["probability"]
    return set_probability(c, c["probability"].to_numpy(float))


def set_probability(c: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """The probability the rule runs on (band, EV, score) — as the probability-level studies did."""
    c = c.copy()
    c["probability"] = p
    has = c["has_odds"].to_numpy(bool)
    push = c["push_probability"].to_numpy(float)
    odds = c["odds"].to_numpy(float)
    c["ev"] = np.where(has, p * odds - (1 - push), np.nan)
    c["eff_odds"] = np.where(has, odds, np.maximum(1 - push, 1e-9) / np.maximum(p, 1e-9))
    return c


def base_tables() -> dict[str, pd.DataFrame]:
    return {u: prepare(pd.read_parquet(C.SHARED_CACHE / f"candidates_{u}.parquet"), u) for u in UNIVERSES}


KEY = ["match_id", "market", "selection"]


def attach(c: pd.DataFrame, new: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace the rule probability by ``new[col]`` where present (rows without it keep the model's)."""
    x = c.merge(new[KEY + [col]].drop_duplicates(KEY), on=KEY, how="left")
    p = x[col].to_numpy(float)
    p = np.where(np.isfinite(p), p, x["p_model"].to_numpy(float))
    x[f"has_{col}"] = np.isfinite(x[col].to_numpy(float))
    return set_probability(x.drop(columns=[col]), p)


# ------------------------------------------------------------------------------------------------ probability parts
def prob_cal(log) -> pd.DataFrame:
    """Calibration study, temperature map per market group fitted on 2025/26 + earlier 2026/27 months (pooled)."""
    path = C.V63 / "calibration" / "cache" / "calibrated_2026.parquet"
    if not path.exists():
        raise SystemExit("calibration/cache/calibrated_2026.parquet missing: run "
                         "python -m experiments.v6_3.calibration.run first")
    cal = pd.read_parquet(path, columns=KEY + ["temperature|group|pooled"])
    cand = pd.read_parquet(C.SHARED_CACHE / "candidates_close.parquet", columns=KEY + ["push_probability"])
    cal = cal.merge(cand, on=KEY, how="left")
    push = pd.to_numeric(cal["push_probability"], errors="coerce").fillna(0.0).to_numpy(float)
    cal["p_cal"] = cal["temperature|group|pooled"].astype(float) * (1 - push)       # calibration.maps.to_p
    log.info("CAL: %d rows", len(cal))
    return cal[KEY + ["p_cal"]]


def prob_v6(u: str, odds, m, log) -> pd.DataFrame:
    """Market-aware study, V6: fixed 50/50 logit pool of the model and the single-book market probability."""
    from experiments.v6_3.market_aware import mkt as M
    from experiments.v6_3.market_aware.run import prepared
    c = prepared(u, "power", odds, m)
    tab = M.market_table(c)
    c = M.back_to_candidates(c, tab, M.fixed_pool(tab, {g: (0.5, 0.5) for g in tab}), "p_5050")
    c = c.loc[c["p_5050_has"]]
    log.info("V6 %s: %d rows with a pooled probability", u, len(c))
    return c[KEY + ["p_5050"]]


def prob_c2(u: str, odds, m, fd26, rule_1x2: str, dec_tab: pd.DataFrame, log) -> pd.DataFrame:
    """Consensus study, C2: monthly walk-forward pool of the model with the multi-book consensus."""
    from experiments.v6_3.consensus import run as CR
    from experiments.v6_3.market_aware import mkt as M
    c = CR.consensus_candidates(u, odds, m, fd26, rule_1x2, dec_tab)
    c["p_model"] = c["probability"]
    tab = M.market_table(c, "q_cons")
    bl, _ = M.walk_forward_pool(tab, "blend")
    c = M.back_to_candidates(c, tab, bl, "p_cons_blend")
    c = c.loc[c["p_cons_blend_has"]]
    log.info("C2 %s: %d rows with a pooled probability", u, len(c))
    return c[KEY + ["p_cons_blend"]]


def probability_parts(log) -> dict[tuple[str, str], pd.DataFrame]:
    from experiments.v6_3.consensus import fd as FD
    from experiments.v6_3.consensus import run as CR
    from experiments.v6_3.market_aware import mkt as M
    out = {}
    cal = prob_cal(log)
    odds, m = M.load_odds(), M.matches()
    fd_all = FD.all_seasons()
    fd26 = fd_all.loc[fd_all["season"] == C.CURRENT]
    rule_1x2 = C.read_json(C.V63 / "consensus" / "results" / "consensus.json")["rule_1x2"]["consensus_used"]
    _, dec_tabs = CR.decision_consensus(odds, m)
    for u in UNIVERSES:
        out[("CAL", u)] = (cal, "p_cal")
        out[("V6", u)] = (prob_v6(u, odds, m, log), "p_5050")
        out[("C2", u)] = (prob_c2(u, odds, m, fd26, rule_1x2, dec_tabs[u], log), "p_cons_blend")
    return out


# ------------------------------------------------------------------------------------------------ model parts
def corner_pover() -> pd.DataFrame:
    """Corner line probabilities of the corners study's 'best' variant (Poisson GLM + NB r by league)."""
    p = pd.read_csv(C.V63 / "corners" / "results" / "corner_predictions_2026.csv")
    out = pd.DataFrame({"match_id": p["match_id"]})
    for line in (8.5, 9.5, 10.5):
        out[line] = p[f"p{line}_best"].astype(float)
    return out


def model_tables(goals: bool, corners: bool, log) -> dict[str, pd.DataFrame]:
    """Candidate tables with the goal means of the goals study (through production's engine) and / or the corner
    probabilities of the corners study (replacing the corner rows, as that study did)."""
    from experiments.v6_3.corners import betting as CB
    tag = "_".join(n for n, f in (("goals", goals), ("corners", corners)) if f)
    out = {}
    for u in UNIVERSES:
        path = CACHE / f"cands_{tag}_{u}.parquet"
        if path.exists():
            out[u] = pd.read_parquet(path)
            continue
        if goals:
            out[u] = goals_candidates(u, log)
        else:
            out[u] = pd.read_parquet(C.SHARED_CACHE / f"candidates_{u}.parquet")
        if corners:
            out[u] = CB.with_probs(out[u], corner_pover())
        out[u].to_parquet(path)
    return out


def goals_candidates(u: str, log) -> pd.DataFrame:
    from experiments.v6_3.recent_weighting import lib as L
    from experiments.v6_3.replay import engine as EN
    path = CACHE / f"cands_goals_{u}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    st = pd.read_parquet(C.V63 / "goals" / "cache" / "stacked_2026.parquet")
    out = L.complete_outputs(st[["match_id", "lam_home", "lam_away"]])
    m, res = L.replay_matches(), L.replay_results()
    batch = EN.batch_from_outputs(m, out, L.odds_lookup(u), "summary_goals_stacked")
    c = EV.candidates(batch.markets, m, res)
    main = batch.picks.loc[batch.picks["role"] == "main", KEY].assign(engine_main=True)
    c = c.merge(main, on=KEY, how="left")
    c["engine_main"] = c["engine_main"].fillna(False).astype(bool)
    c.to_parquet(path)
    log.info("GOALS %s: engine batch in %.0f s, %d rows", u, time.time() - t0, len(c))
    return c


# ------------------------------------------------------------------------------------------------ evaluation
def rule_of(parts: list[str], name: str) -> R.Rule:
    kw = {}
    for p in parts:
        kw.update(RULE_PARTS.get(p, {}))
    return PROD.with_(name=name, **kw)


def baselines() -> dict:
    out = {}
    played = pd.read_parquet(C.SHARED_CACHE / "matches.parquet", columns=["match_id", "played"])
    played = set(played.loc[played["played"].astype(bool), "match_id"])
    for u in UNIVERSES:
        b = pd.read_parquet(C.SHARED_CACHE / f"baseline_b_{u}.parquet")
        out[u] = b.loc[b["match_id"].isin(played) & b["status"].isin(SETTLED)].reset_index(drop=True)
    a = pd.read_parquet(C.SHARED_CACHE / "baseline_a.parquet")
    out["a"] = a.loc[a["status"].isin(SETTLED)].reset_index(drop=True)
    return out


def months(main: pd.DataFrame, base: pd.DataFrame) -> dict:
    def tab(t):
        b = t.loc[t["bet"] == 1].assign(month=pd.to_datetime(t.loc[t["bet"] == 1, "date"]).dt.strftime("%Y-%m")
                                        .replace({"2026-07": "2026-08"}))
        return b.groupby("month").agg(profit=("profit", "sum"), bets=("bet", "sum"))
    a, b = tab(main), tab(base)
    rows = []
    for mth in sorted(set(a.index) | set(b.index)):
        ra = a.loc[mth, "profit"] / a.loc[mth, "bets"] if mth in a.index else None
        rb = b.loc[mth, "profit"] / b.loc[mth, "bets"] if mth in b.index else None
        rows.append({"month": mth, "bets": int(a.loc[mth, "bets"]) if mth in a.index else 0, "roi": ra, "roi_b": rb,
                     "diff": None if ra is None or rb is None else ra - rb})
    d = [r["diff"] for r in rows if r["diff"] is not None]
    return {"months": rows, "months_better": int(sum(x > 0 for x in d)), "months_n": len(d)}


def evaluate(main: pd.DataFrame, base: dict, u: str) -> dict:
    s = EV.summary(main)
    cmp = EV.compare(main, base[u])
    out = {"summary": s, "roi_ci": EV.roi_with_ci(main), "vs_b": cmp, "vs_a": EV.compare(main, base["a"]),
           "changed_share": 1 - cmp["same_main_share"],
           "not_bet_fallback": int(main["note"].fillna("").str.contains("not a bet").sum()),
           "groups_bets": main.loc[main["bet"] == 1, "group"].value_counts().to_dict()}
    out.update(months(main, base[u]))
    b = main.loc[main["bet"] == 1].sort_values("profit", ascending=False)
    out["roi_without_top5_wins"] = float(b["profit"].iloc[5:].sum() / len(b)) if len(b) > 5 else None
    won = b["status"] == "WON"
    out["roi_odds_minus_2pct"] = float(np.where(won, b["odds"] * 0.98 - 1, -1.0).sum() / len(b)) if len(b) else None
    return out


def compact(vid: str, parts: list[str], u: str, ev: dict) -> dict:
    s, rd = ev["summary"], ev["vs_b"]["roi_diff"]
    return {"id": vid, "parts": "+".join(parts), "universe": u, "bets": s["bets"], "roi": s["roi"],
            "roi_lo": ev["roi_ci"]["ci95"][0], "roi_hi": ev["roi_ci"]["ci95"][1], "hit_rate": s["hit_rate"],
            "avg_odds": s["avg_odds"], "profit": s["profit"], "d_roi": rd["diff"], "d_lo": rd["ci95"][0],
            "d_hi": rd["ci95"][1], "p": rd["p"], "d_hit": ev["vs_b"]["hit_rate_diff"]["diff"],
            "changed_share": ev["changed_share"], "months_better": ev["months_better"],
            "months_n": ev["months_n"], "not_bet_fallback": ev["not_bet_fallback"]}


def expected_from_studies() -> dict:
    """ROI of the single parts as the owning studies reported them (reproduction check)."""
    j = C.read_json
    out = {}
    ms = j(C.V63 / "main_selection/results/main_selection.json")["candidates"]
    for name, part in (("bt:reliability", "REL"), ("bt:ev_weight", "EVW"), ("floor_1.40", "FLOOR")):
        for r in ms:
            if r["candidate"] == name:
                out[(part, r["universe"])] = r["roi"]
    for r in ms:
        if r["candidate"] == "combo_bt":
            out[("REL+EVW+FLOOR", r["universe"])] = r["roi"]
    cal = j(C.V63 / "calibration/results/calibration.json")["main"]
    ma = j(C.V63 / "market_aware/results/market_aware.json")["variants"]
    co = j(C.V63 / "consensus/results/consensus.json")["variants"]
    cn = j(C.V63 / "corners/results/results.json")["main"]
    gl = pd.read_csv(C.V63 / "goals/results/main_roi.csv").set_index(["variant", "universe"])
    for u in UNIVERSES:
        out[("CAL", u)] = cal[u]["temperature|group|pooled"]["summary"]["roi"]
        out[("V6", u)] = ma[u]["blend_5050"]["summary"]["roi"]
        out[("C2", u)] = co[u]["cons_blend_wf"]["summary"]["roi"]
        out[("CORNERS", u)] = cn[u]["best"]["summary"]["roi"]
        out[("GOALS", u)] = float(gl.loc[("stacked", u), "roi"])
    return out


def run() -> dict:
    started = time.time()
    log = C.setup_logging("summary.combine", DIRS["logs"] / "combine.log")
    if not PLAN.exists():
        raise SystemExit("results/combinations_plan.json missing: freeze the plan first (summary.plan)")
    plan = C.read_json(PLAN)
    base = baselines()
    tables = base_tables()

    # self-check: vectorised production rule on the shared candidate tables == Baseline B
    check = {}
    for u in UNIVERSES:
        mine = R.select(tables[u], PROD)
        j = mine.merge(base[u][KEY], on="match_id", suffixes=("", "_b"))
        check[u] = float(((j["market"] == j["market_b"]) & (j["selection"] == j["selection_b"])).mean())
    log.info("rule == Baseline B: %s", check)

    probs = probability_parts(log)
    variants = [{"id": r["id"], "parts": r["parts"], "role": r["role"]} for r in plan["references"]] + [
        {"id": k["id"], "parts": k["parts"], "role": "combination"} for k in plan["combinations"]]
    need_models = {tuple(p for p in v["parts"] if p in MODEL_PARTS) for v in variants} - {()}
    mtabs = {}
    for mp in need_models:
        raw = model_tables("GOALS" in mp, "CORNERS" in mp, log)
        mtabs[mp] = {u: prepare(raw[u], u) for u in UNIVERSES}
    # engine check for the goals tables: the vectorised rule == the engine's own Main
    engine_check = {}
    for u in UNIVERSES:
        g = goals_candidates(u, log)
        gt = prepare(g, u)
        mine = R.select(gt, PROD)
        eng = gt.loc[gt["engine_main"], KEY]
        j = mine.merge(eng, on="match_id", suffixes=("", "_e"))
        engine_check[u] = float(((j["market"] == j["market_e"]) & (j["selection"] == j["selection_e"])).mean())
    log.info("GOALS: vectorised rule == engine Main: %s", engine_check)

    mains, details, rows = {}, {}, []
    for v in variants:
        parts = v["parts"]
        mp = tuple(p for p in parts if p in MODEL_PARTS)
        pp = [p for p in parts if p in PROB_PARTS]
        assert len(pp) <= 1, v
        rule = rule_of(parts, v["id"])
        details[v["id"]] = {"parts": parts, "role": v["role"]}
        for u in UNIVERSES:
            c = mtabs[mp][u] if mp else tables[u]
            if pp:
                new, col = probs[(pp[0], u)]
                c = attach(c, new, col)
            main = R.select(c, rule)
            main = main.loc[main["status"].isin(SETTLED + ("UNKNOWN",))].reset_index(drop=True)
            mains[(v["id"], u)] = main
            ev = evaluate(main, base, u)
            details[v["id"]][u] = ev
            row = compact(v["id"], parts, u, ev)
            row["role"] = v["role"]
            rows.append(row)
            log.info("%s %s: ROI %.4f d %.4f p %.3f", v["id"], u, row["roi"], row["d_roi"], row["p"])
    df = pd.DataFrame(rows)

    # Holm over the 12 combinations, per universe and jointly
    combo = df["role"] == "combination"
    for u in UNIVERSES:
        sub = df.loc[combo & (df["universe"] == u)]
        adj = C.holm(sub["p"].to_dict())
        df.loc[sub.index, "p_holm"] = pd.Series(adj)
    adj = C.holm(df.loc[combo, "p"].to_dict())
    df.loc[combo, "p_holm_joint"] = pd.Series(adj)

    # does a combination keep the gains of its parts?
    single = {(r.parts, r.universe): r.d_roi for r in df.loc[df["id"].str.startswith("S_")].itertuples()}
    keep = []
    for k in plan["combinations"]:
        rec = {"id": k["id"], "parts": "+".join(k["parts"])}
        ok_keep, ok_add = True, True
        for u in UNIVERSES:
            d = float(df.loc[(df["id"] == k["id"]) & (df["universe"] == u), "d_roi"].iloc[0])
            dp = {p: single[(p, u)] for p in k["parts"]}
            best_part = max(dp, key=dp.get)
            cmp = EV.compare(mains[(k["id"], u)], mains[(f"S_{best_part}", u)])
            rec[u] = {"d_roi": d, "parts_d_roi": dp, "best_part": best_part, "sum_parts": float(sum(dp.values())),
                      "vs_best_part": cmp["roi_diff"], "same_main_as_best_part": cmp["same_main_share"]}
            ok_keep &= d >= dp[best_part] - 1e-12
            ok_add &= d >= sum(dp.values()) - 1e-12
        rec["keeps_gains_both"] = bool(ok_keep)
        rec["additive_both"] = bool(ok_add)
        keep.append(rec)

    # reproduction of the owning studies by the single parts
    exp = expected_from_studies()
    repro = []
    for (part, u), roi in exp.items():
        vid = "R_combo_bt" if part == "REL+EVW+FLOOR" else f"S_{part}"
        got = df.loc[(df["id"] == vid) & (df["universe"] == u), "roi"]
        if len(got):
            repro.append({"part": part, "universe": u, "study_roi": roi, "here_roi": float(got.iloc[0]),
                          "abs_diff": abs(float(got.iloc[0]) - roi)})

    df.to_csv(RES / "combinations.csv", index=False)
    # per-match tables: Baseline B and the combination with the best evidence (criterion of the plan)
    res = pd.read_parquet(C.SHARED_CACHE / "matches.parquet",
                          columns=["match_id", "played", "home_goals", "away_goals", "total_corners"])
    res = res.loc[res["played"].astype(bool)].set_index("match_id")
    cand = evidence_rank(df.loc[combo])
    best = cand[0]["id"] if cand else None
    for u in UNIVERSES:
        b = base[u].assign(note="")
        EV.per_match_table(b, res, f"baseline_b_{u}", "prod_20261012").to_csv(
            RES / f"per_match_baseline_b_{u}.csv", index=False)
        if best:
            t = mains[(best, u)].copy()
            t["note"] = t["note"].fillna("")
            EV.per_match_table(t, res, f"{best}_{u}", "prod_20261012 + " + "+".join(details[best]["parts"])).to_csv(
                RES / f"per_match_{best}_{u}.csv", index=False)
    cov = []
    m = pd.read_parquet(C.SHARED_CACHE / "matches.parquet", columns=["match_id", "played"])
    for (vid, u), t in mains.items():
        cov.append({"variant": vid, "universe": u, "season_matches": int(len(m)),
                    "played": int(m["played"].astype(bool).sum()), "processed": int(len(t)),
                    "main_with_real_price": int(t["odds"].notna().sum()), "bets": int(t["bet"].sum()),
                    "not_bet_no_price": int((t["odds"].isna() & t["status"].isin(["WON", "LOST"])).sum()),
                    "not_bet_fallback": int(t["note"].fillna("").str.contains("not a bet").sum()),
                    "push": int((t["status"] == "PUSH").sum()), "unknown_result": int((t["status"] == "UNKNOWN").sum()),
                    "main_with_new_probability": int(t.filter(like="has_p_").any(axis=1).sum())
                    if t.filter(like="has_p_").shape[1] else None})
    pd.DataFrame(cov).to_csv(RES / "combinations_coverage.csv", index=False)

    out = {"plan_frozen_at": plan["frozen_at"], "self_check_rule_equals_baseline_b": check,
           "self_check_goals_rule_equals_engine": engine_check, "reproduction": repro,
           "baseline_b": {u: {"summary": EV.summary(base[u]), "roi_ci": EV.roi_with_ci(base[u])} for u in UNIVERSES},
           "baseline_a": {"summary": EV.summary(base["a"]), "roi_ci": EV.roi_with_ci(base["a"])},
           "table": df.to_dict("records"), "keeps_gains": keep, "evidence_rank": cand, "best_by_evidence": best,
           "details": details, "n_combinations": int(combo.sum() // 2)}
    C.write_json(RES / "combinations.json", out)
    C.write_json(RES / "run_info_combine.json", C.run_info("summary.combine", started, {
        "plan": str(PLAN.relative_to(C.ROOT)), "universes": UNIVERSES, "n_boot": C.N_BOOT, "seed": C.SEED}))
    log.info("done in %.0f s", time.time() - started)
    return out


def evidence_rank(df: pd.DataFrame) -> list[dict]:
    """The plan's criterion: positive in both universes, then smallest Holm p at close, then smallest worse raw p."""
    rows = []
    for vid, g in df.groupby("id"):
        g = g.set_index("universe")
        rows.append({"id": vid, "parts": g.loc["close", "parts"], "d_close": float(g.loc["close", "d_roi"]),
                     "d_t09": float(g.loc["t09", "d_roi"]), "p_holm_close": float(g.loc["close", "p_holm"]),
                     "p_close": float(g.loc["close", "p"]), "p_t09": float(g.loc["t09", "p"])})
    pos = [r for r in rows if r["d_close"] > 0 and r["d_t09"] > 0]
    pool = pos or rows
    return sorted(pool, key=lambda r: (r["p_holm_close"], max(r["p_close"], r["p_t09"])))


if __name__ == "__main__":
    run()
