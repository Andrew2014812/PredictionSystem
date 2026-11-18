"""ИССЛЕДОВАНИЕ 3 — Market-Aware Main (FootPredict v6.3).

    .venv/bin/python -m experiments.v6_3.market_aware.run

Light job (no model fitting: production outputs of the shared replay, pandas / scipy on the candidate tables).

1. De-margining: fair probabilities of every complete book at the decision moment by five methods; log loss /
   Brier next to the model on the same markets (both universes).
2. Information test: does the model add information beyond the market?  Pool
   P(y) ∝ exp(a_y + b_q log q_y + b_p log p_y) per market group — full-season fit with a bootstrap interval
   (inference) and a monthly expanding-window fit (out-of-sample log loss of model / market / recalibrated market /
   blend). Candidate-level ROI by model edge p / q − 1.
3. Main variants (both universes, paired with Baseline B): model–market blends (walk-forward, fixed v6.2 weights,
   fixed 50/50), market only, disagreement filter (threshold chosen on earlier months), blended-EV value-first,
   maximum blended EV in the band; de-margin method sensitivity. Holm over all variant × universe tests.
4. Disagreement model − market as an uncertainty signal for the Main picks.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV
from src.markets.recommendation import main_params

from . import mkt as M

STUDY = "market_aware"
D = C.study_dirs(STUDY)
RES, FIG = D["results"], D["figures"]
UNIVERSES = ("close", "t09")
DELTA_GRID = (0.25, 0.5, 0.75, 1.0, np.inf)
MIN_PRIOR_BETS = 300
V62_HYBRIDS = C.ROOT / "experiments/v6_2/bookmaker/results/hybrids.json"
VARIANTS_RU = {
    "baseline_b": "Baseline B (production)",
    "blend_wf": "V1 смесь логитов, веса по прошлым месяцам",
    "blend_v62": "V2 смесь, фиксир. веса v6.2 (2022–2025)",
    "market_only": "V3 только рынок (q вместо p)",
    "disagree_filter": "V4 фильтр расхождения |logit p − logit q| ≤ δ",
    "blend_value_first": "V5 V1 + сначала EV(смеси) ≥ 0",
    "blend_5050": "V6 смесь 50/50 (без подбора)",
    "max_blend_ev": "V7 макс. EV(смеси) в полосе p",
    "blend_wf_prop": "V1 с пропорц. снятием маржи",
    "blend_wf_shin": "V1 со снятием маржи по Шину",
    "blend_wf_log": "V1 с odds-ratio снятием маржи",
}


PREREG = [{
    "id": "v63-market-aware-V6",
    "hypothesis": "A fixed half-and-half logit blend of the model and the single-book market probability improves "
                  "Main ROI versus Baseline B (no fitted weight, so it can be tested forward without re-tuning).",
    "rule": "Production Main rule on p = softmax(0.5 log q + 0.5 log p_model) per market (handicap on the no-refund "
            "scale, double chance from the blended 1X2); q = power de-margining of the first complete book among "
            "bet365 (OddsPapi), Football-Data Bet365, Pinnacle, William Hill at the decision moment; candidates "
            "without a complete book keep the model probability.",
    "primary_metric": "paired ROI difference vs Baseline B at the T universe (bootstrap by match)",
    "secondary": ["the same at the closing universe", "hit rate", "share of Main with a market probability"],
    "minimum_forward_sample": "1000 settled bets",
    "decision": "B (exploratory; chosen after seeing 2026/27, so only a forward result counts)"}]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------ inputs
def prepared(universe: str, method: str = "power", odds=None, m=None) -> pd.DataFrame:
    """Candidates of a universe with the single-book market probability at the universe's decision moment."""
    odds = M.load_odds() if odds is None else odds
    m = M.matches() if m is None else m
    quotes = M.latest_quotes(odds, M.moment(m, universe))
    fair = M.single_book(M.fair_books(quotes, method))
    c = M.load_candidates(universe)
    c = M.attach_q(c, fair)
    hc = c["group"] == "HANDICAP"
    c["q_win"] = np.where(hc, c["q"] * (1 - c["push_probability"]), c["q"])
    return c


def v62_weights() -> dict:
    """The v6.2 pool (H2) fitted on 2022/23–2025/26 (Football-Data Bet365 opening, walk-forward models) and applied
    to 2026/27: theta = [intercepts, b_market, b_model]."""
    h = C.read_json(V62_HYBRIDS)["results"]
    return {"1X2": h["1X2|2026"]["params"]["H2_theta"], "TOTAL": h["OU25|2026"]["params"]["H2_theta"],
            "HANDICAP": h["AH|2026"]["params"]["H2_theta"]}


def v62_pool(tab: dict, th: dict) -> dict:
    out = {}
    for grp, r in tab.items():
        if grp in th:
            out[grp] = M.pool_predict(np.asarray(th[grp]), [M.logs(r["Q"]), M.logs(r["P"])])
        else:
            out[grp] = r["P"].copy()
    return out


# ------------------------------------------------------------------------------------------------ 1. de-margining
def demargin_compare(universe: str, odds, m) -> dict:
    out = {}
    base = None
    for method in M.METHODS_ORDER:
        c = prepared(universe, method, odds, m)
        tab = M.market_table(c)
        for grp, r in tab.items():
            ok = r["y"] >= 0
            ll_q = M.ll_rows(r["Q"][ok], r["y"][ok])
            ll_p = M.ll_rows(r["P"][ok], r["y"][ok])
            Y = np.eye(r["Q"].shape[1])[r["y"][ok]]
            key = f"{grp}"
            d = out.setdefault(key, {"n": int(ok.sum())})
            d[method] = {"ll": float(ll_q.mean()), "brier": float(((r["Q"][ok] - Y) ** 2).sum(1).mean())}
            if method == "power":
                d["model"] = {"ll": float(ll_p.mean()), "brier": float(((r["P"][ok] - Y) ** 2).sum(1).mean())}
                d["model_minus_power"] = C.paired_mean_diff(ll_p, ll_q)
                base = base or {}
                base[grp] = ll_q
            elif base is not None and grp in base and len(base[grp]) == len(ll_q):
                d[f"{method}_minus_power"] = C.paired_mean_diff(ll_q, base[grp])
        if method == "power":
            mg = c.drop_duplicates(["match_id", "market"])
            out["_margin"] = mg.groupby("group")["q_margin"].mean().dropna().to_dict()
            out["_margin_book"] = {f"{g}|{b}": v for (g, b), v in
                                   mg.groupby(["group", "q_book"])["q_margin"].mean().dropna().items()}
            out["_books"] = mg.groupby(["group", "q_book"]).size().unstack(fill_value=0).to_dict("index")
    return out


# ------------------------------------------------------------------------------------------------ 2. information
def boot_pool(r: dict, n_boot: int = 200) -> dict:
    ok = r["y"] >= 0
    Q, P, y = r["Q"][ok], r["P"][ok], r["y"][ok]
    ids = pd.factorize(r["index"].get_level_values(0)[ok])[0]
    th = M.pool_fit([M.logs(Q), M.logs(P)], y)
    thq = M.pool_fit([M.logs(Q)], y)
    k = Q.shape[1]
    nll = M.ll_rows(M.pool_predict(th, [M.logs(Q), M.logs(P)]), y).sum()
    nll_q = M.ll_rows(M.pool_predict(thq, [M.logs(Q)]), y).sum()
    from scipy.stats import chi2
    lr = 2 * (nll_q - nll)
    rng = np.random.default_rng(C.SEED)
    nm = ids.max() + 1
    rows_of = pd.Series(np.arange(len(ids))).groupby(ids).apply(np.array).to_list()
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, nm, nm)
        idx = np.concatenate([rows_of[i] for i in pick])
        bs.append(M.pool_fit([M.logs(Q[idx]), M.logs(P[idx])], y[idx], x0=th)[k - 1:])
    bs = np.array(bs)
    return {"n": int(len(y)), "b_market": float(th[k - 1]), "b_model": float(th[k]),
            "b_market_ci": C.ci(bs[:, 0]), "b_model_ci": C.ci(bs[:, 1]),
            "lr": float(lr), "p_lr": float(chi2.sf(lr, 1)), "ll_gain_per_row": float((nll_q - nll) / len(y))}


def information(c: pd.DataFrame) -> dict:
    tab = M.market_table(c)
    out = {"full_season": {}, "walk_forward": {}, "coefs": {}}
    blend, coefs = M.walk_forward_pool(tab, "blend")
    recal, _ = M.walk_forward_pool(tab, "market")
    model_wf, _ = M.walk_forward_pool(tab, "model")
    out["coefs"] = coefs
    for grp, r in tab.items():
        out["full_season"][grp] = boot_pool(r)
        ok = (r["y"] >= 0) & (r["month"] >= "2026-09")          # months that have a fitted pool for every kind
        y = r["y"][ok]
        ll = {"market": M.ll_rows(r["Q"][ok], y), "model": M.ll_rows(r["P"][ok], y),
              "market_recal": M.ll_rows(recal[grp][ok], y), "model_recal": M.ll_rows(model_wf[grp][ok], y),
              "blend": M.ll_rows(blend[grp][ok], y)}
        out["walk_forward"][grp] = {"n": int(ok.sum()), **{k: float(v.mean()) for k, v in ll.items()},
                                    "model_minus_market": C.paired_mean_diff(ll["model"], ll["market"]),
                                    "blend_minus_market": C.paired_mean_diff(ll["blend"], ll["market"]),
                                    "blend_minus_market_recal": C.paired_mean_diff(ll["blend"], ll["market_recal"]),
                                    "blend_minus_model": C.paired_mean_diff(ll["blend"], ll["model"])}
    return out


def cluster_roi(df: pd.DataFrame) -> dict:
    g = df.groupby("match_id").agg(profit=("profit", "sum"), bets=("bet", "sum"))
    r = C.roi_ci(g["profit"].to_numpy(), g["bets"].to_numpy())
    return {**r, "candidates": int(len(df))}


EDGE_BINS = [-np.inf, -0.10, -0.05, 0.0, 0.05, 0.10, np.inf]
EDGE_LABELS = ["< −10%", "−10…−5%", "−5…0%", "0…+5%", "+5…+10%", "> +10%"]


def edge_analysis(c: pd.DataFrame) -> dict:
    """Every priced candidate inside the Main band (not exact score / European handicap) with a market
    probability: ROI by the model's edge over the market e = p / q − 1 (handicap on the no-refund scale)."""
    ok = (c["bet"] == 1) & c["q"].notna() & c["probability"].between(0.40, 0.85) & ~c["group"].isin(
        ["EHANDICAP", "EXACT_SCORE"])
    d = c.loc[ok].copy()
    d["edge"] = d["p_model"] / d["q_win"] - 1
    d["bin"] = pd.cut(d["edge"], EDGE_BINS, labels=EDGE_LABELS)
    out = {"all": cluster_roi(d), "bins": {}}
    for b in EDGE_LABELS:
        g = d.loc[d["bin"] == b]
        if len(g):
            out["bins"][b] = {**cluster_roi(g), "hit": float((g["status"] == "WON").mean()),
                              "mean_p": float(g["p_model"].mean()), "mean_q": float(g["q_win"].mean())}
    # does the edge carry information about the result beyond the market price?  logistic regression of the
    # outcome on logit q and the edge (log p − log q), clustered bootstrap by match
    import statsmodels.api as sm
    y = (d["status"] == "WON").astype(float).to_numpy()
    X = np.column_stack([np.ones(len(d)), logit(d["q_win"]), np.log(d["p_model"] / d["q_win"])])
    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["match_id"])[0]})
    out["logit_edge"] = {"coef": fit.params.tolist(), "se": fit.bse.tolist(), "p": fit.pvalues.tolist(),
                         "n": int(len(d))}
    return out


# ------------------------------------------------------------------------------------------------ 3. variants
def roi_by_month(main: pd.DataFrame, res) -> pd.Series:
    p = M.played(main, res)
    p = p.assign(month=pd.to_datetime(p["date"]).dt.strftime("%Y-%m"))
    g = p.groupby("month").agg(profit=("profit", lambda s: s.fillna(0).sum()), bets=("bet", "sum"))
    return g


def disagreement_filter(c: pd.DataFrame, res) -> tuple[pd.DataFrame, dict]:
    """Production rule without candidates whose model and market disagree by more than δ in logit (handicap on
    the no-refund scale). δ for month M = the grid value with the best Main ROI over the earlier months (expanding
    window; ∞ = production while fewer than MIN_PRIOR_BETS earlier bets)."""
    pm = np.where(c["group"] == "HANDICAP", M.cond(c["p_model"], c["push_probability"]), c["p_model"])
    qm = np.where(c["group"] == "HANDICAP", c["q"], c["q"])
    dis = np.abs(logit(pm) - logit(qm))
    has = np.isfinite(dis)
    mains = {d: M.main_vec(c, "p_model", extra_eligible=~has | (dis <= d)) for d in DELTA_GRID}
    by_month = {d: roi_by_month(mains[d], res) for d in DELTA_GRID}
    chosen, parts = {}, []
    months = sorted(c["month"].unique())
    for mth in months:
        prior = {d: by_month[d].loc[by_month[d].index < mth] for d in DELTA_GRID}
        nb = prior[np.inf]["bets"].sum()
        if nb < MIN_PRIOR_BETS:
            best = np.inf
        else:
            best = max(DELTA_GRID, key=lambda d: (prior[d]["profit"].sum() / max(prior[d]["bets"].sum(), 1),
                                                  -d if np.isfinite(d) else -99))
        chosen[mth] = None if not np.isfinite(best) else float(best)
        ids = set(c.loc[c["month"] == mth, "match_id"])
        parts.append(mains[best].loc[mains[best]["match_id"].isin(ids)])
    return pd.concat(parts, ignore_index=True), {"delta_by_month": chosen, "grid": [
        None if not np.isfinite(d) else d for d in DELTA_GRID]}


def variants(universe: str, odds, m, res) -> tuple[dict, dict, dict]:
    base = M.baseline(f"baseline_b_{universe}")
    mains, extra = {}, {}
    cands = {}
    for method in ("power", "prop", "shin", "log"):
        c = prepared(universe, method, odds, m)
        tab = M.market_table(c)
        blend, coefs = M.walk_forward_pool(tab, "blend")
        c = M.back_to_candidates(c, tab, blend, "p_blend")
        sfx = "" if method == "power" else f"_{method}"
        mains[f"blend_wf{sfx}"] = M.main_vec(c, "p_blend", method=f"blend_wf{sfx}")
        if method != "power":
            continue
        extra["coefs"] = coefs
        cands["power"] = c
        c = M.back_to_candidates(c, tab, v62_pool(tab, v62_weights()), "p_v62")
        c = M.back_to_candidates(c, tab, M.fixed_pool(tab, {g: (0.5, 0.5) for g in tab}), "p_5050")
        c["p_market"] = c["q_win"].fillna(c["p_model"])
        mains["blend_v62"] = M.main_vec(c, "p_v62", method="blend_v62")
        mains["market_only"] = M.main_vec(c, "p_market", method="market_only")
        mains["blend_5050"] = M.main_vec(c, "p_5050", method="blend_5050")
        mains["blend_value_first"] = M.main_vec(c, "p_blend", main_params().with_(value_first_min_ev=0.0),
                                                method="blend_value_first")
        # maximum blended EV among production-eligible (model band) priced candidates; production rule otherwise
        odds_ = c["odds"].to_numpy(float)
        ev_b = c["p_blend"].to_numpy(float) * odds_ - (1 - c["push_probability"].to_numpy(float))
        score = np.where(np.isfinite(odds_) & (odds_ > 1), ev_b + 10.0, -10.0)   # priced first, then unpriced
        prod_score, _ = M.rule_scores(c, c["p_model"].to_numpy(float), main_params())
        score = np.where(score > 0, score, prod_score - 10.0)
        mains["max_blend_ev"] = M.main_vec(c, "p_model", rank_score=score, method="max_blend_ev")
        f, info = disagreement_filter(c, res)
        mains["disagree_filter"] = f.assign(method="disagree_filter")
        extra["disagree_filter"] = info
        cands["final"] = c
    out = {}
    for k, v in mains.items():
        out[k] = M.evaluate_variant(v, base, res)
        out[k]["stability"] = M.stability(v, base, res)
        mm = M.played(v, res)
        out[k]["main_with_market_prob"] = float(mm["q"].notna().mean())
    out["baseline_b"] = {"summary": EV.summary(M.played(base, res)), "roi_ci": EV.roi_with_ci(M.played(base, res))}
    out["baseline_b"]["stability"] = M.stability(base, base, res)
    return out, mains, {**extra, "cands": cands}


# ------------------------------------------------------------------------------------------------ 4. disagreement
DIS_BINS = [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf]
DIS_LABELS = ["< −0.5", "−0.5…−0.25", "−0.25…0", "0…0.25", "0.25…0.5", "> 0.5"]


def disagreement_main(c: pd.DataFrame, base: pd.DataFrame, res) -> dict:
    b = M.played(base, res).merge(c[["match_id", "market", "selection", "q", "q_win"]],
                                  on=["match_id", "market", "selection"], how="left")
    pm = np.where(b["group"] == "HANDICAP", M.cond(b["probability"], b["push_probability"]), b["probability"])
    b["d"] = logit(pm) - logit(b["q"])
    b["bin"] = pd.cut(b["d"], DIS_BINS, labels=DIS_LABELS)
    s = b.loc[b["q"].notna() & b["status"].isin(["WON", "LOST"])]
    out = {"with_q": int(b["q"].notna().sum()), "total": int(len(b)), "bins": {},
           "overall": {"n": int(len(s)), "hit": float((s["status"] == "WON").mean()),
                       "mean_p": float(s["probability"].mean()), "mean_q": float(s["q_win"].mean()),
                       "share_p_above_q": float((s["probability"] > s["q_win"]).mean())}}
    for lab in DIS_LABELS:
        g = b.loc[(b["bin"] == lab) & b["status"].isin(["WON", "LOST", "PUSH"])]
        if not len(g):
            continue
        r = EV.roi_with_ci(g)
        out["bins"][lab] = {"n": int(len(g)), "hit": float((g["status"] == "WON").mean()),
                            "mean_p": float(g["probability"].mean()), "mean_q": float(g["q_win"].mean()),
                            **r}
    return out


# ------------------------------------------------------------------------------------------------ outputs
def coverage(universe: str, mains: dict, res, m) -> list[dict]:
    rows = []
    for k, v in mains.items():
        p = M.played(v, res)
        rows.append({"universe": universe, "method": k, "season_matches": int(len(m)), "played": int(len(res)),
                     "processed": int(v["match_id"].nunique()), "played_with_main": int(len(p)),
                     "bets_real_price": int(p["bet"].sum()), "no_price": int(p["odds"].isna().sum()),
                     "push": int((p["status"] == "PUSH").sum()), "unknown": int((p["status"] == "UNKNOWN").sum()),
                     "without_statistics": 0, "not_played_excluded": int(len(m) - len(res))})
    return rows


def run() -> dict:
    started = time.time()
    log = C.setup_logging(STUDY, D["logs"] / "run.log")
    res = M.results()
    m = M.matches()
    odds = M.load_odds()
    report = {"self_check": {u: M.self_check(u) for u in UNIVERSES}}
    log.info("self check %s", report["self_check"])
    report["demargin"] = {u: demargin_compare(u, odds, m) for u in UNIVERSES}
    log.info("de-margining done")
    report["variants"], report["information"], report["edge"], report["disagreement"] = {}, {}, {}, {}
    report["extra"] = {}
    cov, pvals = [], {}
    all_mains = {}
    for u in UNIVERSES:
        out, mains, extra = variants(u, odds, m, res)
        report["variants"][u] = out
        report["extra"][u] = {k: v for k, v in extra.items() if k != "cands"}
        c = extra["cands"]["final"]
        report["information"][u] = information(c)
        report["edge"][u] = edge_analysis(c)
        base = M.baseline(f"baseline_b_{u}")
        report["disagreement"][u] = disagreement_main(c, base, res)
        cov += coverage(u, {"baseline_b": base, **mains}, res, m)
        for k, v in out.items():
            if "vs_b" in v:
                pvals[f"{k}|{u}"] = v["vs_b"]["roi_diff"]["p"]
        all_mains[u] = (base, mains, c)
        log.info("%s: %s", u, {k: round(v["summary"]["roi"], 4) for k, v in out.items()})
    adj = C.holm(pvals)
    report["holm"] = {"n_tests": len(adj), "p_holm": adj}
    a = M.baseline("baseline_a")
    report["baseline_a"] = {"summary": EV.summary(a), "roi_ci": EV.roi_with_ci(a)}
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)
    # per-match tables: Baseline B and the study's main candidates of both universes
    for u, (base, mains, c) in all_mains.items():
        EV.per_match_table(M.played(base, res), res, f"baseline_b_{u}", "prod_20261012").to_csv(
            RES / f"per_match_baseline_b_{u}.csv", index=False)
        for k in ("blend_wf", "market_only", "disagree_filter", "blend_5050"):
            t = EV.per_match_table(M.played(mains[k], res), res, f"{k}_{u}", "prod_20261012+market")
            t.to_csv(RES / f"per_match_{k}_{u}.csv", index=False)
    # cumulative profit series for the figure
    series = []
    for u, (base, mains, c) in all_mains.items():
        for k, v in {"baseline_b": base, "blend_wf": mains["blend_wf"], "market_only": mains["market_only"],
                     "disagree_filter": mains["disagree_filter"]}.items():
            p = M.played(v, res).sort_values(["date", "match_id"])
            series.append(pd.DataFrame({"universe": u, "method": k, "date": p["date"],
                                        "cum_profit": p["profit"].fillna(0).cumsum().to_numpy()}))
    pd.concat(series).to_csv(RES / "cumulative_profit.csv", index=False)
    report["variants_ru"] = VARIANTS_RU
    report["n_variants"] = len(VARIANTS_RU) - 1
    C.write_json(RES / "market_aware.json", report)
    C.write_json(RES / "run_info.json", C.run_info(STUDY, started, {
        "universes": UNIVERSES, "default_method": "power", "single_book_order": M.SINGLE_BOOK_ORDER,
        "min_train_rows": M.MIN_TRAIN, "delta_grid": [None if not np.isfinite(d) else d for d in DELTA_GRID],
        "min_prior_bets": MIN_PRIOR_BETS, "v62_weights_source": str(V62_HYBRIDS.relative_to(C.ROOT)),
        "n_boot": C.N_BOOT}))
    M.preregister(RES / "preregistration.json", PREREG)
    from . import figures
    figures.run()
    log.info("done in %.0f s", time.time() - started)
    return report


if __name__ == "__main__":
    run()
