"""ИССЛЕДОВАНИЕ 5 — Timing & Closing Line Value (FootPredict v6.3).

    .venv/bin/python -m experiments.v6_3.timing.run

Light job (production outputs of the shared replay through production's market mathematics; no model fitting).

Prices are kept strictly apart by the moment they existed (UK time, ``collected_at <= moment``):
    early  = kick-off − 48 h;   T (t09) = min(match day 09:00, kick-off − 30 min) — the daily run;
    close  = the last quote before kick-off (diagnostic for a decision at T: never a price available at T).

1. What the archive can measure: quotes per selection and book, lead times, share of selections first quoted
   after T; live prediction versions (every daily run since 22.10.2026 with its own price).
2. Main generated at three moments (early / T / close), each with only the prices of its moment: Baseline B
   replayed three times, paired comparisons on the same matches.
3. The same Main picks (Baseline B at T) valued at the early, T and closing prices (timing of the price only).
4. Closing line value of the Main picks taken at T: price CLV (odds_T / odds_close − 1) and fair CLV
   (odds_T · q_close − (1 − P(refund)), q_close de-margined from a complete closing book), by provider and market;
   ROI by the sign of CLV (diagnostic).
5. Line movement and the model: does the model's disagreement with the market at T / at the opening snapshot predict
   where the market moves by kick-off?  (OddsPapi books quoted at T and again later; Football-Data Bet365
   opening → closing for every match.)
6. Live prediction versions: first vs last recorded version of a match (Main, price, ROI).
7. Price source: ROI of the same picks priced by OddsPapi bet365, Football-Data Bet365 and The Odds API at T.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.consensus import fd as FD
from experiments.v6_3.market_aware import mkt as M
from experiments.v6_3.replay import data as RD
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV

STUDY = "timing"
D = C.study_dirs(STUDY)
RES, FIG, CACHE = D["results"], D["figures"], D["cache"]


# ------------------------------------------------------------------------------------------------ 1. archive
def archive(odds: pd.DataFrame, m: pd.DataFrame) -> dict:
    by = m.set_index("match_id")
    o = odds.loc[odds["provider"] != "football_data"].copy()
    ko = pd.to_datetime(o["match_id"].map(by["kickoff"]))
    T = pd.to_datetime(o["match_id"].map(by["decision_time"]))
    o["lead_h"] = (ko - o["collected_at"]).dt.total_seconds() / 3600
    o["before_T"] = o["collected_at"] <= T
    q = o.groupby(["book", "match_id", "market", "selection"]).agg(n=("odds", "size"), first=("collected_at", "min"),
                                                                     T=("before_T", "any"))
    out = {}
    for b, g in q.groupby(level=0):
        lead = o.loc[o["book"] == b, "lead_h"]
        out[b] = {"selections": int(len(g)), "quotes": int(g["n"].sum()),
                  "share_1_quote": float((g["n"] == 1).mean()), "share_2_quotes": float((g["n"] == 2).mean()),
                  "share_3plus": float((g["n"] >= 3).mean()), "mean_quotes": float(g["n"].mean()),
                  "share_quoted_by_T": float(g["T"].mean()),
                  "lead_h_quantiles": {str(k): float(v) for k, v in lead.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()},
                  "matches": int(g.index.get_level_values(1).nunique())}
    fdq = odds.loc[odds["provider"] == "football_data"]
    out["fd_bet365"] = {"selections": int(fdq.groupby(["match_id", "market", "selection"]).ngroups),
                        "quotes": int(len(fdq)), "note": "quote time unknown (opening snapshot, nominal 00:00)"}
    lead_hist = o.loc[o["book"].isin(["bet365", "pinnacle"]) & o["lead_h"].between(0, 168)]
    hist = {b: np.histogram(g["lead_h"], bins=np.arange(0, 169, 6))[0].tolist() for b, g in lead_hist.groupby("book")}
    return {"books": out, "lead_hist": hist, "lead_bins": np.arange(0, 169, 6).tolist()}


# ------------------------------------------------------------------------------------------------ 2. three moments
def baseline_at(moment: str, m_full: pd.DataFrame, out: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Baseline B at a decision moment with the price set of that moment (production engine + Main rule)."""
    path = CACHE / f"candidates_{moment}.parquet"
    if not path.exists():
        prices = pd.read_parquet(C.SHARED_CACHE / f"prices_{ {'early': 'early', 't09': 'decision', 'close': 'closing'}[moment]}.parquet")
        batch = EN.batch_from_outputs(m_full, out, RD.lookup(prices), EN.PRODUCTION_VERSION)
        cands = EV.candidates(batch.markets, m_full, res)
        cands.to_parquet(path)
    c = pd.read_parquet(path)
    c = c.loc[c["group"] != "EXACT_SCORE"].copy()
    for k in ("probability", "odds", "push_probability"):
        c[k] = c[k].astype(float)
    c["p_model"] = c["probability"]
    return c


def moments_compare(m_full, out, res) -> tuple[dict, dict]:
    mains = {}
    for mo in ("early", "t09", "close"):
        c = baseline_at(mo, m_full, out, res)
        mains[mo] = M.played(M.main_vec(c, method=f"baseline_b_{mo}"), res)
    # the shared Baseline B tables must be reproduced for t09 / close
    chk = {}
    for mo in ("t09", "close"):
        b = M.baseline(f"baseline_b_{mo}")
        j = mains[mo].merge(b[["match_id", "market", "selection"]], on="match_id", suffixes=("", "_b"))
        chk[mo] = float(((j["market"] == j["market_b"]) & (j["selection"] == j["selection_b"])).mean())
    rep = {"self_check_same_main": chk, "summary": {}, "pairs": {}, "all_priced": {}}
    for mo, v in mains.items():
        rep["summary"][mo] = {**EV.summary(v), "roi_ci": EV.roi_with_ci(v)}
    for a, b in (("t09", "early"), ("close", "t09"), ("close", "early")):
        rep["pairs"][f"{a}_vs_{b}"] = EV.compare(mains[a], mains[b])
    priced_all = set.intersection(*[set(v.loc[v["bet"] == 1, "match_id"]) for v in mains.values()])
    for mo, v in mains.items():
        s = v.loc[v["match_id"].isin(priced_all)]
        rep["all_priced"][mo] = {**EV.summary(s), "roi_ci": EV.roi_with_ci(s)}
    for a, b in (("t09", "early"), ("close", "t09")):
        rep["all_priced"][f"{a}_vs_{b}"] = EV.compare(mains[a].loc[mains[a]["match_id"].isin(priced_all)],
                                                     mains[b].loc[mains[b]["match_id"].isin(priced_all)])
    rep["all_priced_matches"] = len(priced_all)
    same = {}
    for a, b in (("t09", "early"), ("close", "t09")):
        j = mains[a].merge(mains[b], on="match_id", suffixes=("", "_o"))
        same[f"{a}_vs_{b}"] = float(((j["market"] == j["market_o"]) & (j["selection"] == j["selection_o"])).mean())
    rep["same_main_share"] = same
    return rep, mains


# ------------------------------------------------------------------------------------------------ 3. same picks, three prices
def same_picks(main_t09: pd.DataFrame, res) -> dict:
    sets = {k: pd.read_parquet(C.SHARED_CACHE / f"prices_{v}.parquet")
            for k, v in (("early", "early"), ("t09", "decision"), ("close", "closing"))}
    t = main_t09[["match_id", "market", "selection", "group", "probability", "push_probability", "date", "league"]].copy()
    for k, p in sets.items():
        t = t.merge(p[["match_id", "market", "selection", "odds"]].rename(columns={"odds": f"odds_{k}"}),
                    on=["match_id", "market", "selection"], how="left")
    out = {}
    for combo in (("t09", "close"), ("early", "t09", "close")):
        ok = t[[f"odds_{k}" for k in combo]].notna().all(1)
        s = t.loc[ok]
        r = {"picks": int(len(s))}
        frames = {}
        for k in combo:
            f = EV.settle_rows(s.assign(odds=s[f"odds_{k}"]), res)
            frames[k] = f
            r[k] = {**EV.summary(f), "roi_ci": EV.roi_with_ci(f)}
        r["close_minus_t09"] = EV.compare(frames["close"], frames["t09"])
        if "early" in combo:
            r["t09_minus_early"] = EV.compare(frames["t09"], frames["early"])
        r["mean_log_ratio_close_t09"] = float(np.log(s["odds_close"] / s["odds_t09"]).mean())
        r["share_close_lower"] = float((s["odds_close"] < s["odds_t09"] - 1e-9).mean())
        r["share_close_higher"] = float((s["odds_close"] > s["odds_t09"] + 1e-9).mean())
        out["+".join(combo)] = r
    return out


# ------------------------------------------------------------------------------------------------ 4. CLV
def clv(main_t09: pd.DataFrame, odds: pd.DataFrame, m: pd.DataFrame, res) -> tuple[dict, pd.DataFrame]:
    close_best = pd.read_parquet(C.SHARED_CACHE / "prices_closing.parquet")
    fair_c = M.single_book(M.fair_books(M.latest_quotes(odds, M.moment(m, "close"))))
    b = main_t09.loc[main_t09["bet"] == 1].copy()
    b = b.merge(close_best[["match_id", "market", "selection", "odds", "provider"]].rename(
        columns={"odds": "odds_close", "provider": "provider_close"}), on=["match_id", "market", "selection"],
        how="left")
    b = M.attach_q(b, fair_c, "qc")
    hc = b["group"] == "HANDICAP"
    b["qc_win"] = np.where(hc, b["qc"] * (1 - b["push_probability"]), b["qc"])
    b["clv_price"] = b["odds"] / b["odds_close"] - 1
    b["clv_fair"] = b["odds"] * b["qc_win"] - (1 - b["push_probability"])
    b["provider"] = b["provider"].fillna("none")
    out = {"bets": int(len(b)), "with_close_price": int(b["odds_close"].notna().sum()),
           "with_close_fair": int(b["qc"].notna().sum())}
    for col in ("clv_price", "clv_fair"):
        v = b[col].dropna().to_numpy()
        out[col] = {"mean": float(v.mean()), "ci95": C.ci(np.concatenate([v[i].mean(1) for i in C.boot_idx(len(v))])),
                    "share_positive": float((v > 0).mean()), "n": int(len(v))}
    out["by_provider"], out["by_group"] = {}, {}
    for key, col in (("by_provider", "provider"), ("by_group", "group")):
        for k, g in b.groupby(col):
            v = g["clv_fair"].dropna()
            vp = g["clv_price"].dropna()
            out[key][k] = {"n": int(len(g)), "clv_price": float(vp.mean()) if len(vp) else None,
                           "clv_fair": float(v.mean()) if len(v) else None, "roi": float(g["profit"].sum() / len(g)),
                           "same_book_family": None}
    for sign, g in (("clv_fair>0", b.loc[b["clv_fair"] > 0]), ("clv_fair<=0", b.loc[b["clv_fair"] <= 0])):
        out[sign] = {"n": int(len(g)), **EV.roi_with_ci(g), "hit": float((g["status"] == "WON").mean())}
    out["corr_clv_profit"] = float(np.corrcoef(b.dropna(subset=["clv_fair"])["clv_fair"],
                                               b.dropna(subset=["clv_fair"])["profit"])[0, 1])
    return out, b


# ------------------------------------------------------------------------------------------------ 5. movement
def slope_ci(x, y, groups, n_boot=500) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], np.asarray(groups)[ok]
    if len(x) < 30:
        return {"n": int(len(x)), "slope": None, "ci95": [None, None]}
    def fit(i):
        xi, yi = x[i], y[i]
        xi = xi - xi.mean()
        return float((xi * (yi - yi.mean())).sum() / max((xi ** 2).sum(), 1e-12))
    g = pd.factorize(groups)[0]
    rows_of = pd.Series(np.arange(len(g))).groupby(g).apply(np.array).to_list()
    rng = np.random.default_rng(C.SEED)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_of), len(rows_of))
        bs.append(fit(np.concatenate([rows_of[i] for i in pick])))
    bs = np.array(bs)
    return {"n": int(len(x)), "slope": fit(np.arange(len(x))), "ci95": C.ci(bs),
            "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
            "mean_abs_move": float(np.abs(y).mean())}


def movement(odds, m, cands_t09) -> dict:
    out = {}
    # (a) OddsPapi / The Odds API books: complete at T and re-quoted later (closing book's latest quote > T)
    fT = M.fair_books(M.latest_quotes(odds, M.moment(m, "t09")))
    fC = M.fair_books(M.latest_quotes(odds, M.moment(m, "close")))
    j = fT.merge(fC, on=["match_id", "market", "book", "selection"], suffixes=("_t", "_c"))
    j = j.loc[j["book"] != "fd_bet365"]
    j["moved"] = j["t_max_c"] > j["t_max_t"]
    p = cands_t09[["match_id", "market", "selection", "p_model", "push_probability", "group"]]
    j = j.merge(p, on=["match_id", "market", "selection"], how="inner")
    j["p_c"] = np.where(j["group_t"] == "HANDICAP", M.cond(j["p_model"], j["push_probability"]), j["p_model"])
    first = j.drop_duplicates(["match_id", "market", "book"])        # one side per book (the first side)
    out["oddspapi_books_complete_at_T_and_close"] = int(len(first))
    out["requoted_after_T"] = int(first["moved"].sum())
    mv = first.loc[first["moved"]]
    out["oddspapi"] = {}
    for grp, g in mv.groupby("group_t"):
        out["oddspapi"][grp] = slope_ci(g["p_c"] - g["q_t"], g["q_c"] - g["q_t"], g["match_id"])
    out["oddspapi"]["all"] = slope_ci(mv["p_c"] - mv["q_t"], mv["q_c"] - mv["q_t"], mv["match_id"])
    # (b) Football-Data Bet365 / market average, opening → closing, every 2026/27 match
    fd = FD.prices(2026)
    outp = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    ou = cands_t09.loc[(cands_t09["market"] == "TOTAL_2.5") & (cands_t09["selection"] == "OVER"),
                       ["match_id", "p_model"]].rename(columns={"p_model": "p_over"})
    out["football_data"] = {}
    for book in ("bet365", "avg"):
        for market in ("1X2", "TOTAL_2.5"):
            o = fd.loc[(fd["book"] == book) & (fd["market"] == market)]
            w = o.pivot_table(index="match_id", columns="snapshot", values=["o1", "o2", "o3"], aggfunc="first")
            if ("o1", "close") not in w.columns:
                continue
            k = 3 if market == "1X2" else 2
            qo = M.demargin(np.column_stack([w[(f"o{i}", "open")] for i in range(1, k + 1)]), "power")
            qc = M.demargin(np.column_stack([w[(f"o{i}", "close")] for i in range(1, k + 1)]), "power")
            ids = w.index.to_numpy()
            if market == "1X2":
                pm = outp.set_index("match_id").reindex(ids)[["p_home", "p_draw", "p_away"]].to_numpy(float)
                x = (pm - qo)[:, [0, 2]].ravel()
                y = (qc - qo)[:, [0, 2]].ravel()
                grp = np.repeat(ids, 2)
            else:
                pm = ou.set_index("match_id").reindex(ids)["p_over"].to_numpy(float)
                x, y, grp = pm - qo[:, 0], qc[:, 0] - qo[:, 0], ids
            out["football_data"][f"{book}|{market}"] = slope_ci(x, y, grp)
            # does the movement predict the result better: closing vs opening log loss
            dfm = fd.loc[(fd["book"] == book) & (fd["market"] == market) & (fd["snapshot"] == "open")].set_index(
                "match_id").reindex(ids)
            yy = FD.outcome(dfm.reset_index())
            okr = (yy >= 0) & np.all(np.isfinite(qo), 1) & np.all(np.isfinite(qc), 1)
            out["football_data"][f"{book}|{market}"]["ll_close_minus_open"] = C.paired_mean_diff(
                M.ll_rows(qc[okr], yy[okr]), M.ll_rows(qo[okr], yy[okr]))
    return out


# ------------------------------------------------------------------------------------------------ 6. live versions
def uk_from_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC").dt.tz_convert("Europe/London").dt.tz_localize(None)


def live_versions(res) -> dict:
    cols = ["match_id", "role", "season", "version", "generated_at", "kickoff", "market", "selection", "group",
            "odds", "odds_priced_at", "provider", "probability", "date", "league", "push_probability"]
    v = pd.read_parquet(C.DATA_FILES["snapshots"].parent / "versions.parquet",
                        columns=[c for c in cols if c != "push_probability"])
    v = v.loc[(v["role"] == "main") & (v["season"] == C.CURRENT)].copy()
    v["gen_uk"] = uk_from_utc(v["generated_at"])
    v = v.loc[v["kickoff"].isna() | (v["gen_uk"] < v["kickoff"])]
    v["lead_h"] = (v["kickoff"] - v["gen_uk"]).dt.total_seconds() / 3600
    v["price_existed"] = v["odds_priced_at"].isna() | (v["odds_priced_at"] <= v["gen_uk"])
    v = v.sort_values(["match_id", "version"])
    n = v.groupby("match_id")["version"].nunique()
    multi = n.index[n >= 2]
    first = v.loc[v["match_id"].isin(multi)].drop_duplicates("match_id", keep="first")
    last = v.loc[v["match_id"].isin(multi)].drop_duplicates("match_id", keep="last")
    first, last = EV.settle_rows(first, res), EV.settle_rows(last, res)
    first = first.loc[first["status"].isin(["WON", "LOST", "PUSH"])]
    last = last.loc[last["match_id"].isin(set(first["match_id"])) & last["status"].isin(["WON", "LOST", "PUSH"])]
    first = first.loc[first["match_id"].isin(set(last["match_id"]))]
    j = first.merge(last, on="match_id", suffixes=("_f", "_l"))
    same = (j["market_f"] == j["market_l"]) & (j["selection_f"] == j["selection_l"])
    sp = j.loc[same & j["odds_f"].notna() & j["odds_l"].notna()]
    out = {"matches_with_versions": int(v["match_id"].nunique()), "matches_2plus": int(len(multi)),
           "settled_pairs": int(len(j)), "versions_per_match": n.value_counts().sort_index().to_dict(),
           "lead_h_first_median": float(first["lead_h"].median()), "lead_h_last_median": float(last["lead_h"].median()),
           "same_main_share": float(same.mean()),
           "same_pick_price_change": {"n": int(len(sp)), "mean_log_ratio_last_first": float(np.log(sp["odds_l"] / sp["odds_f"]).mean()) if len(sp) else None,
                                      "share_last_lower": float((sp["odds_l"] < sp["odds_f"] - 1e-9).mean()) if len(sp) else None,
                                      "share_last_higher": float((sp["odds_l"] > sp["odds_f"] + 1e-9).mean()) if len(sp) else None},
           "price_existed_share_first": float(first["price_existed"].mean()),
           "price_existed_share_last": float(last["price_existed"].mean())}
    # honest prices only: a price is a bet only if it existed when the version was generated
    f2 = first.assign(bet=first["bet"] * first["price_existed"].astype(int))
    l2 = last.assign(bet=last["bet"] * last["price_existed"].astype(int))
    f2.loc[f2["bet"] == 0, "profit"] = np.nan
    l2.loc[l2["bet"] == 0, "profit"] = np.nan
    out["first"] = {**EV.summary(f2), "roi_ci": EV.roi_with_ci(f2)}
    out["last"] = {**EV.summary(l2), "roi_ci": EV.roi_with_ci(l2)}
    out["last_vs_first"] = EV.compare(l2, f2)
    both = set(f2.loc[f2["bet"] == 1, "match_id"]) & set(l2.loc[l2["bet"] == 1, "match_id"])
    out["both_priced"] = {"n": len(both), "first": EV.summary(f2.loc[f2["match_id"].isin(both)]),
                          "last": EV.summary(l2.loc[l2["match_id"].isin(both)]),
                          "last_vs_first": EV.compare(l2.loc[l2["match_id"].isin(both)],
                                                      f2.loc[f2["match_id"].isin(both)])}
    return out


# ------------------------------------------------------------------------------------------------ 7. sources
def sources(main_t09: pd.DataFrame, odds: pd.DataFrame, m: pd.DataFrame, res) -> dict:
    q = M.latest_quotes(odds, M.moment(m, "t09"))
    q["src"] = np.where(q["provider"] == "oddspapi", "oddspapi:" + q["book"],
                        np.where(q["provider"] == "football_data", "football_data:bet365", "the_odds_api:" + q["book"]))
    w = q.pivot_table(index=["match_id", "market", "selection"], columns="src", values="odds", aggfunc="last")
    t = main_t09[["match_id", "market", "selection", "group", "probability", "push_probability", "status",
                  "provider", "date"]].merge(w.reset_index(), on=["match_id", "market", "selection"], how="left")
    out = {"main_by_provider": {}, "pairs": {}}
    for k, g in main_t09.assign(provider=main_t09["provider"].fillna("none")).groupby("provider"):
        out["main_by_provider"][k] = {**EV.summary(g), "roi_ci": EV.roi_with_ci(g)}
    srcs = [c for c in w.columns if (t[c].notna().sum() >= 30)]
    out["availability_at_T"] = {c: int(t[c].notna().sum()) for c in w.columns}
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            s = t.loc[t[a].notna() & t[b].notna()]
            if len(s) < 30:
                continue
            fa = EV.settle_rows(s.assign(odds=s[a]), res)
            fb = EV.settle_rows(s.assign(odds=s[b]), res)
            out["pairs"][f"{a} vs {b}"] = {"n": int(len(s)), "mean_log_ratio": float(np.log(s[a] / s[b]).mean()),
                                           "roi_a": EV.summary(fa)["roi"], "roi_b": EV.summary(fb)["roi"],
                                           "diff": EV.compare(fa, fb)["roi_diff"]}
    return out


def run() -> dict:
    started = time.time()
    log = C.setup_logging(STUDY, D["logs"] / "run.log")
    m = M.matches()
    m_full = pd.read_parquet(C.SHARED_CACHE / "matches.parquet")
    res = M.results()
    out_prod = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    odds = M.load_odds()
    rep = {"archive": archive(odds, m)}
    log.info("archive done")
    rep["moments"], mains = moments_compare(m_full, out_prod, res)
    log.info("moments %s", {k: round(v["roi"] or 0, 4) for k, v in rep["moments"]["summary"].items()})
    rep["same_picks"] = same_picks(mains["t09"], res)
    rep["clv"], clv_tab = clv(mains["t09"], odds, m, res)
    log.info("clv %s", rep["clv"]["clv_fair"])
    c09 = baseline_at("t09", m_full, out_prod, res)
    rep["movement"] = movement(odds, m, c09)
    log.info("movement done")
    rep["live_versions"] = live_versions(res)
    log.info("live versions %s", rep["live_versions"]["last_vs_first"]["roi_diff"])
    rep["sources"] = sources(mains["t09"], odds, m, res)
    pv = {k: v["roi_diff"]["p"] for k, v in rep["moments"]["pairs"].items()}
    pv.update({f"all_priced|{k}": v["roi_diff"]["p"] for k, v in rep["moments"]["all_priced"].items() if "_vs_" in k})
    pv["same_picks|close_minus_t09"] = rep["same_picks"]["t09+close"]["close_minus_t09"]["roi_diff"]["p"]
    pv["live|last_vs_first"] = rep["live_versions"]["last_vs_first"]["roi_diff"]["p"]
    rep["holm"] = {"n_tests": len(pv), "p": pv, "p_holm": C.holm(pv)}
    a = M.baseline("baseline_a")
    rep["baseline_a"] = {"summary": EV.summary(a), "roi_ci": EV.roi_with_ci(a)}
    # outputs
    cov = []
    for mo, v in mains.items():
        cov.append({"moment": mo, "season_matches": int(len(m)), "played": int(len(res)), "processed": int(len(v)),
                    "bets_real_price": int(v["bet"].sum()), "no_price_at_moment": int(v["odds"].isna().sum()),
                    "push": int((v["status"] == "PUSH").sum()), "without_statistics": 0,
                    "not_played_excluded": int(len(m) - len(res))})
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)
    for mo, v in mains.items():
        EV.per_match_table(v, res, f"baseline_b_{mo}", EN.PRODUCTION_VERSION).to_csv(
            RES / f"per_match_baseline_b_{mo}.csv", index=False)
    clv_tab[["match_id", "date", "league", "market", "selection", "probability", "odds", "provider", "odds_close",
             "provider_close", "qc", "qc_book", "clv_price", "clv_fair", "status", "profit"]].to_csv(
        RES / "clv_main_t09.csv", index=False)
    C.write_json(RES / "timing.json", rep)
    C.write_json(RES / "run_info.json", C.run_info(STUDY, started, {
        "moments": {"early": "kick-off - 48 h", "t09": "min(match day 09:00 UK, kick-off - 30 min)",
                    "close": "last quote before kick-off (diagnostic for a decision at T)"},
        "fair_method": "power", "single_book_order": M.SINGLE_BOOK_ORDER, "n_boot": C.N_BOOT,
        "live_versions": "local_data/predictions/versions.parquet (read only)"}))
    M.preregister(RES / "preregistration.json", [])
    from . import figures
    figures.run()
    log.info("done in %.0f s", time.time() - started)
    return rep


if __name__ == "__main__":
    run()
