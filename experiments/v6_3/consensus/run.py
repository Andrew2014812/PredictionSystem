"""ИССЛЕДОВАНИЕ 13 — Market Consensus (FootPredict v6.3).

    .venv/bin/python -m experiments.v6_3.consensus.run

Light job (no model fitting). Several bookmakers at once, never mixing quotes of different times:

1. Availability of every book by market group and league at the decision moment T and at the close (stored quotes:
   OddsPapi bet365 / Pinnacle / William Hill, The Odds API books, Football-Data Bet365) and in Football-Data's
   multi-book snapshots (opening and closing; all books of one snapshot were quoted at the same moment).
2. Consensus rules on Football-Data snapshots 2021/22–2026/27 (market only, no model): single books, mean, median,
   logit mean, 1/margin-weighted mean, Football-Data average and maximum. The rule is chosen on 2021/22–2025/26 and
   checked on 2026/27. Dispersion between books as an uncertainty signal (excess log loss and calibration slope by
   dispersion quintile).
3. Decision-time consensus from the stored quotes: per match each book's latest quote at or before the moment; the
   time gap between books is recorded and the result is shown by gap.
4. Consensus and the model (2026/27, both universes): out-of-sample (monthly expanding window) log loss of the model,
   single book, consensus and their blends; calibration; Main variants (consensus as market probability, blend with
   the consensus, dispersion filter, best price among books), paired with Baseline B, Holm over all tests.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.market_aware import mkt as M
from experiments.v6_3.replay import evaluate as EV
from src.markets.markets import handicap_market

from . import fd as FD

STUDY = "consensus"
D = C.study_dirs(STUDY)
RES, FIG = D["results"], D["figures"]
UNIVERSES = ("close", "t09")
SNAP = {"close": "close", "t09": "open"}
REF = {"max", "avg", "betfair_exchange"}
RULES = ("bet365", "pinnacle", "mean", "median", "logit_mean", "inv_margin", "fd_avg", "fd_max", "exchange")
RULES_RU = {"bet365": "Bet365", "pinnacle": "Pinnacle", "mean": "среднее", "median": "медиана",
            "logit_mean": "среднее логитов", "inv_margin": "взвеш. 1/маржа", "fd_avg": "Avg Football-Data",
            "fd_max": "Max Football-Data", "exchange": "Betfair Exchange"}
SELECT_SEASONS = (2021, 2022, 2023, 2024, 2025)
GAP_BINS = [-0.01, 1, 6, 24, np.inf]
GAP_LABELS = ["≤ 1 ч", "1–6 ч", "6–24 ч", "> 24 ч"]
VARIANTS_RU = {"cons_market_only": "C1 консенсус вместо p", "cons_blend_wf": "C2 смесь модели с консенсусом",
               "single_blend_wf": "S2 смесь с одной книгой (= V1 исслед. 3)",
               "single_market_only": "S1 одна книга вместо p (= V3)",
               "dispersion_filter": "C3 фильтр разброса книг", "best_price": "C4 лучшая цена среди книг"}


PREREG = [{
    "id": "v63-consensus-C2",
    "hypothesis": "Blending the model with a multi-book consensus (instead of one book) raises Main ROI versus "
                  "Baseline B without lowering the hit rate.",
    "rule": "Production Main rule on p = softmax(a + b_q log q_cons + b_p log p_model) per market group, a, b fitted "
            "each month on all settled earlier matches of the season (>= 300 rows, else the model probability); "
            "q_cons = median of the de-margined (power) named books of the Football-Data snapshot for 1X2 / double "
            "chance, Football-Data average for total 2.5 and the Asian line, median of >= 2 stored books, else the "
            "single book (bet365).",
    "primary_metric": "paired ROI difference vs Baseline B at the closing universe (bootstrap by match)",
    "secondary": ["the same at the T universe", "ROI difference vs the single-book blend (V1)", "hit rate"],
    "minimum_forward_sample": "1000 settled bets (about 2 months of the 21 leagues)",
    "decision": "B (promising, not adopted)"}, {
    "id": "v63-consensus-C1",
    "hypothesis": "Using the multi-book consensus probability instead of the model probability in the production "
                  "Main rule does not lower Main ROI versus Baseline B and beats the single-book market version.",
    "rule": "Production Main rule on p = q_cons (as in v63-consensus-C2, handicap on the win scale with the model's "
            "refund probability); candidates without any complete book keep the model probability.",
    "primary_metric": "paired ROI difference vs Baseline B at the closing universe (bootstrap by match)",
    "secondary": ["the same at the T universe", "ROI difference vs the single-book market version (V3)"],
    "minimum_forward_sample": "1000 settled bets",
    "decision": "B (promising, not adopted)"}]


# ------------------------------------------------------------------------------------------------ helpers
def wide_fd(fd: pd.DataFrame, market: str, snapshot: str):
    """Arrays of one FD market / snapshot: ids, books, Q (n, B, k) de-margined (power), margin (n, B), y."""
    d = fd.loc[(fd["market"] == market) & (fd["snapshot"] == snapshot)]
    if d.empty:
        return None
    k = 3 if market == "1X2" else 2
    books = sorted(d["book"].unique())
    ids = d.drop_duplicates("match_id")[["match_id", "season", "league", "date", "hg", "ag", "market", "line"]]
    ids = ids.reset_index(drop=True)
    pos = pd.Series(np.arange(len(ids)), index=ids["match_id"])
    Q = np.full((len(ids), len(books), k), np.nan)
    O = np.full((len(ids), len(books), k), np.nan)
    for j, b in enumerate(books):
        g = d.loc[d["book"] == b]
        o = g[["o1", "o2", "o3"][:k]].to_numpy(float)
        if market == "AH":       # the snapshot line must be the same for the book (one line per snapshot in FD)
            pass
        i = pos.reindex(g["match_id"]).to_numpy()
        Q[i, j] = M.demargin(o, "power")
        O[i, j] = o
    y = FD.outcome(ids.assign(market=market))
    return {"ids": ids, "books": books, "Q": Q, "O": O, "y": y, "k": k}


def consensus(w: dict, rule: str) -> np.ndarray:
    books, Q, O = w["books"], w["Q"], w["O"]
    named = np.array([b not in REF for b in books])
    Qn = Q[:, named]
    if rule in ("bet365", "pinnacle"):
        return Q[:, books.index(rule)] if rule in books else np.full(Q[:, 0].shape, np.nan)
    if rule == "fd_avg":
        return Q[:, books.index("avg")] if "avg" in books else np.full(Q[:, 0].shape, np.nan)
    if rule == "fd_max":
        return Q[:, books.index("max")] if "max" in books else np.full(Q[:, 0].shape, np.nan)
    if rule == "exchange":
        return Q[:, books.index("betfair_exchange")] if "betfair_exchange" in books else np.full(Q[:, 0].shape,
                                                                                                 np.nan)
    with np.errstate(all="ignore"):
        if rule == "mean":
            c = np.nanmean(Qn, 1)
        elif rule == "median":
            c = np.nanmedian(Qn, 1)
        elif rule == "logit_mean":
            c = np.exp(np.nanmean(np.log(np.clip(Qn, 1e-6, 1)), 1))
        elif rule == "inv_margin":
            mg = np.nansum(1 / O[:, named], 2) - 1
            wgt = np.where(np.isfinite(Qn[:, :, 0]), 1 / np.clip(mg, 0.005, None), 0.0)
            c = np.nansum(np.nan_to_num(Qn) * wgt[:, :, None], 1) / wgt.sum(1, keepdims=True)
        else:
            raise ValueError(rule)
    return c / c.sum(1, keepdims=True)


def n_named(w: dict) -> np.ndarray:
    named = np.array([b not in REF for b in w["books"]])
    return np.isfinite(w["Q"][:, named, 0]).sum(1)


def dispersion(w: dict) -> np.ndarray:
    """Mean over sides of the standard deviation of the fair probability across named books."""
    named = np.array([b not in REF for b in w["books"]])
    with np.errstate(all="ignore"):
        return np.nanmean(np.nanstd(w["Q"][:, named], 1, ddof=1), 1)


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


# ------------------------------------------------------------------------------------------------ 1. availability
def availability(odds, m, fd26) -> tuple[pd.DataFrame, dict]:
    rows = []
    played = set(m.loc[m["played"].astype(bool), "match_id"])
    lg = m.set_index("match_id")["league"]
    n_lg = m.loc[m["played"].astype(bool)].groupby("league").size()
    summ = {}
    for u in UNIVERSES:
        f = M.fair_books(M.latest_quotes(odds, M.moment(m, u)))
        f = pd.concat([f, M.dc_from_1x2(f)], ignore_index=True)
        f = f.loc[f["match_id"].isin(played)].drop_duplicates(["match_id", "group", "book"])
        f["league"] = f["match_id"].map(lg)
        for (b, g, l_), x in f.groupby(["book", "group", "league"]):
            rows.append({"moment": u, "source": "stored quotes", "book": b, "group": g, "league": l_,
                         "matches": int(x["match_id"].nunique()), "share": float(x["match_id"].nunique() / n_lg[l_])})
        s = f.groupby(["book", "group"])["match_id"].nunique().unstack(fill_value=0)
        summ[u] = (s / len(played)).round(4).to_dict("index")
        mb = f.loc[f["group"].isin(["1X2", "TOTAL", "HANDICAP"])].groupby(["match_id", "market"])["book"].nunique()
        summ[f"{u}_books_per_market"] = mb.value_counts().sort_index().to_dict()
    fd = fd26.loc[fd26["match_id"].isin(played)].copy()
    fd["league"] = fd["match_id"].map(lg)
    for (snap, b, mk, l_), x in fd.groupby(["snapshot", "book", "market", "league"]):
        rows.append({"moment": f"fd_{snap}", "source": "Football-Data", "book": b, "group": mk, "league": l_,
                     "matches": int(x["match_id"].nunique()), "share": float(x["match_id"].nunique() / n_lg[l_])})
    s = fd.groupby(["snapshot", "book", "market"])["match_id"].nunique().unstack(fill_value=0)
    summ["football_data_2026"] = {f"{a}|{b}": v for (a, b), v in (s / len(played)).round(4).to_dict("index").items()}
    return pd.DataFrame(rows), summ


# ------------------------------------------------------------------------------------------------ 2. FD rules
def fd_rules(fd_all: pd.DataFrame) -> dict:
    out = {"by_key": {}, "selection": {}, "dispersion": {}}
    for market in ("1X2", "TOTAL_2.5", "AH"):
        for snap in ("open", "close"):
            pooled = {r: [] for r in RULES}
            for s in FD.SEASONS:
                w = wide_fd(fd_all.loc[fd_all["season"] == s], market, snap)
                if w is None:
                    continue
                cons = {r: consensus(w, r) for r in RULES}
                need_multi = market == "1X2"
                base_ok = (w["y"] >= 0) & np.all(np.isfinite(cons["bet365"]), 1)
                if need_multi:
                    base_ok &= n_named(w) >= 3
                key = f"{market}|{snap}|{s}"
                rr = {"n": int(base_ok.sum()), "books": [b for b in w["books"]]}
                ll_b = M.ll_rows(cons["bet365"][base_ok], w["y"][base_ok])
                for r in RULES:
                    ok = base_ok & np.all(np.isfinite(cons[r]), 1)
                    if ok.sum() < 200:
                        continue
                    ll = M.ll_rows(cons[r][ok], w["y"][ok])
                    llb = M.ll_rows(cons["bet365"][ok], w["y"][ok])
                    rr[r] = {"n": int(ok.sum()), "ll": float(ll.mean()),
                             "minus_bet365": C.paired_mean_diff(ll, llb) if r != "bet365" else None}
                    if s in SELECT_SEASONS and ok.sum() == base_ok.sum():
                        pooled[r].append(ll)
                out["by_key"][key] = rr
                if market == "1X2":
                    dsp = dispersion(w)
                    okd = base_ok & np.isfinite(dsp)
                    out["dispersion"][f"{snap}|{s}"] = disp_quintiles(cons["median"][okd], w["y"][okd], dsp[okd])
            avail = {r: np.concatenate(v) for r, v in pooled.items() if len(v) == len(SELECT_SEASONS)}
            if avail:
                best = min(avail, key=lambda r: avail[r].mean())
                out["selection"][f"{market}|{snap}"] = {"rules_ll_2021_2025": {r: float(v.mean()) for r, v in
                                                                               avail.items()},
                                                        "chosen": best}
    return out


def disp_quintiles(P, y, dsp) -> dict:
    q = pd.qcut(dsp, 5, labels=False, duplicates="drop")
    out = {}
    for b in sorted(np.unique(q)):
        i = q == b
        ll = M.ll_rows(P[i], y[i])
        ex = ll - entropy(P[i])
        th = M.pool_fit([M.logs(P[i])], y[i])
        out[str(int(b) + 1)] = {"n": int(i.sum()), "disp_mean": float(dsp[i].mean()), "ll": float(ll.mean()),
                                "excess": C.paired_mean_diff(ll, entropy(P[i])),
                                "calib_slope": float(th[P.shape[1] - 1]), "fav_p": float(P[i].max(1).mean())}
    return out


# ------------------------------------------------------------------------------------------------ 3. decision-time
def decision_consensus(odds, m) -> tuple[dict, dict]:
    """Per universe: complete books per (match, market) from the stored quotes (bet365 of OddsPapi and of
    Football-Data merged into one bookmaker, latest quote at or before the moment), median consensus, the time gap
    between the books' latest quotes."""
    out, tabs = {}, {}
    played = set(m.loc[m["played"].astype(bool), "match_id"])
    for u in UNIVERSES:
        q = M.merge_bet365(M.latest_quotes(odds, M.moment(m, u)))
        f = M.fair_books(q)
        f = f.loc[f["match_id"].isin(played)]
        g = f.groupby(["match_id", "market", "selection"])
        t = g.agg(q_med=("q", "median"), q_mean=("q", "mean"), q_sd=("q", "std"), books=("book", "nunique"),
                  t_hi=("t_max", "max"), t_lo=("t_max", "min"))
        t["gap_h"] = (pd.to_datetime(t["t_hi"]) - pd.to_datetime(t["t_lo"])).dt.total_seconds() / 3600
        b365 = f.loc[f["book"] == "bet365"].set_index(["match_id", "market", "selection"])["q"]
        t["q_bet365"] = b365.reindex(t.index)
        t = t.reset_index()
        tabs[u] = t
        multi = t.loc[t["books"] >= 2]
        rr = {"markets_multi_book": int(multi.groupby(["match_id", "market"]).ngroups),
              "matches_multi_book": int(multi["match_id"].nunique()),
              "by_group": multi.assign(group=multi["market"].map(M.group_of)).groupby("group").apply(
                  lambda x: x.groupby(["match_id", "market"]).ngroups).to_dict(),
              "gap_quantiles_h": {str(k): float(v) for k, v in
                                  multi.drop_duplicates(["match_id", "market"])["gap_h"].quantile(
                                      [0.25, 0.5, 0.75, 0.9]).items()}}
        out[u] = rr
    return out, tabs


def gap_sensitivity(tab: pd.DataFrame, res: pd.DataFrame) -> dict:
    """Log loss of the median consensus vs bet365 alone on markets with ≥ 2 books, by time gap between books."""
    t = tab.loc[(tab["books"] >= 2) & tab["q_bet365"].notna()].copy()
    t["group"] = t["market"].map(M.group_of)
    out = {}
    for grp, sides in M.SIDES.items():
        g = t.loc[t["group"] == grp]
        if g.empty:
            continue
        wq = g.pivot_table(index=["match_id", "market"], columns="selection", values="q_med")
        wb = g.pivot_table(index=["match_id", "market"], columns="selection", values="q_bet365")
        gap = g.drop_duplicates(["match_id", "market"]).set_index(["match_id", "market"])["gap_h"].reindex(wq.index)
        if not all(s in wq for s in sides):
            continue
        Q, B = wq[list(sides)].to_numpy(float), wb[list(sides)].to_numpy(float)
        Q = Q / Q.sum(1, keepdims=True)
        y = outcome_side(wq.index, grp, sides, res)
        ok = (y >= 0) & np.all(np.isfinite(Q), 1) & np.all(np.isfinite(B), 1)
        rr = {"n": int(ok.sum())}
        if ok.sum() >= 30:
            rr["all"] = C.paired_mean_diff(M.ll_rows(Q[ok], y[ok]), M.ll_rows(B[ok], y[ok]))
        bins = pd.cut(gap.to_numpy(), GAP_BINS, labels=GAP_LABELS)
        for lab in GAP_LABELS:
            i = ok & (bins == lab)
            if i.sum() >= 30:
                rr[lab] = C.paired_mean_diff(M.ll_rows(Q[i], y[i]), M.ll_rows(B[i], y[i]))
            else:
                rr[lab] = {"n": int(i.sum()), "diff": None}
        out[grp] = rr
    return out


def outcome_side(index, grp, sides, res) -> np.ndarray:
    from src.markets.markets import settle_selection
    r = res.reindex(index.get_level_values(0))
    y = np.full(len(index), -1)
    for i, (mk, hg, ag, co) in enumerate(zip(index.get_level_values(1), r["home_goals"], r["away_goals"],
                                             r["total_corners"])):
        if hg != hg:
            continue
        st = [settle_selection(mk, s, float(hg), float(ag), None if co != co else float(co)) for s in sides]
        if st.count("WON") == 1:
            y[i] = st.index("WON")
    return y


# ------------------------------------------------------------------------------------------------ 4. model + consensus
def fd_consensus_long(fd26: pd.DataFrame, snapshot: str, rule_1x2: str) -> pd.DataFrame:
    """Candidate-keyed consensus from a Football-Data snapshot: 1X2 by the chosen rule over the named books; total 2.5
    and the Asian line by Football-Data's market average (the only multi-book figure of those markets in 2026/27)."""
    parts = []
    for market, rule, sides in (("1X2", rule_1x2, ("H", "D", "A")), ("TOTAL_2.5", "fd_avg", ("OVER", "UNDER")),
                                ("AH", "fd_avg", ("H", "A"))):
        w = wide_fd(fd26, market, snapshot)
        if w is None:
            continue
        cq = consensus(w, rule)
        dsp = dispersion(w) if market == "1X2" else np.full(len(cq), np.nan)
        nb = n_named(w) if market == "1X2" else np.full(len(cq), np.nan)
        mk = (w["ids"]["line"].map(lambda x: handicap_market(x) if x == x else None) if market == "AH"
              else pd.Series(market, index=w["ids"].index))
        for j, s in enumerate(sides):
            parts.append(pd.DataFrame({"match_id": w["ids"]["match_id"], "market": mk.to_numpy(), "selection": s,
                                       "q_fd": cq[:, j], "disp_fd": dsp, "books_fd": nb}))
    out = pd.concat(parts, ignore_index=True)
    return out.loc[out["q_fd"].notna() & out["market"].notna()]


def consensus_candidates(u: str, odds, m, fd26, rule_1x2: str, dec_tab: pd.DataFrame) -> pd.DataFrame:
    """Candidates of a universe with: q (single book, study 3), q_cons (FD snapshot consensus, else decision-time
    median over ≥ 2 stored books, else the single book), its source, the dispersion and number of books."""
    from experiments.v6_3.market_aware.run import prepared
    c = prepared(u, "power", odds, m)
    fdc = fd_consensus_long(fd26, SNAP[u], rule_1x2)
    # double chance from the FD 1X2 consensus
    x = fdc.loc[fdc["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="q_fd")
    dsp1 = fdc.loc[fdc["market"] == "1X2"].drop_duplicates("match_id").set_index("match_id")
    dc = pd.concat([pd.DataFrame({"match_id": x.index, "market": "DOUBLE_CHANCE", "selection": s,
                                  "q_fd": (x[a] + x[b]).to_numpy(),
                                  "disp_fd": dsp1["disp_fd"].reindex(x.index).to_numpy(),
                                  "books_fd": dsp1["books_fd"].reindex(x.index).to_numpy()})
                    for s, (a, b) in {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}.items()])
    fdc = pd.concat([fdc, dc], ignore_index=True)
    c = c.merge(fdc, on=["match_id", "market", "selection"], how="left")
    dt = dec_tab.loc[dec_tab["books"] >= 2, ["match_id", "market", "selection", "q_med", "q_sd", "books", "gap_h"]]
    dt = dt.rename(columns={"q_med": "q_dec", "q_sd": "disp_dec", "books": "books_dec", "gap_h": "gap_dec"})
    c = c.merge(dt, on=["match_id", "market", "selection"], how="left")
    # a consensus must be a complete market: renormalise the decision-time medians within the market
    src = np.where(c["q_fd"].notna(), "football_data", np.where(c["q_dec"].notna(), "stored_multi",
                                                                np.where(c["q"].notna(), "single", "none")))
    c["q_cons"] = np.where(src == "football_data", c["q_fd"], np.where(src == "stored_multi", c["q_dec"], c["q"]))
    c["cons_source"] = src
    c["disp"] = np.where(src == "football_data", c["disp_fd"], np.where(src == "stored_multi", c["disp_dec"], np.nan))
    c["n_books"] = np.where(src == "football_data", c["books_fd"], np.where(src == "stored_multi", c["books_dec"],
                                                                            np.where(c["q"].notna(), 1, 0)))
    # renormalise per (match, market) for partition markets (median of several books need not sum to 1)
    part = c["group"].isin(list(M.SIDES))
    s = c.loc[part].groupby(["match_id", "market"])["q_cons"].transform("sum")
    c.loc[part, "q_cons"] = c.loc[part, "q_cons"] / s
    hc = c["group"] == "HANDICAP"
    c["q_cons_win"] = np.where(hc, c["q_cons"] * (1 - c["push_probability"]), c["q_cons"])
    return c


def ece(p, y, bins=10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        i = idx == b
        if i.any():
            tot += i.sum() * abs(p[i].mean() - y[i].mean())
    return float(tot / len(p))


def model_and_consensus(c: pd.DataFrame) -> dict:
    tab_s = M.market_table(c.loc[c["q_cons"].notna()], "q")
    tab_c = M.market_table(c.loc[c["q"].notna()], "q_cons")
    out = {"walk_forward": {}, "calibration": {}}
    bl_s, _ = M.walk_forward_pool(tab_s, "blend")
    bl_c, coefs = M.walk_forward_pool(tab_c, "blend")
    out["coefs"] = coefs
    for grp in tab_s:
        if grp not in tab_c:
            continue
        a, b = tab_s[grp], tab_c[grp]
        ia = pd.Index(a["index"])
        ib = pd.Index(b["index"])
        common = ia.intersection(ib)
        pa, pb = ia.get_indexer(common), ib.get_indexer(common)
        y = a["y"][pa]
        ok = (y >= 0) & (a["month"][pa] >= "2026-09")
        pa, pb, y = pa[ok], pb[ok], y[ok]
        ll = {"model": M.ll_rows(a["P"][pa], y), "single": M.ll_rows(a["Q"][pa], y),
              "consensus": M.ll_rows(b["Q"][pb], y), "blend_single": M.ll_rows(bl_s[grp][pa], y),
              "blend_consensus": M.ll_rows(bl_c[grp][pb], y)}
        out["walk_forward"][grp] = {"n": int(len(y)), **{k: float(v.mean()) for k, v in ll.items()},
                                    "consensus_minus_single": C.paired_mean_diff(ll["consensus"], ll["single"]),
                                    "blend_cons_minus_blend_single": C.paired_mean_diff(ll["blend_consensus"],
                                                                                        ll["blend_single"]),
                                    "blend_cons_minus_consensus": C.paired_mean_diff(ll["blend_consensus"],
                                                                                     ll["consensus"])}
    # calibration of the candidate sides with all three probabilities (binary, every side)
    d = c.loc[c["q"].notna() & c["q_cons"].notna() & c["status"].isin(["WON", "LOST"]) &
              ~c["group"].isin(["EHANDICAP", "EXACT_SCORE"])]
    y = (d["status"] == "WON").to_numpy(float)
    for name, col in (("model", "p_model"), ("single", "q_win"), ("consensus", "q_cons_win")):
        p = d[col].to_numpy(float)
        out["calibration"][name] = {"ece": ece(p, y), "brier": float(((p - y) ** 2).mean()), "n": int(len(d))}
        pb = pd.cut(p, np.linspace(0, 1, 11))
        out["calibration"][name]["curve"] = pd.DataFrame({"p": p, "y": y, "b": pb}).groupby(
            "b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).reset_index(drop=True).to_dict(
            "records")
    for src, g in c.loc[c["status"].isin(["WON", "LOST"])].groupby("cons_source"):
        out.setdefault("sources", {})[src] = int(g.drop_duplicates(["match_id", "market"]).shape[0])
    return out


def dispersion_filter(c: pd.DataFrame, res) -> tuple[pd.DataFrame, dict]:
    """Production rule without candidates whose market has a book dispersion above δ; δ for month M chosen among
    the 50/75/90 % quantiles of the dispersion of earlier months (and ∞) by the Main ROI of the earlier months."""
    from experiments.v6_3.market_aware.run import MIN_PRIOR_BETS, roi_by_month
    dsp = c["disp"].to_numpy(float)
    months = sorted(c["month"].unique())
    chosen, parts = {}, []
    for mth in months:
        prior = c.loc[c["month"] < mth, "disp"].dropna()
        grid = [np.inf] + ([float(prior.quantile(q)) for q in (0.5, 0.75, 0.9)] if len(prior) > 500 else [])
        mains = {d: M.main_vec(c, "p_model", extra_eligible=~np.isfinite(dsp) | (dsp <= d)) for d in grid}
        best = np.inf
        if len(grid) > 1:
            rb = {d: roi_by_month(mains[d], res) for d in grid}
            prior_b = {d: rb[d].loc[rb[d].index < mth] for d in grid}
            if prior_b[np.inf]["bets"].sum() >= MIN_PRIOR_BETS:
                best = max(grid, key=lambda d: prior_b[d]["profit"].sum() / max(prior_b[d]["bets"].sum(), 1))
        chosen[mth] = None if not np.isfinite(best) else best
        ids = set(c.loc[c["month"] == mth, "match_id"])
        parts.append(mains[best].loc[mains[best]["match_id"].isin(ids)])
    return pd.concat(parts, ignore_index=True), {"delta_by_month": chosen}


def best_price(u: str, c: pd.DataFrame, odds, m, fd26) -> pd.DataFrame:
    """Each candidate's best price among the named books quoting it at the decision moment (stored quotes at or
    before the moment + the Football-Data snapshot of the universe for 1X2 / total 2.5 / Asian line)."""
    q = M.latest_quotes(odds, M.moment(m, u))
    best = q.groupby(["match_id", "market", "selection"]).agg(best=("odds", "max")).reset_index()
    parts = []
    snap = fd26.loc[(fd26["snapshot"] == SNAP[u]) & ~fd26["book"].isin(REF)]
    for market, sides in (("1X2", ("H", "D", "A")), ("TOTAL_2.5", ("OVER", "UNDER")), ("AH", ("H", "A"))):
        g = snap.loc[snap["market"] == market]
        mk = g["line"].map(lambda x: handicap_market(x) if x == x else None) if market == "AH" else market
        for j, s in enumerate(sides):
            parts.append(pd.DataFrame({"match_id": g["match_id"], "market": mk, "selection": s,
                                       "best": g[f"o{j + 1}"]}))
    fdb = pd.concat(parts).dropna().groupby(["match_id", "market", "selection"])["best"].max().reset_index()
    allb = pd.concat([best, fdb]).groupby(["match_id", "market", "selection"])["best"].max().reset_index()
    out = c.merge(allb, on=["match_id", "market", "selection"], how="left")
    out["odds_production"] = out["odds"]
    out["odds"] = np.fmax(out["odds"].to_numpy(float), out["best"].to_numpy(float))
    return EV.settle_rows(out.drop(columns=["status", "profit", "bet"]), M.results())


def main_variants(u: str, c: pd.DataFrame, odds, m, fd26, res) -> tuple[dict, dict, dict]:
    base = M.baseline(f"baseline_b_{u}")
    mains, extra = {}, {}
    c = c.copy()
    c["p_cons_mkt"] = c["q_cons_win"].fillna(c["p_model"])
    c["p_single_mkt"] = c["q_win"].fillna(c["p_model"])
    mains["cons_market_only"] = M.main_vec(c, "p_cons_mkt", method="cons_market_only")
    mains["single_market_only"] = M.main_vec(c, "p_single_mkt", method="single_market_only")
    tab_c = M.market_table(c, "q_cons")
    bl, _ = M.walk_forward_pool(tab_c, "blend")
    c = M.back_to_candidates(c, tab_c, bl, "p_cons_blend")
    mains["cons_blend_wf"] = M.main_vec(c, "p_cons_blend", method="cons_blend_wf")
    tab_s = M.market_table(c, "q")
    bl_s, _ = M.walk_forward_pool(tab_s, "blend")
    c = M.back_to_candidates(c, tab_s, bl_s, "p_single_blend")
    mains["single_blend_wf"] = M.main_vec(c, "p_single_blend", method="single_blend_wf")
    f, info = dispersion_filter(c, res)
    mains["dispersion_filter"] = f.assign(method="dispersion_filter")
    extra["dispersion_filter"] = info
    cb = best_price(u, c, odds, m, fd26)
    mains["best_price"] = M.main_vec(cb, "p_model", method="best_price")
    bp = M.played(mains["best_price"], res)
    extra["best_price_uplift"] = {"mains_with_better_price": int((bp["odds"] > bp["odds_production"] + 1e-9).sum()),
                                  "mean_log_ratio": float(np.log(bp["odds"] / bp["odds_production"]).mean())}
    out = {}
    for k, v in mains.items():
        out[k] = M.evaluate_variant(v, base, res)
        out[k]["stability"] = M.stability(v, base, res)
    out["baseline_b"] = {"summary": EV.summary(M.played(base, res)), "roi_ci": EV.roi_with_ci(M.played(base, res)),
                         "stability": M.stability(base, base, res)}
    out["cons_vs_single"] = {
        "market_only": EV.compare(M.played(mains["cons_market_only"], res), M.played(mains["single_market_only"], res)),
        "blend": EV.compare(M.played(mains["cons_blend_wf"], res), M.played(mains["single_blend_wf"], res))}
    # EV of the Baseline B picks with the model, the single book and the consensus
    b = M.played(base, res).merge(c[["match_id", "market", "selection", "q_win", "q_cons_win", "cons_source", "disp"]],
                                  on=["match_id", "market", "selection"], how="left")
    bb = b.loc[b["bet"] == 1]
    ev_model = bb["probability"] * bb["odds"] - (1 - bb["push_probability"])
    ev_cons = bb["q_cons_win"] * bb["odds"] - (1 - bb["push_probability"])
    ev_single = bb["q_win"] * bb["odds"] - (1 - bb["push_probability"])
    extra["ev_baseline_picks"] = {"n": int(len(bb)), "with_consensus": int(bb["q_cons_win"].notna().sum()),
                                  "mean_ev_model": float(ev_model.mean()), "mean_ev_single": float(ev_single.mean()),
                                  "mean_ev_consensus": float(ev_cons.mean()),
                                  "share_pos_model": float((ev_model > 0).mean()),
                                  "share_pos_consensus": float((ev_cons.dropna() > 0).mean()),
                                  "roi_ev_cons_pos": EV.roi_with_ci(bb.loc[ev_cons > 0]),
                                  "roi_ev_cons_neg": EV.roi_with_ci(bb.loc[ev_cons <= 0]),
                                  "sources": bb["cons_source"].value_counts().to_dict()}
    # dispersion of the picked market and the Main ROI
    bd = bb.loc[bb["disp"].notna()]
    if len(bd) > 100:
        qb = pd.qcut(bd["disp"], 3, labels=["низкий", "средний", "высокий"])
        extra["roi_by_dispersion"] = {str(k): {"n": int(len(g)), **EV.roi_with_ci(g),
                                               "hit": float((g["status"] == "WON").mean()),
                                               "mean_p": float(g["probability"].mean())}
                                      for k, g in bd.groupby(qb, observed=True)}
    return out, mains, {**extra, "cands": c}


def run() -> dict:
    started = time.time()
    log = C.setup_logging(STUDY, D["logs"] / "run.log")
    m = M.matches()
    res = M.results()
    odds = M.load_odds()
    fd_all = FD.all_seasons()
    fd26 = fd_all.loc[fd_all["season"] == C.CURRENT]
    rep = {}
    av, rep["availability"] = availability(odds, m, fd26)
    av.to_csv(RES / "availability.csv", index=False)
    log.info("availability done")
    rep["fd_rules"] = fd_rules(fd_all)
    rule_1x2 = rep["fd_rules"]["selection"].get("1X2|open", {}).get("chosen", "median")
    if rule_1x2 in ("bet365", "pinnacle", "fd_avg", "fd_max", "exchange"):
        rule_1x2_cons = "median"      # a single book is not a consensus; keep the best multi-book rule
        multi = {k: v for k, v in rep["fd_rules"]["selection"]["1X2|open"]["rules_ll_2021_2025"].items()
                 if k in ("mean", "median", "logit_mean", "inv_margin")}
        rule_1x2_cons = min(multi, key=multi.get) if multi else "median"
    else:
        rule_1x2_cons = rule_1x2
    rep["rule_1x2"] = {"best_overall_2021_2025": rule_1x2, "consensus_used": rule_1x2_cons}
    log.info("fd rules done: %s", rep["rule_1x2"])
    rep["decision_time"], dec_tabs = decision_consensus(odds, m)
    rep["gap_sensitivity"] = {u: gap_sensitivity(dec_tabs[u], res) for u in UNIVERSES}
    log.info("decision-time consensus done")
    rep["model"], rep["variants"], rep["extra"] = {}, {}, {}
    pv, cov, all_mains = {}, [], {}
    for u in UNIVERSES:
        c = consensus_candidates(u, odds, m, fd26, rule_1x2_cons, dec_tabs[u])
        rep["model"][u] = model_and_consensus(c)
        out, mains, extra = main_variants(u, c, odds, m, fd26, res)
        rep["variants"][u] = out
        rep["extra"][u] = {k: v for k, v in extra.items() if k != "cands"}
        for k, v in out.items():
            if isinstance(v, dict) and "vs_b" in v:
                pv[f"{k}|{u}"] = v["vs_b"]["roi_diff"]["p"]
        base = M.baseline(f"baseline_b_{u}")
        all_mains[u] = (base, mains)
        for k, v in {"baseline_b": base, **mains}.items():
            p = M.played(v, res)
            cov.append({"universe": u, "method": k, "season_matches": int(len(m)), "played": int(len(res)),
                        "processed": int(v["match_id"].nunique()), "bets_real_price": int(p["bet"].sum()),
                        "no_price": int(p["odds"].isna().sum()), "push": int((p["status"] == "PUSH").sum()),
                        "main_with_consensus": int(p["q_cons"].notna().sum()) if "q_cons" in p else None,
                        "without_statistics": 0, "not_played_excluded": int(len(m) - len(res))})
        log.info("%s: %s", u, {k: round(v["summary"]["roi"], 4) for k, v in out.items() if "summary" in v})
    for u in UNIVERSES:
        for k, v in rep["variants"][u]["cons_vs_single"].items():
            pv[f"cons_vs_single_{k}|{u}"] = v["roi_diff"]["p"]
    rep["holm"] = {"n_tests": len(pv), "p": pv, "p_holm": C.holm(pv)}
    a = M.baseline("baseline_a")
    rep["baseline_a"] = {"summary": EV.summary(a), "roi_ci": EV.roi_with_ci(a)}
    pd.DataFrame(cov).to_csv(RES / "coverage.csv", index=False)
    for u, (base, mains) in all_mains.items():
        EV.per_match_table(M.played(base, res), res, f"baseline_b_{u}", "prod_20261012").to_csv(
            RES / f"per_match_baseline_b_{u}.csv", index=False)
        for k in ("cons_blend_wf", "cons_market_only", "best_price"):
            EV.per_match_table(M.played(mains[k], res), res, f"{k}_{u}", "prod_20261012+consensus").to_csv(
                RES / f"per_match_{k}_{u}.csv", index=False)
    rep["variants_ru"], rep["rules_ru"] = VARIANTS_RU, RULES_RU
    C.write_json(RES / "consensus.json", rep)
    C.write_json(RES / "run_info.json", C.run_info(STUDY, started, {
        "universes": UNIVERSES, "fd_snapshot_by_universe": SNAP, "rules": RULES, "select_seasons": SELECT_SEASONS,
        "fair_method": "power", "gap_bins_h": GAP_LABELS, "n_boot": C.N_BOOT}))
    M.preregister(RES / "preregistration.json", PREREG)
    from . import figures
    figures.run()
    log.info("done in %.0f s", time.time() - started)
    return rep


if __name__ == "__main__":
    run()
