"""Market-specific models vs the derived markets of production, on the same matches, walk-forward.

    python -m experiments.v6_2.market_models.direct      # fits the direct models (cache/direct_<season>.*)
    python -m experiments.v6_2.market_models.run         # evaluation -> results/*.json

Per season s (2023/24 ... 2026/27):

    CUR     production: every goal market from the independent-Poisson score matrix of the goal regressors,
            reconciled with the 1X2 classifier; corners from NB(mu, r)
    BIN     a dedicated binary classifier per selection (BTTS, over 1.5 / 2.5 / 3.5, 1X and X2, corners lines)
    TOT     a 7-class total-goals classifier (all total lines from one coherent distribution)
    GD      a 7-class goal-difference classifier (1X2, double chance and every Asian line incl. refunds)
    ES      a 17-class exact-score classifier
    S_BIN   stacking: logistic regression on (logit CUR, logit BIN), fitted on season s-1 (fold s-1 predictions)
    S_TOT, S_GD, S_ES   log-linear pool p ∝ CUR^a · DIRECT^b over the classes, (a, b) fitted on season s-1

Results: results/market_metrics.json, comparisons.json, calibration.json, coherence.json, cost.json,
leagues.json, main_risk.json, market_reference.json, decisions.json, stacking.json, run_info.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from . import direct as DR
from . import probs as PR
from . import selection as SEL

GROUP_VARIANTS = {
    "1X2": ["GD", "S_GD"],
    "DOUBLE_CHANCE": ["BIN", "GD", "S_BIN", "S_GD"],
    "TOTAL": ["BIN", "TOT", "S_BIN", "S_TOT"],
    "BTTS": ["BIN", "S_BIN"],
    "HANDICAP": ["GD", "S_GD"],
    "EXACT_SCORE": ["ES", "S_ES"],
    "CORNERS": ["BIN", "S_BIN"],
}
BIN_KEYS = {"BTTS": "btts", "TOTAL_1.5": "o15", "TOTAL_2.5": "o25", "TOTAL_3.5": "o35", "DC_1X": "dc_1x",
            "DC_X2": "dc_x2", "CORNERS_8.5": "c85", "CORNERS_9.5": "c95", "CORNERS_10.5": "c105"}
NAMES = {"CUR": "текущий (из матрицы)", "BIN": "бинарный классификатор", "TOT": "мультикласс тотала",
         "GD": "мультикласс разницы", "ES": "мультикласс счёта", "S_BIN": "стекинг с бинарным",
         "S_TOT": "стекинг с тоталом", "S_GD": "стекинг с разницей", "S_ES": "стекинг со счётом"}
CAL_KEYS = ["TOTAL_1.5", "TOTAL_2.5", "TOTAL_3.5", "BTTS", "DC_1X", "DC_X2", "AH_-0.5", "AH_+0.5", "AH_-1.5",
            "CORNERS_9.5", "HOME", "DRAW", "AWAY"]


# ------------------------------------------------------------------------------------------------ books
def season_inputs(s: int):
    b, info = D.base(s)
    d, _ = DR.fold(s)
    d = d.set_index("match_id").reindex(b["match_id"]).reset_index()
    M = PR.production_matrices(b)
    cur = {**PR.from_matrix(M), **PR.corners_from_nb(b["cmu"].to_numpy(), info["nb_size"])}
    out = PR.outcomes(b["home_goals"], b["away_goals"], b["total_corners"])
    return b, d, M, cur, out


def direct_books(d: pd.DataFrame) -> dict:
    bin_ = {k: d[c].to_numpy(float) for k, c in BIN_KEYS.items()}
    bin_["DC_12"] = np.clip(2 - bin_["DC_1X"] - bin_["DC_X2"], 1e-4, 1 - 1e-4)   # P(12) = 1 - P(D) implied
    tot = d[[f"tot_{j}" for j in range(7)]].to_numpy(float)
    gd = d[[f"gd_{j}" for j in range(7)]].to_numpy(float)
    es = d[[f"es_{j}" for j in range(17)]].to_numpy(float)
    return {"BIN": bin_, "TOT": PR.from_tot(tot), "GD": PR.from_gd(gd), "ES": {"ES17": es}}


def fit_stacking(cur: dict, dr: dict, out: dict) -> dict:
    w = {"bin": {}, "pool": {}}
    for k in list(BIN_KEYS) + ["DC_12"]:
        ok = np.isfinite(dr["BIN"][k]) & np.isfinite(cur[k]) & np.isfinite(out[k])
        w["bin"][k] = PR.fit_stack_binary(cur[k][ok], dr["BIN"][k][ok], out[k][ok])
    for name, key in (("TOT", "TOT7"), ("GD", "GD7"), ("ES", "ES17")):
        w["pool"][name] = PR.fit_log_pool(cur[key], dr[name][key], out[key])
    return w


def stacked_books(cur: dict, dr: dict, w: dict) -> dict:
    sb = {k: PR.apply_stack_binary(cur[k], dr["BIN"][k], w["bin"][k]) for k in w["bin"]}
    tot = PR.log_pool(cur["TOT7"], dr["TOT"]["TOT7"], *w["pool"]["TOT"])
    gd = PR.log_pool(cur["GD7"], dr["GD"]["GD7"], *w["pool"]["GD"])
    es = PR.log_pool(cur["ES17"], dr["ES"]["ES17"], *w["pool"]["ES"])
    return {"S_BIN": sb, "S_TOT": PR.from_tot(tot), "S_GD": PR.from_gd(gd), "S_ES": {"ES17": es}}


def variant_book(cur: dict, books: dict, group: str, variant: str) -> dict:
    """The production book with the keys of ``group`` taken from ``variant``."""
    if variant == "CUR":
        return cur
    src = books[variant]
    out = dict(cur)
    for k in PR.GROUP_KEYS[group]:
        if k in src:
            out[k] = src[k]
    if group == "TOTAL" and "TOT7" in src:
        out["TOT7"] = src["TOT7"]
    if group in ("HANDICAP", "1X2") and "GD7" in src:
        out["GD7"] = src["GD7"]
    return out


# ------------------------------------------------------------------------------------------------ coherence
def coherence(cur: dict, books: dict) -> dict:
    def share(x):
        x = np.asarray(x)
        return float(np.nanmean(x)) if x.size else np.nan

    tb = books["BIN"]
    res = {}
    for name, bk in (("BIN", tb), ("S_BIN", books["S_BIN"])):
        o15, o25, o35 = bk["TOTAL_1.5"], bk["TOTAL_2.5"], bk["TOTAL_3.5"]
        c85, c95, c105 = bk["CORNERS_8.5"], bk["CORNERS_9.5"], bk["CORNERS_10.5"]
        ok_c = np.isfinite(c85)
        pdraw = bk["DC_1X"] + bk["DC_X2"] - 1
        res[name] = {
            "totals_not_monotone": share((o15 < o25) | (o25 < o35)),
            "btts_above_over15": share(bk["BTTS"] > o15),
            "btts_minus_over15_max": float(np.max(bk["BTTS"] - o15)),
            "corners_not_monotone": share(((c85 < c95) | (c95 < c105))[ok_c]),
            "dc_implied_draw_outside_0_1": share((pdraw < 0) | (pdraw > 1)),
            "dc_implied_draw_vs_classifier_mae": float(np.mean(np.abs(pdraw - cur["1X2"][:, 1]))),
            # a product that shows BIN totals next to the matrix BTTS / 1X2 (mixed sources)
            "btts_cur_above_over15_bin": share(cur["BTTS"] > o15),
        }
    for name in ("GD", "S_GD"):
        g = books[name]
        res[name] = {"home_win_vs_classifier_mae": float(np.mean(np.abs(g["1X2"][:, 0] - cur["1X2"][:, 0]))),
                     "home_win_vs_classifier_max": float(np.max(np.abs(g["1X2"][:, 0] - cur["1X2"][:, 0]))),
                     "draw_vs_classifier_mae": float(np.mean(np.abs(g["1X2"][:, 1] - cur["1X2"][:, 1]))),
                     # AH -0.5 home (from GD) vs 1X2 home win (classifier): must be equal in a coherent product
                     "ah_m05_vs_1x2_home_mae": float(np.mean(np.abs(g["AH_-0.5"] - cur["1X2"][:, 0])))}
    for name in ("TOT", "S_TOT"):
        g = books[name]
        res[name] = {"over25_vs_matrix_mae": float(np.mean(np.abs(g["TOTAL_2.5"] - cur["TOTAL_2.5"]))),
                     "btts_matrix_above_over15": share(cur["BTTS"] > g["TOTAL_1.5"])}
    res["CUR"] = {"totals_not_monotone": 0.0, "btts_above_over15": share(cur["BTTS"] > cur["TOTAL_1.5"] + 1e-12)}
    return res


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    C.setup_logging()
    T = C.Timer()
    inputs = {s: season_inputs(s) for s in (2022,) + tuple(C.SEASONS)}
    stacking, books_all = {}, {}
    for s in C.SEASONS:
        _, d0, _, cur0, out0 = inputs[s - 1]
        w = fit_stacking(cur0, direct_books(d0), out0)
        stacking[str(s)] = w
        b, d, M, cur, out = inputs[s]
        dr = direct_books(d)
        books_all[s] = {"CUR": cur, **dr, **stacked_books(cur, dr, w)}
    C.save_json("stacking.json", stacking)

    # --- per-match losses of every (group, variant), metrics per season
    losses, metrics, cal = {}, {}, {}
    periods = {str(s): (s,) for s in C.SEASONS}
    periods["select"] = C.SELECT
    for g, vs in GROUP_VARIANTS.items():
        for v in ["CUR"] + vs:
            for s in C.SEASONS:
                b, d, M, cur, out = inputs[s]
                bk = variant_book(cur, books_all[s], g, v)
                L = {"group": PR.group_loss(g, bk, out)}
                for k in PR.GROUP_KEYS[g]:
                    L[k] = PR.key_loss(k, bk, out)
                    L[k + "|brier"] = PR.key_brier(k, bk, out)
                if g == "TOTAL":
                    L["TOT7_RPS"] = C.rps(out["TOT7"], bk["TOT7"])
                if g in ("HANDICAP", "1X2"):
                    L["GD7_RPS"] = C.rps(out["GD7"], bk["GD7"]) if "GD7" in bk else np.nan
                losses[(g, v, s)] = pd.DataFrame(L)
    for (g, v, s), L in losses.items():
        metrics.setdefault(g, {}).setdefault(v, {})[str(s)] = L.mean().to_dict()
    for g, vs in GROUP_VARIANTS.items():
        for v in ["CUR"] + vs:
            metrics[g][v]["select"] = pd.concat([losses[(g, v, s)] for s in C.SELECT]).mean().to_dict()

    # --- calibration (ECE, curves) of single selections
    def cal_pairs(bk, out, k):
        if k in ("HOME", "DRAW", "AWAY"):
            j = {"HOME": 0, "DRAW": 1, "AWAY": 2}[k]
            return (out["1X2"] == j).astype(float), bk["1X2"][:, j]
        return out[k], bk[k]

    key_group = {"TOTAL_1.5": "TOTAL", "TOTAL_2.5": "TOTAL", "TOTAL_3.5": "TOTAL", "BTTS": "BTTS",
                 "DC_1X": "DOUBLE_CHANCE", "DC_X2": "DOUBLE_CHANCE", "AH_-0.5": "HANDICAP", "AH_+0.5": "HANDICAP",
                 "AH_-1.5": "HANDICAP", "CORNERS_9.5": "CORNERS", "HOME": "1X2", "DRAW": "1X2", "AWAY": "1X2"}
    for k in CAL_KEYS:
        g = key_group[k]
        for v in ["CUR"] + GROUP_VARIANTS[g]:
            for p, ss in periods.items():
                o = np.concatenate([cal_pairs(variant_book(inputs[s][3], books_all[s], g, v), inputs[s][4], k)[0]
                                    for s in ss])
                q = np.concatenate([cal_pairs(variant_book(inputs[s][3], books_all[s], g, v), inputs[s][4], k)[1]
                                    for s in ss])
                ok = np.isfinite(o) & np.isfinite(q)
                cal.setdefault(k, {}).setdefault(v, {})[p] = {**C.summary_binary(o[ok], q[ok]),
                                                             "curve": C.calib_curve(o[ok], q[ok])}
    C.save_json("market_metrics.json", {"metrics": metrics, "names": NAMES, "group_variants": GROUP_VARIANTS,
                                        "group_keys": PR.GROUP_KEYS,
                                        "n": {str(s): int(len(inputs[s][0])) for s in inputs}})
    C.save_json("calibration.json", cal)

    # --- paired comparisons vs CUR; Holm over all (group, variant) on the pooled selection seasons
    comp = {}
    for g, vs in GROUP_VARIANTS.items():
        for v in vs:
            c = {}
            for p, ss in periods.items():
                a = pd.concat([losses[(g, v, s)] for s in ss])
                r = pd.concat([losses[(g, "CUR", s)] for s in ss])
                c[p] = {"group": C.paired(a["group"].to_numpy(float), r["group"].to_numpy(float))}
                for k in PR.GROUP_KEYS[g]:
                    c[p][k] = C.paired(a[k].to_numpy(float), r[k].to_numpy(float), n_boot=1000)
            comp[f"{g}|{v}"] = c
    holm = C.holm({k: c["select"]["group"]["p_t"] for k, c in comp.items()})   # normal-approx. paired p
    for k in comp:
        comp[k]["holm_p_select"] = holm[k]
    C.save_json("comparisons.json", {"comparisons": comp, "n_comparisons": len(comp)})

    # --- coherence
    coh = {}
    for s in C.SEASONS:
        coh[str(s)] = coherence(inputs[s][3], books_all[s])
    C.save_json("coherence.json", coh)

    # --- league breakdown of the best variant per group (selection seasons)
    best = {g: min(vs, key=lambda v: metrics[g][v]["select"]["group"]) for g, vs in GROUP_VARIANTS.items()}
    leagues = {}
    lg = pd.concat([inputs[s][0]["league"] for s in C.SELECT], ignore_index=True).to_numpy()
    for g, v in best.items():
        a = pd.concat([losses[(g, v, s)] for s in C.SELECT])["group"].to_numpy(float)
        r = pd.concat([losses[(g, "CUR", s)] for s in C.SELECT])["group"].to_numpy(float)
        leagues[g] = {"variant": v, "leagues": {}}
        for name in sorted(set(lg)):
            m = (lg == name) & np.isfinite(a) & np.isfinite(r)
            if m.sum() >= 50:
                leagues[g]["leagues"][name] = C.paired(a[m], r[m], n_boot=1000)
    C.save_json("leagues.json", {"best_by_rule": best, "by_group": leagues})

    # --- market reference: Bet365 opening (proportional de-vig) vs CUR and the best variant, same matches
    C.save_json("market_reference.json", market_reference(inputs, books_all, best))

    # --- Main / Risk with substituted groups
    C.save_json("main_risk.json", main_risk(inputs, books_all, best, metrics))

    # --- cost
    C.save_json("cost.json", cost())
    C.save_json("run_info.json", {"git_sha": C.git_sha(), "environment": C.environment(),
                                  "data_version": C.data_version(), "seconds": T(),
                                  "seasons": {"select": C.SELECT, "info": C.INFO, "current": C.CURRENT},
                                  "data_check": D.run_check(),
                                  "matches": {str(s): int(len(inputs[s][0])) for s in inputs}})
    decide()
    C.log.info("done in %.0fs", T())


def market_reference(inputs, books_all, best) -> dict:
    res = {}
    for s in C.SEASONS:
        b, d, M, cur, out = inputs[s]
        pr = D.prices(s)
        pr = pr.drop_duplicates(["match_id", "market", "selection"]).set_index(["match_id", "market", "selection"])

        def price(mk, sel):
            key = pd.MultiIndex.from_arrays([b["match_id"], [mk] * len(b), [sel] * len(b)])
            pos = pr.index.get_indexer(key)
            return np.where(pos >= 0, pr["odds"].to_numpy(float)[np.maximum(pos, 0)], np.nan)

        res[str(s)] = {}
        x = np.column_stack([1 / price("1X2", c) for c in "HDA"])
        ok = np.isfinite(x).all(1)
        q = x / x.sum(1, keepdims=True)
        for g, key in (("1X2", "1X2"),):
            entry = {"n": int(ok.sum()), "market": float(C.multi_ll(out[key][ok], q[ok]).mean())}
            for v in ["CUR", best[g]]:
                bk = variant_book(cur, books_all[s], g, v)
                entry[v] = float(C.multi_ll(out[key][ok], bk[key][ok]).mean())
            res[str(s)]["1X2"] = entry
        ov, un = 1 / price("TOTAL_2.5", "OVER"), 1 / price("TOTAL_2.5", "UNDER")
        ok = np.isfinite(ov) & np.isfinite(un)
        q = ov / (ov + un)
        entry = {"n": int(ok.sum()), "market": float(C.bin_ll(out["TOTAL_2.5"][ok], q[ok]).mean())}
        for v in ["CUR", best["TOTAL"]] + [x for x in GROUP_VARIANTS["TOTAL"] if x != best["TOTAL"]]:
            bk = variant_book(cur, books_all[s], "TOTAL", v)
            entry[v] = float(C.bin_ll(out["TOTAL_2.5"][ok], bk["TOTAL_2.5"][ok]).mean())
        res[str(s)]["TOTAL_2.5"] = entry
        if s == C.CURRENT:
            yes, no = 1 / price("BTTS", "YES"), 1 / price("BTTS", "NO")
            ok = np.isfinite(yes) & np.isfinite(no)
            q = yes / (yes + no)
            entry = {"n": int(ok.sum()), "market": float(C.bin_ll(out["BTTS"][ok], q[ok]).mean())}
            for v in ["CUR"] + GROUP_VARIANTS["BTTS"]:
                bk = variant_book(cur, books_all[s], "BTTS", v)
                entry[v] = float(C.bin_ll(out["BTTS"][ok], bk["BTTS"][ok]).mean())
            res[str(s)]["BTTS"] = entry
            ov, un = 1 / price("CORNERS_9.5", "OVER"), 1 / price("CORNERS_9.5", "UNDER")
            ok = np.isfinite(ov) & np.isfinite(un) & np.isfinite(out["CORNERS_9.5"]) & np.isfinite(cur["CORNERS_9.5"])
            q = ov / (ov + un)
            entry = {"n": int(ok.sum()), "market": float(C.bin_ll(out["CORNERS_9.5"][ok], q[ok]).mean())}
            for v in ["CUR"] + GROUP_VARIANTS["CORNERS"]:
                bk = variant_book(cur, books_all[s], "CORNERS", v)
                entry[v] = float(C.bin_ll(out["CORNERS_9.5"][ok], bk["CORNERS_9.5"][ok]).mean())
            res[str(s)]["CORNERS_9.5"] = entry
    return res


def _es_top(M, es17=None):
    top = SEL.top_scores(M)
    if es17 is None:
        return top
    out = []
    for labels, p in top:
        h = np.array([int(x.split("-")[0]) for x in labels])
        a = np.array([int(x.split("-")[1]) for x in labels])
        small = (h <= 3) & (a <= 3)
        q = p.copy()
        q[small] = es17[np.flatnonzero(small), (h * 4 + a)[small]]
        out.append((labels, q))
    return out


def main_risk(inputs, books_all, best, metrics) -> dict:
    """Production Main / Risk with one group (or every improved group) taken from its best variant."""
    subs = {f"SUB_{g}": {g: v} for g, v in best.items()}
    improved = {g: v for g, v in best.items() if metrics[g][v]["select"]["group"] < metrics[g]["CUR"]["select"]["group"]}
    subs["SUB_ALL_IMPROVED"] = improved
    out = {"substitutions": subs, "variants": {}, "self_check": {}}
    picks = {}
    for s in C.SEASONS:
        b, d, M, cur, o = inputs[s]
        prices = D.prices(s)
        for name, sub in [("CUR", {})] + list(subs.items()):
            bk = dict(cur)
            es = None
            for g, v in sub.items():
                bk = variant_book(bk, books_all[s], g, v)
                if g == "EXACT_SCORE":
                    es = books_all[s][v]["ES17"]
            long = SEL.attach_prices(SEL.candidates(b, bk, _es_top(M, es)), prices)
            if name == "CUR":
                out["self_check"][str(s)] = SEL.self_check(long, 200)
            mn, rk = SEL.main_risk(long)
            picks[(name, s)] = (mn, rk)
            res = {"main": SEL.summary(mn), "risk": SEL.summary(rk)}
            if name != "CUR":
                res["main_vs_cur"] = SEL.paired_roi(picks[("CUR", s)][0], mn)
                res["risk_vs_cur"] = SEL.paired_roi(picks[("CUR", s)][1], rk)
                res["main_hit_vs_cur"] = SEL.paired_hit(picks[("CUR", s)][0], mn)
            out["variants"].setdefault(name, {})[str(s)] = res
    for name in ["CUR"] + list(subs):
        cat = lambda k, nn: pd.concat([picks[(nn, s)][k] for s in C.SELECT], ignore_index=True)  # noqa: E731
        res = {"main": SEL.summary(cat(0, name)), "risk": SEL.summary(cat(1, name))}
        if name != "CUR":
            res["main_vs_cur"] = SEL.paired_roi(cat(0, "CUR"), cat(0, name))
            res["risk_vs_cur"] = SEL.paired_roi(cat(1, "CUR"), cat(1, name))
            res["main_hit_vs_cur"] = SEL.paired_hit(cat(0, "CUR"), cat(0, name))
        out["variants"][name]["select"] = res
    return out


def cost() -> dict:
    out = {"direct": {}, "production": {}}
    for s in C.FOLDS:
        _, info = DR.fold(s)
        out["direct"][str(s)] = info["models"]
        _, binfo = D.base(s)
        out["production"][str(s)] = binfo["seconds"]
    return out


RULE = ("Primary metric per market group: mean log loss over the group's lines (per match). A: Holm-adjusted "
        "p < 0.05 over all 18 (group, variant) comparisons on the pooled 2023/24+2024/25, better in both selection "
        "seasons, not worse by point in 2025/26 and 2026/27, Main ROI of the substitution not significantly worse "
        "(pooled selection 95 % CI not entirely below 0). B: better in both selection seasons and pooled 95 % CI "
        "below 0 without the Holm correction, or A-level evidence failing 2025/26 / 2026/27. C: everything else. "
        "A variant that only adds models without a significant gain is C (no complexity without evidence).")


def decide() -> dict:
    comp = C.load_json("comparisons.json")["comparisons"]
    mr = C.load_json("main_risk.json")
    best = C.load_json("leagues.json")["best_by_rule"]
    out = {"rule": RULE, "decisions": {}}
    for key, c in comp.items():
        g, v = key.split("|")
        sc = c["select"]["group"]
        both = all(c[str(s)]["group"]["diff"] < 0 for s in C.SELECT)
        later = all(c[str(s)]["group"]["diff"] <= 0 for s in (C.INFO, C.CURRENT))
        roi_bad = None
        if best.get(g) == v:
            mv = mr["variants"][f"SUB_{g}"]["select"].get("main_vs_cur")
            roi_bad = bool(mv and mv["ci95"][1] < 0)
        sig = c["holm_p_select"] < 0.05 and sc["diff"] < 0
        if sig and both and later and not roi_bad:
            d = "A"
        elif (both and sc["ci95"][1] < 0) or (sig and both):
            d = "B"
        else:
            d = "C"
        out["decisions"][key] = {"decision": d, "diff_select": sc["diff"], "ci95": sc["ci95"],
                                 "holm_p": c["holm_p_select"], "better_both_selection": both,
                                 "not_worse_later": later, "main_roi_significantly_worse": roi_bad}
    C.save_json("decisions.json", out)
    return out


if __name__ == "__main__":
    main()
