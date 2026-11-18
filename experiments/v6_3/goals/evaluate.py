"""Evaluation of the Goal-Model study: selection on 2023/24–2025/26, chronological replay of 2026/27, Main ROI.

Variants (each compared with production = XGB goal means + independent Poisson + reconciliation with the 1X2
classifier, rho = 0):

* distribution / reconciliation (production goal means and 1X2): raw, partial, dc (ρ after reconciliation),
  dc_raw_rho (ρ as production's pipeline would fit it), nb, ipf, sup;
* goal means (production matrix construction): rat (dynamic ratings only), blend (geometric XGB + ratings, w on
  2023–2025), blend_ha (ratings with home/away deviations), ratfeat (XGB with the ratings as features), stacked
  (one team-goals model), monthly (production goal models refitted monthly — study 7's control), hy_v62 (v6.2's
  frozen B candidate: XGB + Maher MH2, w = 0.8458);
* combo: the best goal means and the best matrix construction of the selection seasons together.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV
from experiments.v6_3.recent_weighting import lib as L
from scipy.optimize import minimize_scalar

from . import dists as DS

DIRS = C.study_dirs("goals")
RW_CACHE = C.V63 / "recent_weighting" / "cache"
SEL = (2023, 2024, 2025)
HY_W = 0.8457986642239216          # frozen by v6.2 (experiments/v6_2/goal_distributions/results/preregistration.json)
MATRIX_VARIANTS = ["raw", "partial", "dc", "dc_raw_rho", "nb", "ipf", "sup"]
LAMBDA_VARIANTS = ["rat", "blend", "blend_ha", "ratfeat", "stacked", "monthly", "hy_v62"]
NAMES = {"prod": "Продакшн: XGB + Пуассон + согласование", "raw": "Пуассон без согласования",
         "partial": "Частичное согласование (α)", "dc": "Диксон–Коулс, ρ после согласования",
         "dc_raw_rho": "Диксон–Коулс, ρ по сырым средним", "nb": "Отр. биномиальное + согл.",
         "ipf": "IPF: 1X2 + распределение тотала", "sup": "Подбор перевеса λ под 1X2",
         "rat": "Динамические рейтинги (только)", "blend": "XGB + динам. рейтинги (смесь)",
         "blend_ha": "XGB + рейтинги дом/выезд (смесь)", "ratfeat": "XGB с рейтингами как признаками",
         "stacked": "Единая модель голов команды", "monthly": "Ежемесячное переобучение XGB",
         "hy_v62": "v6.2 HY+R: XGB + Махер (заморожено)", "combo": "Комбинация лучших"}


# ------------------------------------------------------------------------------------------------ data
def _merge_rat(df: pd.DataFrame, name: str, pref: str) -> pd.DataFrame:
    r = pd.read_parquet(DIRS["cache"] / f"ratings_{name}.parquet")[["match_id", "r_lh", "r_la"]]
    return df.merge(r.rename(columns={"r_lh": f"{pref}_lh", "r_la": f"{pref}_la"}), on="match_id", how="left")


def season_frame(season: int) -> pd.DataFrame:
    """Played matches of ``season`` with results, the base model's outputs and every available λ variant."""
    f = L.season_rows(season)
    f = f.loc[f["played"].astype(bool) & f["home_goals"].notna()]
    d = f[["match_id", "date", "league", "season", "home_goals", "away_goals", "h_season_games", "a_season_games"]]
    if season == C.CURRENT:
        base = L.production_outputs()[["match_id", "p_home", "p_draw", "p_away", "lam_home", "lam_away"]]
    else:
        base = pd.read_parquet(DIRS["cache"] / f"base_{season}.parquet")
    d = d.merge(base, on="match_id", how="inner")
    d = _merge_rat(d, "plain", "rat")
    d = _merge_rat(d, "ha", "ratha")
    for name in ("ratfeat", "stacked"):
        p = DIRS["cache"] / f"{name}_{season}.parquet"
        if p.exists():
            x = pd.read_parquet(p).rename(columns={"lam_home": f"{name}_lh", "lam_away": f"{name}_la"})
            d = d.merge(x, on="match_id", how="left")
    p = RW_CACHE / f"wf_{season}_prod.parquet"
    if p.exists():
        x = pd.read_parquet(p)[["match_id", "lam_home", "lam_away"]].rename(
            columns={"lam_home": "monthly_lh", "lam_away": "monthly_la"})
        d = d.merge(x, on="match_id", how="left")
    if season == C.CURRENT:
        p = DIRS["cache"] / "maher_2026.parquet"
        if p.exists():
            d = d.merge(pd.read_parquet(p), on="match_id", how="left")
    return d.reset_index(drop=True)


def geo(a, b, w):
    return np.exp((1 - w) * np.log(np.asarray(a, float)) + w * np.log(np.asarray(b, float)))


def lambdas(d: pd.DataFrame, kind: str, par: dict) -> tuple[np.ndarray, np.ndarray] | None:
    lh, la = d["lam_home"].to_numpy(float), d["lam_away"].to_numpy(float)
    if kind in ("prod",) + tuple(MATRIX_VARIANTS):
        return lh, la
    if kind == "rat":
        return d["rat_lh"].to_numpy(float), d["rat_la"].to_numpy(float)
    if kind == "blend":
        return geo(lh, d["rat_lh"], par["w"]), geo(la, d["rat_la"], par["w"])
    if kind == "blend_ha":
        return geo(lh, d["ratha_lh"], par["w_ha"]), geo(la, d["ratha_la"], par["w_ha"])
    if kind in ("ratfeat", "stacked", "monthly"):
        if f"{kind}_lh" not in d or d[f"{kind}_lh"].isna().all():
            return None
        return d[f"{kind}_lh"].to_numpy(float), d[f"{kind}_la"].to_numpy(float)
    if kind == "hy_v62":
        if "lh_m2" not in d:
            return None
        mh = np.where(np.isfinite(d["lh_m2"]), d["lh_m2"], lh)
        ma = np.where(np.isfinite(d["la_m2"]), d["la_m2"], la)
        return geo(lh, mh, 1 - HY_W), geo(la, ma, 1 - HY_W)
    if kind == "combo":
        return lambdas(d, par["combo_lambda"], par)
    raise ValueError(kind)


def matrix_kind(kind: str, par: dict) -> str:
    if kind in MATRIX_VARIANTS:
        return kind
    if kind == "combo":
        return par["combo_matrix"]
    return "prod"


def matrices(d: pd.DataFrame, kind: str, par: dict):
    lam = lambdas(d, kind, par)
    if lam is None:
        return None, None
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
    return DS.build(matrix_kind(kind, par), lam[0], lam[1], p, par), lam


def losses(d: pd.DataFrame, kind: str, par: dict) -> pd.DataFrame | None:
    m, lam = matrices(d, kind, par)
    if m is None:
        return None
    s = L.score_losses(m, d["home_goals"], d["away_goals"])
    s["dev_home"] = L.poisson_dev(d["home_goals"], lam[0])
    s["dev_away"] = L.poisson_dev(d["away_goals"], lam[1])
    s["abs_home"] = np.abs(d["home_goals"].to_numpy() - lam[0])
    s["abs_away"] = np.abs(d["away_goals"].to_numpy() - lam[1])
    s["dev_goals"] = (s["dev_home"] + s["dev_away"]) / 2
    s["mae_goals"] = (s["abs_home"] + s["abs_away"]) / 2
    s["match_id"] = d["match_id"].to_numpy()
    return s


KEYS = ["score_ll", "totals_ll", "totals_brier", "ou25_ll", "btts_ll", "btts_brier", "hcp_ll", "margin_rps",
        "total_rps", "x12_ll", "dev_goals", "mae_goals"]


def summarise(los: pd.DataFrame) -> dict:
    out = {k: float(los[k].mean()) for k in KEYS}
    out["mae_home"], out["mae_away"] = float(los["abs_home"].mean()), float(los["abs_away"].mean())
    out["dev_home"], out["dev_away"] = float(los["dev_home"].mean()), float(los["dev_away"].mean())
    out["ece_ou25"] = L.ece(los["p_over25"], los["y_over25"])
    out["ece_btts"] = L.ece(los["p_btts"], los["y_btts"])
    out["top_score_p"], out["top_score_hit"] = float(los["top_score_p"].mean()), float(los["top_score_hit"].mean())
    out["n"] = int(len(los))
    return out


def paired(los: pd.DataFrame, base: pd.DataFrame, keys=("score_ll", "totals_ll", "btts_ll", "hcp_ll", "dev_goals",
                                                        "mae_goals", "x12_ll")) -> dict:
    return {k: C.paired_mean_diff(los[k].to_numpy(), base[k].to_numpy()) for k in keys}


# ------------------------------------------------------------------------------------------------ selection
def fit_selection(frames: dict, log) -> dict:
    d = pd.concat([frames[s] for s in SEL], ignore_index=True)
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
    hg, ag = d["home_goals"].to_numpy(int), d["away_goals"].to_numpy(int)
    par = DS.fit_params(d["lam_home"].to_numpy(float), d["lam_away"].to_numpy(float), p, hg, ag)
    par["rho_by_season"] = {}
    for s in SEL:
        x = frames[s]
        px = x[["p_home", "p_draw", "p_away"]].to_numpy(float)
        r = minimize_scalar(lambda v: DS.score_nll(DS.build("dc", x["lam_home"], x["lam_away"], px, {"rho": v}),
                                                   x["home_goals"].to_numpy(int), x["away_goals"].to_numpy(int)),
                            bounds=(-0.3, 0.3), method="bounded", options={"xatol": 1e-4})
        from src.markets.distributions import fit_dixon_coles_rho
        par["rho_by_season"][str(s)] = {"after_reconcile": float(r.x), "raw_pipeline": float(fit_dixon_coles_rho(
            x["lam_home"], x["lam_away"], x["home_goals"], x["away_goals"]))}
    for key, col in (("w", "rat"), ("w_ha", "ratha")):
        def nll(w, col=col):
            m = DS.build("prod", geo(d["lam_home"], d[f"{col}_lh"], w), geo(d["lam_away"], d[f"{col}_la"], w), p)
            return DS.score_nll(m, hg, ag)
        par[key] = float(minimize_scalar(nll, bounds=(0, 1), method="bounded", options={"xatol": 1e-3}).x)
    log.info("selection parameters: %s", par)
    return par


def selection_table(frames: dict, par: dict) -> tuple[list, dict]:
    rows, per = [], {}
    kinds = ["prod"] + MATRIX_VARIANTS + LAMBDA_VARIANTS
    for s in SEL:
        base = losses(frames[s], "prod", par)
        for k in kinds:
            los = losses(frames[s], k, par)
            if los is None:
                continue
            per.setdefault(k, {})[s] = los
            sm = summarise(los)
            pdx = paired(los, base, keys=("score_ll", "totals_ll", "btts_ll", "hcp_ll", "dev_goals"))
            rows.append({"season": s, "variant": k, **{f"m_{x}": sm[x] for x in KEYS},
                         **{f"d_{x}": v["diff"] for x, v in pdx.items()},
                         **{f"d_{x}_lo": v["ci95"][0] for x, v in pdx.items()},
                         **{f"d_{x}_hi": v["ci95"][1] for x, v in pdx.items()},
                         **{f"p_{x}": v["p"] for x, v in pdx.items()}})
    pooled = {}
    for k, by in per.items():
        common = [s for s in SEL if s in by]
        a = pd.concat([by[s] for s in common])
        b = pd.concat([per["prod"][s] for s in common])
        pooled[k] = {"seasons": common, **{x: C.paired_mean_diff(a[x].to_numpy(), b[x].to_numpy())
                                         for x in ("score_ll", "totals_ll", "btts_ll", "hcp_ll", "dev_goals")}}
    return rows, pooled


# ------------------------------------------------------------------------------------------------ Main ROI
def roi_for(kind: str, d26: pd.DataFrame, par: dict):
    """Main tables (both universes) of a variant: production 1X2 / corners, the variant's goal means and
    matrix construction through production's engine."""
    full = L.production_outputs().copy()
    if kind not in ("prod",) + tuple(MATRIX_VARIANTS):
        x = season_frame_all(kind, par)
        if x is None:
            return None
        x = x.set_index("match_id").reindex(full["match_id"])
        for c in ("lam_home", "lam_away"):
            v = x[c].to_numpy(float)
            full[c] = np.where(np.isfinite(v), v, full[c].to_numpy(float))
    mk = matrix_kind(kind, par)
    extras, fn = None, None
    if mk in ("dc", "dc_raw_rho"):
        extras = {"home_goals": {"dc_rho": par["rho"] if mk == "dc" else par["rho_raw"]}}
    elif mk != "prod":
        fn = DS.matrix_fn(mk, par)
    return L.main_tables(full, kind, extras, fn)


def season_frame_all(kind: str, par: dict) -> pd.DataFrame | None:
    """Goal means of a λ variant for every 2026/27 match (played or not)."""
    f = L.season_rows(C.CURRENT)[["match_id"]]
    d = f.merge(L.production_outputs()[["match_id", "p_home", "p_draw", "p_away", "lam_home", "lam_away"]],
                on="match_id", how="left")
    d = _merge_rat(d, "plain", "rat")
    d = _merge_rat(d, "ha", "ratha")
    for name in ("ratfeat", "stacked"):
        p = DIRS["cache"] / f"{name}_{C.CURRENT}.parquet"
        if p.exists():
            d = d.merge(pd.read_parquet(p).rename(columns={"lam_home": f"{name}_lh", "lam_away": f"{name}_la"}),
                        on="match_id", how="left")
    p = RW_CACHE / f"wf_{C.CURRENT}_prod.parquet"
    if p.exists():
        d = d.merge(pd.read_parquet(p)[["match_id", "lam_home", "lam_away"]].rename(
            columns={"lam_home": "monthly_lh", "lam_away": "monthly_la"}), on="match_id", how="left")
    p = DIRS["cache"] / "maher_2026.parquet"
    if p.exists():
        d = d.merge(pd.read_parquet(p), on="match_id", how="left")
    lam = lambdas(d, kind, par)
    if lam is None:
        return None
    return pd.DataFrame({"match_id": d["match_id"], "lam_home": lam[0], "lam_away": lam[1]})


# ------------------------------------------------------------------------------------------------ run
def decide(k: str, pooled: dict, ev26: dict, holm26: dict, roi: dict) -> str:
    sel = pooled.get(k, {}).get("score_ll")
    sel_ok = sel is not None and sel["ci95"][1] is not None and sel["ci95"][1] < 0
    d26 = ev26[k]["score_ll"]
    worse = any(ev26[k][x]["ci95"][0] is not None and ev26[k][x]["ci95"][0] > 0
                for x in ("totals_ll", "btts_ll", "hcp_ll"))
    r = roi.get(k)
    roi_bad = r is not None and any(r[u]["vs_b"]["roi_diff"]["ci95"][1] < 0 for u in r)
    roi_ok = r is not None and all(r[u]["vs_b"]["roi_diff"]["diff"] >= 0 for u in r)
    if sel_ok and d26["diff"] < 0 and holm26.get(k, 1) < 0.05 and not worse and roi_ok:
        return "A"
    if sel_ok and d26["diff"] < 0 and not worse and not roi_bad:
        return "B"
    if (not sel_ok) and d26["diff"] < 0 and holm26.get(k, 1) < 0.05 and not worse and not roi_bad:
        return "B"
    return "C"


def run(log, started: float) -> None:
    frames = {s: season_frame(s) for s in SEL + (C.CURRENT,)}
    par = fit_selection(frames, log)
    sel_rows, pooled = selection_table(frames, par)
    pd.DataFrame(sel_rows).to_csv(DIRS["results"] / "selection_by_season.csv", index=False)

    # choose the combination on the selection seasons: best λ variant and best matrix construction by pooled
    # score log loss (only variants available on all three seasons compete for λ; ratfeat / stacked have 2024–2025)
    def best_of(kinds):
        c = {k: pooled[k]["score_ll"]["diff"] for k in kinds if k in pooled}
        return min(c, key=c.get) if c else None
    par["combo_matrix"] = best_of(MATRIX_VARIANTS)
    par["combo_lambda"] = best_of([k for k in LAMBDA_VARIANTS if k not in ("hy_v62",)])
    log.info("combo: %s + %s", par["combo_lambda"], par["combo_matrix"])

    d26 = frames[C.CURRENT]
    base26 = losses(d26, "prod", par)
    kinds = MATRIX_VARIANTS + LAMBDA_VARIANTS + ["combo"]
    ev26, sm26, per_match = {}, {"prod": summarise(base26)}, {"prod": base26}
    for k in kinds:
        los = losses(d26, k, par)
        if los is None:
            log.info("variant %s: not available for 2026/27", k)
            continue
        per_match[k] = los
        sm26[k] = summarise(los)
        ev26[k] = paired(los, base26)
    holm26 = C.holm({k: v["score_ll"]["p"] for k, v in ev26.items()})

    # by month / by teams' matches played (score log loss and totals)
    month = d26["date"].dt.to_period("M").astype(str).to_numpy()
    games = L.season_games(d26).to_numpy()
    gbin = pd.cut(games, [-1, 2, 5, 9, 14, 99], labels=["0-2", "3-5", "6-9", "10-14", "15+"]).astype(str)
    by_month, by_games = [], []
    for k, los in per_match.items():
        dl = los["score_ll"].to_numpy() - base26["score_ll"].to_numpy()
        for mth in sorted(set(month)):
            sel = month == mth
            by_month.append({"variant": k, "month": mth, "n": int(sel.sum()), "d_score_ll": float(dl[sel].mean()),
                             "d_totals_ll": float((los["totals_ll"] - base26["totals_ll"]).to_numpy()[sel].mean())})
        for g in ["0-2", "3-5", "6-9", "10-14", "15+"]:
            sel = gbin == g
            if sel.sum():
                by_games.append({"variant": k, "games": g, "n": int(sel.sum()), "d_score_ll": float(dl[sel].mean())})
    pd.DataFrame(by_month).to_csv(DIRS["results"] / "by_month_2026.csv", index=False)
    pd.DataFrame(by_games).to_csv(DIRS["results"] / "by_games_2026.csv", index=False)

    # calibration of O/U 2.5 and BTTS
    calib = {k: {"ou25": L.calibration_table(per_match[k]["p_over25"], per_match[k]["y_over25"]),
                 "btts": L.calibration_table(per_match[k]["p_btts"], per_match[k]["y_btts"])}
             for k in per_match}

    # Main ROI (both universes)
    roi, roi_rows, tabs_keep = {}, [], {}
    for k in ["prod"] + [k for k in kinds if k in per_match]:
        t0 = time.time()
        tabs = roi_for(k, d26, par)
        if tabs is None:
            continue
        roi[k] = L.main_eval(tabs)
        tabs_keep[k] = tabs
        for u in L.UNIVERSES:
            roi_rows.append(L.roi_row(k, roi[k], u))
        log.info("ROI %s: close %.4f t09 %.4f (%.0fs)", k, roi[k]["close"]["summary"]["roi"],
                 roi[k]["t09"]["summary"]["roi"], time.time() - t0)
    rr = pd.DataFrame(roi_rows)
    rr["p_holm"] = np.nan
    for u in L.UNIVERSES:
        sub = rr.loc[(rr["universe"] == u) & (rr["variant"] != "prod")]
        h = C.holm(dict(zip(sub["variant"], sub["p"])))
        rr.loc[sub.index, "p_holm"] = sub["variant"].map(h)
    rr.to_csv(DIRS["results"] / "main_roi.csv", index=False)

    decisions = {k: decide(k, pooled, ev26, holm26, roi) for k in ev26}

    # best candidates: lowest 2026 score log loss among B / A; plus the best close ROI; per-match tables
    ranked = sorted(ev26, key=lambda k: ev26[k]["score_ll"]["diff"])
    best_prob = ranked[0]
    best_roi = max([k for k in roi if k != "prod"], key=lambda k: roi[k]["close"]["vs_b"]["roi_diff"]["diff"])
    res = L.replay_results()
    extra = {}
    for k in dict.fromkeys([best_prob, best_roi, "combo"]):
        if k not in tabs_keep:
            continue
        for u, t in tabs_keep[k].items():
            EV.per_match_table(t, res, f"goals_{k}_{u}", "prod_20261012+goals_" + k).to_csv(
                DIRS["results"] / f"per_match_{k}_{u}.csv", index=False)
            b = L.baseline_b(u)
            extra.setdefault(k, {})[u] = {
                "by_month": EV.breakdown(t.assign(month=t["date"].dt.to_period("M").astype(str)), "month")
                .to_dict("records"),
                "by_league": EV.breakdown(t, "league").to_dict("records"),
                "by_group": EV.breakdown(t, "group").to_dict("records"),
                "big_wins": L.big_win_dependence(t, b), "odds_minus_2pct": L.odds_sensitivity(t),
                "base_odds_minus_2pct": L.odds_sensitivity(b)}
    for u in L.UNIVERSES:
        EV.per_match_table(L.baseline_b(u), res, f"baseline_b_{u}", "prod_20261012").to_csv(
            DIRS["results"] / f"per_match_baseline_b_{u}.csv", index=False)

    # per-match probabilistic table of the 2026/27 matches (production vs the variants)
    pm = d26[["match_id", "date", "league", "home_goals", "away_goals", "lam_home", "lam_away"]].copy()
    for k, los in per_match.items():
        pm[f"score_ll_{k}"] = los["score_ll"].to_numpy()
        pm[f"p_over25_{k}"] = los["p_over25"].to_numpy()
        pm[f"p_btts_{k}"] = los["p_btts"].to_numpy()
    pm.to_csv(DIRS["results"] / "per_match_probabilities_2026.csv", index=False)

    coverage = coverage_table(d26, tabs_keep)
    coverage["selection_matches"] = {str(s): int(len(frames[s])) for s in SEL}
    C.write_json(DIRS["results"] / "results.json", {
        "params": par, "pooled_selection": pooled, "summary_2026": sm26, "paired_2026": ev26, "holm_2026": holm26,
        "decisions": decisions, "best_prob": best_prob, "best_roi": best_roi, "roi": roi, "candidates": extra,
        "calibration": calib, "coverage": coverage, "n_variants": len(ev26),
        "names": NAMES})
    preregister(decisions, ev26, pooled, roi, par)
    from . import figures as FG
    FG.make_all(log)
    C.write_json(DIRS["results"] / "run_info.json", C.run_info("goals", started, {
        "selection_seasons": list(SEL), "evaluated_season": C.CURRENT, "params": par,
        "production_version": "prod_20261012"}))
    log.info("done: decisions %s", decisions)


def coverage_table(d26: pd.DataFrame, tabs: dict) -> dict:
    m = L.replay_matches()
    res = L.replay_results()
    g = L.season_games(d26)
    out = {"season_matches": int(len(m)), "played": int(len(res)), "not_played": int(len(m) - len(res)),
           "probabilistic_eval_matches": int(len(d26)), "no_current_season_games": int((g == 0).sum()),
           "fewer_than_3_season_games": int((g < 3).sum()),
           "with_closing_1x2_price": int(res.index.isin(set(pd.read_parquet(L.CACHE / "prices_closing.parquet")
                                                            .query("market == '1X2'")["match_id"])).sum()),
           "with_t09_1x2_price": int(res.index.isin(set(pd.read_parquet(L.CACHE / "prices_decision.parquet")
                                                        .query("market == '1X2'")["match_id"])).sum()),
           "selection_matches": {}}
    for k, t in tabs.items():
        out[k] = {}
        for u, x in t.items():
            out[k][u] = {"main_rows": int(len(x)), "bets": int(x["bet"].sum()),
                         "no_price": int(x["odds"].isna().sum()), "push": int((x["status"] == "PUSH").sum()),
                         "unknown": int((x["status"] == "UNKNOWN").sum()),
                         "no_main": int(len(res) - x["match_id"].nunique())}
    return out


SPEC = {
    "stacked": {"model": "one XGBoost count:poisson model of a team's goals (experiments/v6_3/goals/base.py, "
                         "stage_stacked): two rows per match (home team vs away team with is_home = 1, away team vs "
                         "home team with is_home = 0); features = production home_goals ∪ away_goals sets mapped to "
                         "team (t_) / opponent (o_), diff_* and elo_diff negated and elo_exp_home = 1 − x for the "
                         "away row; production away_goals hyper-parameters (449 trees), recency half-life 2 years, "
                         "trained once on every season before the predicted one",
                "matrix": "production: independent Poisson + reconciliation with the production 1X2 classifier",
                "refit": "once before each season (like production)"},
    "ratfeat": {"model": "production goal models + 6 dynamic-rating features (ratings.py, eta 0.02, r 0.25)"},
}


def preregister(decisions: dict, ev26: dict, pooled: dict, roi: dict, par: dict) -> None:
    now = pd.Timestamp.now(tz="Europe/Berlin").isoformat(timespec="seconds")
    entries = []
    for k, dcs in decisions.items():
        if dcs != "B":
            continue
        entries.append({
            "variant": k, "name": NAMES.get(k, k), "frozen_at": now, "code": "experiments/v6_3/goals/evaluate.py",
            "hypothesis": f"{NAMES.get(k, k)} gives a lower exact-score log loss than production (Poisson + "
                          "reconciliation with the 1X2 classifier) without worse totals / BTTS / handicap "
                          "probabilities and without a lower Main ROI.",
            "exact_rule": SPEC.get(k, {k2: v for k2, v in par.items() if k2 in ("rho", "size", "alpha", "w", "w_ha",
                                                                              "combo_lambda", "combo_matrix")}),
            "primary_metric": "exact-score log loss, paired with production on the same forward matches",
            "secondary": ["totals / BTTS / handicap log loss", "Main ROI vs Baseline B (close and t09)"],
            "minimum_forward_sample": "1500 played matches (about 7 weeks of the 21 leagues)",
            "evidence_2026_27": {"d_score_ll": ev26[k]["score_ll"], "selection_pooled": pooled.get(k, {}).get("score_ll"),
                                 "d_roi_close": roi.get(k, {}).get("close", {}).get("vs_b", {}).get("roi_diff")}})
    C.write_json(DIRS["results"] / "preregistration.json", {"study": "v6.3 Goal Model Improvements",
                                                             "entries": entries})
