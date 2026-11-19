"""Squad Strength study (Prompt 9, study 12): evaluation of the fitted variants (light; the fits are in ``fits.py``).

    .venv/bin/python -I experiments/v6_3/squad/parse_tm.py experiments/v6_3/squad/cache/tm_raw experiments/v6_3/squad/cache/tm
    .venv/bin/python -m experiments.v6_3.squad.build_features
    experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e1.log .venv/bin/python -m experiments.v6_3.squad.fits e1
    experiments/v6_3/heavy.sh 5400 experiments/v6_3/squad/logs/fits_e2.log .venv/bin/python -m experiments.v6_3.squad.fits e2
    .venv/bin/python -m experiments.v6_3.squad.run
    .venv/bin/python -m experiments.v6_3.squad.build_pdf

Writes ``results/*.json|csv`` and ``figures/*.png``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as D
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV
from experiments.v6_3.squad import models as M
from experiments.v6_3.squad import tmdata as T
from experiments.v6_3.squad.sources import SOURCES, API_FOOTBALL_PILOT

DIRS = C.study_dirs("squad")
RES, FIG, CACHE = DIRS["results"], DIRS["figures"], DIRS["cache"]
RC = C.SHARED_CACHE
UNIVERSES = {"close": "closing", "t09": "decision"}
E1_VARIANTS = {"dyn": "DYN — продакшн + динамические признаки состава",
               "pre": "PRE — продакшн + предсезонные признаки состава",
               "sa": "SA — squad-aware компактная модель (динамические)"}
E2_VARIANTS = {"b0r": "B0r — переобучение продакшн (контроль)",
               "pre": "PRE — продакшн + предсезонные признаки состава",
               "sa": "SA-pre — squad-aware компактная модель (предсезонные)"}
KEY_METRICS = ["ll_1x2", "brier_1x2", "dev_home", "dev_away", "ll_ou25", "ll_btts", "dev_corners"]


def table() -> pd.DataFrame:
    f = D.load_features(["match_id", "league", "season", "date", "home_team", "away_team", "played", "result",
                         "home_goals", "away_goals", "total_corners", "elo_diff", "diff_form10_ppg",
                         "diff_form10_gd", "diff_season_ppg", "league_home_win_rate", "league_draw_rate"])
    sq = pd.read_parquet(CACHE / "squad_features.parquet")
    return f.merge(sq, on="match_id", how="left")


def has_dyn(f: pd.DataFrame) -> pd.Series:
    return f["h_sq_xi_value"].notna() & f["a_sq_xi_value"].notna()


def has_pre(f: pd.DataFrame) -> pd.Series:
    return f["h_pre_xi_value"].notna() & f["a_pre_xi_value"].notna()


# ------------------------------------------------------------------------------------------------ stage A
def coverage(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lg, season), g in f.groupby(["league", "season"]):
        rows.append({"league": lg, "season": int(season), "matches": len(g), "played": int(g["played"].sum()),
                     "source_league": T.LEAGUE_MAP.get(lg, ""), "dyn_both": int(has_dyn(g).sum()),
                     "pre_both": int(has_pre(g).sum()),
                     "xi_value_cov": float(pd.concat([g["h_sq_value_cov"], g["a_sq_value_cov"]]).mean())
                     if has_dyn(g).any() else None})
    return pd.DataFrame(rows)


def pilot_plan() -> dict:
    """Requests per day of the forward API-Football pilot, from the 2025/26 calendar of the 21 leagues."""
    f = D.load_features(["league", "season", "date"])
    s = f.loc[f["season"] == 2025]
    per_day = s.groupby("date").size()
    top = s.loc[s["league"].isin(T.LEAGUE_MAP)].groupby("date").size()
    req = np.ceil(per_day / 20)
    return {"season": "2025/26", "matches": int(len(s)), "match_days": int(len(per_day)),
            "matches_per_day_median": float(per_day.median()), "matches_per_day_max": int(per_day.max()),
            "requests_ids_per_day_mean": float(req.mean()), "requests_ids_per_day_max": int(req.max()),
            "requests_ids_per_season": int(req.sum()),
            "fallback_two_per_match_top11_max_per_day": int(2 * top.max()),
            "fallback_two_per_match_top11_per_season": int(2 * top.sum()),
            "daily_limit": 100}


# ------------------------------------------------------------------------------------------------ E1
def outputs(tag: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"outputs_{tag}.parquet")


def e1(f: pd.DataFrame, log) -> dict:
    out, pvals = {}, {}
    for season in (2024, 2025):
        s = f.loc[(f["season"] == season) & f["played"].astype(bool)]
        cov = s.loc[s["league"].isin(T.LEAGUE_MAP) & has_dyn(s) & has_pre(s)]
        unc = s.loc[~s["league"].isin(T.LEAGUE_MAP)]
        base = M.per_match_losses(outputs(f"e1_{season}_b0"), cov)
        base_u = M.per_match_losses(outputs(f"e1_{season}_b0"), unc)
        res = {"matches_covered": int(len(cov)), "matches_uncovered": int(len(unc)), "variants": {}}
        for v in ("dyn", "pre", "sa"):
            tag = f"e1_{season}_{v}"
            if not (CACHE / f"outputs_{tag}.parquet").exists():
                continue
            o = outputs(tag)
            cmp_ = M.compare_losses(M.per_match_losses(o, cov), base)
            cmp_u = M.compare_losses(M.per_match_losses(o, unc), base_u)
            o2 = o.set_index("match_id").loc[cov["match_id"]]
            res["variants"][v] = {"covered": cmp_, "uncovered": cmp_u,
                                  "r2_corners": M.r2(cov["total_corners"], o2["corner_mu"]),
                                  "r2_goals": M.r2(cov["home_goals"] + cov["away_goals"],
                                                   o2["lam_home"] + o2["lam_away"])}
            for m in KEY_METRICS:
                if m in cmp_:
                    pvals[f"{season}|{v}|{m}"] = cmp_[m]["p"]
        b2 = outputs(f"e1_{season}_b0").set_index("match_id").loc[cov["match_id"]]
        res["b0_r2_corners"] = M.r2(cov["total_corners"], b2["corner_mu"])
        res["b0_r2_goals"] = M.r2(cov["home_goals"] + cov["away_goals"], b2["lam_home"] + b2["lam_away"])
        out[str(season)] = res
        log.info("E1 %s: %d covered matches", season, len(cov))
    adj = C.holm(pvals)
    for k, v in adj.items():
        season, var, m = k.split("|")
        out[season]["variants"][var]["covered"][m]["p_holm"] = v
    out["holm_family"] = {"tests": len(pvals), "what": "variant × season × key metric (7 metrics)"}
    return out


# ------------------------------------------------------------------------------------------------ beyond Elo and form
BASE_X = ["elo_diff", "diff_form10_ppg", "diff_form10_gd", "diff_season_ppg", "league_home_win_rate",
          "league_draw_rate"]
SQ_X = ["diff_sq_xi_value", "diff_sq_squad_value", "diff_sq_bench_value", "diff_sq_xi_ga90", "diff_sq_impact",
        "diff_sq_form", "diff_sq_stability", "diff_sq_last_xi_share", "diff_sq_def_value", "diff_sq_att_value"]
PRE_X = ["diff_pre_xi_value", "diff_pre_squad_value", "diff_pre_continuity", "diff_pre_xi_ga90", "diff_pre_impact"]


def _lr_ll(train, test, cols) -> np.ndarray:
    tr, te = train[cols].fillna(train[cols].median()), test[cols].fillna(train[cols].median())
    sc = StandardScaler().fit(tr)
    y = train["result"].map({"H": 0, "D": 1, "A": 2})
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr), y)
    p = lr.predict_proba(sc.transform(te))
    yt = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return -np.log(np.clip(p[np.arange(len(yt)), yt], 1e-9, 1)), dict(zip(cols, lr.coef_[0] - lr.coef_[2]))


def beyond_elo(f: pd.DataFrame) -> dict:
    """Out-of-sample: multinomial logistic regression on Elo + form (+ league rates) vs the same + squad features;
    and stacking on the refitted production model's own probabilities (does squad information add anything the
    production model has not already used?)."""
    p = f.loc[f["played"].astype(bool) & f["league"].isin(T.LEAGUE_MAP) & f["result"].notna()]
    out = {}
    for season, xs, need in ((2024, SQ_X, has_dyn), (2025, SQ_X, has_dyn), (2026, PRE_X, has_pre)):
        tr = p.loc[(p["season"] < season) & need(p) & (p["season"] >= 2021)]
        te = p.loc[(p["season"] == season) & need(p)]
        if len(te) == 0:
            continue
        l0, _ = _lr_ll(tr, te, BASE_X)
        l1, coef = _lr_ll(tr, te, BASE_X + xs)
        lq, _ = _lr_ll(tr, te, ["elo_diff"] + xs)
        lsq, _ = _lr_ll(tr, te, xs)
        out[str(season)] = {"train": int(len(tr)), "test": int(len(te)), "ll_elo_form": float(l0.mean()),
                            "ll_elo_form_squad": float(l1.mean()), "ll_elo_squad": float(lq.mean()),
                            "ll_squad_only": float(lsq.mean()),
                            "diff_squad_vs_elo_form": C.paired_mean_diff(l1, l0),
                            "coef_home_minus_away_std": coef}
    # stacking on the production refit's out-of-sample probabilities: fit on 2024/25, test on 2025/26
    st = {}
    for fam, xs in (("sq", SQ_X), ("pre", PRE_X)):
        frames = []
        for season in (2024, 2025):
            o = outputs(f"e1_{season}_b0")
            s = p.loc[(p["season"] == season) & has_dyn(p) & has_pre(p)].merge(o, on="match_id")
            s["lr_hd"] = np.log(s["p_home"] / s["p_draw"])
            s["lr_ad"] = np.log(s["p_away"] / s["p_draw"])
            frames.append(s)
        tr, te = frames
        l0, _ = _lr_ll(tr, te, ["lr_hd", "lr_ad"])
        l1, coef = _lr_ll(tr, te, ["lr_hd", "lr_ad"] + xs)
        yt = te["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        raw = -np.log(np.clip(te[["p_home", "p_draw", "p_away"]].to_numpy()[np.arange(len(yt)), yt], 1e-9, 1))
        st[fam] = {"train_2024": int(len(tr)), "test_2025": int(len(te)), "ll_production_refit": float(raw.mean()),
                   "ll_recalibrated": float(l0.mean()), "ll_recalibrated_plus_squad": float(l1.mean()),
                   "diff_squad": C.paired_mean_diff(l1, l0), "coef_home_minus_away_std": coef}
    out["stacking_on_production"] = st
    # simple association: home-win rate by quintile of the XI market-value difference within Elo-difference bands
    q = p.loc[has_dyn(p) & (p["season"] <= 2025)].copy()
    q["elo_band"] = pd.qcut(q["elo_diff"], 3, labels=["слабее", "равны", "сильнее"])
    q["xi_q"] = q.groupby("elo_band", observed=True)["diff_sq_xi_value"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    q["home_win"] = (q["result"] == "H").astype(float)
    tab = q.groupby(["elo_band", "xi_q"], observed=True)["home_win"].agg(["mean", "size"]).reset_index()
    tab.to_csv(RES / "home_win_by_elo_and_xi.csv", index=False)
    out["correlations"] = {c: float(q[[c, "elo_diff"]].corr().iloc[0, 1]) for c in SQ_X}
    return out


# ------------------------------------------------------------------------------------------------ E2
def replay_inputs():
    m = pd.read_parquet(RC / "matches.parquet")
    res = D.results(m)
    prices = {u: pd.read_parquet(RC / f"prices_{k}.parquet") for u, k in UNIVERSES.items()}
    return m, res, prices


def main_of(out: pd.DataFrame, m, res, prices: pd.DataFrame, label: str) -> pd.DataFrame:
    batch = EN.batch_from_outputs(m, out, D.lookup(prices), label)
    cands = EV.candidates(batch.markets, m, res)
    main = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
    return cands.merge(main, on=["match_id", "market", "selection"])


def roi_block(new: pd.DataFrame, base: pd.DataFrame, ids) -> dict:
    a = new.loc[new["match_id"].isin(ids) & new["status"].isin(["WON", "LOST", "PUSH"])]
    b = base.loc[base["match_id"].isin(ids) & base["status"].isin(["WON", "LOST", "PUSH"])]
    return {"summary": EV.summary(a), "roi_ci": EV.roi_with_ci(a), "vs_b": EV.compare(a, b)}


def big_wins(main: pd.DataFrame, k: int = 5) -> dict:
    b = main.loc[main["bet"] == 1]
    top = b.sort_values("profit", ascending=False).head(k)
    rest = b.drop(top.index)
    return {"top_profit": float(top["profit"].sum()), "roi_without_top": float(rest["profit"].sum() / max(len(rest), 1))}


def odds_sensitivity(main: pd.DataFrame) -> dict:
    b = main.loc[main["bet"] == 1]
    out = {}
    for f in (0.98, 1.0, 1.02):
        prof = np.where(b["status"] == "WON", b["odds"] * f - 1, -1.0)
        out[str(f)] = float(prof.mean()) if len(b) else None
    return out


def e2(f: pd.DataFrame, log) -> tuple[dict, dict]:
    m, res, prices = replay_inputs()
    cur = f.loc[f["season"] == C.CURRENT]
    covered = set(cur.loc[cur["league"].isin(T.LEAGUE_MAP) & has_pre(cur), "match_id"])
    played = set(res.index)
    a = pd.read_parquet(RC / "baseline_a.parquet")
    report = {"season_matches": int(len(m)), "played": int(len(played)),
              "covered_leagues_matches": int(cur["league"].isin(T.LEAGUE_MAP).sum()),
              "covered_with_preseason_both": int(len(covered)),
              "covered_played": int(len(covered & played)), "universes": {}, "model": {}}
    mains = {}
    truth = cur.loc[cur["match_id"].isin(played & covered)]
    prod = pd.read_parquet(RC / "outputs_production.parquet")
    base_l = M.per_match_losses(prod, truth)
    for v in E2_VARIANTS:
        o = outputs(f"e2_{v}")
        cmp_ = M.compare_losses(M.per_match_losses(o, truth), base_l)
        o2 = o.set_index("match_id").loc[truth["match_id"]]
        report["model"][v] = {"vs_production": cmp_, "r2_corners": M.r2(truth["total_corners"], o2["corner_mu"])}
    hm = C.holm({v: report["model"][v]["vs_production"]["ll_1x2"]["p"] for v in E2_VARIANTS})
    for v, ph in hm.items():
        report["model"][v]["vs_production"]["ll_1x2"]["p_holm"] = ph
    p2 = prod.set_index("match_id").loc[truth["match_id"]]
    report["model"]["production_r2_corners"] = M.r2(truth["total_corners"], p2["corner_mu"])
    pv = {}
    for u, key in UNIVERSES.items():
        b = pd.read_parquet(RC / f"baseline_b_{u}.parquet")
        b = b.loc[b["match_id"].isin(played)]
        ur = {"baseline_b": {"covered": roi_block(b, b, covered), "all": roi_block(b, b, played)},
              "baseline_a": {"covered": {"summary": EV.summary(a.loc[a["match_id"].isin(covered & played)]),
                                         "roi_ci": EV.roi_with_ci(a.loc[a["match_id"].isin(covered & played)])},
                             "all": {"summary": EV.summary(a.loc[a["match_id"].isin(played)])}},
              "variants": {}}
        mains[(u, "baseline_b")] = b
        for v in E2_VARIANTS:
            mm = main_of(outputs(f"e2_{v}"), m, res, prices[u], f"squad_{v}")
            mm = mm.loc[mm["match_id"].isin(played)]
            mains[(u, v)] = mm
            cov_main = mm.loc[mm["match_id"].isin(covered)]
            ur["variants"][v] = {"covered": roi_block(mm, b, covered), "all": roi_block(mm, b, played),
                                 "big_wins_covered": big_wins(cov_main), "odds_sensitivity_covered":
                                 odds_sensitivity(cov_main)}
            pv[f"{u}|{v}"] = ur["variants"][v]["covered"]["vs_b"]["roi_diff"]["p"]
        ur["baseline_b"]["big_wins_covered"] = big_wins(b.loc[b["match_id"].isin(covered)])
        ur["baseline_b"]["odds_sensitivity_covered"] = odds_sensitivity(b.loc[b["match_id"].isin(covered)])
        report["universes"][u] = ur
        log.info("E2 %s done", u)
    for k, v in C.holm(pv).items():
        u, var = k.split("|")
        report["universes"][u]["variants"][var]["covered"]["vs_b"]["roi_diff"]["p_holm"] = v
    report["holm_family"] = {"tests": len(pv), "what": "variant × universe, ROI difference vs Baseline B (covered)"}
    return report, mains


def breakdowns(mains: dict, covered: set) -> None:
    rows = []
    for (u, v), mm in mains.items():
        c = mm.loc[mm["match_id"].isin(covered)].copy()
        c["month"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m")
        for by in ("month", "league"):
            t = EV.breakdown(c, by).rename(columns={by: "key"})
            t["by"], t["universe"], t["method"] = by, u, v
            rows.append(t)
    pd.concat(rows).to_csv(RES / "breakdown_e2.csv", index=False)


def per_match(mains: dict, res, covered: set) -> None:
    for (u, v), mm in mains.items():
        name = "baseline_b" if v == "baseline_b" else f"squad_{v}"
        t = EV.per_match_table(mm, res, f"{name}_{u}", EN.PRODUCTION_VERSION if v == "baseline_b" else f"squad_{v}_2026-07-01")
        t["covered_by_squad_data"] = t["match_id"].isin(covered)
        t.loc[~t["covered_by_squad_data"] & (t["note"].fillna("") == ""), "note"] = \
            "outside the squad-data sample (league not covered or team without pre-season squad)"
        t.to_csv(RES / f"per_match_{name}_{u}.csv", index=False)


def coverage_table(f: pd.DataFrame, e2r: dict, mains: dict, covered: set) -> pd.DataFrame:
    cur = f.loc[f["season"] == C.CURRENT]
    rows = []
    for (u, v), mm in mains.items():
        rows.append({"universe": u, "method": v, "season_matches": e2r["season_matches"], "played": e2r["played"],
                     "processed": int(mm["match_id"].nunique()),
                     "with_real_price": int(mm["odds"].notna().sum()),
                     "bets": int(mm["bet"].sum()),
                     "squad_sample": int(mm["match_id"].isin(covered).sum()),
                     "without_squad_data": int((~mm["match_id"].isin(covered)).sum()),
                     "excluded_no_price": int((mm["odds"].isna() & mm["status"].isin(["WON", "LOST"])).sum()),
                     "push_or_void": int((~mm["status"].isin(["WON", "LOST"])).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "coverage_e2.csv", index=False)
    reasons = cur.assign(reason=np.where(~cur["league"].isin(T.LEAGUE_MAP), "лиги нет в источнике игроков",
                                         np.where(~has_pre(cur), "команда без предсезонного состава (поднялась из "
                                                                 "лиги без данных)", "в выборке состава")))
    reasons.groupby(["league", "reason"]).size().rename("matches").reset_index().to_csv(
        RES / "coverage_e2_reasons.csv", index=False)
    return t


# ------------------------------------------------------------------------------------------------ figures
def figures(f: pd.DataFrame, cov: pd.DataFrame, e1r: dict, e2r: dict, mains: dict, covered: set) -> None:
    import matplotlib.pyplot as plt
    from scripts.report_kit import PALETTE  # noqa: F401  (style)

    # coverage by league
    c = cov.loc[cov["season"].between(2021, 2026)].copy()
    c["share"] = np.where(c["season"] < 2026, c["dyn_both"], c["pre_both"]) / c["matches"]
    piv = c.pivot(index="league", columns="season", values="share").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(len(piv.columns)), [f"{s}/{str(s + 1)[2:]}" for s in piv.columns])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="доля матчей с данными состава обеих команд")
    ax.set_title("Покрытие данными игроков по лигам и сезонам")
    fig.tight_layout()
    fig.savefig(FIG / "coverage.png")
    plt.close(fig)

    # E1 forest: 1X2 log loss and goal deviance differences
    labels, vals, lo, hi = [], [], [], []
    for season in ("2024", "2025"):
        for v, d in e1r[season]["variants"].items():
            for m, name in (("ll_1x2", "log loss 1X2"), ("dev_home", "девиация голов хозяев"),
                            ("ll_ou25", "log loss ТБ/ТМ 2.5"), ("dev_corners", "девиация угловых")):
                x = d["covered"][m]
                labels.append(f"{season}/{int(season) + 1 - 2000} {v.upper()}: {name}")
                vals.append(x["diff"])
                lo.append(x["ci95"][0])
                hi.append(x["ci95"][1])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(labels) + 1))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="o", ms=4)
    ax.axvline(0, color="#4b5a63", lw=0.8)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("разница с B0 (меньше — лучше), 95 % ДИ")
    ax.set_title("Ранние сезоны: изменение метрик от признаков состава")
    fig.tight_layout()
    fig.savefig(FIG / "e1_forest.png")
    plt.close(fig)

    # home-win rate by XI-value quintile within Elo bands
    t = pd.read_csv(RES / "home_win_by_elo_and_xi.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for band, g in t.groupby("elo_band", sort=False):
        ax.plot(g["xi_q"] + 1, g["mean"], marker="o", label=f"Elo: хозяева {band}")
    ax.set_xlabel("квинтиль разницы стоимости типичного XI (хозяева − гости) внутри группы Elo")
    ax.set_ylabel("доля побед хозяев")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=8)
    ax.set_title("Сила состава при одинаковом Elo (2021/22–2025/26)")
    fig.tight_layout()
    fig.savefig(FIG / "xi_within_elo.png")
    plt.close(fig)

    # E2 cumulative profit on the covered sample (close)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for v in ("baseline_b", "pre", "sa"):
        mm = mains.get(("close", v))
        if mm is None:
            continue
        c2 = mm.loc[mm["match_id"].isin(covered) & (mm["bet"] == 1)]
        daily = c2.groupby(pd.to_datetime(c2["date"]))["profit"].sum().sort_index().cumsum()
        ax.step(daily.index, daily.to_numpy(), where="post",
                label={"baseline_b": "Baseline B", "pre": "PRE", "sa": "SA-pre"}[v])
    ax.axhline(0, color="#4b5a63", lw=0.8)
    ax.set_ylabel("накопленная прибыль, ставок")
    ax.legend(fontsize=8)
    ax.set_title("2026/27, общая выборка с данными состава (close)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "e2_cumprofit.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def decide(e1r: dict, e2r: dict) -> dict:
    """Evidence and decision per variant (README §7). Model evidence: a 1X2 log-loss gain significant after Holm on
    a chronological check, same direction on every test season, no significant loss on the other leagues. The
    decision is capped at B when the variant cannot run in production (no maintained player source for the current
    season) or when Main ROI does not improve on 2026/27."""
    out = {}
    for v in ("dyn", "pre", "sa"):
        cells = [e1r[s]["variants"][v] for s in ("2024", "2025") if v in e1r[s]["variants"]]
        ll = [c["covered"]["ll_1x2"] for c in cells]
        sig = any(x.get("p_holm", 1) < 0.05 and x["diff"] < 0 for x in ll)
        same_dir = all(x["diff"] < 0 for x in ll)
        hurt_other = any(c["uncovered"]["ll_1x2"]["diff"] > 0 and c["uncovered"]["ll_1x2"]["p"] < 0.05 for c in cells)
        hurt_goals = any(c["covered"]["dev_home"]["diff"] > 0 and c["covered"]["dev_away"]["diff"] > 0 for c in cells)
        if sig and same_dir and not hurt_other:
            ev = "значимое (Холм) улучшение log loss 1X2 на ранних сезонах"
        elif same_dir and not hurt_other:
            ev = "улучшение log loss 1X2 того же знака, без значимости"
        else:
            ev = "нет устойчивого улучшения"
        if not (same_dir and not hurt_other) or hurt_goals:
            d = "C"
            why = ("ухудшает другие лиги или модели голов" if (hurt_other or hurt_goals) else "нет выигрыша")
        else:
            d = "B"
            why = ("нет поддерживаемого источника составов текущего сезона (датасет заморожен 06.07.2026); "
                   "Main ROI 2026/27 не улучшается" if v == "dyn" else
                   "Main ROI 2026/27 не улучшается; источник стоимости игроков заморожен и лицензионно спорен")
        out[f"e1_{v}"] = {"decision": d, "evidence": ev, "reason": why}
    for v in ("pre", "sa"):
        rc = e2r["universes"]["close"]["variants"][v]["covered"]["vs_b"]["roi_diff"]
        rt = e2r["universes"]["t09"]["variants"][v]["covered"]["vs_b"]["roi_diff"]
        ll = e2r["model"][v]["vs_production"]["ll_1x2"]
        model_better = ll["diff"] < 0 and ll.get("p_holm", 1) < 0.05
        roi_better = rc["diff"] > 0 and rt["diff"] > 0
        if model_better:
            d, ev = "B", "лучше log loss 1X2 (Холм) на 2026/27"
        elif ll["diff"] < 0 and not roi_better:
            d, ev = "C", "log loss 1X2 лучше без значимости, Main ROI не лучше"
        else:
            d, ev = ("B" if roi_better else "C"), "нет значимого улучшения модели"
        why = ("ΔROI close {:+.1f} п.п., t09 {:+.1f} п.п. — не значимо".format(rc["diff"] * 100, rt["diff"] * 100))
        out[f"e2_{v}"] = {"decision": d, "evidence": ev, "reason": why}
    return out


def preregister(dec: dict, e1r: dict, e2r: dict) -> list:
    entries = []
    for k, dd in dec.items():
        if dd["decision"] != "B":
            continue
        entries.append({
            "id": f"squad_{k}", "study": "squad", "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "hypothesis": "Squad-strength features from line-ups and dated market values improve the production "
                          "models (1X2 log loss) and Main ROI on the leagues covered by the player source.",
            "rule": {"e1_dyn": "production feature set + dynamic sq_* features (last 10 club games before the match)",
                     "e1_pre": "production feature set + pre-season pre_* features (squad on 1 July)",
                     "e1_sa": "compact squad-aware model (Elo, league, rest, short form + sq_*)",
                     "e2_pre": "production feature set + pre-season pre_* features, fitted on seasons before 1 July",
                     "e2_sa": "compact squad-aware model + pre_* features"}[k],
            "primary_metric": "paired 1X2 log loss vs Baseline B on covered leagues; secondary Main ROI (close)",
            "minimum_forward_sample": {
                "matches": max(3000, int(e1r["2025"]["variants"][k.split("_")[1]]["covered"]["ll_1x2"]
                                         ["n_for_80pct_power"] or 0)),
                "basis": "80 % power for the 2025/26 log-loss effect (paired sd), at least 3000 covered matches"},
            "decision": dd,
            "blocker": "needs a maintained player source for the forward season (the open dataset stopped on "
                       "2026-07-06); see API-Football pilot plan"})
    return entries


def main() -> None:
    started = time.time()
    log = C.setup_logging("squad.run", DIRS["logs"] / "run.log")
    f = table()
    cov = coverage(f)
    cov.to_csv(RES / "coverage_by_league_season.csv", index=False)
    C.write_json(RES / "sources.json", {"sources": SOURCES, "api_football_pilot": API_FOOTBALL_PILOT,
                                        "pilot_requests": pilot_plan()})
    e1r = e1(f, log)
    C.write_json(RES / "e1_earlier_seasons.json", e1r)
    be = beyond_elo(f)
    C.write_json(RES / "beyond_elo_form.json", be)
    e2r, mains = e2(f, log)
    cur = f.loc[f["season"] == C.CURRENT]
    covered = set(cur.loc[cur["league"].isin(T.LEAGUE_MAP) & has_pre(cur), "match_id"])
    C.write_json(RES / "e2_season_2026.json", e2r)
    breakdowns(mains, covered)
    m, res, _ = replay_inputs()
    per_match(mains, res, covered)
    coverage_table(f, e2r, mains, covered)
    figures(f, cov, e1r, e2r, mains, covered)
    dec = decide(e1r, e2r)
    C.write_json(RES / "decisions.json", dec)
    C.write_json(RES / "preregistration.json", preregister(dec, e1r, e2r))
    C.write_json(RES / "run_info.json", C.run_info("squad", started, {
        "player_source": "dcaribou/transfermarkt-datasets (CC0), frozen 2026-07-06",
        "window_games": 10, "xi": 11, "bench": 7, "per90_prior_minutes": 900, "impact_prior_starts": 10,
        "cutoff": "match date − 1 day", "preseason_date": "1 July of the season",
        "e1_tests": [2024, 2025], "e2_fit_cutoff": "2026-07-01", "variants_tried": {
            "e1": list(E1_VARIANTS), "e2": list(E2_VARIANTS), "sa_grid": len(M.GRID)}},
        {"data_player_source": {"tm_files": sorted(p.name for p in T.TM.glob("*.parquet"))}}))
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
