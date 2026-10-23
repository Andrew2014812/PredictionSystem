"""Q3 — total corners: feature inventory, enriched pre-match features, model variants, line quality.

Variants (corner leagues of each fold only; trees for new feature sets by early stopping on the
inner season; NB size r of every variant = negative_binomial_size on its INNER-season predictions,
as production does on its validation season):

    xgb_production        production total-corners XGBoost (count:poisson, tuned params)
    xgb_enriched          + extra strictly-shifted rolling corner / shot features (below)
    xgb_clean             corner-only subset (corner groups + league corners + shots) + extra features
    xgb_home_away         separate home- and away-corner XGBoost models (prod + extra features),
                          total = sum; line probabilities from the convolution of the two NBs
    poisson_glm           Poisson GLM, production feature set
    poisson_glm_enriched  Poisson GLM, production + extra features
    negbin_glm_clean      negative-binomial GLM (statsmodels), clean subset
    blend                 (1-w) xgb_production + w poisson_glm_enriched, w on the inner season
    naive_league_mean     training-season league average

Extra features (computed here from the per-match history; for every team the matches are
sorted by date and SHIFTED by one before any rolling statistic, so only past matches enter):
EWMA (half-life 5 and 15 matches) of corners for / against, venue-specific EWMA (half-life 5),
EWMA of shots / shots on target for and against, mean match total corners over the last 38
matches, and two match-level expectations (additive and multiplicative strength model).
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.features.sets import GROUPS
from src.modelling.training import negative_binomial_size
from experiments.model_comparison.models import NegBinGLMCandidate, PoissonGLMCandidate

from . import common as C

log = C.setup_logging("corners")
LINES = config.CORNER_LINES
MAXC = 40
CLEAN_GROUPS = ["season_corners", "venue_corners", "form_corners", "trends_corners", "diff_corners",
                "league_corners", "venue_form_corners", "corner_pressure", "opp_adjusted_corners"]


# ---------------------------------------------------------------------------------------
# extra features
# ---------------------------------------------------------------------------------------
def extra_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = df[["match_id", "date", "country", "league", "home_team", "away_team", "home_corners", "away_corners",
               "home_shots", "away_shots", "home_sot", "away_sot"]]
    rows = []
    for side, opp, venue in (("home", "away", "H"), ("away", "home", "A")):
        r = pd.DataFrame({"match_id": base["match_id"], "date": base["date"], "country": base["country"],
                          "team": base[f"{side}_team"], "venue": venue,
                          "cf": base[f"{side}_corners"], "ca": base[f"{opp}_corners"],
                          "sf": base[f"{side}_shots"], "sa": base[f"{opp}_shots"],
                          "tf": base[f"{side}_sot"], "ta": base[f"{opp}_sot"]})
        rows.append(r)
    long = pd.concat(rows, ignore_index=True).sort_values(["date", "match_id"], kind="mergesort")
    long["tot"] = long["cf"] + long["ca"]
    key = [long["country"], long["team"]]
    g = long.groupby(key, sort=False)
    out = pd.DataFrame(index=long.index)
    for col in ("cf", "ca", "sf", "sa", "tf", "ta", "tot"):
        long[f"prev_{col}"] = g[col].shift(1)
    g = long.groupby(key, sort=False)
    for hl in (5, 15):
        for col in ("cf", "ca"):
            out[f"ewm{hl}_{col}"] = g[f"prev_{col}"].transform(
                lambda s, hl=hl: s.ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
    for col in ("sf", "sa", "tf", "ta"):
        out[f"ewm10_{col}"] = g[f"prev_{col}"].transform(
            lambda s: s.ewm(halflife=10, min_periods=1, ignore_na=True).mean())
    out["r38_tot"] = g["prev_tot"].transform(lambda s: s.rolling(38, min_periods=3).mean())
    out["n_hist"] = g.cumcount()
    gv = long.groupby([long["country"], long["team"], long["venue"]], sort=False)
    for col in ("cf", "ca"):
        prev = gv[col].shift(1)
        out[f"venue_ewm5_{col}"] = prev.groupby([long["country"], long["team"], long["venue"]], sort=False).transform(
            lambda s: s.ewm(halflife=5, min_periods=1, ignore_na=True).mean())
    out["match_id"], out["venue"] = long["match_id"], long["venue"]
    h = out.loc[out["venue"] == "H"].drop(columns="venue").set_index("match_id").add_prefix("xh_")
    a = out.loc[out["venue"] == "A"].drop(columns="venue").set_index("match_id").add_prefix("xa_")
    feat = df[["match_id", "league_avg_home_corners", "league_avg_away_corners"]].set_index("match_id").join(h).join(a)
    feat["x_exp_home_add"] = (feat["xh_ewm15_cf"] + feat["xa_ewm15_ca"]) / 2
    feat["x_exp_away_add"] = (feat["xa_ewm15_cf"] + feat["xh_ewm15_ca"]) / 2
    feat["x_exp_total_add"] = feat["x_exp_home_add"] + feat["x_exp_away_add"]
    lh, la = feat["league_avg_home_corners"], feat["league_avg_away_corners"]
    lavg = (lh + la) / 2
    feat["x_exp_home_mult"] = lh * (feat["xh_venue_ewm5_cf"] / lh) * (feat["xa_venue_ewm5_ca"] / lh)
    feat["x_exp_away_mult"] = la * (feat["xa_venue_ewm5_cf"] / la) * (feat["xh_venue_ewm5_ca"] / la)
    feat["x_exp_total_mult"] = feat["x_exp_home_mult"] + feat["x_exp_away_mult"]
    feat["x_tot_hist_mean"] = (feat["xh_r38_tot"] + feat["xa_r38_tot"]) / 2 - lavg * 2
    feat = feat.drop(columns=["league_avg_home_corners", "league_avg_away_corners"])
    names = list(feat.columns)
    merged = df.merge(feat.reset_index(), on="match_id", how="left")
    return merged, names


def leakage_check(df: pd.DataFrame) -> dict:
    """Recompute one feature by brute force for a sample of matches (only strictly earlier matches)."""
    rng = np.random.default_rng(3)
    sample = df.loc[df["season"] >= 2022].sample(40, random_state=3)
    errs = []
    for r in sample.itertuples():
        past = df.loc[(df["date"] < r.date) & (df["country"] == r.country)
                      & ((df["home_team"] == r.home_team) | (df["away_team"] == r.home_team))]
        past = past.sort_values(["date", "match_id"])
        cf = np.where(past["home_team"] == r.home_team, past["home_corners"], past["away_corners"])
        s = pd.Series(cf).ewm(halflife=5, min_periods=1, ignore_na=True).mean()
        ref = s.iloc[-1] if len(s) else np.nan
        val = getattr(r, "xh_ewm5_cf")
        if np.isfinite(ref) or np.isfinite(val):
            errs.append(abs((ref if np.isfinite(ref) else 0) - (val if np.isfinite(val) else 0)))
    _ = rng
    return {"checked": len(errs), "max_abs_diff": float(np.max(errs)) if errs else None}


# ---------------------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------------------
def p_over(mu, r, line):
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    k = int(np.floor(line))
    if r:
        return stats.nbinom.sf(k, r, r / (r + mu))
    return stats.poisson.sf(k, mu)


def pmf_matrix(mu, r):
    k = np.arange(MAXC + 1)[None, :]
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    pmf = stats.nbinom.pmf(k, r, r / (r + mu)) if r else stats.poisson.pmf(k, mu)
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
    return pmf


def p_over_convolved(mu_h, r_h, mu_a, r_a, line):
    ph, pa = pmf_matrix(mu_h, r_h), pmf_matrix(mu_a, r_a)
    cdf_a = np.cumsum(pa, 1)
    k = int(np.floor(line))
    under = sum(ph[:, i] * cdf_a[:, k - i] for i in range(0, k + 1))
    return 1 - under


# ---------------------------------------------------------------------------------------
def numeric_from_groups(groups, cols):
    pats = [re.compile(p) for g in groups for p in GROUPS[g]]
    return [c for c in cols if any(p.search(c) for p in pats)]


def inventory(df: pd.DataFrame, extra: list[str]) -> dict:
    prod = C.task_numeric("corners")
    train = df.loc[df["season"] < 2023]
    corner_cols = [c for c in df.columns if "corner" in c and c not in ("home_corners", "away_corners", "total_corners")]
    rows = []
    for c in corner_cols + extra + [c for c in df.columns if re.search(r"^[ha]_form10_(shots|sot)_", c)]:
        if c not in df:
            continue
        v = train[c]
        corr = float(np.corrcoef(v.fillna(v.mean()), train["total_corners"].fillna(train["total_corners"].mean()))[0, 1])
        rows.append({"feature": c, "in_production": c in prod, "extra": c in extra,
                     "coverage_train": float(v.notna().mean()), "corr_total_corners_train": corr})
    cov = df.groupby("league")["total_corners"].apply(lambda s: float(s.notna().mean())).to_dict()
    return {"production_feature_count": len(prod), "production_features": prod,
            "production_groups": ["table", "season_attack", "season_corners", "venue_corners", "form_corners",
                                  "trends_corners", "rest", "diff_corners", "league_corners", "league_goals",
                                  "venue_form_corners", "corner_pressure", "form_goals", "venue_form_goals"],
            "missing_from_production": [g for g in ("opp_adjusted_corners",)],
            "features": sorted(rows, key=lambda r: -abs(r["corr_total_corners_train"])),
            "corner_coverage_by_league": cov}


def run() -> dict:
    t_start = time.time()
    raw = C.features()
    df, extra = extra_features(raw)
    out = {"extra_features": extra, "leakage_check": leakage_check(df), "inventory": inventory(df, extra), "folds": {}}
    log.info("leakage check %s", out["leakage_check"])
    prod_num = C.task_numeric("corners")
    clean_num = numeric_from_groups(CLEAN_GROUPS, list(df.columns)) + [
        c for c in df.columns if re.search(r"^[ha]_form10_(shots_for|shots_against|sot_for|sot_against)$", c)] + extra
    enriched_num = prod_num + extra
    params, n_prod = C.prod_params("corners")
    store = {}
    for s in C.FOLDS:
        inner, outer, info = C.base_predictions(s)
        cl = info["outer"]["corner_leagues"]
        has = df["total_corners"].notna() & df["league"].isin(cl)
        tr_in, va_in = df.loc[has & (df["season"] < s - 1)], df.loc[has & (df["season"] == s - 1)]
        tr_out, te = df.loc[has & (df["season"] < s)], df.loc[has & (df["season"] == s)]
        y_in, y = va_in["total_corners"].to_numpy(float), te["total_corners"].to_numpy(float)
        mu_in, mu_out, secs, prm = {}, {}, {}, {}

        base_in = inner.set_index("match_id").loc[va_in["match_id"], "cmu"].to_numpy()
        base_out = outer.set_index("match_id").loc[te["match_id"], "cmu"].to_numpy()
        mu_in["xgb_production"], mu_out["xgb_production"] = base_in, base_out
        secs["xgb_production"] = info["outer"]["seconds"]["corners"]

        for name, num in (("xgb_enriched", enriched_num), ("xgb_clean", clean_num)):
            t0 = time.time()
            m = C.XGBFit(num, params, 1, "count:poisson", "poisson-nloglik", False)
            m.n = m.early_stopped_trees(tr_in, tr_in["total_corners"], va_in, va_in["total_corners"])
            mu_in[name] = m.fit(tr_in, tr_in["total_corners"]).predict(va_in)
            mu_out[name] = m.fit(tr_out, tr_out["total_corners"]).predict(te)
            secs[name] = m.seconds
            prm[name] = {"n_trees": m.n, "n_features": len(num)}
            log.info("fold %s %s done in %.0fs", s, name, time.time() - t0)

        # separate home / away corner models
        side_mu_in, side_mu_out, side_r = {}, {}, {}
        side_secs = 0.0
        for side in ("home_corners", "away_corners"):
            ok_in = tr_in[side].notna()
            m = C.XGBFit(enriched_num, params, 1, "count:poisson", "poisson-nloglik", False)
            m.n = m.early_stopped_trees(tr_in.loc[ok_in], tr_in.loc[ok_in, side], va_in, va_in[side])
            side_mu_in[side] = m.fit(tr_in.loc[ok_in], tr_in.loc[ok_in, side]).predict(va_in)
            side_r[side] = negative_binomial_size(va_in[side].to_numpy(float), side_mu_in[side])
            ok_out = tr_out[side].notna()
            side_mu_out[side] = m.fit(tr_out.loc[ok_out], tr_out.loc[ok_out, side]).predict(te)
            side_secs += m.seconds
            prm.setdefault("xgb_home_away", {})[f"{side}_n_trees"] = m.n
        mu_in["xgb_home_away"] = side_mu_in["home_corners"] + side_mu_in["away_corners"]
        mu_out["xgb_home_away"] = side_mu_out["home_corners"] + side_mu_out["away_corners"]
        prm["xgb_home_away"]["nb_size_home"], prm["xgb_home_away"]["nb_size_away"] = (
            side_r["home_corners"], side_r["away_corners"])
        secs["xgb_home_away"] = side_secs

        for name, cand, num in (("poisson_glm", PoissonGLMCandidate, prod_num),
                                ("poisson_glm_enriched", PoissonGLMCandidate, enriched_num),
                                ("negbin_glm_clean", NegBinGLMCandidate, clean_num)):
            t0 = time.time()
            g = cand("count", num)
            if name == "negbin_glm_clean":
                g.grid = [1e-2]                       # one L2 strength: statsmodels NB is slow
            g.tune(tr_in, va_in, tr_in["total_corners"].to_numpy(float), y_in)
            mu_in[name] = g.valid_pred
            t1 = time.time()
            g.fit(tr_out, tr_out["total_corners"].to_numpy(float))
            mu_out[name] = g.predict(te)
            secs[name] = time.time() - t1
            prm[name] = {"l2": g.params["regularisation"], "n_features": len(num)}
            log.info("fold %s %s done in %.0fs", s, name, time.time() - t0)

        grid = np.round(np.arange(0, 1.0001, 0.1), 1)
        dev = {float(w): float(C.poisson_dev(y_in, (1 - w) * base_in + w * mu_in["poisson_glm_enriched"]).mean())
               for w in grid}
        w = min(dev, key=dev.get)
        mu_in["blend"] = (1 - w) * base_in + w * mu_in["poisson_glm_enriched"]
        mu_out["blend"] = (1 - w) * base_out + w * mu_out["poisson_glm_enriched"]
        prm["blend"] = {"weight_glm": w}
        secs["blend"] = secs["xgb_production"] + secs["poisson_glm_enriched"]
        lm = tr_out.groupby("league")["total_corners"].mean()
        lm_in = tr_in.groupby("league")["total_corners"].mean()
        mu_out["naive_league_mean"] = te["league"].map(lm).to_numpy()
        mu_in["naive_league_mean"] = va_in["league"].map(lm_in).fillna(tr_in["total_corners"].mean()).to_numpy()
        secs["naive_league_mean"] = 0.0

        base_rate = {line: float((tr_out["total_corners"] > line).mean()) for line in LINES}
        league_rate = {line: tr_out.assign(o=tr_out["total_corners"] > line).groupby("league")["o"].mean()
                       for line in LINES}
        fold = {"n": int(len(te)), "variants": {}, "params": prm, "train_seconds": secs,
                "line_baselines": {}}
        base_dev = C.poisson_dev(y, base_out)
        line_probs = {}
        for name in mu_out:
            r = negative_binomial_size(y_in, mu_in[name])
            met = C.count_metrics(y, mu_out[name])
            met["delta_deviance_vs_production"] = C.paired_bootstrap(C.poisson_dev(y, mu_out[name]), base_dev)
            met["nb_size_inner"] = r
            met["lines"] = {}
            for line in LINES:
                if name == "xgb_home_away":
                    p = p_over_convolved(side_mu_out["home_corners"], side_r["home_corners"],
                                         side_mu_out["away_corners"], side_r["away_corners"], line)
                else:
                    p = p_over(mu_out[name], r, line)
                o = y > line
                bm = C.binary_metrics(o, p, base_rate[line])
                lr = te["league"].map(league_rate[line]).fillna(base_rate[line]).to_numpy()
                bm["league_rate_log_loss"] = float(np.mean(-(o * np.log(np.clip(lr, 1e-6, 1))
                                                             + (1 - o) * np.log(np.clip(1 - lr, 1e-6, 1)))))
                bm["reliability"] = C.reliability(o, p)
                met["lines"][str(line)] = bm
                line_probs[(name, line)] = p
            met["mean_line_log_loss"] = float(np.mean([met["lines"][str(l)]["log_loss"] for l in LINES]))
            fold["variants"][name] = met
        fold["line_baselines"] = {str(l): {"base_rate_train": base_rate[l]} for l in LINES}
        out["folds"][str(s)] = fold
        store[s] = (y, line_probs)
        pd.DataFrame({f"{n}|{l}": p for (n, l), p in line_probs.items()}).assign(y=y).to_parquet(
            C.CACHE / f"corner_lines_{s}.parquet")
        log.info("fold %s: %s", s, {n: (round(v["rmse"], 3), round(v["mean_line_log_loss"], 4))
                                     for n, v in fold["variants"].items()})

    out["selection"] = select(out)
    out["minutes"] = (time.time() - t_start) / 60
    C.save_json("corners.json", out)
    figures(out, store)
    return out


def select(out: dict) -> dict:
    """Replace production only if CLEARLY better on both selection folds: lower Poisson deviance with the
    bootstrap 95 % CI of the difference below 0, AND lower mean line log loss (8.5/9.5/10.5)."""
    names = [n for n in out["folds"]["2023"]["variants"] if n != "naive_league_mean"]
    sel = {}
    for n in names:
        v = [out["folds"][str(s)]["variants"][n] for s in C.SELECT]
        b = [out["folds"][str(s)]["variants"]["xgb_production"] for s in C.SELECT]
        sel[n] = {"deviance_better_both": all(x["poisson_deviance"] < y_["poisson_deviance"] for x, y_ in zip(v, b)),
                  "deviance_ci_below_zero_both": all(x["delta_deviance_vs_production"]["ci95"][1] < 0 for x in v),
                  "line_log_loss_better_both": all(x["mean_line_log_loss"] < y_["mean_line_log_loss"]
                                                   for x, y_ in zip(v, b)),
                  "rmse_better_both": all(x["rmse"] < y_["rmse"] for x, y_ in zip(v, b)),
                  "pooled_line_log_loss": float(np.mean([x["mean_line_log_loss"] for x in v])),
                  "pooled_deviance": float(np.mean([x["poisson_deviance"] for x in v]))}
    weak = [n for n in names if n != "xgb_production" and sel[n]["deviance_better_both"]
            and sel[n]["line_log_loss_better_both"]]
    ok = [n for n in weak if sel[n]["deviance_ci_below_zero_both"]]
    chosen = min(ok, key=lambda n: sel[n]["pooled_line_log_loss"]) if ok else "xgb_production"
    return {"table": sel, "better_both_point_estimate": weak, "eligible": ok, "chosen": chosen,
            "rule": "Poisson deviance (95 % CI of the difference < 0) AND mean line log loss (8.5/9.5/10.5) "
                    "better than production on both 2023 and 2024; then lowest mean line log loss"}


TITLES = {"xgb_production": "XGB прод.", "xgb_enriched": "XGB + новые", "xgb_clean": "XGB угловой набор",
          "xgb_home_away": "XGB дома+в гостях", "poisson_glm": "Poisson GLM", "poisson_glm_enriched": "Poisson GLM + новые",
          "negbin_glm_clean": "NB GLM", "blend": "Бленд", "naive_league_mean": "Среднее лиги"}


def figures(out, store):
    plt = C.plot_style()
    names = list(out["folds"]["2023"]["variants"])
    metrics = (("rmse", "RMSE"), ("r2", "R²"), ("poisson_deviance", "Пуассоновская девиация"),
               ("mean_line_log_loss", "Средний log loss линий 8.5/9.5/10.5"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = np.arange(len(names))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for k, s in enumerate(C.FOLDS):
            vals = [out["folds"][str(s)]["variants"][n][key] for n in names]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=C.PALETTE[k],
                   label=config.season_label(s) + ("" if s in C.SELECT else " (только для информации)"))
        if key == "mean_line_log_loss":
            for k, s in enumerate(C.FOLDS):
                f = out["folds"][str(s)]["variants"]["xgb_production"]["lines"]
                ax.axhline(np.mean([f[str(l)]["baseline_log_loss"] for l in LINES]), color=C.PALETTE[k], ls="--",
                           lw=0.8)
        vals_all = [out["folds"][str(s)]["variants"][n][key] for s in C.FOLDS for n in names]
        lo, hi = min(vals_all), max(vals_all)
        if key != "r2":
            ax.set_ylim(lo - (hi - lo) * 0.3, hi + (hi - lo) * 0.15)
        ax.set_xticks(x, [TITLES[n] for n in names], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Угловые: сравнение вариантов (пунктир на log loss — базовая частота обучающих сезонов)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "corners_comparison.png"), plt.close(fig)

    chosen = out["selection"]["chosen"]
    show = list(dict.fromkeys(["xgb_production", chosen, "blend", "xgb_home_away", "negbin_glm_clean"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, line in zip(axes, LINES):
        for k, n in enumerate(show):
            y = np.concatenate([store[s][0] for s in C.SELECT])
            p = np.concatenate([store[s][1][(n, line)] for s in C.SELECT])
            rel = C.reliability(y > line, p, bins=10, min_n=30)
            ax.plot([r["predicted"] for r in rel], [r["observed"] for r in rel], "o-", ms=3, color=C.PALETTE[k],
                    label=TITLES[n])
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.set_title(f"Тотал угловых больше {line} (2023/24 + 2024/25)")
        ax.set_xlabel("Предсказанная вероятность")
    axes[0].set_ylabel("Фактическая частота")
    axes[0].legend(fontsize=8)
    fig.suptitle("Калибровка вероятностей линий угловых (NB, r по внутренней валидации)")
    fig.savefig(C.FIGURES / "corners_line_calibration.png"), plt.close(fig)


if __name__ == "__main__":
    run()
