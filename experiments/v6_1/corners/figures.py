"""Figures of the corners study (all numbers from results/*.json and the cached line probabilities)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scripts.report_kit import ACCENT, AMBER, MUTED, NEG, PALETTE, plt, save_chart

from src import config

from . import data as D
from . import dist as Q
from .evaluate import EXT_LINES, REF, TITLES

FOLD_COLORS = {2023: PALETTE[0], 2024: PALETTE[1], 2025: PALETTE[2], 2026: PALETTE[3]}
MAIN = ["xgb_current", "xgb_enriched", "xgb_tuned", "xgb_ts", "lgbm", "catboost", "poisson_glm", "nb_glm",
        "team_strength", "xgb_ts_offset", "xgb_home_away", "ens_weighted", "ens_simple", "naive_league"]


def label(s):
    tag = {2023: "", 2024: "", 2025: " (инф.)", 2026: " (проверка)"}[s]
    return config.season_label(s) + tag


def fig_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    y = df.loc[df["season"] < 2026, "total_corners"].to_numpy(int)
    k = np.arange(0, 26)
    h = np.bincount(np.clip(y, 0, 25), minlength=26) / len(y)
    m, v = y.mean(), y.var()
    r = m * m / (v - m)
    axes[0].bar(k, h, color=PALETTE[0], alpha=0.75, label="Факт 2021/22–2025/26")
    axes[0].plot(k, stats.poisson.pmf(k, m), "o-", ms=3, color=NEG, label=f"Пуассон (μ={m:.2f})")
    axes[0].plot(k, stats.nbinom.pmf(k, r, r / (r + m)), "s--", ms=3, color=AMBER, label=f"NB (r={r:.0f})")
    axes[0].set_xlabel("Всего угловых в матче")
    axes[0].set_ylabel("Доля матчей")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Безусловное распределение тотала угловых", fontsize=9)
    g = df.loc[df["season"] < 2026].groupby("league")["total_corners"].agg(["mean", "var"]).sort_values("mean")
    x = np.arange(len(g))
    axes[1].bar(x - 0.2, g["mean"], 0.4, color=PALETTE[0], label="Среднее")
    axes[1].bar(x + 0.2, g["var"], 0.4, color=PALETTE[1], label="Дисперсия")
    axes[1].set_xticks(x, g.index, rotation=90, fontsize=7)
    axes[1].set_ylim(5, max(g["var"].max(), g["mean"].max()) * 1.08)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Среднее и дисперсия тотала по лигам", fontsize=9)
    return save_chart(fig, D.FIGURES / "distribution.png")


def fig_features(inv):
    rows = [r for r in inv["features"] if r["corr_total"] is not None][:25]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(rows))[::-1]
    colors = [PALETTE[1] if r["new"] else (PALETTE[0] if r["in_production"] else PALETTE[2]) for r in rows]
    ax.barh(y, [r["corr_total"] for r in rows], color=colors)
    ax.set_yticks(y, [r["feature"] for r in rows], fontsize=7)
    ax.set_xlabel("Корреляция Пирсона с тоталом угловых (2021/22–2022/23)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], label="в продакшн-модели"), Patch(color=PALETTE[1], label="новый признак"),
                       Patch(color=PALETTE[2], label="есть в таблице, не в модели")], fontsize=7, loc="lower right")
    ax.set_title("25 признаков с наибольшей |r|", fontsize=9)
    return save_chart(fig, D.FIGURES / "feature_corr.png")


def fig_models(met):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    names = [n for n in MAIN if f"{n}|nb_mom" in met["folds"]["2023"]["dist"]]
    x = np.arange(len(names))
    specs = (("mean_line_log_loss", "Средний log loss линий 8.5/9.5/10.5", "dist"),
             ("log_score", "Log score полного распределения", "dist"),
             ("rmse", "RMSE", "point"), ("r2", "R²", "point"))
    for ax, (key, title, src) in zip(axes.ravel(), specs):
        vals_all = []
        for k, s in enumerate(D.FOLDS):
            f = met["folds"][str(s)]
            vals = [f["dist"][f"{n}|nb_mom"][key] if src == "dist" else f["point"][n][key] for n in names]
            vals_all += vals
            ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, color=FOLD_COLORS[s], label=label(s))
        lo, hi = min(vals_all), max(vals_all)
        if key != "r2":
            ax.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.1)
        ax.set_xticks(x, [TITLES[n] for n in names], rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Модели среднего (распределение NB, r методом моментов — как в продакшне)", fontsize=10)
    return save_chart(fig, D.FIGURES / "models_metrics.png")


def fig_delta(met, keys, fname, title):
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(keys) + 1.2))
    for j, k in enumerate(keys):
        for i, s in enumerate(D.FOLDS):
            d = met["folds"][str(s)]["dist"][k]["delta_line_log_loss"]
            yy = len(keys) - j + (i - 1.5) * 0.18
            ax.errorbar(d["mean"], yy, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o", ms=3,
                        color=FOLD_COLORS[s], label=label(s) if j == 0 else None, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(np.arange(len(keys), 0, -1), [pretty(k) for k in keys], fontsize=7)
    ax.set_xlabel("Δ log loss линий против текущей модели (95 % ДИ; < 0 — лучше)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=9)
    return save_chart(fig, D.FIGURES / fname)


def pretty(k):
    n, d = k.split("|")
    return f"{TITLES.get(n, n)} · {d}"


def fig_dispersion(met):
    models = ["xgb_current", "xgb_home_away", "team_strength", "ens_weighted"]
    dists = ["poisson", "nb_mom", "nb_mle", "nb_league", "nb1", "conv_indep", "nb_joint"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((axes[0], "log_score", "Log score"), (axes[1], "mean_line_log_loss", "Log loss линий")):
        for j, m in enumerate(models):
            vals = []
            for d in dists:
                k = f"{m}|{d}"
                v = [met["folds"][str(s)]["dist"][k][key] for s in D.SELECT if k in met["folds"][str(s)]["dist"]]
                vals.append(np.mean(v) if v else np.nan)
            ax.plot(range(len(dists)), vals, "o-", color=PALETTE[j], label=TITLES[m], ms=4)
        ax.set_xticks(range(len(dists)), dists, rotation=30, fontsize=8)
        ax.set_title(f"{title}, среднее 2023/24 и 2024/25", fontsize=9)
    axes[0].legend(fontsize=7)
    return save_chart(fig, D.FIGURES / "dispersion.png")


def fig_pit(met, keys):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        for s in D.FOLDS:
            h = met["folds"][str(s)]["dist"][k]["pit"]["hist"]
            ax.step(np.arange(0.05, 1.0, 0.1), h, where="mid", color=FOLD_COLORS[s], label=label(s))
        ax.axhline(0.1, color=MUTED, ls=":")
        ax.set_ylim(0.06, 0.14)
        ax.set_title(pretty(k), fontsize=8)
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("Доля матчей")
    axes[0].legend(fontsize=7)
    return save_chart(fig, D.FIGURES / "pit.png")


def fig_line_calibration(keys):
    lp = pd.read_parquet(D.CACHE / "line_probs.parquet")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
    for row, seasons, tag in ((0, D.SELECT, "2023/24 + 2024/25"), (1, (D.CHECK,), "2026/27")):
        for ax, line in zip(axes[row], D.LINES):
            for j, k in enumerate(keys):
                m, d = k.split("|")
                g = lp.loc[(lp["model"] == m) & (lp["dist"] == d) & lp["season"].isin(seasons)]
                q = pd.qcut(g[f"p{line}"].rank(method="first"), 10, labels=False)
                rel = g.assign(o=(g["y"] > line).astype(float), q=q).groupby("q").agg(
                    p=(f"p{line}", "mean"), o=("o", "mean"))
                ax.plot(rel["p"], rel["o"], "o-", ms=3, color=PALETTE[j], label=pretty(k))
            ax.plot([0, 1], [0, 1], ":", color="grey")
            ax.set_xlim(0.15, 0.85)
            ax.set_ylim(0.15, 0.85)
            ax.set_title(f"Больше {line} — {tag} (децили прогноза)", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Предсказанная вероятность")
    for ax in axes[:, 0]:
        ax.set_ylabel("Фактическая частота")
    axes[0, 0].legend(fontsize=7)
    return save_chart(fig, D.FIGURES / "line_calibration.png")


def fig_deciles(met, names):
    fig, axes = plt.subplots(1, len(D.FOLDS), figsize=(13, 3.4), sharey=True)
    for ax, s in zip(axes, D.FOLDS):
        for j, n in enumerate(names):
            dec = met["folds"][str(s)]["point"][n]["deciles"]
            ax.plot([d["predicted"] for d in dec], [d["observed"] for d in dec], "o-", ms=3, color=PALETTE[j],
                    label=TITLES[n])
        ax.plot([7, 13], [7, 13], ":", color="grey")
        ax.set_title(label(s), fontsize=9)
        ax.set_xlabel("Прогноз (среднее в децили)")
    axes[0].set_ylabel("Факт")
    axes[0].legend(fontsize=6)
    return save_chart(fig, D.FIGURES / "deciles.png")


def fig_phase(met, keys):
    phases = ["1_Jul-Sep", "2_Oct-Dec", "3_Jan-Mar", "4_Apr-Jun"]
    names = {"1_Jul-Sep": "июл–сен", "2_Oct-Dec": "окт–дек", "3_Jan-Mar": "янв–мар", "4_Apr-Jun": "апр–июн"}
    fig, axes = plt.subplots(1, len(D.FOLDS), figsize=(13, 3.4), sharey=True)
    for ax, s in zip(axes, D.FOLDS):
        for j, k in enumerate(keys):
            bp = met["folds"][str(s)]["dist"][k]["by_phase"]
            ph = [p for p in phases if p in bp]
            ax.plot([names[p] for p in ph], [bp[p]["mean_line_log_loss"] for p in ph], "o-", color=PALETTE[j],
                    label=pretty(k), ms=4)
        base = met["folds"][str(s)]["baselines"]
        ax.set_title(label(s), fontsize=9)
    axes[0].set_ylabel("Log loss линий")
    axes[0].legend(fontsize=6)
    return save_chart(fig, D.FIGURES / "phase.png")


def fig_league(met, key):
    rows = []
    for lg in met["folds"]["2023"]["dist"][key]["by_league"]:
        d, n = [], 0
        for s in D.SELECT:
            a = met["folds"][str(s)]["dist"][key]["by_league"].get(lg)
            b = met["folds"][str(s)]["dist"]["|".join(REF)]["by_league"].get(lg)
            if a and b:
                d.append((a["mean_line_log_loss"] - b["mean_line_log_loss"]) * a["n"])
                n += a["n"]
        if n:
            rows.append((lg, sum(d) / n, n))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[ACCENT if r[1] < 0 else NEG for r in rows])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Δ log loss линий")
    ax.set_title(f"{pretty(key)} против текущей модели по лигам (2023/24 + 2024/25)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    return save_chart(fig, D.FIGURES / "league.png")


def fig_betting(bet, keys):
    strat = ["value_0", "value_0.05", "best_0", "best_0.05", "ext_value_0"]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(keys)
    for j, k in enumerate(keys):
        for i, s in enumerate(strat):
            r = bet["combos"][k][s]
            if not r["n"]:
                continue
            x = i + (j - (len(keys) - 1) / 2) * w
            ax.errorbar(x, r["roi"] * 100, yerr=[[(r["roi"] - r["ci95"][0]) * 100], [(r["ci95"][1] - r["roi"]) * 100]],
                        fmt="o", color=PALETTE[j], ms=4, label=pretty(k) if i == 0 else None)
            ax.annotate(str(r["n"]), (x, r["ci95"][1] * 100), fontsize=6, ha="center", va="bottom", color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(strat)), ["EV≥0, 3 линии", "EV≥5 %, 3 линии", "лучшая, EV≥0", "лучшая, EV≥5 %",
                                      "EV≥0, 7.5–12.5"], fontsize=8)
    ax.set_ylabel("ROI, % (95 % ДИ)")
    ax.legend(fontsize=7)
    ax.set_title("Ставки по реальным ценам Bet365 на угловые, 2026/27 (число ставок над интервалом)", fontsize=9)
    return save_chart(fig, D.FIGURES / "betting.png")


def fig_market(bet, key):
    mk = bet["market"][key]
    lines = [l for l in map(str, EXT_LINES) if l in mk]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(lines))
    ax.bar(x - 0.2, [mk[l]["market_log_loss"] for l in lines], 0.4, color=PALETTE[2], label="Bet365 без маржи")
    ax.bar(x + 0.2, [mk[l]["model_log_loss"] for l in lines], 0.4, color=PALETTE[0], label=pretty(key))
    ax.set_xticks(x, [f"{l}\nn={mk[l]['n']}" for l in lines], fontsize=7)
    lo = min(min(mk[l]["market_log_loss"], mk[l]["model_log_loss"]) for l in lines)
    ax.set_ylim(lo - 0.03, 0.72)
    ax.set_ylabel("Log loss")
    ax.legend(fontsize=7)
    ax.set_title("Модель и рынок на линиях угловых, 2026/27", fontsize=9)
    return save_chart(fig, D.FIGURES / "market.png")


def fig_history(cur):
    roles = cur["history_2026"]["roles"]
    names = [r for r in ("market", "main", "risk", "all") if r in roles]
    lab = {"market": "Рынок (9.5)", "main": "Main", "risk": "Risk", "all": "Все"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, r in enumerate(names):
        v = roles[r]
        ax.errorbar(i, v["roi"] * 100, yerr=[[(v["roi"] - v["roi_ci95"][0]) * 100], [(v["roi_ci95"][1] - v["roi"]) * 100]],
                    fmt="o", color=PALETTE[i], ms=5)
        ax.annotate(f"n={v['n']}", (i, v["roi_ci95"][1] * 100), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(names)), [lab[n] for n in names])
    ax.set_ylabel("ROI, % (95 % ДИ)")
    ax.set_title("История прогнозов 2026/27: ставки на угловые по реальным ценам", fontsize=9)
    return save_chart(fig, D.FIGURES / "history.png")


def fig_team_strength(grid):
    keys = list(grid)
    seasons = sorted({s for v in grid.values() for s in v})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, s in enumerate(seasons):
        ax.plot(range(len(keys)), [grid[k][s] for k in keys], "o-", ms=3, color=PALETTE[j % len(PALETTE)],
                label=config.season_label(int(s)))
    ax.set_xticks(range(len(keys)), [k.replace("|", "\nhl=") for k in keys], fontsize=6)
    ax.set_ylabel("Пуассоновская девиация")
    ax.set_xlabel("alpha (L2) / период полураспада, дней")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Иерархическая модель: сетка гиперпараметров (скользящее переобучение)", fontsize=9)
    return save_chart(fig, D.FIGURES / "team_strength_grid.png")


def fig_weights(met):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    members = list(met["folds"]["2023"]["infos"]["ens_weighted"]["weights"])
    x = np.arange(len(members))
    for i, s in enumerate(D.FOLDS):
        w = met["folds"][str(s)]["infos"]["ens_weighted"]["weights"]
        ax.bar(x + (i - 1.5) * 0.2, [w.get(m, 0) for m in members], 0.2, color=FOLD_COLORS[s], label=label(s))
    ax.set_xticks(x, [TITLES[m] for m in members], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Вес")
    ax.legend(fontsize=7)
    ax.set_title("Веса ансамбля (минимум девиации на внутреннем сезоне)", fontsize=9)
    return save_chart(fig, D.FIGURES / "ensemble_weights.png")


def run():
    met = D.load_result("metrics.json")
    inv = D.load_result("inventory.json")
    bet = D.load_result("betting.json")
    cur = D.load_result("current.json")
    grid = D.load_result("team_strength_grid.json")
    df = D.with_corners(D.frame())
    ref = "|".join(REF)
    sel = met["selection"]
    tab = sel["table"]
    best = sorted([k for k in tab if k.endswith("|nb_mom") and not k.startswith("naive")],
                  key=lambda k: tab[k]["pooled_select_line_ll"])[:3]
    fig_distribution(df)
    fig_features(inv)
    fig_models(met)
    fig_delta(met, [f"{n}|nb_mom" for n in MAIN if n != "xgb_current" and f"{n}|nb_mom" in met["folds"]["2023"]["dist"]],
              "delta_models.png", "Модели среднего (NB, r методом моментов) против текущей")
    fig_delta(met, sorted(sel["table"], key=lambda k: sel["table"][k]["pooled_select_line_ll"])[:8], "delta_best.png",
              "Восемь лучших пар модель · распределение на фолдах отбора")
    fig_dispersion(met)
    fig_pit(met, [ref] + best[:2])
    fig_line_calibration([ref] + best[:2])
    fig_deciles(met, ["xgb_current", "poisson_glm", "team_strength", "ens_weighted"])
    fig_phase(met, [ref] + best[:1] + ["naive_league|nb_mom"])
    fig_league(met, best[0])
    fig_betting(bet, [ref] + best[:3])
    fig_market(bet, ref)
    fig_history(cur)
    fig_team_strength(grid)
    fig_weights(met)
