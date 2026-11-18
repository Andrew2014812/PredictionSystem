"""Figures of the corners study (read from the results dictionary of run.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import report_kit as K
from experiments.v6_3.corners import data as D
from experiments.v6_3.corners import evaluate as E

F = D.FIGURES
plt = K.plt


def _ci_bar(ax, names, vals, lo, hi, color):
    yv = np.arange(len(names))
    ax.barh(yv, vals, color=color, alpha=0.85)
    ax.errorbar(vals, yv, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                ecolor=K.INK, capsize=3, lw=1)
    ax.set_yticks(yv)
    ax.set_yticklabels(names)
    ax.axvline(0, color=K.MUTED, lw=0.8)


def coverage(out):
    t = pd.read_csv(D.RESULTS / "coverage_league_season.csv")
    piv = t.pivot(index="league", columns="season", values="mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    im = axes[0].imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(piv.columns)))
    axes[0].set_xticklabels([f"{s % 100:02d}/{(s + 1) % 100:02d}" for s in piv.columns])
    axes[0].set_yticks(range(len(piv.index)))
    axes[0].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if v == v:
                axes[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white")
    axes[0].set_title("Среднее угловых за матч по лиге и сезону")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    lg = pd.DataFrame(out["coverage"]["by_league"]).T.sort_values("mean")
    axes[1].barh(lg.index, lg["disp"], color=K.PALETTE[0])
    axes[1].axvline(1, color=K.NEG, lw=1, ls="--")
    axes[1].set_title("Дисперсия / среднее (безусловно), 2021–2026")
    axes[1].set_xlabel("Var / mean (1 = Пуассон)")
    K.save_chart(fig, F / "coverage.png")


def feature_signal():
    t = pd.read_csv(D.RESULTS / "feature_signal.csv").dropna(subset=["corr"]).head(25).iloc[::-1]
    colors = {"prod": K.PALETTE[5], "new": K.PALETTE[0], "market": K.PALETTE[1]}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(t["feature"], t["corr"], color=[colors[g] for g in t["group"]])
    ax.set_xlabel("корреляция с суммой угловых (2021/22–2024/25)")
    ax.set_title("25 признаков с наибольшей |корреляцией|")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label={"prod": "продакшн", "new": "новые", "market": "рынок"}[g])
                       for g, c in colors.items()], loc="lower right")
    K.save_chart(fig, F / "feature_signal.png")


def mean_models(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, title in ((axes[0], "select_mean", "2025/26 (выбор)"), (axes[1], "eval_mean", "2026/27 (проверка)")):
        d = {k: v for k, v in out[key].items() if "dev_diff" in v}
        names = [E.LABELS.get(k, k) for k in d]
        vals = [v["dev_diff"]["diff"] for v in d.values()]
        lo = [v["dev_diff"]["ci95"][0] for v in d.values()]
        hi = [v["dev_diff"]["ci95"][1] for v in d.values()]
        _ci_bar(ax, names, vals, lo, hi, K.PALETTE[2])
        ax.set_title(f"Δ девиации к продакшну, {title}", fontsize=10)
        ax.set_xlabel("меньше — лучше")
    K.save_chart(fig, F / "mean_models.png")


def distributions(out):
    bm = out["best_combo"]["model"]
    keys = [k for k in out["eval_dist"] if k.startswith(f"{bm}|") or k.startswith("xgb_prod|")
            or k.startswith("xgb_ha|conv")]
    keys = list(dict.fromkeys(keys))
    names = [f"{E.LABELS.get(k.split('|')[0], k)}: {E.DLABELS[k.split('|')[1]]}" for k in keys]
    sel = [out["select_dist"][k]["log_score_cf"] - out["select_dist"]["xgb_prod|nb_prod"]["log_score_cf"] for k in keys]
    ev = [out["eval_dist"][k]["log_score_diff"]["diff"] for k in keys]
    lo = [out["eval_dist"][k]["log_score_diff"]["ci95"][0] for k in keys]
    hi = [out["eval_dist"][k]["log_score_diff"]["ci95"][1] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * len(keys) + 2), sharey=True)
    axes[0].barh(np.arange(len(keys)), sel, color=K.PALETTE[1])
    axes[0].set_yticks(np.arange(len(keys)))
    axes[0].set_yticklabels(names, fontsize=7.5)
    axes[0].axvline(0, color=K.MUTED, lw=0.8)
    axes[0].set_title("2025/26, перекрёстная подгонка")
    _ci_bar(axes[1], names, ev, lo, hi, K.PALETTE[0])
    axes[1].set_yticklabels(names, fontsize=7.5)
    axes[1].set_title("2026/27, парный бутстреп [95 % ДИ]")
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.set_xlabel("Δ log score к продакшну (меньше — лучше)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    K.save_chart(fig, F / "distributions.png")


def dispersion(out):
    b = pd.DataFrame(out["dispersion_bins"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(b["mu"], b["var_y"], "o-", color=K.PALETTE[0], label="дисперсия факта")
    ax.plot(b["mu"], b["mean_y"], "s--", color=K.PALETTE[1], label="среднее факта (= Пуассон)")
    r = out["r_prod"]
    ax.plot(b["mu"], b["mu"] + b["mu"] ** 2 / r, ":", color=K.NEG, label=f"NB, r = {r:.0f}")
    ax.set_xlabel("прогноз μ (децили), 2025/26")
    ax.set_ylabel("угловые")
    ax.legend(fontsize=8)
    ax.set_title("Условная дисперсия против среднего")
    ax = axes[1]
    x = np.arange(10) / 10 + 0.05
    for i, (k, lab) in enumerate((("prod", "продакшн"), ("best", "лучший"))):
        h = np.array(out["pit"][k], float)
        ax.bar(x + (i - 0.5) * 0.04, h / h.sum(), width=0.04, color=K.PALETTE[i * 2], label=lab)
    ax.axhline(0.1, color=K.MUTED, ls="--")
    ax.set_title("PIT 2026/27 (равномерно = верное распределение)")
    ax.legend(fontsize=8)
    K.save_chart(fig, F / "dispersion.png")


def reliability(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, line in zip(axes, D.LINES):
        for i, (k, lab) in enumerate((("prod", "продакшн"), ("best", "лучший"))):
            r = pd.DataFrame(out["reliability"][k][str(line)])
            ax.plot(r["p"], r["freq"], "o-", color=K.PALETTE[i * 2], label=lab, ms=4)
        ax.plot([0.2, 0.8], [0.2, 0.8], color=K.MUTED, ls="--", lw=0.8)
        ax.set_title(f"Больше {line}")
        ax.set_xlabel("прогноз P(over)")
        ax.set_ylabel("частота")
        ax.legend(fontsize=8)
    K.save_chart(fig, F / "reliability.png")


def monthly(out):
    m = pd.DataFrame(out["eval_monthly"]).T
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(m.index, m["log_score_diff"], color=[K.PALETTE[0] if v < 0 else K.NEG for v in m["log_score_diff"]])
    for i, (k, r) in enumerate(m.iterrows()):
        ax.text(i, r["log_score_diff"], f"n={int(r['n'])}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color=K.MUTED)
    ax.set_title("Δ log score лучшего варианта к продакшну по месяцам 2026/27 (меньше — лучше)")
    K.save_chart(fig, F / "monthly.png")


def betting(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, u in zip(axes, ("close", "t09")):
        b = out["betting"][u]
        names, vals, lo, hi = [], [], [], []
        for k, v in b.items():
            for sk in ("ev0", "best0"):
                if v[sk]["roi"] is None:
                    continue
                names.append(f"{k} · {sk} (n={v[sk]['bets']})")
                vals.append(v[sk]["roi"] * 100)
                lo.append(v[sk]["ci95"][0] * 100)
                hi.append(v[sk]["ci95"][1] * 100)
        _ci_bar(ax, names, vals, lo, hi, K.PALETTE[3])
        ax.set_title(f"ROI ставок на угловые, {u}")
        ax.set_xlabel("ROI, % [95 % ДИ]")
        ax.tick_params(axis="y", labelsize=7.5)
    K.save_chart(fig, F / "betting.png")


def main_effect(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, u in zip(axes, ("close", "t09")):
        d = {k: v for k, v in out["main"][u].items() if k != "prod"}
        names = list(d)
        vals = [v["vs_b"]["roi_diff"]["diff"] * 100 for v in d.values()]
        lo = [v["vs_b"]["roi_diff"]["ci95"][0] * 100 for v in d.values()]
        hi = [v["vs_b"]["roi_diff"]["ci95"][1] * 100 for v in d.values()]
        _ci_bar(ax, names, vals, lo, hi, K.PALETTE[2])
        ax.set_title(f"Main: ΔROI к Baseline B, {u}")
        ax.set_xlabel("п.п. [95 % ДИ, парный бутстреп]")
    K.save_chart(fig, F / "main_effect.png")


def ratings(out):
    g = pd.DataFrame(out["frame_info"]["rating"]["grid"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (keep, s) in enumerate(g.groupby("keep")):
        ax.plot(s["eta"], s["side_dev_2324_2425"], "o-", color=K.PALETTE[i], label=f"keep = {keep}")
    ax.set_xlabel("шаг обновления η")
    ax.set_ylabel("девиация (угловые стороны)")
    ax.set_title("Подбор динамических рейтингов на 2023/24–2024/25")
    ax.legend(fontsize=8)
    K.save_chart(fig, F / "ratings_grid.png")


def make(out):
    K.plot_style()
    for f in (lambda: coverage(out), feature_signal, lambda: mean_models(out), lambda: distributions(out),
              lambda: dispersion(out), lambda: reliability(out), lambda: monthly(out), lambda: betting(out),
              lambda: main_effect(out), lambda: ratings(out)):
        f()
