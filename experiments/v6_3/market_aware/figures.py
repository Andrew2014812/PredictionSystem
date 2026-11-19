"""Figures of the market-aware Main study (from results/market_aware.json and results/cumulative_profit.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from scripts.report_kit import plt, save_chart

from .run import DIS_LABELS, EDGE_LABELS, FIG, RES

# validated categorical order (three slots, light surface): green, amber, blue; gray = reference
COL = ["#0f8a6c", "#b5760a", "#3b69b0"]
GRAY = "#6b7780"
U_RU = {"close": "close (перед началом)", "t09": "t09 (момент T)"}
GROUP_RU = {"1X2": "1X2", "TOTAL": "Тоталы", "CORNERS": "Угловые", "BTTS": "Обе забьют", "HANDICAP": "Аз. форы"}


def err(ax, x, mean, ci, color, label, marker="o"):
    lo = [m - c[0] for m, c in zip(mean, ci)]
    hi = [c[1] - m for m, c in zip(mean, ci)]
    ax.errorbar(x, mean, yerr=[lo, hi], fmt=marker, color=color, ms=6, capsize=3, lw=1.6, label=label)


def fig_logloss(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        wf = r["information"][u]["walk_forward"]
        groups = list(wf)
        x = np.arange(len(groups))
        for j, (key, lab) in enumerate((("model_minus_market", "модель − рынок"),
                                        ("blend_minus_market", "смесь − рынок"))):
            mean = [wf[g][key]["diff"] * 1000 for g in groups]
            ci = [[v * 1000 for v in wf[g][key]["ci95"]] for g in groups]
            err(ax, x + (j - 0.5) * 0.22, mean, ci, COL[j], lab, "os"[j])
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(x, [GROUP_RU[g] for g in groups])
        ax.set_title(U_RU[u])
    axes[0].set_ylabel("Δ log loss, ×10⁻³ (меньше — лучше)")
    axes[0].legend(frameon=False, loc="upper right")
    return save_chart(fig, FIG / "logloss_vs_market.png")


def fig_coefs(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        fs = r["information"][u]["full_season"]
        groups = list(fs)
        x = np.arange(len(groups))
        err(ax, x - 0.11, [fs[g]["b_market"] for g in groups], [fs[g]["b_market_ci"] for g in groups], COL[2],
            "вес рынка b_q", "s")
        err(ax, x + 0.11, [fs[g]["b_model"] for g in groups], [fs[g]["b_model_ci"] for g in groups], COL[0],
            "вес модели b_p", "o")
        ax.axhline(0, color=GRAY, lw=1)
        ax.axhline(1, color=GRAY, lw=0.8, ls=":")
        ax.set_xticks(x, [GROUP_RU[g] for g in groups])
        ax.set_title(U_RU[u])
    axes[0].set_ylabel("коэффициент пула (95 % ДИ)")
    axes[0].legend(frameon=False, loc="upper left")
    return save_chart(fig, FIG / "pool_coefficients.png")


def fig_edge(r):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(EDGE_LABELS))
    for j, u in enumerate(("close", "t09")):
        b = r["edge"][u]["bins"]
        mean = [b[k]["roi"] * 100 for k in EDGE_LABELS]
        ci = [[v * 100 for v in b[k]["ci95"]] for k in EDGE_LABELS]
        err(ax, x + (j - 0.5) * 0.2, mean, ci, COL[j], U_RU[u], "os"[j])
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xticks(x, EDGE_LABELS)
    ax.set_xlabel("перевес модели над рынком e = p / q − 1")
    ax.set_ylabel("ROI, % (95 % ДИ по матчам)")
    ax.legend(frameon=False)
    return save_chart(fig, FIG / "roi_by_edge.png")


def fig_variants(r):
    keys = [k for k in r["variants_ru"] if k != "baseline_b"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(keys))
    for j, u in enumerate(("close", "t09")):
        v = r["variants"][u]
        mean = [v[k]["vs_b"]["roi_diff"]["diff"] * 100 for k in keys]
        ci = [[c * 100 for c in v[k]["vs_b"]["roi_diff"]["ci95"]] for k in keys]
        lo = [m - c[0] for m, c in zip(mean, ci)]
        hi = [c[1] - m for m, c in zip(mean, ci)]
        ax.errorbar(mean, y + (j - 0.5) * 0.28, xerr=[lo, hi], fmt="os"[j], color=COL[j], ms=6, capsize=3,
                    label=U_RU[u])
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(y, [r["variants_ru"][k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("ROI(вариант) − ROI(Baseline B), п.п. (парный бутстреп, 95 % ДИ)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return save_chart(fig, FIG / "variants_vs_baseline_b.png")


def fig_cumulative():
    d = pd.read_csv(RES / "cumulative_profit.csv", parse_dates=["date"])
    names = {"baseline_b": "Baseline B", "blend_wf": "V1 смесь (walk-forward)", "market_only": "V3 только рынок"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        for j, (k, lab) in enumerate(names.items()):
            s = d.loc[(d["universe"] == u) & (d["method"] == k)]
            ax.plot(s["date"], s["cum_profit"], color=COL[j], lw=2, label=lab)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_title(U_RU[u])
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("накопленная прибыль, ставок по 1 ед.")
    axes[0].legend(frameon=False, loc="lower left")
    return save_chart(fig, FIG / "cumulative_profit.png")


def fig_disagreement(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        b = r["disagreement"][u]["bins"]
        labs = [k for k in DIS_LABELS if k in b and b[k]["n"] >= 30]
        x = np.arange(len(labs))
        ax.plot(x, [b[k]["mean_p"] for k in labs], "o-", color=COL[0], lw=2, label="средняя p модели")
        ax.plot(x, [b[k]["mean_q"] for k in labs], "s-", color=COL[2], lw=2, label="средняя q рынка")
        ax.plot(x, [b[k]["hit"] for k in labs], "D--", color=COL[1], lw=2, label="доля выигрышей")
        ax.set_xticks(x, [f"{k}\n(n={b[k]['n']})" for k in labs])
        ax.set_title(U_RU[u])
        ax.set_xlabel("logit p − logit q выбранного Main")
    axes[0].set_ylabel("вероятность / частота")
    axes[0].legend(frameon=False, loc="lower left")
    return save_chart(fig, FIG / "disagreement_main.png")


def run():
    r = C.read_json(RES / "market_aware.json")
    for f in (fig_logloss, fig_coefs, fig_edge, fig_variants, fig_disagreement):
        f(r)
    fig_cumulative()


if __name__ == "__main__":
    run()
