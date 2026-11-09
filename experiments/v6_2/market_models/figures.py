"""Charts of the market-models study (from results/*.json only)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import save_chart

from . import common as C

COLORS = ["#167c66", "#c88a12", "#3d6fb6", "#c0392b"]       # validated subset of the series palette
MARKERS = ["o", "s", "^", "D"]
MUTED = "#4b5a63"
GROUP_RU = {"1X2": "1X2", "DOUBLE_CHANCE": "Двойной шанс", "TOTAL": "Тотал 1.5/2.5/3.5", "BTTS": "Обе забьют",
            "HANDICAP": "Азиатские форы", "EXACT_SCORE": "Точный счёт", "CORNERS": "Угловые 8.5/9.5/10.5"}


def fig_forest(plt) -> None:
    comp = C.load_json("comparisons.json")["comparisons"]
    names = C.load_json("market_metrics.json")["names"]
    items = list(comp.items())
    fig, ax = plt.subplots(figsize=(8.4, 0.3 * len(items) + 1.3))
    y = np.arange(len(items))
    d = np.array([c["select"]["group"]["diff"] for _, c in items]) * 1e3
    lo = np.array([c["select"]["group"]["ci95"][0] for _, c in items]) * 1e3
    hi = np.array([c["select"]["group"]["ci95"][1] for _, c in items]) * 1e3
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color=COLORS[0], ms=4.5, capsize=2, lw=1.1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{GROUP_RU[k.split('|')[0]]}: {names[k.split('|')[1]]}" for k, _ in items], fontsize=7.3)
    ax.invert_yaxis()
    ax.set_xlabel("Δ log loss к текущему (×10⁻³), 2023/24+2024/25; < 0 — лучше")
    save_chart(fig, C.FIGURES / "forest.png")


def fig_seasons(plt) -> None:
    comp = C.load_json("comparisons.json")["comparisons"]
    best = C.load_json("leagues.json")["best_by_rule"]
    names = C.load_json("market_metrics.json")["names"]
    seasons = [str(s) for s in C.SEASONS]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    groups = list(best)
    w = 0.8 / len(seasons)
    x = np.arange(len(groups))
    pal = COLORS + ["#5f7a85"]
    for i, s in enumerate(seasons):
        d = np.array([comp[f"{g}|{best[g]}"][s]["group"]["diff"] for g in groups]) * 1e3
        lo = np.array([comp[f"{g}|{best[g]}"][s]["group"]["ci95"][0] for g in groups]) * 1e3
        hi = np.array([comp[f"{g}|{best[g]}"][s]["group"]["ci95"][1] for g in groups]) * 1e3
        ax.errorbar(x - 0.4 + w * (i + 0.5), d, yerr=[d - lo, hi - d], fmt=MARKERS[i % 4], color=pal[i], capsize=2,
                    ms=4.5, lw=1, label=C.period_label(int(s)))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GROUP_RU[g]}\n{names[best[g]]}" for g in groups], fontsize=6.8)
    ax.set_ylabel("Δ log loss ×10⁻³")
    ax.legend(fontsize=7, ncol=4)
    save_chart(fig, C.FIGURES / "seasons.png")


def fig_calibration(plt) -> None:
    cal = C.load_json("calibration.json")
    names = C.load_json("market_metrics.json")["names"]
    panels = [("TOTAL_2.5", "Тотал больше 2.5", ["CUR", "BIN", "TOT", "S_TOT"]),
              ("BTTS", "Обе забьют", ["CUR", "BIN", "S_BIN"]),
              ("AH_-0.5", "Фора хозяев −0.5 (= П1)", ["CUR", "GD", "S_GD"]),
              ("CORNERS_9.5", "Угловые больше 9.5", ["CUR", "BIN", "S_BIN"])]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, (k, title, vs) in zip(axes.ravel(), panels):
        allp = []
        for i, v in enumerate(vs):
            pts = cal[k][v]["select"]["curve"]
            p, o = [x["p"] for x in pts], [x["o"] for x in pts]
            allp += p + o
            ax.plot(p, o, marker=MARKERS[i], color=COLORS[i], lw=1.3, ms=4, label=names[v])
        lo, hi = min(allp) * 0.95, max(allp) * 1.03
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("прогноз")
        ax.set_ylabel("факт")
        ax.legend(fontsize=6.8, loc="upper left")
    save_chart(fig, C.FIGURES / "calibration.png")


def fig_main_risk(plt) -> None:
    mr = C.load_json("main_risk.json")["variants"]
    subs = [k for k in mr if k != "CUR"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.34 * len(subs) + 1.6), sharey=True)
    for ax, role, title in ((axes[0], "main_vs_cur", "Main"), (axes[1], "risk_vs_cur", "Risk")):
        y = np.arange(len(subs))
        d = np.array([mr[k]["select"][role]["diff"] for k in subs]) * 100
        lo = np.array([mr[k]["select"][role]["ci95"][0] for k in subs]) * 100
        hi = np.array([mr[k]["select"][role]["ci95"][1] for k in subs]) * 100
        ok = np.isfinite(d)
        ax.errorbar(d[ok], y[ok], xerr=[(d - lo)[ok], (hi - d)[ok]], fmt="o", color=COLORS[2], ms=4, capsize=2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{title}: ΔROI, п.п. (2023/24+2024/25)", fontsize=9)
    axes[0].set_yticks(np.arange(len(subs)))
    axes[0].set_yticklabels([GROUP_RU.get(k.replace("SUB_", ""), "все улучшенные") for k in subs], fontsize=7.5)
    axes[0].invert_yaxis()
    save_chart(fig, C.FIGURES / "main_risk.png")


def build() -> None:
    plt = C.plt()
    fig_forest(plt)
    fig_seasons(plt)
    fig_calibration(plt)
    fig_main_risk(plt)
