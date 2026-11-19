"""Figures of the Goal-Model study (from results/ only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.v6_3 import common as C  # noqa: E402
from scripts.report_kit import plot_style, save_chart  # noqa: E402

DIRS = C.study_dirs("goals")
BLUE, ORANGE, GREEN, GREY = "#2a6f97", "#d17a22", "#3a8d5c", "#8a948f"


def forest(ax, labels, d, lo, hi, title, scale=1.0, unit=""):
    y = np.arange(len(labels))[::-1]
    d, lo, hi = (np.asarray(x, float) * scale for x in (d, lo, hi))
    col = [GREEN if h < 0 else (ORANGE if l > 0 else BLUE) for l, h in zip(lo, hi)]
    for yi, l0, h0, c0 in zip(y, lo, hi, col):
        ax.plot([l0, h0], [yi, yi], color=c0, lw=1.4)
    ax.scatter(d, y, color=col, zorder=3, s=18)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(unit)


def make_all(log=None) -> None:
    plot_style()
    r = C.read_json(DIRS["results"] / "results.json")
    names = r["names"]
    ev = r["paired_2026"]
    keys = sorted(ev, key=lambda k: ev[k]["score_ll"]["diff"])
    labels = [names.get(k, k) for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
    for ax, m, t in zip(axes, ("score_ll", "totals_ll", "btts_ll"),
                        ("Точный счёт: Δ log loss", "Тоталы 0.5–4.5: Δ log loss", "Обе забьют: Δ log loss")):
        forest(ax, labels, [ev[k][m]["diff"] for k in keys], [ev[k][m]["ci95"][0] for k in keys],
               [ev[k][m]["ci95"][1] for k in keys], t, 1000, "×10⁻³ (меньше — лучше)")
    save_chart(fig, DIRS["figures"] / "forest_2026.png")

    rr = pd.read_csv(DIRS["results"] / "main_roi.csv")
    rr = rr.loc[rr["variant"] != "prod"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    order = [k for k in keys if k in set(rr["variant"])]
    for ax, u in zip(axes, ("close", "t09")):
        s = rr.loc[rr["universe"] == u].set_index("variant").reindex(order)
        y = np.arange(len(order))[::-1]
        ax.errorbar(s["d_roi"] * 100, y, xerr=[(s["d_roi"] - s["d_lo"]) * 100, (s["d_hi"] - s["d_roi"]) * 100],
                    fmt="o", color=BLUE, ms=4, capsize=2)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([names.get(k, k) for k in order])
        ax.set_title(f"ΔROI Main к базовой линии B ({u}), п.п.", fontsize=9)
    save_chart(fig, DIRS["figures"] / "roi_2026.png")

    cal = r["calibration"]
    show = ["prod"] + [k for k in dict.fromkeys([r["best_prob"], "combo"]) if k in cal and k != "prod"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, mk, t in zip(axes, ("ou25", "btts"), ("Тотал больше 2.5", "Обе забьют")):
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
        for k, c in zip(show, (BLUE, ORANGE, GREEN)):
            tb = pd.DataFrame(cal[k][mk])
            tb = tb.loc[tb["n"] >= 20]
            ax.plot(tb["p"], tb["freq"], marker="o", ms=3, color=c, label=names.get(k, k))
        ax.set_xlabel("прогноз")
        ax.set_ylabel("частота")
        ax.set_title(t, fontsize=9)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0.1, 0.9)
        ax.legend(fontsize=7)
    save_chart(fig, DIRS["figures"] / "calibration_2026.png")

    sel = C.read_json(DIRS["results"] / "ratings_selection.json")
    g = pd.DataFrame(sel["grid"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for rr_, c in zip(sorted(g.loc[g["eta_ha"] == 0, "r"].unique()), (BLUE, ORANGE, GREEN)):
        x = g.loc[(g["eta_ha"] == 0) & (g["r"] == rr_)].sort_values("eta")
        ax.plot(x["eta"], x["deviance"], marker="o", color=c, label=f"r = {rr_}")
    ax.set_xlabel("η (скорость обновления рейтинга)")
    ax.set_ylabel("девиация Пуассона, 2023–2025")
    ax.legend(fontsize=8)
    save_chart(fig, DIRS["figures"] / "ratings_grid.png")

    bs = pd.read_csv(DIRS["results"] / "selection_by_season.csv")
    vs = [v for v in ["raw", "dc", "nb", "ipf", "sup", "partial", "rat", "blend", "ratfeat", "stacked", "monthly"]
          if v in set(bs["variant"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / 3
    for i, (s, c) in enumerate(zip((2023, 2024, 2025), (BLUE, ORANGE, GREEN))):
        x = bs.loc[bs["season"] == s].set_index("variant").reindex(vs)
        ax.bar(np.arange(len(vs)) + (i - 1) * w, x["d_score_ll"] * 1000, width=w, color=c, label=f"{s}/{s + 1 - 2000}")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(np.arange(len(vs)))
    ax.set_xticklabels([names.get(v, v) for v in vs], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Δ log loss точного счёта, ×10⁻³")
    ax.legend(fontsize=8)
    save_chart(fig, DIRS["figures"] / "selection_seasons.png")

    bm = pd.read_csv(DIRS["results"] / "by_month_2026.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, c in zip([r["best_prob"], "blend", "ipf", "monthly"], (BLUE, ORANGE, GREEN, GREY)):
        x = bm.loc[bm["variant"] == k]
        if len(x):
            ax.plot(x["month"], x["d_score_ll"] * 1000, marker="o", color=c, label=names.get(k, k))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Δ log loss точного счёта, ×10⁻³")
    ax.legend(fontsize=7)
    save_chart(fig, DIRS["figures"] / "by_month_2026.png")
    if log:
        log.info("figures written")
