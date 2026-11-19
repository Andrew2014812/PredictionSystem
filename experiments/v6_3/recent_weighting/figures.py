"""Figures of the Recent-Season-Weighting study (from results/ only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.v6_3 import common as C  # noqa: E402
from scripts.report_kit import plot_style, save_chart  # noqa: E402

DIRS = C.study_dirs("recent_weighting")
BLUE, ORANGE, GREEN, RED, GREY = "#2a6f97", "#d17a22", "#3a8d5c", "#b23a48", "#8a948f"
GROUP_TITLE = {"result": "1X2: Δ log loss", "goals": "Голы: Δ девиации Пуассона", "corners": "Угловые: Δ девиации"}
ORDER = ["hl0", "hl0.5", "hl1", "hl2", "hl4", "last2", "last3", "cur2", "cur4", "win2", "win3", "k250", "form",
         "selected", "prod"]


def make_all(log=None) -> None:
    plot_style()
    r = C.read_json(DIRS["results"] / "results.json")
    names, paired = r["names"], r["paired"]

    # 1. forest: 2026/27 vs Baseline B per task group (monthly schemes + static control)
    for season, ref, fname in ((2026, "vs_B", "forest_2026_vs_B.png"), (2025, "vs_control", "forest_2025.png"),
                               (2026, "vs_control", "forest_2026_vs_control.png")):
        keys = [s for s in ORDER if f"{season}|wf|{s}|{ref}" in paired]
        if ref == "vs_B" and f"{season}|static|prod|vs_B" in paired:
            keys = keys + ["static:prod"]
        if not keys:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True)
        for ax, g in zip(axes, ("result", "goals", "corners")):
            vals = []
            for k in keys:
                kk = f"{season}|static|prod|{ref}" if k == "static:prod" else f"{season}|wf|{k}|{ref}"
                v = paired[kk][g]
                vals.append((v["diff"], v["ci95"][0], v["ci95"][1]))
            v = np.array(vals, float) * 1000
            y = np.arange(len(keys))[::-1]
            col = [GREEN if h < 0 else (RED if lo > 0 else BLUE) for _, lo, h in v]
            for yi, (d0, l0, h0), c0 in zip(y, v, col):
                ax.plot([l0, h0], [yi, yi], color=c0, lw=1.4)
            ax.scatter(v[:, 0], y, color=col, s=16, zorder=3)
            ax.axvline(0, color=GREY, lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([("однократно перед сезоном, веса продакшна" if k == "static:prod" else names.get(k, k))
                                for k in keys], fontsize=7.5)
            ax.set_title(GROUP_TITLE[g], fontsize=9)
            ax.set_xlabel("×10⁻³ (меньше — лучше)")
        save_chart(fig, DIRS["figures"] / fname)

    # 2. when it helps: by refit month (cur4, cur2) and by matches played, both seasons
    w = pd.read_csv(DIRS["results"] / "when_it_helps.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for row, by in enumerate(("refit_month", "games")):
        for ax, g in zip(axes[row], ("result", "goals", "corners")):
            for (s, sc), c in zip([(2025, "cur4"), (2026, "cur4"), (2025, "cur2"), (2025, "monthly_vs_static")],
                                  (BLUE, ORANGE, GREEN, GREY)):
                x = w.loc[(w["season"] == s) & (w["scheme"] == sc) & (w["group"] == g) & (w["by"] == by)]
                if by == "games":
                    x = x.assign(o=x["value"].map({"0-2": 0, "3-5": 1, "6-9": 2, "10-14": 3, "15-19": 4, "20+": 5}))
                    x = x.sort_values("o")
                else:
                    x = x.sort_values("value")
                if len(x) == 0:
                    continue
                lab = {"monthly_vs_static": f"{s}/{s - 1999}: ежемесячно − однократно"}.get(
                    sc, f"{s}/{s - 1999}: {names.get(sc, sc).replace('ежемесячно, ', '')} − контроль")
                ax.errorbar(np.arange(len(x)), x["diff"] * 1000, yerr=1.96 * x["se"] * 1000, marker="o", ms=3,
                            color=c, capsize=2, label=lab)
                ax.set_xticks(np.arange(len(x)))
                ax.set_xticklabels([v[5:] if by == "refit_month" else v for v in x["value"]], fontsize=7,
                                   rotation=45 if by == "refit_month" else 0)
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_title(f"{GROUP_TITLE[g]} — {'месяц переобучения' if by == 'refit_month' else 'сыграно матчей'}",
                         fontsize=8.5)
            if row == 0 and g == "result":
                ax.legend(fontsize=6.5)
    save_chart(fig, DIRS["figures"] / "when_it_helps.png")

    # 3. ROI vs Baseline B (close / t09), monthly schemes
    rr = pd.read_csv(DIRS["results"] / "main_roi.csv")
    keys = [f"wf:{s}" for s in ORDER if f"wf:{s}" in set(rr["variant"])] + \
        [v for v in ("static:prod",) if v in set(rr["variant"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        s = rr.loc[rr["universe"] == u].set_index("variant").reindex(keys)
        y = np.arange(len(keys))[::-1]
        ax.errorbar(s["d_roi"] * 100, y, xerr=[(s["d_roi"] - s["d_lo"]) * 100, (s["d_hi"] - s["d_roi"]) * 100],
                    fmt="o", color=BLUE, ms=4, capsize=2)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([("однократно, веса продакшна" if k == "static:prod" else names.get(k.split(":")[1], k))
                            for k in keys], fontsize=7.5)
        ax.set_title(f"ΔROI Main к базовой линии B ({u}), п.п.", fontsize=9)
    save_chart(fig, DIRS["figures"] / "roi_2026.png")

    # 4. share of the current season in the training weight
    sh = pd.read_csv(DIRS["results"] / "current_weight_share.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for (sc, c) in zip(("prod", "cur2", "cur4"), (GREY, BLUE, ORANGE)):
        for s, ls in ((2025, "-"), (2026, "--")):
            x = sh.loc[(sh["season"] == s) & (sh["scheme"] == sc) & (sh["task"] == "home_goals")]
            ax.plot(np.arange(len(x)), x["weight_share"] * 100, ls=ls, marker="o", ms=3, color=c,
                    label=f"{s}/{s - 1999}, {names.get(sc, sc).replace('ежемесячно, ', '')}")
    ax.set_xlabel("номер ежемесячного переобучения в сезоне (0 = старт сезона)")
    ax.set_ylabel("доля веса текущего сезона, %")
    ax.legend(fontsize=7)
    save_chart(fig, DIRS["figures"] / "weight_share.png")

    # 5. cumulative profit: Baseline B vs the selected variant (close)
    cand = r["candidates"]
    if "wf:selected" in cand:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for name, c in (("wf:selected", ORANGE), ("wf:prod", BLUE)):
            p = DIRS["results"] / f"per_match_{name.replace(':', '_')}_close.csv"
            if p.exists():
                t = pd.read_csv(p).sort_values("date")
                ax.plot(np.arange(len(t)), t["profit"].fillna(0).cumsum(), color=c, label=names.get(name.split(":")[1]))
        b = pd.read_csv(DIRS["results"] / "per_match_baseline_b_close.csv").sort_values("date")
        ax.plot(np.arange(len(b)), b["profit"].fillna(0).cumsum(), color=GREY, label=names["B"])
        ax.set_xlabel("матчи 2026/27 по дате")
        ax.set_ylabel("накопленная прибыль, ед.")
        ax.legend(fontsize=7)
        save_chart(fig, DIRS["figures"] / "cumulative_profit.png")
    if log:
        log.info("figures written")
