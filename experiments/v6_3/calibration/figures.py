"""Figures of the calibration study (from results/calibration.json)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import report_kit as K
from scripts.report_kit import plt

from experiments.v6_3 import common as C

from . import data as D
from . import maps as MP

FIG = D.DIRS["figures"]
UNIV_RU = {"close": "close (цена перед стартом)", "t09": "t09 (цена в момент решения)"}


def vlabel(name: str) -> str:
    if name == "baseline_b":
        return "Baseline B"
    m, s, src = name.split("|")
    src_ru = {"pooled": "2025/26 + месяцы", "backtest": "2025/26", "expanding": "месяцы 2026/27", "-": ""}[src]
    return f"{MP.METHOD_RU.get(m, m)} / {MP.SCOPE_RU.get(s, s)} / {src_ru}".strip(" /")


def vshort(name: str) -> str:
    """Compact label for narrow table columns."""
    if name == "baseline_b":
        return "Baseline B"
    m, s, src = name.split("|")
    ms = {"raw": "исх.", "platt": "Platt", "temperature": "темп.", "beta": "бета", "isotonic": "изотон.",
          "histbin": "гистогр."}[m]
    ss = {"global": "общая", "group": "группа", "key": "исход", "-": ""}[s]
    sr = {"pooled": "25/26+мес.", "backtest": "25/26", "expanding": "мес. 26/27", "-": ""}[src]
    return f"{ms}, {ss}, {sr}"


def reliability(r: dict, primary: str) -> None:
    groups = D.GROUPS_ALL
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True)
    for ax, g in zip(axes.ravel(), groups):
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
        for key, lab, col, ls in (("backtest_raw", "2025/26 без калибровки", K.PALETTE[5], "--"),
                                  ("raw", "2026/27 без калибровки", K.PALETTE[4], "-"),
                                  ("primary", "2026/27 основная карта", K.PALETTE[0], "-")):
            t = pd.DataFrame(r[g][key])
            t = t.loc[t["n"] >= 30]
            ax.plot(t["p"], t["y"], ls, marker="o", ms=3, color=col, label=lab, lw=1.2)
        ax.set_title(D.GROUP_RU[g], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("заявленная вероятность")
    for ax in axes[:, 0]:
        ax.set_ylabel("частота выигрыша")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    K.save_chart(fig, FIG / "reliability_groups.png")


def quality_heatmap(q: dict) -> None:
    rows = [f"{m}|{s}" for m in MP.METHODS[1:] for s in MP.SCOPES]
    raw = q["raw|-|-"]["band"]["log_loss"]
    mat = np.array([[(q[f"{r}|{src}"]["band"]["log_loss"] - raw) * 1000 for src in ("backtest", "expanding", "pooled")]
                    for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lim = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(3), ["2025/26\n(заморожена)", "месяцы 2026/27\n(расширяющееся окно)", "2025/26 + месяцы\n2026/27"])
    ax.set_yticks(range(len(rows)), [f"{MP.METHOD_RU[r.split('|')[0]]} / {MP.SCOPE_RU[r.split('|')[1]]}" for r in rows],
                  fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Δ log loss × 1000 к исходной вероятности")
    K.save_chart(fig, FIG / "quality_heatmap.png")


def ece_groups(q: dict, primary: str) -> None:
    m, s, _ = primary.split("|")
    names = ["raw|-|-", f"{m}|{s}|backtest", f"{m}|{s}|expanding", primary]
    labels = ["без калибровки", "основная, 2025/26", "основная, мес. 2026/27", "основная, 2025/26 + мес."]
    groups = D.MAIN_GROUPS + ["band"]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (n, lab) in enumerate(zip(names, labels)):
        ax.bar(x + (i - 1.5) * 0.2, [q[n][g]["ece"] * 100 for g in groups], 0.2, label=lab)
    ax.set_xticks(x, [D.GROUP_RU.get(g, "кандидаты Main") for g in groups], fontsize=8)
    ax.set_ylabel("ECE, п.п.")
    ax.legend(fontsize=7.5, ncol=2)
    K.save_chart(fig, FIG / "ece_groups.png")


def winners_curse(w: dict) -> None:
    t = pd.DataFrame(w["table"])
    t["kx"] = t["k"].replace(99, 26)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    raw = t.loc[t["rule_on"] == "raw"].sort_values("kx")
    cal = t.loc[t["rule_on"] == "calibrated"].sort_values("kx")
    ax = axes[0]
    ax.errorbar(raw["kx"], raw["gap_stated"] * 100, yerr=raw["gap_stated_sd"].fillna(0) * 100, marker="o", ms=4,
                label="правило на исходных p: заявлено − факт", capsize=2)
    ax.errorbar(raw["kx"], raw["gap_cal"] * 100, yerr=raw["gap_cal_sd"].fillna(0) * 100, marker="s", ms=4,
                label="тот же выбор, откалиброванная p − факт", capsize=2)
    ax.errorbar(cal["kx"], cal["gap_stated"] * 100, yerr=cal["gap_stated_sd"].fillna(0) * 100, marker="^", ms=4,
                label="правило на откалиброванных p", capsize=2)
    ax.axhline(0, color="#777", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 26], ["1", "2", "4", "8", "16", "все"])
    ax.set_xlabel("число случайно оставленных кандидатов k")
    ax.set_ylabel("переоценка Main, п.п.")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(raw["kx"], raw["roi"] * 100, marker="o", ms=4, label="правило на исходных p")
    ax.plot(cal["kx"], cal["roi"] * 100, marker="^", ms=4, label="правило на откалиброванных p")
    ax2 = ax.twinx()
    ax2.plot(raw["kx"], raw["avg_odds"], color=K.PALETTE[5], ls="--", lw=1, label="средний коэф. (исх.)")
    ax2.set_ylabel("средний коэффициент")
    ax2.grid(False)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 26], ["1", "2", "4", "8", "16", "все"])
    ax.set_xlabel("число случайно оставленных кандидатов k")
    ax.set_ylabel("ROI, %")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    K.save_chart(fig, FIG / "winners_curse.png")


def ncand_observed(o: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in ((axes[0], "close_ncand_priced", "2026/27, Baseline B (close)"),
                           (axes[1], "backtest_ncand_priced", "2025/26, правило продакшна (backtest)")):
        t = pd.DataFrame(o[key])
        lo = [g - c[0] for g, c in zip(t["gap"], t["gap_ci"])]
        hi = [c[1] - g for g, c in zip(t["gap"], t["gap_ci"])]
        ax.errorbar(range(len(t)), t["gap"] * 100, yerr=np.array([lo, hi]) * 100, marker="o", capsize=3)
        ax.set_xticks(range(len(t)), [f"{b}\nn={n}" for b, n in zip(t["bin"], t["n"])], fontsize=7)
        ax.axhline(0, color="#777", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("число кандидатов Main с реальной ценой")
        ax.set_ylabel("заявлено − факт, п.п.")
    K.save_chart(fig, FIG / "ncand_observed.png")


def roi_forest(res: dict, family: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        for i, n in enumerate(family):
            v = res["main"][u][n]["vs_b"]["roi_diff"]
            ax.errorbar(v["diff"] * 100, i, xerr=[[(v["diff"] - v["ci95"][0]) * 100], [(v["ci95"][1] - v["diff"]) * 100]],
                        marker="o", ms=4, capsize=2, color=K.PALETTE[0] if n == res["primary"] else K.PALETTE[2])
        ax.axvline(0, color="#777", lw=0.8)
        ax.set_title(UNIV_RU[u], fontsize=9)
        ax.set_xlabel("ΔROI к Baseline B, п.п. (95 % ДИ)")
    axes[0].set_yticks(range(len(family)), [vlabel(n) for n in family], fontsize=7.5)
    K.save_chart(fig, FIG / "roi_forest.png")


def monthly(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    p = res["primary"]
    for ax, u in zip(axes, ("close", "t09")):
        for name, lab in (("baseline_b", "Baseline B"), (p, "основная карта")):
            t = pd.DataFrame(res["robustness"][u][name]["by_month"])
            ax.plot(t["month"], t["roi"] * 100, marker="o", label=lab)
        ax.axhline(0, color="#777", lw=0.8)
        ax.set_title(UNIV_RU[u], fontsize=9)
        ax.set_ylabel("ROI, %")
        ax.legend(fontsize=7.5)
    K.save_chart(fig, FIG / "monthly.png")


def main() -> None:
    res = C.read_json(D.DIRS["results"] / "calibration.json")
    reliability(res["reliability"], res["primary"])
    quality_heatmap(res["quality"])
    ece_groups(res["quality"], res["primary"])
    winners_curse(res["winners_curse"])
    ncand_observed(res["ncand_observed"])
    roi_forest(res, res["family"])
    monthly(res)


if __name__ == "__main__":
    main()
