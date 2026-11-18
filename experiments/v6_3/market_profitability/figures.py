"""Figures of the market-profitability study (read from results/, written to figures/).

    python -m experiments.v6_3.market_profitability.figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.v6_3 import common as C  # noqa: E402
from experiments.v6_3.main_selection import figures as F1  # noqa: E402
from scripts import report_kit as K  # noqa: E402

DIRS = C.study_dirs("market_profitability")
RES, FIG = DIRS["results"], DIRS["figures"]
GROUPS = ["1X2", "TOTAL", "HANDICAP", "DOUBLE_CHANCE", "BTTS", "CORNERS"]


def profile(pr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(GROUPS))
    for k, src in enumerate(("cand", "main")):
        for j, u in enumerate(("bt", "close", "t09")):
            t = pr.loc[(pr["universe"] == u) & (pr["source"] == src)].set_index("group").reindex(GROUPS)
            ok = t["bets"].fillna(0).to_numpy() >= (30 if src == "main" else 1)
            r = t["roi"].to_numpy(float) * 100
            lo, hi = t["ci_lo"].to_numpy(float) * 100, t["ci_hi"].to_numpy(float) * 100
            xx = x + (j - 1) * 0.26
            ax[k].bar(xx[ok], r[ok], 0.26, color=F1.UCOL[u], label=F1.ULAB[u])
            ax[k].errorbar(xx[ok], r[ok], yerr=[r[ok] - lo[ok], hi[ok] - r[ok]], fmt="none", ecolor=K.INK,
                           lw=0.7, capsize=1.5)
        ax[k].axhline(0, color=K.INK, lw=0.7)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(["1X2", "TOTAL", "HCP", "DC", "BTTS", "CORN"])
        ax[k].set_title("каждый кандидат с ценой (флэт)" if src == "cand" else "Main продакшна (≥ 30 ставок)")
    ax[0].set_ylabel("ROI, % (95 % ДИ)")
    ax[0].legend(fontsize=7)
    K.save_chart(fig, FIG / "profile.png")


def gaps(pr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(len(GROUPS))
    for j, (u, src) in enumerate((("close", "cand"), ("close", "main"), ("bt", "cand"), ("bt", "main"))):
        t = pr.loc[(pr["universe"] == u) & (pr["source"] == src)].set_index("group").reindex(GROUPS)
        ok = t["bets"].fillna(0).to_numpy() >= 30
        ax.bar((x + (j - 1.5) * 0.2)[ok], (t["gap"].to_numpy(float) * 100)[ok], 0.2,
               color=K.PALETTE[j], label=f"{F1.ULAB[u]}: {'кандидаты' if src == 'cand' else 'Main'}")
    ax.axhline(0, color=K.INK, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("заявлено − факт, п.п.")
    ax.set_title("Переоценка вероятности: все кандидаты против выбранного Main (≥ 30 ставок)")
    ax.legend(fontsize=7)
    K.save_chart(fig, FIG / "gaps.png")


def history_path() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, sid in enumerate(("eb_cand_group_kEB_b3", "eb_main_group_kEB_b3")):
        p = RES / f"history_path_{sid}_close.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p, parse_dates=["date"])
        for i, g in enumerate(GROUPS):
            x = t.loc[t["cell"] == g]
            if x["n"].max() < 30:
                continue
            ax[k].plot(x["date"], x["roi_shr"] * 100, color=F1.GCOL[g], label=g)
        m = t.drop_duplicates("date")
        ax[k].plot(m["date"], m["m"] * 100, color=K.INK, lw=0.8, ls="--", label="общий ROI")
        ax[k].set_title("сглаженный ROI: все кандидаты" if k == 0 else "сглаженный ROI: Main продакшна")
        ax[k].tick_params(axis="x", labelrotation=30)
    ax[0].set_ylabel("ROI на дату (только прошлые матчи), %")
    ax[0].legend(fontsize=7, ncol=2)
    K.save_chart(fig, FIG / "history_path.png")


def mix(mx: pd.DataFrame) -> None:
    rules = ["production"] + [f"priority_{g}" for g in GROUPS]
    t = mx.loc[mx["universe"] == "close"].drop_duplicates(["rule", "group"])
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    left = np.zeros(len(rules))
    for g in GROUPS:
        v = np.array([t.loc[(t["rule"] == r) & (t["group"] == g), "share"].sum() for r in rules]) * 100
        ax.barh(rules, v, left=left, color=F1.GCOL[g], label=g)
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("доля ставок Main, % (close)")
    ax.set_title("Рынки Main при принудительном приоритете одной группы")
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    K.save_chart(fig, FIG / "priority_mix.png")


def grid_bars(grid: pd.DataFrame) -> None:
    fams = ["priority", "boost", "exclude"]
    g = grid.loc[grid["family"].isin(fams) & (grid["rule"] != "production")]
    rules = list(dict.fromkeys(g["rule"]))
    fig, ax = plt.subplots(figsize=(8.6, 0.3 * len(rules) + 1.2))
    y = np.arange(len(rules))
    for off, u in ((-0.18, "close"), (0.18, "t09")):
        t = g.loc[g["universe"] == u].set_index("rule").reindex(rules)
        d = t["roi_diff"].to_numpy(float) * 100
        ax.errorbar(d, y + off, xerr=[d - t["ci_lo"] * 100, t["ci_hi"] * 100 - d], fmt="o", ms=3.5,
                    color=F1.UCOL[u], label=F1.ULAB[u], capsize=1.5)
    ax.axvline(0, color=K.INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rules)
    ax.invert_yaxis()
    ax.set_xlabel("ΔROI к Baseline B, п.п. (95 % ДИ)")
    ax.set_title("Приоритет, усиление и исключение рынков")
    ax.legend(fontsize=8, loc="lower right")
    K.save_chart(fig, FIG / "rules_forest.png")


def main() -> None:
    res = json.loads((RES / "market_profitability.json").read_text())
    pr = pd.DataFrame(res["profile"])
    profile(pr)
    gaps(pr)
    history_path()
    mix(pd.DataFrame(res["mix"]))
    grid_bars(pd.read_csv(RES / "grid.csv"))
    F1.forest(pd.DataFrame(res["candidates"]), FIG / "candidates_forest.png",
              "Заранее выбранные правила рынков против Baseline B")


if __name__ == "__main__":
    main()
