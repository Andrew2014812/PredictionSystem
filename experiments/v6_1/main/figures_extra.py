"""Overview figures of the strategies (from results/strategies.json and the cached picks of the folds)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C


def run() -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    st = C.load_json("strategies.json")
    folds = st["modes"]["bettable"]["folds"]
    names = [n for n in folds[str(C.FOLDS[0])] if folds[str(C.SELECT[0])][n]["bets"] >= 300]
    # 1. ROI with CI per strategy and season
    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(names))
    for k, s in enumerate(C.FOLDS):
        r = folds[str(s)]
        roi = [r[n]["roi"] * 100 if r[n]["roi"] is not None else np.nan for n in names]
        lo = [(r[n]["roi"] - r[n]["roi_ci95"][0]) * 100 if r[n]["roi_ci95"][0] is not None else 0 for n in names]
        hi = [(r[n]["roi_ci95"][1] - r[n]["roi"]) * 100 if r[n]["roi_ci95"][1] is not None else 0 for n in names]
        ax.errorbar(roi, y + (k - 1) * 0.25, xerr=[lo, hi], fmt="o", ms=3, capsize=1.5, lw=0.8, color=PALETTE[k],
                    label=C.period_label(s))
    ax.axvline(0, color="#555", lw=0.8)
    ax.axvline(folds[str(C.SELECT[1])]["A"]["roi"] * 100, color=PALETTE[1], lw=0.6, ls=":")
    ax.axvline(folds[str(C.SELECT[0])]["A"]["roi"] * 100, color=PALETTE[0], lw=0.6, ls=":")
    ax.set_yticks(y, names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("ROI, % (95 % бутстреп-ДИ)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Стратегии A–G на одних матчах (только кандидаты с ценой Bet365); пунктир — ROI A", fontsize=9.5)
    save_chart(fig, C.FIGURES / "strat_overview.png")
    # 2. reliability of the shown Main and cumulative profit (cached picks)
    path = C.CACHE / "strategy_picks.parquet"
    if not path.exists():
        return
    pk = pd.read_parquet(path)
    pk = pk.loc[pk["mode"] == "bettable"]
    show = ["A", "B3 selection-conditional", "E2 candidate-count penalty", "F1 blend Shin (inner weight)", "C3 max EV"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, s in zip(axes, C.FOLDS):
        for k, n in enumerate(show):
            d = pk.loc[(pk["season"] == s) & (pk["strategy"] == n) & pk["decided"]]
            pc = d["p_display"] / (1 - d["p_push"]).clip(lower=1e-6)
            bins = C.reliability_bins(d["won"], pc, edges=np.arange(0.35, 0.95, 0.05), min_n=40)
            ax.plot([b["predicted"] for b in bins], [b["observed"] for b in bins], marker="o", ms=3, lw=1,
                    color=PALETTE[k % len(PALETTE)], label=n)
        ax.plot([0.35, 0.9], [0.35, 0.9], color="#888", ls="--", lw=0.8)
        ax.set_title(C.period_label(s), fontsize=9.5), ax.set_xlabel("Заявленная вероятность Main")
    axes[0].set_ylabel("Доля выигрышей"), axes[0].legend(fontsize=6.5)
    fig.suptitle("Надёжность показанного Main у разных стратегий (только с ценой)", fontsize=10)
    save_chart(fig, C.FIGURES / "strat_reliability.png")
    show = ["A", "F1 blend Shin (inner weight)", "G7 EV>0 (meta-model)", "C3 max EV", "B3 selection-conditional"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, s in zip(axes, C.FOLDS):
        for k, n in enumerate(show):
            d = pk.loc[(pk["season"] == s) & (pk["strategy"] == n) & pk["bet"] & pk["priced"] & pk["decided"]]
            d = d.sort_values(["date", "mi"])
            ax.plot(np.arange(len(d)), d["profit"].cumsum().to_numpy(), lw=1, color=PALETTE[k % len(PALETTE)], label=n)
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_title(C.period_label(s), fontsize=9.5), ax.set_xlabel("Ставка № (по дате)")
    axes[0].set_ylabel("Накопленная прибыль, ед."), axes[0].legend(fontsize=6.5)
    fig.suptitle("Накопленная прибыль по сезонам (ставка 1 ед., реальные цены Bet365)", fontsize=10)
    save_chart(fig, C.FIGURES / "strat_cumulative.png")


if __name__ == "__main__":
    C.setup_logging("figures")
    run()
