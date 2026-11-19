"""Figures of the meta-selection study (from results/meta_selection.json)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import report_kit as K
from scripts.report_kit import plt

from experiments.v6_3 import common as C

from . import features as F

FIG = F.DIRS["figures"]
NAME_RU = {"baseline_b": "Baseline B (правило продакшна)", "meta_lr_ev": "мета: логрегрессия, max EV",
           "meta_gbm_ev": "мета: бустинг, max EV", "meta_gbm_ret": "мета: бустинг, max доходность",
           "meta_lr_rule": "мета: правило на P мета", "rule_priced": "правило продакшна, только с ценой", "max_p": "max вероятность модели", "max_ev": "max EV",
           "max_q": "max вероятность рынка", "rule_calibrated": "правило на откалибр. p", "random": "случайный выбор"}
UNIV_RU = {"close": "close (цена перед стартом)", "t09": "t09 (цена в момент решения)"}


def forest(res: dict) -> None:
    names = res["meta"] + res["simple"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        for i, n in enumerate(names):
            v = res[u][n]["vs_b"]["roi_diff"]
            ax.errorbar(v["diff"] * 100, i, xerr=[[(v["diff"] - v["ci95"][0]) * 100], [(v["ci95"][1] - v["diff"]) * 100]],
                        marker="o", ms=4, capsize=2, color=K.PALETTE[0] if n in res["meta"] else K.PALETTE[2])
        ax.axvline(0, color="#777", lw=0.8)
        ax.set_title(UNIV_RU[u], fontsize=9)
        ax.set_xlabel("ΔROI к Baseline B, п.п. (95 % ДИ)")
    axes[0].set_yticks(range(len(names)), [NAME_RU[n] for n in names], fontsize=7.5)
    K.save_chart(fig, FIG / "roi_forest.png")


def learning(res: dict) -> None:
    lc = pd.DataFrame(res["checks"]["close"]["learning_curve"]).groupby("frac").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for n in ("meta_lr_ev", "meta_gbm_ev"):
        axes[0].plot(lc["frac"] * 100, lc[f"{n}_log_loss"], marker="o", label=NAME_RU[n])
        axes[1].plot(lc["frac"] * 100, lc[f"{n}_roi"] * 100, marker="o", label=NAME_RU[n])
    pq = res["close"]["prob_quality"]
    axes[0].axhline(pq["raw"]["log_loss"], color=K.PALETTE[4], ls="--", lw=1, label="модель продакшна")
    axes[0].axhline(pq["market_q"]["log_loss"], color=K.PALETTE[5], ls=":", lw=1, label="рынок (без маржи)")
    axes[1].axhline(res["close"]["baseline_b"]["summary"]["roi"] * 100, color=K.PALETTE[4], ls="--", lw=1,
                    label="Baseline B")
    lo = min(lc[[f"{n}_log_loss" for n in ("meta_lr_ev", "meta_gbm_ev")]].min().min(), pq["market_q"]["log_loss"])
    hi = max(lc[[f"{n}_log_loss" for n in ("meta_lr_ev", "meta_gbm_ev")]].max().max(), pq["raw"]["log_loss"])
    axes[0].set_ylim(lo - 0.001, hi + 0.001)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 50, 100], ["5", "10", "25", "50", "100"])
        ax.set_xlabel("доля обучающих матчей, %")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("log loss кандидатов месяца M")
    axes[1].set_ylabel("ROI выбранного Main, %")
    K.save_chart(fig, FIG / "learning_curve.png")


def permutation(res: dict) -> None:
    ch = res["checks"]["close"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    rnd = np.array(res["close"]["random_pick"]["values"]) * 100
    for ax, n in zip(axes, ("meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret")):
        ax.hist(rnd, bins=30, color=K.PALETTE[5], alpha=0.5, label="случайный выбор из пула")
        pv = np.array(ch["permutation"][n]["values"]) * 100
        ax.plot(pv, np.zeros_like(pv) + 1, "|", ms=14, color=K.PALETTE[1], label="перемешанные цели")
        ax.axvline(res["close"][n]["summary"]["roi"] * 100, color=K.PALETTE[0], lw=2, label="мета (walk-forward)")
        ax.axvline(res["close"]["baseline_b"]["summary"]["roi"] * 100, color=K.PALETTE[4], ls="--", lw=1,
                   label="Baseline B")
        ax.set_title(NAME_RU[n], fontsize=8.5)
        ax.set_xlabel("ROI, %")
    axes[0].legend(fontsize=6.5)
    K.save_chart(fig, FIG / "permutation.png")


def monthly(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, u in zip(axes, ("close", "t09")):
        for n in ("baseline_b", "meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret"):
            t = pd.DataFrame(res[u]["robustness"][n]["by_month"])
            ax.plot(t["month"], t["roi"] * 100, marker="o", label=NAME_RU[n])
        ax.axhline(0, color="#777", lw=0.8)
        ax.set_title(UNIV_RU[u], fontsize=9)
        ax.set_ylabel("ROI, %")
    axes[0].legend(fontsize=7)
    K.save_chart(fig, FIG / "monthly.png")


def coefs(res: dict) -> None:
    c = res["checks"]["close"]["lr_coef"]
    items = list(c.items())[:16][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[K.PALETTE[0] if v > 0 else K.PALETTE[4] for _, v in items])
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_xlabel("стандартизированный коэффициент (итоговая логрегрессия)")
    ax.tick_params(axis="y", labelsize=7.5)
    K.save_chart(fig, FIG / "lr_coef.png")


def main() -> None:
    res = C.read_json(F.DIRS["results"] / "meta_selection.json")
    forest(res)
    learning(res)
    permutation(res)
    monthly(res)
    coefs(res)


if __name__ == "__main__":
    main()
