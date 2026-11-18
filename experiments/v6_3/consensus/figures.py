"""Figures of the market-consensus study (from results/consensus.json)."""
from __future__ import annotations

import numpy as np

from experiments.v6_3 import common as C
from experiments.v6_3.market_aware.figures import COL, GRAY, U_RU, err
from scripts.report_kit import plt, save_chart

from .fd import SEASONS
from .run import FIG, GAP_LABELS, RES


def season(s):
    return f"{s}/{(s + 1) % 100:02d}"


def fig_rules(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, snap in zip(axes, ("open", "close")):
        x = np.arange(len(SEASONS))
        for j, (rule, lab) in enumerate((("median", "медиана книг − Bet365"), ("fd_avg", "Avg FD − Bet365"))):
            mean, ci, xs = [], [], []
            for i, s in enumerate(SEASONS):
                v = r["fd_rules"]["by_key"].get(f"1X2|{snap}|{s}", {}).get(rule)
                if v and v.get("minus_bet365"):
                    mean.append(v["minus_bet365"]["diff"] * 1000)
                    ci.append([c * 1000 for c in v["minus_bet365"]["ci95"]])
                    xs.append(i + (j - 0.5) * 0.22)
            err(ax, xs, mean, ci, COL[j], lab, "os"[j])
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(x, [season(s) for s in SEASONS], rotation=30)
        ax.set_title("1X2, снимок " + ("открытия" if snap == "open" else "закрытия"))
    axes[0].set_ylabel("Δ log loss, ×10⁻³ (меньше — лучше)")
    axes[0].legend(frameon=False, loc="upper left")
    return save_chart(fig, FIG / "fd_rules_1x2.png")


def fig_dispersion(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    d = r["fd_rules"]["dispersion"]
    for j, snap in enumerate(("open", "close")):
        ex = {q: [] for q in "12345"}
        sl = {q: [] for q in "12345"}
        for s in SEASONS:
            for q, v in d.get(f"{snap}|{s}", {}).items():
                ex[q].append(v["excess"]["diff"] * 1000)
                sl[q].append(v["calib_slope"])
        x = np.arange(1, 6)
        axes[0].plot(x + (j - 0.5) * 0.1, [np.mean(ex[q]) for q in "12345"], "os"[j] + "-", color=COL[j], lw=2,
                     label="открытие" if snap == "open" else "закрытие")
        axes[1].plot(x + (j - 0.5) * 0.1, [np.mean(sl[q]) for q in "12345"], "os"[j] + "-", color=COL[j], lw=2,
                     label="открытие" if snap == "open" else "закрытие")
    axes[0].axhline(0, color=GRAY, lw=1)
    axes[1].axhline(1, color=GRAY, lw=1)
    for ax in axes:
        ax.set_xticks(range(1, 6), ["1\nнизкий", "2", "3", "4", "5\nвысокий"])
        ax.set_xlabel("квинтиль разброса книг")
    axes[0].set_ylabel("избыточный log loss, ×10⁻³")
    axes[1].set_ylabel("наклон калибровки консенсуса")
    axes[0].legend(frameon=False)
    return save_chart(fig, FIG / "dispersion_quintiles.png")


def fig_variants(r):
    keys = [k for k in r["variants_ru"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(keys))
    for j, u in enumerate(("close", "t09")):
        v = r["variants"][u]
        mean = [v[k]["vs_b"]["roi_diff"]["diff"] * 100 for k in keys]
        ci = [[c * 100 for c in v[k]["vs_b"]["roi_diff"]["ci95"]] for k in keys]
        ax.errorbar(mean, y + (j - 0.5) * 0.28, xerr=[[m - c[0] for m, c in zip(mean, ci)],
                                                       [c[1] - m for m, c in zip(mean, ci)]],
                    fmt="os"[j], color=COL[j], ms=6, capsize=3, label=U_RU[u])
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(y, [r["variants_ru"][k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("ROI(вариант) − ROI(Baseline B), п.п. (парный бутстреп, 95 % ДИ)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return save_chart(fig, FIG / "variants_vs_baseline_b.png")


def fig_calibration(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = {"model": "модель", "single": "одна книга", "consensus": "консенсус"}
    for ax, u in zip(axes, ("close", "t09")):
        cal = r["model"][u]["calibration"]
        for j, (k, lab) in enumerate(names.items()):
            cv = [c for c in cal[k]["curve"] if c["n"] >= 50]
            ax.plot([c["p"] for c in cv], [c["y"] for c in cv], "osD"[j] + "-", color=COL[j], lw=1.8, ms=5,
                    label=f"{lab} (ECE {cal[k]['ece'] * 100:.2f} п.п.)")
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(U_RU[u])
        ax.set_xlabel("вероятность исхода")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_ylabel("частота выигрыша")
    return save_chart(fig, FIG / "calibration.png")


def fig_gaps(r):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, u in enumerate(("close", "t09")):
        q = r["decision_time"][u]["gap_quantiles_h"]
        ks = sorted(q, key=float)
        ax.plot([float(k) * 100 for k in ks], [q[k] for k in ks], "os"[j] + "-", color=COL[j], lw=2, label=U_RU[u])
    ax.set_xlabel("квантиль, %")
    ax.set_ylabel("разрыв во времени между книгами, ч")
    ax.legend(frameon=False)
    return save_chart(fig, FIG / "book_time_gap.png")


def run():
    r = C.read_json(RES / "consensus.json")
    for f in (fig_rules, fig_dispersion, fig_variants, fig_calibration, fig_gaps):
        f(r)


__all__ = ["run", "GAP_LABELS"]

if __name__ == "__main__":
    run()
