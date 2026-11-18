"""Figures of the timing & CLV study (from results/timing.json and results/clv_main_t09.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.market_aware.figures import COL, GRAY, err
from scripts.report_kit import plt, save_chart

from .run import FIG, RES

MOMENT_RU = {"early": "за 48 ч", "t09": "момент T", "close": "закрытие"}


def fig_lead(r):
    a = r["archive"]
    bins = np.array(a["lead_bins"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, b in enumerate(("bet365", "pinnacle")):
        h = np.array(a["lead_hist"].get(b, []), float)
        if not len(h):
            continue
        ax.step(bins[:-1], h / h.sum() * 100, where="post", color=COL[j], lw=2, label=b)
    ax.set_xlabel("время котировки до начала матча, ч (интервалы по 6 ч)")
    ax.set_ylabel("доля котировок, %")
    ax.legend(frameon=False)
    return save_chart(fig, FIG / "quote_lead_time.png")


def fig_moments(r):
    mo = r["moments"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(3)
    for j, (key, lab) in enumerate((("summary", "все матчи (ставка — где есть цена)"),
                                    ("all_priced", f"матчи с ценой во всех трёх моментах ({mo['all_priced_matches']})"))):
        d = mo[key]
        mean = [d[m]["roi"] * 100 for m in ("early", "t09", "close")]
        ci = [[c * 100 for c in d[m]["roi_ci"]["ci95"]] for m in ("early", "t09", "close")]
        err(ax, x + (j - 0.5) * 0.2, mean, ci, COL[j], lab, "os"[j])
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xticks(x, [f"{MOMENT_RU[m]}\n(ставок {mo['summary'][m]['bets']})" for m in ("early", "t09", "close")])
    ax.set_ylabel("ROI Main, % (95 % ДИ)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    return save_chart(fig, FIG / "roi_by_moment.png")


def fig_clv():
    d = pd.read_csv(RES / "clv_main_t09.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bins = np.linspace(-0.25, 0.25, 41)
    for j, (col, lab) in enumerate((("clv_price", "ценовой CLV: odds_T / odds_close − 1"),
                                    ("clv_fair", "справедливый CLV: odds_T · q_close − 1"))):
        v = d[col].dropna().clip(-0.25, 0.25)
        ax.hist(v, bins=bins, histtype="step", lw=2, color=COL[j], label=f"{lab} (ср. {v.mean() * 100:+.1f}%)")
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("CLV Main-ставки (обрезано ±25 %)")
    ax.set_ylabel("число ставок")
    ax.legend(frameon=False, fontsize=8)
    return save_chart(fig, FIG / "clv_distribution.png")


def fig_movement(r):
    mv = r["movement"]
    rows = [(f"OddsPapi {g}", v) for g, v in mv["oddspapi"].items() if v.get("slope") is not None]
    rows += [(f"FD {k.replace('|', ' ')}", v) for k, v in mv["football_data"].items()]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(rows))
    mean = [v["slope"] for _, v in rows]
    ci = [v["ci95"] for _, v in rows]
    ax.errorbar(mean, y, xerr=[[m - c[0] for m, c in zip(mean, ci)], [c[1] - m for m, c in zip(mean, ci)]], fmt="o",
                color=COL[0], ms=6, capsize=3)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(y, [f"{k} (n={v['n']})" for k, v in rows])
    ax.invert_yaxis()
    ax.set_xlabel("наклон: движение рынка (q_close − q_T) на расхождение (p − q_T), 95 % ДИ")
    return save_chart(fig, FIG / "movement_slopes.png")


def run():
    r = C.read_json(RES / "timing.json")
    fig_lead(r)
    fig_moments(r)
    fig_clv()
    fig_movement(r)


if __name__ == "__main__":
    run()
