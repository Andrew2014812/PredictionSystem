"""Figures of the Main-selection study (read from results/, written to figures/).

    python -m experiments.v6_3.main_selection.figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import report_kit as K  # noqa: E402

from . import data as D  # noqa: E402

RES, FIG = D.DIRS["results"], D.DIRS["figures"]
UCOL = {"close": K.PALETTE[0], "t09": K.PALETTE[1], "bt": K.PALETTE[2]}
ULAB = {"close": "2026/27 close", "t09": "2026/27 T", "bt": "2025/26 бэктест"}
GCOL = {"TOTAL": K.PALETTE[0], "HANDICAP": K.PALETTE[1], "1X2": K.PALETTE[2], "DOUBLE_CHANCE": K.PALETTE[3],
        "CORNERS": K.PALETTE[4], "BTTS": K.PALETTE[5]}


def forest(cands: pd.DataFrame, path, title: str) -> None:
    names = list(dict.fromkeys(cands["candidate"]))
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(names) + 1.4))
    y = np.arange(len(names))
    for k, (u, off) in enumerate((("close", -0.17), ("t09", 0.17))):
        t = cands.loc[cands["universe"] == u].set_index("candidate").reindex(names)
        d = t["roi_diff"].to_numpy(float) * 100
        lo = np.array([c[0] if isinstance(c, list) else np.nan for c in t["diff_ci"]], float) * 100
        hi = np.array([c[1] if isinstance(c, list) else np.nan for c in t["diff_ci"]], float) * 100
        ax.errorbar(d, y + off, xerr=[d - lo, hi - d], fmt="o", ms=4, color=UCOL[u], label=ULAB[u], capsize=2)
    ax.axvline(0, color=K.INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("ΔROI к Baseline B, п.п. (95 % ДИ, парный бутстреп)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    K.save_chart(fig, path)


def floors(fl: pd.DataFrame) -> None:
    order = ["production", "floor_1.30", "floor_1.40", "floor_1.50", "floor_1.60"]
    x = [1.0, 1.30, 1.40, 1.50, 1.60]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for u in ("close", "t09", "bt"):
        t = fl.loc[fl["universe"] == u].set_index("rule").reindex(order)
        ax[0].plot(x, t["roi"] * 100, "o-", color=UCOL[u], label=ULAB[u])
        ax[1].plot(x, t["fallback"], "o-", color=UCOL[u], label=ULAB[u])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["нет", "1.30", "1.40", "1.50", "1.60"])
    ax[0].set_xlabel("жёсткий минимальный коэффициент")
    ax[0].set_ylabel("ROI Main, %")
    ax[0].set_title("ROI при минимальном коэффициенте")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["нет", "1.30", "1.40", "1.50", "1.60"])
    ax[1].set_xlabel("жёсткий минимальный коэффициент")
    ax[1].set_ylabel("матчей с резервным Main (не ставка)")
    ax[1].set_title("Резервная логика: сколько матчей")
    ax[0].legend(fontsize=8)
    K.save_chart(fig, FIG / "floors.png")


def curves(grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for k, (fam, pref, base) in enumerate((("power", "power_", 1.0), ("ev_weight", "evw_", 0.5))):
        for u in ("close", "t09", "bt"):
            t = grid.loc[(grid["universe"] == u) & ((grid["family"] == fam) | (grid["rule"] == "production"))]
            t = t.drop_duplicates("rule")
            xs = t["rule"].map(lambda r: base if r == "production" else float(r.replace(pref, "")))
            o = np.argsort(xs.to_numpy())
            ax[k].plot(xs.to_numpy()[o], t["roi"].to_numpy()[o] * 100, "o-", color=UCOL[u], label=ULAB[u])
        ax[k].axvline(base, color=K.MUTED, lw=0.8, ls="--")
        ax[k].set_ylabel("ROI Main, %")
    ax[0].set_xlabel("степень вероятности MAIN_PROBABILITY_POWER")
    ax[0].set_title("Вес вероятности (пунктир — продакшн)")
    ax[1].set_xlabel("вес EV MAIN_EV_WEIGHT")
    ax[1].set_title("Вес EV (пунктир — продакшн)")
    ax[0].legend(fontsize=8)
    K.save_chart(fig, FIG / "power_ev.png")


def consistency(grid: pd.DataFrame) -> None:
    g = grid.loc[(grid["rule"] != "production") & (grid["family"] != "diagnostic")]
    a = g.loc[g["universe"] == "close"].set_index("rule")
    b = g.loc[g["universe"] == "t09"].set_index("rule")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fams = list(dict.fromkeys(a["family"]))
    for i, f in enumerate(fams):
        r = a.index[a["family"] == f]
        ax.scatter(a.loc[r, "roi_diff"] * 100, b.loc[r, "roi_diff"] * 100, s=22,
                   color=plt.cm.tab10(i % 10), label=f)
    ax.axhline(0, color=K.INK, lw=0.7)
    ax.axvline(0, color=K.INK, lw=0.7)
    ax.set_xlabel("ΔROI к Baseline B, close, п.п.")
    ax.set_ylabel("ΔROI к Baseline B, T, п.п.")
    ax.set_title("Вся сетка: согласуется ли эффект в двух вселенных")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    K.save_chart(fig, FIG / "grid_consistency.png")


def cumulative(cp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for k, u in enumerate(("close", "t09")):
        for i, (name, t) in enumerate(cp.loc[cp["universe"] == u].groupby("method", sort=False)):
            ax[k].plot(pd.to_datetime(t["date"]), t["cum_profit"], lw=1.8 if name == "baseline_b" else 1.1,
                       color=K.INK if name == "baseline_b" else K.PALETTE[i % len(K.PALETTE)], label=name)
        ax[k].axhline(0, color=K.MUTED, lw=0.7)
        ax[k].set_title(f"Накопленная прибыль, {ULAB[u]}")
        ax[k].set_ylabel("единиц ставки")
        ax[k].tick_params(axis="x", labelrotation=30)
    ax[0].legend(fontsize=7)
    K.save_chart(fig, FIG / "cumulative.png")


def bands(b: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    labels = list(dict.fromkeys(b["band"]))
    x = np.arange(len(labels))
    for k, u in enumerate(("close", "t09", "bt")):
        t = b.loc[b["universe"] == u].set_index("band").reindex(labels)
        ax.bar(x + (k - 1) * 0.27, t["roi"] * 100, 0.27, color=UCOL[u], label=ULAB[u])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("[", "").replace(")", "").replace(", ", "–").replace("100.0", "∞")
                        for s in labels])
    ax.axhline(0, color=K.INK, lw=0.7)
    ax.set_xlabel("коэффициент Main продакшна")
    ax.set_ylabel("ROI, %")
    ax.set_title("ROI Main продакшна по диапазонам коэффициента")
    ax.legend(fontsize=8)
    K.save_chart(fig, FIG / "odds_bands.png")


def mix(details: dict, base: dict) -> None:
    names = ["Baseline B"] + [n for n in ("floor_1.40", "ew:reliability", "bt:ev_weight", "bt:ranking", "combo_bt")
                              if n in details]
    rows = [base["close"]["groups"]] + [details[n]["close"]["groups"] for n in names[1:]]
    groups = ["TOTAL", "HANDICAP", "1X2", "DOUBLE_CHANCE", "CORNERS", "BTTS"]
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    left = np.zeros(len(names))
    for gname in groups:
        v = np.array([r.get(gname, 0.0) for r in rows]) * 100
        ax.barh(names, v, left=left, color=GCOL[gname], label=gname)
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("доля Main, % (close)")
    ax.set_title("Распределение выбранных рынков")
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    K.save_chart(fig, FIG / "market_mix.png")


def main() -> None:
    import json
    res = json.loads((RES / "main_selection.json").read_text())
    det = json.loads((RES / "candidates_detail.json").read_text())
    grid = pd.read_csv(RES / "grid.csv")
    cands = pd.DataFrame(res["candidates"])
    forest(cands, FIG / "candidates_forest.png", "Заранее выбранные кандидаты против Baseline B")
    floors(pd.DataFrame(res["floors"]))
    curves(grid)
    consistency(grid)
    cumulative(pd.read_csv(RES / "cumulative_profit.csv"))
    bands(pd.DataFrame(res["odds_bands"]))
    mix(det, res["baselines"])


if __name__ == "__main__":
    main()
