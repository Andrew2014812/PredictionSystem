"""Charts of the Main-calibration report (read from results/)."""
from __future__ import annotations

import numpy as np

from . import common as C

SHOW = ["A базовый", "PS-Platt", "PS-Market", "Q3 Q2 + Платт после выбора"]
SHOW26 = ["A как показано", "показано + PS-Group", "показано + PS-Platt по месяцам", "показано + PS-Group по месяцам"]


def _labels():
    return {s: C.period_label(s) for s in C.SEASONS}


def reliability(res, plt, PALETTE, save_chart):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, s in zip(axes, C.SEASONS):
        for k, nm in enumerate(SHOW26 if s == C.CURRENT else SHOW):
            r = res["reliability"].get(f"site|{s}|{nm}", [])
            if r:
                ax.errorbar([x["predicted"] for x in r], [x["observed"] for x in r],
                            yerr=[1.96 * x["se"] for x in r], marker="o", ms=3, lw=1, capsize=1.5,
                            color=PALETTE[k], label=nm)
        ax.plot([0.35, 0.95], [0.35, 0.95], ls="--", color="#888", lw=0.8)
        ax.set_title(C.period_label(s), fontsize=9)
        ax.set_xlabel("Заявленная вероятность Main")
    axes[0].set_ylabel("Доля выигрышей")
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[-1].legend(fontsize=6, loc="upper left")
    fig.suptitle("Надёжность выбранного Main (универсум site): до и после калибровки", fontsize=10)
    save_chart(fig, C.FIGURES / "reliability_main.png")


def gaps(res, plt, PALETTE, save_chart, kind="site"):
    v = res["variants"][kind]
    names = [nm for nm in v[str(C.SELECT[0])] if nm != "A как показано"]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(names) + 1.2))
    for k, s in enumerate(C.SEASONS):
        xs, ys, lo, hi = [], [], [], []
        for i, nm in enumerate(names):
            m = v[str(s)].get(nm)
            if m is None:
                continue
            xs.append(m["gap"] * 100)
            ys.append(i + (k - 1.5) * 0.17)
            lo.append((m["gap"] - m["gap_ci"][0]) * 100)
            hi.append((m["gap_ci"][1] - m["gap"]) * 100)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3, lw=0.9, capsize=1.5, color=PALETTE[k],
                    label=C.period_label(s))
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Заявленная вероятность − доля выигрышей, п.п. (95 % ДИ)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Переоценка выбранного Main по вариантам (универсум {kind})", fontsize=10)
    save_chart(fig, C.FIGURES / f"gap_variants_{kind}.png")


def roi(res, plt, PALETTE, save_chart):
    v = res["variants"]["bet"]
    names = [nm for nm in v[str(C.SELECT[0])]]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(names) + 1.2))
    for k, s in enumerate(C.SEASONS):
        xs, ys, lo, hi = [], [], [], []
        for i, nm in enumerate(names):
            m = v[str(s)].get(nm)
            if m is None or m["roi"] is None:
                continue
            xs.append(m["roi"] * 100)
            ys.append(i + (k - 1.5) * 0.17)
            lo.append((m["roi"] - m["roi_ci"][0]) * 100)
            hi.append((m["roi_ci"][1] - m["roi"]) * 100)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3, lw=0.9, capsize=1.5, color=PALETTE[k],
                    label=C.period_label(s))
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("ROI при реальных ценах, % (95 % ДИ)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Доходность Main по вариантам (универсум bet: только кандидаты с реальной ценой)", fontsize=10)
    save_chart(fig, C.FIGURES / "roi_variants_bet.png")


def ncand(res, plt, PALETTE, save_chart):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, kind in zip(axes, ("site", "bet")):
        series = [(nm, C.FOLDS, f"{nm}, 2023–2025") for nm in ["A базовый", "PS-Platt", "PS-OS"]]
        series += [(nm, (C.CURRENT,), f"{nm}, 2026/27") for nm in ["A базовый", "PS-OS"]]
        keys = []
        for k, (nm, seasons, lab) in enumerate(series):
            agg = {}
            for s in seasons:
                for r in res["ncand"].get(f"{kind}|{s}|{nm}", []):
                    a = agg.setdefault(r["key"], [0, 0.0, 0.0])
                    a[0] += r["n"]
                    a[1] += r["n"] * r["mean_p"]
                    a[2] += r["n"] * r["hit"]
            for key in agg:
                if key not in keys:
                    keys.append(key)
            pts = [(key, v) for key, v in agg.items() if v[0] >= 30]
            xs = [keys.index(key) for key, _ in pts]
            ys = [(sp - sh) / n * 100 for _, (n, sp, sh) in pts]
            es = [1.96 * np.sqrt(sh / n * (1 - sh / n) / n) * 100 for _, (n, sp, sh) in pts]
            ax.errorbar(np.array(xs) + (k - 2) * 0.07, ys, yerr=es, marker="o" if k < 3 else "s", ms=3, capsize=1.5,
                        lw=1, ls="-" if k < 3 else "--", color=PALETTE[k], label=lab)
        order = sorted(range(len(keys)), key=lambda i: float(keys[i].strip("([").split(",")[0]))
        ax.set_xticks(range(len(keys)), keys, fontsize=6.5, rotation=30)
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_title(f"Универсум {kind}", fontsize=9)
        ax.set_xlabel("Число кандидатов в диапазоне 0.40–0.85 (в bet — с ценой)")
        _ = order
    axes[0].set_ylabel("Заявлено − факт, п.п.")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Переоценка выбранного Main в зависимости от числа кандидатов", fontsize=10)
    save_chart(fig, C.FIGURES / "gap_ncand.png")


def ev(plt, PALETTE, save_chart):
    evc = C.load_json("ev_calibration.json", {})
    if not evc:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, s in zip(axes, C.SEASONS):
        for k, nm in enumerate(["сырая p", "PS-Platt", "PS-Market", "рынок (Шин)"]):
            r = evc[str(s)][nm]["bins"]
            ax.errorbar([x["stated_ev"] * 100 for x in r], [x["realised"] * 100 for x in r],
                        yerr=[[(x["realised"] - x["ci"][0]) * 100 for x in r], [(x["ci"][1] - x["realised"]) * 100 for x in r]],
                        marker="o", ms=3, lw=1, capsize=1.5, color=PALETTE[k], label=nm)
        ax.plot([-15, 15], [-15, 15], ls="--", color="#888", lw=0.8)
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_title(C.period_label(s), fontsize=9)
        ax.set_xlabel("Заявленный EV, %")
    axes[0].set_ylabel("Фактическая доходность на ставку, %")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Калибровка EV выбранного Main (универсум bet, Bet365)", fontsize=10)
    save_chart(fig, C.FIGURES / "ev_calibration.png")


def leagues(plt, PALETTE, save_chart):
    lt = C.load_json("league_transfer.json", {})
    if not lt:
        return
    tab = sorted(lt["site"]["2023-2025"]["table"], key=lambda r: r["raw"] - r["hit"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(tab))
    for k, (col, lab) in enumerate((("raw", "без калибровки"), ("pooled", "PS-Platt (все лиги)"),
                                    ("lolo", "PS-Platt без своей лиги"))):
        ax.errorbar(x + (k - 1) * 0.22, [(r[col] - r["hit"]) * 100 for r in tab],
                    yerr=[1.96 * r["se"] * 100 for r in tab] if col == "raw" else None, fmt="o", ms=3.2,
                    capsize=1.5, lw=0.8, color=PALETTE[k], label=lab)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(x, [r["league"] for r in tab], fontsize=7.5)
    ax.set_ylabel("Заявлено − факт, п.п.")
    ax.legend(fontsize=7)
    ax.set_title("Перенос калибровки между лигами (site, 2023/24–2025/26)", fontsize=10)
    save_chart(fig, C.FIGURES / "league_transfer.png")


def all_figures() -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    res = C.load_json("variants.json")
    reliability(res, plt, PALETTE, save_chart)
    gaps(res, plt, PALETTE, save_chart, "site")
    gaps(res, plt, PALETTE, save_chart, "bet")
    roi(res, plt, PALETTE, save_chart)
    ncand(res, plt, PALETTE, save_chart)
    ev(plt, PALETTE, save_chart)
    leagues(plt, PALETTE, save_chart)
