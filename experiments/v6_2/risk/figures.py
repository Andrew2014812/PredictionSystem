"""Charts of the Risk report (read from results/)."""
from __future__ import annotations

import numpy as np

from . import common as C


def ev_decomp(res, plt, PAL, save):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = ["EV модели", "EV по цене рынка", "Факт на ставку"]
    w = 0.2
    for k, s in enumerate(C.SEASONS):
        e = res["diagnostics"][str(s)]["ev_decomposition"]
        vals = [e.get("book_model_ev"), e.get("book_market_ev"), e.get("book_realised")]
        ci = e.get("book_realised_ci") or [np.nan, np.nan]
        x = np.arange(3) + (k - 1.5) * w
        ax.bar(x, [v * 100 if v is not None else np.nan for v in vals], width=w, color=PAL[k], label=C.period_label(s))
        ax.errorbar(x[2], vals[2] * 100, yerr=[[(vals[2] - ci[0]) * 100], [(ci[1] - vals[2]) * 100]], color="#333",
                    capsize=2, lw=0.8)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("% на единицу ставки")
    ax.legend(fontsize=7)
    ax.set_title("Risk: заявленный EV модели, EV по цене без маржи и фактическая доходность", fontsize=10)
    save(fig, C.FIGURES / "ev_decomposition.png")


def flb(res, plt, PAL, save):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, (key, lab) in enumerate((("pooled_history", "2022/23–2025/26 (Bet365, Football-Data)"),
                                    ("2026", "2026/27 (записанные цены)"))):
        rows = res["diagnostics"][key]["favourite_longshot"]
        x = np.arange(len(rows))
        axes[0].errorbar(x + (k - 0.5) * 0.15, [r["roi"] * 100 for r in rows],
                         yerr=[[(r["roi"] - r["roi_ci"][0]) * 100 for r in rows], [(r["roi_ci"][1] - r["roi"]) * 100 for r in rows]],
                         fmt="o", ms=3.5, capsize=2, color=PAL[k], label=lab)
        axes[0].set_xticks(x, [r["band"] for r in rows], rotation=35, fontsize=7)
        axes[1].plot(x, [r["q_gap"] * 100 for r in rows], marker="o", ms=3.5, color=PAL[k], label=f"рынок без маржи, {lab.split(' (')[0]}")
        axes[1].plot(x, [r["model_gap"] * 100 for r in rows], marker="s", ms=3.5, ls=":", color=PAL[k],
                     label=f"модель, {lab.split(' (')[0]}")
        axes[1].set_xticks(x, [r["band"] for r in rows], rotation=35, fontsize=7)
    axes[0].axhline(0, color="#555", lw=0.7)
    axes[0].set_ylabel("ROI ставки на каждый исход, %")
    axes[0].set_title("Слепая ставка на каждый исход с ценой", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="#555", lw=0.7)
    axes[1].set_ylabel("Вероятность − частота, п.п.")
    axes[1].set_title("Смещение вероятности по полосам коэффициента", fontsize=9)
    axes[1].legend(fontsize=6.5)
    fig.suptitle("Эффект фаворит–аутсайдер", fontsize=10)
    save(fig, C.FIGURES / "favourite_longshot.png")


def rare(res, plt, PAL, save):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, key, title in ((axes[0], "rare_pool", "Все кандидаты Risk (p ≥ 0.12)"),
                           (axes[1], "rare_picks", "Выбранный Risk")):
        for k, s in enumerate(C.SEASONS):
            rows = res["diagnostics"][str(s)][key]
            ax.errorbar([r["mean_p"] for r in rows], [r["hit"] for r in rows],
                        yerr=[1.96 * np.sqrt(r["hit"] * (1 - r["hit"]) / r["n"]) for r in rows],
                        marker="o", ms=3, lw=1, capsize=1.5, color=PAL[k], label=C.period_label(s))
        ax.plot([0.1, 0.55], [0.1, 0.55], ls="--", color="#888", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Вероятность модели (без возврата)")
    axes[0].set_ylabel("Доля выигрышей")
    axes[0].legend(fontsize=7)
    fig.suptitle("Калибровка редких исходов", fontsize=10)
    save(fig, C.FIGURES / "rare_calibration.png")


def odds_dep(res, plt, PAL, save):
    rows = res["breakdowns"]["2023-2025"]["odds"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["model_ev"] * 100 for r in rows], width=0.4, color=PAL[1], label="EV модели")
    ax.errorbar(x + 0.2, [r["roi"] * 100 for r in rows],
                yerr=[[(r["roi"] - r["roi_ci"][0]) * 100 for r in rows], [(r["roi_ci"][1] - r["roi"]) * 100 for r in rows]],
                fmt="o", color=PAL[0], capsize=2, label="Фактический ROI (95 % ДИ)")
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(x, [f"{r['band']}\n(n={r['n']})" for r in rows], fontsize=7)
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.set_title("Текущий Risk по полосам коэффициента, 2023/24–2025/26", fontsize=10)
    save(fig, C.FIGURES / "odds_dependence.png")


def strategies(res, plt, PAL, save):
    names = list(res["strategies"][str(C.SELECT[0])])
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(names) + 1.4))
    for k, s in enumerate(C.SEASONS):
        xs, ys, lo, hi = [], [], [], []
        for i, nm in enumerate(names):
            m = res["strategies"][str(s)].get(nm)
            if not m or m["roi"] is None:
                continue
            xs.append(m["roi"] * 100)
            ys.append(i + (k - 1.5) * 0.17)
            lo.append((m["roi"] - m["roi_ci"][0]) * 100)
            hi.append((m["roi_ci"][1] - m["roi"]) * 100)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3, lw=0.9, capsize=1.5, color=PAL[k], label=C.period_label(s))
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("ROI при реальных ценах, % (95 % ДИ)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Правила выбора Risk на одних и тех же матчах", fontsize=10)
    save(fig, C.FIGURES / "strategies_roi.png")


def fewer(res, plt, PAL, save):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, base in zip(axes, ["Макс. EV (стек с рынком)", "Макс. EV (сырая p)", "Макс. перевес над рынком"]):
        for k, s in enumerate(C.SEASONS):
            f = res["fewer"][str(s)]
            pts = [(m["bets"], m["roi"]) for nm, m in f.items()
                   if nm.startswith(base + " | топ") and m["roi"] is not None and m["bets"] >= 20]
            full = res["strategies"][str(s)][base]
            pts.append((full["bets"], full["roi"]))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], marker="o", ms=3, color=PAL[k], label=C.period_label(s))
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xscale("log")
        ax.set_title(base, fontsize=8.5)
        ax.set_xlabel("Число ставок за сезон")
    axes[0].set_ylabel("ROI, %")
    axes[0].legend(fontsize=7)
    fig.suptitle("Устойчивость при ограничении числа ставок: топ-1 / 3 / 5 матчей дня по ключу и все матчи", fontsize=10)
    save(fig, C.FIGURES / "fewer_bets.png")


def post(res, plt, PAL, save):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, s in zip(axes, C.SEASONS):
        for k, nm in enumerate(["сырая p", "PS-Platt", "PS-Market", "PS-Group"]):
            r = res["post"][str(s)][nm]["reliability"]
            ax.errorbar([x["predicted"] for x in r], [x["observed"] for x in r], yerr=[1.96 * x["se"] for x in r],
                        marker="o", ms=3, lw=1, capsize=1.5, color=PAL[k], label=nm)
        ax.plot([0.1, 0.55], [0.1, 0.55], ls="--", color="#888", lw=0.8)
        ax.set_title(C.period_label(s), fontsize=9)
        ax.set_xlabel("Заявленная вероятность Risk")
    axes[0].set_ylabel("Доля выигрышей")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Калибровка выбранного Risk до и после калибровки после выбора", fontsize=10)
    save(fig, C.FIGURES / "post_calibration.png")


def all_figures() -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    res = C.load_json("risk.json")
    for fn in (ev_decomp, flb, rare, odds_dep, strategies, fewer, post):
        fn(res, plt, PALETTE, save_chart)
