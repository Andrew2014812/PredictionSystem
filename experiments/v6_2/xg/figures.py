"""Charts of the xG study (all values read from results/*.json and the caches written by run.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import models as M

FOREST = ["R0", "R0a", "X1", "G1", "X2", "X2a", "X3", "X3a", "X4", "X5", "M1", "M1a", "X2M", "X2Ma", "S0", "S1", "T0", "T1"]
KEY = ["R0a", "X2a", "X3a", "X5", "M1a", "X2Ma"]


def _forest(ax, met: dict, metric: str, title: str, scale: float = 1.0) -> None:
    from scripts.report_kit import ACCENT, NEG, PALETTE
    vs = [v for v in FOREST if v in met["vs_B0"]]
    y = np.arange(len(vs))[::-1]
    for yi, v in zip(y, vs):
        d = met["vs_B0"][v][metric]
        lo, hi = d["ci95"]
        col = NEG if lo is not None and lo > 0 else (ACCENT if hi is not None and hi < 0 else PALETTE[2])
        ax.errorbar(d["mean"] * scale, yi, xerr=[[(d["mean"] - lo) * scale], [(hi - d["mean"]) * scale]], fmt="o",
                    color=col, ms=4, capsize=2)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(vs)
    ax.set_title(title, fontsize=9.5)


def run() -> None:
    plt = C.plt()
    from scripts.report_kit import ACCENT, NEG, PALETTE, save_chart
    met = C.load_json("metrics.json")
    main = C.load_json("main.json")
    cov = C.load_json("features_coverage.json")
    sig = C.load_json("signal.json")
    info = C.load_json("variants_info.json")
    br = C.load_json("breakdowns.json")
    rel = C.load_json("reliability.json")

    # coverage at the season start
    bm = pd.DataFrame(cov["by_month"])
    bm = bm.loc[bm["matches"] >= 20]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(bm))
    ax.bar(x - 0.27, bm["ratings"], 0.27, label="рейтинги xG определены", color=PALETTE[2])
    ax.bar(x, bm["ge3"], 0.27, label="у обеих команд ≥ 3 матчей с xG", color=PALETTE[0])
    ax.bar(x + 0.27, bm["ge5"], 0.27, label="у обеих команд ≥ 5 матчей с xG", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({n} матчей)" for m, n in zip(bm["month"], bm["matches"])], fontsize=8)
    ax.set_ylabel("Доля сыгранных матчей")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Начало сезона: сколько истории xG есть до матча", fontsize=10)
    save_chart(fig, C.FIGURES / "coverage_season_start.png")

    # forest plots
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    _forest(axes[0], met, "goals_dev", "Отклонение Пуассона (голы обеих команд), разница с v6.1")
    _forest(axes[1], met, "logloss_1x2", "Log loss 1X2, разница с v6.1")
    for ax in axes:
        ax.set_xlabel("вариант − v6.1 (меньше = лучше), 95 % ДИ")
    save_chart(fig, C.FIGURES / "forest_main_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    _forest(axes[0], met, "ll_over2.5", "Log loss тотала 2.5, разница с v6.1")
    _forest(axes[1], met, "ll_exact", "Log loss точного счёта, разница с v6.1")
    for ax in axes:
        ax.set_xlabel("вариант − v6.1 (меньше = лучше), 95 % ДИ")
    save_chart(fig, C.FIGURES / "forest_markets.png")

    # cumulative difference over time
    pm = pd.read_parquet(C.CACHE / "per_match.parquet", columns=["match_id", "variant", "goals_dev", "logloss_1x2"])
    dates = M.frame(0).set_index("match_id")["date"]
    base = pm.loc[pm["variant"] == "B0"].set_index("match_id")
    order = base.index[np.argsort(dates.loc[base.index].to_numpy(), kind="stable")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k, v in enumerate(KEY):
        d = pm.loc[pm["variant"] == v].set_index("match_id").loc[order]
        b = base.loc[order]
        for ax, col in zip(axes, ("goals_dev", "logloss_1x2")):
            ax.plot(np.cumsum(d[col].to_numpy() - b[col].to_numpy()), lw=1.1, color=PALETTE[k % len(PALETTE)], label=v)
    for ax, t in zip(axes, ("Накопленная разница отклонения голов", "Накопленная разница log loss 1X2")):
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xlabel("Матч № (по дате)")
        ax.set_title(t + " (вариант − v6.1)", fontsize=9)
    axes[0].legend(fontsize=7)
    save_chart(fig, C.FIGURES / "cumulative_difference.png")

    # coefficient paths
    co = info["in_season"]["coefs"]
    weeks = [w["week"][5:] for w in info["in_season"]["weeks"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(weeks, co["X5_w"], marker="o", color=PALETTE[0])
    axes[0].set_title("X5: вес рейтинга xG w по неделям", fontsize=9.5)
    axes[0].set_ylim(0, max(0.6, max(co["X5_w"]) * 1.1))
    names = ["a", "b−1 (лог λ v6.1)", "рейтинг xG", "xG за (сжат.)", "xG против соп.", "голы − xG"]
    arr = np.array(co["X2a_h"])
    for j in range(arr.shape[1]):
        axes[1].plot(weeks, arr[:, j], marker=".", lw=1, label=names[j], color=PALETTE[j % len(PALETTE)])
    axes[1].axhline(0, color="#555", lw=0.7)
    axes[1].set_title("X2a (голы хозяев): коэффициенты по неделям", fontsize=9.5)
    axes[1].legend(fontsize=6.5, ncol=2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    save_chart(fig, C.FIGURES / "coefficient_paths.png")

    # Main: over-confidence and ROI
    vs = ["B0"] + [v for v in FOREST if v in main["summary"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    y = np.arange(len(vs))[::-1]
    for yi, v in zip(y, vs):
        s = main["summary"][v]
        lo, hi = s["bet_gap_ci95"]
        axes[0].errorbar(s["bet_gap"] * 100, yi, xerr=[[(s["bet_gap"] - lo) * 100], [(hi - s["bet_gap"]) * 100]],
                         fmt="o", ms=4, capsize=2, color=ACCENT if v == "B0" else PALETTE[2])
        lo, hi = s["roi_ci95"]
        axes[1].errorbar(s["roi"] * 100, yi, xerr=[[(s["roi"] - lo) * 100], [(hi - s["roi"]) * 100]],
                         fmt="o", ms=4, capsize=2, color=ACCENT if v == "B0" else PALETTE[2])
    axes[0].axvline(0, color=NEG, lw=0.8, ls="--")
    axes[1].axvline(0, color=NEG, lw=0.8, ls="--")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(vs)
    axes[0].set_xlabel("заявленная вероятность − доля выигрышей, п.п.")
    axes[1].set_xlabel("ROI при реальных ценах, %")
    axes[0].set_title("Переоценка выбранного Main (ставки с ценой)", fontsize=9.5)
    axes[1].set_title("ROI Main, 95 % ДИ", fontsize=9.5)
    save_chart(fig, C.FIGURES / "main_gap_roi.png")

    # reliability
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for k, v in enumerate([v for v in ("B0", "X5", "X3a", "X2Ma") if v in rel]):
        r = rel[v]
        axes[0].plot([b["predicted"] for b in r["1x2"]], [b["observed"] for b in r["1x2"]], marker="o", ms=3, lw=1,
                     color=PALETTE[k], label=v)
        axes[1].plot([b["predicted"] for b in r["main"]], [b["observed"] for b in r["main"]], marker="o", ms=3, lw=1,
                     color=PALETTE[k], label=v)
    for ax, t in zip(axes, ("1X2 (H, D, A вместе)", "Выбранный Main")):
        lim = ax.get_xlim()
        ax.plot(lim, lim, color="#888", ls="--", lw=0.8)
        ax.set_xlabel("Заявленная вероятность")
        ax.set_ylabel("Наблюдаемая частота")
        ax.set_title("Надёжность: " + t, fontsize=9.5)
        ax.legend(fontsize=7)
    save_chart(fig, C.FIGURES / "reliability.png")

    # signal
    rows = [r for r in sig["features"] if r["resid"]["r"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        rr = r["resid"]
        ax.errorbar(rr["r"], yi, xerr=[[rr["r"] - rr["lo"]], [rr["hi"] - rr["r"]]], fmt="o", ms=4, capsize=2,
                    color=PALETTE[0] if r["kind"] == "xG" else PALETTE[1])
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']} → {r['target']}" for r in rows], fontsize=7.5)
    ax.set_xlabel("корреляция с остатком v6.1 (95 % ДИ); зелёный — xG, жёлтый — признаки v6.1")
    ax.set_title("Чего не знает v6.1: связь признаков с ошибкой базовой модели", fontsize=9.5)
    save_chart(fig, C.FIGURES / "signal_residuals.png")

    # league breakdown for X5 and X3a
    lg = pd.DataFrame(br["league"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    y = np.arange(len(lg))[::-1]
    for ax, col, t in zip(axes, ("dgoals", "dll"), ("отклонение голов", "log loss 1X2")):
        for k, v in enumerate(("X5", "X3a")):
            vals = lg[f"{v}_{col}"]
            m = np.array([d["mean"] for d in vals])
            lo = np.array([d["ci95"][0] for d in vals], float)
            hi = np.array([d["ci95"][1] for d in vals], float)
            ax.errorbar(m, y + (0.15 if k else -0.15), xerr=[m - lo, hi - m], fmt="o", ms=3, capsize=1.5,
                        color=PALETTE[k], label=v)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title(f"По лигам: {t} (вариант − v6.1)", fontsize=9.5)
        ax.legend(fontsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{g} ({n})" for g, n in zip(lg["group"], lg["n"])], fontsize=7.5)
    save_chart(fig, C.FIGURES / "league_breakdown.png")

    # sample size
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vs = [v for v in FOREST if v in met["vs_B0"]]
    x = np.arange(len(vs))
    for k, (m, lab) in enumerate((("goals_dev", "отклонение голов"), ("logloss_1x2", "log loss 1X2"))):
        n = [met["vs_B0"][v][m]["n_required"] or np.nan for v in vs]
        ax.scatter(x + (k - 0.5) * 0.25, n, color=PALETTE[k], s=18, label=lab)
    ax.axhline(met["n_matches"], color=NEG, ls="--", lw=0.8, label=f"сейчас: {met['n_matches']} матчей")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(vs, rotation=45, fontsize=8)
    ax.set_ylabel("Матчей для мощности 80 %")
    ax.legend(fontsize=7.5)
    ax.set_title("Сколько матчей нужно, чтобы наблюдаемая разница стала значимой", fontsize=9.5)
    save_chart(fig, C.FIGURES / "sample_size.png")


if __name__ == "__main__":
    C.setup_logging("figures")
    run()
