"""Step 5 — figures of the draw research (PNG in figures/, Russian labels, the report-kit style)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_kit import ACCENT, AMBER, INK, MUTED, PALETTE, plt, save_chart
from src import config

from . import blocks as B
from . import common as C
from . import history as Hm
from . import probs as PR
from .rules import CURRENT, current
from src.modelling.decision import decide

OLD, NEW = "#9aa5ab", ACCENT
LAB = ["П1", "Х", "П2"]


def _cm(ax, cm, title):
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i][j] > cm.max() * 0.6 else INK)
    ax.set_xticks(range(3), LAB), ax.set_yticks(range(3), LAB)
    ax.set_xlabel("Названный исход"), ax.set_ylabel("Фактический исход"), ax.grid(False)
    ax.set_title(title, fontsize=8.5)


def confusion(rules: dict) -> None:
    sets = [("block_2023", "2023/24 (отбор)"), ("block_2024", "2024/25 (отбор)"),
            ("page_backtest_2025", "2025/26, backtest_2526\n(страница Model Analysis)"),
            ("history_2026", "2026/27, история\nпродакшна")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for col, (name, label) in enumerate(sets):
        v = rules["per_set"][name]
        _cm(axes[0, col], v["argmax"]["confusion"], f"argmax — {label}")
        _cm(axes[1, col], v["current"]["confusion"], f"правило v6 — {label}")
    save_chart(fig, C.FIGURES / "confusion_old_vs_current.png")


def season_bars(rules: dict) -> None:
    sets = ["block_2023", "block_2024", "block_2025", "page_backtest_2025", "block_2026", "history_2026"]
    labels = ["2023/24", "2024/25", "2025/26", "2025/26\nстраница", "2026/27\nблок", "2026/27\nистория"]
    metrics = [("macro_f1", "Macro F1"), ("draw_recall", "Полнота ничьих"), ("draw_precision", "Точность ничьих"),
               ("accuracy", "Accuracy")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    x = np.arange(len(sets))
    for ax, (k, title) in zip(axes, metrics):
        a = [rules["per_set"][s]["argmax"][k] for s in sets]
        c = [rules["per_set"][s]["current"][k] for s in sets]
        ax.bar(x - 0.2, a, 0.38, color=OLD, label="argmax (v1–v5)")
        ax.bar(x + 0.2, c, 0.38, color=NEW, label="правило v6")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(title)
    axes[0].legend(fontsize=7, loc="lower left")
    save_chart(fig, C.FIGURES / "metrics_by_season.png")


def pdraw_distribution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    frames = {"2023/24–2025/26 (walk-forward)": pd.concat([B.block(s)[0] for s in (2023, 2024, 2025)]),
              "2026/27 (история продакшна)": Hm.history()}
    bins = np.arange(0.10, 0.42, 0.01)
    for k, (label, fr) in enumerate(frames.items()):
        axes[0].hist(fr["pD"], bins=bins, density=True, histtype="step", lw=1.8, color=PALETTE[k * 2], label=label)
    for v, txt, col in ((CURRENT["t"], "порог правила 0.28", ACCENT), (config.MAIN_MIN_PROBABILITY, "нижняя граница Main 0.40", "#c0392b")):
        axes[0].axvline(v, ls="--", color=col, lw=1.2)
        axes[0].text(v + 0.003, axes[0].get_ylim()[1] * 0.9, txt, fontsize=7, color=col, rotation=90, va="top")
    axes[0].set_xlabel("P(ничья)"), axes[0].set_ylabel("Плотность"), axes[0].set_xlim(0.10, 0.43)
    axes[0].set_title("Распределение P(ничья)")
    axes[0].legend(fontsize=7, loc="upper left")
    fr = frames["2023/24–2025/26 (walk-forward)"]
    gap = np.abs(fr["pH"] - fr["pA"])
    edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0]
    g = pd.cut(gap, edges, include_lowest=True)
    t = fr.assign(g=g, d=(fr["y"] == 1)).groupby("g", observed=True).agg(act=("d", "mean"), pd_=("pD", "mean"), n=("d", "size"))
    xs = np.arange(len(t))
    axes[1].bar(xs - 0.2, t["act"], 0.38, color=OLD, label="фактическая доля ничьих")
    axes[1].bar(xs + 0.2, t["pd_"], 0.38, color=NEW, label="средняя P(ничья)")
    axes[1].set_xticks(xs, [f"{max(iv.left, 0):.2f}–{iv.right:.2f}\nn={n}" for iv, n in zip(t.index, t["n"])], fontsize=6.5)
    axes[1].set_xlabel("|P(П1) − P(П2)|"), axes[1].set_title("Ничьи по близости сил (2023/24–2025/26)")
    axes[1].legend(fontsize=7)
    save_chart(fig, C.FIGURES / "p_draw_distribution.png")


def variants(probs: dict) -> None:
    names = [n for n in probs["folds"]["2023"]["all_rows"] if n not in ("a_baseline", "naive_league")]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    y = np.arange(len(names))
    for k, s in enumerate(C.FOLDS):
        d = [probs["folds"][str(s)]["all_rows"][n]["delta_log_loss"] for n in names]
        m = np.array([v["mean_diff"] for v in d])
        lo = m - np.array([v["ci95"][0] for v in d])
        hi = np.array([v["ci95"][1] for v in d]) - m
        ax.errorbar(m, y + (k - 1.5) * 0.18, xerr=[lo, hi], fmt="o", ms=3.5, lw=1, color=PALETTE[k],
                    label=f"{C.season_label(s)} ({C.ROLE[s]})")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [PR.TITLES[n] for n in names], fontsize=7.5)
    ax.set_xlabel("Δ log loss к базовой модели (меньше нуля — лучше), 95 % бутстрэп-интервал")
    ax.set_xlim(-0.004, 0.011)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Варианты вероятностей: Δ log loss по сезонам (наивная частота лиги: +0.06…+0.07, вне шкалы)")
    save_chart(fig, C.FIGURES / "variants_delta_log_loss.png")


def reliability(probs: dict) -> None:
    show = ["a_baseline", "s2_dixon_coles", "d_draw_xgb_blend", "e_stack"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, s in zip(axes, C.FOLDS):
        fold = probs["folds"][str(s)]
        for k, n in enumerate(show):
            r = fold["all_rows"][n]["reliability_draw"]
            ax.plot([q["predicted"] for q in r], [q["observed"] for q in r], "o-", ms=3, lw=1.4, color=PALETTE[k], label=PR.TITLES[n])
        r = fold["odds_rows"]["bk_proportional"]["reliability_draw"]
        ax.plot([q["predicted"] for q in r], [q["observed"] for q in r], "s--", ms=3, lw=1.2, color=MUTED, label="букмекер Bet365")
        ax.plot([0.1, 0.45], [0.1, 0.45], ":", color="grey")
        ax.set_xlim(0.1, 0.45), ax.set_ylim(0.1, 0.45)
        ax.set_title(f"{C.season_label(s)} ({C.ROLE[s]})", fontsize=9)
        ax.set_xlabel("P(ничья)")
    axes[0].set_ylabel("Фактическая доля ничьих")
    axes[0].legend(fontsize=6.5, loc="upper left")
    save_chart(fig, C.FIGURES / "draw_reliability.png")


def rule_heatmap(rules: dict) -> None:
    grid = [r for r in rules["rule_search"]["a_baseline"]["grid"] if r["params"]["kind"] == "threshold"]
    ts = sorted({r["params"]["t"] for r in grid})
    gs = sorted({r["params"]["g"] for r in grid})
    f1 = np.full((len(gs), len(ts)), np.nan)
    share = np.full_like(f1, np.nan)
    for r in grid:
        i, j = gs.index(r["params"]["g"]), ts.index(r["params"]["t"])
        f1[i, j], share[i, j] = r["pooled"]["macro_f1"], r["pooled"]["predicted_draw_share"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(f1, cmap="Greens", aspect="auto", origin="lower")
    for i in range(len(gs)):
        for j in range(len(ts)):
            over = share[i, j] > rules["draw_share_cap"]
            ax.text(j, i, f"{f1[i, j]:.3f}" + ("*" if over else ""), ha="center", va="center", fontsize=6.3,
                    color="white" if f1[i, j] > np.nanmax(f1) - 0.008 else (MUTED if over else INK))
    j, i = ts.index(CURRENT["t"]), gs.index(CURRENT["g"])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#c0392b", lw=2))
    ax.set_xticks(range(len(ts)), [f"{t:.2f}" for t in ts]), ax.set_yticks(range(len(gs)), [f"{g:.2f}" for g in gs])
    ax.set_xlabel("порог t для P(ничья)"), ax.set_ylabel("макс. разрыв g = |P(П1) − P(П2)|"), ax.grid(False)
    ax.set_title("Macro F1 на объединённых сезонах отбора 2023/24 + 2024/25 (рамка — текущее правило; * — доля ничьих выше допустимой)",
                 fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.03)
    save_chart(fig, C.FIGURES / "rule_grid_macro_f1.png")


def main_score(main: dict) -> None:
    from src.markets.recommendation import odds_preference
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    o = np.linspace(1.05, 6, 300)
    axes[0].plot(o, [odds_preference(v) for v in o], color=INK, lw=1.8)
    fr = B.block(2024)[0]
    q = np.nanquantile(fr["odds_draw"], [0.05, 0.95])
    axes[0].axvspan(q[0], q[1], color=AMBER, alpha=0.18, label="90 % цен на ничью (2024/25)")
    axes[0].axvspan(*config.MAIN_PREFERRED_ODDS, color=ACCENT, alpha=0.12, label="предпочтительное окно Main 1.40–2.50")
    axes[0].set_xlabel("Коэффициент"), axes[0].set_ylabel("odds_preference"), axes[0].legend(fontsize=7)
    axes[0].set_title("Множитель коэффициента в оценке Main")
    names = ["block_2023", "block_2024", "block_2025", "block_2026", "history_2026"]
    lab = ["2023/24", "2024/25", "2025/26", "2026/27\nблок", "2026/27\nистория"]
    w = main["why_not_main"]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, [w[n]["main_score_mean"] for n in names], 0.38, color=NEW, label="оценка выбранного Main")
    axes[1].bar(x + 0.2, [w[n]["draw_score_ungated_mean"] for n in names], 0.38, color=OLD,
                label="оценка ничьей без порога 0.40")
    axes[1].set_xticks(x, lab, fontsize=7.5), axes[1].legend(fontsize=7)
    axes[1].set_title("Средняя оценка правила Main")
    save_chart(fig, C.FIGURES / "main_score_draw.png")


def draw_roi(main: dict) -> None:
    names = ["block_2023", "block_2024", "block_2025", "block_2026", "history_2026"]
    lab = ["2023/24", "2024/25", "2025/26", "2026/27 блок", "2026/27 история"]
    bins = list(main["draw_bets"]["block_2023"]["by_ev"])
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(len(bins))
    for k, n in enumerate(names):
        r = [main["draw_bets"][n]["by_ev"][b] for b in bins]
        m = np.array([v["roi"] if v["roi"] is not None else np.nan for v in r])
        lo = np.array([v["ci95"][0] if v["ci95"][0] is not None else np.nan for v in r])
        hi = np.array([v["ci95"][1] if v["ci95"][1] is not None else np.nan for v in r])
        ax.errorbar(x + (k - 2) * 0.13, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5, lw=1, color=PALETTE[k], label=lab[k])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"EV {b}" for b in bins], fontsize=7)
    ax.set_ylabel("ROI ставки на ничью (95 % CI)")
    ax.set_ylim(-0.8, 0.8)
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    ax.set_title("Ставка 1 на ничью по реальной предматчевой цене: ROI по корзинам EV модели")
    save_chart(fig, C.FIGURES / "draw_roi_by_ev.png")


def policies(main: dict) -> None:
    g = main["policies"]["grid"]
    keys = list(g)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    xs = [g[k]["pooled_selection"]["delta_main_roi"] * 100 for k in keys]
    ys = [g[k]["history_2026"]["delta_main_roi"] * 100 for k in keys]
    sz = [max(g[k]["pooled_selection"]["draw_bets"], 1) ** 0.5 * 2 for k in keys]
    ax.scatter(xs, ys, s=sz, color=ACCENT, alpha=0.6, edgecolor="white", lw=0.5)
    best = main["policies"]["best"]
    if best:
        ax.scatter([g[best]["pooled_selection"]["delta_main_roi"] * 100], [g[best]["history_2026"]["delta_main_roi"] * 100],
                   s=90, facecolor="none", edgecolor="#c0392b", lw=1.6, label=f"лучшая на отборе: {best}")
        ax.legend(fontsize=7)
    ax.axhline(0, color=INK, lw=0.7), ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel("Δ ROI Main, п.п. — сезоны отбора 2023/24 + 2024/25")
    ax.set_ylabel("Δ ROI Main, п.п. — 2026/27 история")
    ax.set_title("Политики «ничья вместо Main» (размер точки ~ число ставок на ничью)")
    save_chart(fig, C.FIGURES / "draw_main_policies.png")


def leagues(rules: dict) -> None:
    lg = rules["by_league"]["blocks_2023_2025"]
    names = sorted(lg, key=lambda k: -lg[k]["current"]["actual_draw_share"])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.27, [lg[n]["current"]["actual_draw_share"] for n in names], 0.26, color=OLD, label="фактическая доля ничьих")
    ax.bar(x, [lg[n]["current"]["draw_recall"] for n in names], 0.26, color=NEW, label="полнота ничьих (правило v6)")
    ax.bar(x + 0.27, [lg[n]["current"]["draw_precision"] for n in names], 0.26, color=AMBER, label="точность ничьих (правило v6)")
    ax.set_xticks(x, names, fontsize=7.5)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("По лигам, 2023/24–2025/26 (walk-forward)")
    save_chart(fig, C.FIGURES / "leagues_draws.png")


def run() -> None:
    rules, probs, main = C.load_json("rules.json"), C.load_json("probabilities.json"), C.load_json("draw_main.json")
    confusion(rules)
    season_bars(rules)
    pdraw_distribution()
    variants(probs)
    reliability(probs)
    rule_heatmap(rules)
    main_score(main)
    draw_roi(main)
    policies(main)
    leagues(rules)
    _ = (decide, current)


if __name__ == "__main__":
    run()
