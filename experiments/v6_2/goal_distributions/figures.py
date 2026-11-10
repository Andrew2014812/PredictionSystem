"""Charts of the goal-distribution study (from results/*.json only)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import save_chart

from . import common as C

COLORS = ["#167c66", "#c88a12", "#3d6fb6", "#c0392b"]       # validated subset of the series palette
MARKERS = ["o", "s", "^", "D"]
MUTED = "#4b5a63"


def _label(v: str) -> str:
    if v == "P+R":
        return "Пуассон + согл. (текущий)"
    names = C.load_json("fits.json")["names"]
    return names.get(v.replace("+R", ""), v) + (" + согл." if v.endswith("+R") else "")


def fig_score(plt) -> None:
    comp = C.load_json("comparisons.json")["comparisons"]
    items = sorted(comp.items(), key=lambda kv: kv[1]["select"]["SCORE"]["diff"])
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(items) + 1.2))
    y = np.arange(len(items))
    d = np.array([c["select"]["SCORE"]["diff"] for _, c in items]) * 1e3
    lo = np.array([c["select"]["SCORE"]["ci95"][0] for _, c in items]) * 1e3
    hi = np.array([c["select"]["SCORE"]["ci95"][1] for _, c in items]) * 1e3
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color=COLORS[0], ms=5, capsize=2, lw=1.2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(k) for k, _ in items], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ log loss точного счёта к текущему (×10⁻³), 2023/24+2024/25; < 0 — лучше")
    save_chart(fig, C.FIGURES / "score_ll.png")


def fig_heat(plt) -> None:
    comp = C.load_json("comparisons.json")["comparisons"]
    keys = [("SCORE", "Счёт"), ("TOTALS", "Тоталы"), ("BTTS", "Обе заб."), ("AH", "Форы"), ("AH_WHOLE", "Целые форы"),
            ("1X2", "1X2"), ("TOT7_RPS", "RPS тотала"), ("GD7_RPS", "RPS разницы")]
    vs = sorted(comp, key=lambda v: comp[v]["select"]["SCORE"]["diff"])
    z = np.array([[comp[v]["select"][k]["diff"] * 1e3 for k, _ in keys] for v in vs])
    sig = np.array([[(comp[v]["select"][k]["ci95"][1] < 0) or (comp[v]["select"][k]["ci95"][0] > 0)
                     for k, _ in keys] for v in vs])
    lim = 2.0                                  # ×10⁻³; larger differences (Maher alone) are clipped
    fig, ax = plt.subplots(figsize=(8.4, 0.32 * len(vs) + 1.6))
    im = ax.imshow(np.clip(z, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:+.2f}" + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=6.2,
                    color="white" if abs(z[i, j]) > 0.6 * lim else "#14202a")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in keys], fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels([_label(v) for v in vs], fontsize=7.2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Δ ×10⁻³ (синий — лучше; шкала обрезана на ±2)")
    save_chart(fig, C.FIGURES / "markets_heat.png")


def fig_calibration(plt, variants: list[str]) -> None:
    cal = C.load_json("calibration.json")
    keys = [("TOTAL_2.5", "Тотал больше 2.5"), ("BTTS", "Обе забьют"), ("DRAW", "Ничья"), ("ZERO_ZERO", "Счёт 0-0")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, (k, title) in zip(axes.ravel(), keys):
        allp = []
        for i, v in enumerate(variants):
            pts = cal[v]["select"][k]
            p = [x["p"] for x in pts]
            o = [x["o"] for x in pts]
            allp += p + o
            ax.plot(p, o, marker=MARKERS[i], color=COLORS[i], lw=1.4, ms=4.5, label=_label(v))
        lo, hi = min(allp) * 0.95, max(allp) * 1.05
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("прогноз")
        ax.set_ylabel("факт")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    save_chart(fig, C.FIGURES / "calibration.png")


def fig_push(plt, variants: list[str]) -> None:
    cal = C.load_json("calibration.json")
    lines = list(cal[variants[0]]["select"]["push"])
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(lines))
    w = 0.8 / (len(variants) + 1)
    obs = [cal[variants[0]]["select"]["push"][l]["observed"] for l in lines]
    ax.bar(x - 0.4 + w / 2, obs, w, color="#9aa7ad", label="факт (доля возвратов)")
    for i, v in enumerate(variants):
        pr = [cal[v]["select"]["push"][l]["predicted"] for l in lines]
        ax.bar(x - 0.4 + w * (i + 1.5), pr, w, color=COLORS[i], label=_label(v))
    ax.set_xticks(x)
    ax.set_xticklabels([f"фора {l}" for l in lines])
    ax.set_ylabel("вероятность возврата")
    ax.legend(fontsize=7, ncol=2)
    save_chart(fig, C.FIGURES / "push.png")


def fig_excess(plt) -> None:
    t = C.load_json("tests.json")["excess"]
    cells = ["0-0", "1-1", "draw", "home_0", "away_0"]
    names = {"0-0": "0-0", "1-1": "1-1", "draw": "ничьи", "home_0": "хозяева 0", "away_0": "гости 0"}
    seasons = [str(s) for s in C.SEASONS]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(cells))
    w = 0.8 / len(seasons)
    for i, s in enumerate(seasons):
        ax.bar(x - 0.4 + w * (i + 0.5), [t[s][c]["ratio"] for c in cells], w, color=(COLORS + ["#5f7a85"])[i],
               label=C.period_label(int(s)))
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[c] for c in cells])
    ax.set_ylabel("факт / ожидание (Пуассон)")
    ax.set_ylim(0.85, 1.2)
    ax.legend(fontsize=7, ncol=4)
    save_chart(fig, C.FIGURES / "excess.png")


def fig_leagues(plt) -> None:
    lg = C.load_json("leagues.json")
    items = sorted(lg["leagues"].items(), key=lambda kv: kv[1]["diff"])
    fig, ax = plt.subplots(figsize=(8.2, 0.26 * len(items) + 1.2))
    y = np.arange(len(items))
    d = np.array([v["diff"] for _, v in items]) * 1e3
    lo = np.array([v["ci95"][0] for _, v in items]) * 1e3
    hi = np.array([v["ci95"][1] for _, v in items]) * 1e3
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color=COLORS[2], ms=4, capsize=2, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} (n={v['n']})" for k, v in items], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ log loss счёта ×10⁻³: {_label(lg['variant'])} − текущий, 2023/24+2024/25")
    save_chart(fig, C.FIGURES / "leagues.png")


def fig_seasons(plt, variants: list[str]) -> None:
    comp = C.load_json("comparisons.json")["comparisons"]
    seasons = [str(s) for s in C.SEASONS]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, v in enumerate(variants):
        d = np.array([comp[v][s]["SCORE"]["diff"] for s in seasons]) * 1e3
        lo = np.array([comp[v][s]["SCORE"]["ci95"][0] for s in seasons]) * 1e3
        hi = np.array([comp[v][s]["SCORE"]["ci95"][1] for s in seasons]) * 1e3
        x = np.arange(len(seasons)) + (i - len(variants) / 2) * 0.12
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt=MARKERS[i], color=COLORS[i], capsize=2, ms=5, label=_label(v))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([C.period_label(int(s)) for s in seasons])
    ax.set_ylabel("Δ log loss счёта ×10⁻³")
    ax.legend(fontsize=7)
    save_chart(fig, C.FIGURES / "seasons.png")


def build(top: list[str]) -> None:
    plt = C.plt()
    fig_score(plt)
    fig_heat(plt)
    fig_calibration(plt, ["P", "P+R"] + top[:2])
    fig_push(plt, ["P", "P+R"] + top[:2])
    fig_excess(plt)
    fig_leagues(plt)
    fig_seasons(plt, top[:4])
