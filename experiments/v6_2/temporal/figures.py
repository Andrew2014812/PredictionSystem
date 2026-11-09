"""Charts of the temporal-features report (read from results/*.json only)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import ACCENT, AMBER, MUTED, NEG, PALETTE, save_chart

from . import common as C


def forest(var: dict) -> None:
    plt = C.plt()
    V = var["variants"]
    names = list(V)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 5.0), sharey=True)
    for ax, m, title in ((axs[0], "d_ll", "log loss 1X2"), (axs[1], "d_dev", "девиация голов")):
        for i, v in enumerate(names):
            d = V[v]["select"][m]
            x, lo, hi = d["diff"] * 1e3, d["ci95"][0] * 1e3, d["ci95"][1] * 1e3
            ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", ms=4, lw=1,
                        color=ACCENT if v.startswith("T") else AMBER)
            for k, s in enumerate(C.SELECT):
                ax.scatter(V[v]["seasons"][str(s)][m]["diff"] * 1e3, i + 0.2 * (2 * k - 1), marker="|", s=40, color=MUTED)
        ax.axvline(0, color=NEG, lw=0.8)
        ax.set_xlabel(f"{title}: вариант − T0, ×10⁻³")
    axs[0].set_yticks(range(len(names)), [f"{v}: {V[v]['name']}" for v in names], fontsize=7.5)
    axs[0].invert_yaxis()
    fig.suptitle("Временные признаки, 2023/24 + 2024/25 (< 0 — лучше производственного набора)", fontsize=10)
    save_chart(fig, C.FIGURES / "variants_forest.png")


def seasons(var: dict) -> None:
    plt = C.plt()
    V = var["variants"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    seas = [str(s) for s in C.EVAL]
    for i, v in enumerate(V):
        y = [V[v]["seasons"][s]["d_ll"]["diff"] * 1e3 for s in seas]
        ax.plot(range(len(seas)), y, marker="o", lw=1, ms=3, color=PALETTE[i % len(PALETTE)],
                ls="-" if v.startswith("T") else "--", label=v)
    noise = [var["seed_noise"][s]["d_ll"]["diff"] * 1e3 for s in seas]
    ax.plot(range(len(seas)), noise, color="black", lw=1.6, label="3 сида − сид 42")
    ax.axhline(0, color=NEG, lw=0.8)
    ax.set_xticks(range(len(seas)), [C.short_period(int(s)) for s in seas])
    ax.set_ylabel("Δ log loss 1X2 к T0, ×10⁻³")
    ax.legend(ncol=5, fontsize=7)
    save_chart(fig, C.FIGURES / "variants_by_season.png")


def phase(var: dict) -> None:
    plt = C.plt()
    V = var["variants"]
    names = list(V)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for k, (lab, col) in enumerate((("early", ACCENT), ("late", AMBER))):
        for i, v in enumerate(names):
            d = V[v]["select"][f"d_ll_{lab}"]
            x, lo, hi = d["diff"] * 1e3, d["ci95"][0] * 1e3, d["ci95"][1] * 1e3
            ax.errorbar(x, i + 0.15 * (2 * k - 1), xerr=[[x - lo], [hi - x]], fmt="o", ms=3, lw=1, color=col,
                        label=("начало сезона (≤ 5 игр у обеих)" if lab == "early" else "остальные матчи") if i == 0 else None)
    ax.axvline(0, color=NEG, lw=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Δ log loss 1X2 к T0, ×10⁻³ (2023/24 + 2024/25)")
    ax.legend(fontsize=8)
    save_chart(fig, C.FIGURES / "variants_phase.png")


def signal(sig: dict) -> None:
    plt = C.plt()
    S = sig["select"]["features"]
    top = sorted(S, key=lambda f: -abs(S[f]["r_model"]["rho"] or 0))[:22]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    y = np.arange(len(top))
    for k, (t, col, lab) in enumerate((("r_model", ACCENT, "остаток модели"), ("r_market", AMBER, "остаток рынка (открытие)"))):
        r = np.array([S[f][t]["rho"] for f in top])
        lo = np.array([S[f][t]["ci95"][0] for f in top])
        hi = np.array([S[f][t]["ci95"][1] for f in top])
        ax.errorbar(r, y + 0.15 * (2 * k - 1), xerr=[r - lo, hi - r], fmt="o", ms=3, lw=1, color=col, label=lab)
    ax.axvline(0, color=NEG, lw=0.8)
    ax.set_yticks(y, top, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("корреляция Пирсона признака с остатком очков хозяев (95 % ДИ, без поправки)")
    ax.legend(fontsize=8)
    ax.set_title("22 признака с наибольшей |r| к остатку модели, 2023/24 + 2024/25", fontsize=9)
    save_chart(fig, C.FIGURES / "signal.png")
    dec = sig["select"]["deciles"]
    fig, axs = plt.subplots(1, len(dec), figsize=(9.0, 3.4), sharey=True)
    for ax, (feat, rows) in zip(axs, dec.items()):
        x = np.arange(len(rows))
        for k, (t, col) in enumerate((("r_model", ACCENT), ("r_market", AMBER))):
            m = np.array([r[t]["mean"] for r in rows])
            lo = np.array([r[t]["ci95"][0] for r in rows])
            hi = np.array([r[t]["ci95"][1] for r in rows])
            ax.errorbar(x + 0.12 * (2 * k - 1), m, yerr=[m - lo, hi - m], fmt="o", ms=3, lw=1, color=col,
                        label={"r_model": "модель", "r_market": "рынок"}[t])
        ax.axhline(0, color=NEG, lw=0.8)
        ax.set_title(feat, fontsize=8.5)
        ax.set_xlabel("дециль признака")
    axs[0].set_ylabel("очки хозяев − ожидание")
    axs[0].legend(fontsize=7)
    save_chart(fig, C.FIGURES / "signal_deciles.png")


def run() -> None:
    var = C.load_json("variants.json")
    sig = C.load_json("signal.json")
    if var:
        forest(var)
        seasons(var)
        phase(var)
    if sig:
        signal(sig)
