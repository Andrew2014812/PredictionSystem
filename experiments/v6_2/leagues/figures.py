"""Charts of the league-specialisation report (read from results/*.json only)."""
from __future__ import annotations

import numpy as np

from scripts.report_kit import ACCENT, AMBER, MUTED, NEG, PALETTE, save_chart

from . import common as C


def _order(d: dict, key) -> list:
    return sorted(d, key=key)


def forest_market(perf: dict) -> None:
    plt = C.plt()
    L = perf["select"]["leagues"]
    meta = perf["select"]["meta"]["by_group"]
    lg = _order(L, lambda x: L[x]["d"]["diff"])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(lg))
    m = np.array([L[x]["d"]["diff"] for x in lg]) * 1e3
    lo = np.array([L[x]["d"]["ci95"][0] for x in lg]) * 1e3
    hi = np.array([L[x]["d"]["ci95"][1] for x in lg]) * 1e3
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="o", color=ACCENT, ms=4, lw=1, label="лига (95 % ДИ)")
    ax.scatter([meta[x]["eb"] * 1e3 for x in lg], y, marker="D", s=16, color=AMBER, zorder=3,
               label="после EB-сжатия")
    ax.axvline(perf["select"]["meta"]["mu"] * 1e3, color=MUTED, ls="--", lw=1, label="среднее (RE)")
    ax.axvline(0, color=NEG, lw=0.8)
    ax.set_yticks(y, [f"{x} (n={L[x]['n']})" for x in lg])
    ax.set_xlabel("log loss модели − рынка, ×10⁻³ (> 0: рынок точнее)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Модель против Bet365 (открытие) по лигам, 2023/24 + 2024/25")
    save_chart(fig, C.FIGURES / "league_vs_market.png")


def persistence(perf: dict) -> None:
    plt = C.plt()
    S = perf["seasons"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    a, b = S["2023"]["leagues"], S["2024"]["leagues"]
    for lg in a:
        ax.scatter(a[lg]["d"]["diff"] * 1e3, b[lg]["d"]["diff"] * 1e3, color=PALETTE[C.tier(lg) - 1], s=20)
        ax.annotate(lg, (a[lg]["d"]["diff"] * 1e3, b[lg]["d"]["diff"] * 1e3), fontsize=7, xytext=(3, 2),
                    textcoords="offset points")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    r = perf["persistence"]["2023-2024"]
    ax.set_xlabel("2023/24: модель − рынок, ×10⁻³")
    ax.set_ylabel("2024/25: модель − рынок, ×10⁻³")
    ax.set_title(f"Повторяемость отставания по лигам (Спирмен ρ = {r['rho']:.2f}, p = {r['p']:.2f})")
    save_chart(fig, C.FIGURES / "league_persistence.png")


def main_by_league(perf: dict) -> None:
    plt = C.plt()
    L = perf["select"]["leagues"]
    lg = _order(L, lambda x: L[x]["main"]["roi"]["mean"] or 0)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 6.0), sharey=True)
    y = np.arange(len(lg))
    roi = np.array([L[x]["main"]["roi"]["mean"] for x in lg]) * 100
    lo = np.array([L[x]["main"]["roi"]["ci95"][0] for x in lg]) * 100
    hi = np.array([L[x]["main"]["roi"]["ci95"][1] for x in lg]) * 100
    axs[0].errorbar(roi, y, xerr=[roi - lo, hi - roi], fmt="o", color=ACCENT, ms=4, lw=1)
    axs[0].axvline(0, color=NEG, lw=0.8)
    axs[0].set_xlabel("ROI Main, % (95 % ДИ)")
    axs[0].set_yticks(y, [f"{x} ({L[x]['main']['bets']})" for x in lg])
    gap = np.array([L[x]["main"]["gap"] for x in lg]) * 100
    glo = np.array([L[x]["main"]["gap_ci"][0] for x in lg]) * 100
    ghi = np.array([L[x]["main"]["gap_ci"][1] for x in lg]) * 100
    axs[1].errorbar(gap, y, xerr=[gap - glo, ghi - gap], fmt="s", color=AMBER, ms=4, lw=1)
    axs[1].axvline(0, color=MUTED, lw=0.8)
    axs[1].set_xlabel("переоценка Main, п.п. (заявлено − факт)")
    fig.suptitle("Main по лигам, 2023/24 + 2024/25 (в скобках — число ставок с ценой)", fontsize=10)
    save_chart(fig, C.FIGURES / "main_by_league.png")


def calibration_slopes(perf: dict, fits: dict) -> None:
    plt = C.plt()
    L = perf["select"]["leagues"]
    lg = _order(L, lambda x: L[x]["slope"])
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    y = np.arange(len(lg))
    b = np.array([L[x]["slope"] for x in lg])
    se = np.array([L[x]["slope_se"] or np.nan for x in lg])
    ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=ACCENT, ms=4, lw=1, label="наклон b по лиге (2023/24+2024/25)")
    f = fits.get(str(C.INFO), {}).get("calibration", {})
    if f:
        ax.scatter([f["league_eb"][x][0] for x in lg], y, marker="D", s=16, color=AMBER, zorder=3,
                   label="карта C3 для 2025/26 (EB, история)")
    ax.axvline(1, color=NEG, lw=0.8)
    ax.set_yticks(y, lg)
    ax.set_xlabel("наклон калибровки b (< 1: модель слишком уверена, > 1: недостаточно)")
    ax.legend(fontsize=8, loc="lower right")
    save_chart(fig, C.FIGURES / "calibration_slopes.png")


def profile_heatmap(prof: dict) -> None:
    plt = C.plt()
    from .profiles import VARS
    a = prof["all_history"]
    rows = {r["league"]: r for r in a["profile"]}
    lg = sorted(rows, key=lambda x: (a["groups"][x], -rows[x]["goals"]))
    X = np.array([[rows[x][v] for v in VARS] for x in lg], float)
    Z = (X - X.mean(0)) / X.std(0)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(VARS)), list(VARS.values()), rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(lg)), [f"{a['groups'][x]} · {x}" for x in lg], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-оценка по лигам")
    ax.set_title("Профили лиг (все сезоны 2021/22–2026/27) и группы")
    save_chart(fig, C.FIGURES / "profiles_heatmap.png")


def variants_forest(var: dict) -> None:
    plt = C.plt()
    V = var["variants"]
    names = list(V)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for i, v in enumerate(names):
        m = V[v]["metric"]
        d = V[v]["select"][f"d_{m}"]
        x, lo, hi = d["diff"] * 1e3, d["ci95"][0] * 1e3, d["ci95"][1] * 1e3
        col = AMBER if m == "dev" else (ACCENT if V[v]["ref"] == "B0" else PALETTE[2])
        ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", color=col, ms=4, lw=1)
        for k, s in enumerate(C.SELECT):
            ax.scatter(V[v]["seasons"][str(s)][f"d_{m}"]["diff"] * 1e3, i + 0.18 * (k * 2 - 1), marker="|", s=40,
                       color=MUTED)
    ax.axvline(0, color=NEG, lw=0.8)
    ax.set_yticks(range(len(names)), [f"{v} ({'девиация голов' if V[v]['metric'] == 'dev' else 'log loss 1X2'})"
                                      for v in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("разница с эталоном, ×10⁻³ (< 0: вариант лучше); штрихи — отдельные сезоны отбора")
    ax.set_title("Варианты специализации, 2023/24 + 2024/25")
    save_chart(fig, C.FIGURES / "variants_forest.png")


def transfer(tr: dict) -> None:
    plt = C.plt()
    P = tr["pooled"]
    kinds = [("own", "только своя лига (M2)"), ("all", "все другие лиги"), ("group", "только своя группа"),
             ("out", "только другие группы")]
    lg = sorted(P["all"]["leagues"], key=lambda x: P["all"]["leagues"][x]["diff"] or 0)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    y = np.arange(len(lg))
    for k, (key, lab) in enumerate(kinds):
        vals = [(P[key]["leagues"].get(x, {}).get("diff") or np.nan) * 1e3 for x in lg]
        ax.scatter(vals, y + (k - 1.5) * 0.17, s=14, color=PALETTE[k], label=lab)
    ax.axvline(0, color=NEG, lw=0.8)
    ax.set_yticks(y, lg)
    ax.set_xlabel("log loss − B0 (глобальная модель со своей историей), ×10⁻³")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Перенос между лигами: 1X2, 2023/24 + 2024/25")
    save_chart(fig, C.FIGURES / "transfer.png")


def league_size(var: dict, prof: dict) -> None:
    """Per-league effect of the specialising variants against the league's training size (fold 2024/25)."""
    plt = C.plt()
    n = {r["league"]: r["n"] for r in prof["folds"]["2024"]["profile"]}
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for k, v in enumerate(("C2", "C3", "M2", "M4a", "M5")):
        if v not in var["variants"]:
            continue
        L = var["variants"][v]["select"]["leagues"]
        lg = sorted(L)
        ax.scatter([n[x] for x in lg], [L[x]["mean"] * 1e3 for x in lg], s=16, color=PALETTE[k], label=v)
    ax.axhline(0, color=NEG, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("матчей лиги в обучении до 2024/25 (лог. шкала)")
    ax.set_ylabel("Δ к эталону по лиге, ×10⁻³ (отбор)")
    ax.legend(fontsize=8, ncol=5)
    ax.set_title("Специализация и объём данных лиги", fontsize=10)
    save_chart(fig, C.FIGURES / "league_size.png")


def run() -> None:
    perf = C.load_json("perf.json")
    prof = C.load_json("profiles.json")
    var = C.load_json("variants.json")
    fits = C.load_json("variant_fits.json", {})
    tr = C.load_json("transfer.json")
    if perf:
        forest_market(perf)
        persistence(perf)
        main_by_league(perf)
        calibration_slopes(perf, fits or {})
    if prof:
        profile_heatmap(prof)
    if var:
        variants_forest(var)
        if prof:
            league_size(var, prof)
    if tr:
        transfer(tr)
