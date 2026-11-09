"""Charts of the odds-timing study (read from results/*.json and cache/*.parquet)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

CLS = ["A1", "A2", "B", "C"]


def _err(ax, x, mean, ci, color, label=None, marker="o"):
    lo = [m - c[0] if c[0] is not None else 0 for m, c in zip(mean, ci)]
    hi = [c[1] - m if c[1] is not None else 0 for m, c in zip(mean, ci)]
    ax.errorbar(x, mean, yerr=[lo, hi], fmt=marker, color=color, capsize=3, ms=5, label=label)


def audit_figs():
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    a = C.load_json("audit.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    br = a["counts"]["by_class_role"]
    roles = ["main", "risk"]
    bottom = np.zeros(2)
    for i, c in enumerate(CLS):
        v = np.array([br.get(c, {}).get(r, 0) for r in roles], float)
        ax.bar(["Main", "Risk"], v, bottom=bottom, color=PALETTE[i], label=c)
        for j, x in enumerate(v):
            if x > 60:
                ax.text(j, bottom[j] + x / 2, f"{c}: {int(x)}", ha="center", va="center", fontsize=8, color="white")
        bottom += v
    ax.set_ylabel("прогнозов с ценой"), ax.legend(fontsize=8), ax.set_title("Классы цен 2026/27 (история сайта)", fontsize=10)
    ax = axes[1]
    labels, keys = [], []
    for role in roles:
        for k, lab in (("live_A1", "A1"), ("live_A2", "A2"), ("live_B", "B"), ("backfill_C", "C"),
                       ("current_all_sources", "все")):
            keys.append((role, k))
            labels.append(f"{'Main' if role == 'main' else 'Risk'} {lab}")
    x = np.arange(len(keys))
    mean = [a["roi"][r][k]["roi"] * 100 if a["roi"][r][k]["roi"] is not None else np.nan for r, k in keys]
    ci = [[(c * 100 if c is not None else None) for c in a["roi"][r][k]["roi_ci95"]] for r, k in keys]
    colors = [PALETTE[0] if r == "main" else PALETTE[1] for r, _ in keys]
    bets = [a["roi"][r][k]["bets"] for r, k in keys]
    for i in range(len(x)):
        if np.isfinite(mean[i]) and ci[i][0] is not None and bets[i] >= 10:
            _err(ax, [x[i]], [mean[i]], [ci[i]], colors[i])
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("ROI, %"), ax.set_title("ROI по классам цены (95 % ДИ; < 10 ставок не показано)", fontsize=10)
    save_chart(fig, C.FIGURES / "audit_classes.png")

    p = pd.read_parquet(C.CACHE / "audit_picks.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    live = p.loc[p["source"] == "live"].drop_duplicates("match_id")
    axes[0].hist(np.clip(live["gen_lead_h"], 0, 96), bins=48, color=PALETTE[0])
    axes[0].set_xlabel("часов до начала матча (обрезано на 96)"), axes[0].set_ylabel("матчей")
    axes[0].set_title("Когда сохранена хранимая версия live-прогноза", fontsize=10)
    for i, c in enumerate(["A1", "B", "C"]):
        v = p.loc[(p["price_class"] == c) & p["quote_lead_h"].notna(), "quote_lead_h"]
        axes[1].hist(np.clip(v, 0, 48), bins=48, histtype="step", lw=1.4, color=PALETTE[i], density=True, label=c)
    axes[1].set_xlabel("за сколько часов до начала опубликована котировка (обрезано на 48)")
    axes[1].set_ylabel("плотность"), axes[1].legend(fontsize=8)
    axes[1].set_title("Время публикации использованной цены", fontsize=10)
    save_chart(fig, C.FIGURES / "lead_times.png")


def drift_figs():
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    d = C.load_json("drift.json")
    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, m in enumerate(["1X2", "TOTAL_2.5", "AH"]):
        v = [d["fd"]["by_season"].get(f"{s}|{m}|b365", {}).get("mean_abs_dq_pp", np.nan) for s in seasons]
        axes[0].plot(seasons, v, marker="o", color=PALETTE[i], label={"TOTAL_2.5": "Тотал 2.5", "AH": "Азиатская фора"}.get(m, m))
        dd = [d["fd"]["by_season"].get(f"{s}|{m}|b365", {}).get("ll_close_minus_open") for s in seasons]
        mean = [x["diff"] * 1000 if x else np.nan for x in dd]
        ci = [[x["ci95"][0] * 1000, x["ci95"][1] * 1000] if x else [None, None] for x in dd]
        _err(axes[1], np.array(seasons) + (i - 1) * 0.15, mean, ci, PALETTE[i],
             {"TOTAL_2.5": "Тотал 2.5", "AH": "Азиатская фора"}.get(m, m))
    axes[0].set_title("Средний |Δq| открытие → закрытие (Bet365), п.п.", fontsize=10)
    axes[0].set_xticks(seasons, [C.season_label(s) for s in seasons], fontsize=8), axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="#555", lw=0.7)
    axes[1].set_title("Log loss закрытие − открытие (×1000), 95 % ДИ", fontsize=10)
    axes[1].set_xticks(seasons, [C.season_label(s) for s in seasons], fontsize=8), axes[1].legend(fontsize=8)
    save_chart(fig, C.FIGURES / "fd_drift.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ctr = d["model_vs_drift_controlled"]
    for i, m in enumerate(["1X2", "TOTAL_2.5"]):
        ss = [s for s in (2022, 2023, 2024, 2025) if f"{s}|{m}" in ctr]
        mean = [ctr[f"{s}|{m}"]["beta"][1] for s in ss]
        ci = [ctr[f"{s}|{m}"]["ci95"][1] for s in ss]
        _err(ax, np.array(ss) + (i - 0.5) * 0.15, mean, ci, PALETTE[i], "1X2" if m == "1X2" else "Тотал 2.5")
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks([2022, 2023, 2024, 2025], [C.season_label(s) for s in (2022, 2023, 2024, 2025)])
    ax.set_ylabel("коэффициент b₁"), ax.legend(fontsize=8)
    ax.set_title("Движение цены Pinnacle к модели: b₁ при контроле других букмекеров", fontsize=10)
    save_chart(fig, C.FIGURES / "model_drift.png")

    lh = d["live_horizons"]["by_group_bucket"]
    from .drift import BUCKET_RU
    groups = ["1X2", "TOTAL", "HANDICAP", "BTTS", "EHANDICAP", "CORNERS"]
    present = [b for b in BUCKET_RU if any((lh.get(f"{g}|{b}") or {}).get("matches", 0) >= 20 for g in groups)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, g in enumerate(groups):
        xs, mean, ci = [], [], []
        for j, b in enumerate(present):
            r = lh.get(f"{g}|{b}")
            if r is None or r["matches"] < 20:
                continue
            xs.append(j + (i - 2.5) * 0.1)
            mean.append(r["mean_abs_dq_pp"]["mean"])
            ci.append(r["mean_abs_dq_pp"]["ci95"])
        if xs:
            _err(ax, xs, mean, ci, PALETTE[i % len(PALETTE)], C.GROUP_RU.get(g, g))
    ax.set_xticks(range(len(present)), present)
    ax.set_xlabel("когда наблюдалась цена (до начала матча)"), ax.set_ylabel("|Δq| к цене ≤ 3 ч до начала, п.п.")
    ax.legend(fontsize=7.5, ncol=3), ax.set_title("OddsPapi 2026/27: насколько ранняя цена отличается от поздней", fontsize=10)
    save_chart(fig, C.FIGURES / "live_horizons.png")

    ca = d["closing_age"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, g in enumerate(["1X2", "TOTAL", "HANDICAP", "BTTS", "DOUBLE_CHANCE", "CORNERS"]):
        if g not in ca:
            continue
        hs = list(ca[g]["share_published_at_least_h_before"].keys())
        ax.plot([float(h) for h in hs], [ca[g]["share_published_at_least_h_before"][h] * 100 for h in hs], marker="o",
                ms=3, color=PALETTE[i % len(PALETTE)], label=C.GROUP_RU.get(g, g))
    ax.set_xscale("log"), ax.set_xlabel("часов до начала (лог. шкала)")
    ax.set_ylabel("% закрывающих цен, уже опубликованных"), ax.legend(fontsize=7.5)
    ax.set_title("Возраст закрывающей цены OddsPapi", fontsize=10)
    save_chart(fig, C.FIGURES / "closing_age.png")


def stability_figs():
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    s = C.load_json("stability.json")
    from .stability import LEAD_RU
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, role in enumerate(["main", "risk"]):
        v = [s["versions"]["by_role_bucket"].get(f"{role}|{b}", {}).get("changed", np.nan) * 100 for b in LEAD_RU]
        axes[0].plot(range(len(LEAD_RU)), v, marker="o", color=PALETTE[i], label="Main" if role == "main" else "Risk")
    axes[0].set_xticks(range(len(LEAD_RU)), LEAD_RU), axes[0].set_ylabel("% версий с другим выбором")
    axes[0].set_xlabel("за сколько часов до начала сохранена версия"), axes[0].legend(fontsize=8)
    axes[0].set_title("Выбор ранней версии ≠ выбору финальной", fontsize=10)
    h = s["history"]
    ss = [2023, 2024, 2025]
    for i, k in enumerate(["open_choice@open", "open_choice@close", "close_choice@close"]):
        mean = [h[str(x)][k]["roi"] * 100 for x in ss]
        ci = [[c * 100 for c in h[str(x)][k]["roi_ci95"]] for x in ss]
        _err(axes[1], np.array(ss) + (i - 1) * 0.15, mean, ci, PALETTE[i],
             {"open_choice@open": "выбор и цена открытия", "open_choice@close": "выбор открытия, цена закрытия",
              "close_choice@close": "выбор и цена закрытия"}[k])
    axes[1].axhline(0, color="#555", lw=0.7)
    axes[1].set_xticks(ss, [C.season_label(x) for x in ss]), axes[1].set_ylabel("ROI Main, %"), axes[1].legend(fontsize=7.5)
    axes[1].set_title("Main по Bet365: открытие vs закрытие (FD)", fontsize=10)
    save_chart(fig, C.FIGURES / "stability.png")


def clv_figs():
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    c = C.load_json("clv.json")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    keys = [("main|A1", "Main A1"), ("risk|A1", "Risk A1"), ("main|B", "Main B"), ("risk|B", "Risk B"),
            ("main|C", "Main C"), ("risk|C", "Risk C")]
    for j, (col, lab) in enumerate((("clv_odds", "CLV к цене того же букмекера"), ("clv_fair", "CLV к справедливой цене закрытия")) ):
        xs, mean, ci = [], [], []
        for i, (k, _) in enumerate(keys):
            r = c.get(k, {}).get(col)
            if r and r.get("mean") is not None and r["ci95"][0] is not None:
                xs.append(i + (j - 0.5) * 0.2)
                mean.append(r["mean"] * 100)
                ci.append([r["ci95"][0] * 100, r["ci95"][1] * 100])
        _err(ax, xs, mean, ci, PALETTE[j], lab)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys]), ax.set_ylabel("%"), ax.legend(fontsize=8)
    ax.set_title("CLV сохранённых Main / Risk 2026/27 (95 % ДИ, кластер = матч)", fontsize=10)
    save_chart(fig, C.FIGURES / "clv.png")


def run():
    audit_figs()
    drift_figs()
    stability_figs()
    clv_figs()


if __name__ == "__main__":
    run()
