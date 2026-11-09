"""Charts of the bookmaker-benchmark study (from results/*.json)."""
from __future__ import annotations

import numpy as np

from . import common as C

SEAS = [2023, 2024, 2025, 2026]
MK_RU = {"1X2": "1X2", "OU25": "Тотал 2.5", "AH": "Аз. фора (линия FD)"}


def _err(ax, x, mean, ci, color, label=None, marker="o"):
    x, mean = np.asarray(x, float), np.asarray(mean, float)
    lo = [m - c[0] if c and c[0] is not None else 0 for m, c in zip(mean, ci)]
    hi = [c[1] - m if c and c[1] is not None else 0 for m, c in zip(mean, ci)]
    ax.errorbar(x, mean, yerr=[lo, hi], fmt=marker, color=color, capsize=3, ms=4.5, label=label)


def run():
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    dm = C.load_json("demargin.json")
    b = C.load_json("benchmark.json")
    enc = C.load_json("encompassing.json")
    hy = C.load_json("hybrids.json")
    bet = C.load_json("betting.json")
    mm = C.load_json("multimarket_2026.json")

    # 1. margin removal
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, market in zip(axes, ("1X2", "OU25")):
        for i, ref in enumerate(("b365_open", "ps_open", "b365_close", "ps_close")):
            keys = [f"{market}|{s}|{ref}" for s in C.SELECT]
            if not all(k in dm["by_key"] for k in keys):
                continue
            vals = []
            for m in ("shin", "power", "log", "add"):
                vals.append(np.mean([dm["by_key"][k][f"{m}_minus_prop"]["diff"] for k in keys]) * 1000)
            ax.plot(range(4), vals, marker="o", color=PALETTE[i], label=ref.replace("_", " "))
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(4), ["Шин", "степенной", "логариф.", "аддитив."])
        ax.set_ylabel("log loss − пропорц. (×1000)"), ax.legend(fontsize=7.5)
        ax.set_title(f"{MK_RU[market]}: методы снятия маржи (2023/24+2024/25)", fontsize=10)
    save_chart(fig, C.FIGURES / "demargin.png")

    # 2. model vs market
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, market in zip(axes, ("1X2", "OU25", "AH")):
        for i, ref in enumerate(("b365_open", "ps_open", "b365_close", "bfe_close")):
            xs, mean, ci = [], [], []
            for j, s in enumerate(SEAS):
                r = b["model_vs_market"].get(f"{market}|{s}|{ref}")
                if r:
                    xs.append(j + (i - 1.5) * 0.12)
                    mean.append(r["d_ll"]["diff"] * 1000)
                    ci.append([r["d_ll"]["ci95"][0] * 1000, r["d_ll"]["ci95"][1] * 1000])
            if xs:
                _err(ax, xs, mean, ci, PALETTE[i], ref.replace("_", " "))
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(len(SEAS)), [C.season_label(s) for s in SEAS], fontsize=8)
        ax.set_title(f"{MK_RU[market]}: log loss модель − рынок (×1000)", fontsize=9.5)
        ax.legend(fontsize=7)
    save_chart(fig, C.FIGURES / "model_vs_market.png")

    # 3. leagues
    lg = b["by_league"]["1X2"]["leagues"]
    names = sorted(lg, key=lambda k: lg[k]["select"]["mean"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    _err(ax, range(len(names)), [lg[k]["select"]["mean"] * 1000 for k in names],
         [[c * 1000 for c in lg[k]["select"]["ci95"]] for k in names], PALETTE[0], "2023/24+2024/25")
    m25 = [lg[k]["2025"]["mean"] * 1000 if lg[k]["2025"].get("mean") is not None else np.nan for k in names]
    ax.plot(np.arange(len(names)) + 0.25, m25, "s", color=PALETTE[1], ms=4, label="2025/26 (инф.)")
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(range(len(names)), names, fontsize=8), ax.legend(fontsize=8)
    ax.set_ylabel("log loss модель − Bet365 откр. (×1000)")
    ax.set_title("1X2 по лигам: ниже нуля = модель лучше рынка", fontsize=10)
    save_chart(fig, C.FIGURES / "leagues.png")

    # 4. odds ranges
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = ["1.0–1.5", "1.5–2.0", "2.0–3.0", "3.0–5.0", "> 5.0"]
    for ax, market in zip(axes, ("1X2", "OU25")):
        for i, per in enumerate(("select", "2025", "2026")):
            xs, mean, ci = [], [], []
            for j, lab in enumerate(labels):
                r = b["by_odds_range"].get(f"{market}|{per}|{lab}")
                if r:
                    xs.append(j + (i - 1) * 0.15)
                    mean.append(r["d_ll"]["mean"] * 1000)
                    ci.append([c * 1000 for c in r["d_ll"]["ci95"]])
            if xs:
                _err(ax, xs, mean, ci, PALETTE[i], {"select": "2023/24+2024/25", "2025": "2025/26 (инф.)",
                                                    "2026": "2026/27 (инф.)"}[per])
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(len(labels)), labels), ax.set_xlabel("коэффициент Bet365 на открытии")
        ax.set_ylabel("Δ log loss исхода (×1000)"), ax.legend(fontsize=7.5)
        ax.set_title(f"{MK_RU[market]}: модель − рынок по диапазонам коэффициента", fontsize=9.5)
    save_chart(fig, C.FIGURES / "odds_range.png")

    # 5. reliability
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, market in zip(axes, ("1X2", "OU25")):
        for i, who in enumerate(("model", "market")):
            r = b["reliability"][f"{market}|{who}"]
            ax.plot([x["pred"] for x in r], [x["obs"] for x in r], marker="o", color=PALETTE[i],
                    label="модель" if who == "model" else "Bet365 откр. (без маржи)")
        lo = min(x["pred"] for x in b["reliability"][f"{market}|model"])
        hi = max(x["pred"] for x in b["reliability"][f"{market}|model"])
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=0.8)
        ax.set_xlabel("прогноз"), ax.set_ylabel("частота"), ax.legend(fontsize=8)
        ax.set_title(f"Надёжность: {'победа хозяев' if market == '1X2' else 'тотал больше 2.5'} (2023/24+2024/25)",
                     fontsize=9.5)
    save_chart(fig, C.FIGURES / "reliability.png")

    # 6. encompassing
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, market in zip(axes, ("1X2", "OU25", "AH")):
        for i, ref in enumerate(("b365_open", "ps_open", "b365_close", "bfe_close")):
            xs, mean, ci = [], [], []
            for j, s in enumerate(SEAS):
                r = enc["tests"].get(f"{market}|{s}|{ref}")
                if r:
                    xs.append(j + (i - 1.5) * 0.12)
                    mean.append(r["b_model"])
                    ci.append(r["b_model_ci95"])
            if xs:
                _err(ax, xs, mean, ci, PALETTE[i], ref.replace("_", " "))
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(len(SEAS)), [C.season_label(s) for s in SEAS], fontsize=8)
        ax.set_title(f"{MK_RU[market]}: вес модели b₂ при рынке", fontsize=9.5), ax.legend(fontsize=7)
    save_chart(fig, C.FIGURES / "encompassing.png")

    # 7. hybrids
    if hy:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        names = ["model", "H1", "H2", "H3", "H4", "H5"]
        for ax, market in zip(axes, ("1X2", "OU25", "AH")):
            for i, nm in enumerate(names):
                xs, mean, ci = [], [], []
                for j, s in enumerate(SEAS):
                    r = hy["results"].get(f"{market}|{s}", {}).get(nm)
                    if r and "d_ll_vs_H0" in r:
                        xs.append(j + (i - 2.5) * 0.1)
                        mean.append(r["d_ll_vs_H0"]["diff"] * 1000)
                        ci.append([c * 1000 for c in r["d_ll_vs_H0"]["ci95"]])
                if xs:
                    _err(ax, xs, mean, ci, PALETTE[i % len(PALETTE)], nm if nm != "model" else "модель")
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_xticks(range(len(SEAS)), [C.season_label(s) for s in SEAS], fontsize=8)
            ax.set_title(f"{MK_RU[market]}: log loss − рынок H0 (×1000)", fontsize=9.5), ax.legend(fontsize=6.5, ncol=2)
        save_chart(fig, C.FIGURES / "hybrids.png")

    # 8. betting
    if bet:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        names = ["model", "H1", "H2", "H3", "H4", "H5"]
        for ax, market in zip(axes, ("1X2", "OU25")):
            for i, nm in enumerate(names):
                xs, mean, ci = [], [], []
                for j, s in enumerate(SEAS):
                    r = bet.get(f"{market}|{s}|{nm}|0.0")
                    if r and r.get("bets", 0) >= 30:
                        xs.append(j + (i - 2.5) * 0.1)
                        mean.append(r["roi"] * 100)
                        ci.append([c * 100 for c in r["roi_ci95"]])
                if xs:
                    _err(ax, xs, mean, ci, PALETTE[i % len(PALETTE)], nm if nm != "model" else "модель")
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_xticks(range(len(SEAS)), [C.season_label(s) for s in SEAS], fontsize=8)
            ax.set_ylabel("ROI, %"), ax.legend(fontsize=6.5, ncol=2)
            ax.set_title(f"{MK_RU[market]}: ставка при EV > 0 по Bet365 откр. (≥ 30 ставок)", fontsize=9.5)
        save_chart(fig, C.FIGURES / "betting.png")

    # 9. 2026 multi-market
    if mm:
        g = mm["by_group"]
        names = list(g)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        _err(ax, range(len(names)), [g[k]["mean"] * 1000 for k in names], [[c * 1000 for c in g[k]["ci95"]] for k in names],
             PALETTE[0])
        ax.axhline(0, color="#555", lw=0.7)
        ax.set_xticks(range(len(names)), [C.GROUP_RU.get(k, k) for k in names])
        ax.set_ylabel("log loss модель − закрытие (×1000)")
        ax.set_title("2026/27 (инф.): модель против закрытия OddsPapi по группам рынков", fontsize=10)
        save_chart(fig, C.FIGURES / "multimarket.png")


if __name__ == "__main__":
    run()
