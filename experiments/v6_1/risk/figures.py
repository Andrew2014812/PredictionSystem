"""Figures of the Risk strategy research (PNG, from results/*.json and the cached picks)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_kit import ACCENT, AMBER, MUTED, NEG, PALETTE, plt, save_chart

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES, CACHE = HERE / "results", HERE / "figures", HERE / "cache"
SEASON_LABEL = {"2023": "2023/24", "2024": "2024/25", "2025": "2025/26 (инф.)", "2026": "2026/27 (незав.)"}
SEASON_COLOR = {"2023": PALETTE[0], "2024": PALETTE[2], "2025": PALETTE[5], "2026": PALETTE[1]}
KEY_STRATEGIES = ["Текущий Risk", "Value Risk (EV ≥ 0)", "Пол EV ≥−5 %", "Max-EV Risk",
                  "Calibrated Risk, пол нет", "Calibrated Risk (бленд с рынком), пол ≥0 %", "Market-specific Risk",
                  "Conservative Risk (2.50–3.50)", "High-odds Risk (4.50–8.00)"]


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def roi_by_strategy(hist, indep=None):
    names = list(hist["seasons"]["2023"])
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(names) + 1.4))
    y = np.arange(len(names))[::-1]
    off = np.linspace(-0.27, 0.27, len(seasons))
    for k, s in enumerate(seasons):
        per = indep["strategies"] if s == "2026" else hist["seasons"][s]
        r = np.array([per[n]["roi"] * 100 for n in names])
        lo = np.array([per[n]["roi_ci95"][0] * 100 for n in names])
        hi = np.array([per[n]["roi_ci95"][1] * 100 for n in names])
        ax.errorbar(r, y + off[k], xerr=[r - lo, hi - r], fmt="o", ms=3.5, lw=0.8, capsize=0,
                    color=SEASON_COLOR[s], label=SEASON_LABEL[s])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xlabel("ROI ставок с реальной ценой, % (линии — 95 % бутстреп-ДИ)")
    ax.legend(fontsize=7, loc="lower left", ncols=2)
    return save_chart(fig, FIGURES / ("roi_by_strategy_2026.png" if indep else "roi_by_strategy.png"))


def band_floor_heatmap(hist):
    pb = hist["pooled_bands_select"]
    bands = list(dict.fromkeys(v["band"] for v in pb.values()))
    floors = list(dict.fromkeys(v["floor"] for v in pb.values()))
    roi = np.array([[pb[f"{b} | {f}"]["roi"] * 100 if pb[f"{b} | {f}"]["roi"] is not None else np.nan
                     for f in floors] for b in bands])
    bets = np.array([[pb[f"{b} | {f}"]["bets"] for f in floors] for b in bands])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    lim = np.nanmax(np.abs(roi))
    im = axes[0].imshow(roi, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(roi):
        axes[0].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5)
    axes[0].set_title("ROI, % (2023/24 + 2024/25)", fontsize=9)
    im2 = axes[1].imshow(bets, cmap="Greys", aspect="auto", alpha=0.6)
    for (i, j), v in np.ndenumerate(bets):
        axes[1].text(j, i, f"{v}", ha="center", va="center", fontsize=7.5)
    axes[1].set_title("Число ставок (2 сезона)", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(floors)), [f"EV {f}" for f in floors], fontsize=7.5)
        ax.set_yticks(range(len(bands)), bands, fontsize=7.5)
        ax.grid(False)
    axes[0].set_ylabel("Диапазон коэффициента")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    del im2
    return save_chart(fig, FIGURES / "band_floor_heatmap.png")


def band_by_season(hist, indep=None):
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    bands = list(dict.fromkeys(v["band"] for v in hist["bands"]["2023"].values()))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(bands))
    w = 0.8 / len(seasons)
    for k, s in enumerate(seasons):
        per = indep["bands"] if s == "2026" else hist["bands"][s]
        r = [per[f"{b} | нет"]["roi"] * 100 if per[f"{b} | нет"]["roi"] is not None else np.nan for b in bands]
        lo = [per[f"{b} | нет"]["roi_ci95"][0] * 100 if per[f"{b} | нет"]["roi_ci95"] else np.nan for b in bands]
        hi = [per[f"{b} | нет"]["roi_ci95"][1] * 100 if per[f"{b} | нет"]["roi_ci95"] else np.nan for b in bands]
        ax.bar(x + (k - (len(seasons) - 1) / 2) * w, r, w, color=SEASON_COLOR[s], label=SEASON_LABEL[s])
        ax.errorbar(x + (k - (len(seasons) - 1) / 2) * w, r, yerr=[np.subtract(r, lo), np.subtract(hi, r)],
                    fmt="none", ecolor="black", lw=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, bands)
    ax.set_xlabel("Диапазон коэффициента (Risk только из этого диапазона, без пола EV)")
    ax.set_ylabel("ROI, %")
    ax.legend(fontsize=7)
    return save_chart(fig, FIGURES / ("band_by_season_2026.png" if indep else "band_by_season.png"))


def calibration_rare(ana, indep=None):
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s in seasons:
        d = (indep["diagnostics"] if s == "2026" else ana["seasons"][s])["calibration_rare"]
        b = d["by_bin"]
        axes[0].plot([r["mean_p"] for r in b], [r["observed"] for r in b], "o-", ms=3.5, color=SEASON_COLOR[s],
                     label=SEASON_LABEL[s])
        b2 = d.get("priced_only_by_bin", [])
        if b2:
            axes[1].plot([r["mean_p"] for r in b2], [r["observed"] for r in b2], "o-", ms=3.5,
                         color=SEASON_COLOR[s], label=f"{SEASON_LABEL[s]}: модель")
            axes[1].plot([r["mean_p_market"] for r in b2], [r["observed"] for r in b2], "s--", ms=3, lw=0.8,
                         color=SEASON_COLOR[s], alpha=0.7, label=f"{SEASON_LABEL[s]}: рынок")
    for ax, t in zip(axes, ("Все кандидаты Risk (p 0.12–0.40)", "Только с полной книгой цен")):
        ax.plot([0.1, 0.42], [0.1, 0.42], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Средняя предсказанная вероятность")
        ax.set_ylabel("Наблюдаемая частота")
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=6.5)
    return save_chart(fig, FIGURES / ("calibration_rare_2026.png" if indep else "calibration_rare.png"))


def calibration_by_key(ana, indep=None):
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    keys = set()
    for s in seasons:
        d = (indep["diagnostics"] if s == "2026" else ana["seasons"][s])["calibration_rare"]["by_key"]
        keys |= {k for k, v in d.items() if v["n"] >= 150}
    keys = sorted(keys)
    fig, ax = plt.subplots(figsize=(8.2, 0.33 * len(keys) + 1.3))
    y = np.arange(len(keys))[::-1]
    off = np.linspace(-0.25, 0.25, len(seasons))
    for k, s in enumerate(seasons):
        d = (indep["diagnostics"] if s == "2026" else ana["seasons"][s])["calibration_rare"]["by_key"]
        xs, lo, hi, yy = [], [], [], []
        for i, key in enumerate(keys):
            v = d.get(key)
            if v and v["n"] >= 150:
                xs.append(v["observed"] - v["mean_p"])
                lo.append(v["observed_ci95"][0] - v["mean_p"])
                hi.append(v["observed_ci95"][1] - v["mean_p"])
                yy.append(y[i] + off[k])
        xs, lo, hi = np.array(xs) * 100, np.array(lo) * 100, np.array(hi) * 100
        ax.errorbar(xs, yy, xerr=[xs - lo, hi - xs], fmt="o", ms=3, lw=0.7, color=SEASON_COLOR[s],
                    label=SEASON_LABEL[s])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, keys, fontsize=7)
    ax.set_xlabel("Наблюдаемая частота − средняя p, п.п. (p 0.12–0.40; < 0 — модель завышает)")
    ax.legend(fontsize=7)
    return save_chart(fig, FIGURES / ("calibration_by_key_2026.png" if indep else "calibration_by_key.png"))


def favourite_longshot(ana, indep=None):
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in seasons:
        rows = (indep["diagnostics"] if s == "2026" else ana["seasons"][s])["favourite_longshot"]
        x = np.arange(len(rows))
        axes[0].plot(x, [r["blind_roi"] * 100 for r in rows], "o-", ms=3.5, color=SEASON_COLOR[s],
                     label=SEASON_LABEL[s])
        axes[0].set_xticks(x, [r["band"] for r in rows], rotation=30, fontsize=7)
        axes[1].plot(x, [(r["observed"] - r["devig"]) * 100 for r in rows], "o-", ms=3.5, color=SEASON_COLOR[s],
                     label=SEASON_LABEL[s])
        axes[1].set_xticks(x, [r["band"] for r in rows], rotation=30, fontsize=7)
    axes[0].set_title("ROI ставки на каждый исход с ценой, %", fontsize=9)
    axes[1].set_title("Наблюдаемая частота − вероятность без маржи, п.п.", fontsize=9)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Коэффициент")
        ax.legend(fontsize=7)
    return save_chart(fig, FIGURES / ("favourite_longshot_2026.png" if indep else "favourite_longshot.png"))


def ev_vs_roi(ana, indep=None):
    src = indep["diagnostics"] if indep else ana["pooled_select"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for key, lab, col in (("ev_bins_model", "EV модели", NEG), ("ev_bins_platt", "EV после калибровки Платта", AMBER),
                          ("ev_bins_blend", "EV бленда модель + рынок", ACCENT)):
        rows = [r for r in src[key]["bins"] if r["n"] >= 100]       # tiny bins are not shown (see the tables)
        x = [r["mean_ev"] * 100 for r in rows]
        y = [r["roi"] * 100 for r in rows]
        lo = [r["roi_ci95"][0] * 100 for r in rows]
        hi = [r["roi_ci95"][1] * 100 for r in rows]
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", ms=3.5, lw=1, capsize=2,
                    color=col, label=f"{lab} (наклон {src[key]['slope']:.2f})")
    lim = [-25, 40]
    ax.plot(lim, lim, color=MUTED, ls=":", lw=0.8, label="ROI = EV")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-25, 45)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Средний ожидаемый EV в корзине, % (корзины с n ≥ 100)")
    ax.set_ylabel("Фактический ROI, %")
    ax.legend(fontsize=7)
    return save_chart(fig, FIGURES / ("ev_vs_roi_2026.png" if indep else "ev_vs_roi.png"))


def mc_inflation(ana):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for s in ("2023", "2024", "2025"):
        rows = ana["seasons"][s]["max_selection_mc"]["rows"]
        sig = [r["sigma"] for r in rows]
        ax.plot(sig, [r["model_ev"] * 100 for r in rows], "o-", ms=3.5, color=SEASON_COLOR[s],
                label=f"{SEASON_LABEL[s]}: EV выбранной ставки по «модели»")
        ax.plot(sig, [r["true_roi"] * 100 for r in rows], "s--", ms=3, color=SEASON_COLOR[s], alpha=0.75,
                label=f"{SEASON_LABEL[s]}: истинный ожидаемый ROI")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("σ шума модели в логитах")
    ax.set_ylabel("%")
    ax.legend(fontsize=6.5)
    return save_chart(fig, FIGURES / "mc_inflation.png")


def variance(ana):
    rows = ana["variance"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    o = [r["odds"] for r in rows]
    axes[0].plot(o, [r["sd_per_bet"] for r in rows], "o-", color=PALETTE[0], label="SD прибыли одной ставки")
    axes[0].plot(o, [r["se_roi_2000"] * 100 for r in rows], "s-", color=PALETTE[1],
                 label="SE ROI за 2000 ставок, п.п.")
    axes[0].set_xlabel("Коэффициент")
    axes[0].legend(fontsize=7)
    axes[1].plot(o, [r["median_max_drawdown"] for r in rows], "o-", color=PALETTE[2], label="медиана")
    axes[1].plot(o, [r["p95_max_drawdown"] for r in rows], "s-", color=NEG, label="95-й перцентиль")
    axes[1].set_xlabel("Коэффициент")
    axes[1].set_ylabel("Макс. просадка за 2000 ставок, ед.")
    axes[1].legend(fontsize=7)
    return save_chart(fig, FIGURES / "variance.png")


def cumulative(names=KEY_STRATEGIES):
    files = [CACHE / f"picks_{s}.parquet" for s in (2023, 2024, 2025, 2026)]
    frames = [pd.read_parquet(f).assign(season=f.stem[-4:]) for f in files if f.exists()]
    if not frames:
        return None
    df = pd.concat(frames)
    hist = df.loc[df["season"] != "2026"]
    fig, axes = plt.subplots(1, 2 if (df["season"] == "2026").any() else 1, figsize=(11, 4), squeeze=False)
    for ax, part, title in ((axes[0][0], hist, "2023/24 – 2025/26 (подряд)"),
                            (axes[0][-1], df.loc[df["season"] == "2026"], "2026/27 (независимая проверка)")):
        if part.empty:
            continue
        for k, n in enumerate(names):
            b = part.loc[(part["strategy"] == n) & part["decided"] & part["priced"]].sort_values(["date", "match_id"])
            ax.plot(np.arange(len(b)), b["profit"].cumsum().to_numpy(), lw=1,
                    color=PALETTE[k % len(PALETTE)], ls="-" if k < 6 else "--", label=n)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Ставка №")
        ax.set_ylabel("Накопленная прибыль, ед.")
    axes[0][0].legend(fontsize=6)
    return save_chart(fig, FIGURES / "cumulative_profit.png")


def current_bands(hist, indep=None):
    seasons = ["2023", "2024", "2025"] + (["2026"] if indep else [])
    labels = ["<2.50", "2.50–3.00", "3.00–3.50", "3.50–5.00", "5.00–6.00", ">6.00"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(labels))
    w = 0.8 / len(seasons)
    for k, s in enumerate(seasons):
        per = (indep["strategies"] if s == "2026" else hist["seasons"][s])["Текущий Risk"]["odds_bands"]
        axes[0].bar(x + (k - (len(seasons) - 1) / 2) * w, [per.get(b, {}).get("bets", 0) for b in labels], w,
                    color=SEASON_COLOR[s], label=SEASON_LABEL[s])
        axes[1].bar(x + (k - (len(seasons) - 1) / 2) * w,
                    [per.get(b, {}).get("roi", np.nan) * 100 if b in per else np.nan for b in labels], w,
                    color=SEASON_COLOR[s], label=SEASON_LABEL[s])
    axes[0].set_title("Ставки текущего Risk по диапазонам", fontsize=9)
    axes[1].set_title("ROI текущего Risk по диапазонам, %", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.legend(fontsize=6.5)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    return save_chart(fig, FIGURES / ("current_bands_2026.png" if indep else "current_bands.png"))


def build():
    hist, ana, indep = _load("history.json"), _load("analysis.json"), _load("independent.json")
    out = []
    if hist:
        out += [roi_by_strategy(hist), band_floor_heatmap(hist), band_by_season(hist), current_bands(hist)]
    if ana:
        out += [calibration_rare(ana), calibration_by_key(ana), favourite_longshot(ana), ev_vs_roi(ana),
                mc_inflation(ana), variance(ana)]
    if hist and indep:
        out += [roi_by_strategy(hist, indep), band_by_season(hist, indep), current_bands(hist, indep)]
    if ana and indep:
        out += [calibration_rare(ana, indep), calibration_by_key(ana, indep), favourite_longshot(ana, indep),
                ev_vs_roi(ana, indep)]
    out.append(cumulative())
    return out


if __name__ == "__main__":
    for p in build():
        print(p)
