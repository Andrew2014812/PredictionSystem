"""Charts of the data-quality report (read only from results/). Palette checked with the dataviz validator:
teal / amber / blue pass lightness, chroma, colour-blind and normal-vision separation on white; amber is below
3:1 contrast, so every chart is backed by a table in the report and carries direct labels or a legend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_kit import GRID, INK, MUTED, save_chart

from . import common as C

import matplotlib.pyplot as plt  # noqa: E402  (style set by report_kit)

TEAL, AMBER, BLUE, GREY = "#00896a", "#c88a12", "#3d6fb6", "#8a969c"
STATUS_RU = {"established": "старожилы", "moved_within": "переход внутри наших лиг", "returner": "вернувшиеся",
             "outside": "новички извне"}


def missing_by_league() -> None:
    m = C.load_csv("raw_missing.csv")
    if m is None:
        return
    w = m.assign(w_shots=m["shots"] * m["played"], w_corners=m["corners"] * m["played"],
                 w_price=m["price_1x2"] * m["played"], w_ah=m["price_ah"] * m["played"])
    g = w.groupby("league")[["w_shots", "w_corners", "w_price", "w_ah", "played"]].sum()
    g = g.assign(shots=g["w_shots"] / g["played"] * 100, price=g["w_price"] / g["played"] * 100,
                 ah=g["w_ah"] / g["played"] * 100).sort_values("shots")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(g))
    ax.barh(y + 0.27, g["shots"], 0.26, color=TEAL, label="нет статистики (удары / угловые)")
    ax.barh(y, g["price"], 0.26, color=AMBER, label="нет цены Bet365 1X2")
    ax.barh(y - 0.27, g["ah"], 0.26, color=BLUE, label="нет цены азиатской форы")
    ax.set_yticks(y, g.index)
    ax.set_xlabel("доля сыгранных матчей, %")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Пропуски в файлах Football-Data, 2021/22–2026/27", loc="left", fontsize=10, color=INK)
    save_chart(fig, C.FIGURES / "missing_by_league.png")


def feature_profile() -> None:
    p = C.load_csv("teams_feature_profile.csv")
    if p is None:
        return
    order = ["1", "2-3", "4-5", "6-10", "11+"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for status, color, marker in (("outside", AMBER, "o"), ("established", TEAL, "s"), ("returner", BLUE, "^")):
        q = p.loc[p["status"] == status].set_index("bucket").reindex(order)
        ax.plot(order, q["missing_share"] * 100, color=color, marker=marker, lw=2, ms=6, label=STATUS_RU[status])
    ax.set_xlabel("номер матча клуба в сезоне")
    ax.set_ylabel("пропущено признаков стороны, %")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Сколько признаков модели 1X2 пусто у клуба", loc="left", fontsize=10, color=INK)
    save_chart(fig, C.FIGURES / "feature_profile.png")


def sensitivity() -> None:
    s = C.load_csv("sensitivity_summary.csv")
    if s is None:
        return
    names = {"season": "таблица сезона", "form": "форма", "stats": "удары / угловые", "elo": "Elo = базовый",
             "rest": "дни отдыха", "newcomer": "всё (новичок)"}
    rows = []
    for g, ru in names.items():
        h = s.loc[s["variant"] == f"{g}_h"]
        a = s.loc[s["variant"] == f"{g}_a"]
        if len(h) and len(a):
            rows.append((ru, float(h["main_changed"].iloc[0]) * 100, float(a["main_changed"].iloc[0]) * 100,
                         float(h["d_log_loss"].iloc[0]) * 1000, float(a["d_log_loss"].iloc[0]) * 1000))
    hh = s.loc[s["variant"] == "h2h"]
    if len(hh):
        rows.append(("личные встречи", float(hh["main_changed"].iloc[0]) * 100, np.nan,
                     float(hh["d_log_loss"].iloc[0]) * 1000, np.nan))
    d = pd.DataFrame(rows, columns=["g", "mh", "ma", "lh", "la"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    y = np.arange(len(d))
    for ax, (ch, ca, title) in zip(axes, (("lh", "la", "рост log loss, ×10⁻³"), ("mh", "ma", "Main изменился, % матчей"))):
        ax.barh(y + 0.2, d[ch], 0.38, color=TEAL, label="пусто у хозяев")
        ax.barh(y - 0.2, d[ca], 0.38, color=AMBER, label="пусто у гостей")
        ax.set_xlabel(title)
        for yy, v in zip(y + 0.2, d[ch]):
            if v == v:
                ax.text(v, yy, f" {v:.0f}" if ch.startswith("m") else f" {v:.1f}", va="center", fontsize=7, color=MUTED)
    axes[0].set_yticks(y, d["g"])
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Что меняется, если у одной команды нет части признаков (2023/24–2026/27)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    save_chart(fig, C.FIGURES / "sensitivity.png")


def newcomer_curves() -> None:
    b = C.load_csv("strata_by_match_no.csv")
    pb = C.load_csv("strata_points_bias.csv")
    if b is None or pb is None:
        return
    order = ["1-3", "4-5", "6-10", "11-20", "21+"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for status, color, dx in (("established", TEAL, -0.08), ("outside", AMBER, 0.08)):
        q = b.loc[b["group"] == status].set_index("bucket").reindex(order)
        x = np.arange(len(order)) + dx
        axes[0].errorbar(x, q["excess_log_loss"] * 1000, yerr=[(q["excess_log_loss"] - q["lo"]) * 1000,
                                                               (q["hi"] - q["excess_log_loss"]) * 1000],
                         color=color, marker="o", lw=2, ms=6, capsize=3, label=STATUS_RU[status])
        r = pb.loc[pb["status"] == status].set_index("bucket").reindex(order)
        axes[1].errorbar(x, r["points_bias"], yerr=[r["points_bias"] - r["lo"], r["hi"] - r["points_bias"]],
                         color=color, marker="o", lw=2, ms=6, capsize=3, label=STATUS_RU[status])
    for ax in axes:
        ax.set_xticks(np.arange(len(order)), order)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel("номер матча клуба в сезоне")
    axes[0].set_ylabel("log loss модели − рынка, ×10⁻³")
    axes[1].set_ylabel("ожид. очки − факт. очки")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Избыточный log loss против рынка", loc="left", fontsize=9.5, color=INK)
    axes[1].set_title("Переоценка очков клуба моделью", loc="left", fontsize=9.5, color=INK)
    save_chart(fig, C.FIGURES / "newcomer_curves.png")


def dq_bins() -> None:
    d = C.load_csv("strata_dq_bins.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(d))
    axes[0].errorbar(x, d["excess_log_loss"] * 1000, yerr=[(d["excess_log_loss"] - d["excess_lo"]) * 1000,
                                                         (d["excess_hi"] - d["excess_log_loss"]) * 1000],
                     color=TEAL, marker="o", lw=2, capsize=3)
    axes[1].errorbar(x, d["main_roi"] * 100, yerr=[(d["main_roi"] - d["roi_lo"]) * 100, (d["roi_hi"] - d["main_roi"]) * 100],
                     color=BLUE, marker="o", lw=2, capsize=3)
    for ax in axes:
        ax.set_xticks(x, [f"{b}\nn={n}" for b, n in zip(d["dq_bin"], d["matches"])], fontsize=8)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel("доля пропущенных признаков матча (n — число матчей)")
    axes[0].set_ylabel("log loss модели − рынка, ×10⁻³")
    axes[1].set_ylabel("ROI Main, %")
    axes[0].set_title("Качество вероятностей", loc="left", fontsize=9.5, color=INK)
    axes[1].set_title("Main по реальной цене Bet365", loc="left", fontsize=9.5, color=INK)
    save_chart(fig, C.FIGURES / "dq_bins.png")


def phantoms() -> None:
    p = C.load_csv("store_phantoms.csv")
    r = C.load_csv("sensitivity_phantoms.csv")
    if p is None or r is None:
        return
    pairs = list(dict.fromkeys(p["pair"]))
    labels = {"B1-20261104-kortrijk-mechelen": "Kortrijk – Mechelen", "E1-20261103-preston-sheffield-united":
              "Preston – Sheffield United", "E2-20261104-oxford-reading": "Oxford – Reading"}
    fig, axes = plt.subplots(1, len(pairs), figsize=(10, 3.4), sharey=True)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        real = p.loc[(p["pair"] == pair) & (p["kind"] == "real")].iloc[0]
        ph = p.loc[(p["pair"] == pair) & (p["kind"] == "phantom")].iloc[0]
        rep = r.loc[(r["pair"] == pair) & (r["variant"] == "phantom_side_as_newcomer")].iloc[0]
        x = np.arange(3)
        ax.bar(x - 0.27, [real.p_home, real.p_draw, real.p_away], 0.26, color=TEAL, label="сохранён: настоящий матч")
        ax.bar(x, [ph.p_home, ph.p_draw, ph.p_away], 0.26, color=AMBER, label="сохранён: фантом")
        ax.bar(x + 0.27, [rep.p_home, rep.p_draw, rep.p_away], 0.26, color=BLUE, label="воспроизведение: сторона = новичок")
        ax.set_xticks(x, ["П1", "Х", "П2"])
        ax.set_title(labels.get(pair, pair), fontsize=9, loc="left", color=INK)
    np.atleast_1d(axes)[0].set_ylabel("вероятность")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="upper left")
    save_chart(fig, C.FIGURES / "phantoms.png")


def calibration() -> None:
    c = C.load_csv("strata_calibration.csv")
    if c is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 0.75], [0, 0.75], color=GREY, lw=1)
    for status, color, marker in (("established", TEAL, "s"), ("outside", AMBER, "o")):
        q = c.loc[c["status"] == status]
        ax.plot(q["predicted"], q["observed"], color=color, marker=marker, lw=2, ms=6, label=STATUS_RU[status])
    ax.set_xlabel("предсказанная P(победа клуба)")
    ax.set_ylabel("наблюдаемая частота побед")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Калибровка P(победа): новички извне и старожилы", loc="left", fontsize=9.5, color=INK)
    save_chart(fig, C.FIGURES / "calibration.png")


def variants() -> None:
    v = C.load_csv("variants_summary.csv")
    if v is None:
        return
    q = v.loc[(v["group"] == "outside_first10") & (v["season"] != "all")]
    if q.empty:
        return
    names = {"elo_new_-50": "Elo новичка −50", "elo_new_-100": "Elo новичка −100", "ec_history": "история National League"}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    seasons = sorted(q["season"].unique())
    colors = [TEAL, AMBER, BLUE]
    for i, (key, ru) in enumerate(names.items()):
        r = q.loc[q["variant"] == key].set_index("season").reindex(seasons)
        x = np.arange(len(seasons)) + (i - 1) * 0.25
        ax.errorbar(x, r["d_log_loss"] * 1000, yerr=[(r["d_log_loss"] - r["lo"]) * 1000, (r["hi"] - r["d_log_loss"]) * 1000],
                    fmt="o", color=colors[i], capsize=3, ms=6, label=ru)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(np.arange(len(seasons)), [f"{s}/{(int(s) + 1) % 100:02d}" for s in seasons])
    ax.set_ylabel("Δ log loss, ×10⁻³ (< 0 лучше)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Матчи новичков извне (первые 10 матчей): вариант − база", loc="left", fontsize=9.5, color=INK)
    save_chart(fig, C.FIGURES / "variants.png")


def run() -> None:
    for fn in (missing_by_league, feature_profile, sensitivity, newcomer_curves, dq_bins, phantoms, calibration, variants):
        try:
            fn()
        except Exception as exc:                                   # noqa: BLE001 - a missing input is reported
            C.log.warning("figure %s not built: %s", fn.__name__, exc)


if __name__ == "__main__":
    C.setup_logging("figures")
    run()
