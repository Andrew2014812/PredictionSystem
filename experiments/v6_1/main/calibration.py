"""Analysis 1 — calibration of the chosen Main picks (strategy A) on the walk-forward folds.

Stated probability vs hit rate, Brier, log loss, ECE — overall and by market group, league, season, odds band,
probability band, line and side, for the Main picks of A in two universes:

* production  — every candidate (as on the site); prices only where Football-Data has them;
* bettable    — only candidates with a real Bet365 price (1X2, total 2.5, the Asian line).

For comparison, the calibration of ALL candidates in the probability band (not only the chosen one) separates
"the model is miscalibrated" from "choosing the maximum makes it look over-confident".
The 2026/27 counterpart is computed in independent.py after the pre-registration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE

ODDS_BANDS = [1.0, 1.30, 1.50, 1.70, 2.00, 2.50, 10.0]
PROB_BANDS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.86, 1.0]


def _cond(df):
    return df["p"].to_numpy(float) / np.clip(1 - df["p_push"].to_numpy(float), 1e-6, None)


def breakdown(o: pd.DataFrame, by: str | list) -> list[dict]:
    dec = o.loc[o["decided"]].copy()
    dec["pc"] = _cond(dec)
    rows = []
    for key, g in dec.groupby(by, observed=True):
        s = C.binary_scores(g["won"], g["pc"])
        bets = g.loc[g["priced"]]
        s.update({"key": key if not isinstance(key, tuple) else " | ".join(map(str, key)),
                  "bets": int(len(bets)), "avg_odds": float(bets["odds"].mean()) if len(bets) else None,
                  "roi": float(bets["profit"].mean()) if len(bets) else None,
                  "se_gap": float(np.sqrt(max(s["hit_rate"] * (1 - s["hit_rate"]), 1e-9) / max(s["n"], 1)))})
        rows.append(s)
    return rows


def annotate(o: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    o["odds_band"] = pd.cut(o["odds"], ODDS_BANDS, right=False).astype(str)
    o["prob_band"] = pd.cut(o["p"], PROB_BANDS, right=False).astype(str)
    fmt = [("" if x != x else (f"{x:+g}" if g == "HANDICAP" else f"{x:g}")) for g, x in zip(o["group"], o["line"])]
    o["line_key"] = np.where(o["line"].notna(), o["group"] + " " + pd.Series(fmt, index=o.index), o["group"])
    side = np.where(o["group"].isin(["TOTAL", "CORNERS"]), o["selection"],
                    np.where(o["group"] == "1X2", o["selection"], np.where(o["group"] == "HANDICAP",
                                                                            "фора " + o["selection"], o["selection"])))
    o["side"] = o["group"] + " " + side
    return o


def season_tables(season: int, mode: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = CD.historical(season)
    priced_only = mode == "bettable"
    if priced_only:
        d = SE.universe(d)
    d = SE.add_ncand(d, priced_only)
    idx = ST.choose(d, ST.CURRENT.with_(priced_only=priced_only))
    o = annotate(ST.outcome(d, idx))
    dec = o.loc[o["decided"]]
    res = {"overall": C.binary_scores(dec["won"], _cond(dec)),
           "overall_bets": C.binary_scores(dec.loc[dec["priced"], "won"], _cond(dec.loc[dec["priced"]])),
           "reliability": C.reliability_bins(dec["won"], _cond(dec)),
           "reliability_bets": C.reliability_bins(dec.loc[dec["priced"], "won"], _cond(dec.loc[dec["priced"]])),
           "by_group": breakdown(o, "group"), "by_league": breakdown(o, "league"),
           "by_odds_band": breakdown(o.loc[o["priced"]], "odds_band"), "by_prob_band": breakdown(o, "prob_band"),
           "by_line": breakdown(o, "line_key"), "by_side": breakdown(o, "side"),
           "reliability_by_group": {g: C.reliability_bins(x["won"], _cond(x)) for g, x in dec.groupby("group")}}
    # all candidates in the probability band (the model's calibration irrespective of the choice)
    p = d["p"].to_numpy(float)
    cand = d.loc[(p >= 0.40) & (p <= 0.85) & ~d["group"].isin(ST.EXCLUDED) & d["status"].isin([0, 1])]
    if priced_only:
        cand = cand.loc[cand["odds"].notna()]
    cw = (cand["status"] == 1).astype(float)
    res["all_candidates"] = C.binary_scores(cw, _cond(cand))
    res["all_candidates_by_group"] = {g: C.binary_scores((x["status"] == 1).astype(float), _cond(x))
                                      for g, x in cand.groupby("group")}
    res["all_candidates_reliability"] = C.reliability_bins(cw, _cond(cand))
    # market (de-margined Bet365, Shin) on the same picks: is the bookmaker better calibrated on them?
    q = d["q_prod_shin"].to_numpy(float)[o["row"].to_numpy()]
    mq = o["decided"].to_numpy() & np.isfinite(q)
    res["market_on_picks"] = C.binary_scores(o["won"].to_numpy()[mq],
                                             q[mq] / np.clip(1 - o["p_push"].to_numpy()[mq], 1e-6, None))
    res["model_on_same_picks"] = C.binary_scores(o["won"].to_numpy()[mq], _cond(o.loc[mq]))
    return res, o, cand


def run() -> dict:
    out = {}
    for mode in ("production", "bettable"):
        out[mode] = {}
        for s in C.FOLDS:
            res, o, cand = season_tables(s, mode)
            out[mode][str(s)] = res
            if mode == "production" and s == 2024:
                o.to_parquet(C.CACHE / "calib_picks_2024.parquet")
    C.save_json("calibration.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    # 1. reliability of A picks per season, both universes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode, title in zip(axes, ("production", "bettable"),
                               ("Все кандидаты (как на сайте)", "Только кандидаты с реальной ценой")):
        for k, s in enumerate(C.FOLDS):
            r = out[mode][str(s)]["reliability"]
            x = [b["predicted"] for b in r]
            y = [b["observed"] for b in r]
            e = [1.96 * b["se"] for b in r]
            ax.errorbar(x, y, yerr=e, marker="o", ms=3.5, lw=1.2, capsize=2, label=C.period_label(s))
        ax.plot([0.35, 0.9], [0.35, 0.9], color="#888", lw=0.8, ls="--")
        ax.set_xlim(0.35, 0.9), ax.set_ylim(0.3, 0.9)
        ax.set_xlabel("Заявленная вероятность Main"), ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("Фактическая доля выигрышей")
    axes[0].legend(fontsize=8)
    fig.suptitle("Диаграмма надёжности выбранных Main (стратегия A), бины по 0.05, 95 % ДИ", fontsize=10)
    from scripts.report_kit import save_chart
    save_chart(fig, C.FIGURES / "calib_reliability_A.png")
    # 2. gap by group: picks vs all candidates (production universe)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    groups = ["TOTAL", "1X2", "HANDICAP"]
    width = 0.13
    for k, s in enumerate(C.FOLDS):
        r = out["production"][str(s)]
        picks = {x["key"]: x for x in r["by_group"]}
        allc = r["all_candidates_by_group"]
        xs = np.arange(len(groups))
        ax.bar(xs + (2 * k - 2.5) * width, [picks.get(g, {}).get("gap", np.nan) * 100 for g in groups], width,
               color=C_PAL(k), label=f"Main {C.period_label(s)}")
        ax.bar(xs + (2 * k - 1.5) * width, [allc.get(g, {}).get("gap", np.nan) * 100 for g in groups], width,
               color=C_PAL(k), alpha=0.4, hatch="//", label=f"все кандидаты {C.period_label(s)}")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(np.arange(len(groups)), [C.GROUP_RU[g] for g in groups])
    ax.set_ylabel("Заявлено − факт, п.п.")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Переоценка вероятности: выбранный Main и все кандидаты диапазона 0.40–0.85", fontsize=10)
    save_chart(fig, C.FIGURES / "calib_gap_groups.png")
    # 3. reliability by group (production, pooled folds 2023+2024)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, g in zip(axes, groups):
        for k, s in enumerate(C.FOLDS):
            r = out["production"][str(s)]["reliability_by_group"].get(g, [])
            ax.plot([b["predicted"] for b in r], [b["observed"] for b in r], marker="o", ms=3, lw=1,
                    label=C.period_label(s), color=C_PAL(k))
        ax.plot([0.35, 0.9], [0.35, 0.9], color="#888", lw=0.8, ls="--")
        ax.set_title(C.GROUP_RU[g], fontsize=9.5), ax.set_xlabel("Заявленная вероятность")
    axes[0].set_ylabel("Доля выигрышей"), axes[0].legend(fontsize=7)
    fig.suptitle("Надёжность Main по группам рынков (все кандидаты, стратегия A)", fontsize=10)
    save_chart(fig, C.FIGURES / "calib_reliability_groups.png")


def C_PAL(k):
    from scripts.report_kit import PALETTE
    return PALETTE[k % len(PALETTE)]


if __name__ == "__main__":
    C.setup_logging("calibration")
    run()
