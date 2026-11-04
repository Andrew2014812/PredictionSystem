"""Analysis 7 — best available price and closing-line value (walk-forward folds; 2026/27 in independent.py).

Same picks (strategy A and a few others, bettable universe), different real prices of the SAME Football-Data
snapshot:
    Bet365 opening (production price) | best of Bet365 / Pinnacle opening | best named book opening (1X2 also
    Bet&Win, William Hill, ...) | Football-Data market maximum / average opening (information: a maximum over
    books a bettor may not hold) | Bet365 / Pinnacle closing (information: not available when the Main is made).

CLV of a pick = price taken / closing price - 1, and the "fair" version: price taken x de-margined Pinnacle
closing probability - 1 (expected return if the closing Pinnacle price is the truth).
Limits: Football-Data's "opening" columns are a Friday / Tuesday snapshot, not the time the prediction was made;
closing prices exist only for 1X2, total 2.5 and one Asian line; the closing Asian line may differ from the opening
one (then there is no closing price for the pick).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as CD
from . import common as C
from . import strategies_eval as SE

PRICES = {"Bet365 откр. (продакшн)": "o_b365", "лучшая Bet365/Pinnacle откр.": "o_best",
          "Pinnacle откр.": "o_ps", "Max FD откр. (инф.)": "o_max", "Avg FD откр.": "o_avg",
          "Bet365 закр. (инф.)": "o_b365_c", "Pinnacle закр. (инф.)": "o_ps_c"}
STRATS = ("A", "F1 blend Shin (inner weight)", "C3 max EV", "B1 isotonic", "E2 candidate-count penalty",
          "G7 EV>0 (meta-model)", "B3 selection-conditional", "C2 value first EV>=0", "D2 only 1X2")


def price_table(d: pd.DataFrame, rows: np.ndarray, bet: np.ndarray) -> dict:
    x = d.iloc[rows]
    st = x["status"].to_numpy()
    dec = np.isin(st, [0, 1]) & bet
    out = {}
    # identical bets for every price: picks with every opening price present (closing separately)
    open_cols = ["o_b365", "o_best", "o_ps", "o_max", "o_avg"]
    common_open = dec & np.all([np.isfinite(x[c].to_numpy(float)) for c in open_cols], axis=0)
    common_close = dec & np.all([np.isfinite(x[c].to_numpy(float)) for c in ("o_b365", "o_b365_c", "o_ps_c")], axis=0)
    for label, col in PRICES.items():
        m = common_close if col.endswith("_c") else common_open
        o = x[col].to_numpy(float)[m]
        won = (st[m] == 1)
        prof = np.where(won, o - 1, -1.0)
        lo, hi = C.roi_ci(prof, 1000)
        out[label] = {"bets": int(m.sum()), "avg_odds": float(o.mean()) if m.any() else None,
                      "roi": float(prof.mean()) if m.any() else None, "roi_ci95": [lo, hi],
                      "subset": "closing" if col.endswith("_c") else "opening"}
    # same closing subset at the Bet365 opening price, for a like-for-like comparison
    o = x["o_b365"].to_numpy(float)[common_close]
    prof = np.where(st[common_close] == 1, o - 1, -1.0)
    out["Bet365 откр. на закр. подвыборке"] = {"bets": int(common_close.sum()), "avg_odds": float(o.mean()),
                                               "roi": float(prof.mean()), "roi_ci95": list(C.roi_ci(prof, 1000)),
                                               "subset": "closing"}
    return out


def clv(d: pd.DataFrame, rows: np.ndarray, bet: np.ndarray) -> dict:
    x = d.iloc[rows]
    st = x["status"].to_numpy()
    o, oc, qc = x["o_b365"].to_numpy(float), x["o_ps_c"].to_numpy(float), x["q_ps_c_shin"].to_numpy(float)
    push = x["p_push"].to_numpy(float)
    m = bet & np.isfinite(o) & np.isfinite(oc) & np.isfinite(qc) & np.isin(st, [0, 1])
    if not m.any():
        return {"n": 0}
    clv_odds = o[m] / oc[m] - 1
    clv_fair = o[m] * qc[m] - (1 - push[m])
    prof = np.where(st[m] == 1, o[m] - 1, -1.0)
    q = pd.qcut(clv_fair, 5, labels=False, duplicates="drop")
    quint = []
    for k in np.unique(q):
        mm = q == k
        quint.append({"quintile": int(k), "n": int(mm.sum()), "mean_clv_fair": float(clv_fair[mm].mean()),
                      "roi": float(prof[mm].mean()), "ci95": list(C.roi_ci(prof[mm], 1000))})
    return {"n": int(m.sum()), "mean_clv_odds": float(clv_odds.mean()), "share_positive_clv_odds": float((clv_odds > 0).mean()),
            "mean_clv_fair": float(clv_fair.mean()), "share_positive_clv_fair": float((clv_fair > 0).mean()),
            "roi_same_bets": float(prof.mean()), "corr_clv_profit": float(np.corrcoef(clv_fair, prof)[0, 1]),
            "quintiles": quint}


def run() -> dict:
    picks = pd.read_parquet(C.CACHE / "strategy_picks.parquet")
    out = {}
    for s in C.FOLDS:
        d = SE.universe(CD.historical(s))
        res = {}
        for name in STRATS:
            p = picks.loc[(picks["mode"] == "bettable") & (picks["season"] == s) & (picks["strategy"] == name)]
            if p.empty:
                continue
            rows, bet = p["row"].to_numpy(int), p["bet"].to_numpy(bool)
            res[name] = {"prices": price_table(d, rows, bet), "clv": clv(d, rows, bet)}
        out[str(s)] = res
        C.log.info("prices / CLV %s", s)
    C.save_json("prices_clv.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    labels = list(PRICES.keys())
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for k, s in enumerate(C.FOLDS):
        t = out[str(s)]["A"]["prices"]
        ax.bar(np.arange(len(labels)) + (k - 1) * 0.27, [t[l]["roi"] * 100 for l in labels], 0.27, color=PALETTE[k],
               label=C.period_label(s))
        for i, l in enumerate(labels):
            lo, hi = t[l]["roi_ci95"]
            ax.plot([i + (k - 1) * 0.27] * 2, [lo * 100, hi * 100], color="#333", lw=0.7)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=7.5)
    ax.axhline(0, color="#555", lw=0.6), ax.set_ylabel("ROI, %"), ax.legend(fontsize=7)
    ax.set_title("ROI одних и тех же Main (A) при разных реальных ценах одного снимка Football-Data", fontsize=10)
    save_chart(fig, C.FIGURES / "prices_roi_by_price.png")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, s in enumerate(C.FOLDS):
        qn = out[str(s)]["A"]["clv"]["quintiles"]
        ax.errorbar([x["mean_clv_fair"] * 100 for x in qn], [x["roi"] * 100 for x in qn],
                    yerr=[[(x["roi"] - x["ci95"][0]) * 100 for x in qn], [(x["ci95"][1] - x["roi"]) * 100 for x in qn]],
                    marker="o", ms=3.5, capsize=2, color=PALETTE[k], label=C.period_label(s))
    ax.plot([-15, 15], [-15, 15], color="#888", ls="--", lw=0.8)
    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xlabel("CLV к закрытию Pinnacle (без маржи), %"), ax.set_ylabel("ROI, %"), ax.legend(fontsize=7)
    ax.set_title("ROI Main по квинтилям CLV", fontsize=10)
    save_chart(fig, C.FIGURES / "prices_clv_quintiles.png")


if __name__ == "__main__":
    C.setup_logging("prices")
    run()
