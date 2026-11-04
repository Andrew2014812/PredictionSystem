"""Analysis 4 — the shown Main prediction vs the betting decision.

The Main stays the one selection per match the site shows (strategy A); a separate decision says "stake" or
"skip" at the real price. Thresholds on raw, calibrated (inner isotonic) and meta-model EV, odds windows and
probability bands — number of bets, total profit, ROI with 95 % bootstrap interval, per season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE
from . import transforms as T

EV_T = (None, 0.0, 0.02, 0.05, 0.10, 0.15)


def decisions(d: pd.DataFrame, fit: SE.Fitted) -> dict[str, np.ndarray]:
    d = SE.add_ncand(d, fit.priced_only)
    idx = ST.choose(d, fit.params)
    rows = d.iloc[idx]
    odds = rows["odds"].to_numpy(float)
    push = rows["p_push"].to_numpy(float)
    priced = np.isfinite(odds)
    p_raw = rows["p"].to_numpy(float)
    p_iso = np.clip(fit.iso.transform(rows), 0.001, 0.999)
    q = T.market_q(rows, fit.book, "shin")
    X = T.meta_frame(rows, "p", q, rows["ncand"].to_numpy())[:, fit.meta_cols]
    p_meta = np.where(rows["group"].isin(fit.meta_groups).to_numpy(), fit.meta.predict(X), np.nan)
    out = {}
    for name, p in (("сырая p", p_raw), ("калибр. p", p_iso), ("мета-модель", p_meta)):
        e = p * odds - (1 - push)
        for t in EV_T:
            label = f"все с ценой" if t is None else f"EV>{t * 100:g}% ({name})"
            if t is None and name != "сырая p":
                continue
            out[label] = priced if t is None else priced & (e > t)
    for lo, hi in ((1.0, 1.4), (1.4, 1.7), (1.7, 2.1), (2.1, 3.0), (3.0, 99)):
        out[f"коэф. {lo:.1f}–{hi:.1f}"] = priced & (odds >= lo) & (odds < hi)
    for lo, hi in ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.86)):
        out[f"p {lo:.1f}–{hi:.2f}"] = priced & (p_raw >= lo) & (p_raw < hi)
    out["_rows"] = idx
    return out


def run() -> dict:
    out = {}
    for mode in ("production", "bettable"):
        out[mode] = {}
        for s in C.FOLDS:
            inner, outer = CD.historical(s - 1), CD.historical(s)
            if mode == "bettable":
                inner, outer = SE.universe(inner), SE.universe(outer)
            fit = SE.Fitted(inner, mode == "bettable")
            dec = decisions(outer, fit)
            idx = dec.pop("_rows")
            o = ST.outcome(outer, idx)
            res = {}
            for label, m in dec.items():
                b = o.loc[m & o["decided"].to_numpy()]
                prof = b["profit"].to_numpy(float)
                lo, hi = C.roi_ci(prof) if len(prof) > 1 else (np.nan, np.nan)
                res[label] = {"bets": int(len(b)), "profit": float(np.nansum(prof)),
                              "roi": float(np.nanmean(prof)) if len(prof) else None, "roi_ci95": [lo, hi],
                              "hit": float(b["won"].mean()) if len(b) else None,
                              "avg_odds": float(b["odds"].mean()) if len(b) else None,
                              "share_of_mains": float(len(b) / max(o["decided"].sum(), 1))}
            out[mode][str(s)] = res
            C.log.info("betting %s %s", mode, s)
    C.save_json("betting.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    labels = [l for l in out["production"][str(C.FOLDS[0])] if l.startswith("EV") or l == "все с ценой"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, s in enumerate(C.FOLDS):
        r = out["production"][str(s)]
        vals = [r[l]["roi"] * 100 if r[l]["roi"] is not None else np.nan for l in labels]
        ax.bar(np.arange(len(labels)) + (k - 1) * 0.27, vals, 0.27, color=PALETTE[k], label=C.period_label(s))
        for i, l in enumerate(labels):
            lo, hi = r[l]["roi_ci95"]
            if lo == lo and lo is not None:
                ax.plot([i + (k - 1) * 0.27] * 2, [lo * 100, hi * 100], color="#333", lw=0.6)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="#555", lw=0.6), ax.set_ylabel("ROI, %"), ax.legend(fontsize=7)
    ax.set_ylim(-40, 40)
    ax.set_title("Решение о ставке поверх показанного Main (A): ROI при порогах EV", fontsize=10)
    save_chart(fig, C.FIGURES / "betting_thresholds.png")


if __name__ == "__main__":
    C.setup_logging("betting")
    run()
