"""Why Risk loses while its model EV is positive: diagnostics on the production Risk and its candidate pool.

All functions take prepared tables (strategies.prepare) and return JSON-ready dicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

P_BANDS = [0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
ODDS_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0, 10.0, 1000.0]


def _band_label(edges, i):
    hi = edges[i + 1]
    return f"{edges[i]:.2f}–{hi:.2f}" if hi < 100 else f"≥{edges[i]:.2f}"


def ev_decomposition(picks: pd.DataFrame) -> dict:
    """Model EV - realised = (model EV - market EV) + (market EV - realised), on priced Risk bets with a full book."""
    b = picks.loc[np.isfinite(picks["stake_profit"].to_numpy(float))]
    full = b.loc[b["p_mkt_u"].notna()]
    out = {"bets": int(len(b)), "with_book": int(len(full)), "avg_odds": float(b["odds"].mean()),
           "mean_p": float(b["p"].mean()), "mean_pc": float(b["pc"].mean()),
           "hit": float(b.loc[b["status"].isin(["WON", "LOST"]), "y"].mean()),
           "model_ev": float(b["ev"].mean()), "realised": float(b["stake_profit"].mean()),
           "realised_ci": C.mean_ci(b["stake_profit"]),
           "roi": float(np.nanmean(b["profit"])), "roi_ci": C.mean_ci(b["profit"])}
    if len(full):
        out.update({"book_model_ev": float(full["ev"].mean()), "book_market_ev": float(full["ev_mkt"].mean()),
                    "book_realised": float(full["stake_profit"].mean()),
                    "book_realised_ci": C.mean_ci(full["stake_profit"]),
                    "model_minus_market": float((full["ev"] - full["ev_mkt"]).mean()),
                    "market_minus_realised": float((full["ev_mkt"] - full["stake_profit"]).mean()),
                    "market_minus_realised_ci": C.mean_ci(full["ev_mkt"] - full["stake_profit"]),
                    "mean_q": float(full["p_mkt_u"].mean()), "mean_p_book": float(full["p"].mean())})
    return out


def calib_by_band(df: pd.DataFrame, p_col: str = "pc", edges=P_BANDS, min_n: int = 30) -> list[dict]:
    """Decided rows: mean stated probability vs hit rate per probability band (no-refund scale)."""
    d = df.loc[df["decided"]]
    x = d[p_col].to_numpy(float)
    k = np.digitize(x, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = k == i
        if m.sum() < min_n:
            continue
        y = d["y"].to_numpy(float)[m]
        hit = float(y.mean())
        se = float(np.sqrt(np.sum(x[m] * (1 - x[m]))) / m.sum())
        row = {"band": _band_label(edges, i), "n": int(m.sum()), "mean_p": float(x[m].mean()), "hit": hit,
               "gap": float(x[m].mean() - hit), "z": float((hit - x[m].mean()) / se) if se > 0 else None}
        qc = d["qc"].to_numpy(float)[m]
        if np.isfinite(qc).sum() >= min_n:
            okq = np.isfinite(qc)
            row.update({"n_q": int(okq.sum()), "mean_q": float(qc[okq].mean()), "hit_q": float(y[okq].mean()),
                        "mean_p_q": float(x[m][okq].mean())})
        out.append(row)
    return out


def favourite_longshot(df: pd.DataFrame) -> list[dict]:
    """Every priced selection of a full book: blind ROI, implied, de-vigged and model probability by odds band."""
    d = df.loc[df["priced"] & df["p_mkt"].notna() & df["decided"]]
    odds = d["odds"].to_numpy(float)
    k = np.digitize(odds, ODDS_BANDS) - 1
    out = []
    for i in range(len(ODDS_BANDS) - 1):
        m = k == i
        if m.sum() < 50:
            continue
        g = d.loc[m]
        pr = g["profit"].to_numpy(float)
        out.append({"band": _band_label(ODDS_BANDS, i), "n": int(m.sum()), "avg_odds": float(g["odds"].mean()),
                    "implied": float((1 / g["odds"]).mean()), "q": float(g["qc"].mean()), "model": float(g["pc"].mean()),
                    "hit": float(g["y"].mean()), "roi": float(pr.mean()), "roi_ci": C.mean_ci(pr),
                    "q_gap": float(g["qc"].mean() - g["y"].mean()), "model_gap": float(g["pc"].mean() - g["y"].mean()),
                    "model_ev": float(np.mean(g["pc"] * g["odds"] - 1))})
    return out


def margins(df: pd.DataFrame) -> list[dict]:
    """Book overround per market group (one row per book) and the share of priced rows by odds band."""
    d = df.loc[df["overround"].notna()].drop_duplicates(["match_id", "market"])
    out = []
    for g, x in d.groupby("group"):
        out.append({"group": g, "books": int(len(x)), "overround": float(x["overround"].mean()),
                    "overround_p10": float(x["overround"].quantile(0.1)),
                    "overround_p90": float(x["overround"].quantile(0.9))})
    return out


def margin_by_band(df: pd.DataFrame) -> list[dict]:
    """Expected loss per unit at the de-vigged price (1 - q odds) by odds band: the margin a bettor pays when the
    de-vigged price is right — proportional de-vig spreads it evenly in relative terms by construction."""
    d = df.loc[df["priced"] & df["p_mkt"].notna()]
    k = np.digitize(d["odds"].to_numpy(float), ODDS_BANDS) - 1
    out = []
    for i in range(len(ODDS_BANDS) - 1):
        m = k == i
        if m.sum() < 50:
            continue
        g = d.loc[m]
        out.append({"band": _band_label(ODDS_BANDS, i), "n": int(m.sum()),
                    "margin_at_q": float(np.mean(1 - g["qc"] * g["odds"])),
                    "overround": float(g["overround"].mean())})
    return out


def selection_effect(df: pd.DataFrame, picks: pd.DataFrame) -> dict:
    """Chosen Risk vs not-chosen pool candidates at the same stated probability; dependence on the pool size."""
    pool = df.loc[df["cand"] & (df["p"] >= 0.12) & df["decided"]].copy()
    chosen_key = set(zip(picks["match_id"], picks["market"], picks["selection"]))
    pool["chosen"] = [k in chosen_key for k in zip(pool["match_id"], pool["market"], pool["selection"])]
    out = {}
    for label, sub in (("all", pool), ("priced", pool.loc[pool["priced"]])):
        bins = np.digitize(sub["pc"].to_numpy(float), P_BANDS)
        ch, rest = sub["chosen"].to_numpy(), ~sub["chosen"].to_numpy()
        w = np.zeros(len(sub))
        for b in np.unique(bins[ch]):
            nc, nr = (ch & (bins == b)).sum(), (rest & (bins == b)).sum()
            if nr:
                w[rest & (bins == b)] = nc / nr
        err = sub["pc"].to_numpy(float) - sub["y"].to_numpy(float)
        out[label] = {"n_chosen": int(ch.sum()), "n_rest": int(rest.sum()),
                      "gap_chosen": float(err[ch].mean()) if ch.any() else None,
                      "gap_chosen_ci": C.mean_ci(err[ch]) if ch.sum() > 1 else None,
                      "gap_rest_reweighted": float(np.sum(w[rest] * err[rest]) / max(w[rest].sum(), 1e-9)),
                      "gap_rest": float(err[rest].mean()) if rest.any() else None}
        if label == "priced":
            ev = sub["ev"].to_numpy(float)
            out[label].update({"ev_chosen": float(np.nanmean(ev[ch])), "ev_rest_reweighted":
                               float(np.nansum(w[rest] * ev[rest]) / max(w[rest][np.isfinite(ev[rest])].sum(), 1e-9))})
    dec = picks.loc[picks["status"].isin(["WON", "LOST"])]
    qs = pd.qcut(dec["ncand"].rank(method="first"), 4, labels=False)
    rows = []
    for k, g in dec.groupby(qs):
        h = float(g["y"].mean())
        rows.append({"quartile": int(k), "range": f"{int(g['ncand'].min())}–{int(g['ncand'].max())}",
                     "n": int(len(g)), "mean_p": float(g["pc"].mean()), "hit": h, "gap": float(g["pc"].mean() - h),
                     "se": float(np.sqrt(h * (1 - h) / len(g)))})
    out["by_pool_size"] = rows
    return out


def odds_dependence(picks: pd.DataFrame) -> list[dict]:
    b = picks.loc[np.isfinite(picks["profit"].to_numpy(float))]
    k = np.digitize(b["odds"].to_numpy(float), ODDS_BANDS) - 1
    out = []
    for i in range(len(ODDS_BANDS) - 1):
        m = k == i
        if m.sum() < 20:
            continue
        g = b.loc[m]
        pr = g["profit"].to_numpy(float)
        out.append({"band": _band_label(ODDS_BANDS, i), "n": int(m.sum()), "avg_odds": float(g["odds"].mean()),
                    "mean_p": float(g["pc"].mean()), "hit": float(g["y"].mean()),
                    "gap": float(g["pc"].mean() - g["y"].mean()), "model_ev": float(g["ev"].mean()),
                    "q": float(g["p_mkt_u"].mean()) if g["p_mkt_u"].notna().any() else None,
                    "roi": float(pr.mean()), "roi_ci": C.mean_ci(pr), "small": bool(m.sum() < 100)})
    return out


def breakdown(picks: pd.DataFrame, by: str) -> list[dict]:
    out = []
    for k, g in picks.groupby(by):
        dec = g.loc[g["status"].isin(["WON", "LOST"])]
        b = g.loc[np.isfinite(g["profit"].to_numpy(float))]
        if not len(dec):
            continue
        out.append({"key": str(k), "n": int(len(dec)), "mean_p": float(dec["pc"].mean()), "hit": float(dec["y"].mean()),
                    "gap": float(dec["pc"].mean() - dec["y"].mean()), "bets": int(len(b)),
                    "avg_odds": float(b["odds"].mean()) if len(b) else None,
                    "roi": float(b["profit"].mean()) if len(b) else None,
                    "roi_ci": C.mean_ci(b["profit"]) if len(b) > 1 else None,
                    "model_ev": float(b["ev"].mean()) if len(b) else None, "small": bool(len(b) < 100)})
    return out
