"""Metrics of a set of Main picks on one season / universe, with paired bootstrap against the baseline.

All variants of a season are evaluated on the same matches with the SAME bootstrap resamples (by match), so
differences are paired. Calibration uses decided picks (won / lost); ROI follows production: a priced pick
that is won or lost is one unit stake, a refund is not a stake. The EV check uses the per-stake return
including refunds (profit 0), which is what EV = p_win odds - (1 - p_push) predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

NCAND_BINS = {"site": [0, 17, 19, 21, 22, 100], "bet": [0, 1, 2, 3, 4, 5, 15, 18, 20, 100]}


class Boot:
    def __init__(self, n: int, n_boot: int = C.N_BOOT, seed: int = C.SEED):
        self.n = n
        self.chunks = [i.astype(np.int32) for i in C.boot_idx(n, n_boot, seed)]

    def dist(self, fn, *arrays) -> np.ndarray:
        out = []
        for idx in self.chunks:
            with np.errstate(all="ignore"):
                out.append(fn(*[a[idx] for a in arrays]))
        return np.concatenate(out)


def _nanmean(x):
    return np.nanmean(x, axis=1)


def arrays(f: pd.DataFrame, n: int) -> dict:
    out = {}
    mi = f["mi"].to_numpy()
    for col in ("pc", "y", "profit", "stake_profit", "ev"):
        a = np.full(n, np.nan)
        a[mi] = f[col].to_numpy(float)
        out[col] = a
    pc = np.clip(out["pc"], 1e-6, 1 - 1e-6)
    y = out["y"]
    out["err"] = np.where(np.isfinite(y), pc - y, np.nan)
    out["ll"] = np.where(np.isfinite(y), -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), np.nan)
    out["br"] = np.where(np.isfinite(y), (pc - y) ** 2, np.nan)
    out["evgap"] = np.where(np.isfinite(out["stake_profit"]), out["ev"] - out["stake_profit"], np.nan)
    return out


def metrics(f: pd.DataFrame, a: dict, boot: Boot, base: dict | None = None, same_pick: bool = False) -> dict:
    dec = f.loc[np.isfinite(f["y"].to_numpy(float))]
    bets = f.loc[np.isfinite(f["profit"].to_numpy(float))].sort_values("date")
    g = boot.dist(_nanmean, a["err"])
    r = boot.dist(_nanmean, a["profit"])
    out = {"settled": int(len(dec)), "mean_p": float(dec["pc"].mean()), "hit": float(dec["y"].mean()),
           "gap": float(np.nanmean(a["err"])), "gap_ci": C.ci(g), "gap_p": C.p_two_sided(g),
           "ece": C.ece(dec["y"], dec["pc"]), "log_loss": float(np.nanmean(a["ll"])), "brier": float(np.nanmean(a["br"])),
           "bets": int(len(bets)), "bet_hit": float(bets["y"].mean()) if len(bets) else np.nan,
           "avg_odds": float(bets["odds"].mean()) if len(bets) else np.nan,
           "roi": float(np.nanmean(a["profit"])) if len(bets) else np.nan, "roi_ci": C.ci(r),
           "max_drawdown": C.max_drawdown(bets["profit"].to_numpy(float)),
           "mean_ev": float(np.nanmean(a["ev"])) if len(bets) else np.nan,
           "ev_gap": float(np.nanmean(a["evgap"])) if len(bets) else np.nan,
           "ev_gap_ci": C.ci(boot.dist(_nanmean, a["evgap"])) if len(bets) else [np.nan, np.nan],
           "same_pick_as_base": bool(same_pick)}
    if base is not None:
        dg = boot.dist(lambda x, b: np.abs(np.nanmean(x, 1)) - np.abs(np.nanmean(b, 1)), a["err"], base["err"])
        dr = boot.dist(lambda x, b: np.nanmean(x, 1) - np.nanmean(b, 1), a["profit"], base["profit"])
        out.update({"d_absgap": float(abs(np.nanmean(a["err"])) - abs(np.nanmean(base["err"]))),
                    "d_absgap_ci": C.ci(dg), "d_absgap_p": C.p_two_sided(dg),
                    "d_roi": float(np.nanmean(a["profit"]) - np.nanmean(base["profit"])),
                    "d_roi_ci": C.ci(dr), "d_roi_p": C.p_two_sided(dr)})
        if same_pick:
            dl = boot.dist(lambda x, b: np.nanmean(x, 1) - np.nanmean(b, 1), a["ll"], base["ll"])
            out.update({"d_logloss": float(np.nanmean(a["ll"]) - np.nanmean(base["ll"])), "d_logloss_ci": C.ci(dl),
                        "d_logloss_p": C.p_two_sided(dl)})
    return out


def breakdown(f: pd.DataFrame, by: str, min_n: int = 30) -> list[dict]:
    dec = f.loc[np.isfinite(f["y"].to_numpy(float))]
    rows = []
    for k, g in dec.groupby(by, observed=True):
        hit = float(g["y"].mean())
        bets = g.loc[np.isfinite(g["profit"].to_numpy(float))]
        rows.append({"key": str(k), "n": int(len(g)), "mean_p": float(g["pc"].mean()), "hit": hit,
                     "gap": float(g["pc"].mean() - hit), "se": float(np.sqrt(max(hit * (1 - hit), 1e-9) / len(g))),
                     "bets": int(len(bets)), "roi": float(bets["profit"].mean()) if len(bets) else None,
                     "small": bool(len(g) < min_n)})
    return rows


def by_ncand(f: pd.DataFrame, kind: str) -> list[dict]:
    edges = NCAND_BINS[kind]
    f = f.assign(nb=pd.cut(f["ncand"], edges, right=True, include_lowest=True).astype(str))
    rows = breakdown(f, "nb")
    order = {str(c): i for i, c in enumerate(pd.cut(pd.Series(edges[1:]), edges, right=True,
                                                    include_lowest=True).astype(str))}
    return sorted(rows, key=lambda r: order.get(r["key"], 99))


def ev_bins(f: pd.DataFrame, ev_col: str, edges=(-1, -0.10, -0.05, 0.0, 0.05, 0.10, 10)) -> list[dict]:
    b = f.loc[np.isfinite(f["stake_profit"].to_numpy(float))]
    e = b[ev_col].to_numpy(float)
    k = np.digitize(e, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = k == i
        if m.sum() >= 30:
            pr = b["stake_profit"].to_numpy(float)[m]
            lo, hi = C.ci(C.boot_stat(lambda x: np.nanmean(x, 1), pr, n_boot=1000))
            out.append({"band": f"[{edges[i]:+.2f}; {edges[i + 1]:+.2f})", "n": int(m.sum()),
                        "stated_ev": float(e[m].mean()), "realised": float(pr.mean()), "ci": [lo, hi]})
    return out


def ev_slope(f: pd.DataFrame, ev_col: str) -> dict:
    b = f.loc[np.isfinite(f["stake_profit"].to_numpy(float)) & np.isfinite(f[ev_col].to_numpy(float))]
    x, y = b[ev_col].to_numpy(float), b["stake_profit"].to_numpy(float)
    if len(x) < 30 or np.std(x) == 0:
        return {"n": int(len(x))}
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * resid.var()
    return {"n": int(len(x)), "intercept": float(beta[0]), "slope": float(beta[1]),
            "slope_se": float(np.sqrt(cov[1, 1])), "mean_ev": float(x.mean()), "mean_return": float(y.mean())}
