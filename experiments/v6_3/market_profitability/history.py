"""Historical profitability of markets, known before each match (expanding window by day).

For a match played on day d the history is every bet settled on a day < d of the same universe, plus (optionally)
the whole 2025/26 backtest as a seed (Football-Data prices: 1X2, total 2.5, one Asian line). The season's final
ROI is never used for its early matches.

Cells: ``group`` (1X2, TOTAL, ...), ``market`` (TOTAL_2.5, HANDICAP_-1, ...) or ``group_odds`` (group x price band).
Sources: ``cand`` — flat ROI of every Main-eligible candidate with a real price (independent of the Main rule);
``main`` — ROI of production's Main bets in the cell.

Empirical-Bayes shrinkage (normal approximation): with n_j bets and ROI r_j in cell j, pooled ROI m and per-bet
variance s², the between-cell variance is τ² = max(0, Var_w(r_j) − mean(s²/n_j)); k = s²/τ² (fixed k optional);
r̃_j = (n_j r_j + k m)/(n_j + k). A cell without history gets m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ODDS_BANDS = [1.0, 1.4, 1.7, 2.2, 1000.0]
BAND = (0.40, 0.85)


def cell_of(c: pd.DataFrame, cell: str) -> pd.Series:
    if cell == "group":
        return c["group"].astype(str)
    if cell == "market":
        return c["market"].astype(str)
    if cell == "group_odds":
        b = pd.cut(c["eff_odds"], ODDS_BANDS, right=False, labels=["<1.4", "1.4-1.7", "1.7-2.2", ">=2.2"])
        return c["group"].astype(str) + "|" + b.astype(str)
    raise ValueError(cell)


def events(c: pd.DataFrame, source: str, cell: str, base_main: pd.DataFrame | None = None) -> pd.DataFrame:
    """Settled bets (date, cell, profit) of a universe."""
    if source == "cand":
        ok = ((~c["excluded"]) & c["has_odds"] & c["status"].isin(["WON", "LOST"])
              & c["probability"].between(*BAND))
        e = c.loc[ok]
    else:
        e = base_main.loc[base_main["bet"] == 1]
    return pd.DataFrame({"date": pd.to_datetime(e["date"]).to_numpy(), "cell": cell_of(e, cell).to_numpy(),
                         "profit": e["profit"].to_numpy(float)})


def _shrink(S: pd.Series, n: pd.Series, SS: pd.Series, k_fixed: float | None) -> tuple[pd.Series, float, float, pd.Series]:
    N = n.sum()
    if N == 0:
        z = pd.Series(0.0, index=S.index)
        return z, 0.0, np.inf, z
    m = S.sum() / N
    s2 = max(SS.sum() / N - m ** 2, 1e-6)
    has = n > 0
    r = (S / n.where(has)).fillna(m)
    if k_fixed is None:
        w = n[has] / N
        var_w = float((w * (r[has] - m) ** 2).sum())
        tau2 = max(0.0, var_w - float((w * s2 / n[has]).sum()))
        k = s2 / tau2 if tau2 > 0 else np.inf
    else:
        k = float(k_fixed)
    shr = (S + (k * m if np.isfinite(k) else 0)) / (n + k) if np.isfinite(k) else pd.Series(m, index=S.index)
    se = np.sqrt(s2 / n.where(has))
    return shr.fillna(m), m, k, se


def expanding(c: pd.DataFrame, ev_cur: pd.DataFrame, ev_seed: pd.DataFrame | None, cell: str,
              k_fixed: float | None = None) -> pd.DataFrame:
    """Per (date, cell) of ``c``: shrunk ROI, raw ROI, n, pooled ROI m, k, standard error — history < date."""
    dates = np.sort(pd.to_datetime(c["date"]).unique())
    cells = sorted(set(cell_of(c, cell).unique()) | set(ev_cur["cell"].unique())
                   | (set(ev_seed["cell"].unique()) if ev_seed is not None else set()))
    g = ev_cur.assign(sq=ev_cur["profit"] ** 2).groupby(["date", "cell"]).agg(S=("profit", "sum"),
                                                                              n=("profit", "size"), SS=("sq", "sum"))
    seed = pd.DataFrame(0.0, index=cells, columns=["S", "n", "SS"])
    if ev_seed is not None and len(ev_seed):
        sd = ev_seed.assign(sq=ev_seed["profit"] ** 2).groupby("cell").agg(
            S=("profit", "sum"), n=("profit", "size"), SS=("sq", "sum"))
        seed = seed.add(sd.reindex(cells).fillna(0), fill_value=0)
    acc = seed.copy()
    by_date = {d: x.droplevel(0) for d, x in g.groupby(level=0)}
    rows = []
    for d in dates:
        shr, m, k, se = _shrink(acc["S"], acc["n"], acc["SS"], k_fixed)
        raw = (acc["S"] / acc["n"].where(acc["n"] > 0))
        rows.append(pd.DataFrame({"date": d, "cell": cells, "roi_shr": shr.to_numpy(), "roi_raw": raw.to_numpy(),
                                  "n": acc["n"].to_numpy(), "m": m, "k": k, "se": se.to_numpy()}))
        if d in by_date:                      # matches of day d enter the history of later days only
            acc = acc.add(by_date[d].reindex(cells).fillna(0), fill_value=0)
    return pd.concat(rows, ignore_index=True)


def attach(c: pd.DataFrame, table: pd.DataFrame, cell: str) -> pd.DataFrame:
    key = pd.DataFrame({"date": pd.to_datetime(c["date"]).to_numpy(), "cell": cell_of(c, cell).to_numpy()})
    return key.merge(table, on=["date", "cell"], how="left")
