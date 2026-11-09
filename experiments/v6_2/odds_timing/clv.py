"""Closing-line value (CLV) of the stored 2026/27 Main / Risk predictions.

For every stored Main / Risk with a price (audit classes A1, A2, B, C):

    CLV_odds = price taken / closing price of the SAME bookmaker - 1
               (OddsPapi archive closing quote; for Bet365 1X2 / total 2.5 / Asian line also the Football-Data
               Bet365 closing column)
    CLV_fair = price taken x q_close - 1
               q_close = de-margined (Shin) closing probability of the selection:
               1) Betfair-exchange closing price from Football-Data (1X2, total 2.5; margin ~1 %),
               2) otherwise the complete closing market of one OddsPapi bookmaker (Pinnacle preferred, then Bet365).
               Double chance from the 1X2 probabilities. Whole-line Asian handicaps (refund) are left out.

CLV_fair is the expected return of the bet if the closing market is the truth; for B and C prices (themselves the
closing quote) CLV_odds is 0 by construction and CLV_fair is just minus the closing margin share of that price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import fd
from . import quotes as Q
from .drift import SELS

GROUP_SELS = SELS


def _fd_close() -> pd.DataFrame:
    """Long table of Football-Data 2026/27 closing prices: Bet365 price and exchange probability."""
    d = fd.prices(C.LIVE)
    rows = []
    for market, cols, sels in (("1X2", ("x_H", "x_D", "x_A"), ("H", "D", "A")),
                               ("TOTAL_2.5", ("ou_OVER", "ou_UNDER"), ("OVER", "UNDER"))):
        qb = C.demargin(d[[f"{c}_bfe_c" for c in cols]].to_numpy(float), "shin")
        for j, s in enumerate(sels):
            rows.append(pd.DataFrame({"match_id": d["match_id"], "market": market, "selection": s,
                                      "b365_close": d[f"{cols[j]}_b365_c"].to_numpy(float), "q_bfe": qb[:, j]}))
    # Asian line (closing line) — Bet365 only
    line = d["ah_line_c"].to_numpy(float)
    ok = np.isfinite(line)
    from src.markets.markets import handicap_market
    mk = np.array([handicap_market(x) if f else None for x, f in zip(line, ok)], dtype=object)
    for s, col in (("H", "ah_H_b365_c"), ("A", "ah_A_b365_c")):
        rows.append(pd.DataFrame({"match_id": d["match_id"][ok], "market": mk[ok], "selection": s,
                                  "b365_close": d[col].to_numpy(float)[ok], "q_bfe": np.nan}))
    return pd.concat(rows, ignore_index=True)


def _odp_fair(clos: pd.DataFrame) -> pd.DataFrame:
    """Shin probabilities of every complete closing market of one OddsPapi bookmaker (Pinnacle first)."""
    clos = clos.copy()
    clos["group"] = clos["market"].map(C.market_group)
    out = []
    for grp, sels in GROUP_SELS.items():
        s = clos.loc[clos["group"] == grp]
        w = s.pivot_table(index=["match_id", "market", "bookmaker"], columns="selection", values="odds", aggfunc="last")
        if not set(sels) <= set(w.columns):
            continue
        w = w[list(sels)].dropna()
        q = C.demargin(w.to_numpy(float), "shin")
        long = pd.DataFrame(q, index=w.index, columns=pd.Index(list(sels), name="selection")).stack().rename("q").reset_index()
        long["rank"] = (long["bookmaker"] != "pinnacle").astype(int)
        out.append(long)
    f = pd.concat(out, ignore_index=True).sort_values("rank").drop_duplicates(["match_id", "market", "selection"])
    return f.rename(columns={"bookmaker": "q_book"})[["match_id", "market", "selection", "q", "q_book"]]


NEAR_CLOSE_H = 3.0


def near_closing(q: pd.DataFrame) -> pd.DataFrame:
    """For matches without an archive row: the last live observation within NEAR_CLOSE_H hours of kick-off
    (a strictly known price at that fetch time), in the same layout as Q.closing."""
    from .drift import live_snapshots
    snap = live_snapshots(q)
    snap = snap.loc[snap["lead_h"] <= NEAR_CLOSE_H]
    last_t = snap.groupby("match_id")["t"].transform("max")
    snap = snap.loc[snap["t"] == last_t]
    snap = snap.loc[~snap["match_id"].isin(set(Q.closing(q)["match_id"]))]
    return snap.assign(priced_at_utc=snap["t"])[["match_id", "bookmaker", "market", "selection", "odds",
                                                  "priced_at_utc", "lead_h"]]


def attach(picks: pd.DataFrame) -> pd.DataFrame:
    q = Q.load()
    arch = Q.closing(q)
    arch = arch.assign(close_kind="archive")
    near = near_closing(q).assign(close_kind="live_le_3h")
    clos = pd.concat([arch[["match_id", "bookmaker", "market", "selection", "odds", "priced_at_utc", "close_kind"]],
                      near[["match_id", "bookmaker", "market", "selection", "odds", "priced_at_utc", "close_kind"]]],
                     ignore_index=True)
    out = picks.copy()
    out["bk"] = out["bookmaker"].fillna("").str.lower().str.replace(" ", "")
    same = clos.assign(bk=clos["bookmaker"].str.lower())[["match_id", "market", "selection", "bk", "odds", "close_kind"]]
    out = out.merge(same.rename(columns={"odds": "close_same_odp"}), on=["match_id", "market", "selection", "bk"],
                    how="left")
    fdc = _fd_close()
    out = out.merge(fdc, on=["match_id", "market", "selection"], how="left")
    out["close_same"] = out["close_same_odp"]
    is_b365 = out["bk"].eq("bet365")
    out.loc[out["close_same"].isna() & is_b365, "close_same"] = out["b365_close"]
    out["close_same_from"] = np.where(out["close_same_odp"].notna(), "oddspapi_" + out["close_kind"].fillna(""),
                                      np.where(out["close_same"].notna(), "football_data", None))
    fair = _odp_fair(clos)
    out = out.merge(fair, on=["match_id", "market", "selection"], how="left")
    # double chance from the 1X2 closing probabilities (exchange first)
    one = pd.concat([fdc.loc[fdc["market"] == "1X2", ["match_id", "selection", "q_bfe"]].rename(columns={"q_bfe": "q1"}),
                     ], ignore_index=True)
    w = one.pivot_table(index="match_id", columns="selection", values="q1")
    w2 = fair.loc[fair["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="q")
    w = w.combine_first(w2) if not w.empty else w2
    dc = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
    mdc = out["market"] == "DOUBLE_CHANCE"
    for sel, (a, b) in dc.items():
        m = mdc & (out["selection"] == sel)
        if m.any() and not w.empty:
            out.loc[m, "q_dc"] = (out.loc[m, "match_id"].map(w[a]) + out.loc[m, "match_id"].map(w[b])).to_numpy()
    out["q_close"] = out["q_bfe"].combine_first(out["q"])
    if "q_dc" in out:
        out["q_close"] = out["q_close"].combine_first(out["q_dc"])
    out["q_from"] = np.where(out["q_bfe"].notna(), "betfair_exchange",
                             np.where(out["q"].notna(), "oddspapi_" + out["q_book"].fillna(""),
                                      np.where(out.get("q_dc", pd.Series(np.nan, index=out.index)).notna(),
                                               "from_1x2", None)))
    whole = out["market"].str.startswith("HANDICAP_") & out["market"].map(
        lambda m: m.startswith("HANDICAP_") and float(m.split("_")[1]) % 1 == 0)
    out.loc[whole, "q_close"] = np.nan
    out["clv_odds"] = out["odds"] / out["close_same"] - 1
    out["clv_fair"] = out["odds"] * out["q_close"] - 1
    return out


def summary(df: pd.DataFrame) -> dict:
    out = {}
    dec = df["status"].isin(["WON", "LOST"])
    for (role, cls), g in df.groupby(["role", "price_class"]):
        r = {"n": int(len(g))}
        for col in ("clv_odds", "clv_fair"):
            x = g[col].to_numpy(float)
            r[col] = C.grouped_mean_ci(x, g["match_id"].to_numpy(), 1000)
            r[col + "_share_pos"] = float((x[np.isfinite(x)] > 0).mean()) if np.isfinite(x).any() else None
        gd = g.loc[dec.loc[g.index]]
        x = gd["clv_fair"].to_numpy(float)
        ok = np.isfinite(x) & gd["profit"].notna().to_numpy()
        if ok.sum() >= 30:
            pr = gd["profit"].to_numpy(float)[ok]
            r["corr_clv_fair_profit"] = float(np.corrcoef(x[ok], pr)[0, 1])
            r["roi_clv_fair_pos"] = C.roi_summary(np.where(x[ok] > 0, pr, np.nan), n_boot=1000)
            r["roi_clv_fair_neg"] = C.roi_summary(np.where(x[ok] <= 0, pr, np.nan), n_boot=1000)
            r["roi_same_bets"] = C.roi_summary(pr, n_boot=1000)
            r["mean_clv_fair_same_bets"] = float(x[ok].mean())
        out[f"{role}|{cls}"] = r
    # A1 by provider and by market group
    a1 = df.loc[df["price_class"].isin(["A1", "A2"])]
    for (role, prov), g in a1.groupby(["role", a1["provider"].fillna("—")]):
        out[f"{role}|A|provider={prov}"] = {"n": int(len(g)),
                                            "clv_odds": C.grouped_mean_ci(g["clv_odds"], g["match_id"], 1000),
                                            "clv_fair": C.grouped_mean_ci(g["clv_fair"], g["match_id"], 1000)}
    for (role, grp), g in a1.groupby(["role", "group"]):
        if len(g) >= 20:
            out[f"{role}|A|group={grp}"] = {"n": int(len(g)),
                                            "clv_odds": C.grouped_mean_ci(g["clv_odds"], g["match_id"], 1000),
                                            "clv_fair": C.grouped_mean_ci(g["clv_fair"], g["match_id"], 1000)}
    out["coverage"] = {"close_same_from": df.groupby([df["price_class"], df["close_same_from"].fillna("none")])
                       .size().unstack(fill_value=0).to_dict("index"),
                       "q_from": df.groupby([df["price_class"], df["q_from"].fillna("none")]).size()
                       .unstack(fill_value=0).to_dict("index")}
    return out


def run() -> dict:
    picks = pd.read_parquet(C.CACHE / "audit_picks.parquet")
    picks = picks.loc[picks["price_class"].isin(["A1", "A2", "B", "C"])]
    df = attach(picks)
    df.to_parquet(C.CACHE / "clv_picks.parquet")
    out = summary(df)
    C.save_json("clv.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("clv")
    run()
