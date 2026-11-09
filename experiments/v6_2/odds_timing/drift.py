"""How market prices move between an early quote and the close.

1. Football-Data, 2021/22–2026/27: the "opening" snapshot (Friday / Tuesday) vs the closing snapshot of the same
   book, for 1X2, total 2.5 and the Asian line (only when the line did not move). De-margined probabilities
   (Shin), their drift dq = q_close - q_open, margins, and which snapshot predicts the result better (log loss,
   paired by match).
2. The model and the drift: does the model's disagreement with the opening price, p - q_open, predict where the
   price moves (dq)? Walk-forward predictions 2022/23–2025/26 (v6.1 cache) and the stored 2026/27 predictions.
   A positive slope means the model holds information the market only prices in later (the source of positive CLV).
3. OddsPapi 2026/27 by horizon: prices observed by the live collection (22.10–09.11.2026) at different times before
   kick-off vs the last observation within 3 h of kick-off, for every market group (strictly known prices, quotes.py).
4. The age of the closing price: the share of OddsPapi closing quotes already published h hours before kick-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import fd
from . import quotes as Q

MKT = {"1X2": (("x_H", "x_D", "x_A"), ("H", "D", "A")),
       "TOTAL_2.5": (("ou_OVER", "ou_UNDER"), ("OVER", "UNDER")),
       "AH": (("ah_H", "ah_A"), ("H", "A"))}
BOOK_RU = {"b365": "Bet365", "ps": "Pinnacle", "avg": "среднее рынка", "bfe": "Betfair Exchange", "max": "максимум"}


def _odds(d: pd.DataFrame, market: str, book: str, closing: bool) -> np.ndarray:
    suf = "_c" if closing else ""
    return d[[f"{c}_{book}{suf}" for c in MKT[market][0]]].to_numpy(float)


def _outcome(d: pd.DataFrame, market: str, line=None) -> np.ndarray:
    hg, ag = d["hg"].to_numpy(float), d["ag"].to_numpy(float)
    if market == "1X2":
        return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    if market == "TOTAL_2.5":
        return np.where(hg + ag > 2.5, 0, 1)
    m = hg + np.asarray(line, float) - ag
    return np.where(m > 0, 0, np.where(m < 0, 1, -1))       # -1 = refund (whole line)


def _ll(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    ok = (y >= 0) & np.all(np.isfinite(q), 1)
    out[ok] = -np.log(np.clip(q[ok, y[ok]], 1e-9, 1))
    return out


def fd_market(d: pd.DataFrame, market: str, book: str, method: str = "shin") -> pd.DataFrame | None:
    """Per match: q_open, q_close (k columns each), margins, outcome; only matches with both snapshots complete."""
    try:
        o_open, o_close = _odds(d, market, book, False), _odds(d, market, book, True)
    except KeyError:
        return None
    ok = np.all(np.isfinite(o_open) & (o_open > 1), 1) & np.all(np.isfinite(o_close) & (o_close > 1), 1)
    line = None
    if market == "AH":
        line, line_c = d["ah_line"].to_numpy(float), d["ah_line_c"].to_numpy(float)
        half_or_whole = np.isfinite(line) & ((np.abs(line * 2 - np.round(line * 2)) < 1e-9))
        ok &= half_or_whole & np.isclose(line, line_c)
    if ok.sum() < 30:
        return None
    x = d.loc[ok].reset_index(drop=True)
    qo, qc = C.demargin(o_open[ok], method), C.demargin(o_close[ok], method)
    y = _outcome(x, market, line[ok] if line is not None else None)
    out = pd.DataFrame({"match_id": x["match_id"], "league": x["league"], "season": x["season"], "y": y,
                        "margin_open": (1 / o_open[ok]).sum(1) - 1, "margin_close": (1 / o_close[ok]).sum(1) - 1,
                        "ll_open": _ll(qo, y), "ll_close": _ll(qc, y)})
    for j, s in enumerate(MKT[market][1]):
        out[f"qo_{s}"], out[f"qc_{s}"] = qo[:, j], qc[:, j]
        out[f"oo_{s}"], out[f"oc_{s}"] = o_open[ok][:, j], o_close[ok][:, j]
    if market == "AH":
        out["line"] = line[ok]
    return out


def _drift_stats(t: pd.DataFrame, market: str) -> dict:
    sels = MKT[market][1]
    dq = np.concatenate([(t[f"qc_{s}"] - t[f"qo_{s}"]).to_numpy() for s in (sels if market == "1X2" else sels[:1])])
    absd = np.abs(dq)
    res = {"matches": int(len(t)), "mean_abs_dq_pp": float(absd.mean() * 100), "sd_dq_pp": float(dq.std() * 100),
           "p90_abs_dq_pp": float(np.quantile(absd, 0.9) * 100), "share_abs_gt_2pp": float((absd > 0.02).mean()),
           "share_abs_gt_5pp": float((absd > 0.05).mean()),
           "margin_open": float(t["margin_open"].mean()), "margin_close": float(t["margin_close"].mean()),
           "ll_open": float(t["ll_open"].mean()), "ll_close": float(t["ll_close"].mean())}
    res["ll_close_minus_open"] = C.paired_diff_ci(t["ll_open"].to_numpy(), t["ll_close"].to_numpy(), 1000)
    if market == "1X2":
        fo = t[[f"qo_{s}" for s in sels]].to_numpy().argmax(1)
        fc = t[[f"qc_{s}" for s in sels]].to_numpy().argmax(1)
        res["favourite_changes"] = float((fo != fc).mean())
    # odds change of a selection taken at the opening price: price_open / price_close - 1 (CLV of a random bet)
    s0 = sels[0]
    res["mean_odds_ratio_minus1"] = float((t[f"oo_{s0}"] / t[f"oc_{s0}"] - 1).mean())
    return res


def fd_drift() -> dict:
    out = {"by_season": {}, "by_league": {}, "by_bucket": {}}
    pooled = {}
    for s in fd.SEASONS:
        d = fd.prices(s)
        for market in MKT:
            for book in ("b365", "ps", "avg", "bfe"):
                t = fd_market(d, market, book)
                if t is None:
                    continue
                out["by_season"][f"{s}|{market}|{book}"] = _drift_stats(t, market)
                if book == "b365":
                    pooled.setdefault(market, []).append(t)
    for market, frames in pooled.items():
        t = pd.concat(frames, ignore_index=True)
        t.to_parquet(C.CACHE / f"fd_drift_b365_{market}.parquet")
        out["all_seasons_b365|" + market] = _drift_stats(t, market)
        for lg, g in t.groupby("league"):
            out["by_league"][f"{market}|{lg}"] = _drift_stats(g, market)
        if market == "1X2":
            q = np.concatenate([t[f"qo_{s}"] for s in "HDA"])
            dq = np.concatenate([t[f"qc_{s}"] - t[f"qo_{s}"] for s in "HDA"])
            mid = np.concatenate([t["match_id"]] * 3)
            edges = [0, .2, .35, .5, .65, 1.0]
            b = np.digitize(q, edges[1:-1])
            for k in range(len(edges) - 1):
                m = b == k
                r = C.grouped_mean_ci(dq[m] * 100, mid[m], 1000)
                out["by_bucket"][f"{edges[k]:.2f}-{edges[k + 1]:.2f}"] = r
    return out


# ------------------------------------------------------------------------------------------------ model vs drift
def model_probs() -> pd.DataFrame:
    """match_id, season, p_H, p_D, p_A, p_OVER: walk-forward 2022–2025 (v6.1 cache) + stored 2026/27."""
    from experiments.v6_1.main import common as V61
    frames = []
    for s in (2022, 2023, 2024, 2025):
        c = pd.read_parquet(V61.CACHE / f"cand_{s}.parquet", columns=["match_id", "market", "selection", "p"])
        c = c.loc[c["market"].isin(["1X2", "TOTAL_2.5"])]
        w = c.pivot_table(index="match_id", columns="selection", values="p", aggfunc="first")
        frames.append(pd.DataFrame({"match_id": w.index, "season": s, "p_H": w["H"].to_numpy(),
                                    "p_D": w["D"].to_numpy(), "p_A": w["A"].to_numpy(),
                                    "p_OVER": w["OVER"].to_numpy()}))
    frames.append(live_model_probs())
    return pd.concat(frames, ignore_index=True)


def live_model_probs() -> pd.DataFrame:
    """Probabilities of the stored 2026/27 predictions (live when recorded, otherwise backfill)."""
    from src.analytics.performance import unified_history
    picks = pd.read_parquet(C.PRED_DIR / "picks.parquet", columns=None)
    hist = unified_history(picks)
    ids = set(hist.loc[hist["season"] == C.LIVE, "match_id"])
    src = hist.drop_duplicates("match_id").set_index("match_id")["source"]
    mk = pd.read_parquet(C.PRED_DIR / "markets.parquet", columns=["match_id", "market", "selection", "probability"])
    mk = mk.loc[mk["match_id"].isin(ids) & mk["market"].isin(["1X2", "TOTAL_2.5"])]
    w = mk.pivot_table(index="match_id", columns="selection", values="probability", aggfunc="last")
    return pd.DataFrame({"match_id": w.index, "season": C.LIVE, "p_H": w["H"].to_numpy(), "p_D": w["D"].to_numpy(),
                         "p_A": w["A"].to_numpy(), "p_OVER": w["OVER"].to_numpy(),
                         "source": w.index.map(src)})


def _slope_ci(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 1000) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], pd.factorize(pd.Series(groups[ok]))[0]
    k = g.max() + 1
    sx, sy, sxx, sxy, n = (np.bincount(g, w, k) for w in (x, y, x * x, x * y, None))

    def slope(i):
        SX, SY, SXX, SXY, N = sx[i].sum(-1), sy[i].sum(-1), sxx[i].sum(-1), sxy[i].sum(-1), n[i].sum(-1)
        return (SXY - SX * SY / N) / (SXX - SX * SX / N)
    b = slope(np.arange(k))
    rng = np.random.default_rng(C.SEED)
    bs = np.concatenate([slope(rng.integers(0, k, size=(200, k))) for _ in range(n_boot // 200)])
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(x)), "matches": int(k), "slope": float(b), "ci95": [float(np.quantile(bs, .025)),
                                                                           float(np.quantile(bs, .975))],
            "corr": corr, "p_slope_le_0": float((bs <= 0).mean())}


def model_vs_drift() -> dict:
    mp = model_probs()
    out = {}
    for s in (2022, 2023, 2024, 2025, 2026):
        d = fd.prices(s)
        m = mp.loc[mp["season"] == s].drop_duplicates("match_id").set_index("match_id")
        for market, sels in (("1X2", "HDA"), ("TOTAL_2.5", ("OVER",))):
            for book in ("b365", "ps", "bfe"):
                t = fd_market(d, market, book)
                if t is None:
                    continue
                t = t.loc[t["match_id"].isin(m.index)]
                if len(t) < 100:
                    continue
                xs, ys, gs = [], [], []
                for sel in sels:
                    p = t["match_id"].map(m[f"p_{sel}"]).to_numpy(float)
                    xs.append(p - t[f"qo_{sel}"].to_numpy())
                    ys.append(t[f"qc_{sel}"].to_numpy() - t[f"qo_{sel}"].to_numpy())
                    gs.append(t["match_id"].to_numpy())
                x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
                r = _slope_ci(x, y, g)
                # direction: when the model is above the opening price by > 3 pp, does the price move towards it?
                big = np.abs(x) > 0.03
                r["share_moves_towards_model_if_gap_gt_3pp"] = float((np.sign(y[big]) == np.sign(x[big])).mean()) \
                    if big.any() else None
                r["n_gap_gt_3pp"] = int(big.sum())
                r["mean_abs_gap_pp"] = float(np.nanmean(np.abs(x)) * 100)
                out[f"{s}|{market}|{book}"] = r
    return out


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 500) -> dict:
    """OLS with an intercept; 95 % intervals from a cluster (match) bootstrap."""
    Xc = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    g = pd.factorize(pd.Series(groups))[0]
    k = g.max() + 1
    order = np.argsort(g)
    starts = np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]
    members = np.split(order, starts[1:])
    rng = np.random.default_rng(C.SEED)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, k, k)])
        bs.append(np.linalg.lstsq(Xc[idx], y[idx], rcond=None)[0])
    bs = np.array(bs)
    return {"beta": beta.tolist(), "ci95": np.quantile(bs, [0.025, 0.975], axis=0).T.tolist(),
            "p_le_0": (bs <= 0).mean(0).tolist(), "n": int(len(y)), "matches": int(k)}


def model_vs_drift_controlled() -> dict:
    """dq of the Pinnacle (sharp) price regressed on the model gap AND on the gaps of the other books at the same
    opening snapshot: d q_ps = a + b1 (p - q_ps_open) + b2 (q_b365_open - q_ps_open) + b3 (q_avg_open - q_ps_open).
    b1 > 0 = the model predicts the move beyond what the other books already show (not just noise reversion)."""
    mp = model_probs()
    out = {}
    for s in (2022, 2023, 2024, 2025):
        d = fd.prices(s)
        m = mp.loc[mp["season"] == s].drop_duplicates("match_id").set_index("match_id")
        for market, sels in (("1X2", "HDA"), ("TOTAL_2.5", ("OVER",))):
            ts = {b: fd_market(d, market, b) for b in ("ps", "b365", "avg")}
            if any(t is None for t in ts.values()):
                continue
            base = ts["ps"].set_index("match_id")
            ids = base.index.intersection(ts["b365"]["match_id"]).intersection(ts["avg"]["match_id"]).intersection(m.index)
            if len(ids) < 300:
                continue
            b365, avg = ts["b365"].set_index("match_id").loc[ids], ts["avg"].set_index("match_id").loc[ids]
            ps = base.loc[ids]
            X, y, g = [], [], []
            for sel in sels:
                qo = ps[f"qo_{sel}"].to_numpy()
                X.append(np.column_stack([m.loc[ids, f"p_{sel}"].to_numpy(float) - qo, b365[f"qo_{sel}"].to_numpy() - qo,
                                          avg[f"qo_{sel}"].to_numpy() - qo]))
                y.append(ps[f"qc_{sel}"].to_numpy() - qo)
                g.append(np.asarray(ids))
            X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(g)
            ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
            out[f"{s}|{market}"] = _ols_cluster(X[ok], y[ok], g[ok])
            out[f"{s}|{market}"]["names"] = ["const", "model-ps", "b365-ps", "avg-ps"]
    return out


# ------------------------------------------------------------------------------------------------ OddsPapi horizons
SELS = {"1X2": ("H", "D", "A"), "BTTS": ("YES", "NO"), "TOTAL": ("OVER", "UNDER"), "CORNERS": ("OVER", "UNDER"),
        "HANDICAP": ("H", "A"), "EHANDICAP": ("H", "D", "A")}
BUCKETS = [3, 6, 12, 24, 48, 72, 1e9]
BUCKET_RU = ["3–6 ч", "6–12 ч", "12–24 ч", "24–48 ч", "48–72 ч", "> 72 ч"]
REF_MAX_LEAD = 3.0


def live_snapshots(q: pd.DataFrame) -> pd.DataFrame:
    """Prices known to be valid at every live fetch time of each match (long: match, bookmaker, market,
    selection, t, odds)."""
    live = q.loc[~q["archive"] & q["market"].map(C.market_group).isin(SELS)]
    times = live[["match_id", "fetched_at_utc", "kickoff_utc"]].drop_duplicates(["match_id", "fetched_at_utc"])
    keys = live[Q.KEY].drop_duplicates()
    grid = keys.merge(times, on="match_id").rename(columns={"fetched_at_utc": "t"})
    m = grid.merge(live[Q.KEY + ["odds", "valid_from", "valid_to", "priced_at_utc"]], on=Q.KEY)
    m = m.loc[(m["valid_from"] <= m["t"]) & (m["t"] <= m["valid_to"])]
    m = m.sort_values("priced_at_utc").drop_duplicates(Q.KEY + ["t"], keep="last")
    m["lead_h"] = (m["kickoff_utc"] - m["t"]).dt.total_seconds() / 3600
    return m[Q.KEY + ["t", "lead_h", "odds"]]


def live_horizon_drift(q: pd.DataFrame) -> dict:
    snap = live_snapshots(q)
    snap["group"] = snap["market"].map(C.market_group)
    rows = []
    for grp, sels in SELS.items():
        s = snap.loc[snap["group"] == grp]
        w = s.pivot_table(index=["match_id", "bookmaker", "market", "t", "lead_h"], columns="selection",
                          values="odds", aggfunc="first")
        if not set(sels) <= set(w.columns):
            continue
        w = w[list(sels)].dropna()
        qv = C.demargin(w.to_numpy(float), "shin")
        df = w.reset_index()[["match_id", "bookmaker", "market", "t", "lead_h"]]
        df["group"] = grp
        df["q0"] = qv[:, 0]
        df["o0"] = w.to_numpy(float)[:, 0]
        if grp == "1X2":
            df["q1"], df["q2"] = qv[:, 1], qv[:, 2]
        rows.append(df)
    allq = pd.concat(rows, ignore_index=True)
    ref = (allq.loc[allq["lead_h"] <= REF_MAX_LEAD].sort_values("t")
           .drop_duplicates(["match_id", "bookmaker", "market"], keep="last"))
    ref = ref.rename(columns={"t": "t_ref", "lead_h": "lead_ref", "q0": "q0_ref", "o0": "o0_ref",
                              "q1": "q1_ref", "q2": "q2_ref"}).drop(columns="group")
    pair = allq.merge(ref, on=["match_id", "bookmaker", "market"])
    pair = pair.loc[pair["lead_h"] > pair["lead_ref"] + 1.0].copy()
    pair["dq"] = pair["q0_ref"] - pair["q0"]
    pair["bucket"] = pd.cut(pair["lead_h"], BUCKETS, labels=BUCKET_RU, right=False)
    pair.to_parquet(C.CACHE / "live_drift_pairs.parquet")
    out = {"reference_lead_h": {"median": float(ref["lead_ref"].median()), "q75": float(ref["lead_ref"].quantile(.75))},
           "matches": int(pair["match_id"].nunique()), "by_group_bucket": {}}
    for (grp, b), g in pair.groupby(["group", "bucket"], observed=True):
        if len(g) < 20:
            continue
        absd = np.abs(g["dq"].to_numpy())
        out["by_group_bucket"][f"{grp}|{b}"] = {
            "n": int(len(g)), "matches": int(g["match_id"].nunique()),
            "mean_abs_dq_pp": C.grouped_mean_ci(absd * 100, g["match_id"].to_numpy(), 1000),
            "share_abs_gt_2pp": float((absd > 0.02).mean()), "share_unchanged": float((g["o0"] == g["o0_ref"]).mean()),
            "mean_odds_ratio_minus1": float((g["o0"] / g["o0_ref"] - 1).mean())}
    # does the price observed earlier predict the result worse than the near-closing one? (1X2, same matches)
    res = _results()
    x = pair.loc[pair["group"] == "1X2"].merge(res, on="match_id")
    if not x.empty:
        y = x["y"].to_numpy(int)
        q_early = x[["q0", "q1", "q2"]].to_numpy()
        q_ref = x[["q0_ref", "q1_ref", "q2_ref"]].to_numpy()
        x["ll_early"], x["ll_ref"] = _ll(q_early, y), _ll(q_ref, y)
        for b, g in x.groupby("bucket", observed=True):
            g1 = g.sort_values("t").drop_duplicates(["match_id", "bookmaker"], keep="first")
            if len(g1) >= 30:
                out.setdefault("ll_1x2_by_bucket", {})[str(b)] = C.paired_diff_ci(g1["ll_early"], g1["ll_ref"], 1000)
    return out


def _results() -> pd.DataFrame:
    f = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                        columns=["match_id", "season", "played", "home_goals", "away_goals"])
    f = f.loc[(f["season"] == C.LIVE) & f["played"].astype(bool) & f["home_goals"].notna()]
    hg, ag = f["home_goals"].to_numpy(), f["away_goals"].to_numpy()
    return pd.DataFrame({"match_id": f["match_id"], "y": np.where(hg > ag, 0, np.where(hg == ag, 1, 2))})


def closing_age(q: pd.DataFrame) -> dict:
    a = Q.closing(q)
    a = a.loc[a["market"].map(C.market_group).isin(["1X2", "TOTAL", "BTTS", "HANDICAP", "CORNERS", "DOUBLE_CHANCE"])]
    a["group"] = a["market"].map(C.market_group)
    hs = [0.25, 0.5, 1, 2, 3, 6, 12, 24, 48, 72]
    out = {}
    for grp, g in list(a.groupby("group")) + [("ALL", a)]:
        out[grp] = {"n": int(len(g)), "matches": int(g["match_id"].nunique()),
                    "share_published_at_least_h_before": {str(h): float((g["lead_h"] >= h).mean()) for h in hs},
                    "median_lead_h": float(g["lead_h"].median())}
    return out


def run() -> dict:
    q = Q.load()
    out = {"fd": fd_drift(), "model_vs_drift": model_vs_drift(),
           "model_vs_drift_controlled": model_vs_drift_controlled(), "live_horizons": live_horizon_drift(q),
           "closing_age": closing_age(q)}
    C.save_json("drift.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("drift")
    run()
