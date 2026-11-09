"""Model vs bookmaker: margin removal, log loss / Brier / calibration by market, season, league and odds range.

Reference probabilities are de-margined prices of one snapshot of one book. The decision-time references are the
OPENING snapshots (Bet365 = the production book, Pinnacle, the market average); the CLOSING snapshots (Bet365,
Pinnacle, Betfair exchange) are the market's final estimate and serve as the hardest benchmark (not available
at decision time). Every comparison is on the same matches; intervals are paired bootstraps by match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import data as D

SEASONS = (2022, 2023, 2024, 2025, 2026)
REFS = [("b365", False), ("ps", False), ("avg", False), ("bfe", False), ("b365", True), ("ps", True), ("bfe", True)]
REF_RU = {"b365": "Bet365", "ps": "Pinnacle", "avg": "среднее", "bfe": "Betfair Exch."}


def ref_name(book: str, closing: bool) -> str:
    return f"{book}_{'close' if closing else 'open'}"


# ------------------------------------------------------------------------------------------------ margin removal
def demargin_compare() -> dict:
    out = {}
    for market in D.MARKETS:
        for s in SEASONS:
            t = D.MARKETS[market][0](s)
            y = t["y"].to_numpy(int)
            for book, closing in REFS:
                o = D.odds_matrix(t, market, book, closing)
                ok = np.all(np.isfinite(o) & (o > 1), 1)
                if ok.sum() < 200:
                    continue
                res = {"n": int(ok.sum()), "margin": float(((1 / o[ok]).sum(1) - 1).mean())}
                base = None
                for m in C.METHODS:
                    q = C.demargin(o[ok], m)
                    ll = C.logloss_rows(q, y[ok])
                    res[m] = float(ll.mean())
                    if m == "prop":
                        base = ll
                    else:
                        res[f"{m}_minus_prop"] = C.paired_diff_ci(base, ll, 1000)
                out[f"{market}|{s}|{ref_name(book, closing)}"] = res
    # choice on the selection seasons: lowest pooled log loss of 2023/24 + 2024/25, per market and book snapshot
    choice = {}
    for market in D.MARKETS:
        for book, closing in REFS:
            keys = [f"{market}|{s}|{ref_name(book, closing)}" for s in C.SELECT]
            if not all(k in out for k in keys):
                continue
            tot = {m: sum(out[k][m] * out[k]["n"] for k in keys) / sum(out[k]["n"] for k in keys) for m in C.METHODS}
            choice[f"{market}|{ref_name(book, closing)}"] = min(tot, key=tot.get)
    return {"by_key": out, "choice": choice}


def market_probs(t: pd.DataFrame, market: str, book: str, closing: bool, method: str) -> np.ndarray:
    return C.demargin(D.odds_matrix(t, market, book, closing), method)


# ------------------------------------------------------------------------------------------------ model vs market
def _calib(y_bin: np.ndarray, p: np.ndarray) -> dict:
    """Calibration slope / intercept (logistic regression of the event on logit p) and ECE."""
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    try:
        fit = sm.GLM(y_bin, sm.add_constant(x), family=sm.families.Binomial()).fit()
        a, b = fit.params
    except Exception:
        a = b = np.nan
    return {"slope": float(b), "intercept": float(a), "ece": C.ece(y_bin, p), "mean_p": float(p.mean()),
            "rate": float(y_bin.mean())}


def compare(t: pd.DataFrame, market: str, book: str, closing: bool, method: str) -> dict | None:
    q = market_probs(t, market, book, closing, method)
    ok = np.all(np.isfinite(q), 1)
    if ok.sum() < 100:
        return None
    p = D.model_matrix(t, market)[ok]
    q = q[ok]
    y = t["y"].to_numpy(int)[ok]
    llm, llq = C.logloss_rows(p, y), C.logloss_rows(q, y)
    brm, brq = C.brier_rows(p, y), C.brier_rows(q, y)
    res = {"n": int(ok.sum()), "ll_model": float(llm.mean()), "ll_market": float(llq.mean()),
           "brier_model": float(brm.mean()), "brier_market": float(brq.mean()),
           "d_ll": C.paired_diff_ci(llq, llm, 1000), "d_brier": C.paired_diff_ci(brq, brm, 1000),
           "share_model_better_ll": float((llm < llq).mean())}
    k0 = 0                                         # home win / over / home cover
    yb = (y == k0).astype(float)
    res["calib_model"], res["calib_market"] = _calib(yb, p[:, k0]), _calib(yb, q[:, k0])
    return res


def model_vs_market(choice: dict) -> dict:
    out = {}
    for market in D.MARKETS:
        for s in SEASONS[1:]:
            t = D.MARKETS[market][0](s)
            for book, closing in REFS:
                method = choice.get(f"{market}|{ref_name(book, closing)}", "shin")
                r = compare(t, market, book, closing, method)
                if r is not None:
                    r["method"] = method
                    out[f"{market}|{s}|{ref_name(book, closing)}"] = r
    return out


def by_league(choice: dict, book: str = "b365", closing: bool = False) -> dict:
    """Delta log loss (model - market) per league: selection seasons pooled (Holm across leagues),
    2025/26 and 2026/27 shown for information."""
    out = {}
    for market in ("1X2", "OU25"):
        method = choice.get(f"{market}|{ref_name(book, closing)}", "shin")
        frames = {}
        for label, seasons in (("select", C.SELECT), ("2025", (2025,)), ("2026", (2026,))):
            t = D.load(market, seasons)
            q = market_probs(t, market, book, closing, method)
            ok = np.all(np.isfinite(q), 1)
            t, q = t.loc[ok].reset_index(drop=True), q[ok]
            y = t["y"].to_numpy(int)
            t["d"] = C.logloss_rows(D.model_matrix(t, market), y) - C.logloss_rows(q, y)
            frames[label] = t
        res, pv = {}, {}
        for lg, g in frames["select"].groupby("league"):
            r = C.mean_ci(g["d"].to_numpy(), 1000)
            boot_p = None
            if r["ci95"][0] is not None:
                # two-sided bootstrap p-value of mean d = 0
                x = g["d"].to_numpy()
                rng = np.random.default_rng(C.SEED)
                ms = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
                boot_p = float(min(1.0, max(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1e-3)))
            res[lg] = {"select": r, "p": boot_p}
            pv[lg] = boot_p
            for lab in ("2025", "2026"):
                gg = frames[lab].loc[frames[lab]["league"] == lg, "d"].to_numpy()
                res[lg][lab] = C.mean_ci(gg, 1000) if len(gg) >= 20 else {"n": int(len(gg)), "mean": None}
        adj = C.holm(pv)
        for lg in res:
            res[lg]["p_holm"] = adj.get(lg)
        out[market] = {"reference": ref_name(book, closing), "method": method, "leagues": res,
                       "n_leagues_tested": len(pv)}
    return out


def by_odds_range(choice: dict, book: str = "b365") -> dict:
    """Selection-level binary log loss (event k vs not) of model and market, by the opening price of the selection."""
    edges = [1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    labels = ["1.0–1.5", "1.5–2.0", "2.0–3.0", "3.0–5.0", "> 5.0"]
    out = {}
    for market in D.MARKETS:
        method = choice.get(f"{market}|{ref_name(book, False)}", "shin")
        for label, seasons in (("select", C.SELECT), ("2025", (2025,)), ("2026", (2026,))):
            t = D.load(market, seasons)
            o = D.odds_matrix(t, market, book, False)
            q = C.demargin(o, method)
            ok = np.all(np.isfinite(q), 1)
            p = D.model_matrix(t, market)[ok]
            q, o, y, mid = q[ok], o[ok], t["y"].to_numpy(int)[ok], t["match_id"].to_numpy()[ok]
            k = p.shape[1]
            rows = []
            for j in range(k if market == "1X2" else 1):
                e = (y == j).astype(float)
                pj, qj = np.clip(p[:, j], 1e-9, 1 - 1e-9), np.clip(q[:, j], 1e-9, 1 - 1e-9)
                d = -(e * np.log(pj) + (1 - e) * np.log(1 - pj)) + (e * np.log(qj) + (1 - e) * np.log(1 - qj))
                rows.append(pd.DataFrame({"odds": o[:, j], "d": d, "mid": mid, "gap": pj - qj, "e": e, "p": pj,
                                          "q": qj}))
            r = pd.concat(rows, ignore_index=True)
            r["bucket"] = pd.cut(r["odds"], edges, labels=labels, right=False)
            for b, g in r.groupby("bucket", observed=True):
                if len(g) < 50:
                    continue
                out[f"{market}|{label}|{b}"] = {"d_ll": C.grouped_mean_ci(g["d"].to_numpy(), g["mid"].to_numpy(), 1000),
                                                "mean_p": float(g["p"].mean()), "mean_q": float(g["q"].mean()),
                                                "rate": float(g["e"].mean()), "n": int(len(g))}
    return out


def reliability(choice: dict) -> dict:
    """Reliability curves (10 bins) of model and Bet365-opening probability, selection seasons pooled, 1X2 home and over."""
    out = {}
    for market in ("1X2", "OU25"):
        method = choice.get(f"{market}|b365_open", "shin")
        t = D.load(market, C.SELECT)
        q = market_probs(t, market, "b365", False, method)
        ok = np.all(np.isfinite(q), 1)
        p = D.model_matrix(t, market)[ok]
        y = (t["y"].to_numpy(int)[ok] == 0).astype(float)
        for name, v in (("model", p[:, 0]), ("market", q[ok][:, 0])):
            edges = np.quantile(v, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(v, edges[1:-1]), 0, 9)
            out[f"{market}|{name}"] = [{"pred": float(v[idx == b].mean()), "obs": float(y[idx == b].mean()),
                                        "n": int((idx == b).sum())} for b in range(10) if (idx == b).any()]
    return out


def run() -> dict:
    dm = demargin_compare()
    C.save_json("demargin.json", dm)
    out = {"model_vs_market": model_vs_market(dm["choice"]), "by_league": by_league(dm["choice"]),
           "by_league_ps_close": by_league(dm["choice"], "b365", True),
           "by_odds_range": by_odds_range(dm["choice"]), "reliability": reliability(dm["choice"]),
           "choice": dm["choice"]}
    C.save_json("benchmark.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("bench")
    run()
