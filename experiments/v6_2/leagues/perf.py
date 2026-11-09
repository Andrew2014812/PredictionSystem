"""How evenly does the production model perform across leagues?

For the production setup B0 (seed 42, walk-forward) on the evaluation matches of every season:
log loss of the model and of the market (Bet365 opening, de-margined) and their difference per league,
calibration per league (slope b of the multinomial map z_k = b·log p_k + a_k: b < 1 = over-confident),
predicted vs observed draw share, and the production Main per league (hit rate, gap = claimed − observed,
ROI at the Bet365 opening price).

Heterogeneity across leagues (random-effects meta-analysis, DerSimonian–Laird):
    d_L = mean (log loss model − market) of league L,  v_L = var / n_L,
    Q = Σ w_L (d_L − d̄_w)²,  w_L = 1/v_L,  τ² = max(0, (Q − (L−1)) / (Σw − Σw²/Σw)),  I² = max(0, (Q−(L−1))/Q),
    shrunken league effect d_L^EB = μ̂ + (d_L − μ̂) · τ² / (τ² + v_L),  μ̂ = Σ w*_L d_L / Σ w*_L, w* = 1/(v_L+τ²).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import calib as CB
from . import common as C
from . import profiles as PR
from . import wf


def season_block(s: int, pred: pd.DataFrame | None = None, with_main: bool = True) -> pd.DataFrame:
    """Per-match evaluation rows of season s (B0 by default) with per-match losses and the Main outcome."""
    fr = wf.frame(s)
    pred = pred if pred is not None else wf.base_predictions(s, C.PROD_SEED)
    pred = fr[["match_id"]].merge(pred, on="match_id", how="left")
    ev = fr.loc[fr["in_eval"]].reset_index(drop=True)
    pe = pred.loc[fr["in_eval"].to_numpy()].reset_index(drop=True)
    y = ev["y"].to_numpy(int)
    p, q = wf.P(pe), wf.Q(ev)
    out = ev[["match_id", "league", "season", "y", "home_goals", "away_goals"]].copy()
    out[["pH", "pD", "pA"]] = p
    out[["qH", "qD", "qA"]] = q
    out["ll"], out["ll_mkt"] = C.logloss_rows(p, y), C.logloss_rows(q, y)
    qc = wf.Q(ev, closing=True)
    okc = np.all(np.isfinite(qc), 1)
    out["ll_close"] = np.where(okc, C.logloss_rows(np.nan_to_num(qc, nan=1 / 3), y), np.nan)
    if "lh" in pe:
        out["lh"], out["la"] = pe["lh"].to_numpy(), pe["la"].to_numpy()
        out["dev"] = C.poisson_dev_rows(ev["home_goals"], pe["lh"]) + C.poisson_dev_rows(ev["away_goals"], pe["la"])
    if with_main:
        o = wf.main_outcome(fr, pred, s)
        out = out.merge(o.add_prefix("m_").rename(columns={"m_match_id": "match_id"}), on="match_id", how="left")
    return out


def dl_meta(d: pd.Series, groups: pd.Series) -> dict:
    """DerSimonian–Laird random-effects summary of per-group means of d."""
    g = pd.DataFrame({"d": d.to_numpy(float), "g": groups.to_numpy()}).dropna().groupby("g")["d"]
    m, v, n = g.mean(), g.var(ddof=1) / g.size(), g.size()
    w = 1 / v
    dw = (w * m).sum() / w.sum()
    Q = float((w * (m - dw) ** 2).sum())
    k = len(m)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = float((ws * m).sum() / ws.sum())
    eb = mu + (m - mu) * tau2 / (tau2 + v)
    return {"k": k, "Q": Q, "p_Q": float(1 - stats.chi2.cdf(Q, k - 1)), "tau2": tau2, "tau": float(np.sqrt(tau2)),
            "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, "mu": mu,
            "mu_se": float(np.sqrt(1 / ws.sum())),
            "by_group": {str(x): {"n": int(n[x]), "mean": float(m[x]), "se": float(np.sqrt(v[x])), "eb": float(eb[x])}
                         for x in m.index}}


def league_rows(b: pd.DataFrame) -> dict:
    """Per-league metrics of one block (one season or pooled seasons)."""
    out = {}
    for lg, g in b.groupby("league"):
        y = g["y"].to_numpy(int)
        p = g[["pH", "pD", "pA"]].to_numpy()
        r = {"n": int(len(g)), "small": bool(len(g) < C.SMALL_N), "ll": float(g["ll"].mean()),
             "ll_mkt": float(g["ll_mkt"].mean()), "d": C.paired_diff_ci(g["ll_mkt"], g["ll"]),
             "ece": C.ece_multi(p, y), "ece_mkt": C.ece_multi(g[["qH", "qD", "qA"]].to_numpy(), y),
             "draw_pred": float(g["pD"].mean()), "draw_obs": float((y == 1).mean()),
             "home_pred": float(g["pH"].mean()), "home_obs": float((y == 0).mean())}
        th = CB.fit(p, y)
        cov = CB.covariance(th, p, y)
        r["slope"] = float(th[0])
        r["slope_se"] = float(np.sqrt(max(cov[0, 0], 0))) if np.isfinite(cov[0, 0]) else None
        if "dev" in g:
            r["dev"] = float(g["dev"].mean())
        if "m_won" in g:
            dec = g.loc[g["m_decided"].fillna(False).astype(bool)]
            bets = dec.loc[dec["m_priced"].astype(bool)]
            r["main"] = {"settled": int(len(dec)), "hit": float(dec["m_won"].mean()) if len(dec) else None,
                         "p": float(dec["m_p"].mean()) if len(dec) else None,
                         "gap": float(dec["m_p"].mean() - dec["m_won"].mean()) if len(dec) else None,
                         "gap_ci": C.paired_diff_ci(dec["m_won"], dec["m_p"])["ci95"] if len(dec) > 1 else [None, None],
                         "bets": int(len(bets)), "roi": C.mean_ci(bets["m_profit"])}
        out[lg] = r
    return out


def run() -> dict:
    blocks = {s: season_block(s) for s in C.EVAL}
    for s, b in blocks.items():
        b.to_parquet(C.CACHE / f"b0_block_{s}.parquet")
    sel = pd.concat([blocks[s] for s in C.SELECT], ignore_index=True)
    res = {"seasons": {}, "select": {}}
    for s, b in blocks.items():
        res["seasons"][str(s)] = {"n": int(len(b)), "ll": float(b["ll"].mean()), "ll_mkt": float(b["ll_mkt"].mean()),
                                  "d": C.paired_diff_ci(b["ll_mkt"], b["ll"]), "leagues": league_rows(b),
                                  "meta": dl_meta(b["ll"] - b["ll_mkt"], b["league"]),
                                  "meta_main_roi": dl_meta(b["m_profit"], b["league"]),
                                  "coverage": {"played": int(len(wf.frame(s))),
                                               "with_b365_open": int(wf.frame(s)["in_eval"].sum())}}
    res["select"] = {"n": int(len(sel)), "leagues": league_rows(sel), "meta": dl_meta(sel["ll"] - sel["ll_mkt"], sel["league"]),
                     "meta_main_roi": dl_meta(sel["m_profit"], sel["league"]),
                     "meta_main_gap": dl_meta(sel["m_p"] - sel["m_won"], sel["league"]),
                     "d": C.paired_diff_ci(sel["ll_mkt"], sel["ll"])}
    # persistence of league differences between seasons (are they real or noise?)
    dd = {s: (blocks[s]["ll"] - blocks[s]["ll_mkt"]).groupby(blocks[s]["league"]).mean() for s in C.EVAL}
    pers = {}
    for a, b in ((2023, 2024), (2024, 2025), (2023, 2025), (2025, 2026)):
        r = stats.spearmanr(dd[a], dd[b].reindex(dd[a].index))
        pers[f"{a}-{b}"] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    res["persistence"] = pers
    # what explains league differences: league profile (seasons < 2023) and training size
    prof = PR.profile(wf.features().loc[lambda d: d["season"] < 2023])
    dsel = (sel["ll"] - sel["ll_mkt"]).groupby(sel["league"]).mean()
    prof["train_n"] = wf.features().loc[lambda d: d["season"] < 2023].groupby("league").size()
    prof["tier"] = [C.tier(x) for x in prof.index]
    cors, pv = {}, {}
    for v in list(PR.VARS) + ["train_n", "tier"]:
        r = stats.spearmanr(prof[v].reindex(dsel.index), dsel)
        cors[v] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        pv[v] = float(r.pvalue)
    adj = C.holm(pv)
    for v in cors:
        cors[v]["p_holm"] = adj[v]
    res["explain"] = cors
    C.save_json("perf.json", res)
    return res
