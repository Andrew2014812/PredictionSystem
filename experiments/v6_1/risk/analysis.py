"""Why does a higher model EV not give a better realised ROI?  Diagnostics on the candidate tables.

1. calibration of rare outcomes: model p in 0.12-0.40 by selection type (all Risk candidates, and
   only the priced ones next to the de-vigged market probability as P(win), ``p_mkt_u``);
2. favourite-longshot bias of the prices: blind ROI of every priced selection by odds band, implied
   probability 1/odds and de-vigged probability against the observed frequency;
3. realised ROI by model EV bin (raw model, Platt-calibrated, model + market blend) and the slope
   of the realised profit on EV;
4. who is right when model and market disagree: logistic regression of the outcome on
   logit(p_model) and logit(p_market);
5. EV inflation from taking the maximum: an exact Monte-Carlo on the real candidate sets
   (truth = de-vigged market probability, model = truth + logit noise of size sigma);
6. variance of Risk by odds level: SD of a 1-unit bet, standard error of ROI, bets needed to see an
   edge, and simulated drawdowns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibration import RANGE, logit

P_BINS = [0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40]
EV_BINS = [-np.inf, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
EV_LABELS = ["< −10 %", "−10…−5 %", "−5…0 %", "0…5 %", "5…10 %", "10…20 %", "≥ 20 %"]
FLB_BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 6.0, np.inf]
FLB_LABELS = ["1.00–1.50", "1.50–2.00", "2.00–2.50", "2.50–3.00", "3.00–3.50", "3.50–5.00", "5.00–6.00", ">6.00"]


def _ci_mean(x, seed=5, b=1000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def _wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def _cal_row(g: pd.DataFrame) -> dict:
    n = len(g)
    k = float(g["won"].sum())
    mp = float(g["p"].mean())
    se = float(np.sqrt((g["p"] * (1 - g["p"])).sum())) / n
    row = {"n": n, "mean_p": mp, "observed": k / n, "ratio": (k / n) / mp if mp else None,
           "z": ((k / n) - mp) / se if se else None, "observed_ci95": _wilson(k, n)}
    pm = g["p_mkt_u"].dropna()
    if len(pm):
        gm = g.loc[g["p_mkt_u"].notna()]
        row.update({"n_priced": int(len(gm)), "mean_p_priced": float(gm["p"].mean()),
                    "mean_p_market": float(gm["p_mkt_u"].mean()), "observed_priced": float(gm["won"].mean())})
    return row


def calibration_rare(df: pd.DataFrame) -> dict:
    """df: prepared candidate rows of ONE season (settled)."""
    r = df.loc[df["cand"] & df["status"].notna() & df["p"].between(*RANGE)]
    out = {"all": _cal_row(r), "by_key": {}, "by_bin": [], "priced_only": {}}
    for k, g in r.groupby("key"):
        if len(g) >= 50:
            out["by_key"][k] = _cal_row(g)
    for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
        g = r.loc[(r["p"] >= lo) & (r["p"] < hi)]
        if len(g):
            out["by_bin"].append({"bin": f"{lo:.2f}–{hi:.2f}", **_cal_row(g)})
    pr = r.loc[r["p_mkt_u"].notna()]
    if len(pr):
        out["priced_only"] = _cal_row(pr)
        out["priced_only_by_bin"] = []
        for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
            g = pr.loc[(pr["p"] >= lo) & (pr["p"] < hi)]
            if len(g):
                out["priced_only_by_bin"].append({"bin": f"{lo:.2f}–{hi:.2f}", **_cal_row(g),
                                                  "market_observed_gap": float(g["won"].mean() - g["p_mkt_u"].mean())})
    return out


def favourite_longshot(df: pd.DataFrame) -> list[dict]:
    """Every priced selection with a complete book (no model involved)."""
    r = df.loc[df["p_mkt"].notna() & df["decided"]]
    out = []
    for lo, hi, lab in zip(FLB_BANDS[:-1], FLB_BANDS[1:], FLB_LABELS):
        g = r.loc[(r["odds"] >= lo) & (r["odds"] < hi)]
        if len(g) < 30:
            continue
        out.append({"band": lab, "n": int(len(g)), "avg_odds": float(g["odds"].mean()),
                    "implied": float((1 / g["odds"]).mean()), "devig": float(g["p_mkt"].mean()),
                    "model_p": float(g["p"].mean()), "observed": float(g["won"].mean()),
                    "blind_roi": float(g["profit"].mean()), "blind_roi_ci95": _ci_mean(g["profit"]),
                    "overround": float(g["overround"].mean())})
    return out


def ev_bins(df: pd.DataFrame, ev_col: str = "ev") -> dict:
    """Priced Risk candidates (production pool, p >= 0.12) by EV bin; realised ROI per bin."""
    r = df.loc[df["cand"] & df["priced"] & df["decided"] & (df["p"] >= 0.12)].copy()
    r["bin"] = pd.cut(r[ev_col], EV_BINS, labels=EV_LABELS, right=False)
    rows = []
    for lab, g in r.groupby("bin", observed=True):
        rows.append({"bin": str(lab), "n": int(len(g)), "mean_ev": float(g[ev_col].mean()),
                     "roi": float(g["profit"].mean()), "roi_ci95": _ci_mean(g["profit"]),
                     "avg_odds": float(g["odds"].mean())})
    x, y = r[ev_col].to_numpy(float), r["profit"].to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2)
    return {"bins": rows, "n": int(len(r)), "slope": float(beta[1]), "slope_se": float(np.sqrt(cov[1, 1])),
            "intercept": float(beta[0]), "mean_ev": float(x.mean()), "roi": float(y.mean())}


def who_is_right(df: pd.DataFrame) -> dict:
    """Outcome ~ a·logit(p_model) + b·logit(p_market) + c on priced Risk candidates with a complete book."""
    r = df.loc[df["cand"] & df["p_mkt_u"].notna() & df["status"].notna() & (df["p"] >= 0.12)]
    X = np.column_stack([logit(r["p"]), logit(r["p_mkt_u"])])
    y = r["won"].to_numpy()
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    pm, pk = np.clip(r["p"], 1e-6, 1 - 1e-6), np.clip(r["p_mkt_u"], 1e-6, 1 - 1e-6)
    ll = lambda p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))  # noqa: E731
    pb = m.predict_proba(X)[:, 1]
    disagree = r.assign(ratio=r["p"] / r["p_mkt_u"])
    rows = []
    for lo, hi in ((0, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.35), (1.35, np.inf)):
        g = disagree.loc[(disagree["ratio"] >= lo) & (disagree["ratio"] < hi)]
        if len(g) >= 30:
            rows.append({"ratio": f"{lo:.2f}–{hi:.2f}" if np.isfinite(hi) else f"≥ {lo:.2f}", "n": int(len(g)),
                         "mean_p_model": float(g["p"].mean()), "mean_p_market": float(g["p_mkt_u"].mean()),
                         "observed": float(g["won"].mean())})
    return {"n": int(len(r)), "w_model": float(m.coef_[0][0]), "w_market": float(m.coef_[0][1]),
            "intercept": float(m.intercept_[0]), "log_loss_model": ll(pm), "log_loss_market": ll(pk),
            "log_loss_blend_in_sample": ll(pb), "by_disagreement": rows}


def max_selection_mc(df: pd.DataFrame, sigmas=(0.0, 0.1, 0.2, 0.3, 0.4), reps: int = 20, seed: int = 3) -> dict:
    """Exact winner's-curse simulation on the real priced candidate sets of a season.

    truth  q = de-vigged market probability (so every bet has true EV = 1/(1+overround) - 1 < 0);
    model  p = sigmoid(logit q + e), e ~ N(0, sigma^2) independent per selection;
    rule   among priced candidates take the max of p·odds - 1 (as Max-EV Risk) or the first with
           p·odds - 1 >= 0 by p^0.5 (as Value Risk, without the odds window).
    Reported: mean model EV of the chosen bet, its true expected ROI (q·odds - 1) and the inflation.
    """
    r = df.loc[df["cand"] & df["p_mkt"].notna() & (df["p"] >= 0.12) & (df["push_p"] < 1e-9)][
        ["match_id", "p_mkt", "odds", "push_p"]]                      # no refundable (whole) lines
    codes = pd.factorize(r["match_id"])[0]
    q, o, push = r["p_mkt"].to_numpy(), r["odds"].to_numpy(), r["push_p"].to_numpy()
    true_ev = q * o - (1 - push)
    k_per_match = np.bincount(codes)
    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(k_per_match)[:-1]]
    out = []
    for s in sigmas:
        model_ev, real_ev, chosen_k = [], [], []
        for _ in range(reps):
            p = 1 / (1 + np.exp(-(logit(q) + rng.normal(0, s, len(q)))))
            ev = p * o - (1 - push)
            evs = ev[order]
            best = np.maximum.reduceat(evs, starts)
            is_best = evs == np.repeat(best, k_per_match)
            idx = order[is_best]
            _, first = np.unique(codes[idx], return_index=True)
            pick = idx[first]
            model_ev.append(ev[pick].mean())
            real_ev.append(true_ev[pick].mean())
        out.append({"sigma": s, "model_ev": float(np.mean(model_ev)), "true_roi": float(np.mean(real_ev)),
                    "inflation": float(np.mean(model_ev) - np.mean(real_ev))})
    ok = df["p_mkt"].notna() & df["cand"] & (df["push_p"] < 1e-9)
    lm, lq = logit(df.loc[ok, "p"]), logit(df.loc[ok, "p_mkt"])
    return {"matches": int(len(k_per_match)), "mean_candidates": float(k_per_match.mean()),
            "sd_logit_model_minus_market": float(np.std(lm - lq)), "rows": out}


def variance_table(odds_levels=(2.75, 3.25, 4.25, 5.5, 7.0), roi=-0.05, n_season=2000, reps=2000, seed=9) -> list[dict]:
    """For a bet at odds o with true ROI r (win probability q = (1 + r) / o)."""
    rng = np.random.default_rng(seed)
    out = []
    for o in odds_levels:
        q = (1 + roi) / o
        sd = o * np.sqrt(q * (1 - q))
        wins = rng.random((reps, n_season)) < q
        prof = np.where(wins, o - 1, -1.0)
        cum = np.cumsum(prof, 1)
        dd = np.max(np.maximum.accumulate(np.maximum(cum, 0), 1) - cum, 1)
        season_roi = prof.mean(1)
        out.append({"odds": o, "win_probability": float(q), "sd_per_bet": float(sd),
                    "se_roi_2000": float(sd / np.sqrt(n_season)),
                    "bets_to_see_5pp": float((1.96 * sd / 0.05) ** 2),
                    "p_positive_season": float((season_roi > 0).mean()),
                    "median_max_drawdown": float(np.median(dd)), "p95_max_drawdown": float(np.quantile(dd, 0.95)),
                    "expected_longest_losing": float(np.mean([_longest(~w) for w in wins[:300]]))})
    return out


def _longest(lost: np.ndarray) -> int:
    best = run = 0
    for x in lost:
        run = run + 1 if x else 0
        best = max(best, run)
    return best
