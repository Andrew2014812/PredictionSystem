"""Does the model carry information that is not in the price, and can it be combined with the price?

Encompassing test (per season, in-sample hypothesis test): a pooled multinomial logit

    P(k) = softmax_k( a_k + b1 log q_k + b2 log p_k )          (a_1 = 0; binary markets: k = 2)

q = de-margined market probability, p = model probability. b2 = 0 means the market encompasses the model;
likelihood-ratio test of b2 = 0 (chi^2, 1 df) and a bootstrap interval of b2 (resampling matches).

Hybrids (walk-forward: every parameter for season s is fitted on seasons 2022/23 .. s-1 only; market input = the
Bet365 OPENING snapshot, i.e. the production book at decision time):
    H0  market only (de-margined, method chosen on 2023/24 + 2024/25)
    H1  linear blend  w p + (1 - w) q
    H2  log-linear pool  softmax(a + b1 log q + b2 log p)
    H3  market correction  softmax(a + b1 log q)   (favourite-longshot correction, no model)
    H4  meta-model: multinomial logistic regression on log q, log p, margin and league
    H5  residual learning: gradient boosting on the match features with the market logit as base margin
        (learns only what the market misses; trees on seasons 2021/22 .. s-1, rounds by early stopping on s-1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import common as C
from . import data as D

MARKETS = ("1X2", "OU25", "AH")
HYBRIDS = ("H0", "H1", "H2", "H3", "H4", "H5")
N_ENC_BOOT = 100
HYB_RU = {"model": "модель FootPredict", "H0": "H0 рынок (Bet365 откр.)", "H1": "H1 линейная смесь",
          "H2": "H2 лог-линейный пул", "H3": "H3 коррекция рынка", "H4": "H4 мета-модель",
          "H5": "H5 остатки к рынку (XGB)"}


# ------------------------------------------------------------------------------------------------ pooled logit
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pool_fit(logs: list[np.ndarray], y: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Parameters [a_2..a_k, b_1..b_m] of softmax(a + sum_j b_j log X_j) by maximum likelihood (analytic gradient);
    returns (theta, negative log-likelihood)."""
    k = logs[0].shape[1]
    m = len(logs)
    n = len(y)
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1.0
    L = np.stack(logs, 0)                                   # (m, n, k)

    def f(theta):
        a = np.r_[0.0, theta[:k - 1]]
        z = a + np.tensordot(theta[k - 1:], L, axes=1)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        P = np.exp(z - lse[:, None])
        nll = -(z[np.arange(n), y] - lse).sum()
        R = P - Y
        g = np.r_[R[:, 1:].sum(0), (L * R[None]).sum((1, 2))]
        return nll, g
    x0 = np.r_[np.zeros(k - 1), np.ones(m) / m] if x0 is None else x0
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B")
    return res.x, float(res.fun)


def pool_predict(theta: np.ndarray, logs: list[np.ndarray]) -> np.ndarray:
    k = logs[0].shape[1]
    a = np.r_[0.0, theta[:k - 1]]
    return _softmax(a + sum(theta[k - 1 + j] * logs[j] for j in range(len(logs))))


def _logs(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1.0))


# ------------------------------------------------------------------------------------------------ data
def frame(market: str, seasons, book: str = "b365", closing: bool = False, method: str = "power") -> pd.DataFrame:
    t = D.load(market, seasons)
    q = C.demargin(D.odds_matrix(t, market, book, closing), method)
    ok = np.all(np.isfinite(q), 1)
    t = t.loc[ok].reset_index(drop=True)
    t.attrs["q"] = q[ok]
    t.attrs["p"] = D.model_matrix(t, market)
    t.attrs["o"] = D.odds_matrix(t, market, "b365", False)
    return t


def encompassing(choice: dict) -> dict:
    out, pv = {}, {}
    refs = [("b365", False), ("ps", False), ("b365", True), ("ps", True), ("bfe", True)]
    for market in MARKETS:
        for s in (2023, 2024, 2025, 2026, "select"):
            seasons = C.SELECT if s == "select" else (s,)
            for book, closing in refs:
                ref = f"{book}_{'close' if closing else 'open'}"
                method = choice.get(f"{market}|{ref}", "shin")
                t = frame(market, seasons, book, closing, method)
                if len(t) < 300:
                    continue
                y = t["y"].to_numpy(int)
                lq, lp = _logs(t.attrs["q"]), _logs(t.attrs["p"])
                th_full, f_full = pool_fit([lq, lp], y)
                th_res, f_res = pool_fit([lq], y)
                k = lq.shape[1]
                lr = 2 * (f_res - f_full)
                p_lr = float(stats.chi2.sf(max(lr, 0.0), 1))
                rng = np.random.default_rng(C.SEED)
                bs = []
                for _ in range(N_ENC_BOOT):
                    i = rng.integers(0, len(y), len(y))
                    bs.append(pool_fit([lq[i], lp[i]], y[i], th_full)[0][k])
                key = f"{market}|{s}|{ref}"
                print("encompassing", key, flush=True)
                out[key] = {"n": int(len(y)), "b_market": float(th_full[k - 1]), "b_model": float(th_full[k]),
                            "b_model_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                            "lr": float(lr), "p_lr": p_lr, "b_market_alone": float(th_res[k - 1]),
                            "ll_gain_per_match": float(lr / 2 / len(y))}
                if s in (2023, 2024):
                    pv[key] = p_lr
    adj = C.holm(pv)
    for k_, v in adj.items():
        out[k_]["p_holm"] = v
    return {"tests": out, "n_tests_holm": len(pv)}


# ------------------------------------------------------------------------------------------------ hybrids
def _blend_w(p, q, y) -> float:
    grid = np.linspace(0, 1, 51)
    ll = [C.logloss_rows(w * p + (1 - w) * q, y).mean() for w in grid]
    return float(grid[int(np.argmin(ll))])


def _meta_X(t: pd.DataFrame, leagues: list[str]) -> np.ndarray:
    q, p, o = t.attrs["q"], t.attrs["p"], t.attrs["o"]
    margin = (1 / o).sum(1, keepdims=True) - 1
    lg = np.column_stack([(t["league"].to_numpy() == l_).astype(float) for l_ in leagues])
    return np.column_stack([_logs(q), _logs(p), margin, lg])


_FEATS = None


def _features() -> pd.DataFrame:
    global _FEATS
    if _FEATS is None:
        from src.features.store import load_features
        from src.features.sets import model_features
        f = load_features()
        num = model_features("result", list(f.columns))[0]
        f = f.loc[f["played"].astype(bool) & f["home_goals"].notna(), ["match_id", "league", "season"] + num]
        f["league_code"] = f["league"].astype("category").cat.codes
        _FEATS = (f, num + ["league_code"])
    return _FEATS


def _market_only(market: str, seasons, method: str) -> pd.DataFrame:
    """Matches with a complete Bet365 opening market (no model needed): match_id, season, y, q columns."""
    from experiments.v6_2.odds_timing import fd
    frames = []
    for s in seasons:
        d = fd.prices(s)
        if market == "1X2":
            o = d[["x_H_b365", "x_D_b365", "x_A_b365"]].to_numpy(float)
            y = np.where(d["hg"] > d["ag"], 0, np.where(d["hg"] == d["ag"], 1, 2))
        else:
            o = d[["ou_OVER_b365", "ou_UNDER_b365"]].to_numpy(float)
            y = np.where(d["hg"] + d["ag"] > 2.5, 0, 1)
        q = C.demargin(o, method)
        ok = np.all(np.isfinite(q), 1)
        f = pd.DataFrame({"match_id": d["match_id"][ok].to_numpy(), "season": s, "y": y[ok]})
        for j in range(q.shape[1]):
            f[f"q{j}"] = q[ok, j]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _residual_xgb(market: str, s: int, method: str, test: pd.DataFrame) -> np.ndarray | None:
    """H5 probabilities for the rows of ``test`` (season s): boosting on the match features with the de-margined
    Bet365 opening logit as base margin; trained on seasons 2021/22 .. s-1 (rounds by early stopping on s-1)."""
    import xgboost as xgb
    if market == "AH":
        return None
    f, cols = _features()
    tr = _market_only(market, range(2021, s), method)
    k = 3 if market == "1X2" else 2
    te = pd.DataFrame({"match_id": test["match_id"].to_numpy(), "season": s, "y": test["y"].to_numpy(int)})
    for j in range(k):
        te[f"q{j}"] = test.attrs["q"][:, j]
    allr = pd.concat([tr, te], ignore_index=True)
    X = allr[["match_id"]].merge(f.drop(columns=["season"]), on="match_id", how="left")
    have = X[cols].notna().any(axis=1).to_numpy()
    q = allr[[f"q{j}" for j in range(k)]].to_numpy(float)
    y = allr["y"].to_numpy(int)
    base = np.log(np.clip(q, 1e-6, 1))
    if k == 2:
        base_m, yy = base[:, 0] - base[:, 1], (y == 0).astype(int)
        params = {"objective": "binary:logistic"}
    else:
        base_m, yy = base, y
        params = {"objective": "multi:softprob", "num_class": 3}
    params.update({"eta": 0.03, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 50,
                   "lambda": 10.0, "nthread": C.N_JOBS, "seed": C.SEED, "verbosity": 0})
    season = allr["season"].to_numpy()
    is_test = np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)]
    Xv = X[cols].to_numpy(np.float32)

    def dm(mask):
        d = xgb.DMatrix(Xv[mask], label=yy[mask])
        d.set_base_margin(base_m[mask].ravel())
        return d
    inner_tr = (season < s - 1) & have & ~is_test
    inner_va = (season == s - 1) & have & ~is_test
    if inner_tr.sum() < 1000 or inner_va.sum() < 500:
        return None
    b = xgb.train(params, dm(inner_tr), 1500, evals=[(dm(inner_va), "va")], early_stopping_rounds=50,
                  verbose_eval=False)
    rounds = max(int(b.best_iteration) + 1, 1)
    final = xgb.train(params, dm((season < s) & have & ~is_test), rounds)
    out = q[is_test].copy()
    idx = np.flatnonzero(have[is_test])
    if len(idx):
        pr = final.predict(dm(is_test & have))
        if k == 2:
            out[idx, 0], out[idx, 1] = pr, 1 - pr
        else:
            out[idx] = pr.reshape(-1, 3)
    C_ROUNDS[f"{market}|{s}"] = {"rounds": rounds, "train_rows": int(((season < s) & have & ~is_test).sum()),
                                 "test_with_features": int(len(idx)), "test_rows": int(is_test.sum())}
    return out


C_ROUNDS: dict = {}


def hybrid_probs(market: str, s: int, method: str) -> tuple[pd.DataFrame, dict, dict]:
    train = frame(market, tuple(range(C.FIRST, s)), "b365", False, method)
    test = frame(market, (s,), "b365", False, method)
    ytr = train["y"].to_numpy(int)
    ptr, qtr = train.attrs["p"], train.attrs["q"]
    p, q = test.attrs["p"], test.attrs["q"]
    probs, params = {"model": p, "H0": q}, {}
    w = _blend_w(ptr, qtr, ytr)
    probs["H1"], params["H1_w"] = w * p + (1 - w) * q, w
    th2, _ = pool_fit([_logs(qtr), _logs(ptr)], ytr)
    probs["H2"], params["H2_theta"] = pool_predict(th2, [_logs(q), _logs(p)]), th2.tolist()
    th3, _ = pool_fit([_logs(qtr)], ytr)
    probs["H3"], params["H3_theta"] = pool_predict(th3, [_logs(q)]), th3.tolist()
    from sklearn.linear_model import LogisticRegression
    leagues = sorted(train["league"].unique())
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(_meta_X(train, leagues), ytr)
    probs["H4"] = lr.predict_proba(_meta_X(test, leagues))
    h5 = _residual_xgb(market, s, method, test)
    if h5 is not None and len(h5) == len(test):
        probs["H5"] = h5
    params["train_rows"], params["test_rows"] = int(len(train)), int(len(test))
    return test, probs, params


def evaluate(choice: dict) -> dict:
    out, pv, cache = {}, {}, {}
    for market in MARKETS:
        method = choice.get(f"{market}|b365_open", "power")
        for s in C.EVAL:
            test, probs, params = hybrid_probs(market, s, method)
            y = test["y"].to_numpy(int)
            ll = {k: C.logloss_rows(v, y) for k, v in probs.items()}
            res = {"n": int(len(y)), "params": params, "method": method}
            for k, v in probs.items():
                res[k] = {"ll": float(ll[k].mean()), "brier": float(C.brier_rows(v, y).mean())}
                if k != "H0":
                    res[k]["d_ll_vs_H0"] = C.paired_diff_ci(ll["H0"], ll[k], 1000)
                    if s in C.SELECT and k != "model":
                        pv[f"{market}|{s}|{k}"] = res[k]["d_ll_vs_H0"]["p_two_sided"]
            out[f"{market}|{s}"] = res
            cache[(market, s)] = (test, probs)
            print(f"hybrids {market} {s}", flush=True)
    adj = C.holm(pv)
    for key, v in adj.items():
        m, s, k = key.split("|")
        out[f"{m}|{s}"][k]["p_holm"] = v
    return {"results": out, "n_tests_holm": len(pv), "xgb_rounds": C_ROUNDS}, cache


# ------------------------------------------------------------------------------------------------ betting
def _close_fair(test: pd.DataFrame, market: str) -> np.ndarray:
    """Closing reference probability: Betfair exchange close (Shin), else Pinnacle close, else Bet365 close."""
    out = np.full((len(test), len(D.MARKETS[market][1])), np.nan)
    for book in ("bfe", "ps", "b365"):
        q = C.demargin(D.odds_matrix(test, market, book, True), "shin")
        fill = ~np.all(np.isfinite(out), 1) & np.all(np.isfinite(q), 1)
        out[fill] = q[fill]
    return out


def betting(cache: dict) -> dict:
    out = {}
    for (market, s), (test, probs) in cache.items():
        o = test.attrs["o"]
        y = test["y"].to_numpy(int)
        qc = _close_fair(test, market)
        mid = test["match_id"].to_numpy()
        for name, p in probs.items():
            ev = p * o - 1
            j = np.nanargmax(np.where(np.isfinite(ev), ev, -9), 1)
            best = ev[np.arange(len(y)), j]
            for tau in (0.0, 0.03, 0.06):
                bet = best > tau
                if bet.sum() == 0:
                    out[f"{market}|{s}|{name}|{tau}"] = {"bets": 0}
                    continue
                oj = o[np.arange(len(y)), j]
                won = (y == j)
                profit = np.where(bet, np.where(won, oj - 1, -1.0), np.nan)
                clv = np.where(bet, oj * qc[np.arange(len(y)), j] - 1, np.nan)
                r = C.roi_summary(profit, oj, won.astype(float), n_boot=1000)
                r["clv_fair"] = C.mean_ci(clv, 1000)
                r["mean_ev"] = float(best[bet].mean())
                # odds ranges of the bets
                r["by_odds"] = {}
                for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 100)):
                    mm = bet & (oj >= lo) & (oj < hi)
                    if mm.sum() >= 30:
                        r["by_odds"][f"{lo}-{hi}"] = {"bets": int(mm.sum()), "roi": float(profit[mm].mean()),
                                                      "clv_fair": float(np.nanmean(clv[mm]))}
                out[f"{market}|{s}|{name}|{tau}"] = r
        _ = mid
    return out


def run() -> dict:
    bench = C.load_json("demargin.json")
    choice = bench["choice"]
    import sys
    enc = C.load_json("encompassing.json") if "--reuse-encompassing" in sys.argv else None
    if enc is None:
        enc = encompassing(choice)
        C.save_json("encompassing.json", enc)
    ev, cache = evaluate(choice)
    C.save_json("hybrids.json", ev)
    bets = betting(cache)
    C.save_json("betting.json", bets)
    # per-match probabilities of every variant (for figures / preregistration checks)
    frames = []
    for (market, s), (test, probs) in cache.items():
        f = pd.DataFrame({c: test[c].to_numpy() for c in ("match_id", "league", "season", "y")})
        f["market"] = market
        for k, v in probs.items():
            for j in range(v.shape[1]):
                f[f"{k}_{j}"] = v[:, j]
        frames.append(f)
    pd.concat(frames, ignore_index=True).to_parquet(C.CACHE / "hybrid_probs.parquet")
    return {"encompassing": enc, "hybrids": ev, "betting": bets}


if __name__ == "__main__":
    C.setup_logging("hybrids")
    run()
