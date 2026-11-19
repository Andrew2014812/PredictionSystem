"""Helpers shared by the Recent-Season-Weighting and Goal-Model studies (v6.3).

* seasons / cut-offs / matches of a replayed season;
* probabilistic metrics (1X2, count targets, score-matrix markets) with per-match losses for paired bootstrap;
* ``main_eval`` — any model outputs through production's engine and Main rule in both universes (close, t09),
  paired with Baseline B of the same universe, Baseline A alongside.
"""
from __future__ import annotations

import contextlib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from experiments.v6_3 import common as C
from experiments.v6_3.replay import data as D
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV
from src.markets import distributions as DIST
from src.prediction import engine as E

CACHE = C.SHARED_CACHE
UNIVERSES = {"close": "closing", "t09": "decision"}
TASK_COLS = {"result": ["p_home", "p_draw", "p_away"], "home_goals": ["lam_home"], "away_goals": ["lam_away"],
             "corners": ["corner_mu"]}
PROD_HALF_LIFE = {"result": 0, "home_goals": 2, "away_goals": 2, "corners": 2}
MAXG = 10


# ------------------------------------------------------------------------------------------------ data
@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    return D.load_features()


def season_rows(season: int) -> pd.DataFrame:
    f = features()
    return f.loc[f["season"] == season].sort_values(["date", "kickoff", "match_id"]).reset_index(drop=True)


def monthly_cutoffs(season: int) -> list[pd.Timestamp]:
    """First match day of the season, then the 1st of every following month up to the last match."""
    m = season_rows(season)
    start, end = m["date"].min(), m["date"].max()
    months = pd.date_range(start.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1), end, freq="MS")
    return [start] + [d for d in months if d > start + pd.Timedelta(days=14)]


def k_match_cutoffs(season: int, k: int) -> list[pd.Timestamp]:
    """Refit days: season start, then the first match day after every further k played matches of the season."""
    m = season_rows(season)
    played = m.loc[m["played"].astype(bool)]
    per_day = played.groupby("date").size().sort_index()
    before = per_day.cumsum().shift(1, fill_value=0)
    cuts, nxt = [m["date"].min()], k
    for day, n in before.items():
        if n >= nxt:
            cuts.append(day)
            nxt = (n // k + 1) * k
    return cuts


@lru_cache(maxsize=1)
def replay_matches() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "matches.parquet")


@lru_cache(maxsize=1)
def replay_results() -> pd.DataFrame:
    return D.results(replay_matches())


@lru_cache(maxsize=2)
def odds_lookup(universe: str) -> dict:
    return D.lookup(pd.read_parquet(CACHE / f"prices_{UNIVERSES[universe]}.parquet"))


@lru_cache(maxsize=1)
def production_outputs() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "outputs_production.parquet")


@lru_cache(maxsize=2)
def baseline_b(universe: str) -> pd.DataFrame:
    b = pd.read_parquet(CACHE / f"baseline_b_{universe}.parquet")
    return b.loc[b["match_id"].isin(replay_results().index)].reset_index(drop=True)


@lru_cache(maxsize=1)
def baseline_a() -> pd.DataFrame:
    a = pd.read_parquet(CACHE / "baseline_a.parquet")
    return a.loc[a["match_id"].isin(replay_results().index)].reset_index(drop=True)


def season_games(rows: pd.DataFrame) -> pd.Series:
    """Matches of the season already played by the less experienced of the two teams (features of the match)."""
    return np.minimum(rows["h_season_games"].fillna(0), rows["a_season_games"].fillna(0))


# ------------------------------------------------------------------------------------------------ metrics
RES_IDX = {"H": 0, "D": 1, "A": 2}


def onex2_losses(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    p = p / p.sum(1, keepdims=True)
    oh = np.eye(3)[y]
    return pd.DataFrame({"logloss": -np.log(p[np.arange(len(y)), y]), "brier": ((p - oh) ** 2).sum(1),
                         "correct": (p.argmax(1) == y).astype(float)})


def onex2_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    los = onex2_losses(p, y)
    pred = np.asarray(p).argmax(1)
    conf = np.asarray(p).max(1)
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = sum(abs(los["correct"][bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in np.unique(bins))
    return {"n": int(len(y)), "accuracy": float(los["correct"].mean()), "log_loss": float(los["logloss"].mean()),
            "brier": float(los["brier"].mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "ece_top": float(ece),
            "mean_p": [float(x) for x in np.asarray(p).mean(0)], "freq": [float((y == k).mean()) for k in range(3)]}


def poisson_dev(y, mu) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def count_losses(mu, y) -> pd.DataFrame:
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    return pd.DataFrame({"abs": np.abs(mu - y), "sq": (mu - y) ** 2, "dev": poisson_dev(y, mu),
                         "nll": -stats.poisson.logpmf(y.astype(int), np.clip(mu, 1e-9, None))})


def count_metrics(mu, y) -> dict:
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    ok = np.isfinite(mu) & np.isfinite(y)
    mu, y = mu[ok], y[ok]
    los = count_losses(mu, y)
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": int(len(y)), "mae": float(los["abs"].mean()), "rmse": float(np.sqrt(los["sq"].mean())),
            "poisson_deviance": float(los["dev"].mean()), "r2": float(1 - los["sq"].sum() / ss),
            "mean_pred": float(mu.mean()), "mean_true": float(y.mean())}


# ------------------------------------------------------------------------------------------------ score matrices
TOTAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
HCP_LINES = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
_I, _J = np.indices((MAXG + 1, MAXG + 1))


def poisson_matrices(lh, la) -> np.ndarray:
    k = np.arange(MAXG + 1)
    ph = stats.poisson.pmf(k[None, :], np.asarray(lh, float)[:, None])
    pa = stats.poisson.pmf(k[None, :], np.asarray(la, float)[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    return m / m.sum((1, 2), keepdims=True)


def reconcile(mats: np.ndarray, p: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Production reconciliation (alpha = 1): H / D / A regions rescaled to the classifier's probabilities;
    alpha < 1 moves the regions only part of the way (geometric)."""
    out = mats.copy()
    for k, mask in enumerate(DIST.outcome_masks(MAXG + 1)):
        mass = mats[:, mask].sum(1)
        f = np.where(mass > 0, p[:, k] / np.maximum(mass, 1e-15), 1.0) ** alpha
        out[:, mask] *= f[:, None]
    return out / out.sum((1, 2), keepdims=True)


def score_losses(mats: np.ndarray, hg, ag) -> pd.DataFrame:
    """Per-match losses of the markets derived from score matrices (n x 11 x 11)."""
    hg = np.clip(np.asarray(hg, int), 0, MAXG)
    ag = np.clip(np.asarray(ag, int), 0, MAXG)
    n = len(hg)
    eps = 1e-12
    out = {"score_ll": -np.log(np.clip(mats[np.arange(n), hg, ag], eps, 1))}
    tot = hg + ag
    ll_t, br_t = [], []
    for line in TOTAL_LINES:
        p = mats[:, (_I + _J) > line].sum(1)
        y = (tot > line).astype(float)
        ll_t.append(-(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
        br_t.append((p - y) ** 2)
        if line == 2.5:
            out["ou25_ll"], out["ou25_brier"], out["p_over25"], out["y_over25"] = ll_t[-1], br_t[-1], p, y
    out["totals_ll"], out["totals_brier"] = np.mean(ll_t, 0), np.mean(br_t, 0)
    p = mats[:, (_I > 0) & (_J > 0)].sum(1)
    y = ((hg > 0) & (ag > 0)).astype(float)
    out["btts_ll"] = -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    out["btts_brier"] = (p - y) ** 2
    out["p_btts"], out["y_btts"] = p, y
    ll_h, br_h = [], []
    for line in HCP_LINES:
        p = mats[:, (_I + line - _J) > 0].sum(1)
        y = ((hg + line - ag) > 0).astype(float)
        ll_h.append(-(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
        br_h.append((p - y) ** 2)
    out["hcp_ll"], out["hcp_brier"] = np.mean(ll_h, 0), np.mean(br_h, 0)
    # margin distribution (ranked probability score over the goal difference, -5..+5)
    diffs = np.arange(-5, 6)
    pm = np.stack([mats[:, np.clip(_I - _J, -5, 5) == d].sum(1) for d in diffs], 1)
    ym = (np.clip(hg - ag, -5, 5)[:, None] <= diffs[None, :]).astype(float)
    out["margin_rps"] = ((np.cumsum(pm, 1) - ym) ** 2).sum(1) / (len(diffs) - 1)
    # total-goals distribution RPS (0..8+)
    tg = np.arange(0, 9)
    pt = np.stack([mats[:, np.clip(_I + _J, 0, 8) == t].sum(1) for t in tg], 1)
    yt = (np.clip(tot, 0, 8)[:, None] <= tg[None, :]).astype(float)
    out["total_rps"] = ((np.cumsum(pt, 1) - yt) ** 2).sum(1) / (len(tg) - 1)
    # 1X2 implied by the matrix
    ph = np.stack([mats[:, msk].sum(1) for msk in DIST.outcome_masks(MAXG + 1)], 1)
    y3 = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    out["x12_ll"] = -np.log(np.clip(ph[np.arange(n), y3], eps, 1))
    # top exact score: probability vs hit
    flat = mats.reshape(n, -1)
    top = flat.argmax(1)
    out["top_score_p"] = flat[np.arange(n), top]
    out["top_score_hit"] = (top == hg * (MAXG + 1) + ag).astype(float)
    return pd.DataFrame(out)


SCORE_KEYS = ["score_ll", "totals_ll", "totals_brier", "ou25_ll", "ou25_brier", "btts_ll", "btts_brier", "hcp_ll",
              "hcp_brier", "margin_rps", "total_rps", "x12_ll"]


def calibration_table(p, y, bins=10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": int(b), "n": int((idx == b).sum()), "p": float(p[idx == b].mean()), "freq": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).sum() > 0]


def ece(p, y, bins=10) -> float:
    t = calibration_table(p, y, bins)
    n = sum(r["n"] for r in t)
    return float(sum(r["n"] / n * abs(r["p"] - r["freq"]) for r in t))


# ------------------------------------------------------------------------------------------------ Main ROI
@contextlib.contextmanager
def matrix_patch(matrix_fn):
    """Inside the block production's engine builds every score matrix with ``matrix_fn(lh, la, p1x2)``
    instead of ``reconcile_with_1x2(score_matrix(lh, la, rho), p1x2)``; all the rest is production's code."""
    if matrix_fn is None:
        yield
        return
    keep_sm, keep_rc = E.score_matrix, E.reconcile_with_1x2
    E.score_matrix = lambda lh, la, rho=0.0, **k: (float(lh), float(la))
    E.reconcile_with_1x2 = lambda raw, p: matrix_fn(raw[0], raw[1], np.asarray(p, float))
    try:
        yield
    finally:
        E.score_matrix, E.reconcile_with_1x2 = keep_sm, keep_rc


def complete_outputs(outputs: pd.DataFrame) -> pd.DataFrame:
    """Outputs for every replay match: given columns where present, production's otherwise."""
    prod = production_outputs().set_index("match_id")
    o = outputs.set_index("match_id")
    full = prod.copy()
    for c in EN.OUTPUT_COLS:
        if c in o:
            v = o[c].reindex(full.index)
            full[c] = v.where(v.notna(), full[c])
    return full.reset_index()


def main_tables(outputs: pd.DataFrame, label: str, extras: dict | None = None, matrix_fn=None) -> dict:
    """{universe: Main table (played matches)} of the given outputs (missing columns from production)."""
    m, res = replay_matches(), replay_results()
    out = complete_outputs(outputs)
    tabs = {}
    for u in UNIVERSES:
        with matrix_patch(matrix_fn):
            batch = EN.batch_from_outputs(m, out, odds_lookup(u), label, extras)
        cands = EV.candidates(batch.markets, m, res)
        main = batch.picks.loc[batch.picks["role"] == "main", ["match_id", "market", "selection"]]
        t = cands.merge(main, on=["match_id", "market", "selection"])
        t = t.loc[t["match_id"].isin(res.index)].reset_index(drop=True)
        t["method"] = label
        tabs[u] = t
    return tabs


def main_eval(tabs: dict) -> dict:
    out = {}
    for u, t in tabs.items():
        s = EV.summary(t)
        out[u] = {"summary": s, "roi_ci": EV.roi_with_ci(t), "vs_b": EV.compare(t, baseline_b(u)),
                  "vs_a": EV.compare(t, baseline_a()), "groups": t["group"].value_counts().to_dict()}
    return out


def roi_row(name: str, ev: dict, u: str) -> dict:
    s, r, c = ev[u]["summary"], ev[u]["roi_ci"], ev[u]["vs_b"]
    return {"variant": name, "universe": u, "bets": s["bets"], "hit_rate": s["hit_rate"], "avg_odds": s["avg_odds"],
            "roi": s["roi"], "roi_lo": r["ci95"][0], "roi_hi": r["ci95"][1], "d_roi": c["roi_diff"]["diff"],
            "d_lo": c["roi_diff"]["ci95"][0], "d_hi": c["roi_diff"]["ci95"][1], "p": c["roi_diff"]["p"],
            "same_main": c["same_main_share"], "profit": s["profit"]}


def big_win_dependence(t: pd.DataFrame, base: pd.DataFrame, top: int = 5) -> dict:
    """ROI difference to the baseline after removing the ``top`` largest single profits of the variant."""
    b = t.loc[t["bet"] == 1]
    drop = b.nlargest(top, "profit")["match_id"]
    a2 = t.loc[~t["match_id"].isin(drop)]
    b2 = base.loc[~base["match_id"].isin(drop)]
    def roi(x):
        x = x.loc[x["bet"] == 1]
        return float(x["profit"].sum() / len(x)) if len(x) else None
    return {"top": top, "roi_without": roi(a2), "base_roi_same_matches": roi(b2),
            "d_roi_without": (roi(a2) - roi(b2)) if roi(a2) is not None else None}


def odds_sensitivity(t: pd.DataFrame, cut: float = 0.02) -> dict:
    """ROI if every price were 2 % lower (a worse bookmaker) — robustness to the odds."""
    b = t.loc[t["bet"] == 1]
    won = b["status"] == "WON"
    prof = np.where(won, b["odds"] * (1 - cut) - 1, -1.0)
    return {"odds_cut": cut, "roi": float(prof.mean()) if len(b) else None}
