"""Main picks of every season and the uncertainty measures of each pick.

Main picks
    2023/24–2025/26 (and 2022/23 as calibration history): the v6.1 walk-forward candidate tables
    (experiments/v6_1/main/cache/cand_<s>.parquet: production mathematics on models fitted on seasons < s, real
    prices = Football-Data Bet365 opening, as production attaches them) and the production Main rule
    (experiments/v6_1/main/strategies.py, checked against src.markets.recommendation there).
    2026/27: the Main production showed (unified history, latest pre-kick-off version) with its stored
    probability and its recorded price (missing prices filled with the last OddsPapi quote before kick-off, as
    production's history repair does: cand_live.parquet of v6.1).

Uncertainty measures of one Main (all known before kick-off)
    u_seed, u_boot, u_sub   standard deviation of the Main's probability over the members of an ensemble family
                            (ensemble.py; the probability of the SAME selection, production mathematics)
    u_1x2                   mean standard deviation of the three 1X2 probabilities over the bootstrap members
    u_missing               share of missing model inputs of the match (1X2 model features)
    u_va                    width p1 - p0 of the inductive Venn-Abers interval of the Main probability, calibrated
                            on the previous season's Mains (walk-forward)
    conf_size               size of the split-conformal 1X2 prediction set (alpha = 0.2), calibrated on the
                            previous season
    2026/27 only (recorded predictions): n_versions, main_changed, p_drift (|p last - p first|), p_sd_versions
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiments.v6_1.main import candidates as CD
from experiments.v6_1.main import strategies as ST

from . import common as C
from . import ensemble as E

CAND_COLS = ["match_id", "league", "season", "date", "market", "selection", "group", "line", "p", "p_push",
             "status", "odds"]
CONF_ALPHA = 0.2


def candidates(season: int) -> pd.DataFrame:
    name = "cand_live.parquet" if season == C.CURRENT else f"cand_{season}.parquet"
    d = pd.read_parquet(C.V61_CACHE / name, columns=CAND_COLS)
    for c in ("league", "market", "selection", "group"):
        d[c] = d[c].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    return d.reset_index(drop=True)


def _stored_keys() -> tuple[set, pd.DataFrame]:
    from src.analytics.performance import unified_history
    picks = pd.read_parquet(C.ROOT / "local_data/predictions/picks.parquet")
    h = unified_history(picks)
    h = h.loc[(h["role"] == "main") & (h["season"] == C.CURRENT)]
    return set(zip(h["match_id"], h["market"], h["selection"])), h


def mains(season: int) -> pd.DataFrame:
    """One Main per match: match_id, league, date, market, selection, group, p, p_push, status, odds."""
    d = candidates(season)
    if season == C.CURRENT:
        keys, h = _stored_keys()
        m = d.loc[[k in keys for k in zip(d["match_id"], d["market"], d["selection"])]]
        m = m.drop_duplicates("match_id").copy()
        src = h.drop_duplicates("match_id").set_index("match_id")["source"]
        m["source"] = m["match_id"].map(src)
    else:
        d["mi"] = pd.factorize(d["match_id"])[0]
        m = d.iloc[ST.choose(d)].copy()
        m["source"] = "walk_forward"
    m["y"] = np.where(m["status"] == 1, 1.0, np.where(m["status"] == 0, 0.0, np.nan))
    m["pc"] = m["p"] / np.clip(1 - m["p_push"].fillna(0.0), 1e-6, None)
    odds = m["odds"].to_numpy(float)
    priced = np.isfinite(odds) & (odds > 1.0)
    m["profit"] = np.where(priced & m["y"].notna(), np.where(m["y"] == 1, odds - 1, -1.0), np.nan)
    return m.sort_values(["date", "match_id"]).reset_index(drop=True)


def outer_1x2(season: int) -> pd.DataFrame:
    """Walk-forward 1X2 probabilities and result of a past season (v6.1 outer predictions)."""
    o = pd.read_parquet(C.V61_CACHE / f"outer_{season}.parquet", columns=["match_id", "y", "pH", "pD", "pA"])
    return o


def live_1x2() -> pd.DataFrame:
    d = candidates(C.CURRENT)
    d = d.loc[d["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="p", aggfunc="first")
    return pd.DataFrame({"match_id": d.index, "pH": d["H"].to_numpy(), "pD": d["D"].to_numpy(),
                         "pA": d["A"].to_numpy()})


# ------------------------------------------------------------------------------------------------ ensemble measures
def member_main_probs(m: pd.DataFrame, family: str, season: int) -> np.ndarray:
    """(n_mains, K) probability of each Main's selection under every member of the family (NaN if missing)."""
    ens, infos = E.load(family, season)
    K = C.FAMILIES[family]
    out = np.full((len(m), K), np.nan)
    if ens.empty:
        return out
    keys = list(zip(m["market"], m["selection"]))
    for info in infos:
        k = info["k"]
        pred = ens.loc[ens["member"] == k].set_index("match_id").reindex(m["match_id"]).reset_index()
        ok = pred["pH"].notna().to_numpy()
        if not ok.any():
            continue
        probs = CD._probabilities(pred.loc[ok].reset_index(drop=True), info["nb_size"])
        idx_ok = np.flatnonzero(ok)
        for j, i in enumerate(idx_ok):
            pair = probs.get(keys[i])
            if pair is not None:
                out[i, k] = pair[0][j]
    return out


def member_1x2(m: pd.DataFrame, family: str, season: int) -> np.ndarray:
    ens, _ = E.load(family, season)
    K = C.FAMILIES[family]
    out = np.full((len(m), K, 3), np.nan)
    for k in range(K):
        sub = ens.loc[ens["member"] == k].set_index("match_id").reindex(m["match_id"])
        out[:, k, :] = sub[["pH", "pD", "pA"]].to_numpy(float)
    return out


def missing_share(match_ids) -> pd.Series:
    from experiments.v6_research import common as V6
    cols = V6.task_numeric("result")
    f = pd.read_parquet(C.ROOT / "local_data/processed/features.parquet", columns=["match_id"] + cols)
    f = f.drop_duplicates("match_id", keep="last").set_index("match_id")
    return f[cols].isna().mean(axis=1).reindex(list(match_ids))


def venn_abers_width(cal_p, cal_y, test_p, grid: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inductive Venn-Abers (p0, p1) for every test probability (computed on a 0.001 grid of p)."""
    cal_p, cal_y = np.asarray(cal_p, float), np.asarray(cal_y, float)
    test_p = np.asarray(test_p, float)
    rounded = np.round(test_p / grid) * grid
    uniq = np.unique(rounded[np.isfinite(rounded)])
    p0, p1 = {}, {}
    for u in uniq:
        x = np.r_[cal_p, u]
        for lab, store in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, np.r_[cal_y, lab])
            store[u] = float(iso.predict([u])[0])
    a = np.array([p0.get(r, np.nan) for r in rounded])
    b = np.array([p1.get(r, np.nan) for r in rounded])
    return a, b, b - a


def conformal_set_size(cal: pd.DataFrame, test_p: np.ndarray, alpha: float = CONF_ALPHA) -> tuple[np.ndarray, float]:
    """Split-conformal 1X2 sets {k : 1 - p_k <= q}, q = the (1 - alpha) quantile of 1 - p_y on ``cal``."""
    p = cal[["pH", "pD", "pA"]].to_numpy(float)
    y = cal["y"].to_numpy(int)
    s = 1 - p[np.arange(len(y)), y]
    n = len(s)
    q = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
    size = ((1 - np.asarray(test_p, float)) <= q).sum(1)
    return size, q


def version_stability(match_ids) -> pd.DataFrame:
    """Recorded (live) Main of 2026/27 across its pre-kick-off versions."""
    v = pd.read_parquet(C.ROOT / "local_data/predictions/versions.parquet",
                        columns=["match_id", "role", "market", "selection", "probability", "version", "generated_at",
                                 "source"])
    v = v.loc[(v["role"] == "main") & v["match_id"].isin(set(match_ids))].sort_values(["match_id", "version"])
    g = v.groupby("match_id")
    out = pd.DataFrame({
        "n_versions": g["version"].nunique(),
        "n_distinct_main": g.apply(lambda x: len(set(zip(x["market"], x["selection"]))), include_groups=False),
        "p_first": g["probability"].first(), "p_last": g["probability"].last(),
        "p_sd_versions": g["probability"].std(ddof=0)})
    out["main_changed"] = (out["n_distinct_main"] > 1).astype(float)
    out["p_drift"] = (out["p_last"] - out["p_first"]).abs()
    return out.reset_index()


def build(season: int) -> pd.DataFrame:
    """Mains of a season with every uncertainty measure (cached)."""
    path = C.CACHE / f"mains_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    m = mains(season)
    for fam in C.FAMILIES:
        P = member_main_probs(m, fam, season)
        m[f"pm_{fam}"] = np.nanmean(P, 1)
        m[f"u_{fam}"] = np.nanstd(P, 1)
        m[f"lo_{fam}"] = np.nanmin(P, 1)
        m[f"hi_{fam}"] = np.nanmax(P, 1)
        m[f"k_{fam}"] = np.isfinite(P).sum(1)
    P3 = member_1x2(m, "boot", season)
    m["u_1x2"] = np.nanmean(np.nanstd(P3, 1), 1)
    m["u_missing"] = missing_share(m["match_id"]).to_numpy()
    prev = mains(season - 1)
    cal = prev.loc[prev["y"].notna()]
    p0, p1, w = venn_abers_width(cal["pc"], cal["y"], m["pc"])
    m["va_p0"], m["va_p1"], m["u_va"] = p0, p1, w
    cal1 = outer_1x2(season - 1)
    test = live_1x2() if season == C.CURRENT else outer_1x2(season)
    t = test.set_index("match_id").reindex(m["match_id"])
    ok = t[["pH", "pD", "pA"]].notna().all(1).to_numpy()
    size = np.full(len(m), np.nan)
    s, q = conformal_set_size(cal1, t.loc[ok, ["pH", "pD", "pA"]].to_numpy(float))
    size[ok] = s
    m["conf_size"] = size
    m.attrs["conformal_q"] = q
    if season == C.CURRENT:
        st = version_stability(m["match_id"])
        m = m.merge(st, on="match_id", how="left")
    m.to_parquet(path)
    (C.CACHE / f"mains_{season}.json").write_text(json.dumps({"conformal_q": q, "rows": int(len(m))}))
    return m
