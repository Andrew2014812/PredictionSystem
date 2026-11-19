"""Data of the v6.3 corners study: the match frame, strictly pre-match corner features and dynamic ratings.

Seasons (season = start year): 2021/22 … 2025/26 are history, 2026/27 is the replayed season.

    select stage   fit on 2021/22–2024/25, predict 2025/26  -> every choice (model, distribution, weights, maps)
    eval stage     fit on 2021/22–2025/26, predict 2026/27  -> chronological check (season already studied)
    wf stage       monthly refits inside 2026/27 on everything played before the 1st of the month

Statistics come from the Football-Data match files (HC/AC corners, HS/AS shots, HST/AST on target, HF/AF fouls,
HY/AY cards). Fouls are not in the feature table; they are read once from the same raw files through
``src.data.dataset.build_matches`` (read only) and cached in ``cache/fouls.parquet``.

Every new feature of a match is computed from the team's EARLIER matches only: the per-team series is shifted by
one match before any average (a team plays at most once a day), and the dynamic ratings are updated day by day
after all matches of the day were predicted.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from src.features.sets import model_features

NAME = "corners"
DIRS = C.study_dirs(NAME)
CACHE, RESULTS, FIGURES, LOGS = DIRS["cache"], DIRS["results"], DIRS["figures"], DIRS["logs"]
FIRST = 2021
SELECT, EVAL = 2025, 2026
LINES = (8.5, 9.5, 10.5)

ID_COLS = ["match_id", "league", "country", "season", "date", "kickoff", "home_team", "away_team", "played",
           "home_goals", "away_goals", "total_corners", "home_corners", "away_corners", "home_shots", "away_shots",
           "home_sot", "away_sot", "home_yellows", "away_yellows", "elo_diff", "market_prob_home", "market_prob_draw",
           "market_prob_away", "market_prob_over25", "league_avg_corners", "league_avg_home_corners",
           "league_avg_away_corners"]


def fouls() -> pd.DataFrame:
    path = CACHE / "fouls.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from src.data.dataset import build_matches
    m = build_matches()
    out = m.loc[:, ["match_id", "home_fouls", "away_fouls"]].drop_duplicates("match_id")
    out.to_parquet(path)
    return out


def raw_features() -> tuple[pd.DataFrame, list[str]]:
    import pyarrow.parquet as pq
    cols = pq.read_schema(C.DATA_FILES["features"]).names
    prod = model_features("corners", cols)[0]
    use = list(dict.fromkeys([c for c in ID_COLS if c in cols] + prod))
    f = pd.read_parquet(C.DATA_FILES["features"], columns=use)
    f["date"] = pd.to_datetime(f["date"])
    f = f.loc[f["season"] >= FIRST].copy()
    for c in f.columns:
        if f[c].dtype == "float64" and c not in ("total_corners",):
            f[c] = f[c].astype("float32")
    return f, prod


# ---------------------------------------------------------------------------------------------- EWMA features
EWM_SPECS = (("cf", 5), ("cf", 15), ("ca", 5), ("ca", 15), ("sf", 10), ("sa", 10), ("tf", 10), ("ta", 10),
             ("ff", 10), ("fa", 10), ("tot", 10), ("gf", 10), ("ga", 10))


def _long(df: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for side, opp, venue in (("home", "away", "H"), ("away", "home", "A")):
        sides.append(pd.DataFrame({
            "match_id": df["match_id"].to_numpy(), "date": df["date"].to_numpy(), "country": df["country"].to_numpy(),
            "team": df[f"{side}_team"].to_numpy(), "venue": venue,
            "cf": df[f"{side}_corners"].to_numpy(float), "ca": df[f"{opp}_corners"].to_numpy(float),
            "sf": df[f"{side}_shots"].to_numpy(float), "sa": df[f"{opp}_shots"].to_numpy(float),
            "tf": df[f"{side}_sot"].to_numpy(float), "ta": df[f"{opp}_sot"].to_numpy(float),
            "ff": df[f"{side}_fouls"].to_numpy(float), "fa": df[f"{opp}_fouls"].to_numpy(float),
            "gf": df[f"{side}_goals"].to_numpy(float), "ga": df[f"{opp}_goals"].to_numpy(float)}))
    long = pd.concat(sides, ignore_index=True).sort_values(["date", "match_id"], kind="mergesort")
    long["tot"] = long["cf"] + long["ca"]
    return long


def ewma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team form from earlier matches only (series shifted by one match, EWMA by half-life in matches)."""
    long = _long(df)
    key = [long["country"], long["team"]]
    g = long.groupby(key, sort=False)
    out = pd.DataFrame(index=long.index)
    for col, hl in EWM_SPECS:
        prev = g[col].shift(1)
        out[f"x{hl}_{col}"] = prev.groupby(key, sort=False).transform(
            lambda s: s.ewm(halflife=hl, min_periods=1, ignore_na=True).mean())
    out["x_nhist"] = g.cumcount()
    vkey = [long["country"], long["team"], long["venue"]]
    gv = long.groupby(vkey, sort=False)
    for col in ("cf", "ca"):
        out[f"xv5_{col}"] = gv[col].shift(1).groupby(vkey, sort=False).transform(
            lambda s: s.ewm(halflife=5, min_periods=1, ignore_na=True).mean())
    out["x_cps"] = out["x15_cf"] / out["x10_sf"].where(out["x10_sf"] > 0)
    out["x_shot_share"] = out["x10_sf"] / (out["x10_sf"] + out["x10_sa"])
    out["match_id"], out["venue"] = long["match_id"].to_numpy(), long["venue"].to_numpy()
    h = out.loc[out["venue"] == "H"].drop(columns="venue").set_index("match_id").add_prefix("h_")
    a = out.loc[out["venue"] == "A"].drop(columns="venue").set_index("match_id").add_prefix("a_")
    return h.join(a)


# ---------------------------------------------------------------------------------------------- dynamic ratings
RATING_GRID = {"eta": (0.01, 0.02, 0.035, 0.05), "keep": (0.5, 0.8, 1.0)}


def dynamic_ratings(df: pd.DataFrame, home_col: str, away_col: str, eta: float, keep: float,
                    eta_league: float = 0.01, init: tuple[float, float] | None = None) -> pd.DataFrame:
    """Online team ratings of a side count (corners or shots), updated after every match day.

        log μ_home = b_L^H + att_i + con_j        log μ_away = b_L^A + att_j + con_i

    att = how many corners the team wins, con = how many its opponents win against it (log scale, 0 = league
    average). After the day's matches: att_i += η (y_i − μ_i) / μ_i (and the same for con of the opponent);
    league bases move with η_L. At a new season all team ratings are multiplied by ``keep`` (regression to the
    mean). A match is always predicted with the ratings BEFORE its day, so nothing of the day itself is used.
    """
    d = df[["match_id", "date", "season", "league", "country", "home_team", "away_team", home_col, away_col]]
    d = d.sort_values(["date", "match_id"], kind="mergesort")
    init = init or ((5.3, 4.5) if "corners" in home_col else (12.8, 10.9))
    att, con, base = {}, {}, {}
    out = np.full((len(d), 6), np.nan)
    season_seen = None
    dates = d["date"].to_numpy()
    idx_by_day = np.split(np.arange(len(d)), np.flatnonzero(np.diff(dates.astype("int64"))) + 1)
    lg, ct, ht, at = d["league"].to_numpy(), d["country"].to_numpy(), d["home_team"].to_numpy(), d["away_team"].to_numpy()
    yh, ya, ss = d[home_col].to_numpy(float), d[away_col].to_numpy(float), d["season"].to_numpy()
    for day in idx_by_day:
        if season_seen is not None and ss[day[0]] != season_seen:
            for k in att:
                att[k] *= keep
            for k in con:
                con[k] *= keep
        season_seen = ss[day[0]]
        upd = []
        for i in day:
            L = lg[i]
            if L not in base:          # first match of a league: a neutral start (no later match is used)
                base[L] = [np.log(init[0]), np.log(init[1])]
            hk, ak = (ct[i], ht[i]), (ct[i], at[i])
            ah, ch, aa, ca = att.get(hk, 0.0), con.get(hk, 0.0), att.get(ak, 0.0), con.get(ak, 0.0)
            mh, ma = np.exp(base[L][0] + ah + ca), np.exp(base[L][1] + aa + ch)
            out[i] = (mh, ma, ah, ch, aa, ca)
            if np.isfinite(yh[i]) and np.isfinite(ya[i]):
                upd.append((L, hk, ak, yh[i], ya[i], mh, ma))
        for L, hk, ak, y1, y2, mh, ma in upd:
            r1, r2 = (y1 - mh) / mh, (y2 - ma) / ma
            att[hk] = att.get(hk, 0.0) + eta * r1
            con[ak] = con.get(ak, 0.0) + eta * r1
            att[ak] = att.get(ak, 0.0) + eta * r2
            con[hk] = con.get(hk, 0.0) + eta * r2
            base[L][0] += eta_league * r1
            base[L][1] += eta_league * r2
    res = pd.DataFrame(out, columns=["mu_h", "mu_a", "h_att", "h_con", "a_att", "a_con"], index=d.index)
    res["match_id"] = d["match_id"].to_numpy()
    return res.set_index("match_id")


def poisson_dev(y, mu) -> float:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-6, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))


def choose_rating_params(df: pd.DataFrame, log=None) -> dict:
    """Grid of (η, keep) chosen on the side-count Poisson deviance of 2023/24 + 2024/25 (before the selection
    season); 2025/26 and 2026/27 are not looked at."""
    rows = []
    for eta in RATING_GRID["eta"]:
        for keep in RATING_GRID["keep"]:
            r = dynamic_ratings(df, "home_corners", "away_corners", eta, keep)
            j = df.set_index("match_id").join(r)
            sel = j.loc[j["season"].isin([2023, 2024]) & j["total_corners"].notna()]
            dev = (poisson_dev(sel["home_corners"], sel["mu_h"]) + poisson_dev(sel["away_corners"], sel["mu_a"])) / 2
            dev_t = poisson_dev(sel["total_corners"], sel["mu_h"] + sel["mu_a"])
            rows.append({"eta": eta, "keep": keep, "side_dev_2324_2425": dev, "total_dev_2324_2425": dev_t})
            if log:
                log.info("rating grid eta=%.3f keep=%.2f side dev %.5f", eta, keep, dev)
    best = min(rows, key=lambda r: r["side_dev_2324_2425"])
    return {"grid": rows, "best": best}


# ---------------------------------------------------------------------------------------------- frame
def build_frame(log=None) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    f, prod = raw_features()
    f = f.merge(fouls(), on="match_id", how="left")
    f = f.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    ew = ewma_features(f)
    grid = choose_rating_params(f, log)
    eta, keep = grid["best"]["eta"], grid["best"]["keep"]
    rc = dynamic_ratings(f, "home_corners", "away_corners", eta, keep).add_prefix("rc_")
    rs = dynamic_ratings(f, "home_shots", "away_shots", eta, keep).add_prefix("rs_")
    feat = ew.join(rc).join(rs)
    j = f.set_index("match_id").join(feat)
    lh, la = j["league_avg_home_corners"].astype(float), j["league_avg_away_corners"].astype(float)
    j["n_exp_home_add"] = (j["h_x15_cf"] + j["a_x15_ca"]) / 2
    j["n_exp_away_add"] = (j["a_x15_cf"] + j["h_x15_ca"]) / 2
    j["n_exp_total_add"] = j["n_exp_home_add"] + j["n_exp_away_add"]
    j["n_exp_total_mult"] = j["h_xv5_cf"] * j["a_xv5_ca"] / lh + j["a_xv5_cf"] * j["h_xv5_ca"] / la
    j["n_exp_total_shots"] = (j["h_x10_sf"] + j["a_x10_sa"] + j["a_x10_sf"] + j["h_x10_sa"]) / 2
    j["n_oppadj_h"] = j["h_x15_cf"] - j["a_x15_ca"]
    j["n_oppadj_a"] = j["a_x15_cf"] - j["h_x15_ca"]
    j["n_tot_hist"] = (j["h_x10_tot"] + j["a_x10_tot"]) / 2 - (lh + la)
    j["n_rc_total"] = j["rc_mu_h"] + j["rc_mu_a"]
    j["n_rs_total"] = j["rs_mu_h"] + j["rs_mu_a"]
    j["n_abs_elo"] = j["elo_diff"].abs()
    j["n_fouls_sum"] = j["h_x10_ff"] + j["a_x10_ff"]
    # market profile of the match (Football-Data pre-match prices; production treats them as pre-match)
    ph, pa, po = (j[f"market_prob_{k}"].astype(float) for k in ("home", "away", "over25"))
    j["m_diff"] = ph - pa
    j["m_abs_diff"] = (ph - pa).abs()
    j["m_fav"] = np.maximum(ph, pa)
    j["m_over25"] = po
    j["m_exp_goals"] = _goals_from_over25(po.to_numpy())
    j = j.reset_index()
    new = [c for c in j.columns if c.startswith(("h_x", "a_x", "h_xv", "a_xv", "rc_", "rs_", "n_"))]
    mkt = [c for c in j.columns if c.startswith("m_")]
    sets = {"prod": prod, "stats": list(dict.fromkeys(prod + new)), "stats_mkt": list(dict.fromkeys(prod + new + mkt)),
            "new": new, "market": mkt}
    info = {"rows": int(len(j)), "seconds": round(time.time() - t0, 1), "rating": grid,
            "feature_counts": {k: len(v) for k, v in sets.items()}}
    j.to_parquet(CACHE / "frame.parquet")
    C.write_json(CACHE / "sets.json", sets)
    C.write_json(RESULTS / "frame_info.json", info)
    return j, sets


def _goals_from_over25(p: np.ndarray) -> np.ndarray:
    """Total-goal Poisson rate λ with P(N ≥ 3) = p (bisection, vectorised)."""
    from scipy import stats
    lo, hi = np.full(len(p), 0.2), np.full(len(p), 6.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        over = 1 - stats.poisson.cdf(2, mid)
        lo = np.where(over < p, mid, lo)
        hi = np.where(over < p, hi, mid)
    out = (lo + hi) / 2
    out[~np.isfinite(p)] = np.nan
    return out


def load_frame() -> tuple[pd.DataFrame, dict]:
    path = CACHE / "frame.parquet"
    if not path.exists():
        return build_frame()
    return pd.read_parquet(path), C.read_json(CACHE / "sets.json")


def leakage_check(df: pd.DataFrame, n: int = 40, seed: int = 3) -> dict:
    """Recompute h_x5_cf and a_x10_sf by brute force from strictly earlier matches of the team."""
    sample = df.loc[df["season"] >= 2022].sample(n, random_state=seed)
    errs = []
    for r in sample.itertuples():
        for side, col, raw, hl in (("home", "h_x5_cf", "corners", 5), ("away", "a_x10_sf", "shots", 10)):
            team = getattr(r, f"{side}_team")
            past = df.loc[(df["date"] < r.date) & (df["country"] == r.country)
                          & ((df["home_team"] == team) | (df["away_team"] == team))].sort_values(["date", "match_id"])
            vals = np.where(past["home_team"] == team, past[f"home_{raw}"], past[f"away_{raw}"])
            ref = pd.Series(vals, dtype=float).ewm(halflife=hl, min_periods=1, ignore_na=True).mean()
            ref = ref.iloc[-1] if len(ref) else np.nan
            val = getattr(r, col)
            if not np.isfinite(ref) and not np.isfinite(val):
                errs.append(0.0)
            else:
                errs.append(abs(float(ref) - float(val)) if np.isfinite(ref) and np.isfinite(val) else np.inf)
    return {"checked": len(errs), "max_abs_diff": float(np.max(errs))}


if __name__ == "__main__":
    lg = C.setup_logging("corners.data", LOGS / "data.log")
    frame, s = build_frame(lg)
    lg.info("frame %s, sets %s", frame.shape, {k: len(v) for k, v in s.items()})
    C.write_json(RESULTS / "leakage_check.json", leakage_check(frame))
