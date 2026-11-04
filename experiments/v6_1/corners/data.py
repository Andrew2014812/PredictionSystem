"""Data, folds and feature sets of the v6.1 corners study.

Folds (season = start year), as in experiments/v6_research/common.py:

    fold s:  OUTER  fit on seasons 2021 .. s-1, score season s
             INNER  fit on seasons 2021 .. s-2, predict season s-1   (early stopping, NB size,
                    blend weights, calibration and tuning are fitted here only)

    2023, 2024   selection folds (choices are made here)
    2025         information only (looked at during earlier development)
    2026         2026/27 to date: independent check, the only season with real corner prices

All statistics come from the Football-Data match files (HC/AC corners, HS/AS shots, HST/AST shots
on target, HF/AF fouls, HY/AY/HR/AR cards). Fouls are not in the feature table; they are read from the
same raw files through ``src.data.dataset.build_matches`` (read only).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features.sets import GROUPS, model_features
from src.features.store import load_features

log = logging.getLogger("corners")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CACHE = HERE / "cache"
for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(exist_ok=True)

FOLDS = (2023, 2024, 2025, 2026)
SELECT = (2023, 2024)
INFO = 2025
CHECK = 2026
LINES = tuple(config.CORNER_LINES)          # product lines 8.5 / 9.5 / 10.5
N_JOBS = 3                                  # shared machine


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    for noisy in ("hyperopt", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log


# ---------------------------------------------------------------------------------------------- frame
_FRAME: pd.DataFrame | None = None


def fouls() -> pd.DataFrame:
    """home_fouls / away_fouls per match from the raw Football-Data files (cached)."""
    path = CACHE / "fouls.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from src.data.dataset import build_matches
    m = build_matches()
    out = m.loc[:, ["match_id", "home_fouls", "away_fouls"]].drop_duplicates("match_id")
    out.to_parquet(path)
    return out


def frame() -> pd.DataFrame:
    """Played matches from 2021/22 with corner statistics, plus fouls and the extra features."""
    global _FRAME
    if _FRAME is not None:
        return _FRAME
    path = CACHE / "frame.parquet"
    if path.exists():
        _FRAME = pd.read_parquet(path)
        return _FRAME
    t0 = time.time()
    df = load_features()
    df = df.loc[df["played"].astype(bool) & df["home_goals"].notna() & (df["season"] >= config.FIRST_SEASON)]
    df = df.merge(fouls(), on="match_id", how="left")
    df = df.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    df, _ = add_extra_features(df)
    df.to_parquet(path)
    log.info("frame: %d rows, %d columns (%.0fs)", len(df), df.shape[1], time.time() - t0)
    _FRAME = df
    return df


def with_corners(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["total_corners"].notna() & df["home_corners"].notna() & df["away_corners"].notna()]


def fold_rows(season: int) -> dict[str, pd.DataFrame]:
    df = with_corners(frame())
    return {"inner_train": df.loc[df["season"] < season - 1], "inner_valid": df.loc[df["season"] == season - 1],
            "outer_train": df.loc[df["season"] < season], "test": df.loc[df["season"] == season]}


# ---------------------------------------------------------------------------------------------- extra features
EWM_SPECS = (("cf", 5), ("cf", 15), ("ca", 5), ("ca", 15), ("sf", 10), ("sa", 10), ("tf", 10), ("ta", 10),
             ("ff", 10), ("fa", 10), ("tot", 10))


def _ewm(series_by_team, hl: int):
    return series_by_team.transform(lambda s: s.ewm(halflife=hl, min_periods=1, ignore_na=True).mean())


def add_extra_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strictly pre-match team statistics from the match history.

    For every team (country, team) the matches are sorted by date and the series is SHIFTED by one
    match before any average, so a match never sees itself or later matches (a team plays at most
    once per day, so same-day matches of other teams are irrelevant here).

        x[hl]_cf / _ca     EWMA of corners won / conceded (half-life 5 and 15 matches)
        x10_sf / _sa       EWMA of shots for / against; x10_tf / _ta shots on target
        x10_ff / _fa       EWMA of fouls committed / suffered (fouls by the opponent)
        x10_tot            EWMA of the total corners of the team's matches
        xv5_cf / _ca       venue-specific EWMA (half-life 5) of corners for / against
        x_cps              corners per shot (x15_cf / x10_sf)
        x_oppadj_cf        own EWMA corners minus what the opponents usually concede (pressure vs strength)
    Match level: additive and multiplicative corner expectations, a shots-based expectation and
    |Elo difference| (lopsided matches).
    """
    sides = []
    for side, opp, venue in (("home", "away", "H"), ("away", "home", "A")):
        sides.append(pd.DataFrame({
            "match_id": df["match_id"].to_numpy(), "date": df["date"].to_numpy(),
            "country": df["country"].to_numpy(), "team": df[f"{side}_team"].to_numpy(), "venue": venue,
            "cf": df[f"{side}_corners"].to_numpy(float), "ca": df[f"{opp}_corners"].to_numpy(float),
            "sf": df[f"{side}_shots"].to_numpy(float), "sa": df[f"{opp}_shots"].to_numpy(float),
            "tf": df[f"{side}_sot"].to_numpy(float), "ta": df[f"{opp}_sot"].to_numpy(float),
            "ff": df[f"{side}_fouls"].to_numpy(float), "fa": df[f"{opp}_fouls"].to_numpy(float)}))
    long = pd.concat(sides, ignore_index=True).sort_values(["date", "match_id"], kind="mergesort")
    long["tot"] = long["cf"] + long["ca"]
    key = [long["country"], long["team"]]
    g = long.groupby(key, sort=False)
    prev = {c: g[c].shift(1) for c in ("cf", "ca", "sf", "sa", "tf", "ta", "ff", "fa", "tot")}
    out = pd.DataFrame(index=long.index)
    for col, hl in EWM_SPECS:
        out[f"x{hl}_{col}"] = _ewm(prev[col].groupby(key, sort=False), hl)
    out["x_nhist"] = g.cumcount()
    vkey = [long["country"], long["team"], long["venue"]]
    gv = long.groupby(vkey, sort=False)
    for col in ("cf", "ca"):
        out[f"xv5_{col}"] = _ewm(gv[col].shift(1).groupby(vkey, sort=False), 5)
    out["x_cps"] = out["x15_cf"] / out["x10_sf"].where(out["x10_sf"] > 0)
    out["match_id"], out["venue"] = long["match_id"].to_numpy(), long["venue"].to_numpy()
    h = out.loc[out["venue"] == "H"].drop(columns="venue").set_index("match_id").add_prefix("h_")
    a = out.loc[out["venue"] == "A"].drop(columns="venue").set_index("match_id").add_prefix("a_")
    feat = h.join(a)
    feat = df[["match_id", "league_avg_home_corners", "league_avg_away_corners", "elo_diff",
               "league_avg_corners"]].set_index("match_id").join(feat)
    lh, la = feat["league_avg_home_corners"], feat["league_avg_away_corners"]
    lavg = (lh + la) / 2
    # opponent-adjusted pressure: own corners vs what the CURRENT opponent usually concedes
    feat["h_x_oppadj_cf"] = feat["h_x15_cf"] - feat["a_x15_ca"]
    feat["a_x_oppadj_cf"] = feat["a_x15_cf"] - feat["h_x15_ca"]
    feat["x_exp_home_add"] = (feat["h_x15_cf"] + feat["a_x15_ca"]) / 2
    feat["x_exp_away_add"] = (feat["a_x15_cf"] + feat["h_x15_ca"]) / 2
    feat["x_exp_total_add"] = feat["x_exp_home_add"] + feat["x_exp_away_add"]
    feat["x_exp_home_mult"] = feat["h_xv5_cf"] * feat["a_xv5_ca"] / lh
    feat["x_exp_away_mult"] = feat["a_xv5_cf"] * feat["h_xv5_ca"] / la
    feat["x_exp_total_mult"] = feat["x_exp_home_mult"] + feat["x_exp_away_mult"]
    feat["x_exp_total_shots"] = (feat["h_x10_sf"] + feat["a_x10_sa"] + feat["a_x10_sf"] + feat["h_x10_sa"]) / 2
    feat["x_tot_hist"] = (feat["h_x10_tot"] + feat["a_x10_tot"]) / 2 - 2 * lavg
    feat["x_abs_elo_diff"] = feat["elo_diff"].abs()
    feat = feat.drop(columns=["league_avg_home_corners", "league_avg_away_corners", "elo_diff",
                              "league_avg_corners"])
    names = list(feat.columns)
    merged = df.merge(feat.reset_index(), on="match_id", how="left")
    return merged, names


def extra_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"^([ha]_x|x_)", c)]


def leakage_check(df: pd.DataFrame, n: int = 60) -> dict:
    """Recompute h_x5_cf and a_x10_sf by brute force from strictly earlier matches of the team."""
    sample = df.loc[df["season"] >= 2022].sample(n, random_state=11)
    errs = []
    for r in sample.itertuples():
        for side, col, raw_for, hl in (("home", "h_x5_cf", "corners", 5), ("away", "a_x10_sf", "shots", 10)):
            team = getattr(r, f"{side}_team")
            past = df.loc[(df["date"] < r.date) & (df["country"] == r.country)
                          & ((df["home_team"] == team) | (df["away_team"] == team))].sort_values(["date", "match_id"])
            vals = np.where(past["home_team"] == team, past[f"home_{raw_for}"], past[f"away_{raw_for}"])
            ref = pd.Series(vals, dtype=float).ewm(halflife=hl, min_periods=1, ignore_na=True).mean()
            ref = ref.iloc[-1] if len(ref) else np.nan
            val = getattr(r, col)
            if np.isnan(ref) and np.isnan(val):
                errs.append(0.0)
            else:
                errs.append(abs(float(ref) - float(val)) if np.isfinite(ref) and np.isfinite(val) else np.inf)
    return {"checked": len(errs), "max_abs_diff": float(np.max(errs))}


# ---------------------------------------------------------------------------------------------- feature sets
def group_cols(groups, cols) -> list[str]:
    pats = [re.compile(p) for g in groups for p in GROUPS[g]]
    return [c for c in cols if any(p.search(c) for p in pats)]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)
    prod = model_features("corners", cols)[0]
    extra = extra_names(df)
    added = group_cols(["opp_adjusted_corners", "elo", "discipline"], cols)
    enriched = list(dict.fromkeys(prod + added + extra))
    compact = ["league_avg_corners", "league_avg_home_corners", "league_avg_away_corners",
               "h_x15_cf", "h_x15_ca", "a_x15_cf", "a_x15_ca", "h_x5_cf", "h_x5_ca", "a_x5_cf", "a_x5_ca",
               "h_xv5_cf", "h_xv5_ca", "a_xv5_cf", "a_xv5_ca", "h_x10_sf", "h_x10_sa", "a_x10_sf", "a_x10_sa",
               "h_x10_tf", "a_x10_tf", "h_x10_ff", "a_x10_ff", "x_exp_total_add", "x_exp_total_mult",
               "x_exp_total_shots", "x_tot_hist", "x_abs_elo_diff", "h_season_corners_for",
               "a_season_corners_for", "h_season_corners_against", "a_season_corners_against",
               "h_form10_corner_share", "a_form10_corner_share", "h_form10_shot_share", "a_form10_shot_share"]
    compact = [c for c in compact if c in cols]
    return {"prod": prod, "enriched": enriched, "compact": compact, "extra": extra, "added_groups": added}


def inventory(df: pd.DataFrame, sets: dict) -> dict:
    """Which statistics exist where, and how strongly each feature relates to total corners (2021–22)."""
    raw = ["home_corners", "away_corners", "home_shots", "away_shots", "home_sot", "away_sot", "home_fouls",
           "away_fouls", "home_yellows", "away_yellows", "home_reds", "away_reds"]
    full = load_features()
    full = full.loc[full["played"].astype(bool) & (full["season"] >= config.FIRST_SEASON)].merge(
        fouls(), on="match_id", how="left")
    cov_league = full.groupby("league")[raw].apply(lambda d: d.notna().mean()).round(4)
    cov_season = full.groupby("season")[raw].apply(lambda d: d.notna().mean()).round(4)
    counts = full.groupby(["league", "season"]).size().unstack(fill_value=0)
    train = df.loc[df["season"] < 2023]
    y = train["total_corners"]
    rows = []
    for c in sets["enriched"]:
        v = train[c].astype(float)
        ok = v.notna() & y.notna()
        r = float(np.corrcoef(v[ok], y[ok])[0, 1]) if ok.sum() > 100 and v[ok].std() > 0 else None
        rows.append({"feature": c, "in_production": c in sets["prod"], "new": c in sets["extra"],
                     "coverage": float(v.notna().mean()), "corr_total": r})
    rows.sort(key=lambda r: -abs(r["corr_total"] or 0))
    df = with_corners(df)
    stats = df.groupby("season").agg(n=("total_corners", "size"), mean=("total_corners", "mean"),
                                     var=("total_corners", "var"), home=("home_corners", "mean"),
                                     away=("away_corners", "mean"))
    hc = {s: float(np.corrcoef(d["home_corners"], d["away_corners"])[0, 1]) for s, d in df.groupby("season")}
    stats["corr_home_away"] = pd.Series(hc)
    by_league = df.groupby("league").agg(n=("total_corners", "size"), mean=("total_corners", "mean"),
                                         var=("total_corners", "var"))
    return {"coverage_by_league": cov_league.to_dict(orient="index"),
            "coverage_by_season": {str(k): v for k, v in cov_season.to_dict(orient="index").items()},
            "matches_by_league_season": {lg: {str(s): int(n) for s, n in r.items()} for lg, r in counts.iterrows()},
            "season_stats": {str(k): v for k, v in stats.to_dict(orient="index").items()},
            "league_stats": by_league.to_dict(orient="index"),
            "feature_counts": {k: len(v) for k, v in sets.items()},
            "features": rows, "extra_features": sets["extra"], "added_groups": sets["added_groups"],
            "compact": sets["compact"]}


# ---------------------------------------------------------------------------------------------- IO
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_result(name: str, obj) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def load_result(name: str, default=None):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
