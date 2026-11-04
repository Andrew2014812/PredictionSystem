"""Risk strategies as vectorised rules over a candidate table (one chosen row per match).

Every strategy starts from the production candidate pool of src/markets/recommendation.py::
risk_prediction: no European handicap, never the Main selection, never a selection with the same
settlement footprint as Main or a bet against Main (column ``eligible``); candidates are the pool
rows with p >= RISK_MIN_PROBABILITY, and when a match has none the whole pool is used.

The production rule score is reproduced exactly (``score``); ``check_current`` verifies the
vectorised "current" strategy against the production function match by match.

Strategy families (all run on the same matches):

    current            production risk_prediction()
    floor F            priced candidates with model EV >= F -> best production score; otherwise current.
                       F = 0 is "Value Risk" (positive EV first); F = none is the current rule.
    max-EV             priced candidates with EV >= 0 -> the highest EV; otherwise current (diagnostic:
                       the most EV-greedy rule)
    calibrated F       probabilities replaced by a calibration fitted on the previous season
                       (Platt per selection type; "blend": model + market price), EV recomputed with
                       them; priced candidates with calibrated EV >= F first, else the best score with
                       calibrated probabilities
    market-specific    candidates restricted to selection types whose probabilities in 0.12-0.40 were
                       NOT significantly over-estimated in the previous season; otherwise current
    conservative       production score with the preferred odds window 2.50-3.50 (instead of 3.0-6.0)
    high-odds          production score with the preferred odds window 4.50-8.00
    band B, floor F    only priced candidates with odds in band B (and EV >= F); the best of them by
                       reliability x p^0.5 (inside the band the odds preference is flat); no fallback, so
                       coverage < 100 % (analysis of odds bands, not a production candidate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

RISK_MIN = config.RISK_MIN_PROBABILITY
PREFERRED = tuple(config.RISK_PREFERRED_ODDS)
SOFTNESS = config.RISK_ODDS_SOFTNESS
POWER = config.RISK_PROBABILITY_POWER
EV_WEIGHT = config.RISK_EV_WEIGHT
UNPRICED = config.RISK_UNPRICED_FACTOR
EV_CLIP = 0.3

FLOORS = {"≥0 %": 0.0, "≥−3 %": -0.03, "≥−5 %": -0.05, "≥−8 %": -0.08, "нет": None}
BANDS = {"2.50–3.00": (2.50, 3.00), "3.00–3.50": (3.00, 3.50), "3.50–5.00": (3.50, 5.00),
         "5.00–6.00": (5.00, 6.00), ">6.00": (6.00, np.inf)}
CONSERVATIVE = (2.50, 3.50)
HIGH_ODDS = (4.50, 8.00)


def selection_key(df: pd.DataFrame) -> pd.Series:
    """Selection type used for calibration / reliability: 1X2:D, TOTAL_3.5:OVER, HANDICAP:H, ..."""
    grp = df["group"]
    by_market = grp.isin(["TOTAL", "CORNERS"])
    base = np.where(by_market, df["market"], grp)
    sel = np.where(grp == "EXACT_SCORE", "", df["selection"])
    return pd.Series([f"{b}:{s}" if s else b for b, s in zip(base, sel)], index=df.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priced"] = df["odds"].notna() & (df["odds"] > 1.0)
    df["key"] = selection_key(df)
    df["rel"] = df["group"].map(lambda g: config.MARKET_RELIABILITY.get(g, 0.8)).astype(float)
    df["decided"] = df["status"].isin(["WON", "LOST"])
    df["won"] = (df["status"] == "WON").astype(float)
    df["profit"] = np.where(df["status"] == "WON", df["odds"] - 1.0,
                            np.where(df["status"] == "LOST", -1.0, 0.0))
    # candidate pool of risk_prediction: eligible and p >= min, else the whole eligible pool
    cand = df["eligible"] & (df["p"] >= RISK_MIN)
    has = cand.groupby(df["match_id"]).transform("any")
    df["cand"] = df["eligible"] & (cand | ~has)
    return df


def odds_preference(odds, low: float, high: float, soft: float) -> np.ndarray:
    odds = np.asarray(odds, float)
    x = np.log(np.maximum(odds, 1.0 + 1e-12))
    d = np.where(odds < low, np.log(low) - x, np.where(odds > high, x - np.log(high), 0.0))
    out = np.exp(-0.5 * (d / soft) ** 2) if soft > 0 else (d == 0).astype(float)
    return np.where(odds > 1.0, out, 0.0)


def score(df: pd.DataFrame, p_col: str = "p", ev_col: str = "ev", preferred=PREFERRED, soft=SOFTNESS,
          power=POWER, ev_weight=EV_WEIGHT, unpriced=UNPRICED, flat=False) -> np.ndarray:
    p = df[p_col].to_numpy(float)
    eff = np.where(df["priced"], df["odds"], 1.0 / np.maximum(p, 1e-9))
    pref = 1.0 if flat else odds_preference(eff, *preferred, soft)
    base = df["rel"].to_numpy() * np.maximum(p, 0.0) ** power * pref
    ev = np.clip(np.nan_to_num(df[ev_col].to_numpy(float)), -EV_CLIP, EV_CLIP)
    return np.where(df["priced"], base * (1 + ev_weight * ev), base * unpriced)


def choose(df: pd.DataFrame, tiers: list[np.ndarray], key: np.ndarray, p_col: str = "p") -> pd.DataFrame:
    """Per match: the first tier with any row, inside it the row with the highest key
    (ties: priced, then probability) - the ordering of recommendation._choose."""
    rank = np.full(len(df), np.inf)
    for i, t in reversed(list(enumerate(tiers))):
        rank = np.where(np.asarray(t, bool), i, rank)
    sub = df.assign(_rank=rank, _key=key).loc[np.isfinite(rank)]
    if sub.empty:
        return sub
    sub = sub.loc[sub["_rank"] == sub.groupby("match_id")["_rank"].transform("min")]
    sub = sub.sort_values(["match_id", "_key", "priced", p_col], ascending=[True, False, False, False],
                          kind="stable")
    out = sub.drop_duplicates("match_id", keep="first").copy()
    out["tier"] = out["_rank"].astype(int)
    return out.drop(columns=["_rank", "_key"])


# ---------------------------------------------------------------------------------------- strategies
def current(df):
    return choose(df, [df["cand"]], score(df))


def floor(df, f):
    if f is None:
        return current(df)
    s = score(df)
    return choose(df, [df["cand"] & df["priced"] & (df["ev"] >= f), df["cand"]], s)


def max_ev(df):
    first = df["cand"] & df["priced"] & (df["ev"] >= 0)
    key = np.where(first, df["ev"].fillna(-9), score(df))
    return choose(df, [first, df["cand"]], key)


def calibrated(df, f, p_col="p_cal"):
    ev_col = f"ev_{p_col}"
    s = score(df, p_col=p_col, ev_col=ev_col)
    tiers = [df["cand"]] if f is None else [df["cand"] & df["priced"] & (df[ev_col] >= f), df["cand"]]
    return choose(df, tiers, s, p_col=p_col)


def market_specific(df, keep_col="reliable"):
    s = score(df)
    return choose(df, [df["cand"] & df[keep_col], df["cand"]], s)


def window(df, preferred):
    return choose(df, [df["cand"]], score(df, preferred=preferred))


def band(df, lo_hi, f=None):
    lo, hi = lo_hi
    m = df["cand"] & df["priced"] & (df["odds"] >= lo) & (df["odds"] < hi)
    if f is not None:
        m &= df["ev"] >= f
    return choose(df, [m], score(df, flat=True))


def catalogue(cal_floor=None, blend_floor=None) -> dict:
    """Name -> (family, function). Floors of the calibrated rules are chosen on 2023 + 2024 (run.py)."""
    out = {"Текущий Risk": ("current", current)}
    for label, f in FLOORS.items():
        if f is not None:
            name = "Value Risk (EV ≥ 0)" if f == 0 else f"Пол EV {label}"
            out[name] = ("floor", lambda d, f=f: floor(d, f))
    out["Max-EV Risk"] = ("diagnostic", max_ev)
    for label, f in FLOORS.items():
        out[f"Calibrated Risk, пол {label}"] = ("calibrated", lambda d, f=f: calibrated(d, f, "p_cal"))
        out[f"Calibrated Risk (бленд с рынком), пол {label}"] = ("calibrated_blend",
                                                                lambda d, f=f: calibrated(d, f, "p_blend"))
    out["Market-specific Risk"] = ("market", market_specific)
    out["Conservative Risk (2.50–3.50)"] = ("window", lambda d: window(d, CONSERVATIVE))
    out["High-odds Risk (4.50–8.00)"] = ("window", lambda d: window(d, HIGH_ODDS))
    return out


def band_catalogue() -> dict:
    out = {}
    for bl, b in BANDS.items():
        for fl, f in FLOORS.items():
            out[(bl, fl)] = lambda d, b=b, f=f: band(d, b, f)
    return out


def check_current(df: pd.DataFrame, n_max: int | None = None) -> dict:
    """The vectorised current rule against production risk_prediction(), match by match."""
    from src.markets.odds import PricedSelection
    from src.markets.recommendation import main_prediction, risk_prediction
    picked = current(df).set_index("match_id")[["market", "selection"]]
    agree = total = 0
    for mid, g in df.groupby("match_id", sort=False):
        if n_max is not None and total >= n_max:
            break
        sels = [PricedSelection(r.market, r.selection, r.p, r.odds if r.odds == r.odds else None,
                                push_probability=r.push_p) for r in g.itertuples()]
        shown = [s for s, m in zip(sels, g["is_main"]) if m]          # the Main of the table
        main = shown[0] if shown else (lambda r: r.pick if r else None)(main_prediction(sels))
        rec = risk_prediction(sels, main)
        total += 1
        agree += rec is not None and mid in picked.index and \
            tuple(picked.loc[mid]) == (rec.pick.market, rec.pick.selection)
    return {"checked": total, "agree": agree}
