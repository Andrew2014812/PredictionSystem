"""Long candidate tables: one row per (match, Main-eligible selection) with probability, result and real prices.

Historical seasons (2022–2025): probabilities from the walk-forward production models (base.py), built with the
same mathematics as src/prediction/engine.py (independent-Poisson score matrix, MAX_GOALS = 10, reconciled with
the 1X2 classifier; negative-binomial corners), checked against the production functions (``self_check``).
Prices: Football-Data opening and closing prices (fd_prices.py). The PRODUCTION price of a selection is the
opening Bet365 price exactly as src/providers/football_data.py attaches it (1X2, total 2.5, the one half/whole
Asian line of the match).

Season 2026/27: probabilities and recorded prices from the production store (markets.parquet, the latest
pre-kick-off version of every match of the unified history); a selection without a recorded price gets the last
OddsPapi quote strictly before kick-off from the price archive (the same repair production applies to its stored
predictions, src/prediction/history_repair.py). Football-Data 2026/27 opening / closing prices are attached
to the markets they cover (for CLV and the like-for-like universe).

Column ``status``: 1 won, 0 lost, 2 refund (Asian whole line), -1 unknown (e.g. corners not reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_research import scores as SC
from src import config
from src.markets.distributions import corners_distribution
from src.markets.markets import handicap_market, is_half_line

from . import base
from . import common as C
from . import fd_prices

BOOK_KEYS = ("b365", "ps", "max", "avg")
TOTAL_LINES = config.TOTAL_GOAL_LINES
AH_LINES = config.HANDICAP_LINES
CORNER_LINES = config.CORNER_LINES


# ------------------------------------------------------------------------------------------------ de-margining
def shin(implied: np.ndarray) -> np.ndarray:
    """Shin (1993) probabilities from an (n, k) array of 1/odds (rows with NaN stay NaN)."""
    pi = np.asarray(implied, float)
    s = pi.sum(1, keepdims=True)
    lo, hi = np.zeros((len(pi), 1)), np.full((len(pi), 1), 0.4)

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / s) - z) / (2 * (1 - z))

    for _ in range(60):
        mid = (lo + hi) / 2
        tot = probs(mid).sum(1, keepdims=True)
        hi = np.where(tot < 1, mid, hi)        # sum decreases in z
        lo = np.where(tot >= 1, mid, lo)
    p = probs((lo + hi) / 2)
    return p / p.sum(1, keepdims=True)


def proportional(implied: np.ndarray) -> np.ndarray:
    pi = np.asarray(implied, float)
    return pi / pi.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ historical seasons
def _status_1x2(hg, ag, sel):
    out = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    return (out == sel).astype(np.int8)


def _settle(market: str, sel: str, hg, ag, corners) -> np.ndarray:
    hg, ag = np.asarray(hg, float), np.asarray(ag, float)
    if market == "1X2":
        return _status_1x2(hg, ag, sel)
    if market == "DOUBLE_CHANCE":
        o = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
        allowed = {"1X": "HD", "X2": "DA", "12": "HA"}[sel]
        return np.isin(o, list(allowed)).astype(np.int8)
    if market.startswith("TOTAL_"):
        line = float(market.split("_")[1])
        return (((hg + ag) > line) == (sel == "OVER")).astype(np.int8)
    if market == "BTTS":
        return (((hg > 0) & (ag > 0)) == (sel == "YES")).astype(np.int8)
    if market.startswith("HANDICAP_"):
        line = float(market.split("_")[1])
        m = hg + line - ag
        won = (m > 0) if sel == "H" else (m < 0)
        return np.where(m == 0, 2, won.astype(int)).astype(np.int8)
    if market.startswith("CORNERS_"):
        line = float(market.split("_")[1])
        c = np.asarray(corners, float)
        return np.where(np.isnan(c), -1, ((c > line) == (sel == "OVER")).astype(int)).astype(np.int8)
    raise ValueError(market)


def _probabilities(pred: pd.DataFrame, nb_size) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(market, selection) -> (win probability, refund probability), production mathematics."""
    m = SC.reconcile(SC.matrices(pred["lh"].to_numpy(), pred["la"].to_numpy()),
                     pred[["pH", "pD", "pA"]].to_numpy())
    I, J = SC._I, SC._J
    z = np.zeros(len(pred))
    ph, pd_, pa = (m * (I > J)).sum((1, 2)), (m * (I == J)).sum((1, 2)), (m * (I < J)).sum((1, 2))
    out = {("1X2", "H"): (ph, z), ("1X2", "D"): (pd_, z), ("1X2", "A"): (pa, z),
           ("DOUBLE_CHANCE", "1X"): (ph + pd_, z), ("DOUBLE_CHANCE", "X2"): (pd_ + pa, z),
           ("DOUBLE_CHANCE", "12"): (ph + pa, z)}
    for line in TOTAL_LINES:
        over = (m * ((I + J) > line)).sum((1, 2))
        out[(f"TOTAL_{line}", "OVER")] = (over, z)
        out[(f"TOTAL_{line}", "UNDER")] = (1 - over, z)
    btts = (m * ((I > 0) & (J > 0))).sum((1, 2))
    out[("BTTS", "YES")], out[("BTTS", "NO")] = (btts, z), (1 - btts, z)
    for line in AH_LINES:
        margin = I + line - J
        wh, push, wa = (m * (margin > 0)).sum((1, 2)), (m * (margin == 0)).sum((1, 2)), (m * (margin < 0)).sum((1, 2))
        mk = handicap_market(line)
        out[(mk, "H")], out[(mk, "A")] = (wh, push), (wa, push)
    cmu = pred["cmu"].to_numpy(float)
    ok = ~np.isnan(cmu)
    if ok.any():
        pmfs = np.full((len(pred), 41), np.nan)
        for i in np.flatnonzero(ok):
            pmfs[i] = corners_distribution(cmu[i], nb_size)
        k = np.arange(41)
        for line in CORNER_LINES:
            over = (pmfs * (k > line)).sum(1)
            over[~ok] = np.nan
            out[(f"CORNERS_{line}", "OVER")] = (over, z)
            out[(f"CORNERS_{line}", "UNDER")] = (1 - over, z)
    return out


def _attach_fd(long: pd.DataFrame, fd: pd.DataFrame, closing: bool) -> pd.DataFrame:
    """Add price columns o_<book>[_c] for the markets Football-Data covers."""
    suf = "_c" if closing else ""
    fd = fd.set_index("match_id")
    keyed = fd.reindex(long["match_id"])
    for book in BOOK_KEYS:
        col = f"o_{book}{suf}"
        v = np.full(len(long), np.nan)
        mk, sel = long["market"].to_numpy(), long["selection"].to_numpy()
        for s in "HDA":
            m = (mk == "1X2") & (sel == s)
            v[m] = keyed[f"x_{s}_{book}{suf}"].to_numpy()[m]
        for s in ("OVER", "UNDER"):
            m = (mk == "TOTAL_2.5") & (sel == s)
            v[m] = keyed[f"ou_{s}_{book}{suf}"].to_numpy()[m]
        line = keyed[f"ah_line{suf}"].to_numpy(float)
        ok_line = np.array([x == x and (is_half_line(x) or x % 1 == 0) for x in line])
        row_line = long["line"].to_numpy(float)
        for s in "HA":
            m = (long["group"].to_numpy() == "HANDICAP") & (sel == s) & ok_line & np.isclose(row_line, line)
            v[m] = keyed[f"ah_{s}_{book}{suf}"].to_numpy()[m]
        long[col] = v
    named_books = [b for b in fd_prices.NAMED if f"x_H_{b}{suf}" in fd.columns]
    # best named price at the same snapshot (1X2 has more books; totals / AH: Bet365 and Pinnacle)
    with np.errstate(all="ignore"):
        v = np.fmax(long[f"o_b365{suf}"].to_numpy(float), long[f"o_ps{suf}"].to_numpy(float))
    mk, sel = long["market"].to_numpy(), long["selection"].to_numpy()
    for s in "HDA":
        m = (mk == "1X2") & (sel == s)
        stack = np.vstack([keyed[f"x_{s}_{b}{suf}"].to_numpy() for b in named_books])
        with np.errstate(all="ignore"):
            v[m] = np.fmax.reduce(stack, 0)[m]
    long[f"o_best{suf}"] = v
    return long


def historical(season: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"cand_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    pred, info = base.outer(season)
    long = long_from_pred(pred, info["nb_size"], season)
    long.to_parquet(path)
    return long


def long_from_pred(pred: pd.DataFrame, nb_size, season: int, fd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate table from one row per match with pH/pD/pA, lh/la, cmu and the results."""
    probs = _probabilities(pred, nb_size)
    hg, ag, tc = pred["home_goals"].to_numpy(), pred["away_goals"].to_numpy(), pred["total_corners"].to_numpy()
    frames = []
    for (mk, sel), (p, push) in probs.items():
        ok = ~np.isnan(p)
        frames.append(pd.DataFrame({"mi": np.flatnonzero(ok), "market": mk, "selection": sel,
                                    "p": p[ok].astype(float), "p_push": push[ok].astype(float),
                                    "status": _settle(mk, sel, hg, ag, tc)[ok]}))
    long = pd.concat(frames, ignore_index=True)
    meta = pred[["match_id", "league", "season", "date"]].reset_index(drop=True)
    long = pd.concat([meta.iloc[long["mi"]].reset_index(drop=True), long], axis=1)
    long["group"] = long["market"].map(lambda m: "1X2" if m == "1X2" else ("DOUBLE_CHANCE" if m == "DOUBLE_CHANCE"
                                                                           else ("BTTS" if m == "BTTS" else m.split("_")[0])))
    long["line"] = long["market"].map(lambda m: float(m.split("_")[1]) if "_" in m and m.split("_")[0] in
                                      ("TOTAL", "HANDICAP", "CORNERS") else np.nan)
    fd = fd_prices.prices(season) if fd is None else fd
    long = _attach_fd(long, fd, closing=False)
    long = _attach_fd(long, fd, closing=True)
    # production price: opening Bet365 (src/providers/football_data.py); identical to the feature-table columns
    long["odds"] = long["o_b365"]
    long["o_prod"] = long["odds"]
    long["source"] = "walk_forward"
    long = add_market_probabilities(long)
    return long.sort_values(["date", "match_id", "market", "selection"]).reset_index(drop=True)


def add_market_probabilities(long: pd.DataFrame) -> pd.DataFrame:
    """De-margined bookmaker probabilities q_<book>_<method> for every fully priced market of a match.

    For an Asian whole line the two prices are conditional on no refund; q is that conditional probability and
    is converted with the model's refund probability: q_win = q_cond x (1 - p_push)."""
    for book in [c[2:] for c in long.columns if c.startswith("o_")]:
        col = f"o_{book}"
        for method, fn in (("prop", proportional), ("shin", shin)):
            long[f"q_{book}_{method}"] = np.nan
        sub = long.loc[long[col].notna(), ["match_id", "market", "selection", col]]
        if sub.empty:
            continue
        for mk_kind, sels in (("1X2", ["H", "D", "A"]), ("TOTAL", ["OVER", "UNDER"]), ("BTTS", ["YES", "NO"]),
                              ("HANDICAP", ["H", "A"]), ("CORNERS", ["OVER", "UNDER"]),
                              ("DOUBLE_CHANCE", ["1X", "X2", "12"])):
            part = sub.loc[(sub["market"] == mk_kind) if mk_kind in ("1X2", "BTTS", "DOUBLE_CHANCE")
                           else sub["market"].str.startswith(mk_kind + "_")]
            if part.empty:
                continue
            wide = part.pivot_table(index=["match_id", "market"], columns="selection", values=col, aggfunc="first")
            wide = wide.reindex(columns=sels).dropna()
            if wide.empty:
                continue
            implied = 1.0 / wide.to_numpy(float)
            for method, fn in (("prop", proportional), ("shin", shin)):
                if mk_kind == "DOUBLE_CHANCE":
                    # each 1X2 outcome is in two of the three double-chance selections: probabilities sum to 2
                    q = implied / implied.sum(1, keepdims=True) * 2 if method == "prop" else shin(implied / 2) * 2
                else:
                    q = fn(implied)
                qs = pd.DataFrame(q, index=wide.index, columns=pd.Index(sels, name="selection")).stack().rename("q").reset_index()
                key = pd.MultiIndex.from_frame(qs[["match_id", "market", "selection"]])
                lk = pd.MultiIndex.from_frame(long[["match_id", "market", "selection"]])
                pos = key.get_indexer(lk)
                hit = pos >= 0
                vals = long[f"q_{book}_{method}"].to_numpy(float)
                vals[hit] = qs["q"].to_numpy()[pos[hit]]
                long[f"q_{book}_{method}"] = vals
        whole = (long["group"] == "HANDICAP") & (long["p_push"] > 0)
        for method in ("prop", "shin"):
            c = f"q_{book}_{method}"
            long.loc[whole, c] = long.loc[whole, c] * (1 - long.loc[whole, "p_push"])
    return long


def self_check(season: int = 2024, n: int = 60) -> dict:
    """Probabilities and the Main pick of the vectorised table vs the production functions on n matches."""
    from src.markets.distributions import reconcile_with_1x2, score_matrix
    from src.markets.markets import corner_markets, goal_markets
    from src.markets.odds import price
    from src.markets.recommendation import main_prediction
    from src.providers.football_data import odds_from_matches
    from src.providers.schema import odds_lookup
    pred, info = base.outer(season)
    long = historical(season)
    feats = base.V6.seasons(base.V6.features(), [season])
    lookup = odds_lookup(odds_from_matches(feats))
    rng = np.random.default_rng(0)
    rows = pred.iloc[rng.choice(len(pred), n, replace=False)]
    max_diff, same_main, odds_diff = 0.0, 0, 0.0
    from . import strategies as ST
    for r in rows.itertuples():
        m = reconcile_with_1x2(score_matrix(r.lh, r.la), np.array([r.pH, r.pD, r.pA]))
        sels = goal_markets(m)
        if r.cmu == r.cmu:
            sels += corner_markets(corners_distribution(r.cmu, info["nb_size"]))
        priced = price(sels, lookup.get(r.match_id))
        ref = main_prediction(priced)
        mine = long.loc[long["match_id"] == r.match_id].set_index(["market", "selection"])
        for s in priced:
            if (s.market, s.selection) in mine.index:
                row = mine.loc[(s.market, s.selection)]
                max_diff = max(max_diff, abs(row["p"] - s.probability))
                if s.has_odds:
                    odds_diff = max(odds_diff, abs(row["odds"] - s.odds))
        pick = ST.pick_current(long.loc[long["match_id"] == r.match_id])
        same_main += (pick["market"].iloc[0], pick["selection"].iloc[0]) == (ref.pick.market, ref.pick.selection)
    return {"matches": n, "max_probability_diff": max_diff, "max_odds_diff": odds_diff, "same_main": same_main}


# ------------------------------------------------------------------------------------------------ 2026/27 (store)
def live(refresh: bool = False) -> pd.DataFrame:
    """Candidate table of the 2026/27 unified history (latest pre-kick-off version of every match)."""
    path = C.CACHE / "cand_live.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from src.analytics.performance import unified_history
    from src.features.store import load_features
    from src.prediction import store
    from src.prediction.history_repair import archive_prices
    picks = unified_history(store.load_picks())
    mains = picks.loc[(picks["role"] == "main") & (picks["season"] == C.LIVE)]
    ids = set(mains["match_id"])
    mk = store.load_markets()
    mk = mk.loc[mk["match_id"].isin(ids) & ~mk["group"].isin(["EHANDICAP", "EXACT_SCORE"])].copy()
    mk = mk.drop_duplicates(["match_id", "market", "selection"], keep="last")
    feats = load_features()
    feats = feats.loc[feats["match_id"].isin(ids)]
    # fill missing prices from the archive: last OddsPapi quote strictly before kick-off
    need = mk.loc[mk["odds"].isna(), ["match_id", "market", "selection"]].assign(source="live", odds=np.nan)
    arch = archive_prices(need, feats)
    mk["odds_origin"] = np.where(mk["odds"].notna(), "recorded", None)
    if not arch.empty:
        a = arch.set_index(["match_id", "market", "selection"])
        key = pd.MultiIndex.from_frame(mk[["match_id", "market", "selection"]])
        pos = a.index.get_indexer(key)
        fill = (pos >= 0) & mk["odds"].isna().to_numpy()
        mk.loc[fill, "odds"] = a["odds"].to_numpy()[pos[fill]]
        mk.loc[fill, "bookmaker"] = a["bookmaker"].to_numpy()[pos[fill]]
        mk.loc[fill, "provider"] = "oddspapi"
        mk.loc[fill, "priced_at"] = pd.to_datetime(a["collected_at"].to_numpy()[pos[fill]])
        mk.loc[fill, "odds_origin"] = "archive"
    meta = feats.set_index("match_id")
    res = mains.drop_duplicates("match_id").set_index("match_id")
    out = pd.DataFrame({"match_id": mk["match_id"].to_numpy(), "market": mk["market"].to_numpy(),
                        "selection": mk["selection"].to_numpy(), "group": mk["group"].to_numpy(),
                        "p": mk["probability"].astype(float).to_numpy(),
                        "p_push": mk["push_probability"].fillna(0).astype(float).to_numpy(),
                        "odds": mk["odds"].astype(float).to_numpy(), "bookmaker": mk["bookmaker"].to_numpy(),
                        "provider": mk["provider"].to_numpy(), "odds_origin": mk["odds_origin"].to_numpy(),
                        "priced_at": pd.to_datetime(mk["priced_at"]).to_numpy()})
    out["league"] = out["match_id"].map(res["league"])
    out["season"] = C.LIVE
    out["date"] = pd.to_datetime(out["match_id"].map(res["date"]))
    out["source"] = out["match_id"].map(res["source"])
    out["model_version"] = out["match_id"].map(res["model_version"])
    out["kickoff"] = pd.to_datetime(out["match_id"].map(meta["kickoff"]))
    hg = out["match_id"].map(meta["home_goals"]).to_numpy(float)
    ag = out["match_id"].map(meta["away_goals"]).to_numpy(float)
    tc = out["match_id"].map(meta["total_corners"]).to_numpy(float)
    st = np.full(len(out), -1, np.int8)
    for (mkt, sel), idx in out.groupby(["market", "selection"]).indices.items():
        s = _settle(mkt, sel, hg[idx], ag[idx], tc[idx])
        s = np.where(np.isnan(hg[idx]), -1, s)
        st[idx] = s
    out["status"] = st
    out["line"] = out["market"].map(lambda m: float(m.split("_")[1]) if "_" in m and m.split("_")[0] in
                                    ("TOTAL", "HANDICAP", "CORNERS") else np.nan)
    out["mi"] = pd.factorize(out["match_id"])[0]
    fd = fd_prices.prices(C.LIVE)
    out = _attach_fd(out, fd, closing=False)
    out = _attach_fd(out, fd, closing=True)
    out["o_prod"] = out["odds"]
    out = add_market_probabilities(out)
    out = out.sort_values(["date", "match_id", "market", "selection"]).reset_index(drop=True)
    out.to_parquet(path)
    return out


def stored_live_mains() -> pd.DataFrame:
    """The Main predictions exactly as production stored them (unified history, 2026/27)."""
    from src.analytics.performance import unified_history
    from src.prediction import store
    picks = unified_history(store.load_picks())
    return picks.loc[(picks["role"] == "main") & (picks["season"] == C.LIVE)].copy()


if __name__ == "__main__":
    C.setup_logging("cand")
    for s in C.OUTER_SEASONS:
        d = historical(s, refresh=True)
        C.log.info("season %s: %d rows, %d matches", s, len(d), d["match_id"].nunique())
    d = live(refresh=True)
    C.log.info("live: %d rows, %d matches", len(d), d["match_id"].nunique())
    print(self_check())
