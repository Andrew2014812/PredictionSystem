"""Completing stored predictions afterwards, without changing what was predicted.

Three repairs, all safe to run on every pipeline run (they only fill or correct prices, never picks):

* **Archived prices.** A Main / Risk / market forecast stored without a price gets the last real
  OddsPapi quote strictly before kick-off once the price archive has it (``odds_origin = "archive"``).
  Selection, probability, model version and generation time stay as stored; EV and — for a settled
  prediction — profit are computed from the added price. A prediction that has a price keeps it.
* **Missing Risk predictions.** Matches recorded before the Risk rule existed have a Main but no Risk.
  The Risk is chosen by the current Risk rule from the probabilities and pre-match prices stored for
  that very version (``markets.parquet``) — never from the result — and stored with
  ``origin = "risk_reconstructed"`` and the reconstruction time; the Main is never touched.
* **Market averages.** A price that was Football-Data's market average (not one bookmaker's price) is
  replaced by a real bookmaker quote from before kick-off, or removed (the prediction is then not a bet);
  see ``replace_average_prices``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..markets.markets import PUSH, WON
from ..markets.odds import PricedSelection
from ..markets.recommendation import risk_prediction
from ..providers import load_api_odds, pre_match_only
from ..providers.schema import best_prices
from ..storage import predictions_storage
from . import store
from .price_timing import label_prices

log = logging.getLogger(__name__)

ARCHIVE = "archive"
RECONSTRUCTED = "risk_reconstructed"
PROVIDER_ODP = "oddspapi"
REPAIRED_SOURCES = ("live", "backfill")


def _profit(status: pd.Series, odds: np.ndarray) -> np.ndarray:
    priced = ~np.isnan(np.asarray(odds, dtype=float))
    return np.where(status == WON, odds - 1, np.where(status == "LOST", -1.0,
                                                    np.where(status.isin([PUSH, "VOID"]) & priced, 0.0, np.nan)))


def clear_refunds_without_price(picks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A push / void without a real price was no bet: its profit is empty, not 0 (older runs stored 0)."""
    wrong = picks["status"].isin([PUSH, "VOID"]) & picks["odds"].isna() & picks["profit"].notna()
    if wrong.any():
        picks = picks.copy()
        picks.loc[wrong, "profit"] = np.nan
    return picks, int(wrong.sum())


def archive_prices(picks: pd.DataFrame, matches: pd.DataFrame, api: pd.DataFrame | None = None) -> pd.DataFrame:
    """Last pre-kick-off OddsPapi quote per (match, market, selection) of the given picks."""
    api = load_api_odds() if api is None else api
    if api.empty:
        return api
    api = api.loc[(api["provider"] == PROVIDER_ODP) & api["match_id"].isin(picks["match_id"])]
    return best_prices(pre_match_only(api, matches))


def fill_archive_prices(picks: pd.DataFrame, prices: pd.DataFrame, push: pd.Series,
                        sources=REPAIRED_SOURCES) -> tuple[pd.DataFrame, int]:
    out = picks.copy()
    if "odds_origin" not in out:
        out["odds_origin"] = None
    if prices.empty or out.empty:
        return out, 0
    quotes = prices.set_index(["match_id", "market", "selection"])
    keys = pd.MultiIndex.from_frame(out[["match_id", "market", "selection"]])
    todo = (out["source"].isin(sources) & out["odds"].isna() & keys.isin(quotes.index)).to_numpy()
    if not todo.any():
        return out, 0
    q = quotes.loc[keys[todo]]
    idx = out.index[todo]
    odds = q["odds"].to_numpy(float)
    p_push = push.reindex(keys[todo]).fillna(0.0).to_numpy(float) if len(push) else np.zeros(len(idx))
    out.loc[idx, "odds"] = odds
    out.loc[idx, "bookmaker"] = q["bookmaker"].to_numpy()
    out.loc[idx, "provider"] = PROVIDER_ODP
    out.loc[idx, "real_odds"] = True
    out.loc[idx, "odds_origin"] = ARCHIVE
    if "odds_priced_at" in out:
        out.loc[idx, "odds_priced_at"] = pd.to_datetime(q["collected_at"]).to_numpy()
    out.loc[idx, "ev"] = out.loc[idx, "probability"].to_numpy(float) * odds - (1 - p_push)
    out.loc[idx, "profit"] = _profit(out.loc[idx, "status"], odds)
    return out, int(todo.sum())


AVERAGE = "Market average"
AVERAGE_REPLACED = "average_replaced"
AVERAGE_REMOVED = "average_removed"


def replace_average_prices(picks: pd.DataFrame, prices: pd.DataFrame, push: pd.Series,
                           sources=REPAIRED_SOURCES) -> tuple[pd.DataFrame, int, int]:
    """Early v3 predictions were priced with Football-Data's market AVERAGE — real market data, but not a
    price any one bookmaker offered, so it cannot stand for a bet. Such a price is replaced by the real
    bookmaker quote stored for the same match, market and selection before kick-off (the same choice as
    for archived prices: provider priority, then the latest quote); without one the prediction keeps its
    selection and result but is no longer a bet. The average stays in ``replaced_average_odds``."""
    out = picks.copy()
    for col in ("odds_origin", "replaced_average_odds"):
        if col not in out:
            out[col] = None if col == "odds_origin" else np.nan
    todo = (out["source"].isin(sources) & (out["bookmaker"] == AVERAGE)).to_numpy()
    if not todo.any():
        return out, 0, 0
    idx = out.index[todo]
    out.loc[idx, "replaced_average_odds"] = out.loc[idx, "odds"].to_numpy(float)
    quotes = prices.set_index(["match_id", "market", "selection"]) if not prices.empty else None
    keys = pd.MultiIndex.from_frame(out.loc[idx, ["match_id", "market", "selection"]])
    found = keys.isin(quotes.index) if quotes is not None else np.zeros(len(idx), bool)
    have, lack = idx[found], idx[~found]
    if len(have):
        q = quotes.loc[keys[found]]
        odds = q["odds"].to_numpy(float)
        p_push = push.reindex(keys[found]).fillna(0.0).to_numpy(float) if len(push) else np.zeros(len(have))
        out.loc[have, "odds"] = odds
        out.loc[have, "bookmaker"] = q["bookmaker"].to_numpy()
        out.loc[have, "provider"] = q["provider"].to_numpy()
        out.loc[have, "real_odds"] = True
        out.loc[have, "odds_origin"] = AVERAGE_REPLACED
        if "odds_priced_at" in out:
            out.loc[have, "odds_priced_at"] = pd.to_datetime(q["collected_at"]).to_numpy()
        out.loc[have, "ev"] = out.loc[have, "probability"].to_numpy(float) * odds - (1 - p_push)
        out.loc[have, "profit"] = _profit(out.loc[have, "status"], odds)
    if len(lack):
        out.loc[lack, ["odds", "ev", "profit"]] = np.nan
        out.loc[lack, ["bookmaker", "provider"]] = None
        out.loc[lack, "real_odds"] = False
        out.loc[lack, "odds_origin"] = AVERAGE_REMOVED
        if "odds_priced_at" in out:
            out.loc[lack, "odds_priced_at"] = pd.NaT
    return out, len(have), len(lack)


def _selections(rows: pd.DataFrame) -> list[PricedSelection]:
    out = []
    for r in rows.itertuples(index=False):
        odds = None if pd.isna(r.odds) else float(r.odds)
        push = 0.0 if pd.isna(getattr(r, "push_probability", np.nan)) else float(r.push_probability)
        out.append(PricedSelection(r.market, r.selection, float(r.probability), odds, r.bookmaker, r.provider,
                                   push, getattr(r, "priced_at", None)))
    return out


def restore_missing_risk(picks: pd.DataFrame, markets: pd.DataFrame, now: pd.Timestamp | None = None,
                         sources=REPAIRED_SOURCES) -> tuple[pd.DataFrame, int]:
    """Adds one Risk to every stored match of ``sources`` that has a Main but no Risk."""
    if picks.empty or markets.empty:
        return picks, 0
    now = now or pd.Timestamp.utcnow().tz_localize(None)
    scoped = picks.loc[picks["source"].isin(sources)]
    mains = scoped.loc[scoped["role"] == "main"]
    have_risk = set(scoped.loc[scoped["role"] == "risk", "match_id"])
    todo = mains.loc[~mains["match_id"].isin(have_risk)]
    if todo.empty:
        return picks, 0
    by_match = {m: g for m, g in markets.loc[markets["match_id"].isin(todo["match_id"])].groupby("match_id")}
    rows = []
    for main in todo.itertuples(index=False):
        sels = by_match.get(main.match_id)
        if sels is None:
            continue
        priced = _selections(sels)
        main_sel = next((s for s in priced if s.market == main.market and s.selection == main.selection), None)
        if main_sel is None:                       # the Main's own selection must be known to stay apart from it
            continue
        risk = risk_prediction(priced, main_sel)
        if risk is None:
            continue
        s = risk.pick
        row = main._asdict()
        row.update({"pick_id": f"{main.match_id}|risk|{s.market}|{s.selection}", "role": "risk", "kind": risk.kind,
                    "market": s.market, "group": s.group, "selection": s.selection, "probability": s.probability,
                    "odds": s.odds if s.has_odds else np.nan, "bookmaker": s.bookmaker if s.has_odds else None,
                    "provider": s.provider if s.has_odds else None, "real_odds": s.has_odds,
                    "ev": s.ev if s.has_odds else np.nan, "status": "UPCOMING", "profit": np.nan,
                    "settled_at": pd.NaT, "origin": RECONSTRUCTED, "reconstructed_at": now,
                    "odds_origin": None})
        if "odds_priced_at" in row:
            row["odds_priced_at"] = pd.Timestamp(s.priced_at) if s.has_odds and s.priced_at is not None else pd.NaT
        rows.append(row)
    if not rows:
        return picks, 0
    out = pd.concat([picks, pd.DataFrame(rows)], ignore_index=True)
    return out, len(rows)


def repair_history(features: pd.DataFrame, now: pd.Timestamp | None = None, write: bool = True,
                   out: dict | None = None) -> dict:
    """The repairs on the stored tables; restored Risk predictions of finished matches are settled.
    With ``write=False`` nothing is stored; ``out["picks"]`` then receives the repaired table (audits)."""
    picks = store.load_picks()
    if picks.empty:
        return {"risk_restored": 0, "prices_filled": 0}
    markets = store.load_markets()
    picks, n_risk = restore_missing_risk(picks, markets, now)
    scoped = picks.loc[picks["source"].isin(REPAIRED_SOURCES)]
    prices = archive_prices(scoped, features.loc[features["match_id"].isin(scoped["match_id"])])
    push = (markets.loc[markets["match_id"].isin(scoped["match_id"])]
            .drop_duplicates(["match_id", "market", "selection"], keep="last")
            .set_index(["match_id", "market", "selection"])["push_probability"]) if not markets.empty else pd.Series()
    picks, n_odds = fill_archive_prices(picks, prices, push)
    averaged = picks.loc[picks["source"].isin(REPAIRED_SOURCES) & (picks["bookmaker"] == AVERAGE)]
    real = pd.DataFrame()
    if not averaged.empty:
        api = load_api_odds()
        api = api.loc[api["match_id"].isin(averaged["match_id"])] if not api.empty else api
        real = best_prices(pre_match_only(api, features.loc[features["match_id"].isin(averaged["match_id"])])) \
            if not api.empty else api
    picks, n_replaced, n_removed = replace_average_prices(picks, real, push)
    picks, n_cleared = clear_refunds_without_price(picks)
    before = picks["odds_timing"].copy() if "odds_timing" in picks else None
    picks, timing = label_prices(picks)
    relabelled = before is None or not before.fillna("").equals(picks["odds_timing"].fillna(""))
    summary = {"risk_restored": n_risk, "prices_filled": n_odds, "average_prices_replaced": n_replaced,
               "average_prices_removed": n_removed, "refund_profit_cleared": n_cleared, "price_timing": timing}
    if out is not None:
        out["picks"] = picks
    if write and (n_risk or n_odds or n_replaced or n_removed or n_cleared or relabelled):
        predictions_storage.write_parquet(store.PICKS, picks)
        if n_risk:
            store.settle(features)                     # restored Risk of finished matches gets its result
    log.info("history repair: %s", summary)
    return summary
