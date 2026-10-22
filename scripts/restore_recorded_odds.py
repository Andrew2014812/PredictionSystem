"""Fill the missing prices of recorded (pre-match) predictions from the OddsPapi price archive.

A recorded prediction keeps its pick, probability, model version and generation time; only an empty
price is filled, with the last OddsPapi quote strictly before kick-off (the archive downloaded by
``scripts/oddspapi_history.py``). Such a row gets ``odds_origin = "archive"`` so the interface can say
the price was added later; EV and — for a settled prediction — profit are recomputed from it. A
prediction that already has a price is never changed. Safe to run more than once.

    python scripts/restore_recorded_odds.py            # dry run: how many prices would be filled
    python scripts/restore_recorded_odds.py --go
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.store import load_features  # noqa: E402
from src.markets.markets import PUSH, WON  # noqa: E402
from src.prediction import store  # noqa: E402
from src.providers import load_api_odds, pre_match_only  # noqa: E402
from src.providers.collect import PROVIDER_ODP  # noqa: E402
from src.providers.schema import best_prices  # noqa: E402
from src.storage import predictions_storage  # noqa: E402

ARCHIVE = "archive"


def archive_prices(picks: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Last pre-kick-off OddsPapi quote per (match, market, selection) of the given picks."""
    api = load_api_odds()
    if api.empty:
        return api
    api = api.loc[(api["provider"] == PROVIDER_ODP) & api["match_id"].isin(picks["match_id"])]
    return best_prices(pre_match_only(api, matches))


def restore(picks: pd.DataFrame, prices: pd.DataFrame, push: pd.Series) -> tuple[pd.DataFrame, int]:
    out = picks.copy()
    if "odds_origin" not in out:
        out["odds_origin"] = None
    if prices.empty:
        return out, 0
    quotes = prices.set_index(["match_id", "market", "selection"])
    keys = pd.MultiIndex.from_frame(out[["match_id", "market", "selection"]])
    found = keys.isin(quotes.index)
    todo = (out["source"] == "live") & out["odds"].isna() & found
    if not todo.any():
        return out, 0
    q = quotes.loc[keys[todo.to_numpy()]]
    idx = out.index[todo]
    odds = q["odds"].to_numpy(float)
    p_push = push.reindex(keys[todo.to_numpy()]).fillna(0.0).to_numpy(float)
    out.loc[idx, "odds"] = odds
    out.loc[idx, "bookmaker"] = q["bookmaker"].to_numpy()
    out.loc[idx, "provider"] = PROVIDER_ODP
    out.loc[idx, "real_odds"] = True
    out.loc[idx, "odds_origin"] = ARCHIVE
    out.loc[idx, "ev"] = out.loc[idx, "probability"].to_numpy(float) * odds - (1 - p_push)
    status = out.loc[idx, "status"]
    profit = np.where(status == WON, odds - 1, np.where(status == "LOST", -1.0,
                                                        np.where(status.isin([PUSH, "VOID"]), 0.0, np.nan)))
    out.loc[idx, "profit"] = profit
    return out, int(todo.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--go", action="store_true", help="write the filled prices (otherwise only count)")
    args = parser.parse_args()
    picks = store.load_picks()
    live = picks.loc[picks["source"] == "live"]
    features = load_features()
    prices = archive_prices(live, features.loc[features["match_id"].isin(live["match_id"])])
    markets = store.load_markets()
    push = (markets.loc[markets["match_id"].isin(live["match_id"])].drop_duplicates(["match_id", "market", "selection"], keep="last")
            .set_index(["match_id", "market", "selection"])["push_probability"]) if not markets.empty else pd.Series()
    restored, n = restore(picks, prices, push)
    missing = int((live["odds"].isna()).sum())
    print(f"recorded predictions without a price: {missing}; filled from the archive: {n}")
    by = restored.loc[restored["odds_origin"] == ARCHIVE].groupby("role").size().to_dict()
    print("by type:", by)
    if args.go and n:
        predictions_storage.write_parquet(store.PICKS, restored)
        print("saved", store.PICKS)
    elif not args.go:
        print("dry run: nothing written (add --go)")


if __name__ == "__main__":
    main()
