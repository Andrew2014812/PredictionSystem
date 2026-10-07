"""One-off migration of the prediction history to the real-odds-only scheme.

Before v3 the history stored three kinds of prices: "market" (real
Football-Data averages), "derived" and "simulated". After migration:

* real prices are kept (bookmaker "Market average", provider football_data);
* derived / simulated prices, their EV and profit are removed — the
  prediction and its WON / LOST result stay, but it no longer counts in
  profit or ROI;
* predictions of leagues that are no longer supported are removed.

Safe to run more than once.

    python scripts/migrate_real_odds.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leagues import league_codes  # noqa: E402
from src.prediction import store  # noqa: E402
from src.storage import predictions_storage  # noqa: E402


def migrate(df):
    if df.empty or "odds_source" not in df.columns:
        return df, 0
    synthetic = df["odds_source"].isin(["derived", "simulated"])
    real = df["odds_source"] == "market"
    for col in ("odds", "ev", "profit"):
        if col in df.columns:
            df.loc[synthetic, col] = np.nan
    df["bookmaker"] = np.where(real, "Market average", None)
    df["provider"] = np.where(real, "football_data", None)
    df["real_odds"] = real
    drop = [c for c in ("odds_source", "fair_odds", "implied_probability") if c in df.columns]
    return df.drop(columns=drop), int(synthetic.sum())


def drop_unsupported_leagues() -> None:
    snapshots = predictions_storage.read_parquet_or_empty(store.SNAPSHOTS)
    if snapshots.empty:
        return
    gone = set(snapshots.loc[~snapshots["league"].isin(league_codes()), "match_id"])
    for name in (store.SNAPSHOTS, store.MARKETS, store.PICKS):
        df = predictions_storage.read_parquet_or_empty(name)
        if not df.empty:
            predictions_storage.write_parquet(name, df.loc[~df["match_id"].isin(gone)])
    print(f"{len(gone)} matches of unsupported leagues removed")


def main() -> None:
    drop_unsupported_leagues()
    for name in (store.PICKS, store.MARKETS):
        df = predictions_storage.read_parquet_or_empty(name)
        df, changed = migrate(df)
        if changed or not df.empty:
            predictions_storage.write_parquet(name, df)
        print(f"{name}: {changed} synthetic prices removed")


if __name__ == "__main__":
    main()
