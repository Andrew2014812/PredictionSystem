"""Reclassify "live" predictions that were generated after kick-off.

Before v4 the pipeline chose upcoming matches by DATE, so a run during a match
day could create or refresh a "live" prediction for a match that had already
started. Those are not genuine pre-match predictions: this moves them to
``backfill`` (after-the-fact evaluation) in snapshots and picks. Nothing is
deleted and the script is idempotent.

``generated_at`` is read as UTC (all scheduled runs were GitHub Actions, UTC);
the rare local run was on a clock ahead of UTC, so the check errs towards
reclassifying. ``kickoff`` is UK local time.

    python scripts/migrate_prediction_history.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.prediction.store import PICKS, SNAPSHOTS, load_picks, load_snapshots  # noqa: E402
from src.storage import predictions_storage  # noqa: E402


def late_live_ids(snapshots: pd.DataFrame) -> set[str]:
    live = snapshots.loc[snapshots["source"] == "live"]
    if live.empty:
        return set()
    generated_uk = (pd.to_datetime(live["generated_at"]).dt.tz_localize("UTC")
                    .dt.tz_convert("Europe/London").dt.tz_localize(None))
    kickoff = pd.to_datetime(live["kickoff"]).fillna(pd.to_datetime(live["date"]))
    return set(live.loc[generated_uk >= kickoff, "match_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    snapshots, picks = load_snapshots(), load_picks()
    ids = late_live_ids(snapshots)
    print(f"live predictions generated after kick-off: {len(ids)} of {(snapshots['source'] == 'live').sum()}")
    if not ids or args.dry_run:
        return
    for name, df in ((SNAPSHOTS, snapshots), (PICKS, picks)):
        df = df.copy()
        df.loc[df["match_id"].isin(ids) & (df["source"] == "live"), "source"] = "backfill"
        predictions_storage.write_parquet(name, df)
    print("reclassified as backfill")


if __name__ == "__main__":
    main()
