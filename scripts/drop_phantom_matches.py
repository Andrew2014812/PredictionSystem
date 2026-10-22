"""Remove matches that never existed: a provider team name resolved to the wrong club created them.

A phantom match is removed from the provider fixtures and from every prediction table (it was never a
real match, so its predictions are not a track record). Its prices can be moved to the real match with
``--move-odds`` when the provider fixture was the real one under a wrong team.

    python scripts/drop_phantom_matches.py SC0-20261017-st-johnstone-dundee
    python scripts/drop_phantom_matches.py SP1-20261024-santander-barcelona \
        --move-odds SP1-20261024-santander-barcelona=SP1-20261024-santander-espanol --forget-id oddspapi:2817
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction import store  # noqa: E402
from src.providers.collect import LIVE_FILE, ODDS_FILE, live_storage, odds_storage  # noqa: E402
from src.providers.teams_registry import TeamRegistry  # noqa: E402
from src.storage import predictions_storage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_ids", nargs="+")
    parser.add_argument("--move-odds", nargs="*", default=[], help="phantom_id=real_id")
    parser.add_argument("--forget-id", nargs="*", default=[], help="provider:id of the mis-learned team id")
    args = parser.parse_args()
    gone = set(args.match_ids)
    live = live_storage.read_parquet_or_empty(LIVE_FILE)
    if not live.empty:
        live_storage.write_parquet(LIVE_FILE, live.loc[~live["match_id"].isin(gone)])
        print("provider fixtures removed:", int(live["match_id"].isin(gone).sum()))
    odds = odds_storage.read_parquet_or_empty(ODDS_FILE)
    if not odds.empty:
        moves = dict(m.split("=", 1) for m in args.move_odds)
        moved = odds["match_id"].isin(moves)
        odds.loc[moved, "match_id"] = odds.loc[moved, "match_id"].map(moves)
        dropped = odds["match_id"].isin(gone)
        odds_storage.write_parquet(ODDS_FILE, odds.loc[~dropped])
        print("prices moved:", int(moved.sum()), "prices removed:", int(dropped.sum()))
    for name in (store.SNAPSHOTS, store.MARKETS, store.PICKS, store.VERSIONS):
        df = predictions_storage.read_parquet_or_empty(name)
        if not df.empty:
            predictions_storage.write_parquet(name, df.loc[~df["match_id"].isin(gone)])
            print(f"{name}: {int(df['match_id'].isin(gone).sum())} rows removed")
    if args.forget_id:
        registry = TeamRegistry()
        for item in args.forget_id:
            provider, pid = item.split(":", 1)
            print("forgot", item, "->", registry.data.get(provider, {}).pop(pid, None))
        registry.save()


if __name__ == "__main__":
    main()
