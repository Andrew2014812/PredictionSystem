"""Carry a downloaded OddsPapi price archive over to newer data.

The ``history`` workflow downloads for hours; meanwhile the daily run may commit new data to main.
Before the recorded prices are filled and the backtest regenerated, the job moves to the newest main
and merges in only what the download produced:

    python scripts/merge_history_archive.py save <dir>    # after the download: keep the archive aside
    git reset --hard origin/main                          # the newest data
    python scripts/merge_history_archive.py merge <dir>   # add the archive to it

Prices are added (never replaced): the union of both price tables, a quote recorded twice kept once.
Ledger requests and learned team ids are added where missing; the history progress and coverage files
are the download's own.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()                  # run from the repository root (the workflow may run a copy of this file)
sys.path.insert(0, str(ROOT))

from src.providers.schema import ODDS_COLUMNS  # noqa: E402

FILES = {
    "odds": ROOT / "local_data/odds/odds.parquet",
    "ledger": ROOT / "local_data/api/oddspapi/ledger.json",
    "ids": ROOT / "local_data/teams/provider_ids.json",
    "progress": ROOT / "local_data/api/oddspapi/history_progress.json",
    "coverage": ROOT / "local_data/api/oddspapi/history_coverage.json",
}
KEY = ["match_id", "provider", "bookmaker", "market", "selection", "collected_at"]


def save(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        if path.exists():
            shutil.copy2(path, target / path.name)
            print("saved", path.name)


def _merge_odds(saved: Path) -> None:
    if not saved.exists():
        return
    new = pd.read_parquet(saved)
    path = FILES["odds"]
    old = pd.read_parquet(path) if path.exists() else new.iloc[0:0]
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values("collected_at").drop_duplicates(KEY, keep="last")
    merged[[c for c in ODDS_COLUMNS if c in merged]].to_parquet(path, index=False)
    print(f"prices: {len(old)} before, {len(merged)} after the merge")


def _merge_json(saved: Path, path: Path, nested: str | None) -> None:
    if not saved.exists():
        return
    theirs = json.loads(saved.read_text(encoding="utf-8"))
    ours = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if nested is None:                        # provider -> {id: team}: add learned ids, never re-assign
        for provider, ids in theirs.items():
            mine = ours.setdefault(provider, {})
            for pid, team in ids.items():
                mine.setdefault(pid, team)
    else:                                     # ledger: add the requests the download made
        mine = ours.setdefault(nested, {})
        for k, v in theirs.get(nested, {}).items():
            mine.setdefault(k, v)
        for k, v in theirs.items():
            if k != nested and k not in ours:
                ours[k] = v
    path.write_text(json.dumps(ours, indent=1, ensure_ascii=False), encoding="utf-8")
    print("merged", path.name)


def merge(source: Path) -> None:
    _merge_odds(source / FILES["odds"].name)
    _merge_json(source / FILES["ledger"].name, FILES["ledger"], "requests")
    _merge_json(source / FILES["ids"].name, FILES["ids"], None)
    for name in ("progress", "coverage"):
        saved = source / FILES[name].name
        if saved.exists():
            shutil.copy2(saved, FILES[name])
            print("copied", saved.name)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("save", "merge"):
        sys.exit(__doc__)
    (save if sys.argv[1] == "save" else merge)(Path(sys.argv[2]))
