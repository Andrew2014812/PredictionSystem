"""v5: whole-line handicap rows stored before v5 were European 3-way handicaps.

Before v5 ``HANDICAP_-1`` meant the European 3-way market (home / draw / away
after the handicap). From v5 ``HANDICAP_<line>`` is the Asian handicap (with a
refund on whole lines) and the European market is ``EHANDICAP_<line>``. This
renames the stored whole-line rows of earlier predictions so that they keep
their original meaning and settlement; half lines are identical in both and
are not touched. Idempotent; nothing is deleted.

    python scripts/migrate_v5_handicap.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.prediction.store import MARKETS, PICKS, VERSIONS  # noqa: E402
from src.storage import predictions_storage  # noqa: E402


def whole_line(market: pd.Series) -> pd.Series:
    line = pd.to_numeric(market.str.extract(r"^HANDICAP_([+-]?[\d.]+)$")[0], errors="coerce")
    return line.notna() & (line % 1 == 0)


def migrate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or "market" not in df:
        return df, 0
    mask = whole_line(df["market"])
    if not mask.any():
        return df, 0
    df = df.copy()
    df.loc[mask, "market"] = "E" + df.loc[mask, "market"]
    if "group" in df:
        df.loc[mask, "group"] = "EHANDICAP"
    return df, int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (MARKETS, PICKS, VERSIONS):
        df = predictions_storage.read_parquet_or_empty(name)
        new, n = migrate(df)
        print(f"{name}: {n} whole-line handicap rows -> EHANDICAP")
        if n and not args.dry_run:
            predictions_storage.write_parquet(name, new)


if __name__ == "__main__":
    main()
