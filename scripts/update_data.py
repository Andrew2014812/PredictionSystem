"""Download fresh results and fixtures from Football-Data.

    python scripts/update_data.py                 # current season + fixtures
    python scripts/update_data.py --all-seasons   # every season since FIRST_SEASON
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.download import download_fixtures, download_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--leagues", nargs="*", help="league codes (default: all enabled)")
    args = parser.parse_args()

    current = config.current_season()
    seasons = list(range(config.FIRST_SEASON, current + 1)) if args.all_seasons else [current]
    saved = download_results(seasons, args.leagues)
    print(f"results: {len(saved)} files updated")
    print("fixtures:", "updated" if download_fixtures() else "not available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    main()
