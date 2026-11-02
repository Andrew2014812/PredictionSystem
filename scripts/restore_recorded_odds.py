"""Complete stored predictions without changing them: archived prices and missing Risk predictions.

* A prediction stored without a price gets the last real OddsPapi quote strictly before kick-off
  (``odds_origin = "archive"``); its selection, probability, model version and generation time stay.
* A match recorded with a Main but no Risk gets the Risk chosen by the current rule from the
  probabilities and pre-match prices stored for that version (``origin = "risk_reconstructed"``).

The daily pipeline runs the same repair; this script runs it on demand. Safe to run more than once.

    python scripts/restore_recorded_odds.py            # dry run: what would be filled
    python scripts/restore_recorded_odds.py --go
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.store import load_features  # noqa: E402
from src.prediction.history_repair import repair_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--go", action="store_true", help="write the result (otherwise only count)")
    args = parser.parse_args()
    summary = repair_history(load_features(), write=args.go)
    print(f"Risk predictions restored: {summary['risk_restored']}; prices filled from the archive: "
          f"{summary['prices_filled']}" + ("" if args.go else " (dry run: nothing written, add --go)"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
