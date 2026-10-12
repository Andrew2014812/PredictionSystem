"""Re-apply the current decision rules without retraining.

Regenerates the evaluation history (backtest test season and backfill current
season) and the predictions of matches that have not kicked off, then settles.
Recorded pre-match predictions of started matches are never touched.

    python scripts/regenerate_predictions.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.store import load_features  # noqa: E402
from src.modelling.pipeline import season_split  # noqa: E402
from src.modelling.registry import list_versions, load_bundles  # noqa: E402
from src.prediction.engine import predict_matches  # noqa: E402
from src.prediction.pipeline import generate_live_predictions, settle_predictions  # noqa: E402
from src.prediction.store import drop_source, save_batch  # noqa: E402
from src.providers import real_odds_lookup  # noqa: E402


def main() -> None:
    features = load_features()
    split = season_split()
    backtests = sorted(v for v, m in list_versions().items() if m.get("role") == "backtest")
    if backtests:
        version = backtests[-1]
        rows = features.loc[features["season"].isin(split["test"])]
        batch = predict_matches(rows, load_bundles(version), real_odds_lookup(rows), version, "backtest")
        save_batch(batch, replace_source="backtest")
        print("backtest predictions:", len(batch.snapshots))
    print("backfill predictions dropped:", drop_source("backfill"))
    print("backfill + upcoming predictions:", generate_live_predictions(features))
    settle_predictions(features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
