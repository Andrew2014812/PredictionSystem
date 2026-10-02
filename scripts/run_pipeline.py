"""Daily pipeline: update data -> features -> predictions -> settlement.

    python scripts/run_pipeline.py             # download + predict + settle
    python scripts/run_pipeline.py --offline   # skip the download step

Does not retrain models (see scripts/retrain_models.py).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.download import download_fixtures, download_results  # noqa: E402
from src.features.store import rebuild_features  # noqa: E402
from src.prediction.pipeline import generate_live_predictions, settle_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        download_results([config.current_season()])
        download_fixtures()
    features = rebuild_features()
    settle_predictions(features)
    print("predicted matches:", generate_live_predictions(features))
    picks = settle_predictions(features)
    if not picks.empty:
        print(picks["status"].value_counts().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
