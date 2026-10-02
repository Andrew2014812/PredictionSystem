"""Tune, evaluate and retrain the four XGBoost models.

    python scripts/retrain_models.py               # Hyperopt budgets from config
    python scripts/retrain_models.py --quick       # 5 trials per model (smoke test)

Run it when a season has finished (new completed season = new training
data). It rebuilds features first, so it always trains on the latest data.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.features.store import rebuild_features  # noqa: E402
from src.modelling.pipeline import train_all  # noqa: E402
from src.prediction.pipeline import generate_live_predictions, settle_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="5 Hyperopt trials per model")
    args = parser.parse_args()
    evals = {k: 5 for k in config.HYPEROPT_EVALS} if args.quick else None

    features = rebuild_features()
    versions = train_all(features, evals)
    print("trained:", versions)
    print("live predictions:", generate_live_predictions(features))
    settle_predictions(features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    main()
