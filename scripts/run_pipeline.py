"""Daily pipeline — updates data and predictions, never retrains models.

    python scripts/run_pipeline.py             # full update
    python scripts/run_pipeline.py --offline   # use local files only

Order:
1. fresh fixtures / results and real odds from the API providers
   (API-Football, The Odds API — only if keys are configured);
2. current-season Football-Data results and fixtures;
3. rebuild features (Football-Data + API rows, deduplicated);
4. settle stored predictions of finished matches;
5. generate predictions for upcoming matches (and backfill);
6. everything is written to local_data/ (the web app only reads these files).

Settlement runs once, before prediction: it only touches predictions of
matches that are already finished, while prediction only (re)writes
predictions of matches that have not started, so the order between them
does not change the result.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.dataset import build_matches  # noqa: E402
from src.data.download import download_fixtures, download_results  # noqa: E402
from src.features.store import rebuild_features  # noqa: E402
from src.prediction.pipeline import generate_live_predictions, settle_predictions  # noqa: E402
from src.providers.collect import collect  # noqa: E402
from src.storage import LocalStorage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    summary = {}
    if not args.offline:
        summary["providers"] = collect(build_matches())
        download_results([config.current_season()])
        download_fixtures()
    features = rebuild_features()
    picks = settle_predictions(features)
    summary["predicted_matches"] = generate_live_predictions(features)
    summary["picks"] = picks["status"].value_counts().to_dict() if not picks.empty else {}
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    LocalStorage(config.DATA_DIR).write_json("data_status.json", {
        "updated_at": summary["updated_at"],
        "last_result_date": str(features.loc[features["played"], "date"].max().date()),
        "upcoming_matches": int((~features["played"]).sum()),
    })
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
