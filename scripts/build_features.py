"""Rebuild the cleaned match table and pre-match features.

    python scripts/build_features.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.store import rebuild_features  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    features = rebuild_features()
    print(f"features: {len(features)} matches, {int((~features['played']).sum())} upcoming")
