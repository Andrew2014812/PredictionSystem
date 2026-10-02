"""Central configuration of the prediction system.

Everything that may need tuning (paths, seasons, model settings, market
lines, recommendation thresholds) lives here so the rest of the code never
hard-codes these values.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "local_data"

RESULTS_DIR = DATA_DIR / "results"          # raw Football-Data season files
FIXTURES_DIR = DATA_DIR / "fixtures"        # raw Football-Data fixtures file
PROCESSED_DIR = DATA_DIR / "processed"      # cleaned matches + features
MODELS_DIR = DATA_DIR / "models"            # trained models + metadata.json
PREDICTIONS_DIR = DATA_DIR / "predictions"  # prediction snapshots, picks, SHAP

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
FOOTBALL_DATA_URL = "https://www.football-data.co.uk"
FOOTBALL_DATA_RESULTS_PATH = "mmz4281"      # /mmz4281/2526/E0.csv
FOOTBALL_DATA_FIXTURES_FILE = "fixtures.csv"
HTTP_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
# A season is identified by its starting year (2025 -> season 2025/26,
# code "2526" in Football-Data URLs).
FIRST_SEASON = 2021


def current_season(today: date | None = None) -> int:
    """Season that is in progress on ``today`` (new seasons start in July)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(season: int) -> str:
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def season_of(day) -> int:
    """Season a match date belongs to."""
    return day.year if day.month >= 7 else day.year - 1


# Temporal evaluation scheme (relative to the current season):
#   TRAIN       all completed seasons before VALIDATION
#   VALIDATION  current - 2   -> used by Hyperopt only
#   TEST        current - 1   -> untouched until the final evaluation
#   CURRENT     current       -> live out-of-sample predictions
VALIDATION_OFFSET = 2
TEST_OFFSET = 1

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FORM_WINDOWS = (5, 10, 15)
H2H_WINDOW = 5
LEAGUE_PRIOR_WINDOW = 300     # matches used for running league averages
REST_DAYS_CAP = 30            # long breaks (season start) are capped
# Odds are NOT model inputs. Flip to True to append market-implied
# probabilities to every model's feature set (see features/sets.py).
USE_ODDS_FEATURES = False

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
HYPEROPT_EVALS = {
    "result": 30,
    "home_goals": 20,
    "away_goals": 20,
    "corners": 20,
}
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 1500
# A league needs this share of matches with corner statistics before the
# corners market is offered for it.
MIN_STAT_COVERAGE = 0.8
MAX_GOALS = 10                 # score matrix is (MAX_GOALS + 1) x (MAX_GOALS + 1)

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
TOTAL_GOAL_LINES = (1.5, 2.5, 3.5)
CORNER_LINES = (8.5, 9.5, 10.5)
DEFAULT_CORNER_LINE = 9.5
HANDICAP_LINES = (-2, -1, 1, 2)   # European (3-way) handicap, home perspective
EXACT_SCORE_TOP_N = 5

# Margin applied to simulated odds (a "naive bookmaker" that prices every
# selection from the league's historical frequency of that event).
SIMULATED_MARGIN = 0.06

# Markets whose single most likely selection is stored in the prediction
# history for every match (role = "market").
HISTORY_MARKETS = ("1X2", "TOTAL_2.5", "BTTS", "CORNERS_9.5", "EXACT_SCORE")

# ---------------------------------------------------------------------------
# Recommendation layer (main / risk prediction)
# ---------------------------------------------------------------------------
MAIN_MIN_PROBABILITY = 0.45
MAIN_MIN_ODDS = 1.30
MAIN_MAX_ODDS = 3.00
MAIN_MIN_EV = 0.03
RISK_MIN_ODDS = 3.00
RISK_MAX_ODDS = 8.00
RISK_MIN_PROBABILITY = 0.22
RISK_MIN_EV = 0.05

# How much each market's model output is trusted when ranking candidates.
MARKET_RELIABILITY = {
    "1X2": 1.00,
    "DOUBLE_CHANCE": 1.00,
    "TOTAL": 1.00,
    "BTTS": 0.95,
    "HANDICAP": 0.90,
    "CORNERS": 0.85,
    "EXACT_SCORE": 0.50,
}
# How much an odds value is trusted, by where it came from.
ODDS_SOURCE_RELIABILITY = {
    "market": 1.00,
    "derived": 0.90,
    "simulated": 0.75,
}

# Predictions are settled as VOID when no result appears this long after
# the scheduled date (postponed / abandoned match).
VOID_AFTER_DAYS = 21
