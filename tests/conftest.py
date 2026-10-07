import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import normalise  # noqa: E402


def raw_row(div, day, home, away, hg=None, ag=None, time="15:00", **extra):
    row = {"Div": div, "Date": day, "Time": time, "HomeTeam": home, "AwayTeam": away,
           "FTHG": hg, "FTAG": ag, "FTR": None, "HS": 10, "AS": 8, "HST": 4, "AST": 3,
           "HF": 11, "AF": 12, "HC": 5, "AC": 4, "HY": 1, "AY": 2, "HR": 0, "AR": 0,
           "B365H": 2.0, "B365D": 3.4, "B365A": 3.8, "B365>2.5": 1.9, "B365<2.5": 1.95,
           "AHh": -0.5, "B365AHH": 1.95, "B365AHA": 1.9}
    if hg is None:
        for k in ("FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"):
            row[k] = np.nan
    row.update(extra)
    return row


@pytest.fixture
def make_matches():
    """Build canonical matches from (date, home, away, hg, ag) tuples."""
    def build(rows, div="E0"):
        raw = pd.DataFrame([raw_row(div, d, h, a, hg, ag) for d, h, a, hg, ag in rows])
        return normalise(raw)
    return build


def pytest_addoption(parser):
    parser.addoption("--e2e", action="store_true", help="run browser end-to-end tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--e2e"):
        return
    skip = pytest.mark.skip(reason="browser test: run with --e2e")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)
