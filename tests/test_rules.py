"""Project-level rules: one settlement per run, full league support, no synthetic odds in the UI."""
import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from src.leagues import LEAGUES, enabled_leagues

ROOT = Path(__file__).resolve().parents[1]


def _calls(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name)


@pytest.mark.parametrize("script", ["run_pipeline.py", "retrain_models.py"])
def test_settlement_runs_exactly_once(script):
    assert _calls(ROOT / "scripts" / script, "settle_predictions") == 1


def test_disabled_leagues_explain_why():
    for lg in LEAGUES:
        if not lg.enabled:
            assert lg.disabled_reason
    assert not next(lg for lg in LEAGUES if lg.code == "EC").enabled


def test_enabled_leagues_have_full_statistics():
    path = ROOT / "local_data" / "processed" / "features.parquet"
    if not path.exists():
        pytest.skip("no processed data")
    f = pd.read_parquet(path, columns=["league", "season", "played", "home_shots", "home_sot", "home_corners",
                                       "home_yellows"])
    recent = f.loc[f["played"] & (f["season"] >= f["season"].max() - 3)]
    for lg in enabled_leagues():
        rows = recent.loc[recent["league"] == lg.code]
        assert len(rows) > 300, lg.code
        for col in ("home_shots", "home_sot", "home_corners", "home_yellows"):
            assert rows[col].notna().mean() >= 0.9, (lg.code, col)
        assert lg.calendar in ("european", "calendar_year")


def test_no_synthetic_odds_wording_in_the_interface():
    banned = re.compile(r"\b(BOOK|DERIVED|SIM)\b|Fair odds|Estimated odds|Simulated odds|odds_source|"
                        r"Not a value bet|No selection passed the minimum EV")
    for path in [*sorted((ROOT / "views").glob("*.py")), *sorted((ROOT / "src" / "ui").glob("*.py"))]:
        text = path.read_text(encoding="utf-8")
        assert not banned.search(text), (path.name, banned.search(text).group())


def test_odds_layer_has_no_synthetic_prices():
    from src import config
    from src.markets import odds
    assert not hasattr(config, "SIMULATED_MARGIN") and not hasattr(config, "ODDS_SOURCE_RELIABILITY")
    assert not hasattr(odds, "fit_market_score_matrix")              # no prices derived from other markets


def test_history_shows_completed_predictions_by_default():
    import importlib.util
    spec = importlib.util.spec_from_file_location("history_view", ROOT / "views" / "history.py")
    source = (ROOT / "views" / "history.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "default_statuses")
    namespace: dict = {}
    exec(compile(ast.Module([fn], type_ignores=[]), "history", "exec"), namespace)
    assert namespace["default_statuses"](False) == ["WON", "LOST"]
    assert "UPCOMING" in namespace["default_statuses"](True)
    assert namespace["default_statuses"](False, ["VOID"]) == ["VOID"]
    assert spec is not None
