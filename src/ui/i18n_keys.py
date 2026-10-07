"""Collect every English text the interface translates (used by tests and tooling)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "app.py", *sorted((ROOT / "views").glob("*.py")), *sorted((ROOT / "src" / "ui").glob("*.py"))]
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# texts passed to t() through variables
EXTRA = {
    "Won", "Lost", "Void", "Upcoming", "Live (before kick-off)", "Backfill (current season)",
    "Backtest (test season)", "League position", "Points", "Points / game", "Points / game (last 5)",
    "Goals / game", "Conceded / game", "Shots / game", "Shots on target / game", "Corners / game",
    "Yellow cards / game", "Home", "Draw", "Away",
}


def literal_keys() -> set[str]:
    """String literals passed directly to t(...)."""
    keys = set()
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
    return keys


def registry_keys() -> set[str]:
    """Texts that reach t() through lookup tables."""
    from ..analytics.performance import DATE_PRESETS, PERIODS
    from ..leagues import LEAGUES
    from .labels import MARKET_GROUP_TITLES
    keys = set(DATE_PRESETS) | set(PERIODS) | set(WEEKDAYS) | set(MARKET_GROUP_TITLES.values())
    keys |= {lg.country for lg in LEAGUES}
    for module, names in (("views.history", ("ROLES",)), ("views.analytics", ("PREDICTION_SETS",)),
                          ("views.teams", ("PERIODS",)), ("views.model_analysis",
                                                            ("METRIC_HELP", "GROUP_NAMES", "TASK_TITLES"))):
        tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(tg, "id", None) in names for tg in node.targets):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        value = sub.value
                        is_code = value.replace("_", "").isupper() and " " not in value
                        if not is_code and (" " in value or value[:1].isupper()):
                            keys.add(value)
    return keys | EXTRA


def all_keys() -> set[str]:
    return literal_keys() | registry_keys()
