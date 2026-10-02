"""Market probabilities derived from the model distributions, and settlement.

Every selection is identified by ``(market, selection)``:

    1X2              H | D | A
    DOUBLE_CHANCE    1X | X2 | 12
    TOTAL_<line>     OVER | UNDER          line in TOTAL_GOAL_LINES
    BTTS             YES | NO
    EXACT_SCORE      "<home>-<away>"       e.g. "2-1"
    HANDICAP_<line>  H | D | A             European 3-way handicap, home perspective
    CORNERS_<line>   OVER | UNDER          line in CORNER_LINES (or any line)

European handicap ``-1`` adds -1 to the home goals: "H(-1)" wins when the
home team wins by 2+, "D(-1)" when it wins by exactly 1, "A(+1)" otherwise.
Three outcomes always cover every score, so there is no push / refund.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import config
from .distributions import outcome_masks, prob_over

GROUPS = {
    "1X2": "1X2", "DOUBLE_CHANCE": "DOUBLE_CHANCE", "BTTS": "BTTS", "EXACT_SCORE": "EXACT_SCORE",
}


def market_group(market: str) -> str:
    """'TOTAL_2.5' -> 'TOTAL', 'HANDICAP_-1' -> 'HANDICAP', '1X2' -> '1X2'."""
    return GROUPS.get(market, market.split("_")[0])


def market_line(market: str) -> float | None:
    parts = market.rsplit("_", 1)
    try:
        return float(parts[1]) if len(parts) == 2 else None
    except ValueError:
        return None


@dataclass
class Selection:
    market: str
    selection: str
    probability: float

    @property
    def group(self) -> str:
        return market_group(self.market)

    def to_dict(self) -> dict:
        return {**asdict(self), "group": self.group}


def goal_markets(matrix: np.ndarray, top_scores: int = config.EXACT_SCORE_TOP_N) -> list[Selection]:
    """1X2, double chance, totals, BTTS, handicap and exact score from one score matrix."""
    size = matrix.shape[0]
    home, draw, away = outcome_masks(size)
    ph, pd_, pa = matrix[home].sum(), matrix[draw].sum(), matrix[away].sum()
    i, j = np.indices((size, size))
    total = i + j
    out = [
        Selection("1X2", "H", ph), Selection("1X2", "D", pd_), Selection("1X2", "A", pa),
        Selection("DOUBLE_CHANCE", "1X", ph + pd_), Selection("DOUBLE_CHANCE", "X2", pd_ + pa),
        Selection("DOUBLE_CHANCE", "12", ph + pa),
    ]
    for line in config.TOTAL_GOAL_LINES:
        over = matrix[total > line].sum()
        out += [Selection(f"TOTAL_{line}", "OVER", over), Selection(f"TOTAL_{line}", "UNDER", 1 - over)]
    btts = matrix[(i > 0) & (j > 0)].sum()
    out += [Selection("BTTS", "YES", btts), Selection("BTTS", "NO", 1 - btts)]
    for line in config.HANDICAP_LINES:
        margin = i + line - j
        market = f"HANDICAP_{line:+d}"
        out += [Selection(market, "H", matrix[margin > 0].sum()),
                Selection(market, "D", matrix[margin == 0].sum()),
                Selection(market, "A", matrix[margin < 0].sum())]
    flat = np.argsort(matrix, axis=None)[::-1][:top_scores]
    for idx in flat:
        h, a = np.unravel_index(idx, matrix.shape)
        out.append(Selection("EXACT_SCORE", f"{h}-{a}", matrix[h, a]))
    return [Selection(s.market, s.selection, float(s.probability)) for s in out]


def exact_score_probability(matrix: np.ndarray, score: str) -> float:
    h, a = (int(x) for x in score.split("-"))
    if h >= matrix.shape[0] or a >= matrix.shape[1]:
        return 0.0
    return float(matrix[h, a])


def corner_markets(pmf: np.ndarray, lines=config.CORNER_LINES) -> list[Selection]:
    out = []
    for line in lines:
        over = prob_over(pmf, line)
        out += [Selection(f"CORNERS_{line}", "OVER", over), Selection(f"CORNERS_{line}", "UNDER", 1 - over)]
    return out


def expected_corners(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(len(pmf)), pmf))


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def selection_won(market: str, selection: str, home_goals: float, away_goals: float,
                  corners: float | None = None) -> bool | None:
    """True / False when the selection is decided by the result, None if unknown."""
    if home_goals is None or away_goals is None or home_goals != home_goals or away_goals != away_goals:
        return None
    hg, ag = int(home_goals), int(away_goals)
    group = market_group(market)
    line = market_line(market)
    outcome = "H" if hg > ag else ("A" if hg < ag else "D")
    if group == "1X2":
        return selection == outcome
    if group == "DOUBLE_CHANCE":
        return outcome in {"1X": "HD", "X2": "DA", "12": "HA"}[selection]
    if group == "TOTAL":
        return (hg + ag > line) == (selection == "OVER")
    if group == "BTTS":
        return (hg > 0 and ag > 0) == (selection == "YES")
    if group == "EXACT_SCORE":
        return selection == f"{hg}-{ag}"
    if group == "HANDICAP":
        margin = hg + line - ag
        return selection == ("H" if margin > 0 else ("A" if margin < 0 else "D"))
    if group == "CORNERS":
        if corners is None or corners != corners:
            return None
        return (corners > line) == (selection == "OVER")
    raise ValueError(f"unknown market {market}")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
MARKET_TITLES = {
    "1X2": "Match result", "DOUBLE_CHANCE": "Double chance", "TOTAL": "Total goals",
    "BTTS": "Both teams to score", "EXACT_SCORE": "Exact score", "HANDICAP": "Handicap",
    "CORNERS": "Total corners",
}


def market_title(market: str) -> str:
    group = market_group(market)
    line = market_line(market)
    title = MARKET_TITLES.get(group, market)
    if group in ("TOTAL", "CORNERS") and line is not None:
        return f"{title} {line:g}"
    if group == "HANDICAP" and line is not None:
        return f"{title} {int(line):+d}"
    return title


def selection_label(market: str, selection: str, home: str = "Home", away: str = "Away") -> str:
    group = market_group(market)
    line = market_line(market)
    if group == "1X2":
        return {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}[selection]
    if group == "DOUBLE_CHANCE":
        return {"1X": f"{home} or draw", "X2": f"{away} or draw", "12": f"{home} or {away}"}[selection]
    if group == "TOTAL":
        return f"{selection.title()} {line:g} goals"
    if group == "BTTS":
        return f"Both teams to score: {selection.title()}"
    if group == "EXACT_SCORE":
        return f"Exact score {selection.replace('-', ':')}"
    if group == "HANDICAP":
        hcp = int(line)
        return {"H": f"{home} ({hcp:+d})", "D": f"Draw ({hcp:+d})", "A": f"{away} ({-hcp:+d})"}[selection]
    if group == "CORNERS":
        return f"{selection.title()} {line:g} corners"
    return f"{market} {selection}"
