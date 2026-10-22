"""Market probabilities derived from the model distributions, and settlement.

Every selection is identified by ``(market, selection)``:

    1X2              H | D | A
    DOUBLE_CHANCE    1X | X2 | 12
    TOTAL_<line>     OVER | UNDER          line in TOTAL_GOAL_LINES
    BTTS             YES | NO
    EXACT_SCORE      "<home>-<away>"       e.g. "2-1"
    HANDICAP_<line>  H | A                 Asian handicap (half and whole lines)
    EHANDICAP_<line> H | D | A             European 3-way handicap (whole lines)
    CORNERS_<line>   OVER | UNDER          line in CORNER_LINES (or any line)

The handicap line is added to the home team's goals (home perspective).

* Asian half line ``-0.5``: "H(-0.5)" wins when the home team wins, "A(+0.5)"
  otherwise — two outcomes, no refund.
* Asian whole line ``-1``: "H(-1)" wins when the home team wins by 2+, is
  refunded (PUSH) when it wins by exactly 1 and loses otherwise; "A(+1)" is
  the mirror image. The selection's ``probability`` is the probability of a
  WIN and ``push_probability`` the probability of a refund.
* European whole line ``-1`` (3-way): "H(-1)" home wins by 2+, "D(-1)" home
  wins by exactly 1, "A(+1)" otherwise — every score decides, no refund. Kept
  as a separate market because its prices and rules differ from the Asian one.

Settlement (:func:`settle_selection`) returns WON, LOST or PUSH; a PUSH returns
the stake (profit 0) and is not a staked bet in ROI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import config
from .distributions import outcome_masks, prob_over

GROUPS = {
    "1X2": "1X2", "DOUBLE_CHANCE": "DOUBLE_CHANCE", "BTTS": "BTTS", "EXACT_SCORE": "EXACT_SCORE",
}
WON, LOST, PUSH = "WON", "LOST", "PUSH"


def market_group(market: str) -> str:
    """'TOTAL_2.5' -> 'TOTAL', 'HANDICAP_-1' -> 'HANDICAP', '1X2' -> '1X2'."""
    return GROUPS.get(market, market.split("_")[0])


def handicap_market(line: float) -> str:
    """Asian handicap market key: 'HANDICAP_-1', 'HANDICAP_+0.5', ..."""
    return f"HANDICAP_{line:+g}"


def european_handicap_market(line: float) -> str:
    """European 3-way handicap market key (whole lines): 'EHANDICAP_-1'."""
    return f"EHANDICAP_{line:+g}"


def is_half_line(line: float) -> bool:
    return abs(line * 2) % 2 == 1


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
    push_probability: float = 0.0       # Asian whole-line handicap: chance of a refund

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
        win_h, push, win_a = matrix[margin > 0].sum(), matrix[margin == 0].sum(), matrix[margin < 0].sum()
        market = handicap_market(line)                       # Asian: refund on an exact whole-line margin
        out += [Selection(market, "H", win_h, push), Selection(market, "A", win_a, push)]
        if not is_half_line(line):                           # European 3-way, kept separately
            em = european_handicap_market(line)
            out += [Selection(em, "H", win_h), Selection(em, "D", push), Selection(em, "A", win_a)]
    flat = np.argsort(matrix, axis=None)[::-1][:top_scores]
    for idx in flat:
        h, a = np.unravel_index(idx, matrix.shape)
        out.append(Selection("EXACT_SCORE", f"{h}-{a}", matrix[h, a]))
    return [Selection(s.market, s.selection, float(s.probability), float(s.push_probability)) for s in out]


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
def settle_selection(market: str, selection: str, home_goals: float, away_goals: float,
                     corners: float | None = None) -> str | None:
    """WON / LOST / PUSH when the result decides the selection, None if unknown."""
    if home_goals is None or away_goals is None or home_goals != home_goals or away_goals != away_goals:
        return None
    hg, ag = int(home_goals), int(away_goals)
    group = market_group(market)
    line = market_line(market)
    outcome = "H" if hg > ag else ("A" if hg < ag else "D")

    def won(flag: bool) -> str:
        return WON if flag else LOST

    if group == "1X2":
        return won(selection == outcome)
    if group == "DOUBLE_CHANCE":
        return won(outcome in {"1X": "HD", "X2": "DA", "12": "HA"}[selection])
    if group == "TOTAL":
        return won((hg + ag > line) == (selection == "OVER"))
    if group == "BTTS":
        return won((hg > 0 and ag > 0) == (selection == "YES"))
    if group == "EXACT_SCORE":
        return won(selection == f"{hg}-{ag}")
    if group == "HANDICAP":                                   # Asian
        if selection not in ("H", "A"):
            raise ValueError(f"{market} has no {selection} selection")
        margin = hg + line - ag
        if margin == 0:
            return PUSH                                       # only possible on whole lines
        return won(selection == ("H" if margin > 0 else "A"))
    if group == "EHANDICAP":                                  # European 3-way
        margin = hg + line - ag
        return won(selection == ("H" if margin > 0 else ("A" if margin < 0 else "D")))
    if group == "CORNERS":
        if corners is None or corners != corners:
            return None
        return won((corners > line) == (selection == "OVER"))
    raise ValueError(f"unknown market {market}")


def selection_won(market: str, selection: str, home_goals: float, away_goals: float,
                  corners: float | None = None) -> bool | None:
    """True / False for a decided selection; None if unknown or refunded (PUSH)."""
    status = settle_selection(market, selection, home_goals, away_goals, corners)
    return None if status in (None, PUSH) else status == WON


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
MARKET_TITLES = {
    "1X2": "Match result", "DOUBLE_CHANCE": "Double chance", "TOTAL": "Total goals",
    "BTTS": "Both teams to score", "EXACT_SCORE": "Exact score", "HANDICAP": "Asian handicap",
    "EHANDICAP": "European handicap", "CORNERS": "Total corners",
}


def market_title(market: str) -> str:
    group = market_group(market)
    line = market_line(market)
    title = MARKET_TITLES.get(group, market)
    if group in ("TOTAL", "CORNERS") and line is not None:
        return f"{title} {line:g}"
    # no line here: the line belongs to the side that is chosen (shown by the selection label,
    # e.g. "Köln (+0.5)"); printing the home line next to an away pick read as a contradiction

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
    if group in ("HANDICAP", "EHANDICAP"):
        return {"H": f"{home} ({line:+g})", "D": f"Draw ({line:+g})", "A": f"{away} ({-line:+g})"}[selection]
    if group == "CORNERS":
        return f"{selection.title()} {line:g} corners"
    return f"{market} {selection}"
