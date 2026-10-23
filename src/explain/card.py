"""The "Why …?" card: the model's main factors for one selection, each with the numbers behind it.

A factor is one football quantity. The home and away versions of a statistic ("h_corners_for_form10",
"a_corners_for_form10") form one factor whose evidence shows both teams' values. Its influence is its
share of the total |SHAP| of the model output being explained (all factors of that output add up to
100 %); it is NOT a change of the selection's probability in percentage points.

Which output is explained depends on the market (``shap_explainer.explanation_plan``):

* 1X2, double chance, Asian handicap — the 1X2 classifier (log-odds of the class concerned);
* totals and BTTS — the home- and away-goals models together (log of the expected goals); both
  probabilities rise or fall monotonically with the expected goals, so the direction is exact;
* corner totals — the corners model (log of the expected corners);
* exact score — no share is shown: P(2-1) is not monotonic in the expected goals, so a SHAP share of
  the goal models would suggest a precision that does not exist. The card then shows the expected
  goals and the recent record instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..features.names import feature_label
from ..markets.markets import market_group

_SIDE = re.compile(r"^(?P<side>[ha])_(?P<rest>.+)$")
_PREFIX = re.compile(r"^(Home team|Away team) · ")
NO_SHARE_GROUPS = {"EXACT_SCORE"}
MAX_FACTORS = 5
MIN_SHARE = 0.03


@dataclass
class CardFactor:
    key: str
    name: str
    evidence: str
    share: float          # share of the explained model output's total |SHAP| (0..1)
    supports: bool


def shares_are_meaningful(market: str) -> bool:
    return market_group(market) not in NO_SHARE_GROUPS


def _fmt(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, str):
        return value
    v = float(value)
    name = feature.lower()
    if "rate" in name or "share" in name or name.endswith("_pct"):
        return f"{v * 100:.0f}%" if abs(v) <= 1.0 else f"{v:.0f}%"
    if "position" in name and "norm" not in name:
        return f"{int(round(v))}."
    if "days" in name or name.endswith("_n") or "games" in name or name.endswith("points"):
        return f"{v:.0f}"
    if name.endswith("elo") or "_elo" in name and "diff" not in name:
        return f"{v:.0f}"
    if name.startswith("diff_") or "elo_diff" in name:
        return f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def _group_key(feature: str) -> tuple[str, str | None]:
    m = _SIDE.match(feature)
    return (m.group("rest"), m.group("side")) if m else (feature, None)


def card_factors(explanation: pd.DataFrame, home: str, away: str, lang: str = "en",
                 limit: int = MAX_FACTORS) -> list[CardFactor]:
    """Top factors (supporting ones first in size order, then the strongest opposing one when it is
    substantial). ``explanation``: columns feature, shap, value (from ``explain_selection``)."""
    if explanation is None or explanation.empty:
        return []
    total = float(explanation["shap"].abs().sum()) or 1.0
    groups: dict[str, dict] = {}
    for r in explanation.itertuples(index=False):
        key, side = _group_key(str(r.feature))
        g = groups.setdefault(key, {"shap": 0.0, "abs": 0.0, "values": {}, "features": []})
        g["shap"] += float(r.shap)
        g["abs"] += abs(float(r.shap))
        g["values"][side] = (str(r.feature), r.value)
        g["features"].append(str(r.feature))
    out = []
    for key, g in groups.items():
        first = g["features"][0]
        name = _PREFIX.sub("", feature_label(first, lang))
        name = name[:1].upper() + name[1:]
        vals = g["values"]
        if key == "league":
            from ..ui.labels import league_name
            evidence = league_name(str(vals[None][1])) if None in vals else ""
        elif "h" in vals or "a" in vals:
            parts = [f"{team}: {_fmt(*vals[s])}" for s, team in (("h", home), ("a", away)) if s in vals]
            evidence = " · ".join(parts)
        else:
            evidence = _fmt(*vals[None])
        out.append(CardFactor(key, name, evidence, g["abs"] / total, g["shap"] > 0))
    support = sorted((f for f in out if f.supports), key=lambda f: -f.share)
    against = sorted((f for f in out if not f.supports), key=lambda f: -f.share)
    chosen = [f for i, f in enumerate(support[:limit]) if i < 3 or f.share >= MIN_SHARE]
    if against and against[0].share >= 0.10:            # a factor against is shown only when it really matters
        chosen = (chosen[:limit - 1] if len(chosen) >= limit else chosen) + [against[0]]
    return chosen
