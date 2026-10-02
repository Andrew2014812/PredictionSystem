"""Plain-language summary of a SHAP explanation ("Why this prediction?")."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.names import feature_topic


def _level(row: pd.Series) -> str:
    value, mean = row["value"], row.get("mean")
    if isinstance(value, str) or value is None or value != value or mean is None or mean != mean:
        return ""
    if abs(value - mean) < 1e-9:
        return ""
    return " (above average)" if value > mean else " (below average)"


@dataclass
class Factor:
    label: str
    topic: str
    direction: str          # "up" | "down"
    value_text: str
    strength: float         # share of the total absolute contribution


@dataclass
class Narrative:
    sentence: str
    supporting: list[Factor]
    opposing: list[Factor]


def _value_text(row: pd.Series) -> str:
    value, mean = row["value"], row.get("mean")
    if isinstance(value, str):
        return value
    if value is None or value != value:
        return "no data"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if mean is not None and mean == mean:
        level = "above" if value > mean else "below"
        text += f" ({level} the usual {mean:.2f})"
    return text


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_narrative(explanation: pd.DataFrame, selection_text: str, n_support: int = 3,
                    n_oppose: int = 2) -> Narrative:
    total = explanation["shap"].abs().sum() or 1.0

    def factors(frame: pd.DataFrame, direction: str, n: int) -> list[Factor]:
        out, topics = [], set()
        for _, row in frame.iterrows():
            topic = feature_topic(row["feature"]) + _level(row)
            if topic in topics:          # one factor per topic keeps the text short
                continue
            topics.add(topic)
            out.append(Factor(row["label"], topic, direction, _value_text(row), abs(row["shap"]) / total))
            if len(out) == n:
                break
        return out

    supporting = factors(explanation.loc[explanation["shap"] > 0], "up", n_support)
    opposing = factors(explanation.loc[explanation["shap"] < 0], "down", n_oppose)
    if supporting:
        sentence = (f"The model favours {selection_text} mainly because of "
                    f"{_join([f.topic for f in supporting])}.")
    else:
        sentence = f"No single factor clearly supports {selection_text}."
    if opposing:
        sentence += f" Working against it: {_join([f.topic for f in opposing])}."
    return Narrative(sentence, supporting, opposing)
