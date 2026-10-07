"""Plain-language summary of a SHAP explanation ("Why this prediction?").

Raw SHAP magnitudes (log-odds / log-rate units) mean little to a reader, so
each factor gets a *relative influence*: its |SHAP| divided by the largest
|SHAP| of the explanation (0–100 %), shown as a bar with a word
(strong / moderate / slight). Raw values stay in the technical view.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.names import feature_label, feature_topic

TEXT = {
    "en": {
        "lead": "FootPredict favours {sel} mainly because of {topics}.",
        "none": "No single factor clearly supports {sel}.",
        "against": " Working against it: {topics}.",
        "and": "and", "above": "above average", "below": "below average",
        "strong": "Strong", "moderate": "Moderate", "slight": "Slight",
        "pos": "positive influence", "neg": "negative influence",
    },
    "uk": {
        "lead": "FootPredict схиляється до прогнозу «{sel}». Головні чинники: {topics}.",
        "none": "Жоден окремий фактор явно не підтримує прогноз «{sel}».",
        "against": " Проти цього прогнозу свідчать: {topics}.",
        "and": "і", "above": "вище середнього", "below": "нижче середнього",
        "strong": "Сильний", "moderate": "Помірний", "slight": "Слабкий",
        "pos": "позитивний вплив", "neg": "негативний вплив",
    },
}


@dataclass
class Factor:
    feature: str
    label: str
    topic: str
    direction: str          # "up" | "down"
    relative: float         # |SHAP| / max |SHAP| of the explanation (0..1)
    level: str              # e.g. "Strong positive influence"
    shap: float


@dataclass
class Narrative:
    sentence: str
    supporting: list[Factor]
    opposing: list[Factor]


def _join(items: list[str], word: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def _position(row: pd.Series, text: dict) -> str:
    value, mean = row["value"], row.get("mean")
    if isinstance(value, str) or value is None or value != value or mean is None or mean != mean:
        return ""
    if abs(value - mean) < 1e-9:
        return ""
    return f" ({text['above'] if value > mean else text['below']})"


def influence_level(relative: float, positive: bool, lang: str = "en") -> str:
    text = TEXT.get(lang, TEXT["en"])
    strength = text["strong"] if relative >= 0.66 else (text["moderate"] if relative >= 0.33 else text["slight"])
    return f"{strength} {text['pos'] if positive else text['neg']}"


def influence_table(explanation: pd.DataFrame, n: int = 8, lang: str = "en") -> list[Factor]:
    """Top-n factors with relative influence, for the bar view."""
    top = explanation.head(n)
    peak = top["shap"].abs().max() or 1.0
    out = []
    for _, row in top.iterrows():
        rel = abs(row["shap"]) / peak
        out.append(Factor(row["feature"], feature_label(row["feature"], lang), feature_topic(row["feature"], lang),
                          "up" if row["shap"] > 0 else "down", rel,
                          influence_level(rel, row["shap"] > 0, lang), float(row["shap"])))
    return out


def build_narrative(explanation: pd.DataFrame, selection_text: str, lang: str = "en", n_support: int = 3,
                    n_oppose: int = 2) -> Narrative:
    text = TEXT.get(lang, TEXT["en"])
    peak = explanation["shap"].abs().max() or 1.0

    def factors(frame: pd.DataFrame, direction: str, n: int) -> list[Factor]:
        out, topics = [], set()
        for _, row in frame.iterrows():
            topic = feature_topic(row["feature"], lang) + _position(row, text)
            if topic in topics:          # one factor per topic keeps the text short
                continue
            topics.add(topic)
            rel = abs(row["shap"]) / peak
            out.append(Factor(row["feature"], feature_label(row["feature"], lang), topic, direction, rel,
                              influence_level(rel, direction == "up", lang), float(row["shap"])))
            if len(out) == n:
                break
        return out

    supporting = factors(explanation.loc[explanation["shap"] > 0], "up", n_support)
    opposing = factors(explanation.loc[explanation["shap"] < 0], "down", n_oppose)
    if supporting:
        sentence = text["lead"].format(sel=selection_text, topics=_join([f.topic for f in supporting], text["and"]))
    else:
        sentence = text["none"].format(sel=selection_text)
    if opposing:
        sentence += text["against"].format(topics=_join([f.topic for f in opposing], text["and"]))
    return Narrative(sentence, supporting, opposing)
