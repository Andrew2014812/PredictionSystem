"""Plain-language summary of a SHAP explanation ("Why this prediction?").

Raw SHAP magnitudes (log-odds / log-rate units) mean little to a reader, so
each factor gets its SHARE OF INFLUENCE: |SHAP of the factor| divided by the
sum of |SHAP| over all features of that explanation (all shares add up to
100 %). It says how much of the model's reasoning for THIS prediction a factor
carries — it is not a change of the probability in percentage points. Raw
values stay in the technical view.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..features.names import feature_label, feature_topic

TEXT = {
    "en": {
        "lead": "FootPredict favours {sel} mainly because of {topics}.",
        "none": "No single factor clearly supports {sel}.",
        "against": " Working against it: {topics}.",
        "and": "and", "above": "above average", "below": "below average",
        "pos": "Supports this prediction", "neg": "Works against this prediction",
    },
    "uk": {
        "lead": "FootPredict схиляється до прогнозу «{sel}». Головні чинники: {topics}.",
        "none": "Жоден окремий фактор явно не підтримує прогноз «{sel}».",
        "against": " Проти цього прогнозу свідчать: {topics}.",
        "and": "і", "above": "вище середнього", "below": "нижче середнього",
        "pos": "На користь прогнозу", "neg": "Проти прогнозу",
    },
}


TARGETS = {
    "en": {"result": "the probability of this result", "goals": "the expected number of goals",
           "corners": "the expected number of corners",
           "note": "Green helps this prediction, red works against it. The bar shows how strongly a factor "
                   "moved the model's estimate of {target} compared with the strongest factor shown — not a "
                   "change in percentage points."},
    "uk": {"result": "ймовірність цього результату", "goals": "очікувану кількість голів",
           "corners": "очікувану кількість кутових",
           "note": "Відсоток — частка фактора в міркуваннях моделі щодо показника «{target}» порівняно з "
                   "усіма іншими факторами (разом 100%). Це не зміна ймовірності."},
}


def model_target(market: str, lang: str = "en") -> str:
    """Which model output the SHAP factors of a market explain."""
    from ..markets.markets import market_group
    group = market_group(market)
    key = "goals" if group in ("TOTAL", "BTTS", "EXACT_SCORE") else ("corners" if group == "CORNERS" else "result")
    return TARGETS.get(lang, TARGETS["en"])[key]


def influence_note(market: str, lang: str = "en") -> str:
    tx = TARGETS.get(lang, TARGETS["en"])
    return tx["note"].format(target=model_target(market, lang))


@dataclass
class Factor:
    feature: str
    label: str
    topic: str
    direction: str          # "up" | "down"
    relative: float         # share of total |SHAP| of the explanation (0..1)
    level: str              # "Supports this prediction" / "Works against this prediction"
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
    """Plain words for the direction of a factor (no ML jargon on regular pages)."""
    text = TEXT.get(lang, TEXT["en"])
    return text["pos"] if positive else text["neg"]


# Elo ratings stay inside the models (removing them made every model clearly worse on
# validation, experiments/xgb_v5) but are hard to interpret, so they are never shown
# to users as a separate argument. Their share still counts in the total.
HIDDEN_FEATURES = re.compile(r"(?:^[ha]_elo$|^elo_|_opp_elo$|_elo_surprise$)")
MIN_SUPPORT_SHARE = 0.03     # a supporting factor below 3 % of the reasoning is not worth showing
MIN_OPPOSE_SHARE = 0.10      # an opposing factor is shown only when it really matters


def key_factors(explanation: pd.DataFrame, lang: str = "en", max_support: int = 4) -> list[Factor]:
    """2–4 main factors FOR the prediction, plus the strongest factor against it when it is
    substantial (>= MIN_OPPOSE_SHARE of the reasoning) — no artificial pros-and-cons balance."""
    total = explanation["shap"].abs().sum() or 1.0
    text = TEXT.get(lang, TEXT["en"])

    def make(row, direction):
        rel = abs(row["shap"]) / total
        return Factor(row["feature"], feature_label(row["feature"], lang),
                      feature_topic(row["feature"], lang) + _position(row, text), direction, rel,
                      influence_level(rel, direction == "up", lang), float(row["shap"]))

    shown = explanation.loc[~explanation["feature"].astype(str).str.contains(HIDDEN_FEATURES)]
    ups = shown.loc[shown["shap"] > 0].sort_values("shap", ascending=False)
    support = [make(r, "up") for _, r in ups.head(max_support).iterrows()]
    support = [f for i, f in enumerate(support) if i < 2 or f.relative >= MIN_SUPPORT_SHARE]
    downs = shown.loc[shown["shap"] < 0].sort_values("shap")
    against = [make(r, "down") for _, r in downs.head(1).iterrows()]
    return support + [f for f in against if f.relative >= MIN_OPPOSE_SHARE]


def influence_table(explanation: pd.DataFrame, n: int = 8, lang: str = "en") -> list[Factor]:
    """Top-n factors with their share of the total influence."""
    top = explanation.head(n)
    total = explanation["shap"].abs().sum() or 1.0
    out = []
    for _, row in top.iterrows():
        rel = abs(row["shap"]) / total
        out.append(Factor(row["feature"], feature_label(row["feature"], lang), feature_topic(row["feature"], lang),
                          "up" if row["shap"] > 0 else "down", rel,
                          influence_level(rel, row["shap"] > 0, lang), float(row["shap"])))
    return out


def build_narrative(explanation: pd.DataFrame, selection_text: str, lang: str = "en", n_support: int = 3,
                    n_oppose: int = 1) -> Narrative:
    text = TEXT.get(lang, TEXT["en"])
    peak = explanation["shap"].abs().sum() or 1.0

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

    shown = explanation.loc[~explanation["feature"].astype(str).str.contains(HIDDEN_FEATURES)]
    supporting = factors(shown.loc[shown["shap"] > 0], "up", n_support)
    opposing = [f for f in factors(shown.loc[shown["shap"] < 0], "down", n_oppose)
                if f.relative >= MIN_OPPOSE_SHARE]
    if supporting:
        sentence = text["lead"].format(sel=selection_text, topics=_join([f.topic for f in supporting], text["and"]))
    else:
        sentence = text["none"].format(sel=selection_text)
    if opposing:
        sentence += text["against"].format(topics=_join([f.topic for f in opposing], text["and"]))
    return Narrative(sentence, supporting, opposing)
