"""Readable catalogue of every feature family the models use.

A family groups the technical columns that differ only by team side (home /
away) or window (last 5 / 10 / 15), e.g. ``h_form5_gf`` and ``a_form10_gf`` both
belong to "Goals scored — recent form". The technical names stay in the code
and metadata; this catalogue is what the Model Analysis page shows.
"""
from __future__ import annotations

import re
from collections import defaultdict

# (regex on the column name without side / window, EN name, EN text, UA name, UA text)
FAMILIES: list[tuple[str, str, str, str, str]] = [
    (r"^season_games$", "Matches played", "Number of league matches the team has played this season.",
     "Зіграні матчі", "Кількість матчів команди в лізі цього сезону."),
    (r"^points$", "League points", "Points in the league table before the match.",
     "Очки в таблиці", "Очки команди в турнірній таблиці перед матчем."),
    (r"^season_ppg$", "Points per game (season)", "Average points per match this season.",
     "Очки за матч (сезон)", "Середня кількість очок за матч у цьому сезоні."),
    (r"^season_gd$", "Goal difference per game (season)", "Goals scored minus conceded, per match, this season.",
     "Різниця голів за матч (сезон)", "Забиті мінус пропущені голи за матч у цьому сезоні."),
    (r"^position(_norm)?$", "League position", "Place in the table before the match (also relative to the number of teams).",
     "Місце в таблиці", "Місце в таблиці перед матчем (також відносно кількості команд)."),
    (r"^season_(gf|ga)$", "Goals scored / conceded (season)", "Average goals scored and conceded per match this season.",
     "Забиті / пропущені голи (сезон)", "Середня кількість забитих і пропущених голів за матч у сезоні."),
    (r"^season_(shots|sot)_(for|against)$", "Shots and shots on target (season)",
     "Average shots and shots on target made and allowed per match this season.",
     "Удари та удари в площину (сезон)", "Середні удари та удари в площину — свої й суперника — за матч у сезоні."),
    (r"^season_corners_(for|against)$", "Corners (season)", "Average corners won and conceded per match this season.",
     "Кутові (сезон)", "Середня кількість поданих і дозволених кутових за матч у сезоні."),
    (r"^season_(fouls|yellows|reds)$", "Discipline", "Fouls, yellow and red cards per match this season.",
     "Дисципліна", "Фоли, жовті та червоні картки за матч у сезоні."),
    (r"^venue_(games|ppg)$", "Home / away record", "Points per match at home (home team) or away (away team) this season.",
     "Результати вдома / на виїзді", "Очки за матч удома (господарі) або на виїзді (гості) в сезоні."),
    (r"^venue_(gf|ga|shots_for|sot_for)$", "Home / away attack and defence",
     "Goals, shots and shots on target at the venue of the match.",
     "Атака й оборона вдома / на виїзді", "Голи, удари та удари в площину на полі, де проходить матч."),
    (r"^venue_corners_(for|against)$", "Home / away corners", "Corners won and conceded at the venue of the match.",
     "Кутові вдома / на виїзді", "Подані й дозволені кутові на полі, де проходить матч."),
    (r"^venue_form5_.*$", "Recent home / away form", "Goals, shots and corners in the last 5 matches at the same venue.",
     "Поточна форма вдома / на виїзді", "Голи, удари й кутові в останніх 5 матчах на тому ж полі."),
    (r"^form\d+_(n|ppg|win_rate|draw_rate|loss_rate)$", "Recent results",
     "Points per game and win / draw / loss rates over the last 5, 10 and 15 matches.",
     "Останні результати", "Очки за матч і частки перемог / нічиїх / поразок за останні 5, 10 і 15 матчів."),
    (r"^form\d+_gd$", "Recent goal difference", "Goal difference per match over the last 5, 10 and 15 matches.",
     "Поточна різниця голів", "Різниця голів за матч за останні 5, 10 і 15 матчів."),
    (r"^form\d+_(gf|ga)$", "Goals scored / conceded (recent form)", "Average goals over the last 5, 10 and 15 matches.",
     "Забиті / пропущені голи (форма)", "Середні голи за останні 5, 10 і 15 матчів."),
    (r"^form\d+_(shots|sot)_(for|against)$", "Shots (recent form)",
     "Shots and shots on target made and allowed over the last 5, 10 and 15 matches.",
     "Удари (форма)", "Удари та удари в площину — свої й суперника — за останні 5, 10 і 15 матчів."),
    (r"^form\d+_corners_(for|against)$", "Corners (recent form)", "Corners won and conceded over the last 5, 10 and 15 matches.",
     "Кутові (форма)", "Подані й дозволені кутові за останні 5, 10 і 15 матчів."),
    (r"^form10_(corner_diff|corner_share|shot_share)$", "Attacking pressure",
     "Share of corners and shots in the team's matches and the corner difference (last 10).",
     "Атакувальний тиск", "Частка кутових і ударів у матчах команди та різниця кутових (останні 10)."),
    (r"^form10_(opp_elo|elo_surprise)$", "Opponent strength and results vs expectation",
     "Average Elo of the last 10 opponents and how much better or worse the results were than Elo expected.",
     "Сила суперників і результати проти очікувань",
     "Середній Elo останніх 10 суперників і наскільки результати кращі чи гірші за очікування Elo."),
    (r"^form10_adj_(gf|ga)$", "Opponent-adjusted goals",
     "Goals scored / conceded minus what those opponents usually concede / score (last 10).",
     "Голи з поправкою на суперника", "Забиті / пропущені голи мінус те, що ці суперники зазвичай пропускають / забивають (останні 10)."),
    (r"^form10_adj_c[fa]$", "Opponent-adjusted corners",
     "Corners won / conceded minus what those opponents usually allow / win (last 10).",
     "Кутові з поправкою на суперника", "Подані / дозволені кутові мінус звичні показники цих суперників (останні 10)."),
    (r"^freq_.*$", "Event frequencies", "How often the team's last 10 matches had BTTS, over 1.5 / 2.5 / 3.5, clean sheets or no goal.",
     "Частоти подій", "Як часто в останніх 10 матчах команди були «обидві заб'ють», тотали 1.5/2.5/3.5, сухі матчі чи матчі без голів."),
    (r"^trend_.*$", "Trends", "Last 5 matches compared with the season average (scoring, conceding, shots, corners).",
     "Тенденції", "Останні 5 матчів порівняно із середнім за сезон (голи, пропущені, удари, кутові)."),
    (r"^rest_days$|^diff_rest_days$", "Rest days", "Days since the previous league match and the difference between the teams.",
     "Дні відпочинку", "Дні від попереднього матчу в лізі та різниця між командами."),
    (r"^elo$|^elo_(diff|exp_home)$", "Elo strength rating",
     "Club strength rating updated after every match; the gap includes home advantage and gives an expected result.",
     "Рейтинг сили Elo", "Рейтинг сили клубу, що оновлюється після кожного матчу; різниця враховує перевагу поля."),
    (r"^h2h_.*$", "Head-to-head", "Results, goals, BTTS and over 2.5 rates of the last 5 meetings.",
     "Особисті зустрічі", "Результати, голи, частки «обидві заб'ють» і тоталу більше 2.5 в останніх 5 зустрічах."),
    (r"^diff_.*$", "Strength gaps", "Differences between the teams: points, position, goal difference, form, shots, corners.",
     "Різниця в силі", "Різниця між командами: очки, місце, різниця голів, форма, удари, кутові."),
    (r"^(attack_vs|form_attack_vs|sot_vs)_.*$", "Attack vs defence",
     "One team's attack against the other team's defence (goals and shots on target).",
     "Атака проти оборони", "Атака однієї команди проти оборони іншої (голи та удари в площину)."),
    (r"^corners_vs_.*$|^form_corners_sum$", "Corner match-up", "Corners one team wins against the corners the other concedes.",
     "Протистояння за кутовими", "Кутові однієї команди проти кутових, які дозволяє інша."),
    (r"^league_(avg_|home_win|draw|over25|btts).*$", "League context",
     "Recent league averages: goals, corners, home-win, draw, over 2.5 and BTTS rates.",
     "Контекст ліги", "Недавні середні показники ліги: голи, кутові, частки перемог господарів, нічиїх, тоталу 2.5 і «обидві заб'ють»."),
    (r"^league(=.*)?$", "League", "Which league the match is played in.", "Ліга", "У якій лізі проходить матч."),
    (r"^market_prob_.*$", "Bookmaker probabilities", "Margin-free bookmaker probabilities (only if odds features are enabled).",
     "Ймовірності букмекерів", "Ймовірності букмекерів без маржі (лише якщо ввімкнено ознаки з коефіцієнтів)."),
]
_COMPILED = [(re.compile(p), en, et, uk, ut) for p, en, et, uk, ut in FAMILIES]


def _core(name: str) -> str:
    """Column name without the team side prefix."""
    return re.sub(r"^[ha]_", "", name)


def family_of(name: str) -> int | None:
    core = _core(name)
    for i, (rx, *_rest) in enumerate(_COMPILED):
        if rx.search(core) or rx.search(name):
            return i
    return None


def catalogue(model_features: dict[str, list[str]], lang: str = "en") -> list[dict]:
    """One entry per feature family: name, description, models, technical columns."""
    columns: dict[int, set[str]] = defaultdict(set)
    models: dict[int, set[str]] = defaultdict(set)
    other: set[str] = set()
    for model, feats in model_features.items():
        for f in feats:
            i = family_of(f)
            if i is None:
                other.add(f)
                continue
            columns[i].add(f)
            models[i].add(model)
    out = []
    for i in sorted(columns):
        _, en, et, uk, ut = _COMPILED[i]
        out.append({"name": uk if lang == "uk" else en, "description": ut if lang == "uk" else et,
                    "models": sorted(models[i]), "columns": sorted(columns[i])})
    if other:
        out.append({"name": "Other", "description": "", "models": [], "columns": sorted(other)})
    return out
