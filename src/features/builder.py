"""Pre-match feature engineering.

Matches are processed day by day in chronological order:

1. features of *every* match on day D are computed from the state built
   from matches strictly before D;
2. only then are the *played* matches of day D added to the state.

Future fixtures therefore get features but never update tables, form or
head-to-head records, and matches of the same round cannot see each
other's results.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from .state import (IDX, HeadToHead, LeaguePrior, SeasonTables, TeamHistory, nanmean_rows,
                    team_record)

log = logging.getLogger(__name__)

SEASON_STATS = ("gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against",
                "corners_for", "corners_against", "fouls", "yellows", "reds")
VENUE_STATS = ("gf", "ga", "shots_for", "sot_for", "corners_for", "corners_against")
FORM_STATS = ("gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against",
              "corners_for", "corners_against")
FREQ_STATS = ("btts", "over15", "over25", "over35", "clean_sheet", "failed_to_score")
FREQ_WINDOW = 10

ID_COLUMNS = ["match_id", "league", "country", "season", "date", "time", "kickoff",
              "home_team", "away_team", "played"]
TARGET_COLUMNS = ["result", "home_goals", "away_goals", "total_corners"]
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25"]
MATCH_STAT_COLUMNS = ["home_shots", "away_shots", "home_sot", "away_sot", "home_corners",
                      "away_corners", "home_yellows", "away_yellows", "home_reds", "away_reds"]


def _value(v) -> float:
    return float(v) if v is not None and v == v else np.nan


class FeatureBuilder:
    def __init__(self):
        self.windows = config.FORM_WINDOWS
        self.history = TeamHistory(max(self.windows))
        self.tables = SeasonTables()
        self.h2h = HeadToHead(config.H2H_WINDOW)
        self.prior = LeaguePrior(config.LEAGUE_PRIOR_WINDOW)

    # -- reading state ------------------------------------------------------
    def _team_features(self, side: str, match: dict, venue: str) -> dict[str, float]:
        team = match[f"{venue}_team"]
        key = (match["league"], match["season"])
        out: dict[str, float] = {}

        season = self.tables.table(key).get(team)
        if season is not None and season.games > 0:
            mean = season.mean()
            standings = self.tables.standings(key)
            out[f"{side}_season_games"] = season.games
            out[f"{side}_points"] = season.sum("points")
            out[f"{side}_season_ppg"] = mean[IDX["points"]]
            out[f"{side}_season_gd"] = mean[IDX["gf"]] - mean[IDX["ga"]]
            for stat in SEASON_STATS:
                out[f"{side}_season_{stat}"] = mean[IDX[stat]]
            out[f"{side}_position"] = standings[team]
            out[f"{side}_position_norm"] = standings[team] / len(standings)
        else:
            out[f"{side}_season_games"] = 0

        venue_tables = self.tables.home if venue == "home" else self.tables.away
        venue_acc = venue_tables.get(key, {}).get(team)
        if venue_acc is not None and venue_acc.games > 0:
            mean = venue_acc.mean()
            out[f"{side}_venue_games"] = venue_acc.games
            out[f"{side}_venue_ppg"] = mean[IDX["points"]]
            for stat in VENUE_STATS:
                out[f"{side}_venue_{stat}"] = mean[IDX[stat]]
        else:
            out[f"{side}_venue_games"] = 0

        team_key = (match["country"], team)
        for k in self.windows:
            rows = self.history.window(team_key, k)
            out[f"{side}_form{k}_n"] = len(rows)
            if not rows:
                continue
            mean = nanmean_rows(rows)
            out[f"{side}_form{k}_ppg"] = mean[IDX["points"]]
            out[f"{side}_form{k}_win_rate"] = mean[IDX["win"]]
            out[f"{side}_form{k}_draw_rate"] = mean[IDX["draw"]]
            out[f"{side}_form{k}_loss_rate"] = mean[IDX["loss"]]
            out[f"{side}_form{k}_gd"] = mean[IDX["gf"]] - mean[IDX["ga"]]
            for stat in FORM_STATS:
                out[f"{side}_form{k}_{stat}"] = mean[IDX[stat]]

        freq_rows = self.history.window(team_key, FREQ_WINDOW)
        if freq_rows:
            mean = nanmean_rows(freq_rows)
            for stat in FREQ_STATS:
                out[f"{side}_freq_{stat}"] = mean[IDX[stat]]

        # Trend = recent level (last 5) minus season level.
        for stat, name in (("gf", "scoring"), ("ga", "conceding"),
                           ("shots_for", "shots"), ("corners_for", "corners")):
            recent = out.get(f"{side}_form5_{stat}")
            season_level = out.get(f"{side}_season_{stat}")
            if recent is not None and season_level is not None:
                out[f"{side}_trend_{name}"] = recent - season_level

        last = self.history.last_date.get(team_key)
        if last is not None:
            out[f"{side}_rest_days"] = min((match["date"] - last).days, config.REST_DAYS_CAP)
        return out

    def _h2h_features(self, match: dict) -> dict[str, float]:
        meetings = self.h2h.get(match["country"], match["home_team"], match["away_team"])
        out: dict[str, float] = {"h2h_n": len(meetings)}
        if not meetings:
            return out
        home = match["home_team"]
        wins = draws = losses = goals_for = 0
        totals, btts = [], []
        for m_home, _, hg, ag in meetings:
            gf, ga = (hg, ag) if m_home == home else (ag, hg)
            wins += gf > ga
            draws += gf == ga
            losses += gf < ga
            goals_for += gf
            totals.append(hg + ag)
            btts.append(hg > 0 and ag > 0)
        n = len(meetings)
        out.update({
            "h2h_home_win_rate": wins / n,
            "h2h_draw_rate": draws / n,
            "h2h_away_win_rate": losses / n,
            "h2h_avg_goals": float(np.mean(totals)),
            "h2h_home_team_goals": goals_for / n,
            "h2h_btts_rate": float(np.mean(btts)),
            "h2h_over25_rate": float(np.mean(np.array(totals) > 2.5)),
        })
        return out

    @staticmethod
    def _derived_features(f: dict) -> dict[str, float]:
        def diff(a, b):
            return f[a] - f[b] if a in f and b in f else np.nan

        def avg(a, b):
            return (f[a] + f[b]) / 2 if a in f and b in f else np.nan

        out = {
            "diff_season_ppg": diff("h_season_ppg", "a_season_ppg"),
            "diff_position": diff("h_position", "a_position"),
            "diff_points": diff("h_points", "a_points"),
            "diff_season_gd": diff("h_season_gd", "a_season_gd"),
            "diff_venue_ppg": diff("h_venue_ppg", "a_venue_ppg"),
            "diff_form5_ppg": diff("h_form5_ppg", "a_form5_ppg"),
            "diff_form10_ppg": diff("h_form10_ppg", "a_form10_ppg"),
            "diff_form10_gd": diff("h_form10_gd", "a_form10_gd"),
            "diff_form10_shots": diff("h_form10_shots_for", "a_form10_shots_for"),
            "diff_form10_sot": diff("h_form10_sot_for", "a_form10_sot_for"),
            "diff_form10_corners": diff("h_form10_corners_for", "a_form10_corners_for"),
            "diff_rest_days": diff("h_rest_days", "a_rest_days"),
            # attack of one side against the defence of the other
            "h_attack_vs_a_defence": avg("h_venue_gf", "a_venue_ga"),
            "a_attack_vs_h_defence": avg("a_venue_gf", "h_venue_ga"),
            "h_form_attack_vs_a_form_defence": avg("h_form10_gf", "a_form10_ga"),
            "a_form_attack_vs_h_form_defence": avg("a_form10_gf", "h_form10_ga"),
            "h_sot_vs_a_sot_conceded": avg("h_form10_sot_for", "a_form10_sot_against"),
            "a_sot_vs_h_sot_conceded": avg("a_form10_sot_for", "h_form10_sot_against"),
            "h_corners_vs_a_corners_conceded": avg("h_venue_corners_for", "a_venue_corners_against"),
            "a_corners_vs_h_corners_conceded": avg("a_venue_corners_for", "h_venue_corners_against"),
            "form_corners_sum": (f["h_form10_corners_for"] + f["h_form10_corners_against"]
                                 + f["a_form10_corners_for"] + f["a_form10_corners_against"]) / 2
            if all(k in f for k in ("h_form10_corners_for", "h_form10_corners_against",
                                    "a_form10_corners_for", "a_form10_corners_against"))
            else np.nan,
        }
        return out

    @staticmethod
    def _market_features(match: dict) -> dict[str, float]:
        """Bookmaker-implied probabilities (margin removed).

        Only used as model inputs when ``config.USE_ODDS_FEATURES`` is on.
        """
        out = {}
        odds = [_value(match.get(c)) for c in ("odds_home", "odds_draw", "odds_away")]
        if all(o > 1 for o in odds):
            inv = [1 / o for o in odds]
            for name, p in zip(("home", "draw", "away"), inv):
                out[f"market_prob_{name}"] = p / sum(inv)
        over, under = _value(match.get("odds_over25")), _value(match.get("odds_under25"))
        if over > 1 and under > 1:
            out["market_prob_over25"] = (1 / over) / (1 / over + 1 / under)
        return out

    def match_features(self, match: dict) -> dict[str, float]:
        f: dict[str, float] = {}
        f.update(self._team_features("h", match, "home"))
        f.update(self._team_features("a", match, "away"))
        f.update(self._h2h_features(match))
        f.update(self.prior.summary(match["league"]))
        f.update(self._derived_features(f))
        f.update(self._market_features(match))
        return f

    # -- updating state ------------------------------------------------------
    def add_result(self, match: dict) -> None:
        hg, ag = match["home_goals"], match["away_goals"]
        s = {c: _value(match[c]) for c in MATCH_STAT_COLUMNS + ["home_fouls", "away_fouls"]}
        home_rec = team_record(hg, ag, s["home_shots"], s["away_shots"], s["home_sot"], s["away_sot"],
                               s["home_corners"], s["away_corners"], s["home_fouls"],
                               s["home_yellows"], s["home_reds"])
        away_rec = team_record(ag, hg, s["away_shots"], s["home_shots"], s["away_sot"], s["home_sot"],
                               s["away_corners"], s["home_corners"], s["away_fouls"],
                               s["away_yellows"], s["away_reds"])
        key = (match["league"], match["season"])
        self.tables.add(key, match["home_team"], match["away_team"], home_rec, away_rec)
        self.history.add((match["country"], match["home_team"]), match["date"], home_rec)
        self.history.add((match["country"], match["away_team"]), match["date"], away_rec)
        self.h2h.add(match["country"], match["home_team"], match["away_team"], hg, ag)
        self.prior.add(match["league"], hg, ag, s["home_corners"] + s["away_corners"])

    # -- driver -------------------------------------------------------------
    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        matches = matches.sort_values(["date", "kickoff", "league", "home_team"])
        rows = []
        for _, day in matches.groupby("date", sort=True):
            records = day.to_dict("records")
            # 1) features for the whole day from state *before* the day
            for match in records:
                rows.append({**{c: match[c] for c in ID_COLUMNS + ODDS_COLUMNS + MATCH_STAT_COLUMNS},
                             **self.match_features(match)})
            # 2) only played matches change the state
            for match in records:
                if match["played"]:
                    self.add_result(match)
        out = pd.DataFrame(rows)
        played = matches.set_index("match_id")
        out["result"] = out["match_id"].map(played["result"])
        out["home_goals"] = out["match_id"].map(played["home_goals"])
        out["away_goals"] = out["match_id"].map(played["away_goals"])
        out["total_corners"] = out["home_corners"] + out["away_corners"]
        log.info("features: %d matches x %d columns", *out.shape)
        return out


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    return FeatureBuilder().build(matches)
