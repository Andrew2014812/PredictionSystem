import json

import pandas as pd
import pytest

from src.providers import aliases
from src.providers.api_football import parse_bet, parse_fixture, parse_odds
from src.providers.cache import ApiCache, fetch_json
from src.providers.football_data import odds_from_matches
from src.providers.schema import best_prices, odds_frame, odds_lookup
from src.providers.the_odds_api import parse_event, parse_event_odds, parse_score

AF_FIXTURE = {
    "fixture": {"id": 1208021, "date": "2026-10-04T14:00:00+00:00", "status": {"short": "FT"}},
    "league": {"id": 39, "season": 2026},
    "teams": {"home": {"name": "Manchester United"}, "away": {"name": "Nottingham Forest"}},
    "goals": {"home": 2, "away": 1}, "score": {"fulltime": {"home": 2, "away": 1}},
}


def test_api_football_fixture_is_mapped_to_league_and_uk_time():
    fx = parse_fixture(AF_FIXTURE)
    assert fx["league"] == "E0" and fx["date"] == "2026-10-04" and fx["time"] == "15:00"   # BST
    assert fx["played"] and (fx["home_goals"], fx["away_goals"]) == (2, 1)
    assert parse_fixture({**AF_FIXTURE, "league": {"id": 999999}}) is None               # league not supported


def test_api_football_unplayed_fixture_has_no_score():
    item = {**AF_FIXTURE, "fixture": {**AF_FIXTURE["fixture"], "status": {"short": "NS"}},
            "goals": {"home": None, "away": None}, "score": {"fulltime": {"home": None, "away": None}}}
    fx = parse_fixture(item)
    assert not fx["played"] and fx["home_goals"] is None


def test_api_football_bets_map_to_system_markets():
    rows = {(m, s): o for m, _, s, o in
            parse_bet("Match Winner", [{"value": "Home", "odd": "2.10"}, {"value": "Draw", "odd": "3.40"},
                                       {"value": "Away", "odd": "3.60"}])
            + parse_bet("Goals Over/Under", [{"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.95"},
                                             {"value": "Over 4.5", "odd": "5.0"}])
            + parse_bet("Both Teams Score", [{"value": "Yes", "odd": "1.80"}, {"value": "No", "odd": "2.00"}])
            + parse_bet("Double Chance", [{"value": "Home/Draw", "odd": "1.30"}, {"value": "Draw/Away", "odd": "1.70"},
                                          {"value": "Home/Away", "odd": "1.33"}])
            + parse_bet("Exact Score", [{"value": "2:1", "odd": "9.00"}])
            + parse_bet("Handicap Result", [{"value": "Home -1", "odd": "3.80"}, {"value": "Draw -1", "odd": "3.70"},
                                            {"value": "Away +1", "odd": "1.80"}])
            + parse_bet("Asian Handicap", [{"value": "Home -0.5", "odd": "2.10"}, {"value": "Away +0.5", "odd": "1.80"},
                                           {"value": "Home -0.25", "odd": "1.9"}, {"value": "Home -1", "odd": "3.0"}])
            + parse_bet("Corners Over Under", [{"value": "Over 9.5", "odd": "1.85"}, {"value": "Under 9.5", "odd": "1.95"}])}
    assert rows[("1X2", "H")] == 2.10 and rows[("TOTAL_2.5", "UNDER")] == 1.95
    assert ("TOTAL_4.5", "OVER") not in rows                       # line not offered by the system
    assert rows[("BTTS", "YES")] == 1.80 and rows[("DOUBLE_CHANCE", "X2")] == 1.70
    assert rows[("EXACT_SCORE", "2-1")] == 9.00
    # "Handicap Result" is the European 3-way market; "Away +1" = home line -1
    assert rows[("EHANDICAP_-1", "H")] == 3.80 and rows[("EHANDICAP_-1", "D")] == 3.70
    assert rows[("EHANDICAP_-1", "A")] == 1.80
    assert rows[("HANDICAP_-1", "H")] == 3.0                       # Asian whole line (refund on a 1-goal win)
    assert rows[("HANDICAP_-0.5", "H")] == 2.10 and rows[("HANDICAP_-0.5", "A")] == 1.80
    assert not any(m in ("HANDICAP_-0.25",) for m, _ in rows)      # quarter Asian lines are skipped
    assert rows[("CORNERS_9.5", "OVER")] == 1.85


def test_api_football_odds_item():
    item = {"fixture": {"id": 7}, "bookmakers": [{"id": 8, "name": "Bet365", "bets": [
        {"name": "Match Winner", "values": [{"value": "Home", "odd": "1.50"}]}]}]}
    fixture_id, rows = parse_odds(item)
    assert fixture_id == "7" and rows == [("1X2", None, "H", 1.5, "Bet365")]


ODDS_EVENT = {
    "id": "e1", "sport_key": "soccer_epl", "commence_time": "2026-10-04T14:00:00Z",
    "home_team": "Arsenal", "away_team": "Chelsea",
    "bookmakers": [
        {"key": "skybet", "title": "Sky Bet", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0}, {"name": "Chelsea", "price": 3.8},
                                        {"name": "Draw", "price": 3.5}]}]},
        {"key": "williamhill", "title": "William Hill", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.05}, {"name": "Chelsea", "price": 3.7},
                                        {"name": "Draw", "price": 3.4}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5},
                                           {"name": "Under", "price": 1.95, "point": 2.5}]},
            {"key": "spreads", "outcomes": [{"name": "Arsenal", "price": 2.1, "point": -0.5},
                                            {"name": "Chelsea", "price": 1.8, "point": 0.5}]}]},
    ],
}


def test_the_odds_api_prefers_bookmaker_priority_and_maps_markets():
    rows = {(m, s): (o, b) for m, _, s, o, b in parse_event_odds(ODDS_EVENT)}
    assert rows[("1X2", "H")] == (2.05, "William Hill")           # williamhill is first in the priority list
    assert rows[("1X2", "D")][0] == 3.4
    assert rows[("TOTAL_2.5", "OVER")][0] == 1.9
    assert rows[("HANDICAP_-0.5", "H")][0] == 2.1 and rows[("HANDICAP_-0.5", "A")][0] == 1.8


def test_the_odds_api_event_and_score():
    ev = parse_event(ODDS_EVENT, "soccer_epl")
    assert ev["league"] == "E0" and ev["time"] == "15:00"
    done = {**ODDS_EVENT, "completed": True, "scores": [{"name": "Arsenal", "score": "3"},
                                                         {"name": "Chelsea", "score": "1"}]}
    assert parse_score(done) == (3, 1)
    assert parse_score({**ODDS_EVENT, "completed": False}) is None


@pytest.mark.parametrize("provider_name,known,expected", [
    ("Manchester United", {"Man United", "Man City"}, "Man United"),
    ("Nottingham Forest", {"Nott'm Forest"}, "Nott'm Forest"),
    ("AFC Bournemouth", {"Bournemouth"}, "Bournemouth"),            # exact match after removing "AFC"
    ("Borussia Mönchengladbach", {"M'gladbach"}, "M'gladbach"),
    ("Paris Saint-Germain", {"Paris SG", "Paris FC"}, "Paris SG"),
    ("Paris FC", {"Paris SG", "Paris FC"}, "Paris FC"),
    ("KVC Westerlo", {"Westerlo", "Mechelen"}, "Westerlo"),
    ("SpVgg Greuther Fürth", {"Greuther Furth", "Bochum"}, "Greuther Furth"),
    ("Estac Troyes", {"Troyes", "Metz"}, "Troyes"),
    ("Aris Thessalonikis", {"Aris", "PAOK"}, "Aris"),                 # unique word-set match
    ("Iraklis 1908", {"Iraklis", "Larisa"}, "Iraklis"),               # founding year ignored
    ("Queen's Park", {"Queens Park", "Partick"}, "Queens Park"),
    ("Lommel United", {"Lommel SK", "Genk"}, "Lommel SK"),
    ("TSG 1899 Hoffenheim", {"Hoffenheim", "Mainz"}, "Hoffenheim"),
    ("Stade Brestois 29", {"Brest", "Lens"}, "Brest"),
    ("Gençlerbirliği S.K.", {"Genclerbirligi", "Konyaspor"}, "Genclerbirligi"),
    ("Real", {"Real Madrid", "Real Sociedad"}, None),                 # ambiguous -> no match
    ("Some Unknown Club", {"Arsenal"}, None),                        # never guessed
    ("Manchester United", {"Arsenal"}, None),                        # alias outside the league is rejected
])
def test_team_alias_resolution(provider_name, known, expected):
    assert aliases.resolve(provider_name, known) == expected


def test_football_data_odds_only_real_and_half_handicap_lines():
    matches = pd.DataFrame([
        {"match_id": "m1", "date": pd.Timestamp("2026-01-01"), "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.6,
         "odds_over25": 1.9, "odds_under25": 1.95, "odds_ah_line": -0.5, "odds_ah_home": 2.0, "odds_ah_away": 1.85},
        {"match_id": "m2", "date": pd.Timestamp("2026-01-01"), "odds_home": float("nan"), "odds_draw": None,
         "odds_away": None, "odds_over25": None, "odds_under25": None, "odds_ah_line": -0.25, "odds_ah_home": 1.9,
         "odds_ah_away": 1.9},
    ])
    odds = odds_from_matches(matches)
    keys = set(zip(odds["match_id"], odds["market"], odds["selection"]))
    assert ("m1", "HANDICAP_-0.5", "H") in keys and ("m1", "1X2", "D") in keys
    assert not any(k[0] == "m2" for k in keys)                      # quarter line + missing prices -> nothing
    assert set(odds["bookmaker"]) == {"Bet365"}


def test_provider_priority():
    rows = odds_frame([
        {"match_id": "m", "provider": "football_data", "bookmaker": "Bet365", "market": "1X2", "selection": "H",
         "odds": 2.0, "collected_at": "2026-01-01"},
        {"match_id": "m", "provider": "api_football", "bookmaker": "Bet365", "market": "1X2", "selection": "H",
         "odds": 2.1, "collected_at": "2026-01-02"},
        {"match_id": "m", "provider": "the_odds_api", "bookmaker": "Sky Bet", "market": "BTTS", "selection": "YES",
         "odds": 1.8, "collected_at": "2026-01-02"},
        {"match_id": "m", "provider": "api_football", "bookmaker": "Bet365", "market": "BTTS", "selection": "NO",
         "odds": 1.0, "collected_at": "2026-01-02"},                 # not a price -> dropped
    ])
    lookup = odds_lookup(rows)["m"]
    assert lookup[("1X2", "H")]["odds"] == 2.1 and lookup[("1X2", "H")]["provider"] == "api_football"
    assert lookup[("BTTS", "YES")]["bookmaker"] == "Sky Bet"
    assert ("BTTS", "NO") not in lookup
    assert len(best_prices(rows)) == 2


def test_api_cache_avoids_repeated_requests(tmp_path, monkeypatch):
    calls = []

    class Response:
        status_code = 200
        headers = {"x-requests-last": "3", "x-requests-remaining": "497"}

        def json(self):
            return {"response": [1, 2]}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        return Response()

    monkeypatch.setattr("src.providers.cache.requests.get", fake_get)
    cache = ApiCache("test", root=tmp_path)
    for _ in range(3):
        body = fetch_json(cache, "https://api.example/odds", {"league": 1, "apiKey": "secret"}, {}, 3600,
                          cost_header="x-requests-last", remaining_header="x-requests-remaining")
    assert body == {"response": [1, 2]} and len(calls) == 1
    assert cache.requests_today() == 1
    stored = json.loads(next(p for p in tmp_path.rglob("*.json")).read_text())
    assert "secret" not in json.dumps(stored)                        # API keys are never written to disk
    assert cache.get("https://api.example/odds", {"league": 1, "apiKey": "other"}, 3600) is not None
    assert cache.get("https://api.example/odds", {"league": 1}, -1) is None   # expired


def _collect_env(tmp_path, monkeypatch):
    from src.providers import collect as col
    from src.providers.ledger import CreditLedger
    from src.storage import LocalStorage
    monkeypatch.setattr(col, "live_storage", LocalStorage(tmp_path / "live"))
    monkeypatch.setattr(col, "odds_storage", LocalStorage(tmp_path / "odds"))
    monkeypatch.setattr(col, "api_storage", LocalStorage(tmp_path / "api"))
    monkeypatch.setattr(col.ApiFootball, "available", property(lambda self: False))
    monkeypatch.setattr(col.OddsPapi, "available", property(lambda self: False))    # never a real request
    from src.providers import teams_registry
    monkeypatch.setattr(teams_registry, "PATH", tmp_path / "provider_ids.json")
    ledger_path = tmp_path / "ledger.json"
    return col, (lambda: CreditLedger(ledger_path))


def _fake_api(col, ledger_factory, calls, league_prices=True, event_body=None, remaining=400):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    kickoff = datetime.now(ZoneInfo("UTC")) + timedelta(hours=20)
    event = {"id": "ev1", "commence_time": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "home_team": "Arsenal", "away_team": "Chelsea"}

    class FakeOddsApi(col.TheOddsApi):
        def __init__(self):
            super().__init__(key="test", ledger=ledger_factory())

        def events(self, sport_key):
            return [event] if sport_key == "soccer_epl" else []

        def _request(self, path, params):
            calls.append(path)
            if path.endswith("/events/ev1/odds"):
                body = event_body if event_body is not None else {**event, "bookmakers": []}
                n = len({m["key"] for b in body.get("bookmakers", []) for m in b["markets"]})
                return body, {"x-requests-last": str(n), "x-requests-used": "1", "x-requests-remaining": str(remaining)}
            body = [{**event, "bookmakers": [{"key": "williamhill", "title": "William Hill", "markets": [
                {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0}, {"name": "Chelsea", "price": 3.6},
                                            {"name": "Draw", "price": 3.4}]}]}]}] if league_prices else []
            return body, {"x-requests-last": "2", "x-requests-used": "2", "x-requests-remaining": str(remaining)}

    return FakeOddsApi, event


MATCHES = pd.DataFrame({"league": ["E0"], "season": [2026], "home_team": ["Arsenal"], "away_team": ["Chelsea"]})


def test_league_odds_are_not_requested_again_even_after_an_empty_answer(tmp_path, monkeypatch):
    col, ledger = _collect_env(tmp_path, monkeypatch)
    calls = []
    fake, _ = _fake_api(col, ledger, calls, league_prices=False)
    monkeypatch.setattr(col, "TheOddsApi", fake)
    col.collect(MATCHES, mode="odds")
    league_calls = [c for c in calls if c.endswith("soccer_epl/odds")]
    assert len(league_calls) == 1
    col.collect(MATCHES, mode="odds")                 # a new CI run: the ledger file survives
    assert len([c for c in calls if c.endswith("soccer_epl/odds")]) == 1


def test_extended_markets_are_bought_once_per_match(tmp_path, monkeypatch):
    import json as _json
    from pathlib import Path
    col, ledger = _collect_env(tmp_path, monkeypatch)
    sample = _json.loads((Path(__file__).parent / "data" / "the_odds_api_event_odds.json").read_text())
    calls = []
    fake, event = _fake_api(col, ledger, calls)
    sample = {**sample, "id": "ev1", "home_team": "Arsenal", "away_team": "Chelsea",
              "bookmakers": [{**b, "markets": [{**m, "outcomes": [
                  {**o, "name": o["name"].replace("Coventry City", "Arsenal").replace("Newcastle United", "Chelsea")}
                  for o in m["outcomes"]]} for m in b["markets"]]} for b in sample["bookmakers"]]}
    monkeypatch.setattr(col, "TheOddsApi", _fake_api(col, ledger, calls, event_body=sample)[0])
    first = col.collect(MATCHES, mode="odds")
    stored = col.load_api_odds()
    markets = set(stored["market"])
    assert {"1X2", "BTTS", "DOUBLE_CHANCE", "TOTAL_1.5", "TOTAL_3.5", "CORNERS_9.5"} <= markets, markets
    assert set(stored["provider"]) == {"the_odds_api"} and stored["bookmaker"].notna().all()
    assert first["odds_rows"] > 10
    col.collect(MATCHES, mode="odds")
    assert len([c for c in calls if c.endswith("/events/ev1/odds")]) == 1


def test_results_mode_spends_no_credits(tmp_path, monkeypatch):
    col, ledger = _collect_env(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(col, "TheOddsApi", _fake_api(col, ledger, calls)[0])
    summary = col.collect(MATCHES, mode="results")
    assert calls == [] and summary["live_rows"] == 1


def test_monthly_reserve_stops_spending(tmp_path, monkeypatch):
    from src import config
    col, ledger = _collect_env(tmp_path, monkeypatch)
    l = ledger()
    l.data["remaining"] = config.ODDS_API_MONTHLY_RESERVE + 1
    l.save()
    calls = []
    monkeypatch.setattr(col, "TheOddsApi", _fake_api(col, ledger, calls)[0])
    col.collect(MATCHES, mode="odds")
    assert calls == []                                  # 2 credits would break the reserve


def test_extended_market_parsing_from_a_real_response():
    import json as _json
    from pathlib import Path
    sample = _json.loads((Path(__file__).parent / "data" / "the_odds_api_event_odds.json").read_text())
    rows = {(m, s): (o, b) for m, _, s, o, b in parse_event_odds(sample)}
    assert rows[("BTTS", "YES")][0] > 1 and rows[("BTTS", "NO")][0] > 1
    assert {("DOUBLE_CHANCE", "1X"), ("DOUBLE_CHANCE", "X2"), ("DOUBLE_CHANCE", "12")} <= set(rows)
    assert ("TOTAL_1.5", "OVER") in rows and ("TOTAL_3.5", "UNDER") in rows
    assert ("CORNERS_9.5", "OVER") in rows and ("CORNERS_10.5", "UNDER") in rows
    assert ("EXACT_SCORE", "1-0") in rows
    assert all(b != "Betfair" and "Exchange" not in b for _, b in rows.values())     # exchanges skipped
    assert not any(m.startswith("HANDICAP") and float(m.split("_")[1]) % 0.5 for m, _ in rows)   # no quarter lines
