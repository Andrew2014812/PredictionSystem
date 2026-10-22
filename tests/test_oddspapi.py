"""OddsPapi: market catalogue mapping, odds parsing (live and historical), quota and team registry.

The payloads follow the documented v4 formats (oddspapi.io/en/docs: get-markets, get-odds,
get-historical-odds); they are replaced by recorded real answers once a key is available."""
from datetime import datetime, timezone

import pytest

from src.providers.ledger import CreditLedger
from src.providers.oddspapi import outcome_map, parse_odds
from src.providers.teams_registry import TeamRegistry

CATALOGUE = [
    {"marketId": 101, "marketName": "1X2", "marketType": "1x2", "period": "result", "handicap": 0,
     "outcomes": [{"outcomeId": 101, "outcomeName": "1"}, {"outcomeId": 102, "outcomeName": "X"},
                  {"outcomeId": 103, "outcomeName": "2"}]},
    {"marketId": 104, "marketName": "Both Teams To Score", "marketType": "bothteamsscore", "period": "result",
     "outcomes": [{"outcomeId": 104, "outcomeName": "Yes"}, {"outcomeId": 105, "outcomeName": "No"}]},
    {"marketId": 1010, "marketName": "Over/Under", "marketType": "totals", "period": "result", "handicap": 2.5,
     "outcomes": [{"outcomeId": 1010, "outcomeName": "Over"}, {"outcomeId": 1011, "outcomeName": "Under"}]},
    {"marketId": 1068, "marketName": "Asian Handicap", "marketType": "spreads", "period": "result",
     "handicap": -0.5, "outcomes": [{"outcomeId": 1068, "outcomeName": "1"}, {"outcomeId": 1069, "outcomeName": "2"}]},
    {"marketId": 1064, "marketName": "Asian Handicap", "marketType": "spreads", "period": "result",
     "handicap": -1, "outcomes": [{"outcomeId": 1064, "outcomeName": "1"}, {"outcomeId": 1065, "outcomeName": "2"}]},
    {"marketId": 1070, "marketName": "Asian Handicap", "marketType": "spreads", "period": "result",
     "handicap": -0.25, "outcomes": [{"outcomeId": 1070, "outcomeName": "1"}, {"outcomeId": 1071, "outcomeName": "2"}]},
    {"marketId": 10140, "marketName": "European Handicap", "marketType": "handicap", "period": "result",
     "handicap": -1, "outcomes": [{"outcomeId": 10140, "outcomeName": "1"}, {"outcomeId": 10141, "outcomeName": "X"},
                                  {"outcomeId": 10142, "outcomeName": "2"}]},
    {"marketId": 101902, "marketName": "Double Chance", "marketType": "doublechance", "period": "result",
     "outcomes": [{"outcomeId": 101902, "outcomeName": "1X"}, {"outcomeId": 101903, "outcomeName": "12"},
                  {"outcomeId": 101904, "outcomeName": "X2"}]},
    {"marketId": 10336, "marketName": "Correct Score", "marketType": "correctscore", "period": "result",
     "outcomes": [{"outcomeId": 10337, "outcomeName": "1:0"}, {"outcomeId": 10338, "outcomeName": "2:1"}]},
    {"marketId": 10803, "marketName": "Total Corners", "marketType": "totals", "period": "result", "handicap": 9.5,
     "outcomes": [{"outcomeId": 10803, "outcomeName": "Over"}, {"outcomeId": 10804, "outcomeName": "Under"}]},
    {"marketId": 10208, "marketName": "1X2 1st Half", "marketType": "1x2", "period": "1st half",
     "outcomes": [{"outcomeId": 10208, "outcomeName": "1"}, {"outcomeId": 10209, "outcomeName": "X"},
                  {"outcomeId": 10210, "outcomeName": "2"}]},
    {"marketId": 20001, "marketName": "Total Cards", "marketType": "totals", "period": "result", "handicap": 4.5,
     "outcomes": [{"outcomeId": 20001, "outcomeName": "Over"}, {"outcomeId": 20002, "outcomeName": "Under"}]},
]


def _book(prices: dict, active=True) -> dict:
    return {"bookmakerIsActive": True, "markets": {"m": {"marketActive": True, "outcomes": {
        str(o): {"players": {"0": {"price": p, "active": active, "changedAt": "2026-10-17T10:00:00Z"}}}
        for o, p in prices.items()}}}}


def test_catalogue_maps_every_supported_market_and_nothing_else():
    m = outcome_map(CATALOGUE)
    assert m["101"] == ("1X2", None, "H") and m["102"][2] == "D" and m["103"][2] == "A"
    assert m["104"] == ("BTTS", None, "YES") and m["105"][2] == "NO"
    assert m["1010"] == ("TOTAL_2.5", 2.5, "OVER") and m["1011"][2] == "UNDER"
    assert m["1068"] == ("HANDICAP_-0.5", -0.5, "H") and m["1069"][2] == "A"
    assert m["1064"] == ("HANDICAP_-1", -1.0, "H")                          # whole Asian line kept
    assert "1070" not in m                                                    # quarter line skipped
    assert m["10141"] == ("EHANDICAP_-1", -1.0, "D")                          # European 3-way kept apart
    assert m["101902"][2] == "1X" and m["101903"][2] == "12" and m["101904"][2] == "X2"
    assert m["10338"] == ("EXACT_SCORE", None, "2-1")
    assert m["10803"] == ("CORNERS_9.5", 9.5, "OVER")
    assert "10208" not in m and "20001" not in m                              # half time and cards ignored


def test_live_odds_use_bookmaker_priority_and_skip_exchanges():
    body = {"bookmakerOdds": {"betfair-ex": _book({101: 2.30}), "pinnacle": _book({101: 2.10, 104: 1.80}),
                              "bet365": _book({101: 2.05})}}
    rows = {(m, s): (p, b) for m, _, s, p, b, _ in parse_odds(body, outcome_map(CATALOGUE))}
    assert rows[("1X2", "H")] == (2.05, "bet365")                             # first in priority
    assert rows[("BTTS", "YES")] == (1.80, "pinnacle")                        # next bookmaker fills the gap
    assert all(b != "betfair-ex" for _, b in rows.values())


def test_historical_odds_use_the_last_price_before_kick_off():
    snaps = [{"createdAt": "2026-10-16T09:00:00Z", "price": 2.20, "active": True},
             {"createdAt": "2026-10-17T13:55:00Z", "price": 2.05, "active": True},
             {"createdAt": "2026-10-17T14:30:00Z", "price": 1.40, "active": True}]   # in-play: never used
    body = {"bookmakers": {"bet365": {"markets": {"101": {"outcomes": {"101": {"players": {"0": snaps}}}}}}}}
    kickoff = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)
    (row,) = parse_odds(body, outcome_map(CATALOGUE), kickoff)
    assert row[3] == 2.05 and row[5] == datetime(2026, 10, 17, 13, 55, tzinfo=timezone.utc)


def test_no_price_is_invented_for_inactive_or_missing_quotes():
    body = {"bookmakerOdds": {"bet365": _book({101: 2.05, 102: 3.4}, active=False)}}
    assert parse_odds(body, outcome_map(CATALOGUE)) == []
    assert parse_odds({}, outcome_map(CATALOGUE)) == []


def test_team_registry_never_merges_two_clubs(tmp_path):
    reg = TeamRegistry(tmp_path / "ids.json")
    known = {"Man United", "Man City"}
    assert reg.resolve("oddspapi", "E0", "England", "Manchester United", 35, known) == "Man United"
    assert reg.resolve("oddspapi", "E0", "England", "Man Utd FC", 35, known) == "Man United"   # learned id
    # the same provider id now claimed by another club: refused and reported, nothing attached
    assert reg.resolve("oddspapi", "E0", "England", "Manchester City", 35, known) is None
    assert reg.conflicts and reg.conflicts[0][2] == "England|Man United"
    assert reg.resolve("oddspapi", "E0", "England", "Real", 99, {"Real Madrid", "Real Sociedad"}) is None
    assert ("oddspapi", "E0", "Real") in reg.unmatched
    reg.save()
    assert TeamRegistry(tmp_path / "ids.json").data["oddspapi"]["35"] == "England|Man United"


def test_team_registry_gives_one_club_one_id(tmp_path):
    reg = TeamRegistry(tmp_path / "ids.json")
    known = {"Barcelona", "Santander"}                     # the league's Espanol is spelt differently
    assert reg.resolve("oddspapi", "SP1", "Spain", "FC Barcelona", 2814, known) == "Barcelona"
    # a second id for the same club is a mis-read name of another club: refused, not learned
    assert reg.resolve("oddspapi", "SP1", "Spain", "Barcelona", 2817, known) is None
    assert "2817" not in reg.data["oddspapi"] and reg.conflicts


def test_quota_ledger_retries_errors_but_not_empty_answers(tmp_path):
    ledger = CreditLedger(tmp_path / "l.json", monthly_reserve=20)
    now = datetime(2026, 10, 17, 12, 0)
    ledger.record("odds|1", 1, 0, now, status="not_found")                    # answered: no market
    ledger.record("odds|2", 1, 0, now, status=500)                             # temporary error
    assert ledger.requested_within("odds|1", 12, now)                          # not asked again
    assert not ledger.requested_within("odds|2", 12, now)                      # may be retried
    ledger.data["remaining"] = 21
    assert ledger.can_spend(1, 10) and not ledger.can_spend(2, 10)             # monthly reserve kept


@pytest.mark.parametrize("league,name", [("E0", "Wolverhampton Wanderers"), ("SP1", "Athletic Club")])
def test_registry_uses_aliases_for_provider_names(tmp_path, league, name):
    known = {"Wolves", "Ath Bilbao"}
    assert TeamRegistry(tmp_path / "x.json").resolve("oddspapi", league, "C", name, None, known) in known


# -- transport: rate limits, retries, quota --------------------------------------------------------
class _Resp:
    def __init__(self, status, body=None, text="", headers=None):
        self.status_code, self._body, self.text, self.headers = status, body, text, headers or {}

    def json(self):
        return self._body


def _client(tmp_path, monkeypatch, responses):
    import requests as rq

    from src.providers import oddspapi as op
    from src.providers.cache import ApiCache
    calls, sleeps = [], []
    seq = iter(responses)

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(op.requests, "get", fake_get)
    monkeypatch.setattr(op.time, "sleep", lambda s: sleeps.append(s))
    client = op.OddsPapi("k", ApiCache("oddspapi", tmp_path), CreditLedger(tmp_path / "l.json", 20))
    client.ledger.data["remaining"] = 200
    return client, calls, sleeps, rq


NOW = datetime(2026, 10, 22, 9, 0)


def test_rate_limit_waits_as_asked_and_retries(tmp_path, monkeypatch):
    client, calls, sleeps, _ = _client(tmp_path, monkeypatch, [
        _Resp(429, text='{"code":"RATE_LIMITED","message":"Please wait 0.12 seconds"}'),
        _Resp(200, {"bookmakers": {}})])
    body, status = client._get("historical-odds", {"fixtureId": "f1"}, None, NOW, 0)
    assert status == 200 and body == {"bookmakers": {}} and len(calls) == 2
    assert any(0.12 < s < 1.0 for s in sleeps)                    # waited about what the API asked
    assert client.rate_limited == 1


def test_retry_after_header_is_respected(tmp_path, monkeypatch):
    from src.providers.oddspapi import retry_after
    assert retry_after(_Resp(429, headers={"Retry-After": "2"})) == 2.0
    assert retry_after(_Resp(429, text="Please wait 0.4 seconds")) == 0.4
    assert retry_after(_Resp(429, text="slow down")) == 1.0


def test_rate_limit_never_loops_forever(tmp_path, monkeypatch):
    client, calls, _, _ = _client(tmp_path, monkeypatch, [_Resp(429, text="RATE_LIMITED wait 0.1 s")] * 10)
    body = client.historical_odds("f1", NOW)
    assert body is None and len(calls) == 4                       # bounded attempts
    assert "historical|f1" not in client.ledger.data["requests"]   # not remembered: retried next run


def test_monthly_quota_stops_the_run(tmp_path, monkeypatch):
    client, calls, _, _ = _client(tmp_path, monkeypatch, [_Resp(429, text='{"code":"REQUEST_LIMIT_EXCEEDED"}')])
    assert client._get("odds", {"fixtureId": "f1"}, "odds|f1", NOW, 50) == (None, "quota")
    assert client._get("odds", {"fixtureId": "f2"}, "odds|f2", NOW, 50) == (None, "quota")
    assert len(calls) == 1 and client.quota_exhausted


def test_server_errors_back_off_and_are_not_recorded_as_answers(tmp_path, monkeypatch):
    import requests as rq
    client, calls, sleeps, _ = _client(tmp_path, monkeypatch, [
        _Resp(502, text="bad gateway"), rq.ConnectionError("down"), _Resp(200, {"bookmakers": {"bet365": {}}})])
    body, status = client._get("odds", {"fixtureId": "f1"}, "odds|f1", NOW, 50)
    assert status == 200 and len(calls) == 3 and any(s >= 2 for s in sleeps)   # exponential backoff
    client2, _, _, _ = _client(tmp_path / "b", monkeypatch, [rq.ConnectionError("down")] * 4)
    assert client2._get("odds", {"fixtureId": "f1"}, "odds|f1", NOW, 50) == (None, "network")
    assert not client2.ledger.requested_within("odds|f1", 12, NOW)


def test_empty_answer_is_remembered_for_the_refresh_window(tmp_path, monkeypatch):
    client, _, _, _ = _client(tmp_path, monkeypatch, [_Resp(200, {"bookmakers": {}})])
    client.historical_odds("f9", NOW)
    assert client.ledger.requested_within("historical|f9", 12, NOW)          # "no odds" is an answer


def test_quotes_that_contradict_other_lines_are_dropped():
    from src.providers.oddspapi import inconsistent_quotes
    ok = {("TOTAL_2.5", "OVER"): (2.5, 1.85, None), ("TOTAL_2.5", "UNDER"): (2.5, 1.95, None),
          ("TOTAL_3.5", "OVER"): (3.5, 3.40, None), ("TOTAL_3.5", "UNDER"): (3.5, 1.30, None)}
    assert inconsistent_quotes(ok) == set()
    swapped = {**ok, ("TOTAL_3.5", "OVER"): (3.5, 1.30, None), ("TOTAL_3.5", "UNDER"): (3.5, 3.40, None)}
    assert ("TOTAL_3.5", "UNDER") in inconsistent_quotes(swapped)          # "under 3.5 at 3.40" is not used
    hcp = {("HANDICAP_-1.5", "H"): (-1.5, 3.2, None), ("HANDICAP_-1.5", "A"): (-1.5, 1.35, None),
           ("HANDICAP_-0.5", "H"): (-0.5, 1.6, None), ("HANDICAP_-0.5", "A"): (-0.5, 2.4, None)}
    assert inconsistent_quotes(hcp) == set()
    flipped = {**hcp, ("HANDICAP_-1.5", "H"): (-1.5, 1.35, None), ("HANDICAP_-1.5", "A"): (-1.5, 3.2, None)}
    assert inconsistent_quotes(flipped)
