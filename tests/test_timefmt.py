"""Visitor time zones: UK kick-offs shown in the browser's zone, including the calendar date."""
import pandas as pd
import pytest

from src.ui import timefmt


@pytest.fixture
def zone(monkeypatch):
    def set_zone(name):
        monkeypatch.setattr(timefmt, "user_tz_name", lambda: name)
    return set_zone


def test_late_uk_kick_off_moves_to_the_next_day_in_kyiv(zone):
    zone("Europe/Kyiv")
    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-17"]), "time": ["23:00"],
                       "kickoff": pd.to_datetime(["2026-10-17 23:00"])})
    out = timefmt.with_local_time(df).iloc[0]
    assert out["local_date"] == pd.Timestamp("2026-10-18") and out["local_time"] == "01:00"


def test_dst_is_respected_on_both_sides(zone):
    zone("Europe/Kyiv")
    winter = pd.DataFrame({"date": pd.to_datetime(["2026-12-05"]), "time": ["15:00"]})     # UK GMT, Kyiv +2
    summer = pd.DataFrame({"date": pd.to_datetime(["2026-08-15"]), "time": ["15:00"]})     # UK BST, Kyiv +3
    assert timefmt.with_local_time(winter).iloc[0]["local_time"] == "17:00"
    assert timefmt.with_local_time(summer).iloc[0]["local_time"] == "17:00"
    zone("America/New_York")
    assert timefmt.with_local_time(summer).iloc[0]["local_time"] == "10:00"


def test_unknown_time_keeps_the_stored_date(zone):
    zone("Asia/Tokyo")
    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-17"]), "time": [None]})
    out = timefmt.with_local_time(df).iloc[0]
    assert out["local_date"] == pd.Timestamp("2026-10-17")


def test_quote_time_is_shown_in_the_visitor_zone(zone):
    zone("Europe/Kyiv")
    assert timefmt.local_stamp(pd.Timestamp("2026-10-17 13:42")) == "17 Oct, 15:42"      # UK 13:42 -> Kyiv
    assert timefmt.local_stamp(None) == ""


def test_invalid_browser_zone_falls_back_to_kyiv(monkeypatch):
    class Ctx:
        timezone = "Not/AZone"
    monkeypatch.setattr(timefmt.st, "context", Ctx())
    assert timefmt.user_tz_name() == "Europe/Kyiv"
