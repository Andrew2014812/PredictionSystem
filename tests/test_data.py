import pandas as pd

from src.data.cleaning import deduplicate, drop_stale_fixtures, make_match_id, normalise
from tests.conftest import raw_row


def test_same_pairing_in_different_seasons_is_not_a_duplicate(make_matches):
    m = make_matches([("10/08/2024", "Arsenal", "Chelsea", 2, 1),
                      ("09/08/2025", "Arsenal", "Chelsea", 0, 0)])
    assert len(deduplicate(m)) == 2


def test_true_duplicate_keeps_the_played_row():
    raw = pd.DataFrame([raw_row("E0", "09/08/2025", "Arsenal", "Chelsea"),               # fixture
                        raw_row("E0", "09/08/2025", "Arsenal", "Chelsea", 1, 0, "17:30")])  # result
    out = deduplicate(normalise(raw))
    assert len(out) == 1
    assert bool(out.iloc[0]["played"]) and out.iloc[0]["home_goals"] == 1


def test_match_id_ignores_kickoff_time():
    a = normalise(pd.DataFrame([raw_row("E0", "09/08/2025", "Arsenal", "Chelsea", time="15:00")]))
    b = normalise(pd.DataFrame([raw_row("E0", "09/08/2025", "Arsenal", "Chelsea", time="17:30")]))
    assert a["match_id"].iloc[0] == b["match_id"].iloc[0] == make_match_id(
        "E0", pd.Timestamp("2025-08-09"), "Arsenal", "Chelsea")


def test_rescheduled_fixture_is_dropped_but_later_meeting_kept(make_matches):
    m = make_matches([("09/08/2025", "Arsenal", "Chelsea", None, None),   # postponed fixture
                      ("20/08/2025", "Arsenal", "Chelsea", 1, 1),         # played later
                      ("01/03/2026", "Arsenal", "Chelsea", None, None)])  # far later meeting
    out = drop_stale_fixtures(deduplicate(m))
    assert len(out) == 2
    assert set(out["date"].dt.strftime("%Y-%m-%d")) == {"2025-08-20", "2026-03-01"}


def test_unplayed_rows_have_no_result_or_stats(make_matches):
    m = make_matches([("09/08/2025", "Arsenal", "Chelsea", None, None)])
    row = m.iloc[0]
    assert not row["played"] and row["result"] is None and pd.isna(row["home_corners"])
    assert row["odds_home"] == 2.0      # market information is kept


def test_two_digit_years_and_season():
    m = normalise(pd.DataFrame([raw_row("E0", "09/08/21", "A", "B", 1, 0),
                                raw_row("E0", "09/02/22", "A", "B", 1, 0)]))
    assert list(m["season"]) == [2021, 2021]
