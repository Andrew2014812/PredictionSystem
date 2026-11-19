"""Convert the downloaded open Transfermarkt dataset (CSV.gz, untrusted input) into compact typed parquet files.

Run isolated, with the raw folder and the output folder as arguments (nothing is imported from the raw folder):

    .venv/bin/python -I experiments/v6_3/squad/parse_tm.py experiments/v6_3/squad/cache/tm_raw experiments/v6_3/squad/cache/tm

Only the columns the study needs are kept and only games from 2019-07-01 on (two seasons of history before the
first FootPredict season 2021/22). Values are coerced to numbers / dates; anything unparsable becomes missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

START = pd.Timestamp("2019-07-01")
CHUNK = 400_000


def _num(s, kind="int64"):
    return pd.to_numeric(s, errors="coerce").astype("float64" if kind == "float" else "Int64")


def _date(s):
    return pd.to_datetime(s, errors="coerce")


def _read(path: Path, cols: list[str], date_col: str | None = None, keep=None) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, dtype=str, keep_default_na=True):
        if date_col:
            ch[date_col] = _date(ch[date_col])
            ch = ch.loc[ch[date_col] >= START]
        if keep is not None:
            ch = keep(ch)
        parts.append(ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(raw / "competitions.csv.gz", dtype=str)[["competition_id", "name", "type", "country_name"]]
    comp.to_parquet(out / "competitions.parquet")

    clubs = pd.read_csv(raw / "clubs.csv.gz", dtype=str, usecols=["club_id", "name", "domestic_competition_id"])
    clubs["club_id"] = _num(clubs["club_id"])
    clubs.to_parquet(out / "clubs.parquet")

    g = _read(raw / "games.csv.gz", ["game_id", "competition_id", "competition_type", "season", "round", "date",
                                     "home_club_id", "away_club_id", "home_club_goals", "away_club_goals",
                                     "home_club_name", "away_club_name"], "date")
    for c in ("game_id", "season", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"):
        g[c] = _num(g[c])
    g.to_parquet(out / "games.parquet")

    a = _read(raw / "appearances.csv.gz", ["game_id", "player_id", "player_club_id", "date", "competition_id",
                                           "yellow_cards", "red_cards", "goals", "assists", "minutes_played"], "date")
    for c in ("game_id", "player_id", "player_club_id", "yellow_cards", "red_cards", "goals", "assists",
              "minutes_played"):
        a[c] = _num(a[c])
    a.to_parquet(out / "appearances.parquet")
    del a

    lu = _read(raw / "game_lineups.csv.gz", ["date", "game_id", "player_id", "club_id", "type", "position"], "date")
    for c in ("game_id", "player_id", "club_id"):
        lu[c] = _num(lu[c])
    lu["type"] = lu["type"].astype("category")
    lu["position"] = lu["position"].astype("category")
    lu.to_parquet(out / "lineups.parquet")
    del lu

    v = pd.read_csv(raw / "player_valuations.csv.gz", dtype=str, usecols=["player_id", "date", "market_value_in_eur"])
    v["player_id"] = _num(v["player_id"])
    v["date"] = _date(v["date"])
    v["value"] = _num(v.pop("market_value_in_eur"), "float")
    v.dropna().to_parquet(out / "valuations.parquet")

    p = pd.read_csv(raw / "players.csv.gz", dtype=str,
                    usecols=["player_id", "position", "sub_position", "date_of_birth"])
    p["player_id"] = _num(p["player_id"])
    p["date_of_birth"] = _date(p["date_of_birth"])
    p.to_parquet(out / "players.parquet")

    t = pd.read_csv(raw / "transfers.csv.gz", dtype=str,
                    usecols=["player_id", "transfer_date", "from_club_id", "to_club_id"])
    for c in ("player_id", "from_club_id", "to_club_id"):
        t[c] = _num(t[c])
    t["transfer_date"] = _date(t["transfer_date"])
    t.to_parquet(out / "transfers.parquet")
    print("parsed into", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
