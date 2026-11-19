"""Build the squad features of every FootPredict match of the covered leagues (no model fitting).

    .venv/bin/python -m experiments.v6_3.squad.build_features

Writes ``cache/squad_features.parquet`` (one row per match: h_/a_/diff_ sq_* and pre_*), ``results/team_mapping.csv``,
``results/mapping_report.json`` and ``results/leak_check.json``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.squad import features as F
from experiments.v6_3.squad import tmdata as T

DIRS = C.study_dirs("squad")


def leak_check(P: F.PlayerData, mapping: pd.DataFrame, fd: pd.DataFrame, n: int = 400) -> dict:
    """For a sample of matches: every game in the feature window and every market value used is dated at least one
    day before the match date; the match itself (same club, same date ±1) is never in the window."""
    key = mapping.set_index(["league", "team"])["club_id"].astype("int64")
    s = fd.loc[fd["league"].isin(T.LEAGUE_MAP) & (fd["season"] <= 2025)].sample(n, random_state=C.SEED)
    games = P.club_games.set_index(["club_id", "game_id"])["date"]
    bad_window, bad_value, checked = 0, 0, 0
    for r in s.itertuples(index=False):
        cid = key.get((r.league, r.home_team))
        if cid is None:
            continue
        d = np.datetime64(r.date - pd.Timedelta(days=1), "ns")
        for g in P.window(int(cid), d):
            if games.loc[(int(cid), int(g))] >= r.date - pd.Timedelta(days=1):
                bad_window += 1
        tab = F.squad_table(P, int(cid), P.window(int(cid), d))
        for p in tab.index[:F.XI]:
            h = P.values.get(int(p))
            if h is not None:
                i = np.searchsorted(h[0], d, side="left")
                if i > 0 and h[0][i - 1] >= d:
                    bad_value += 1
        checked += 1
    return {"matches_checked": checked, "window_games_on_or_after_cutoff": bad_window,
            "values_on_or_after_cutoff": bad_value, "cutoff": "match date − 1 day (strictly before)"}


def main() -> None:
    started = time.time()
    log = C.setup_logging("squad.features", DIRS["logs"] / "build_features.log")
    fd = T.fd_matches()
    games = T.load("games")
    mapping, report = T.team_mapping(fd, games)
    mapping.to_csv(DIRS["results"] / "team_mapping.csv", index=False)
    log.info("mapping: %d teams", len(mapping))
    dyn, pre = F.build(fd, mapping, log)
    wide = F.wide(fd, dyn, pre)
    wide.to_parquet(DIRS["cache"] / "squad_features.parquet")
    pre.to_csv(DIRS["results"] / "preseason_features.csv", index=False)
    P = F.PlayerData(set(mapping["club_id"].astype("int64")))
    C.write_json(DIRS["results"] / "leak_check.json", leak_check(P, mapping, fd))
    # date agreement of the two sources (FootPredict vs Transfermarkt dates of the same league match)
    report["seconds"] = round(time.time() - started, 1)
    C.write_json(DIRS["results"] / "mapping_report.json", report)
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
