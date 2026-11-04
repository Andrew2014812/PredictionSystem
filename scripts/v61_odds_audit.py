"""Odds coverage audit of the 2026/27 season for the v6.1 report (reports/v6_1/Odds_Coverage_Audit.pdf).

Everything is read from the committed data: the one prediction history (unified_history), the stored
prices (local_data/odds), the OddsPapi archive progress (fixtures per league, with the provider's team
names) and the tournament lookup. For every settled Main / Risk without a real price the reason is
named and whether it can still be recovered; La Liga gets a match-by-match table.

    python scripts/v61_odds_audit.py --out reports/v6_1/data/odds_audit_after.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.analytics.performance import (rescheduled_duplicates, same_day_duplicates, summarize,  # noqa: E402
                                       unified_history)
from src.leagues import enabled_leagues  # noqa: E402
from src.markets.markets import market_group  # noqa: E402
from src.prediction import store  # noqa: E402
from src.providers.collect import load_api_odds  # noqa: E402

API = config.DATA_DIR / "api" / "oddspapi"
NO_TOURNAMENT = "league not found at OddsPapi (no tournament id)"
ID_CONFLICT = "club refused by the id-conflict guard (Barcelona / Espanyol, fixed in v6.1)"
NOT_MAPPED = "OddsPapi fixture exists but was not matched to our match"
NOT_LISTED = "match not in the OddsPapi fixture list"
NO_SNAPSHOT = "matched, but no pre-match price snapshot"
NO_MARKET = "market not offered for the match"
NO_LINE = "this line of the market not offered"
NO_SELECTION = "selection not offered"
RECOVERABLE = {ID_CONFLICT: "yes — the corrected ids map these fixtures at the next history run",
               NOT_MAPPED: "maybe — after a name fix, at the next history run",
               NO_TOURNAMENT: "only if OddsPapi lists the league (checked monthly, see candidates)",
               NOT_LISTED: "no (OddsPapi has no such fixture)", NO_SNAPSHOT: "no (the archive holds no price)",
               NO_MARKET: "no", NO_LINE: "no", NO_SELECTION: "no"}


def _num(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    return v


def load_json(name: str, default):
    path = API / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def archive_items() -> pd.DataFrame:
    progress = load_json("history_progress.json", {"fixtures": {}})
    rows = [{**item, "league": lg, "period": "–".join(e.get("period", []))}
            for lg, e in progress["fixtures"].items() for item in e.get("items", [])]
    return pd.DataFrame(rows)


def reason_for(pick, odds_by_match: dict, tours: dict, archive: pd.DataFrame) -> str:
    sels = odds_by_match.get(pick.match_id)
    if sels is None:
        if pick.league not in tours:
            return NO_TOURNAMENT
        clubs = {pick.home_team, pick.away_team}
        if pick.league == "SP1" and clubs & {"Barcelona", "Espanol"}:
            return ID_CONFLICT
        day = archive.loc[(archive["league"] == pick.league)
                          & (archive["kickoff_utc"].str[:10] == str(pd.Timestamp(pick.date).date()))]
        if not day.empty and day["match_id"].isna().any():
            return NOT_MAPPED
        if archive.empty or pick.match_id not in set(archive["match_id"].dropna()):
            return NOT_LISTED
        return NO_SNAPSHOT
    if any(m == pick.market and s == pick.selection for m, s in sels):
        return NO_SNAPSHOT
    if any(m == pick.market for m, _ in sels):
        return NO_SELECTION
    if any(market_group(m) == market_group(pick.market) for m, _ in sels):
        return NO_LINE
    return NO_MARKET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-repairs", action="store_true",
                        help="apply the history repairs of the next pipeline run in memory (nothing is written)")
    args = parser.parse_args()
    season = config.current_season()
    feats = pd.read_parquet(config.PROCESSED_DIR / "features.parquet",
                            columns=["match_id", "league", "season", "played", "date", "kickoff"])
    played = feats.loc[(feats["season"] == season) & feats["played"].astype(bool)]
    picks = store.load_picks()
    if args.with_repairs:
        from src.prediction.history_repair import repair_history
        held: dict = {}
        print("repairs applied in memory:", repair_history(pd.read_parquet(config.PROCESSED_DIR / "features.parquet"),
                                                           write=False, out=held))
        picks = held.get("picks", picks)
    live = picks.loc[picks["source"] == "live"]
    raw = pd.concat([live, picks.loc[(picks["source"] == "backfill") & ~picks["match_id"].isin(set(live["match_id"]))]])
    raw = raw.loc[raw["season"] == season]
    moved, spelt = rescheduled_duplicates(raw), same_day_duplicates(raw)
    history = unified_history(picks)
    history = history.loc[history["season"] == season]
    odds = load_api_odds()
    odds = odds.loc[odds["match_id"].isin(set(history["match_id"]) | set(played["match_id"]))] if not odds.empty else odds
    odds_by_match = {m: set(zip(g["market"], g["selection"])) for m, g in odds.groupby("match_id")} if len(odds) else {}
    tours = {**load_json("tournaments.json", {}), **config.ODDSPAPI_TOURNAMENTS}
    archive = archive_items()
    candidates = load_json("tournament_candidates.json", {})
    settled = history.loc[history["status"].isin(["WON", "LOST", "PUSH"])]
    roles = settled.loc[settled["role"].isin(["main", "risk"])]

    # reasons for every settled Main / Risk without a real price
    unpriced = roles.loc[roles["odds"].isna()].copy()
    unpriced["reason"] = [reason_for(p, odds_by_match, tours, archive) for p in unpriced.itertuples(index=False)]
    unpriced["recoverable"] = unpriced["reason"].map(RECOVERABLE)
    reasons = (unpriced.groupby(["role", "reason"]).size().rename("n").reset_index()
               .assign(recoverable=lambda d: d["reason"].map(RECOVERABLE)))

    # price provenance of the priced Main / Risk
    priced = roles.loc[roles["odds"].notna()].copy()
    kick = played.set_index("match_id")["kickoff"]
    priced["kickoff"] = priced["match_id"].map(kick)
    when = pd.to_datetime(priced.get("odds_priced_at"), errors="coerce")
    after_kickoff = int(((when.notna()) & (pd.to_datetime(priced["kickoff"]) < when)).sum())

    leagues = []
    for lg in enabled_leagues():
        g = played.loc[played["league"] == lg.code]
        h = settled.loc[settled["league"] == lg.code]
        row = {"league": lg.code, "name": f"{lg.country} {lg.name}", "played": len(g),
               "tournament": tours.get(lg.code),
               "archive_fixtures": int((archive["league"] == lg.code).sum()) if len(archive) else 0,
               "archive_mapped": int(archive.loc[archive["league"] == lg.code, "match_id"].notna().sum())
               if len(archive) else 0,
               "with_any_price": int(odds.loc[odds["match_id"].isin(set(g["match_id"])), "match_id"].nunique())
               if len(odds) else 0}
        for role in ("main", "risk"):
            s = summarize(h.loc[h["role"] == role])
            row[f"{role}_settled"], row[f"{role}_bets"] = s["settled"], s["bets"]
            row[f"{role}_push"], row[f"{role}_roi"] = s["push"], s["roi"]
        leagues.append(row)

    main = settled.loc[settled["role"] == "main"].assign(group=lambda d: d["market"].map(market_group))
    by_group = (main.groupby("group").agg(predictions=("match_id", "size"), priced=("odds", lambda s: int(s.notna().sum())))
                .reset_index())
    by_line = (main.assign(line=main["market"]).groupby("line")
               .agg(predictions=("match_id", "size"), priced=("odds", lambda s: int(s.notna().sum()))).reset_index()
               .sort_values("predictions", ascending=False))

    laliga = history.loc[(history["league"] == "SP1") & (history["role"] == "main")].copy()
    laliga["has_result"] = laliga["status"].isin(["WON", "LOST", "PUSH"])
    laliga["is_bet"] = laliga["status"].isin(["WON", "LOST"]) & laliga["odds"].notna() & laliga["profit"].notna()
    detail = laliga.loc[~laliga["is_bet"]].copy()

    def why(p):
        if p.status == "UPCOMING":
            return "no result yet"
        if p.status == "VOID":
            return "void (no result in time)"
        if p.status == "PUSH":
            return "refund (push): not a won / lost bet"
        return reason_for(p, odds_by_match, tours, archive)
    detail["why_not_a_bet"] = [why(p) for p in detail.itertuples(index=False)]
    detail["recoverable"] = detail["why_not_a_bet"].map(RECOVERABLE).fillna("—")
    cols = ["date", "home_team", "away_team", "match_id", "market", "selection", "probability", "status",
            "odds", "has_result", "why_not_a_bet", "recoverable"]
    s_ll = summarize(laliga)

    out = {
        "built_at": datetime.now().isoformat(timespec="seconds"), "season": season,
        "played_matches": len(played), "history_matches": int(history["match_id"].nunique()),
        "duplicates_removed": {"moved_matches": moved.to_dict("records"), "club_spelt_twice": spelt.to_dict("records")},
        "roles": {r: summarize(history.loc[history["role"] == r]) for r in ("main", "risk")},
        "main_without_risk": int(len(set(history.loc[history["role"] == "main", "match_id"])
                                     - set(history.loc[history["role"] == "risk", "match_id"]))),
        "unpriced_reasons": reasons.to_dict("records"),
        "priced_provenance": {
            "provider": priced["provider"].fillna("unknown").value_counts().to_dict() if "provider" in priced else {},
            "bookmaker": priced["bookmaker"].fillna("unknown").value_counts().head(12).to_dict()
            if "bookmaker" in priced else {},
            "odds_origin": priced["odds_origin"].fillna("recorded").value_counts().to_dict()
            if "odds_origin" in priced else {},
            "with_quote_time": int(when.notna().sum()), "quoted_after_kickoff": after_kickoff},
        "leagues": leagues,
        "main_by_group": by_group.to_dict("records"),
        "main_by_line": by_line.head(40).to_dict("records"),
        "laliga": {"main_predictions": s_ll["settled"], "bets": s_ll["bets"], "push": s_ll["push"],
                   "pending": s_ll["pending"], "not_a_bet": detail[cols].sort_values("date").to_dict("records")},
        "tournament_candidates": candidates,
        "price_rows": {"total": len(odds), "by_provider": odds["provider"].value_counts().to_dict() if len(odds) else {}},
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=_num, ensure_ascii=False), encoding="utf-8")
    unpriced[["league", "date", "match_id", "role", "market", "selection", "probability", "reason", "recoverable"]] \
        .sort_values(["league", "date"]).to_csv(path.with_name(path.stem + "_unpriced.csv"), index=False)
    print(f"season {season}: {len(played)} played; Main {out['roles']['main']['settled']} settled, "
          f"{out['roles']['main']['bets']} bets; Risk {out['roles']['risk']['settled']} settled, "
          f"{out['roles']['risk']['bets']} bets; La Liga {s_ll['settled']} Main, {s_ll['bets']} bets")
    print(reasons.to_string(index=False))


if __name__ == "__main__":
    main()
