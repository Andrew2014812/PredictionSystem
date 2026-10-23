"""Data audit for the v6 report: price coverage of season 2026/27, the one prediction history, reasons
for missing prices. Read-only; writes one JSON.

    python scripts/v6_audit.py --out reports/v6/audit_before.json     # before the archive / repairs
    python scripts/v6_audit.py --out reports/v6/audit_after.json      # after them
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.analytics.performance import group_summary, summarize, unified_history  # noqa: E402
from src.markets.markets import market_group  # noqa: E402
from src.prediction import store  # noqa: E402
from src.providers import load_api_odds  # noqa: E402

GROUPS = ["1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "EXACT_SCORE", "HANDICAP", "EHANDICAP", "CORNERS"]
PROGRESS = config.API_DIR / "oddspapi" / "history_progress.json"


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if v != v else float(v)
    return v


def role_stats(picks: pd.DataFrame) -> dict:
    out = {}
    for role in ("main", "risk", "market"):
        s = summarize(picks.loc[picks["role"] == role])
        out[role] = {k: _num(v) for k, v in s.items()}
        out[role]["odds_coverage"] = (s["bets"] / s["settled"]) if s["settled"] else None
    return out


def missing_reasons(picks: pd.DataFrame, odds: pd.DataFrame, mapped: set, fixtures_known: bool) -> dict:
    """Why a settled Main / Risk of the history has no real price."""
    todo = picks.loc[picks["role"].isin(["main", "risk"]) & picks["status"].isin(["WON", "LOST", "PUSH"])
                     & picks["odds"].isna()]
    by_match = odds.groupby("match_id") if not odds.empty else None
    priced_matches = set(odds["match_id"]) if not odds.empty else set()
    markets = {m: set(zip(g["market"], g["selection"])) for m, g in by_match} if by_match is not None else {}
    reasons = {}
    for r in todo.itertuples(index=False):
        if r.match_id not in priced_matches:
            if fixtures_known and r.match_id not in mapped:
                why = "match not found at OddsPapi (or not mapped safely)"
            else:
                why = "no pre-match price snapshot for the match"
        else:
            sels = markets.get(r.match_id, set())
            if any(m == r.market for m, _ in sels):
                why = "selection / line not offered by the bookmakers"
            elif any(market_group(m) == market_group(r.market) for m, _ in sels):
                why = "this line of the market not offered"
            else:
                why = "market not offered for the match"
        reasons.setdefault(r.role, {}).setdefault(why, 0)
        reasons[r.role][why] += 1
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    season = config.current_season()
    feats = pd.read_parquet(config.PROCESSED_DIR / "features.parquet",
                            columns=["match_id", "league", "season", "played", "date"])
    played = feats.loc[(feats["season"] == season) & feats["played"].astype(bool)]
    picks = store.load_picks()
    history = unified_history(picks)
    history = history.loc[history["season"] == season]
    odds = load_api_odds()
    odds = odds.loc[odds["match_id"].isin(played["match_id"])] if not odds.empty else odds
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"fixtures": {}}
    mapped = {i["match_id"] for v in progress["fixtures"].values() for i in v.get("items", []) if i.get("match_id")}
    fixtures = {k: len(v.get("items", [])) for k, v in progress["fixtures"].items()}
    odp = odds.loc[odds["provider"] == "oddspapi"] if not odds.empty else odds
    groups_by_match = (odp.assign(g=odp["market"].map(market_group)).groupby("match_id")["g"].agg(set)
                       if not odp.empty else pd.Series(dtype=object))
    leagues = []
    for code, g in played.groupby("league"):
        ids = set(g["match_id"])
        h = history.loc[history["match_id"].isin(ids)]
        settled = h.loc[h["status"].isin(["WON", "LOST", "PUSH"])]
        row = {"league": code, "matches": len(g), "oddspapi_fixtures": fixtures.get(code, 0),
               "mapped": len(ids & mapped),
               "with_any_price": int(odds.loc[odds["match_id"].isin(ids), "match_id"].nunique()) if not odds.empty else 0,
               "with_oddspapi_price": int(sum(1 for m in ids if m in groups_by_match.index)),
               "main_priced": int(settled.loc[settled["role"] == "main", "odds"].notna().sum()),
               "main_settled": int((settled["role"] == "main").sum()),
               "risk_priced": int(settled.loc[settled["role"] == "risk", "odds"].notna().sum()),
               "risk_settled": int((settled["role"] == "risk").sum()),
               "archive_prices": int((h.get("odds_origin", pd.Series(dtype=object)) == "archive").sum())}
        for grp in GROUPS:
            row[f"m_{grp}"] = int(sum(1 for m in ids if m in groups_by_match.index and grp in groups_by_match[m]))
        leagues.append(row)
    hist_main = history.loc[history["role"] == "main"]
    by_market = group_summary(hist_main, "group")
    by_league = group_summary(hist_main, "league")
    out = {
        "built_at": datetime.now().isoformat(timespec="seconds"), "season": season,
        "played_matches": len(played), "history_matches": int(history["match_id"].nunique()),
        "history_by_source": history.groupby("source")["match_id"].nunique().to_dict(),
        "history_roles": role_stats(history),
        "recorded_roles": role_stats(history.loc[history["source"] == "live"]),
        "rebuilt_roles": role_stats(history.loc[history["source"] == "backfill"]),
        "main_without_risk": int(len(set(hist_main["match_id"]) - set(history.loc[history["role"] == "risk", "match_id"]))),
        "risk_reconstructed": int((history.get("origin", pd.Series(dtype=object)) == "risk_reconstructed").sum()),
        "archive_prices": int((history.get("odds_origin", pd.Series(dtype=object)) == "archive").sum()),
        "price_rows": {"total": len(odds), "by_provider": odds["provider"].value_counts().to_dict() if len(odds) else {}},
        "bookmakers": odds["bookmaker"].value_counts().head(10).to_dict() if len(odds) else {},
        "leagues": leagues,
        "missing_reasons": missing_reasons(history, odds, mapped, bool(progress["fixtures"])),
        "main_by_market": by_market.to_dict("records") if not by_market.empty else [],
        "main_by_league": by_league.to_dict("records") if not by_league.empty else [],
        "risk_by_market": group_summary(history.loc[history["role"] == "risk"], "group").to_dict("records"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=_num, ensure_ascii=False), encoding="utf-8")
    t = out["history_roles"]
    print(f"history: {out['history_matches']} matches; Main {t['main']['settled']} settled, {t['main']['bets']} priced, "
          f"ROI {t['main']['roi']}; Risk {t['risk']['settled']} settled, {t['risk']['bets']} priced, ROI {t['risk']['roi']}; "
          f"Main without Risk {out['main_without_risk']}")


if __name__ == "__main__":
    main()
