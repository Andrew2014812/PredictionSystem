"""Audit of the prediction store (local_data/predictions) and of the data the live predictions were made from.

1. Phantom predictions (a club spelt two ways), moved matches, predictions whose match id is not in the
   feature table, results that were withdrawn after a prediction was settled, invariants of the history.
2. Which prediction the unified history keeps for a moved match, and whether that prediction can be settled.
3. Freshness: for every recorded (live) Main of the unified history, the feature table committed by the same
   daily run (git history, read only) is compared with today's table: were all earlier results of both clubs
   already in the data when the prediction was made?

Writes results/store_*.csv and store_audit.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.invariants import check_history
from src.analytics.performance import rescheduled_duplicates, same_day_duplicates, unified_history

from . import common as C

PICK_COLS = ["pick_id", "match_id", "league", "season", "date", "home_team", "away_team", "role", "kind", "market",
             "selection", "probability", "odds", "real_odds", "status", "home_goals", "away_goals", "profit",
             "source", "generated_at", "bookmaker", "odds_priced_at", "settled_at", "time", "group", "provider",
             "origin", "odds_origin"]


def load_picks() -> pd.DataFrame:
    import pyarrow.parquet as pq
    have = pq.ParquetFile(C.PRED_DIR / "picks.parquet").schema.names
    return pd.read_parquet(C.PRED_DIR / "picks.parquet", columns=[c for c in PICK_COLS if c in have])


def history_inputs(pk: pd.DataFrame) -> pd.DataFrame:
    live = pk.loc[pk["source"] == "live"]
    rebuilt = pk.loc[(pk["source"] == "backfill") & ~pk["match_id"].isin(set(live["match_id"]))]
    return pd.concat([live, rebuilt], ignore_index=True)


def roi(picks: pd.DataFrame) -> dict:
    main = picks.loc[picks["role"] == "main"]
    done = main.loc[main["status"].isin(["WON", "LOST"])]
    bets = done.loc[done["odds"].notna() & done["profit"].notna()]
    return {"mains": int(len(main)), "settled": int(len(done)), "bets": int(len(bets)),
            "hit_rate": float((done["status"] == "WON").mean()) if len(done) else None,
            "profit": float(bets["profit"].sum()), "roi": float(bets["profit"].mean()) if len(bets) else None}


def moved_audit(hist: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For every moved-match cluster: the id kept, whether it is a played match of the feature table, and the
    proposed rule: a recorded prediction first (as now), then the prediction whose own match id is the played
    match (the real date), then the latest date and the latest generation."""
    moved = rescheduled_duplicates(hist)
    played_ids = set(feats.loc[feats["played"].astype(bool), "match_id"])
    main = hist.loc[hist["role"] == "main"].drop_duplicates("match_id").set_index("match_id")
    rows = []
    for kept, g in moved.groupby("kept_match_id"):
        ids = [kept] + g["match_id"].tolist()
        frame = main.loc[ids].reset_index()
        frame["order"] = (frame["source"] != "live").astype(int)
        frame["not_played"] = (~frame["match_id"].isin(played_ids)).astype(int)
        fix = frame.sort_values(["order", "not_played", "date", "generated_at"],
                                ascending=[True, True, False, False]).iloc[0]["match_id"]
        for i in ids:
            r = main.loc[i]
            rows.append({"cluster": kept, "match_id": i, "kept": i == kept, "date": str(pd.Timestamp(r["date"]).date()),
                         "source": r["source"], "status": r["status"], "odds": r["odds"], "profit": r["profit"],
                         "generated_at": str(r["generated_at"]), "own_id_played": i in played_ids,
                         "fix_keeps": i == fix})
    t = pd.DataFrame(rows)
    if len(t):
        played_cluster = t.groupby("cluster")["own_id_played"].transform("any")
        stuck = t.loc[t["kept"] & ~t["own_id_played"] & (t["status"] == "UPCOMING") & played_cluster]
    return t, {"clusters": int(moved["kept_match_id"].nunique()), "dropped": int(len(moved)),
               "kept_not_settleable": stuck["match_id"].tolist() if len(t) else []}


def withdrawn_results(pk: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Settled predictions whose match has no played row (by id or by the same pairing within 60 days)."""
    done = pk.loc[pk["status"].isin(["WON", "LOST", "PUSH"]) & (pk["role"] == "main")]
    played = feats.loc[feats["played"].astype(bool)]
    ids = set(played["match_id"])
    by_pair = {k: g["date"].tolist() for k, g in played.groupby(["league", "home_team", "away_team"])}
    rows = []
    for r in done.itertuples():
        if r.match_id in ids:
            continue
        dates = [d for d in by_pair.get((r.league, r.home_team, r.away_team), [])
                 if 0 <= (d - pd.Timestamp(r.date)).days <= 60]
        if not dates:
            rows.append({"match_id": r.match_id, "source": r.source, "status": r.status, "market": r.market,
                         "selection": r.selection, "odds": r.odds, "profit": r.profit,
                         "settled_score": f"{r.home_goals:.0f}-{r.away_goals:.0f}"})
    return pd.DataFrame(rows)


def phantoms(pk: pd.DataFrame) -> pd.DataFrame:
    sn = pd.read_parquet(C.PRED_DIR / "snapshots.parquet")
    rows = []
    for ph, (real, side) in C.PHANTOMS.items():
        for mid, kind in ((ph, "phantom"), (real, "real")):
            s = sn.loc[sn["match_id"] == mid]
            m = pk.loc[(pk["match_id"] == mid) & (pk["role"] == "main")]
            if s.empty:
                continue
            s, m = s.iloc[-1], (m.iloc[-1] if len(m) else None)
            rows.append({"pair": real, "kind": kind, "match_id": mid, "phantom_side": side,
                         "generated_at": str(s["generated_at"]), "p_home": s["p_home"], "p_draw": s["p_draw"],
                         "p_away": s["p_away"], "xg_home": s["xg_home"], "xg_away": s["xg_away"],
                         "exp_corners": s["exp_corners"],
                         "main": f"{m['market']} {m['selection']}" if m is not None else None,
                         "main_prob": m["probability"] if m is not None else None,
                         "main_odds": m["odds"] if m is not None else None,
                         "status": m["status"] if m is not None else None,
                         "profit": m["profit"] if m is not None else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ freshness
def freshness(hist: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    path = "local_data/processed/features.parquet"
    versions = C.file_versions(path)
    live_main = hist.loc[(hist["role"] == "main") & (hist["source"] == "live")].copy()
    live_main["generated_at"] = pd.to_datetime(live_main["generated_at"])
    played = feats.loc[feats["played"].astype(bool), ["match_id", "league", "country", "date", "home_team", "away_team"]]
    sides = pd.concat([played.rename(columns={"home_team": "team"})[["match_id", "country", "date", "team"]],
                       played.rename(columns={"away_team": "team"})[["match_id", "country", "date", "team"]]])
    by_team = {k: g.sort_values("date") for k, g in sides.groupby(["country", "team"])}
    country = feats.drop_duplicates("league").set_index("league")["country"]
    cache: dict[str, set] = {}
    ts = pd.Series([v[1] for v in versions])
    rows = []
    for r in live_main.itertuples():
        after = np.where(ts >= r.generated_at)[0]
        if not len(after):
            continue
        sha, vts, _ = versions[int(after[0])]
        if (vts - r.generated_at) > pd.Timedelta(hours=6):
            continue                                        # no commit of the same run
        if sha not in cache:
            v = C.read_version_parquet(sha, path, columns=["match_id", "played"])
            cache[sha] = set(v.loc[v["played"].astype(bool), "match_id"])
        known = cache[sha]
        cutoff = pd.Timestamp(r.generated_at).normalize()          # results of earlier days must be there
        missing, lag = 0, []
        for team in (r.home_team, r.away_team):
            g = by_team.get((country[r.league], team))
            if g is None:
                continue
            prev = g.loc[(g["date"] < cutoff) & (g["date"] < pd.Timestamp(r.date))]
            for mid, d in zip(prev["match_id"], prev["date"]):
                if mid not in known:
                    missing += 1
                    lag.append((cutoff - d).days)
        rows.append({"match_id": r.match_id, "league": r.league, "date": str(pd.Timestamp(r.date).date()),
                     "generated_at": str(r.generated_at), "features_commit": sha[:7],
                     "missing_previous_results": missing, "max_lag_days": max(lag) if lag else 0,
                     "status": r.status, "odds": r.odds, "profit": r.profit})
    return pd.DataFrame(rows)


def run() -> dict:
    pk = load_picks()
    feats = C.features(["match_id", "league", "country", "season", "date", "kickoff", "home_team", "away_team",
                        "played", "home_goals", "away_goals"])
    hist_in = history_inputs(pk)
    uh = unified_history(pk)
    main = pk.loc[pk["role"] == "main"]
    counts = main.groupby(["source", "status"]).size().unstack(fill_value=0)
    ph = phantoms(pk)
    C.save_csv("store_phantoms.csv", ph)
    moved_t, moved_s = moved_audit(hist_in, feats)
    C.save_csv("store_moved.csv", moved_t)
    same = same_day_duplicates(hist_in.loc[~hist_in["match_id"].isin(set(rescheduled_duplicates(hist_in)["match_id"]))])
    ids = set(feats["match_id"])
    orphans = main.loc[~main["match_id"].isin(ids), ["match_id", "source", "status", "date", "generated_at"]]
    C.save_csv("store_orphans.csv", orphans.assign(date=orphans["date"].astype(str), generated_at=orphans["generated_at"].astype(str)))
    wd = withdrawn_results(uh, feats)
    C.save_csv("store_withdrawn_results.csv", wd)
    inv = check_history(pk, feats)
    # ROI of the unified history now, and with the moved-match rule fixed (prefer the played id)
    base = roi(uh)
    if len(moved_t):
        drop_now = set(moved_t.loc[~moved_t["kept"], "match_id"])
        drop_fix = set(moved_t.loc[~moved_t["fix_keeps"], "match_id"])
        hist_now = hist_in.loc[~hist_in["match_id"].isin(drop_now)]
        hist_fix = hist_in.loc[~hist_in["match_id"].isin(drop_fix)]
        sd = set(same["match_id"])
        fixed = roi(hist_fix.loc[~hist_fix["match_id"].isin(sd)])
        current = roi(hist_now.loc[~hist_now["match_id"].isin(sd)])
    else:
        fixed = current = base
    # phantoms if they had not been removed
    with_phantoms = roi(pd.concat([uh, pk.loc[pk["match_id"].isin(list(C.PHANTOMS)) & (pk["source"] == "live")]]))
    fr = freshness(uh, feats)
    C.save_csv("store_freshness.csv", fr)
    stale = fr.loc[fr["missing_previous_results"] > 0] if len(fr) else fr
    void_profit = pk.loc[(pk["status"] == "VOID") & pk["odds"].isna() & pk["profit"].notna(), "pick_id"].tolist()
    out = {"picks": int(len(pk)), "main_by_source_status": {k: v for k, v in counts.to_dict("index").items()},
           "unified_mains": int((uh["role"] == "main").sum()),
           "unified_by_source": uh.loc[uh["role"] == "main", "source"].value_counts().to_dict(),
           "phantoms": ph.to_dict("records"),
           "moved": moved_s, "same_day_duplicates": same.to_dict("records"),
           "orphan_mains": int(len(orphans)), "orphan_ids": orphans["match_id"].tolist(),
           "withdrawn_results": wd.to_dict("records"),
           "invariants": {k: (len(v), v[:8]) for k, v in inv.items()},
           "void_with_profit_without_price": void_profit,
           "roi_unified_now": current, "roi_unified_moved_rule_fixed": fixed, "roi_unified_reported": base,
           "roi_if_phantoms_kept": with_phantoms,
           "freshness": {"live_mains_checked": int(len(fr)), "with_missing_previous_result": int(len(stale)),
                         "share": float(len(stale) / len(fr)) if len(fr) else None,
                         "by_league": stale["league"].value_counts().to_dict() if len(stale) else {},
                         "max_lag_days": int(stale["max_lag_days"].max()) if len(stale) else 0,
                         "median_lag_days": float(stale["max_lag_days"].median()) if len(stale) else None}}
    if len(fr):
        for flag, g in fr.groupby(fr["missing_previous_results"] > 0):
            done = g.loc[g["status"].isin(["WON", "LOST"])]
            bets = done.loc[done["odds"].notna() & done["profit"].notna()]
            out["freshness"]["stale" if flag else "fresh"] = {
                "mains": int(len(g)), "settled": int(len(done)),
                "hit_rate": float((done["status"] == "WON").mean()) if len(done) else None,
                **{k: v for k, v in C.roi_summary(bets["profit"]).items()}}
    C.save_json("store_audit.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("store_audit")
    run()
