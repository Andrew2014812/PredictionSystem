"""Recheck of the 2026/27 Main / Risk history after the last OddsPapi import (read only).

Universe: ``unified_history`` (src/analytics/performance.py) — per match the predictions recorded before kick-off
(source = live) when they exist, otherwise the retrospective reconstruction (source = backfill); one prediction
per real match. Roles main / risk. Nothing is re-selected and no data is written: the stored selection, price,
status and profit are counted as they are.

Definitions
    bet                     WON / LOST with a real price and a profit (``performance.staked``)
    ROI                     sum(profit) / bets × 100 (stake 1); PUSH and VOID are not staked
    mean EV                 mean of the stored ``ev`` over the bets (EV at the stored price)
    match odds coverage     the match has at least one real bookmaker market: a quote in odds.parquet (OddsPapi,
                            The Odds API) or a Bet365 price in the Football-Data row (features.parquet odds_*)
    prediction coverage     the stored Main / Risk has a real price for exactly its selection and line
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C
from .football_data import LEAGUES

SETTLED = ("WON", "LOST", "PUSH")
MOVED = "перенесённый матч: цена хранится под другой датой"


def load_history() -> pd.DataFrame:
    from src.analytics.performance import unified_history
    from src.markets.markets import market_group
    p = pd.read_parquet(C.PICKS_FILE)
    h = unified_history(p)
    h = h.loc[(h["season"] == C.SEASON) & h["role"].isin(["main", "risk"])].copy()
    h["group"] = h["market"].map(market_group)
    h["is_bet"] = h["status"].isin(["WON", "LOST"]) & h["odds"].notna() & h["profit"].notna()
    h["priced"] = h["odds"].notna()
    if C.TIMING_AUDIT.exists():
        a = pd.read_csv(C.TIMING_AUDIT, usecols=["pick_id", "price_class"]).set_index("pick_id")["price_class"]
        h["price_class"] = h["pick_id"].map(a)
    else:
        h["price_class"] = np.nan
    h.loc[h["priced"] & h["price_class"].isna(), "price_class"] = "нет в аудите"
    h.loc[~h["priced"], "price_class"] = "без цены"
    h["price_class"] = h["price_class"].replace({"-": "не определён"})
    return h


def stats(g: pd.DataFrame) -> dict:
    from src.analytics.performance import summarize
    s = summarize(g)
    bets = g.loc[g["is_bet"]]
    settled = g["status"].isin(SETTLED)
    return {"predictions": int(len(g)), "settled": int(settled.sum()), "won": s["won"], "lost": s["lost"],
            "push": s["push"], "void": s["void"], "pending": s["pending"], "priced": int(g["priced"].sum()),
            "priced_settled": int((g["priced"] & settled).sum()), "bets": s["bets"],
            "profit": s["profit"], "roi": s["roi"], "mean_ev": float(bets["ev"].mean()) if len(bets) else None,
            "avg_odds": s["avg_odds"], "hit_rate": s["hit_rate"],
            "coverage": float((g["priced"] & settled).sum() / settled.sum()) if settled.sum() else None}


def table(h: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in h.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **stats(g)})
    return pd.DataFrame(rows)


def match_coverage(h: pd.DataFrame) -> dict:
    o = pd.read_parquet(C.ODDS_FILE, columns=["match_id", "provider", "collected_at", "kickoff"])
    f = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                        columns=["match_id", "league", "season", "played", "kickoff", "odds_home", "odds_over25",
                                 "odds_ah_home"])
    # older rows (The Odds API) have no stored kick-off: take it from the match row
    ko = o["kickoff"].fillna(o["match_id"].map(f.drop_duplicates("match_id").set_index("match_id")["kickoff"]))
    o = o.loc[o["collected_at"] < ko]
    f = f.loc[(f["season"] == C.SEASON) & f["played"].astype(bool)]
    fd = set(f.loc[f[["odds_home", "odds_over25", "odds_ah_home"]].notna().any(axis=1), "match_id"])
    providers = {p: set(g["match_id"]) for p, g in o.groupby("provider")}
    hm = h.loc[h["status"].isin(SETTLED) & (h["role"] == "main"), ["match_id", "league", "source"]].drop_duplicates("match_id")
    rows = []
    for lg, g in hm.groupby("league"):
        ids = set(g["match_id"])
        rows.append({"league": lg, "matches": len(ids),
                     "any_real_market": len(ids & (fd | set().union(*providers.values()))),
                     "oddspapi": len(ids & providers.get("oddspapi", set())),
                     "the_odds_api": len(ids & providers.get("the_odds_api", set())),
                     "football_data_bet365": len(ids & fd)})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"league": "Всего", **{c: int(df[c].sum()) for c in df.columns if c != "league"}}
    df["share"] = df["any_real_market"] / df["matches"]
    C.save_csv("bets_match_coverage.csv", df)
    tot = df.iloc[-1]
    return {"matches_settled": int(tot["matches"]), "with_any_real_market": int(tot["any_real_market"]),
            "share": float(tot["share"]), "by_league": df.to_dict("records"),
            "without": sorted(set(hm["match_id"]) - (fd | set().union(*providers.values())))}


def reasons(h: pd.DataFrame) -> pd.DataFrame:
    """Every settled or upcoming Main / Risk without a real price, with the reason (v6.1 audit rules)."""
    from scripts.v61_odds_audit import RECOVERABLE, archive_items, load_json, reason_for
    from src import config
    from src.providers.collect import load_api_odds
    odds = load_api_odds()
    odds = odds.loc[odds["match_id"].isin(set(h["match_id"]))]
    odds_by_match = {m: set(zip(g["market"], g["selection"])) for m, g in odds.groupby("match_id")}
    cand = load_json("tournament_candidates.json", {})
    seen = {code: e["found"] for code, e in cand.get("leagues", {}).items() if e.get("found")}
    tours = {**seen, **load_json("tournaments.json", {}), **config.ODDSPAPI_TOURNAMENTS}
    arch = archive_items()
    un = h.loc[~h["priced"]].copy()
    data_day = pd.Timestamp(C.git("log", "-1", "--format=%aI", "--", "local_data").strip()).tz_convert("UTC").tz_localize(None).normalize()
    all_odds = load_api_odds()[["match_id", "market", "selection", "collected_at", "kickoff"]]
    all_odds = all_odds.loc[all_odds["collected_at"] < all_odds["kickoff"]]
    pair = all_odds["match_id"].str.replace(r"-\d{8}-", "|", regex=True)

    def moved_price(p) -> str | None:
        """The same pairing priced under another date (a moved match): the id that holds the price."""
        key = pd.Series([p.match_id]).str.replace(r"-\d{8}-", "|", regex=True).iloc[0]
        cand = all_odds.loc[(pair == key) & (all_odds["match_id"] != p.match_id)
                            & (all_odds["market"] == p.market) & (all_odds["selection"] == p.selection)]
        if cand.empty:
            return None
        days = (pd.to_datetime(cand["match_id"].str.extract(r"-(\d{8})-")[0]) - pd.Timestamp(p.date)).dt.days.abs()
        cand = cand.loc[days <= 7]
        return cand["match_id"].iloc[0] if len(cand) else None

    def why(p):
        if p.status == "UPCOMING":
            if pd.Timestamp(p.date) < data_day - pd.Timedelta(days=1):
                return "дата прошла, результата нет ни из одного источника"
            return "матч ещё не сыгран (цена ещё может прийти)"
        if p.status == "VOID":
            return "VOID: результата нет (матч не состоялся)"
        moved = moved_price(p)
        if moved:
            return f"{MOVED}: {moved}"
        return reason_for(p, odds_by_match, tours, arch)
    un["reason"] = [why(p) for p in un.itertuples(index=False)]
    un["recoverable"] = un["reason"].map(RECOVERABLE).fillna("—")
    un.loc[un["reason"].str.startswith(MOVED), "recoverable"] = "да — цена той же ставки есть под id дня игры"
    cols = ["league", "date", "match_id", "home_team", "away_team", "role", "source", "market", "selection",
            "probability", "status", "reason", "recoverable"]
    un = un[cols].sort_values(["date", "league", "match_id", "role"])
    C.save_csv("bets_unpriced.csv", un)
    return un


def laliga(h: pd.DataFrame, unpriced: pd.DataFrame) -> dict:
    g = h.loc[h["league"] == "SP1"]
    out = {"overall": {r: stats(g.loc[g["role"] == r]) for r in ("main", "risk")}}
    out["by_source"] = table(g, ["role", "source"]).to_dict("records")
    out["by_group_main"] = table(g.loc[g["role"] == "main"], ["group"]).to_dict("records")
    nb = g.loc[~g["is_bet"]].copy()

    def why(p):
        if p.status == "UPCOMING":
            return "ещё не сыгран"
        if p.status == "PUSH":
            return "PUSH (возврат ставки)"
        if p.status == "VOID":
            return "VOID"
        if pd.isna(p.odds):
            r = unpriced.loc[unpriced["match_id"] == p.match_id]
            r = r.loc[r["role"] == p.role, "reason"]
            return r.iloc[0] if len(r) else "без цены"
        return "?"
    nb["why"] = [why(p) for p in nb.itertuples(index=False)]
    cols = ["date", "home_team", "away_team", "role", "source", "market", "selection", "status", "odds", "why"]
    C.save_csv("bets_laliga_not_bets.csv", nb[cols].sort_values(["date", "role"]))
    out["not_bets"] = nb[cols].sort_values(["date", "role"]).astype(str).to_dict("records")
    out["not_bets_by_reason"] = nb.groupby(["role", "why"]).size().rename("n").reset_index().to_dict("records")
    return out


def run() -> dict:
    h = load_history()
    C.save_csv("bets_history_main_risk.csv", h[["pick_id", "match_id", "league", "date", "role", "source", "group",
                                                 "market", "selection", "probability", "odds", "ev", "status",
                                                 "profit", "provider", "bookmaker", "odds_origin", "price_class",
                                                 "is_bet"]])
    overall = table(h, ["role"])
    by_source = table(h, ["role", "source"])
    by_league = table(h, ["league", "role"])
    by_group = table(h, ["group", "role"])
    by_line = table(h, ["market", "role"])
    by_class = table(h, ["price_class", "role"])
    by_class_source = table(h, ["price_class", "source", "role"])
    by_provider = table(h.loc[h["priced"]], ["provider", "role"])
    by_league_source = table(h, ["league", "source", "role"])
    by_group_source = table(h, ["group", "source", "role"])
    by_line_source = table(h, ["market", "source", "role"])
    for name, df in (("bets_overall", overall), ("bets_by_source", by_source), ("bets_by_league", by_league),
                     ("bets_by_group", by_group), ("bets_by_line", by_line), ("bets_by_price_class", by_class),
                     ("bets_by_price_class_source", by_class_source), ("bets_by_provider", by_provider),
                     ("bets_by_league_source", by_league_source), ("bets_by_group_source", by_group_source),
                     ("bets_by_line_source", by_line_source)):
        C.save_csv(f"{name}.csv", df)
    un = reasons(h)
    settled_unpriced = un.loc[~un["status"].isin(["UPCOMING"])]
    prev = json.loads((C.ROOT / "reports" / "v6_1" / "data" / "odds_audit_after.json").read_text())
    out = {"definitions": __doc__.strip(),
           "history_matches": int(h["match_id"].nunique()),
           "overall": overall.to_dict("records"), "by_source": by_source.to_dict("records"),
           "match_coverage": match_coverage(h),
           "prediction_coverage": {r: {"settled": int(g["status"].isin(SETTLED).sum()),
                                       "priced_settled": int((g["priced"] & g["status"].isin(SETTLED)).sum()),
                                       "share": float((g["priced"] & g["status"].isin(SETTLED)).sum()
                                                      / max(int(g["status"].isin(SETTLED).sum()), 1))}
                                   for r, g in h.groupby("role")},
           "unpriced": {"total": int(len(un)), "settled_or_void": int(len(settled_unpriced)),
                        "by_reason": un.groupby(["role", "reason"]).size().rename("n").reset_index().to_dict("records"),
                        "rows": un.astype(str).to_dict("records")},
           "laliga": laliga(h, un),
           "previous_v61_audit": {"built_at": prev.get("built_at"), "roles": prev.get("roles"),
                                  "unpriced_reasons": prev.get("unpriced_reasons"),
                                  "laliga": next((x for x in prev.get("leagues", []) if x["league"] == "SP1"), None)},
           "price_class_available": bool(C.TIMING_AUDIT.exists())}
    C.save_json("bets.json", out)
    return out


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(r["overall"]).T)
    print(pd.DataFrame(r["by_source"]).T)
    print({k: v for k, v in r["match_coverage"].items() if k != "by_league"})
    print(r["prediction_coverage"])
    print(pd.DataFrame(r["unpriced"]["by_reason"]))
    print(pd.DataFrame(r["unpriced"]["rows"]).to_string())
    print(json.dumps(r["laliga"]["overall"], indent=0))
    print(pd.DataFrame(r["laliga"]["not_bets_by_reason"]))
