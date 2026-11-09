"""OddsPapi: markets stored vs used, coverage by league, cost of updates, the price archive and price timing.

Inputs: ``local_data/odds/odds.parquet`` (every real quote with provider / bookmaker / quote time),
``local_data/api/oddspapi/*.json`` (ledger, archive progress and coverage, outcome map, tournaments),
``local_data/api/diagnostics/history.csv``, the git history of the ledger and the predictions.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd

from . import common as C
from .football_data import LEAGUES

GROUPS = ["1X2", "DOUBLE_CHANCE", "TOTAL", "BTTS", "HANDICAP", "EHANDICAP", "CORNERS", "EXACT_SCORE"]
FREE_PLAN = 250                                       # billable requests a month (README, "Data sources")


def load_odds() -> pd.DataFrame:
    from src.markets.markets import market_group
    o = pd.read_parquet(C.ODDS_FILE, columns=["match_id", "provider", "bookmaker", "market", "selection", "odds",
                                              "collected_at", "fetched_at", "kickoff"])
    o["group"] = o["market"].map(market_group)
    return o


def season_matches() -> pd.DataFrame:
    f = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                        columns=["match_id", "league", "season", "played", "date", "kickoff"])
    return f.loc[f["season"] == C.SEASON]


def coverage(o: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    played = matches.loc[matches["played"].astype(bool)]
    progress = json.loads((C.API / "oddspapi" / "history_progress.json").read_text())["fixtures"]
    # the same lookup as src.providers.collect.oddspapi_tournaments: candidates found, stored map, configured ids
    cand = json.loads((C.API / "oddspapi" / "tournament_candidates.json").read_text())
    tours = {code: e["found"] for code, e in cand.get("leagues", {}).items() if e.get("found")}
    tours.update(json.loads((C.API / "oddspapi" / "tournaments.json").read_text()))
    from src import config
    tours.update(config.ODDSPAPI_TOURNAMENTS)
    rows = []
    op = o.loc[o["provider"] == "oddspapi"]
    any_ids = set(o["match_id"])
    for lg in LEAGUES:
        ids = set(played.loc[played["league"] == lg, "match_id"])
        items = progress.get(lg, {}).get("items", [])
        g = op.loc[op["match_id"].isin(ids)]
        row = {"league": lg, "tournament": tours.get(lg), "played": len(ids),
               "archive_fixtures": len(items), "archive_mapped": sum(1 for i in items if i.get("match_id")),
               "with_oddspapi": int(g["match_id"].nunique()), "with_any_provider": len(ids & any_ids)}
        for grp in GROUPS:
            row[grp] = int(g.loc[g["group"] == grp, "match_id"].nunique())
        rows.append(row)
    df = pd.DataFrame(rows)
    tot = {"league": "Всего", "tournament": None, **{c: int(df[c].sum()) for c in df.columns if c not in ("league", "tournament")}}
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


def markets(o: pd.DataFrame) -> dict:
    """What the provider gives (catalogue size), what FootPredict maps and stores, what predictions use."""
    ledger = json.loads((C.API / "oddspapi" / "ledger.json").read_text())
    omap = json.loads((C.API / "oddspapi" / "outcome_map.json").read_text())
    mapped_markets = Counter(v[0] for v in omap["outcomes"].values())
    picks = pd.read_parquet(C.PICKS_FILE, columns=["season", "role", "market", "group", "source"])
    picks = picks.loc[picks["season"] == C.SEASON]
    used = picks.groupby("market")["role"].agg(lambda s: ",".join(sorted(set(s)))).to_dict()
    op = o.loc[o["provider"] == "oddspapi"]
    stored = op.groupby(["group", "market"]).agg(rows=("odds", "size"), matches=("match_id", "nunique")).reset_index()
    stored["used_by"] = stored["market"].map(used).fillna("—")
    by_group = (stored.assign(used=stored["used_by"] != "—")
                .groupby("group").agg(markets=("market", "size"), used_markets=("used", "sum"), rows=("rows", "sum"),
                                      rows_used=("rows", lambda s: int(s[stored.loc[s.index, "used_by"] != "—"].sum())))
                .reset_index())
    C.save_csv("oddspapi_markets.csv", stored)
    return {"catalogue_markets": ledger["requests"].get("markets", {}).get("items"),
            "catalogue_fetched_at": ledger["requests"].get("markets", {}).get("at"),
            "outcomes_mapped": len(omap["outcomes"]), "markets_mapped": len(mapped_markets),
            "markets_stored": int(stored["market"].nunique()),
            "markets_used_by_predictions": int(stored.loc[stored["used_by"] != "—", "market"].nunique()),
            "rows_stored": int(len(op)), "rows_in_used_markets": int(stored.loc[stored["used_by"] != "—", "rows"].sum()),
            "by_group": by_group.to_dict("records"),
            "bookmakers": op["bookmaker"].value_counts().to_dict(),
            "picks_markets_not_stored": sorted(set(used) - set(stored["market"]))}


def costs() -> dict:
    """Billable vs free requests per run from the committed ledger and the run diagnostics."""
    ledger = json.loads((C.API / "oddspapi" / "ledger.json").read_text())
    per_run: dict[str, Counter] = defaultdict(Counter)
    kinds: dict[str, Counter] = defaultdict(Counter)
    for key, v in ledger["requests"].items():
        at = v["at"][:13]
        per_run[at]["billable" if v.get("cost") else "free"] += 1
        kinds[at][key.split("|")[0] + ("" if not key.startswith("fixtures|") or key.startswith("fixtures|all")
                                       else "(league)")] += 1
    runs = pd.DataFrame([{"run_hour": k, "billable": c.get("billable", 0), "free": c.get("free", 0),
                          "kinds": ", ".join(f"{a} {b}" for a, b in sorted(kinds[k].items()))}
                         for k, c in sorted(per_run.items())])
    runs["month"] = runs["run_hour"].str[:7]
    runs["archive_import"] = runs["kinds"].str.contains(r"fixtures\(league\)")
    monthly = runs.groupby("month").agg(billable=("billable", "sum"), free=("free", "sum"), runs=("run_hour", "size")).reset_index()
    C.save_csv("oddspapi_runs.csv", runs)
    # month_credits as the ledger itself counts it, at each committed version
    hist = []
    cm = C.commits("local_data/api/oddspapi/ledger.json")
    for r in cm.itertuples():
        raw = C.blob(r.sha, "local_data/api/oddspapi/ledger.json")
        d = json.loads(raw)
        hist.append({"sha": r.sha[:7], "at_utc": str(r.at_utc), "subject": r.subject,
                     "month": d.get("month"), "month_credits": d.get("month_credits"),
                     "remaining": d.get("remaining"), "used": d.get("used"), "entries": len(d.get("requests", {}))})
    hist = pd.DataFrame(hist)
    C.save_csv("oddspapi_ledger_versions.csv", hist)
    diag = pd.read_csv(C.API / "diagnostics" / "history.csv")
    d = diag.loc[diag["source"] == "oddspapi", ["run_at", "n_requests", "n_free", "n_billable", "n_returned",
                                                "n_in_leagues", "n_matches_asked", "n_matches_priced",
                                                "n_quotes_new", "n_quotes_updated"]]
    daily = runs.loc[~runs["archive_import"]]
    return {"free_plan_per_month": FREE_PLAN, "per_month": monthly.to_dict("records"),
            "daily_runs": {"n": int(len(daily)), "billable_mean": float(daily["billable"].mean()),
                           "billable_max": int(daily["billable"].max()), "free_mean": float(daily["free"].mean()),
                           "free_max": int(daily["free"].max())},
            "archive_imports": runs.loc[runs["archive_import"]].to_dict("records"),
            "ledger_month_credits_now": ledger.get("month_credits"), "ledger_month": ledger.get("month"),
            "ledger_has_remaining": any(pd.notna(hist["remaining"])) if "remaining" in hist else False,
            "ledger_versions": int(len(hist)),
            "diagnostics_runs": d.to_dict("records")}


def archive(o: pd.DataFrame) -> dict:
    cov = json.loads((C.API / "oddspapi" / "history_coverage.json").read_text())
    op = o.loc[o["provider"] == "oddspapi"].copy()
    op["after_kickoff_fetch"] = op["fetched_at"] > op["kickoff"]
    op["lead_h"] = (op["kickoff"] - op["collected_at"]).dt.total_seconds() / 3600
    q = lambda s, x: float(s.quantile(x)) if len(s) else None   # noqa: E731
    out = {}
    for name, g in (("archive (fetched after kick-off)", op.loc[op["after_kickoff_fetch"]]),
                    ("live (fetched before kick-off)", op.loc[~op["after_kickoff_fetch"]])):
        out[name] = {"rows": int(len(g)), "matches": int(g["match_id"].nunique()),
                     "lead_median_h": q(g["lead_h"], .5), "lead_p10_h": q(g["lead_h"], .1),
                     "lead_p90_h": q(g["lead_h"], .9), "quoted_at_or_after_kickoff": int((g["lead_h"] <= 0).sum())}
    return {"coverage_file": cov, "rows": out}


def timing_classes() -> dict:
    if not C.TIMING_AUDIT.exists():
        return {"available": False}
    a = pd.read_csv(C.TIMING_AUDIT, usecols=["pick_id", "role", "source", "provider", "price_class"])
    return {"available": True, "file": str(C.TIMING_AUDIT.relative_to(C.ROOT)),
            "by_role": a.pivot_table(index="price_class", columns="role", values="pick_id", aggfunc="count",
                                     fill_value=0).reset_index().to_dict("records"),
            "by_provider": a.pivot_table(index="provider", columns="price_class", values="pick_id", aggfunc="count",
                                         fill_value=0).reset_index().to_dict("records")}


def run() -> dict:
    o = load_odds()
    matches = season_matches()
    cov = coverage(o, matches)
    C.save_csv("oddspapi_coverage.csv", cov)
    total = cov.loc[cov["league"] == "Всего"].iloc[0]
    upcoming = matches.loc[~matches["played"].astype(bool)]
    out = {"coverage": cov.to_dict("records"),
           "played": int(total["played"]), "with_oddspapi": int(total["with_oddspapi"]),
           "with_any_provider": int(total["with_any_provider"]),
           "share_oddspapi": float(total["with_oddspapi"] / total["played"]),
           "upcoming": int(len(upcoming)),
           "upcoming_with_oddspapi": int(upcoming["match_id"].isin(set(o.loc[o["provider"] == "oddspapi", "match_id"])).sum()),
           "tournament_candidates_checked_at": json.loads((C.API / "oddspapi" / "tournament_candidates.json").read_text()).get("checked_at"),
           "markets": markets(o), "costs": costs(), "archive": archive(o), "timing": timing_classes()}
    missing = matches.loc[matches["played"].astype(bool)
                          & ~matches["match_id"].isin(set(o.loc[o["provider"] == "oddspapi", "match_id"]))]
    C.save_csv("oddspapi_missing_matches.csv", missing[["match_id", "league", "date"]].sort_values("date"))
    out["missing_matches"] = missing[["match_id", "league", "date"]].astype(str).to_dict("records")
    C.save_json("oddspapi.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print(pd.DataFrame(r["coverage"]).to_string())
    print(json.dumps({k: r[k] for k in ("played", "with_oddspapi", "with_any_provider", "share_oddspapi", "upcoming",
                                        "upcoming_with_oddspapi", "missing_matches")}, indent=1))
    m = dict(r["markets"]); m.pop("by_group")
    print(json.dumps(m, indent=1))
    print(pd.DataFrame(r["markets"]["by_group"]))
    c = dict(r["costs"]); c.pop("diagnostics_runs")
    print(json.dumps(c, indent=1, default=str))
    print(json.dumps(r["archive"]["rows"], indent=1))
    print(json.dumps(r["timing"], indent=1))
