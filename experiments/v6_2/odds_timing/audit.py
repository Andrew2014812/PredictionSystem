"""Price-timing audit of every stored 2026/27 Main / Risk with a price (the product owner's audit).

Universe: the one history the website shows (src.analytics.performance.unified_history: a live prediction when
one was recorded before kick-off, otherwise the backfill reconstruction), season 2026/27, roles main / risk,
with a real price.

Decision time:
    live      generated_at of the stored version (the latest version before kick-off; UTC);
    backfill  the prediction was generated AFTER the match (retrospective evaluation; features use only
              pre-match data). Its generation time is not a decision time: no bet could have been placed then.
              We therefore treat its price as the closing price and also report how many of these closing
              quotes had already been published at nominal decision times (24 h and the median live lead
              before kick-off) — at those times the closing price was already on offer.

Classes of the price of a pick:
    A1  attached by the live pipeline at generation (odds_origin empty): known to the system when it decided;
    A2  added later from the archive, but the quote was published at or before generation (it existed in the
        market at decision time; the system did not have it);
    B   added later from the archive: the last pre-kick-off quote, published AFTER generation;
    C   closing price of a backfill prediction (decision after the match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import quotes as Q

WON_LOST = ("WON", "LOST", "PUSH")
CLASS_RU = {"A1": "A1 — записана при генерации", "A2": "A2 — из архива, опубликована до генерации",
            "B": "B — из архива, опубликована после генерации", "C": "C — закрытие (backfill)",
            "U": "не определено"}


def load_history() -> pd.DataFrame:
    from src.analytics.performance import unified_history
    picks = pd.read_parquet(C.PRED_DIR / "picks.parquet")
    hist = unified_history(picks)
    hist = hist.loc[(hist["season"] == C.LIVE) & hist["role"].isin(["main", "risk"])].copy()
    snaps = pd.read_parquet(C.PRED_DIR / "snapshots.parquet", columns=["match_id", "kickoff", "version",
                                                                          "first_generated_at"])
    snaps = snaps.drop_duplicates("match_id", keep="last").set_index("match_id")
    hist["kickoff"] = pd.to_datetime(hist["match_id"].map(snaps["kickoff"]))
    hist["version"] = hist["match_id"].map(snaps["version"])
    hist["first_generated_at"] = pd.to_datetime(hist["match_id"].map(snaps["first_generated_at"]))
    hist["kickoff_utc"] = C.uk_to_utc(hist["kickoff"]).to_numpy()
    hist["quote_utc_stored"] = C.uk_to_utc(hist["odds_priced_at"]).to_numpy()
    hist["gen_lead_h"] = (hist["kickoff_utc"] - hist["generated_at"]).dt.total_seconds() / 3600
    hist["group"] = hist["market"].map(C.market_group)
    hist["odds_origin"] = hist["odds_origin"].fillna("recorded")
    return hist.reset_index(drop=True)


def recover_quote_times(hist: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Quote time of OddsPapi prices stored without ``odds_priced_at``: the quote of the same match, market,
    selection, bookmaker and price strictly before kick-off (archive picks were filled with exactly that quote)."""
    out = hist.copy()
    out["quote_utc"] = out["quote_utc_stored"]
    out["quote_time_from"] = np.where(out["quote_utc"].notna(), "stored", None)
    need = out["odds"].notna() & out["quote_utc"].isna() & (out["provider"] == "oddspapi")
    if need.any():
        cand = q.loc[q["priced_at_utc"] < q["kickoff_utc"], Q.KEY + ["odds", "priced_at_utc", "archive"]]
        m = out.loc[need, Q.KEY + ["odds"]].reset_index().merge(cand, on=Q.KEY, suffixes=("", "_q"))
        m = m.loc[np.isclose(m["odds"], m["odds_q"])]
        # the archive repair takes the latest pre-kick-off quote: prefer the archive row, then the latest
        m = m.sort_values(["archive", "priced_at_utc"]).drop_duplicates("index", keep="last")
        out.loc[m["index"].to_numpy(), "quote_utc"] = m["priced_at_utc"].to_numpy()
        out.loc[m["index"].to_numpy(), "quote_time_from"] = "matched_in_odds_parquet"
    return out


def verify_against_versions(hist: pd.DataFrame) -> pd.Series:
    """True when the stored live price equals the price of the same version in the version archive
    (written at generation), False when it differs, NaN when the version is not archived (pre-v5)."""
    v = pd.read_parquet(C.PRED_DIR / "versions.parquet", columns=["match_id", "role", "version", "market",
                                                                   "selection", "odds"])
    m = hist[["match_id", "role", "version", "market", "selection", "odds"]].reset_index().merge(
        v, on=["match_id", "role", "version"], how="left", suffixes=("", "_v"))
    m = m.drop_duplicates("index").set_index("index")
    same = np.isclose(m["odds"].fillna(-1), m["odds_v"].fillna(-1)) & (m["market"] == m["market_v"])
    res = pd.Series(np.where(m["market_v"].isna(), np.nan, same.astype(float)), index=m.index)
    return res.reindex(hist.index)


def classify(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    priced = h["odds"].notna()
    live, back = h["source"] == "live", h["source"] == "backfill"
    restored = h["odds_origin"].isin(["archive", "average_replaced"])
    cls = np.full(len(h), "U", dtype=object)
    cls[(live & priced & ~restored).to_numpy()] = "A1"
    before = (h["quote_utc"] <= h["generated_at"]).to_numpy()
    after = (h["quote_utc"] > h["generated_at"]).to_numpy()
    cls[(live & priced & restored).to_numpy() & before] = "A2"
    cls[(live & priced & restored).to_numpy() & after] = "B"
    cls[(back & priced).to_numpy()] = "C"
    cls[~priced.to_numpy()] = "-"
    h["price_class"] = cls
    h["quote_after_gen_h"] = (h["quote_utc"] - h["generated_at"]).dt.total_seconds() / 3600
    h["quote_lead_h"] = (h["kickoff_utc"] - h["quote_utc"]).dt.total_seconds() / 3600
    return h


def _roi(df: pd.DataFrame) -> dict:
    d = df.loc[df["status"].isin(WON_LOST) & df["odds"].notna() & df["profit"].notna()]
    return C.roi_summary(d["profit"], d["odds"], (d["status"] == "WON").astype(float))


def counts(h: pd.DataFrame) -> dict:
    p = h.loc[h["price_class"] != "-"]
    out = {"picks_total": int(len(h)), "picks_priced": int(len(p)),
           "by_class": p.groupby("price_class").size().to_dict(),
           "by_class_role": p.groupby(["price_class", "role"]).size().unstack(fill_value=0).to_dict("index"),
           "by_source_class": p.groupby(["source", "price_class"]).size().unstack(fill_value=0).to_dict("index"),
           "by_provider_class": p.groupby([p["provider"].fillna("—"), "price_class"]).size()
           .unstack(fill_value=0).to_dict("index"),
           "by_bookmaker_class": p.groupby([p["bookmaker"].fillna("—"), "price_class"]).size()
           .unstack(fill_value=0).to_dict("index"),
           "by_group_class": p.groupby(["group", "price_class"]).size().unstack(fill_value=0).to_dict("index"),
           "by_origin_class": p.groupby(["odds_origin", "price_class"]).size().unstack(fill_value=0).to_dict("index"),
           "by_quote_time_from": p.groupby([p["quote_time_from"].fillna("unknown"), "price_class"]).size()
           .unstack(fill_value=0).to_dict("index"),
           "unpriced_by_source_role": h.loc[h["price_class"] == "-"].groupby(["source", "role"]).size().to_dict()}
    out["by_class_role"] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in out["by_class_role"].items()}
    return out


def timing_stats(h: pd.DataFrame) -> dict:
    out = {}
    live = h.loc[h["source"] == "live"]
    lead = live.drop_duplicates("match_id")["gen_lead_h"]
    out["live_generation_lead_h"] = {"n_matches": int(len(lead)), "median": float(lead.median()),
                                     "q25": float(lead.quantile(.25)), "q75": float(lead.quantile(.75)),
                                     "min": float(lead.min()), "max": float(lead.max()),
                                     "share_le_1h": float((lead <= 1).mean()), "share_le_6h": float((lead <= 6).mean()),
                                     "share_le_24h": float((lead <= 24).mean())}
    first = live.drop_duplicates("match_id")
    fl = (first["kickoff_utc"] - first["first_generated_at"]).dt.total_seconds() / 3600
    out["live_first_version_lead_h"] = {"median": float(fl.median()), "q25": float(fl.quantile(.25)),
                                        "q75": float(fl.quantile(.75))}
    out["live_versions_per_match"] = {"median": float(first["version"].median()), "mean": float(first["version"].mean())}
    for cls in ("A1", "A2", "B", "C"):
        x = h.loc[h["price_class"] == cls]
        out[f"quote_after_gen_h_{cls}"] = {"n_known": int(x["quote_after_gen_h"].notna().sum()),
                                           "median": float(x["quote_after_gen_h"].median()) if len(x) else None,
                                           "q25": float(x["quote_after_gen_h"].quantile(.25)) if len(x) else None,
                                           "q75": float(x["quote_after_gen_h"].quantile(.75)) if len(x) else None}
        out[f"quote_lead_h_{cls}"] = {"median": float(x["quote_lead_h"].median()) if x["quote_lead_h"].notna().any() else None,
                                      "q25": float(x["quote_lead_h"].quantile(.25)) if x["quote_lead_h"].notna().any() else None,
                                      "q75": float(x["quote_lead_h"].quantile(.75)) if x["quote_lead_h"].notna().any() else None}
    # A1 consistency: a quote recorded at generation must not be published after generation
    a1 = h.loc[(h["price_class"] == "A1") & h["quote_utc"].notna()]
    out["A1_quote_after_generation"] = int((a1["quote_utc"] > a1["generated_at"]).sum())
    out["A1_with_quote_time"] = int(len(a1))
    # the same check on the RAW columns (UK local quote time vs UTC generation time) — what a naive check sees
    raw = h.loc[(h["price_class"] == "A1") & h["odds_priced_at"].notna()]
    out["A1_raw_uk_vs_utc_after_generation"] = int((raw["odds_priced_at"] > raw["generated_at"]).sum())
    fd = h.loc[(h["price_class"] == "A1") & (h["provider"] == "football_data") & h["odds_priced_at"].notna()]
    out["football_data_placeholder"] = {
        "n_with_time": int(len(fd)),
        "time_is_midnight": int((fd["odds_priced_at"].dt.hour.eq(0) & fd["odds_priced_at"].dt.minute.eq(0)).sum()),
        "shown_time_after_generation": int((fd["quote_utc"] > fd["generated_at"]).sum())}
    # backfill: nominal decision times
    back = h.loc[h["price_class"] == "C"]
    med = out["live_generation_lead_h"]["median"]
    out["backfill_closing_published_before"] = {
        "n": int(len(back)), "n_known": int(back["quote_lead_h"].notna().sum()),
        "share_ge_24h": float((back["quote_lead_h"] >= 24).mean()),
        "share_ge_median_live_lead": float((back["quote_lead_h"] >= med).mean()), "median_live_lead_h": med,
        "share_ge_1h": float((back["quote_lead_h"] >= 1).mean())}
    return out


def roi_tables(h: pd.DataFrame) -> dict:
    out = {}
    for role in ("main", "risk", "all"):
        r = h if role == "all" else h.loc[h["role"] == role]
        live = r.loc[r["source"] == "live"]
        res = {
            "current_all_sources": _roi(r.loc[r["price_class"] != "-"]),
            "current_live": _roi(live.loc[live["price_class"] != "-"]),
            "live_A1": _roi(live.loc[live["price_class"] == "A1"]),
            "live_A1_A2": _roi(live.loc[live["price_class"].isin(["A1", "A2"])]),
            "live_A2": _roi(live.loc[live["price_class"] == "A2"]),
            "live_B": _roi(live.loc[live["price_class"] == "B"]),
            "backfill_C": _roi(r.loc[r["price_class"] == "C"]),
        }
        # generation-time policy on the same live picks: a B pick has no price available at generation -> no bet
        a = live.loc[live["price_class"] != "-"]
        a = a.loc[a["status"].isin(WON_LOST) & a["profit"].notna()]
        gen_only = np.where(a["price_class"].isin(["A1", "A2"]), a["profit"], np.nan)
        res["live_diff_genonly_minus_current"] = C.diff_ci(a["profit"].to_numpy(float), gen_only)
        # decided picks with any price vs ROI on the strict set
        res["current_all_minus_live_A1"] = C.diff_ci(
            r.loc[(r["price_class"] != "-") & r["status"].isin(WON_LOST) & r["profit"].notna(), "profit"],
            live.loc[(live["price_class"] == "A1") & live["status"].isin(WON_LOST) & live["profit"].notna(), "profit"])
        out[role] = res
    # breakdowns of the strict live set
    live = h.loc[(h["source"] == "live") & (h["price_class"] != "-")]
    out["live_by_class_provider"] = {f"{c}|{p}": _roi(g) for (c, p), g in
                                     live.groupby(["price_class", live["provider"].fillna("—")]) if len(g) >= 5}
    out["live_by_class_group"] = {f"{c}|{g_}": _roi(g) for (c, g_), g in live.groupby(["price_class", "group"])
                                  if len(g) >= 5}
    return out


def ev_by_class(h: pd.DataFrame) -> dict:
    out = {}
    for (cls, role), g in h.loc[h["price_class"] != "-"].groupby(["price_class", "role"]):
        out[f"{cls}|{role}"] = {"n": int(len(g)), "mean_ev": float(g["ev"].mean()), "mean_odds": float(g["odds"].mean()),
                                "mean_prob": float(g["probability"].mean())}
    return out


def run() -> dict:
    q = Q.load()
    hist = load_history()
    hist = recover_quote_times(hist, q)
    hist["versions_match"] = verify_against_versions(hist)
    h = classify(hist)
    a1 = h.loc[h["price_class"] == "A1"]
    out = {"counts": counts(h), "timing": timing_stats(h), "roi": roi_tables(h), "ev": ev_by_class(h),
           "A1_version_check": {"verified_equal": int((a1["versions_match"] == 1).sum()),
                                "differs": int((a1["versions_match"] == 0).sum()),
                                "not_in_version_archive": int(a1["versions_match"].isna().sum())},
           "definitions": {k: v for k, v in CLASS_RU.items()}}
    # the restored prices: did any quote valid at generation exist for B picks (any OddsPapi bookmaker)?
    b = h.loc[h["price_class"] == "B", ["match_id", "market", "selection", "generated_at"]].copy()
    if not b.empty:
        qq = q.loc[q["match_id"].isin(b["match_id"])]
        when = b.rename(columns={"generated_at": "t"})
        hits = 0
        for book in qq["bookmaker"].unique():
            w = when.assign(bookmaker=book)
            hits_b = Q.price_at(qq, w)["odds_at"].notna()
            hits += int(hits_b.sum())
        out["B_any_quote_valid_at_generation"] = hits
    cols = ["pick_id", "match_id", "league", "role", "source", "group", "market", "selection", "probability", "odds",
            "ev", "status", "profit", "provider", "bookmaker", "odds_origin", "price_class", "generated_at",
            "kickoff_utc", "quote_utc", "quote_time_from", "quote_after_gen_h", "quote_lead_h", "gen_lead_h",
            "versions_match", "model_version"]
    h[cols].to_parquet(C.CACHE / "audit_picks.parquet")
    C.save_csv("audit_picks.csv", h[cols])
    C.save_json("audit.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("audit")
    import json
    r = run()
    print(json.dumps(C.clean(r["counts"]), indent=1)[:3000])
    print(json.dumps(C.clean(r["timing"]), indent=1)[:3000])
