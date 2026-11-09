"""Stability of the Main / Risk choice with respect to WHEN the price is taken.

1. Versions (2026/27 live): every pre-match refresh is stored (versions.parquet). How often does the Main / Risk of
   an earlier version differ from the final one, by lead time and by whether the model version changed?
2. History (Football-Data, walk-forward 2023/24–2025/26, v6.1 candidate tables): the production Main rule run
   with the Bet365 opening snapshot (what production attaches) and with the Bet365 closing snapshot (what a run
   just before kick-off would see). Share of different Mains, EV and ROI of each policy, paired by match.
3. 2026/27 live, archived matches: the stored Main / Risk (prices of generation time) vs the production rules re-run
   on the same stored probabilities with the closing OddsPapi prices of every selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import quotes as Q

LEAD_BUCKETS = [0, 6, 24, 48, 96, 1e9]
LEAD_RU = ["< 6 ч", "6–24 ч", "24–48 ч", "48–96 ч", "> 96 ч"]


# ------------------------------------------------------------------------------------------------ 1. versions
def versions() -> dict:
    v = pd.read_parquet(C.PRED_DIR / "versions.parquet",
                        columns=["match_id", "role", "market", "selection", "odds", "version", "generated_at",
                                 "kickoff", "model_version", "season"])
    v = v.loc[(v["season"] == C.LIVE) & v["role"].isin(["main", "risk"])].copy()
    v["kickoff_utc"] = C.uk_to_utc(v["kickoff"]).to_numpy()
    v["lead_h"] = (v["kickoff_utc"] - v["generated_at"]).dt.total_seconds() / 3600
    v = v.sort_values(["match_id", "role", "version"])
    final = v.drop_duplicates(["match_id", "role"], keep="last").set_index(["match_id", "role"])
    key = pd.MultiIndex.from_frame(v[["match_id", "role"]])
    v["final_sel"] = (final["market"] + "|" + final["selection"]).reindex(key).to_numpy()
    v["final_model"] = final["model_version"].reindex(key).to_numpy()
    v["final_version"] = final["version"].reindex(key).to_numpy()
    v["sel"] = v["market"] + "|" + v["selection"]
    earlier = v.loc[v["version"] < v["final_version"]].copy()
    earlier["changed"] = earlier["sel"] != earlier["final_sel"]
    earlier["same_model"] = earlier["model_version"] == earlier["final_model"]
    earlier["bucket"] = pd.cut(earlier["lead_h"], LEAD_BUCKETS, labels=LEAD_RU, right=False)
    out = {"matches": int(v["match_id"].nunique()), "by_role_bucket": {}, "by_role_model": {}}
    for (role, b), g in earlier.groupby(["role", "bucket"], observed=True):
        out["by_role_bucket"][f"{role}|{b}"] = {"n": int(len(g)), "changed": float(g["changed"].mean()),
                                                "changed_same_model": float(g.loc[g["same_model"], "changed"].mean())
                                                if g["same_model"].any() else None,
                                                "n_same_model": int(g["same_model"].sum())}
    for (role, sm), g in earlier.groupby(["role", "same_model"]):
        out["by_role_model"][f"{role}|{'same' if sm else 'other'}"] = {"n": int(len(g)), "changed": float(g["changed"].mean())}
    # consecutive versions of the same model: how often does the choice flip from one refresh to the next?
    v["prev_sel"] = v.groupby(["match_id", "role"])["sel"].shift()
    v["prev_model"] = v.groupby(["match_id", "role"])["model_version"].shift()
    cons = v.loc[v["prev_sel"].notna() & (v["prev_model"] == v["model_version"])]
    out["consecutive_same_model"] = {r: {"n": int(len(g)), "flip": float((g["sel"] != g["prev_sel"]).mean())}
                                     for r, g in cons.groupby("role")}
    # the first published version vs the final one (a reader who bet when the prediction first appeared)
    first = v.drop_duplicates(["match_id", "role"], keep="first")
    first = first.loc[first["version"] < first["final_version"]]
    out["first_vs_final"] = {r: {"n": int(len(g)), "changed": float((g["sel"] != g["final_sel"]).mean()),
                                 "median_lead_first_h": float(g["lead_h"].median())} for r, g in first.groupby("role")}
    out["first_vs_final_roi"] = _first_vs_final_roi(v)
    return out


def _first_vs_final_roi(v: pd.DataFrame) -> dict:
    """A reader who bets on the FIRST published version (its own recorded price) vs on the final version
    (its own recorded price), same matches; prices as recorded at each version's generation."""
    from src.markets.markets import settle_selection
    f = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                        columns=["match_id", "home_goals", "away_goals", "total_corners", "played"])
    f = f.loc[f["played"].astype(bool)].set_index("match_id")
    out = {}
    for role, g in v.groupby("role"):
        first = g.drop_duplicates("match_id", keep="first").set_index("match_id")
        last = g.drop_duplicates("match_id", keep="last").set_index("match_id")
        ids = [m for m in first.index if m in f.index and first.at[m, "version"] < last.at[m, "version"]]
        res = {}
        for tag, t in (("first", first), ("final", last)):
            prof = []
            for m in ids:
                r = f.loc[m]
                st = settle_selection(t.at[m, "market"], t.at[m, "selection"], r["home_goals"], r["away_goals"],
                                      r["total_corners"])
                o = t.at[m, "odds"]
                prof.append(np.nan if (pd.isna(o) or st is None) else (o - 1 if st == "WON" else (0.0 if st == "PUSH" else -1.0)))
            res[tag] = np.array(prof, float)
        both = np.isfinite(res["first"]) & np.isfinite(res["final"])
        out[role] = {"matches": int(len(ids)), "first": C.roi_summary(res["first"], n_boot=1000),
                     "final": C.roi_summary(res["final"], n_boot=1000),
                     "both_priced": int(both.sum()),
                     "diff_final_minus_first_same_matches": C.paired_diff_ci(np.where(both, res["first"], np.nan),
                                                                            np.where(both, res["final"], np.nan), 1000)}
    return out


# ------------------------------------------------------------------------------------------------ 2. history
def history_open_vs_close() -> dict:
    from experiments.v6_1.main import common as V61
    from experiments.v6_1.main import strategies as ST
    cols = ["match_id", "league", "season", "date", "mi", "market", "selection", "p", "p_push", "status", "group",
            "line", "o_b365", "o_b365_c", "q_ps_c_shin"]
    out = {}
    for s in (2023, 2024, 2025):
        d = pd.read_parquet(V61.CACHE / f"cand_{s}.parquet", columns=cols)
        # same universe for both policies: matches whose FD markets are priced at BOTH snapshots
        fdm = d["market"].isin(["1X2", "TOTAL_2.5"])
        ok = d.loc[fdm].assign(h=d["o_b365"].notna() & d["o_b365_c"].notna()).groupby("match_id")["h"].all()
        d = d.loc[d["match_id"].isin(ok.index[ok])].copy()
        # a Bet365 price exists at one snapshot only for an Asian line that moved: keep each snapshot's own price
        d["mi"] = pd.factorize(d["match_id"])[0]
        d = d.reset_index(drop=True)
        n = d["mi"].max() + 1
        res = {"matches": int(n)}
        picks = {}
        for name, col in (("open", "o_b365"), ("close", "o_b365_c")):
            idx = ST.choose(d, ST.CURRENT, odds_col=col)
            picks[name] = d.iloc[idx].assign(row=idx)
        po, pc = picks["open"].set_index("mi"), picks["close"].set_index("mi")
        common = po.index.intersection(pc.index)
        res["share_different_main"] = float((po.loc[common, "row"] != pc.loc[common, "row"]).mean())

        def per_match(p: pd.DataFrame, col: str) -> np.ndarray:
            prof = np.full(n, np.nan)
            o = p[col].to_numpy(float)
            st = p["status"].to_numpy()
            ok_ = np.isfinite(o) & np.isin(st, [0, 1])
            prof[p.index.to_numpy()[ok_]] = np.where(st[ok_] == 1, o[ok_] - 1, -1.0)
            return prof
        prof = {"open_choice@open": per_match(po, "o_b365"), "open_choice@close": per_match(po, "o_b365_c"),
                "close_choice@close": per_match(pc, "o_b365_c")}
        for k, v in prof.items():
            res[k] = C.roi_summary(v, n_boot=1000)
            p_ = po if k.startswith("open") else pc
            col = "o_b365" if k.endswith("@open") else "o_b365_c"
            o = p_[col].to_numpy(float)
            res[k]["mean_ev"] = float(np.nanmean(p_["p"].to_numpy() * o - (1 - p_["p_push"].to_numpy())))
            # CLV of the price taken against the de-margined Pinnacle close (no refund markets)
            q = p_["q_ps_c_shin"].to_numpy(float)
            res[k]["mean_clv_fair_ps"] = float(np.nanmean(np.where(p_["p_push"].to_numpy() == 0, o * q - 1, np.nan)))
        res["diff_close_choice_minus_open_choice"] = C.paired_diff_ci(prof["open_choice@open"],
                                                                      prof["close_choice@close"], 1000)
        res["diff_open_choice_close_minus_open_price"] = C.paired_diff_ci(prof["open_choice@open"],
                                                                          prof["open_choice@close"], 1000)
        out[str(s)] = res
        C.log.info("history open/close %s", s)
    return out


# ------------------------------------------------------------------------------------------------ 3. live re-run
def _priced_list(rows: pd.DataFrame, odds_col: str):
    from src.markets.odds import PricedSelection
    out = []
    for r in rows.itertuples(index=False):
        o = getattr(r, odds_col)
        out.append(PricedSelection(r.market, r.selection, float(r.probability), None if pd.isna(o) else float(o),
                                   None, None, 0.0 if pd.isna(r.push_probability) else float(r.push_probability)))
    return out


def live_rerun() -> dict:
    """The CURRENT production rules re-run on the stored probabilities of archived live matches in three price worlds:

    gen          the prices recorded at generation (markets.parquet; many selections unpriced then);
    close_same   the closing OddsPapi price, but only for selections that had a price at generation
                 (same coverage -> isolates the timing of the price);
    close_full   the closing price of every selection the archive has (timing + coverage).

    The stored picks were made by the rules in force at the time (several rule changes in October 2026), so the
    comparison uses the current rule in every world; the stored pick is kept for information."""
    from src.analytics.performance import unified_history
    from src.markets.markets import settle_selection
    from src.markets.recommendation import main_prediction, risk_prediction
    picks = pd.read_parquet(C.PRED_DIR / "picks.parquet")
    hist = unified_history(picks)
    live = hist.loc[(hist["season"] == C.LIVE) & (hist["source"] == "live") & hist["role"].isin(["main", "risk"])]
    q = Q.load()
    clos = Q.closing(q)
    ids = sorted(set(live["match_id"]) & set(clos["match_id"]))
    mk = pd.read_parquet(C.PRED_DIR / "markets.parquet",
                         columns=["match_id", "market", "selection", "group", "probability", "odds", "push_probability"])
    mk = mk.loc[mk["match_id"].isin(ids)].drop_duplicates(["match_id", "market", "selection"], keep="last")
    best = clos.sort_values("priced_at_utc").drop_duplicates(["match_id", "market", "selection"], keep="last")
    mk = mk.merge(best[["match_id", "market", "selection", "odds"]].rename(columns={"odds": "close_full"}),
                  on=["match_id", "market", "selection"], how="left")
    mk["close_same"] = np.where(mk["odds"].notna(), mk["close_full"], np.nan)
    f = pd.read_parquet(C.DATA / "processed" / "features.parquet",
                        columns=["match_id", "home_goals", "away_goals", "total_corners", "played"])
    f = f.loc[f["played"].astype(bool)].set_index("match_id")
    stored = live.drop_duplicates(["match_id", "role"]).set_index(["match_id", "role"])
    rows = []
    for mid, g in mk.groupby("match_id"):
        if mid not in f.index:
            continue
        r = f.loc[mid]
        world = {}
        for w in ("odds", "close_same", "close_full"):
            sels = _priced_list(g, w)
            m = main_prediction(sels)
            world[w] = {"main": m, "risk": risk_prediction(sels, m.pick if m else None)}
        close_px = g.set_index(["market", "selection"])["close_full"]
        for role in ("main", "risk"):
            row = {"match_id": mid, "role": role, "close_priced_share": float(g["close_full"].notna().mean()),
                   "gen_priced_share": float(g["odds"].notna().mean())}
            for w, tag in (("odds", "gen"), ("close_same", "same"), ("close_full", "full")):
                rec = world[w][role]
                if rec is None:
                    row[f"{tag}_sel"] = None
                    continue
                s_ = rec.pick
                row[f"{tag}_sel"] = f"{s_.market}|{s_.selection}"
                row[f"{tag}_odds"] = s_.odds
                row[f"{tag}_p"] = s_.probability
                row[f"{tag}_ev"] = s_.ev
                row[f"{tag}_close_odds"] = close_px.get((s_.market, s_.selection), np.nan)
                row[f"{tag}_result"] = settle_selection(s_.market, s_.selection, r["home_goals"], r["away_goals"],
                                                        r["total_corners"])
            if (mid, role) in stored.index:
                srow = stored.loc[(mid, role)]
                row["stored_sel"] = f"{srow['market']}|{srow['selection']}"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(C.CACHE / "live_rerun.parquet")

    def prof(odds, result):
        o = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(float)
        res = np.asarray(result, dtype=object)
        return np.where((res == "WON") & np.isfinite(o), o - 1,
                        np.where((res == "LOST") & np.isfinite(o), -1.0,
                                 np.where((res == "PUSH") & np.isfinite(o), 0.0, np.nan)))
    out = {"matches": int(df["match_id"].nunique()),
           "gen_priced_share": float(df.drop_duplicates("match_id")["gen_priced_share"].mean()),
           "close_priced_share": float(df.drop_duplicates("match_id")["close_priced_share"].mean())}
    for role, g in df.groupby("role"):
        p_gen = prof(g["gen_odds"], g["gen_result"])
        p_gen_at_close = prof(g["gen_close_odds"], g["gen_result"])
        p_same = prof(g["same_odds"], g["same_result"])
        p_full = prof(g["full_odds"], g["full_result"])
        out[role] = {"n": int(len(g)),
                     "rerun_gen_equals_stored": float((g["gen_sel"] == g["stored_sel"]).mean()),
                     "changed_gen_vs_close_same": float((g["gen_sel"] != g["same_sel"]).mean()),
                     "changed_gen_vs_close_full": float((g["gen_sel"] != g["full_sel"]).mean()),
                     "gen@gen_price": C.roi_summary(p_gen, n_boot=1000),
                     "gen_choice@close_price": C.roi_summary(np.where(np.isfinite(p_gen), p_gen_at_close, np.nan),
                                                            n_boot=1000),
                     "close_same@close": C.roi_summary(p_same, n_boot=1000),
                     "close_full@close": C.roi_summary(p_full, n_boot=1000),
                     "mean_ev_gen": float(np.nanmean(pd.to_numeric(g["gen_ev"], errors="coerce"))),
                     "mean_ev_close_same": float(np.nanmean(pd.to_numeric(g["same_ev"], errors="coerce"))),
                     "mean_ev_close_full": float(np.nanmean(pd.to_numeric(g["full_ev"], errors="coerce"))),
                     "diff_close_same_minus_gen": C.paired_diff_ci(p_gen, p_same, 1000),
                     "diff_gen_choice_close_minus_gen_price": C.paired_diff_ci(
                         p_gen, np.where(np.isfinite(p_gen), p_gen_at_close, np.nan), 1000)}
    return out


def run() -> dict:
    out = {"versions": versions(), "history": history_open_vs_close(), "live_rerun": live_rerun()}
    C.save_json("stability.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("stability")
    run()
