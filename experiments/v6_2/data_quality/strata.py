"""Does data quality change the quality of predictions and the result of Main at real prices?

Matches of 2023/24–2026/27 (out-of-sample predictions, predictions.py) are split by
* the history the system has for the clubs (teams.py): a club from outside our leagues (no history at all),
  a club returning after a season away, a club promoted / relegated inside our leagues, established clubs;
* the club's match number in the season (the start of a season leaves season features empty for everyone,
  so newcomers are compared with established clubs at the same match number);
* a per-match data-quality score: the share of missing result-model features (both sides).

Measured per group: 1X2 log loss, the same against the bookmaker's de-margined Bet365 probabilities
(excess log loss = model − market, so harder matches do not look like worse data), the bias of the expected
points of the newcomer, and Main: hit rate and ROI at the real Football-Data Bet365 price. Intervals: 95 %
bootstrap over matches. The 2026/27 production record (unified history, recorded prices) is split the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.performance import unified_history
from src.features.sets import model_features

from . import common as C
from . import predictions as PR

PRIORITY = ["outside", "returner", "moved_within", "established"]
BUCKETS = [0, 3, 5, 10, 20, 60]
BUCKET_LABELS = ["1-3", "4-5", "6-10", "11-20", "21+"]
DQ_EDGES = [-0.001, 0.0005, 0.05, 0.2, 0.5, 1.0]
DQ_LABELS = ["0 %", "0–5 %", "5–20 %", "20–50 %", "> 50 %"]


def frame() -> pd.DataFrame:
    pred = PR.all_predictions()
    ms = pd.read_parquet(C.CACHE / "team_status.parquet")
    f = PR.all_features()
    num, _ = model_features("result", list(f.columns))
    q = f[["match_id"]].copy()
    q["dq_missing"] = f[num].isna().mean(axis=1).to_numpy()
    q["h_missing"] = f[[c for c in num if c.startswith("h_")]].isna().mean(axis=1).to_numpy()
    q["a_missing"] = f[[c for c in num if c.startswith("a_")]].isna().mean(axis=1).to_numpy()
    x = pred.merge(ms, on="match_id", how="left").merge(q, on="match_id", how="left")
    rank = {s: i for i, s in enumerate(PRIORITY)}
    worst = [min((h, a), key=lambda s: rank.get(s, 9)) for h, a in zip(x["h_status"], x["a_status"])]
    x["group"] = worst
    nh = np.where(x["h_status"] == x["group"], x["h_season_match_no"], np.inf)
    na = np.where(x["a_status"] == x["group"], x["a_season_match_no"], np.inf)
    x["group_match_no"] = np.minimum(nh, na)
    x["early"] = x["group_match_no"] <= 10
    x["min_match_no"] = np.minimum(x["h_season_match_no"], x["a_season_match_no"])
    x["bucket"] = pd.cut(x["min_match_no"], BUCKETS, labels=BUCKET_LABELS)
    x["dq_bin"] = pd.cut(x["dq_missing"], DQ_EDGES, labels=DQ_LABELS)
    x["excess_ll"] = x["ll"] - x["ll_market"]
    return x


def _metrics(g: pd.DataFrame) -> dict:
    return {"matches": int(len(g)), "log_loss": C.mean_ci(g["ll"]), "market_log_loss": C.mean_ci(g["ll_market"]),
            "excess_log_loss": C.mean_ci(g["excess_ll"]),
            "main": C.roi_summary(g["main_profit"], g["main_won"])}


def newcomer_side_bias(x: pd.DataFrame, status: str) -> pd.DataFrame:
    """Per match and newcomer side: predicted win probability and expected points vs what happened."""
    rows = []
    for s, ps_win, ps_lose in (("h", "p_home", "p_away"), ("a", "p_away", "p_home")):
        sel = x.loc[x[f"{s}_status"] == status]
        gf = sel["home_goals"] if s == "h" else sel["away_goals"]
        ga = sel["away_goals"] if s == "h" else sel["home_goals"]
        pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
        exp_pts = 3 * sel[ps_win] + sel["p_draw"]
        rows.append(pd.DataFrame({"match_id": sel["match_id"], "season": sel["season"], "side": s,
                                  "match_no": sel[f"{s}_season_match_no"], "p_win": sel[ps_win],
                                  "won": (gf > ga).astype(float), "points": pts, "exp_points": exp_pts,
                                  "market_p_win": (1 / sel["odds_home" if s == "h" else "odds_away"])
                                  / (1 / sel["odds_home"] + 1 / sel["odds_draw"] + 1 / sel["odds_away"])}))
    out = pd.concat(rows, ignore_index=True)
    out["bias_points"] = out["exp_points"] - out["points"]
    out["bucket"] = pd.cut(out["match_no"], BUCKETS, labels=BUCKET_LABELS)
    return out


def live_record(x: pd.DataFrame) -> dict:
    """2026/27 production record (unified history, recorded or archive prices) split by club history."""
    import pyarrow.parquet as pq
    cols = [c for c in ("pick_id", "match_id", "league", "home_team", "away_team", "date", "role", "status", "odds",
                        "profit", "source", "generated_at", "home_goals", "away_goals", "market", "selection")
            if c in pq.ParquetFile(C.PRED_DIR / "picks.parquet").schema.names]
    pk = pd.read_parquet(C.PRED_DIR / "picks.parquet", columns=cols)
    uh = unified_history(pk)
    main = uh.loc[uh["role"] == "main"]
    ms = pd.read_parquet(C.CACHE / "team_status.parquet")
    j = main.merge(ms, on="match_id", how="inner")
    rank = {s: i for i, s in enumerate(PRIORITY)}
    j["group"] = [min((h, a), key=lambda s: rank.get(s, 9)) for h, a in zip(j["h_status"], j["a_status"])]
    j["won"] = np.where(j["status"] == "WON", 1.0, np.where(j["status"] == "LOST", 0.0, np.nan))
    j["bet_profit"] = np.where(j["status"].isin(["WON", "LOST"]) & j["odds"].notna(), j["profit"], np.nan)
    out = {"matched": int(len(j)), "unified_mains": int(len(main))}
    for name, g in j.groupby("group"):
        out[name] = C.roi_summary(g["bet_profit"], g["won"])
    newc = j["group"].isin(["outside", "returner"])
    out["newcomer_vs_rest_roi"] = C.group_diff_ci(j.loc[~newc, "bet_profit"], j.loc[newc, "bet_profit"])
    return out


def run() -> dict:
    x = frame()
    x.to_parquet(C.CACHE / "strata_frame.parquet")
    out = {"matches": int(len(x)), "seasons": list(C.EVAL_SEASONS)}
    # 1) club history groups (whole season and first ten matches of the club)
    grp = {}
    for (g, early), part in x.groupby(["group", "early"]):
        grp[f"{g}|{'1-10' if early else '11+'}"] = _metrics(part)
    for g, part in x.groupby("group"):
        grp[f"{g}|all"] = _metrics(part)
    out["groups"] = grp
    est = x.loc[x["group"] == "established"]
    comp = {}
    for g in ("outside", "returner", "moved_within"):
        for early in (True, False):
            a = est.loc[est["min_match_no"] <= 10] if early else est.loc[est["min_match_no"] > 10]
            b = x.loc[(x["group"] == g) & (x["early"] == early)]
            comp[f"{g}|{'1-10' if early else '11+'}"] = {
                "excess_log_loss_diff": C.group_diff_ci(a["excess_ll"], b["excess_ll"]),
                "log_loss_diff": C.group_diff_ci(a["ll"], b["ll"]),
                "main_roi_diff": C.group_diff_ci(a["main_profit"], b["main_profit"]),
                "main_hit_diff": C.group_diff_ci(a["main_won"], b["main_won"])}
    out["vs_established"] = comp
    pv = {k: v["excess_log_loss_diff"].get("p_b_greater") for k, v in comp.items()}
    out["holm_excess_ll"] = C.holm({k: (2 * min(p, 1 - p) if p is not None else None) for k, p in pv.items()})
    # 2) by season match number: outside newcomers vs established-only matches
    rows = []
    for b, part in x.groupby("bucket", observed=True):
        for g in ("established", "outside", "returner", "moved_within"):
            sel = part.loc[part["group"] == g]
            if len(sel) < 5:
                continue
            m = C.mean_ci(sel["excess_ll"])
            rows.append({"bucket": str(b), "group": g, "matches": int(len(sel)), "log_loss": float(sel["ll"].mean()),
                         "excess_log_loss": m["mean"], "lo": m["ci95"][0], "hi": m["ci95"][1],
                         "main_hit": float(sel["main_won"].mean()), "main_roi": float(np.nanmean(sel["main_profit"]))
                         if sel["main_profit"].notna().any() else None,
                         "bets": int(sel["main_profit"].notna().sum())})
    C.save_csv("strata_by_match_no.csv", pd.DataFrame(rows))
    # 3) bias of newcomers' expected points
    bias_rows, bias = [], {}
    for status in ("outside", "returner", "moved_within", "established"):
        b = newcomer_side_bias(x, status)
        bias[status] = {"sides": int(len(b)), "points_bias": C.mean_ci(b["bias_points"]),
                        "p_win_mean": float(b["p_win"].mean()), "won_rate": float(b["won"].mean()),
                        "market_p_win_mean": float(b["market_p_win"].mean()),
                        "early_points_bias": C.mean_ci(b.loc[b["match_no"] <= 10, "bias_points"]),
                        "late_points_bias": C.mean_ci(b.loc[b["match_no"] > 10, "bias_points"])}
        for bk, part in b.groupby("bucket", observed=True):
            m = C.mean_ci(part["bias_points"])
            bias_rows.append({"status": status, "bucket": str(bk), "sides": int(len(part)), "points_bias": m["mean"],
                              "lo": m["ci95"][0], "hi": m["ci95"][1], "p_win": float(part["p_win"].mean()),
                              "won": float(part["won"].mean()), "market_p_win": float(part["market_p_win"].mean())})
    out["points_bias"] = bias
    C.save_csv("strata_points_bias.csv", pd.DataFrame(bias_rows))
    # 4) data-quality score bins
    dq_rows = []
    for b, part in x.groupby("dq_bin", observed=True):
        m = _metrics(part)
        dq_rows.append({"dq_bin": str(b), "matches": m["matches"], "log_loss": m["log_loss"]["mean"],
                        "excess_log_loss": m["excess_log_loss"]["mean"], "excess_lo": m["excess_log_loss"]["ci95"][0],
                        "excess_hi": m["excess_log_loss"]["ci95"][1], "main_hit": m["main"].get("hit_rate"),
                        "bets": m["main"]["bets"], "main_roi": m["main"]["roi"], "roi_lo": m["main"]["roi_ci95"][0],
                        "roi_hi": m["main"]["roi_ci95"][1]})
    C.save_csv("strata_dq_bins.csv", pd.DataFrame(dq_rows))
    clean = x.loc[x["dq_missing"] <= 0.0005]
    poor = x.loc[x["dq_missing"] > 0.2]
    out["dq_poor_vs_clean"] = {"excess_log_loss_diff": C.group_diff_ci(clean["excess_ll"], poor["excess_ll"]),
                               "main_roi_diff": C.group_diff_ci(clean["main_profit"], poor["main_profit"]),
                               "main_hit_diff": C.group_diff_ci(clean["main_won"], poor["main_won"]),
                               "share_poor": float(len(poor) / len(x))}
    out["dq_spearman_excess_ll"] = float(pd.Series(x["dq_missing"]).corr(x["excess_ll"], method="spearman"))
    # within the first matches only (the season start is when data are thin for everybody)
    early = x.loc[x["min_match_no"] <= 5]
    ec, ep = early.loc[early["dq_missing"] <= 0.2], early.loc[early["dq_missing"] > 0.5]
    out["dq_within_first5"] = {"excess_log_loss_diff": C.group_diff_ci(ec["excess_ll"], ep["excess_ll"]),
                               "main_roi_diff": C.group_diff_ci(ec["main_profit"], ep["main_profit"])}
    # 5) per season, share of matches with an outside / returning club among their first ten matches
    out["share_newcomer_early_by_season"] = {int(s): float(((g["group"].isin(["outside", "returner"])) & g["early"]).mean())
                                             for s, g in x.groupby("season")}
    out["live_record"] = live_record(x)
    # 6) calibration of P(win) of outside newcomers vs established clubs
    cal = []
    for status in ("outside", "established"):
        b = newcomer_side_bias(x, status)
        b["bin"] = pd.cut(b["p_win"], [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
        for k, part in b.groupby("bin", observed=True):
            if len(part) >= 15:
                cal.append({"status": status, "bin": str(k), "sides": int(len(part)), "predicted": float(part["p_win"].mean()),
                            "observed": float(part["won"].mean()), "market": float(part["market_p_win"].mean())})
    C.save_csv("strata_calibration.csv", pd.DataFrame(cal))
    C.save_json("strata.json", out)
    return out


if __name__ == "__main__":
    C.setup_logging("strata")
    run()
