"""Audit of the xG data (step 1 of the study).

Writes results/audit.json, results/audit_by_league.csv, results/audit_by_week.csv and figures/audit_*.png.
Nothing here is a model input; post-match xG is only described.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import data as D


def _league_comparability(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, g in x.groupby("league"):
        xg = np.r_[g["xg_home"], g["xg_away"]]
        gl = np.r_[g["home_goals"], g["away_goals"]]
        n = len(g)
        # bootstrap the goals / xG ratio by match
        ratios = []
        for idx in C.boot_index(n, 1000, C.SEED):
            num = g["home_goals"].to_numpy()[idx].sum(1) + g["away_goals"].to_numpy()[idx].sum(1)
            den = g["xg_home"].to_numpy()[idx].sum(1) + g["xg_away"].to_numpy()[idx].sum(1)
            ratios.append(num / den)
        ratios = np.concatenate(ratios)
        mean_mu = np.full_like(gl, gl.mean(), dtype=float)
        dev_xg = C.poisson_dev(gl, np.clip(xg, 0.05, None)).mean()
        dev_mean = C.poisson_dev(gl, mean_mu).mean()
        slope = np.polyfit(xg, gl, 1)
        rows.append({"league": lg, "matches": n, "xg_home": g["xg_home"].mean(), "xg_away": g["xg_away"].mean(),
                     "goals_home": g["home_goals"].mean(), "goals_away": g["away_goals"].mean(),
                     "xg_total": (g["xg_home"] + g["xg_away"]).mean(),
                     "goals_total": (g["home_goals"] + g["away_goals"]).mean(),
                     "ratio": gl.sum() / xg.sum(), "ratio_lo": np.quantile(ratios, 0.025),
                     "ratio_hi": np.quantile(ratios, 0.975), "corr": np.corrcoef(xg, gl)[0, 1],
                     "slope": slope[0], "intercept": slope[1],
                     "dev_xg": dev_xg, "dev_mean": dev_mean, "pseudo_r2": 1 - dev_xg / dev_mean,
                     "decimals": int(pd.Series(xg).map(lambda v: len(f"{v:.6g}".split(".")[1]) if "." in f"{v:.6g}" else 0).max()),
                     "share_zero": float(np.mean(xg == 0)),
                     "home_share_xg": g["xg_home"].sum() / xg.sum(), "home_share_goals": g["home_goals"].sum() / max(gl.sum(), 1)})
    return pd.DataFrame(rows)


def run() -> dict:
    t = C.Timer()
    cols = D.xg_columns_by_season()
    by_season = cols.groupby("season_code").agg(files=("league", "size"), with_xg=("has_hxg", "sum")).reset_index()
    raw = D.raw_xg("2627")
    enabled = D.enabled_leagues()
    feats = D.season_features(columns=["match_id", "league", "date", "played", "home_goals", "away_goals"])
    played = feats.loc[feats["played"].astype(bool)]
    xr = raw.loc[raw["league"].isin(enabled)]

    # ---- anomalies
    both = xr["xg_home"].notna() & xr["xg_away"].notna()
    one = xr["xg_home"].notna() ^ xr["xg_away"].notna()
    has_col = xr["has_xg_column"]
    nonnum = has_col & (xr["xg_home"].isna() & ~xr["xg_home_raw"].isin(["nan", "None", ""]))
    anomalies = {
        "rows_in_files": int(len(xr)),
        "rows_in_leagues_with_xg_column": int(has_col.sum()),
        "both_xg_present": int(both.sum()),
        "only_one_side_present": int(one.sum()),
        "non_numeric_values": int(nonnum.sum()),
        "negative": int(((xr["xg_home"] < 0) | (xr["xg_away"] < 0)).sum()),
        "above_5": int(((xr["xg_home"] > 5) | (xr["xg_away"] > 5)).sum()),
        "above_8_impossible": int(((xr["xg_home"] > 8) | (xr["xg_away"] > 8)).sum()),
        "exact_zero": int(((xr["xg_home"] == 0) | (xr["xg_away"] == 0)).sum()),
        "total_above_7": int(((xr["xg_home"] + xr["xg_away"]) > 7).sum()),
        "xg_but_no_goals": int((both & ~xr["played"].astype(bool)).sum()),
        "goals_but_no_xg_in_xg_leagues": int((has_col & xr["played"].astype(bool) & ~both).sum()),
        "duplicate_match_ids": int(xr["match_id"].duplicated().sum()),
        "goal_minus_xg_above_4": int((((xr["home_goals"] - xr["xg_home"]).abs() > 4)
                                      | ((xr["away_goals"] - xr["xg_away"]).abs() > 4)).sum()),
        "xg_without_shots": int((both & (xr["home_shots"].isna() | xr["away_shots"].isna())).sum()),
        "xg_positive_with_zero_shots": int((both & (((xr["home_shots"] == 0) & (xr["xg_home"] > 0))
                                                    | ((xr["away_shots"] == 0) & (xr["xg_away"] > 0)))).sum()),
        "zero_xg_with_goals": int((both & (((xr["xg_home"] == 0) & (xr["home_goals"] > 0))
                                           | ((xr["xg_away"] == 0) & (xr["away_goals"] > 0)))).sum()),
        "sot_exceeds_shots": int((xr["home_sot"] > xr["home_shots"]).sum() + (xr["away_sot"] > xr["away_shots"]).sum()),
        "goals_exceed_sot": int((both & ((xr["home_goals"] > xr["home_sot"]) | (xr["away_goals"] > xr["away_sot"]))).sum()),
    }
    ext = xr.loc[both & ((xr["xg_home"] > 4.5) | (xr["xg_away"] > 4.5)),
                 ["match_id", "xg_home", "xg_away", "home_goals", "away_goals", "home_shots", "away_shots",
                  "home_sot", "away_sot"]].sort_values("xg_home", ascending=False)

    # ---- coverage by league
    x = D.clean_xg(raw)
    lg_rows = []
    for lg in enabled:
        f = played.loc[played["league"] == lg]
        r = xr.loc[xr["league"] == lg]
        lg_rows.append({"league": lg, "has_xg_column": bool(r["has_xg_column"].any()) if len(r) else False,
                        "played_in_system": int(len(f)), "rows_in_file": int(len(r)),
                        "with_xg": int(f["match_id"].isin(x["match_id"]).sum()),
                        "fill_rate": float(f["match_id"].isin(x["match_id"]).mean()) if len(f) else np.nan,
                        "last_played": str(f["date"].max().date()) if len(f) else None,
                        "last_xg": str(x.loc[x["league"] == lg, "date"].max().date()) if (x["league"] == lg).any() else None})
    by_league = pd.DataFrame(lg_rows)
    xg_leagues = sorted(by_league.loc[by_league["has_xg_column"], "league"])

    # ---- coverage by week (played matches of the xG leagues)
    pl = played.loc[played["league"].isin(xg_leagues)].copy()
    pl["week"] = pl["date"].dt.to_period("W-SUN").dt.start_time
    pl["with_xg"] = pl["match_id"].isin(x["match_id"])
    by_week = pl.groupby("week").agg(played=("match_id", "size"), with_xg=("with_xg", "sum")).reset_index()
    by_week["fill_rate"] = by_week["with_xg"] / by_week["played"]
    by_week["week"] = by_week["week"].dt.date.astype(str)

    comp = _league_comparability(x)
    allx = np.r_[x["xg_home"], x["xg_away"]]
    allg = np.r_[x["home_goals"], x["away_goals"]]
    overall = {"matches_with_xg": int(len(x)), "teams_matches": int(len(allx)), "mean_xg": float(allx.mean()),
               "mean_goals": float(allg.mean()), "ratio": float(allg.sum() / allx.sum()),
               "corr": float(np.corrcoef(allx, allg)[0, 1]),
               "xg_home_mean": float(x["xg_home"].mean()), "xg_away_mean": float(x["xg_away"].mean()),
               "goals_home_mean": float(x["home_goals"].mean()), "goals_away_mean": float(x["away_goals"].mean()),
               "first_date": str(x["date"].min().date()), "last_date": str(x["date"].max().date()),
               "xg_quantiles": {str(q): float(np.quantile(allx, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
               "ratio_spread_leagues": [float(comp["ratio"].min()), float(comp["ratio"].max())]}
    # heterogeneity of the goals / xG ratio across leagues: Poisson chi-square of goals given scaled xG
    from scipy.stats import chi2
    exp_goals = comp["matches"] * comp["xg_total"] * overall["ratio"]
    obs_goals = comp["matches"] * comp["goals_total"]
    chi = float((((obs_goals - exp_goals) ** 2) / exp_goals).sum())
    overall["ratio_heterogeneity"] = {"chi2": chi, "df": int(len(comp) - 1), "p": float(chi2.sf(chi, len(comp) - 1))}

    # ---- xG versus shots (how much of xG the existing shot features already carry)
    s = x.dropna(subset=["home_shots", "away_shots", "home_sot", "away_sot"])
    shots = np.r_[s["home_shots"], s["away_shots"]]
    sot = np.r_[s["home_sot"], s["away_sot"]]
    xs = np.r_[s["xg_home"], s["xg_away"]]
    gs = np.r_[s["home_goals"], s["away_goals"]]
    A = np.c_[np.ones_like(shots), shots, sot]
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    r2 = 1 - np.sum((xs - A @ coef) ** 2) / np.sum((xs - xs.mean()) ** 2)
    shots_vs_xg = {"n_team_matches": int(len(xs)), "coef": coef.tolist(), "r2_xg_on_shots_sot": float(r2),
                   "corr_goals_xg": float(np.corrcoef(gs, xs)[0, 1]),
                   "corr_goals_sot": float(np.corrcoef(gs, sot)[0, 1]),
                   "corr_goals_shots": float(np.corrcoef(gs, shots)[0, 1])}

    # ---- pre-match model expectation (B0) versus the match's xG and goals
    b = D.baseline()
    m = x.merge(b[["match_id", "lh", "la"]], on="match_id")
    lam = np.r_[m["lh"], m["la"]]
    mx = np.r_[m["xg_home"], m["xg_away"]]
    mg = np.r_[m["home_goals"], m["away_goals"]]
    b0_vs_xg = {"n_team_matches": int(len(lam)), "corr_lambda_xg": float(np.corrcoef(lam, mx)[0, 1]),
                "corr_lambda_goals": float(np.corrcoef(lam, mg)[0, 1]), "mean_lambda": float(lam.mean()),
                "mean_xg": float(mx.mean()), "mean_goals": float(mg.mean()),
                "dev_goals_given_lambda": float(C.poisson_dev(mg, lam).mean()),
                "dev_goals_given_xg": float(C.poisson_dev(mg, np.clip(mx, 0.05, None)).mean())}

    out = {"seasons": by_season.to_dict("records"), "xg_leagues": xg_leagues,
           "no_xg_leagues": [lg for lg in enabled if lg not in xg_leagues], "anomalies": anomalies,
           "extreme_examples": ext.head(12).to_dict("records"), "overall": overall, "shots_vs_xg": shots_vs_xg,
           "b0_vs_xg": b0_vs_xg, "played_in_system_2026": int(len(played)),
           "played_in_xg_leagues": int(len(pl)), "played_with_xg": int(pl["with_xg"].sum()),
           "minutes": t.minutes()}
    C.save_json("audit.json", out)
    C.save_csv("audit_by_league.csv", by_league.merge(comp, on="league", how="left"))
    C.save_csv("audit_by_week.csv", by_week)
    figures(by_week, comp, x)
    return out


def figures(by_week: pd.DataFrame, comp: pd.DataFrame, x: pd.DataFrame) -> None:
    plt = C.plt()
    from scripts.report_kit import NEG, PALETTE, save_chart
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = np.arange(len(by_week))
    ax.bar(w, by_week["played"], color="#c9d1cd", label="сыграно (лиги с xG)")
    ax.bar(w, by_week["with_xg"], color=PALETTE[0], label="с xG в файле")
    ax.set_xticks(w)
    ax.set_xticklabels(by_week["week"], rotation=60, fontsize=7)
    ax.set_ylabel("Матчей за неделю")
    ax.legend(fontsize=8)
    ax.set_title("Наличие xG по неделям 2026/27 (18 лиг с колонками HxG/AxG)", fontsize=10)
    save_chart(fig, C.FIGURES / "audit_by_week.png")

    c = comp.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(c))
    ax.errorbar(c["ratio"], y, xerr=[c["ratio"] - c["ratio_lo"], c["ratio_hi"] - c["ratio"]], fmt="o",
                color=PALETTE[0], ms=4, capsize=2)
    ax.axvline(1.0, color=NEG, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(c["league"])
    ax.set_xlabel("Голы / xG (95 % бутстреп-интервал)")
    ax.set_title("Сопоставимость xG между лигами: отношение голов к xG", fontsize=10)
    save_chart(fig, C.FIGURES / "audit_ratio_by_league.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    xs = np.r_[x["xg_home"], x["xg_away"]]
    gs = np.r_[x["home_goals"], x["away_goals"]]
    bins = np.r_[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 9.0]
    idx = np.digitize(xs, bins) - 1
    mx = [xs[idx == k].mean() for k in range(len(bins) - 1) if (idx == k).sum() >= 10]
    mg = [gs[idx == k].mean() for k in range(len(bins) - 1) if (idx == k).sum() >= 10]
    se = [gs[idx == k].std() / np.sqrt((idx == k).sum()) for k in range(len(bins) - 1) if (idx == k).sum() >= 10]
    ax.errorbar(mx, mg, yerr=1.96 * np.array(se), marker="o", color=PALETTE[0], capsize=2, label="средние голы в корзине xG")
    ax.plot([0, 4.5], [0, 4.5], color="#888", ls="--", lw=0.8, label="голы = xG")
    ax.set_xlabel("xG команды в матче")
    ax.set_ylabel("Голы команды")
    ax.legend(fontsize=8)
    ax.set_title("xG и голы (одна команда в одном матче)", fontsize=10)
    save_chart(fig, C.FIGURES / "audit_goals_vs_xg.png")


if __name__ == "__main__":
    C.setup_logging("audit")
    run()
