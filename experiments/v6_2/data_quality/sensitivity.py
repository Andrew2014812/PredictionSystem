"""Controlled experiments: what the production models do when part of ONE team's features is missing.

For every played match of 2023/24–2026/27 the features of one side (home or away) are blanked group by group
the way the feature builder leaves them for a club without that history, the derived features are recomputed
with the builder's own formulas (src/features/builder.FeatureBuilder._derived_features), and the match is
predicted again with the same models and the same real prices (predictions.predict). Compared with the
untouched prediction of the same match: change of the 1X2 probabilities and goal rates, change of log loss,
how often Main changes, Main hit rate and ROI.

Groups (side s = h or a):
    season     s_season_*, s_points, s_position*, s_venue_* ; s_season_games = s_venue_games = 0
    form       s_form*_*, s_venue_form5_*, s_freq_*, s_trend_* ; s_form{5,10,15}_n = 0
    stats      every shots / shots-on-target / corners feature of the side (goals and points kept)
    elo        s_elo set to the base rating of the league's tier (what a club unknown to the system gets)
    rest       s_rest_days missing
    newcomer   all of the above + no head-to-head: exactly the features of a club the system has never seen
    h2h        head-to-head of the pair missing (both sides)

The phantom predictions of 03–04.11.2026 are reproduced the same way: the real match row with the misspelt
side turned into a newcomer, predicted with the production bundle, compared with the stored phantom snapshot.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.builder import FeatureBuilder
from src.leagues import get_league

from . import common as C
from . import predictions as PR

SIDES = ("h", "a")
GROUPS = ("season", "form", "stats", "elo", "rest", "newcomer")
DERIVED = ["diff_season_ppg", "diff_position", "diff_points", "diff_season_gd", "diff_venue_ppg", "diff_form5_ppg",
           "diff_form10_ppg", "diff_form10_gd", "diff_form10_shots", "diff_form10_sot", "diff_form10_corners",
           "diff_rest_days", "h_attack_vs_a_defence", "a_attack_vs_h_defence", "h_form_attack_vs_a_form_defence",
           "a_form_attack_vs_h_form_defence", "h_sot_vs_a_sot_conceded", "a_sot_vs_h_sot_conceded",
           "h_corners_vs_a_corners_conceded", "a_corners_vs_h_corners_conceded", "form_corners_sum"]


def _cols(df: pd.DataFrame, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [c for c in df.columns if rx.search(c)]


def blank(df: pd.DataFrame, side: str | None, group: str) -> pd.DataFrame:
    x = df.copy()
    s = side
    if group in ("season", "newcomer"):
        cols = _cols(x, rf"^{s}_(season_|points$|position|venue_(?!form)|trend_)")
        x[cols] = np.nan
        x[f"{s}_season_games"] = 0.0
        x[f"{s}_venue_games"] = 0.0
    if group in ("form", "newcomer"):
        cols = _cols(x, rf"^{s}_(form\d+_|venue_form5_|freq_|trend_)")
        x[cols] = np.nan
        for k in (5, 10, 15):
            x[f"{s}_form{k}_n"] = 0.0
    if group == "stats":
        cols = _cols(x, rf"^{s}_.*(shot|sot|corner)")
        x[cols] = np.nan
    if group in ("elo", "newcomer"):
        x[f"{s}_elo"] = x["league"].map(lambda lg: 1500.0 - 100.0 * (get_league(lg).tier - 1))
    if group in ("rest", "newcomer"):
        x[f"{s}_rest_days"] = np.nan
    if group in ("h2h", "newcomer"):
        x[_cols(x, r"^h2h_")] = np.nan
        x["h2h_n"] = 0.0
    if group in ("elo", "newcomer"):
        x["elo_diff"] = x["h_elo"] + 60.0 - x["a_elo"]
        x["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(x["elo_diff"]) / 400.0))
    src = [c for c in x.columns if c.startswith(("h_", "a_"))]
    rows = x[src].to_dict("records")
    derived = pd.DataFrame([FeatureBuilder._derived_features({k: v for k, v in r.items() if v == v}) for r in rows],
                           index=x.index)
    x[DERIVED] = derived[DERIVED]
    return x


def _summary(base: pd.DataFrame, var: pd.DataFrame) -> dict:
    pb = base[["p_home", "p_draw", "p_away"]].to_numpy(float)
    pv = var[["p_home", "p_draw", "p_away"]].to_numpy(float)
    tv = 0.5 * np.abs(pv - pb).sum(1)
    main_changed = ((base["main_market"].fillna("") != var["main_market"].fillna(""))
                    | (base["main_selection"].fillna("") != var["main_selection"].fillna("")))
    out = {"matches": int(len(base)), "mean_tv": float(tv.mean()), "p90_tv": float(np.quantile(tv, 0.9)),
           "mean_dp_home": float((pv[:, 0] - pb[:, 0]).mean()), "mean_dp_draw": float((pv[:, 1] - pb[:, 1]).mean()),
           "mean_dp_away": float((pv[:, 2] - pb[:, 2]).mean()),
           "mean_dxg_home": float((var["xg_home"] - base["xg_home"]).mean()),
           "mean_dxg_away": float((var["xg_away"] - base["xg_away"]).mean()),
           "mean_abs_dxg": float(((var["xg_home"] - base["xg_home"]).abs()
                                  + (var["xg_away"] - base["xg_away"]).abs()).mean() / 2),
           "log_loss_base": float(base["ll"].mean()), "log_loss_var": float(var["ll"].mean()),
           "d_log_loss": C.paired_diff_ci(base["ll"], var["ll"]),
           "main_changed": float(main_changed.mean()),
           "main_base": C.roi_summary(base["main_profit"], base["main_won"]),
           "main_var": C.roi_summary(var["main_profit"], var["main_won"]),
           "d_main_hit": C.paired_diff_ci(base["main_won"], var["main_won"])}
    pa, pv_ = base["main_profit"].to_numpy(float), var["main_profit"].to_numpy(float)
    rng = np.random.default_rng(C.SEED)
    diffs = []
    for _ in range(C.N_BOOT // 200):
        idx = rng.integers(0, len(pa), size=(200, len(pa)))
        with np.errstate(all="ignore"):
            diffs.append(np.nanmean(pv_[idx], 1) - np.nanmean(pa[idx], 1))
    d = np.concatenate(diffs)
    out["d_main_roi"] = {"diff": float(np.nanmean(pv_) - np.nanmean(pa)),
                         "ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}
    return out


def run() -> dict:
    results, per_season, cache = {}, [], {}
    for season in C.EVAL_SEASONS:
        rows = PR.played(season)
        bundles = PR.bundles_for(season)
        from src.providers.football_data import odds_from_matches
        from src.providers.schema import odds_lookup
        odds = odds_lookup(odds_from_matches(rows))
        base = PR.season_predictions(season).set_index("match_id").loc[rows["match_id"]].reset_index()
        cache[(season, "base")] = base
        variants = [(s, g) for g in GROUPS for s in SIDES] + [(None, "h2h")]
        for side, group in variants:
            key = f"{group}_{side}" if side else group
            path = C.CACHE / f"sens_{season}_{key}.parquet"
            if path.exists():
                var = pd.read_parquet(path)
            else:
                var = PR.predict(blank(rows, side or "h", group), bundles, odds)
                var.to_parquet(path)
            cache[(season, key)] = var
            s = _summary(base, var)
            per_season.append({"season": season, "variant": key, "matches": s["matches"], "mean_tv": s["mean_tv"],
                               "d_log_loss": s["d_log_loss"]["mean"], "main_changed": s["main_changed"]})
            C.log.info("sensitivity %s %s: tv %.3f, dLL %.4f, main changed %.2f", season, key, s["mean_tv"],
                       s["d_log_loss"]["mean"], s["main_changed"])
    keys = sorted({k for (_, k) in cache if k != "base"})
    for key in keys:
        base = pd.concat([cache[(s, "base")] for s in C.EVAL_SEASONS], ignore_index=True)
        var = pd.concat([cache[(s, key)] for s in C.EVAL_SEASONS], ignore_index=True)
        results[key] = _summary(base, var)
    C.save_csv("sensitivity_by_season.csv", pd.DataFrame(per_season))
    flat = [{"variant": k, "matches": v["matches"], "mean_tv": v["mean_tv"], "p90_tv": v["p90_tv"],
             "dp_home": v["mean_dp_home"], "dp_draw": v["mean_dp_draw"], "dp_away": v["mean_dp_away"],
             "dxg_home": v["mean_dxg_home"], "dxg_away": v["mean_dxg_away"],
             "d_log_loss": v["d_log_loss"]["mean"], "d_log_loss_lo": v["d_log_loss"]["ci95"][0],
             "d_log_loss_hi": v["d_log_loss"]["ci95"][1], "main_changed": v["main_changed"],
             "main_hit_base": v["main_base"]["hit_rate"], "main_hit_var": v["main_var"]["hit_rate"],
             "main_roi_base": v["main_base"]["roi"], "main_roi_var": v["main_var"]["roi"],
             "d_main_roi": v["d_main_roi"]["diff"], "d_main_roi_lo": v["d_main_roi"]["ci95"][0],
             "d_main_roi_hi": v["d_main_roi"]["ci95"][1], "bets_base": v["main_base"]["bets"],
             "bets_var": v["main_var"]["bets"]} for k, v in results.items()]
    C.save_csv("sensitivity_summary.csv", pd.DataFrame(flat))
    phantom = phantom_reproduction()
    C.save_csv("sensitivity_phantoms.csv", phantom)
    out = {"variants": results, "seasons": list(C.EVAL_SEASONS), "phantoms": phantom.to_dict("records")}
    C.save_json("sensitivity.json", out)
    return out


def phantom_reproduction() -> pd.DataFrame:
    sn = pd.read_parquet(C.PRED_DIR / "snapshots.parquet")
    f = PR.all_features()
    bundles = PR.bundles_for(C.LIVE_SEASON)
    rows = []
    for ph, (real, side) in C.PHANTOMS.items():
        r = f.loc[f["match_id"] == real]
        if r.empty:
            continue
        r = r.copy()
        variants = {"real_features": r, "phantom_side_as_newcomer": blank(r, side, "newcomer")}
        for name, x in variants.items():
            x = x.copy()
            x["played"] = False                       # predict as a fixture: no settlement needed here
            p = PR.predict(x, bundles, odds={})
            stored = sn.loc[sn["match_id"] == (real if name == "real_features" else ph)].iloc[-1]
            rows.append({"pair": real, "variant": name,
                         "p_home": float(p["p_home"].iloc[0]), "p_draw": float(p["p_draw"].iloc[0]),
                         "p_away": float(p["p_away"].iloc[0]), "xg_home": float(p["xg_home"].iloc[0]),
                         "xg_away": float(p["xg_away"].iloc[0]),
                         "stored_id": stored["match_id"], "stored_p_home": float(stored["p_home"]),
                         "stored_p_draw": float(stored["p_draw"]), "stored_p_away": float(stored["p_away"]),
                         "stored_xg_home": float(stored["xg_home"]), "stored_xg_away": float(stored["xg_away"]),
                         "max_abs_diff_p": float(np.max(np.abs(p[["p_home", "p_draw", "p_away"]].to_numpy(float)[0]
                                                               - stored[["p_home", "p_draw", "p_away"]].to_numpy(float))))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    C.setup_logging("sensitivity")
    run()
