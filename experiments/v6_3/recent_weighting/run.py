"""Recent Season Weighting (v6.3, study 7) — evaluation of every weighting / retraining variant.

    # heavy (model fitting), one after the other:
    experiments/v6_3/heavy.sh 4500  experiments/v6_3/recent_weighting/logs/fits_2026.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026
    experiments/v6_3/heavy.sh 4500  experiments/v6_3/recent_weighting/logs/fits_2026k.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2026 --k 250
    experiments/v6_3/heavy.sh 10800 experiments/v6_3/recent_weighting/logs/fits_2025.log .venv/bin/python -m experiments.v6_3.recent_weighting.fits --season 2025
    # light (no fitting): metrics, selection on 2025/26, 2026/27 replay, Main ROI, figures
    .venv/bin/python -m experiments.v6_3.recent_weighting.run

Selection: every parameter (half-life, season multipliers, window, form prior) is chosen on the 2025/26 replay with
the same monthly scheme (trained on everything before the 1st of each month of 2025/26); 2026/27 is reported for
every variant, with the 2025/26 choice marked. 2026/27 is not a holdout (already examined by v6.1 / v6.2).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.replay import evaluate as EV

from . import fits as F
from . import form as FM
from . import lib as L

DIRS = C.study_dirs("recent_weighting")
SEASONS = (2025, 2026)
TASK_GROUPS = {"result": ["result"], "goals": ["home_goals", "away_goals"], "corners": ["corners"]}
NAMES = {"prod": "ежемесячно, веса продакшна (контроль)", "hl0": "ежемесячно, без затухания",
         "hl0.5": "ежемесячно, полураспад 0.5 г.", "hl1": "ежемесячно, полураспад 1 г.",
         "hl2": "ежемесячно, полураспад 2 г.", "hl4": "ежемесячно, полураспад 4 г.",
         "last2": "ежемесячно, прошлый сезон ×2", "last3": "ежемесячно, прошлый сезон ×3",
         "cur2": "ежемесячно, текущий сезон ×2", "cur4": "ежемесячно, текущий сезон ×4",
         "win2": "ежемесячно, окно 2 сезона + текущий", "win3": "ежемесячно, окно 3 сезона + текущий",
         "k250": "переобучение каждые 250 матчей", "selected": "выбор 2025/26 по задачам",
         "form": "давность + текущая форма (V8)", "B": "базовая линия B (продакшн, без переобучения)"}


# ------------------------------------------------------------------------------------------------ outputs
def load_outputs(season: int, scheme: str, kind: str = "wf") -> pd.DataFrame | None:
    """Outputs of a scheme for every match of ``season``; tasks the scheme shares with production's weights
    (e.g. hl0 for the result task) are taken from the control (``prod``) of the same kind."""
    p = DIRS["cache"] / f"{kind}_{season}_{scheme}.parquet"
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    base = pd.read_parquet(DIRS["cache"] / f"{kind if not kind.startswith('k') else 'wf'}_{season}_prod.parquet")
    o = o.set_index("match_id")
    base = base.set_index("match_id").reindex(o.index)
    for task, cols in L.TASK_COLS.items():
        for c in cols:
            if c not in o or o[c].isna().all():
                o[c] = base[c]
    return o.reset_index()


def truth(season: int) -> pd.DataFrame:
    m = L.season_rows(season)
    m = m.loc[m["played"].astype(bool) & m["home_goals"].notna()]
    return m[["match_id", "date", "league", "result", "home_goals", "away_goals", "total_corners", "h_season_games",
              "a_season_games"]].reset_index(drop=True)


def per_match_losses(out: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    o = out.set_index("match_id").reindex(t["match_id"])
    y = t["result"].map(L.RES_IDX).to_numpy()
    r = L.onex2_losses(o[["p_home", "p_draw", "p_away"]].to_numpy(float), y)
    gh = L.count_losses(o["lam_home"].to_numpy(float), t["home_goals"].to_numpy(float))
    ga = L.count_losses(o["lam_away"].to_numpy(float), t["away_goals"].to_numpy(float))
    c = L.count_losses(o["corner_mu"].to_numpy(float), t["total_corners"].to_numpy(float))
    return pd.DataFrame({"match_id": t["match_id"].to_numpy(), "ll": r["logloss"].to_numpy(),
                         "brier": r["brier"].to_numpy(), "correct": r["correct"].to_numpy(),
                         "dev_goals": ((gh["dev"] + ga["dev"]) / 2).to_numpy(),
                         "mae_goals": ((gh["abs"] + ga["abs"]) / 2).to_numpy(),
                         "dev_corners": c["dev"].to_numpy(), "mae_corners": c["abs"].to_numpy()})


PRIMARY = {"result": "ll", "goals": "dev_goals", "corners": "dev_corners"}


def task_metrics(out: pd.DataFrame, t: pd.DataFrame) -> dict:
    o = out.set_index("match_id").reindex(t["match_id"])
    y = t["result"].map(L.RES_IDX).to_numpy()
    res = {"result": L.onex2_metrics(o[["p_home", "p_draw", "p_away"]].to_numpy(float), y),
           "home_goals": L.count_metrics(o["lam_home"], t["home_goals"]),
           "away_goals": L.count_metrics(o["lam_away"], t["away_goals"]),
           "corners": L.count_metrics(o["corner_mu"], t["total_corners"])}
    return res


def paired_vs(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for g, k in PRIMARY.items():
        x, y = a[k].to_numpy(), b[k].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        out[g] = C.paired_mean_diff(x[ok], y[ok])
    ok = np.isfinite(a["mae_goals"]) & np.isfinite(b["mae_goals"])
    out["goals_mae"] = C.paired_mean_diff(a["mae_goals"][ok].to_numpy(), b["mae_goals"][ok].to_numpy())
    out["accuracy"] = C.paired_mean_diff(a["correct"].to_numpy(), b["correct"].to_numpy())
    out["brier"] = C.paired_mean_diff(a["brier"].to_numpy(), b["brier"].to_numpy())
    ok = np.isfinite(a["mae_corners"]) & np.isfinite(b["mae_corners"])
    out["corners_mae"] = C.paired_mean_diff(a["mae_corners"][ok].to_numpy(), b["mae_corners"][ok].to_numpy())
    return out


def combine(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outputs with each task group taken from its own scheme's table."""
    base = parts["result"][["match_id", "p_home", "p_draw", "p_away"]]
    g = parts["goals"].set_index("match_id").reindex(base["match_id"])
    c = parts["corners"].set_index("match_id").reindex(base["match_id"])
    out = base.copy()
    out["lam_home"], out["lam_away"] = g["lam_home"].to_numpy(), g["lam_away"].to_numpy()
    out["corner_mu"] = c["corner_mu"].to_numpy()
    return out


# ------------------------------------------------------------------------------------------------ main
def run(log) -> dict:
    started = time.time()
    tr = {s: truth(s) for s in SEASONS}
    outs: dict[tuple, pd.DataFrame] = {}
    for s in SEASONS:
        for sc in F.SCHEMES:
            for kind in ("wf", "static"):
                o = load_outputs(s, sc, kind)
                if o is not None:
                    outs[(s, kind, sc)] = o
        o = load_outputs(s, "k250", "wf")
        if o is not None:
            outs[(s, "wf", "k250")] = o
    prod_b = L.production_outputs()
    outs[(2026, "B", "B")] = prod_b
    log.info("loaded %d output tables", len(outs))

    # ---- form correction: k chosen on 2025/26 on top of the best half-life scheme of each task group
    los = {key: per_match_losses(o, tr[key[0]]) for key, o in outs.items()}
    hl = [sc for sc in F.SCHEMES if sc.startswith("hl") or sc == "prod"]
    chosen_hl = {}
    for g, k in PRIMARY.items():
        cand = {sc: np.nanmean(los[(2025, "wf", sc)][k]) for sc in hl if (2025, "wf", sc) in los}
        chosen_hl[g] = min(cand, key=cand.get) if cand else "prod"
    form_sel = {}
    if (2025, "wf", "prod") in outs:
        base25 = combine({g: outs[(2025, "wf", chosen_hl[g])] for g in TASK_GROUPS})
        sums25 = FM.team_sums(2025, base25)
        grid = []
        for k in FM.K_GRID:
            for kc in FM.KC_GRID:
                lo = per_match_losses(FM.apply(base25, sums25, k, kc), tr[2025])
                grid.append({"k": k, "kc": kc, **{g: float(np.nanmean(lo[m])) for g, m in PRIMARY.items()}})
        gdf = pd.DataFrame(grid)
        form_sel = {"grid": grid, "k_result": float(gdf.loc[gdf["result"].idxmin(), "k"]),
                    "k_goals": float(gdf.loc[gdf["goals"].idxmin(), "k"]),
                    "kc": float(gdf.loc[gdf["corners"].idxmin(), "kc"]), "half_life_schemes": chosen_hl}
        for s in SEASONS:
            if (s, "wf", "prod") not in outs:
                continue
            base = combine({g: outs[(s, "wf", chosen_hl[g])] for g in TASK_GROUPS})
            sums = FM.team_sums(s, base)
            fr = FM.apply(base, sums, form_sel["k_result"], form_sel["kc"])
            fg = FM.apply(base, sums, form_sel["k_goals"], form_sel["kc"])
            fr[["lam_home", "lam_away"]] = fg[["lam_home", "lam_away"]].to_numpy()
            outs[(s, "wf", "form")] = fr
            los[(s, "wf", "form")] = per_match_losses(fr, tr[s])
        log.info("form: %s", {k: v for k, v in form_sel.items() if k != "grid"})

    # ---- selection on 2025/26: best monthly scheme per task group (primary metric)
    selected = {}
    for g, k in PRIMARY.items():
        cand = {sc: float(np.nanmean(los[(2025, "wf", sc)][k])) for sc in F.SCHEMES if (2025, "wf", sc) in los}
        selected[g] = min(cand, key=cand.get) if cand else "prod"
    for s in SEASONS:
        if all((s, "wf", selected[g]) in outs for g in TASK_GROUPS):
            outs[(s, "wf", "selected")] = combine({g: outs[(s, "wf", selected[g])] for g in TASK_GROUPS})
            los[(s, "wf", "selected")] = per_match_losses(outs[(s, "wf", "selected")], tr[s])
    log.info("selected on 2025/26: %s", selected)

    # ---- metrics table and paired comparisons (vs control and, for 2026/27, vs Baseline B)
    rows, paired = [], {}
    for key, lo in los.items():
        s, kind, sc = key
        tm = task_metrics(outs[key], tr[s])
        rows.append({"season": s, "kind": kind, "scheme": sc, "result_ll": tm["result"]["log_loss"],
                     "result_brier": tm["result"]["brier"], "result_acc": tm["result"]["accuracy"],
                     "result_f1": tm["result"]["macro_f1"], "result_bacc": tm["result"]["balanced_accuracy"],
                     "home_mae": tm["home_goals"]["mae"], "away_mae": tm["away_goals"]["mae"],
                     "home_dev": tm["home_goals"]["poisson_deviance"], "away_dev": tm["away_goals"]["poisson_deviance"],
                     "home_r2": tm["home_goals"]["r2"], "away_r2": tm["away_goals"]["r2"],
                     "corners_mae": tm["corners"]["mae"], "corners_dev": tm["corners"]["poisson_deviance"],
                     "corners_r2": tm["corners"]["r2"], "corners_n": tm["corners"]["n"]})
        ctrl = los.get((s, "wf", "prod"))
        if ctrl is not None and key != (s, "wf", "prod"):
            paired[f"{s}|{kind}|{sc}|vs_control"] = paired_vs(lo, ctrl)
        if s == 2026 and kind != "B":
            paired[f"{s}|{kind}|{sc}|vs_B"] = paired_vs(lo, los[(2026, "B", "B")])
    metrics = pd.DataFrame(rows).sort_values(["season", "kind", "scheme"])
    metrics.to_csv(DIRS["results"] / "metrics.csv", index=False)

    # Holm within 2026/27 vs Baseline B, per task group, over every variant tried
    holm = {}
    for g in PRIMARY:
        pv = {k: v[g]["p"] for k, v in paired.items() if k.startswith("2026|") and k.endswith("vs_B")}
        holm[g] = C.holm(pv)
    pv = {k: v["goals"]["p"] for k, v in paired.items() if k.startswith("2026|") and k.endswith("vs_control")}
    holm["goals_vs_control"] = C.holm(pv)
    holm["result_vs_control"] = C.holm({k: v["result"]["p"] for k, v in paired.items()
                                        if k.startswith("2026|") and k.endswith("vs_control")})
    holm["corners_vs_control"] = C.holm({k: v["corners"]["p"] for k, v in paired.items()
                                         if k.startswith("2026|") and k.endswith("vs_control")})

    # ---- when does the current season help: by refit month and by matches already played by the teams
    when = []
    for s in SEASONS:
        ctrl = los.get((s, "wf", "prod"))
        if ctrl is None:
            continue
        cut = outs[(s, "wf", "prod")].set_index("match_id").reindex(tr[s]["match_id"])["refit_cutoff"]
        month = pd.to_datetime(cut).dt.strftime("%Y-%m-%d").to_numpy()
        games = L.season_games(tr[s]).to_numpy()
        gbin = pd.cut(games, [-1, 2, 5, 9, 14, 19, 99], labels=["0-2", "3-5", "6-9", "10-14", "15-19", "20+"]
                      ).astype(str)
        for sc in ("cur2", "cur4", "last2", "hl1", "hl0.5", "form", "selected"):
            if (s, "wf", sc) not in los:
                continue
            lo = los[(s, "wf", sc)]
            for g, k in PRIMARY.items():
                d = lo[k].to_numpy() - ctrl[k].to_numpy()
                for by, lab in (("refit_month", month), ("games", gbin)):
                    for v in pd.unique(lab):
                        sel = (lab == v) & np.isfinite(d)
                        if sel.sum() < 30:
                            continue
                        when.append({"season": s, "scheme": sc, "group": g, "by": by, "value": v, "n": int(sel.sum()),
                                     "diff": float(d[sel].mean()),
                                     "se": float(d[sel].std(ddof=1) / np.sqrt(sel.sum()))})
        # static (no new data) vs monthly (new data) with production weights
        st = los.get((s, "static", "prod"))
        if st is not None:
            for g, k in PRIMARY.items():
                d = ctrl[k].to_numpy() - st[k].to_numpy()
                for v in pd.unique(month):
                    sel = (month == v) & np.isfinite(d)
                    if sel.sum() >= 30:
                        when.append({"season": s, "scheme": "monthly_vs_static", "group": g, "by": "refit_month",
                                     "value": v, "n": int(sel.sum()), "diff": float(d[sel].mean()),
                                     "se": float(d[sel].std(ddof=1) / np.sqrt(sel.sum()))})
    pd.DataFrame(when).to_csv(DIRS["results"] / "when_it_helps.csv", index=False)

    # share of the current season in the training weight at each cut-off (production weights vs cur4)
    share = []
    feats = L.features()
    for s in SEASONS:
        for cut in L.monthly_cutoffs(s):
            rows_ = feats.loc[feats["played"].astype(bool) & (feats["date"] < cut) & feats["home_goals"].notna()]
            for sc in ("prod", "cur2", "cur4"):
                for task in ("result", "home_goals"):
                    keep, w = F.scheme_weights(sc, task, rows_, s)
                    cur = rows_["season"].to_numpy() == s
                    share.append({"season": s, "cutoff": str(cut.date()), "scheme": sc, "task": task,
                                  "current_rows": int(cur.sum()), "weight_share": float(w[cur].sum() / w.sum())})
    pd.DataFrame(share).to_csv(DIRS["results"] / "current_weight_share.csv", index=False)

    # ---- Main ROI on 2026/27 (both universes): every scheme for all tasks, static schemes, k250, selected, form,
    # and per-task swaps of the selected variant
    variants = {}
    for (s, kind, sc), o in outs.items():
        if s != 2026 or kind == "B":
            continue
        variants[f"{kind}:{sc}"] = o
    if (2026, "wf", "selected") in outs:
        for g, cols in (("result", ["p_home", "p_draw", "p_away"]), ("goals", ["lam_home", "lam_away"]),
                        ("corners", ["corner_mu"])):
            variants[f"swap_{g}:selected"] = outs[(2026, "wf", "selected")][["match_id"] + cols]
            variants[f"swap_{g}:prod"] = outs[(2026, "wf", "prod")][["match_id"] + cols]
    roi, roi_rows, tabs_keep = {}, [], {}
    ctrl_tabs = None
    for name, o in variants.items():
        t0 = time.time()
        tabs = L.main_tables(o, name)
        roi[name] = L.main_eval(tabs)
        tabs_keep[name] = tabs
        if name == "wf:prod":
            ctrl_tabs = tabs
        for u in L.UNIVERSES:
            roi_rows.append(L.roi_row(name, roi[name], u))
        log.info("ROI %-22s close %+.4f t09 %+.4f (%.0fs)", name, roi[name]["close"]["summary"]["roi"],
                 roi[name]["t09"]["summary"]["roi"], time.time() - t0)
    rr = pd.DataFrame(roi_rows)
    rr["p_holm"] = np.nan
    rr["d_roi_vs_control"] = np.nan
    rr["p_vs_control"] = np.nan
    for u in L.UNIVERSES:
        sub = rr.loc[rr["universe"] == u]
        h = C.holm(dict(zip(sub["variant"], sub["p"])))
        rr.loc[sub.index, "p_holm"] = sub["variant"].map(h)
        if ctrl_tabs is not None:
            for i, r in sub.iterrows():
                if r["variant"] == "wf:prod":
                    continue
                c = EV.compare(tabs_keep[r["variant"]][u], ctrl_tabs[u])
                rr.loc[i, "d_roi_vs_control"] = c["roi_diff"]["diff"]
                rr.loc[i, "p_vs_control"] = c["roi_diff"]["p"]
    rr.to_csv(DIRS["results"] / "main_roi.csv", index=False)

    # ---- candidates: selected, form, control, best close ROI; per-match tables and robustness
    best_roi = rr.loc[rr["universe"] == "close"].sort_values("d_roi", ascending=False)["variant"].iloc[0]
    res = L.replay_results()
    cand = {}
    decisions = decide(paired, holm, rr, selected, chosen_hl)
    b_cands = [f"wf:{k.split('|')[0]}" for k, d in decisions.items() if d in ("A", "B")]
    for name in dict.fromkeys(["wf:selected", "wf:form", "wf:prod", best_roi] + b_cands):
        if name not in tabs_keep:
            continue
        for u, t in tabs_keep[name].items():
            tag = name.replace(":", "_")
            EV.per_match_table(t, res, f"recent_{tag}_{u}", f"refit({name})").to_csv(
                DIRS["results"] / f"per_match_{tag}_{u}.csv", index=False)
            b = L.baseline_b(u)
            cand.setdefault(name, {})[u] = {
                "by_month": EV.breakdown(t.assign(month=t["date"].dt.to_period("M").astype(str)), "month")
                .to_dict("records"),
                "base_by_month": EV.breakdown(b.assign(month=b["date"].dt.to_period("M").astype(str)), "month")
                .to_dict("records"),
                "by_league": EV.breakdown(t, "league").to_dict("records"),
                "base_by_league": EV.breakdown(b, "league").to_dict("records"),
                "big_wins": L.big_win_dependence(t, b), "odds_minus_2pct": L.odds_sensitivity(t),
                "base_odds_minus_2pct": L.odds_sensitivity(b)}
    for u in L.UNIVERSES:
        EV.per_match_table(L.baseline_b(u), res, f"baseline_b_{u}", "prod_20261012").to_csv(
            DIRS["results"] / f"per_match_baseline_b_{u}.csv", index=False)

    g26 = L.season_games(tr[2026])
    coverage = {"season_matches": int(len(L.replay_matches())), "played": int(len(res)),
                "not_played": int(len(L.replay_matches()) - len(res)),
                "no_current_season_games": int((g26 == 0).sum()), "fewer_than_3_season_games": int((g26 < 3).sum()),
                "evaluated_probabilistic": {str(s): int(len(tr[s])) for s in SEASONS},
                "corners_evaluated": {str(s): int(np.isfinite(los[(s, "wf", "prod")]["dev_corners"]).sum())
                                      for s in SEASONS if (s, "wf", "prod") in los},
                "main": {name: {u: {"main_rows": int(len(t)), "bets": int(t["bet"].sum()),
                                    "no_price": int(t["odds"].isna().sum()),
                                    "push": int((t["status"] == "PUSH").sum()),
                                    "unknown": int((t["status"] == "UNKNOWN").sum()),
                                    "no_main": int(len(res) - t["match_id"].nunique())}
                                for u, t in tabs.items()} for name, tabs in tabs_keep.items()}}
    out = {"selected_2025": selected, "chosen_half_life_2025": chosen_hl, "form": form_sel, "paired": paired,
           "holm": holm, "roi": roi, "candidates": cand, "coverage": coverage, "decisions": decisions,
           "names": NAMES, "n_variants": {"monthly_schemes": len(F.SCHEMES), "static_schemes": len(F.SCHEMES),
                                          "other": ["k250", "selected", "form"],
                                          "roi_variants": int(rr["variant"].nunique())},
           "cutoffs": {str(s): [str(c.date()) for c in L.monthly_cutoffs(s)] for s in SEASONS},
           "k250_cutoffs": [str(c.date()) for c in L.k_match_cutoffs(2026, 250)]}
    C.write_json(DIRS["results"] / "results.json", out)
    preregister(decisions, paired, rr, selected, form_sel)
    from . import figures as FG
    FG.make_all(log)
    C.write_json(DIRS["results"] / "run_info.json", C.run_info("recent_weighting", started, {
        "schemes": F.SCHEMES, "selection_season": 2025, "evaluated_season": 2026,
        "production_half_life": L.PROD_HALF_LIFE, "production_version": "prod_20261012", "n_jobs": C.N_JOBS}))
    return out


def decide(paired: dict, holm: dict, rr: pd.DataFrame, selected: dict, chosen_hl: dict) -> dict:
    """A / B / C per (variant, task group). The rule (written before looking at 2026/27):
    A — chosen on 2025/26 for that task, better than Baseline B AND than the control on 2026/27 with Holm p < 0.05,
        Main ROI not significantly worse than B in both universes;
    B — better than the control on 2025/26 and on 2026/27 by point (not significant), Main ROI not significantly
        worse; tightened after the first run (can only remove candidates): also better than Baseline B on
        2026/27 by point — a scheme that loses to the production model is not a candidate for production;
    C — everything else."""
    out = {}
    for key in [k for k in paired if k.startswith("2026|wf|") and k.endswith("vs_B")]:
        sc = key.split("|")[2]
        if sc == "prod":
            continue
        rv = rr.loc[rr["variant"] == f"wf:{sc}"]
        roi_bad = bool((rv["d_hi"] < 0).any()) if len(rv) else False
        for g in PRIMARY:
            vb = paired[key][g]
            vc = paired.get(f"2026|wf|{sc}|vs_control", {}).get(g)
            v25 = paired.get(f"2025|wf|{sc}|vs_control", {}).get(g)
            better25 = v25 is not None and v25["diff"] is not None and v25["diff"] < 0
            sig = (holm[g].get(key, 1) < 0.05 and vb["diff"] < 0 and vc is not None and vc["diff"] < 0
                   and holm.get(f"{g}_vs_control", {}).get(f"2026|wf|{sc}|vs_control", 1) < 0.05)
            chosen = selected.get(g) == sc or sc in ("selected", "form")
            if chosen and sig and not roi_bad:
                d = "A"
            elif (better25 and vc is not None and vc["diff"] is not None and vc["diff"] < 0 and vb["diff"] < 0
                  and not roi_bad):
                d = "B"
            else:
                d = "C"
            out[f"{sc}|{g}"] = d
    return out


def preregister(decisions: dict, paired: dict, rr: pd.DataFrame, selected: dict, form_sel: dict) -> None:
    now = pd.Timestamp.now(tz="Europe/Berlin").isoformat(timespec="seconds")
    entries = []
    for key, d in decisions.items():
        if d not in ("A", "B"):
            continue
        sc, g = key.split("|")
        entries.append({
            "variant": sc, "task_group": g, "decision": d, "name": NAMES.get(sc, sc), "frozen_at": now,
            "hypothesis": f"Retraining the {g} model(s) on the 1st of every month with the weighting '{sc}' gives a "
                          f"lower {PRIMARY[g]} than the production model trained once before the season, and a "
                          "Main ROI not lower than Baseline B.",
            "exact_rule": {"scheme": sc, "selected_2025": selected, "form": {k: v for k, v in form_sel.items()
                                                                           if k != "grid"},
                           "code": "experiments/v6_3/recent_weighting/fits.py (scheme_weights), form.py"},
            "primary_metric": f"{PRIMARY[g]} paired with production (Baseline B) on forward matches",
            "minimum_forward_sample": "2000 played matches (two monthly refits)",
            "evidence_2026_27": {"vs_B": paired.get(f"2026|wf|{sc}|vs_B", {}).get(g),
                                 "vs_control": paired.get(f"2026|wf|{sc}|vs_control", {}).get(g),
                                 "roi": rr.loc[rr["variant"] == f"wf:{sc}"].to_dict("records")}})
    C.write_json(DIRS["results"] / "preregistration.json", {"study": "v6.3 Recent Season Weighting",
                                                             "entries": entries})


def main() -> None:
    log = C.setup_logging("recent_weighting.run", DIRS["logs"] / "run.log")
    run(log)


if __name__ == "__main__":
    main()
