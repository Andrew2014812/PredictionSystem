"""Explainability research of the "Why this prediction?" card (FootPredict v6.1).

    .venv/bin/python experiments/v6_1/explain/run_research.py all          # every experiment
    .venv/bin/python experiments/v6_1/explain/run_research.py e1 e2         # some of them

E0  the chain values on the card reproduce the stored market probabilities;
E1  old (v6) vs chain vs linear vs output-Shapley on a sample of real matches: additivity, the share of
    influence the old card ignored, sign agreement, non-monotonic markets, time;
E2  every method against a reference: permutation Shapley values of the whole composition g(u(x));
E3  stability: small perturbations of the features and the most similar match;
E4  football facts (recent form) and the pre-match guarantee;
E5  the example matches of the report (old vs new factors).

Results: results/*.json|csv, figures: by build_pdf.py, log: cache/run_research.log.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, RESULTS, Data, chain, factor_groups, g_of, jaccard, linear_contributions,  # noqa: E402
                    log, model_outputs, mshap_contributions, occlusion, old_explain, old_factors, overlap, panel,
                    permutation_reference, plan_outputs, raw_columns, save_json, setup_logging, sign_agreement,
                    spearman, to_factors, top)

from src.analytics.teams import team_matches  # noqa: E402
from src.explain.card import card_factors  # noqa: E402
from src.explain.facts import market_facts  # noqa: E402
from src.markets.markets import market_group, settle_selection  # noqa: E402

SEED = 20261104
N_SAMPLE = 300          # E1 / E4 matches
N_REF = 30              # E2 matches (reference Shapley)
N_REF_REPEAT = 8        # E2 matches computed with a second seed
N_PERM, N_BG = 16, 24   # reference: permutations (8 antithetic pairs) and background matches
N_STAB = 40             # E3 matches
N_NOISE, NOISE = 6, 0.05
EXAMPLES = ["I1-20261011-como-roma", "E1-20261011-southampton-portsmouth", "SP1-20260916-barcelona-santander",
            "G1-20261011-aris-volos-nfc"]


def sample(data: Data, n: int, seed: int) -> pd.DataFrame:
    pool = data.pool()
    per = n // len(pool["model_version"].unique())
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in pool.groupby("model_version")]
    return pd.concat(parts).reset_index(drop=True)


# ----------------------------------------------------------------------------- E0
def e0(data: Data) -> None:
    """Chain values from the stored model outputs vs the stored probabilities (every snapshot)."""
    t = time.time()
    mk = data.markets.set_index(["match_id", "market", "selection"])["probability"]
    errs, rows = [], 0
    for snap in data.snaps.itertuples(index=False):
        s = snap._asdict()
        for role in ("main", "risk"):
            m, sel = s.get(f"{role}_market"), s.get(f"{role}_selection")
            if not isinstance(m, str):
                continue
            v = chain.chain_values(m, sel, s)
            stored = mk.get((s["match_id"], m, sel))
            if v is not None and stored is not None:
                errs.append({"version": s["model_version"], "source": s["source"], "group": market_group(m),
                             "err": abs(v.probability - float(stored))})
            rows += 1
    df = pd.DataFrame(errs)
    # every stored selection of 200 random matches (all markets, not only Main / Risk)
    ids = data.snaps.sample(200, random_state=SEED)["match_id"]
    full = []
    for mid in ids:
        snap = data.snaps.loc[data.snaps["match_id"] == mid].iloc[0]
        for r in data.match_markets(mid).itertuples(index=False):
            v = chain.chain_values(r.market, r.selection, snap)
            if v is not None:
                full.append(abs(v.probability - float(r.probability)))
    # live predictions: does today's feature row still reproduce the stored probability?
    live = data.snaps.loc[(data.snaps["source"] == "live") & data.snaps["match_id"].isin(data.by_id.index)
                          & (data.snaps["model_version"] == "prod_20261012")]
    lrows = data.by_id.loc[live["match_id"]].reset_index(drop=True)
    U = model_outputs(lrows, data.bundles["prod_20261012"])
    drift = []
    for i, snap in enumerate(live.itertuples(index=False)):
        if not isinstance(snap.main_market, str):
            continue
        sub = {k: v[i:i + 1] for k, v in U.items()}
        p_now = float(g_of(snap.main_market, snap.main_selection, sub, data.bundles["prod_20261012"])[0])
        stored = mk.get((snap.match_id, snap.main_market, snap.main_selection))
        if stored is not None:
            drift.append(abs(p_now - float(stored)))
    drift = np.array(drift)
    out = {
        "main_risk_selections": int(len(df)), "max_abs_error": float(df["err"].max()),
        "by_group": df.groupby("group")["err"].max().to_dict(),
        "by_source": df.groupby("source")["err"].max().to_dict(),
        "all_markets_200_matches": {"selections": len(full), "max_abs_error": float(max(full))},
        "live_drift": {"n": int(len(drift)), "share_exact_1e-4": float((drift < 1e-4).mean()),
                       "share_over_1pp": float((drift > 0.01).mean()), "median": float(np.median(drift)),
                       "p90": float(np.quantile(drift, 0.9)), "max": float(drift.max())},
        "seconds": time.time() - t,
    }
    save_json("e0_reproduction.json", out)
    log.info("E0 %s", out)


# ----------------------------------------------------------------------------- E1
def _path_sign_change(market, selection, inputs) -> bool:
    """Does ∂g/∂(log λ) change sign along ū → u (the market is not monotonic in the expected goals)?"""
    if market_group(market) == "CORNERS":
        idx = [0]
    else:
        idx = [3, 4]
    t = np.linspace(0, 1, 13)
    pts = inputs.base[None, :] + t[:, None] * (inputs.u - inputs.base)[None, :]
    grad = chain.gradient(market, selection, pts, inputs.rho, inputs.nb_size)[:, idx]
    big = np.abs(grad) > 1e-6
    return bool(any((np.sign(grad[big[:, k], k]).min() < 0 < np.sign(grad[big[:, k], k]).max())
                    for k in range(len(idx)) if big[:, k].any()))


def e1(data: Data) -> None:
    t0 = time.time()
    rows = []
    sm = sample(data, N_SAMPLE, SEED)
    for n, snap in enumerate(sm.itertuples(index=False)):
        bundles = data.bundles[snap.model_version]
        row = data.row(snap.match_id)
        mk = data.match_markets(snap.match_id).set_index(["market", "selection"])["probability"]
        t = time.time()
        goal_in = chain.chain_inputs(row, bundles, chain.GOAL_OUTPUTS)
        t_goal = time.time() - t
        t = time.time()
        corner_in = chain.chain_inputs(row, bundles, chain.CORNER_OUTPUTS)
        t_corner = time.time() - t
        for market, sel, role in panel(snap._asdict(), data.match_markets(snap.match_id)):
            group = market_group(market)
            inputs = corner_in if group == "CORNERS" else goal_in
            t = time.time()
            ex = chain.attribute(inputs, market, sel)
            t_attr = time.time() - t
            t = time.time()
            old, old_shown = old_explain(row, market, sel, bundles)
            t_old = time.time() - t
            new = to_factors(ex.contributions)
            lin_df, lin_sum = linear_contributions(inputs, market, sel)
            lin = to_factors(lin_df)
            msh = to_factors(mshap_contributions(inputs, market, sel))
            delta = ex.probability - ex.base_probability
            eff = ex.output_effect
            omitted = [k for k in eff.index if k not in plan_outputs(market, sel)]
            new_shown = [f.key for f in card_factors(ex.contributions, "H", "A")]
            rows.append({
                "match_id": snap.match_id, "version": snap.model_version, "role": role, "market": market,
                "selection": sel, "group": group, "p_stored": float(mk.get((market, sel), np.nan)),
                "p_chain": ex.probability, "p_base": ex.base_probability, "delta": delta,
                "complete_err": abs(ex.total - delta),
                "linear_err": abs(lin_sum - delta), "linear_rel_err": abs(lin_sum - delta) / max(abs(delta), 1e-9),
                "omitted_share": float(eff[omitted].abs().sum() / max(eff.abs().sum(), 1e-12)),
                "result_share": float(eff[[k for k in eff.index if k.startswith("z_")]].abs().sum()
                                      / max(eff.abs().sum(), 1e-12)) if group != "CORNERS" else 0.0,
                "goals_share": float(eff[[k for k in eff.index if k.startswith("log_lambda")]].abs().sum()
                                     / max(eff.abs().sum(), 1e-12)) if group != "CORNERS" else 0.0,
                "old_sign_agree_new": float(np.mean([np.sign(old[k]) == np.sign(new.get(k, 0.0)) for k in old_shown]))
                if old_shown else np.nan,
                "old_new_top5": overlap(old, new), "old_new_top1": float(top(old, 1) == top(new, 1)),
                "old_new_shown_jaccard": jaccard(old_shown, new_shown),
                "mshap_new_spearman": spearman(new, msh), "linear_new_spearman": spearman(new, lin),
                "mshap_new_top5": overlap(new, msh), "linear_new_top5": overlap(new, lin),
                "new_top_share": float(new.abs().max() / new.abs().sum()),
                "new_top5_share": float(new.abs().sort_values().iloc[-5:].sum() / new.abs().sum()),
                "non_monotonic": _path_sign_change(market, sel, inputs),
                "t_shap_inputs": t_goal if group != "CORNERS" else t_corner, "t_attr": t_attr, "t_old": t_old,
            })
        if n % 25 == 0:
            log.info("E1 %d/%d matches (%.0fs)", n, len(sm), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1_selections.csv", index=False)
    agg = {"matches": int(df["match_id"].nunique()), "selections": int(len(df)), "seconds": time.time() - t0,
           "max_p_diff_stored": float((df["p_chain"] - df["p_stored"]).abs().max()),
           "max_complete_err": float(df["complete_err"].max()),
           "linear_rel_err_median": float(df["linear_rel_err"].median()),
           "linear_rel_err_p90": float(df["linear_rel_err"].quantile(0.9)),
           "linear_rel_err_over10pct": float((df["linear_rel_err"] > 0.10).mean())}
    by = df.groupby("group").agg(n=("market", "size"), omitted=("omitted_share", "mean"),
                                 omitted_over20=("omitted_share", lambda s: float((s > 0.2).mean())),
                                 result_share=("result_share", "mean"), goals_share=("goals_share", "mean"),
                                 old_sign=("old_sign_agree_new", "mean"), old_top5=("old_new_top5", "mean"),
                                 old_top1=("old_new_top1", "mean"), shown_jacc=("old_new_shown_jaccard", "mean"),
                                 lin_rel=("linear_rel_err", "median"), lin_sp=("linear_new_spearman", "mean"),
                                 msh_sp=("mshap_new_spearman", "mean"), msh_top5=("mshap_new_top5", "mean"),
                                 nonmono=("non_monotonic", "mean"), top_share=("new_top_share", "median"),
                                 top5_share=("new_top5_share", "median"))
    agg["by_group"] = by.round(4).reset_index().to_dict("records")
    agg["by_role"] = df.groupby("role").agg(n=("market", "size"), omitted=("omitted_share", "mean"),
                                            old_sign=("old_sign_agree_new", "mean"),
                                            old_top5=("old_new_top5", "mean")).round(4).reset_index().to_dict("records")
    agg["time_ms"] = {"shap_inputs_goal_median": float(df.loc[df.group != "CORNERS", "t_shap_inputs"].median() * 1000),
                      "shap_inputs_corner_median": float(df.loc[df.group == "CORNERS", "t_shap_inputs"].median() * 1000),
                      "attribution_median": float(df["t_attr"].median() * 1000),
                      "old_explain_selection_median": float(df["t_old"].median() * 1000)}
    save_json("e1_summary.json", agg)
    log.info("E1 done %s", {k: v for k, v in agg.items() if k != "by_group"})


# ----------------------------------------------------------------------------- E2
def _background(data: Data, version: str, n: int, seed: int) -> pd.DataFrame:
    to = pd.Timestamp(data.meta[version]["trained_to"]) if data.meta[version] else data.features["date"].max()
    played = data.features.loc[data.features["played"].astype(bool) & (data.features["date"] < to)
                               & (data.features["date"] >= to - pd.Timedelta(days=365))]
    return played.sample(n, random_state=seed).reset_index(drop=True)


def e2(data: Data) -> None:
    t0 = time.time()
    sm = sample(data, N_SAMPLE, SEED).sample(N_REF, random_state=SEED + 1).reset_index(drop=True)
    recs, repeat = [], []
    for n, snap in enumerate(sm.itertuples(index=False)):
        bundles = data.bundles[snap.model_version]
        row = data.row(snap.match_id)
        cols = raw_columns(bundles)
        groups = factor_groups(cols)
        means = {}
        for b in bundles.values():
            means.update(b.extra.get("feature_means", {}))
        bg = _background(data, snap.model_version, N_BG, SEED + n)
        sels = [(m, s, r) for m, s, r in panel(snap._asdict(), data.match_markets(snap.match_id))]
        t = time.time()
        ref = permutation_reference(row, bg, bundles, groups, [(m, s) for m, s, _ in sels], N_PERM, SEED + n)
        t_ref = time.time() - t
        if n < N_REF_REPEAT:
            ref2 = permutation_reference(row, bg, bundles, groups, [(m, s) for m, s, _ in sels], N_PERM, SEED + 1000 + n)
        goal_in = chain.chain_inputs(row, bundles, chain.GOAL_OUTPUTS)
        corner_in = chain.chain_inputs(row, bundles, chain.CORNER_OUTPUTS)
        for i, (market, sel, role) in enumerate(sels):
            inputs = corner_in if market_group(market) == "CORNERS" else goal_in
            R = pd.Series(ref["phi"][i], index=ref["keys"])
            se = pd.Series(ref["se"][i], index=ref["keys"])
            ex = chain.attribute(inputs, market, sel)
            t = time.time()
            occ = occlusion(row, market, sel, bundles, groups, means)
            t_occ = time.time() - t
            methods = {"old": old_factors(row, market, sel, bundles), "chain": to_factors(ex.contributions),
                       "linear": to_factors(linear_contributions(inputs, market, sel)[0]),
                       "mshap": to_factors(mshap_contributions(inputs, market, sel)), "occl": occ}
            ref_top = top(R, 5)
            rec = {"match_id": snap.match_id, "market": market, "selection": sel, "role": role,
                   "group": market_group(market), "p": ex.probability, "ref_sum": float(R.sum()),
                   "ref_top5_se_rel": float((se[ref_top] / R[ref_top].abs()).median()),
                   "ref_top5_share": float(R.abs()[ref_top].sum() / R.abs().sum()),
                   "t_ref": t_ref / len(sels), "t_occl": t_occ}
            for name, M in methods.items():
                M = M.reindex(R.index).fillna(0.0) if name != "old" else M
                rec[f"{name}_sign5"] = sign_agreement(M, R, 5)
                rec[f"{name}_sign3"] = sign_agreement(M, R, 3)
                rec[f"{name}_top5"] = overlap(M, R, 5)
                rec[f"{name}_top1"] = float(top(M, 1) == top(R, 1))
                rec[f"{name}_spearman"] = spearman(M, R)
                # sign agreement on the reference's own top-5 (does the method get the main factors' direction?)
                rec[f"{name}_refsign5"] = float(np.mean([np.sign(M.get(k, 0.0)) == np.sign(R[k]) for k in ref_top]))
            recs.append(rec)
            if n < N_REF_REPEAT:
                R2 = pd.Series(ref2["phi"][i], index=ref2["keys"])
                repeat.append({"match_id": snap.match_id, "market": market, "top5": overlap(R, R2),
                               "sign5": sign_agreement(R2, R, 5), "spearman": spearman(R, R2),
                               "corr": float(np.corrcoef(R.to_numpy(), R2.to_numpy())[0, 1])})
        log.info("E2 %d/%d %s (%.0fs, ref %.1fs)", n + 1, len(sm), snap.match_id, time.time() - t0, t_ref)
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "e2_reference.csv", index=False)
    pd.DataFrame(repeat).to_csv(RESULTS / "e2_reference_repeat.csv", index=False)
    methods = ["old", "chain", "linear", "mshap", "occl"]
    summary = {"matches": int(df["match_id"].nunique()), "selections": int(len(df)), "seconds": time.time() - t0,
               "n_perm": N_PERM, "n_background": N_BG,
               "ref_top5_se_rel_median": float(df["ref_top5_se_rel"].median()),
               "ref_top5_share_median": float(df["ref_top5_share"].median()),
               "ref_seconds_per_match": float(df.groupby("match_id")["t_ref"].sum().median()),
               "occl_ms_median": float(df["t_occl"].median() * 1000),
               "overall": {m: {k: float(df[f"{m}_{k}"].mean()) for k in ("sign5", "sign3", "refsign5", "top5", "top1",
                                                                          "spearman")} for m in methods},
               "by_group": {g: {m: {k: float(x[f"{m}_{k}"].mean()) for k in ("refsign5", "top5", "spearman")}
                                for m in methods} for g, x in df.groupby("group")},
               "repeat": pd.DataFrame(repeat).mean(numeric_only=True).to_dict() if repeat else {}}
    save_json("e2_summary.json", summary)
    log.info("E2 done %s", summary["overall"])


# ----------------------------------------------------------------------------- E3
def _noisy(row: pd.DataFrame, sd: pd.Series, rng) -> pd.DataFrame:
    out = row.copy()
    cols = [c for c in sd.index if c in out.columns and pd.notna(out[c].iloc[0])]
    out[cols] = out[cols].astype(float) + rng.normal(0, NOISE, len(cols)) * sd[cols].to_numpy()
    return out


def e3(data: Data) -> None:
    t0 = time.time()
    sm = sample(data, N_SAMPLE, SEED).sample(N_STAB, random_state=SEED + 2).reset_index(drop=True)
    recent = data.features.loc[data.features["played"].astype(bool)
                               & (data.features["date"] >= data.features["date"].max() - pd.Timedelta(days=730))]
    rng = np.random.default_rng(SEED)
    recs = []
    pool = data.pool()
    for n, snap in enumerate(sm.itertuples(index=False)):
        bundles = data.bundles[snap.model_version]
        cols = [c for c in raw_columns(bundles) if c != "league"]
        sd = recent[cols].std(numeric_only=True)
        groups = factor_groups(raw_columns(bundles))
        means = {}
        for b in bundles.values():
            means.update(b.extra.get("feature_means", {}))
        row = data.row(snap.match_id)
        sels = panel(snap._asdict(), data.match_markets(snap.match_id))
        # the most similar match: nearest stored prediction of the same version in model-output space
        same = pool.loc[(pool["model_version"] == snap.model_version) & (pool["match_id"] != snap.match_id)]
        key = np.column_stack([same["p_home"], same["p_draw"], np.log(same["xg_home"]), np.log(same["xg_away"])])
        me = np.array([snap.p_home, snap.p_draw, np.log(snap.xg_home), np.log(snap.xg_away)])
        twin = same.iloc[int(np.argmin(((key - me) ** 2).sum(1)))]
        twin_row = data.row(twin["match_id"])

        def explain(r):
            gi = chain.chain_inputs(r, bundles, chain.GOAL_OUTPUTS)
            ci = chain.chain_inputs(r, bundles, chain.CORNER_OUTPUTS)
            out = {}
            for market, sel, _ in sels:
                inp = ci if market_group(market) == "CORNERS" else gi
                out[(market, sel)] = {"chain": to_factors(chain.attribute(inp, market, sel).contributions),
                                      "old": old_factors(r, market, sel, bundles)}
            return out

        base = explain(row)
        occ_base = {(m, s): occlusion(row, m, s, bundles, groups, means) for m, s, _ in sels}
        noisy = []
        for _ in range(N_NOISE):
            r = _noisy(row, sd, rng)
            e = explain(r)
            for m, s, _ in sels:
                e[(m, s)]["occl"] = occlusion(r, m, s, bundles, groups, means)
            noisy.append(e)
        twin_e = explain(twin_row)
        for market, sel, role in sels:
            b = {**base[(market, sel)], "occl": occ_base[(market, sel)]}
            rec = {"match_id": snap.match_id, "market": market, "selection": sel, "role": role,
                   "group": market_group(market), "twin": twin["match_id"]}
            for name in ("old", "chain", "occl"):
                B = b[name]
                share = B.abs() / B.abs().sum()
                tops = top(B, 3)
                jac, dshare, flips = [], [], []
                for e in noisy:
                    N = e[(market, sel)][name]
                    nshare = N.abs() / N.abs().sum()
                    jac.append(jaccard(tops, top(N, 3)))
                    dshare.append(float(np.mean([abs(share[k] - nshare.get(k, 0.0)) for k in tops])))
                    flips.append(float(np.mean([np.sign(B[k]) != np.sign(N.get(k, 0.0)) for k in tops])))
                rec[f"{name}_jacc3"] = float(np.mean(jac))
                rec[f"{name}_dshare_pp"] = float(np.mean(dshare) * 100)
                rec[f"{name}_flip"] = float(np.mean(flips))
                if name != "occl":
                    T = twin_e[(market, sel)][name] if (market, sel) in twin_e else None
                    rec[f"{name}_twin_jacc3"] = jaccard(tops, top(T, 3)) if T is not None else np.nan
            recs.append(rec)
        log.info("E3 %d/%d (%.0fs)", n + 1, len(sm), time.time() - t0)
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "e3_stability.csv", index=False)
    cols = [c for c in df.columns if c.split("_", 1)[0] in ("old", "chain", "occl")]
    out = {"matches": int(df["match_id"].nunique()), "selections": int(len(df)), "noise_sd_fraction": NOISE,
           "draws": N_NOISE, "seconds": time.time() - t0,
           "overall": df[cols].mean().round(4).to_dict(),
           "by_group": df.groupby("group")[cols].mean().round(4).reset_index().to_dict("records"),
           "dshare_pp_p90": {n: float(df[f"{n}_dshare_pp"].quantile(0.9)) for n in ("old", "chain", "occl")}}
    save_json("e3_summary.json", out)
    log.info("E3 done %s", out["overall"])


# ----------------------------------------------------------------------------- E4
def _fact_rate(market, selection, rows_h, rows_a) -> tuple[int, int] | None:
    """How often the selection would have won in the teams' recent home / away matches (x, k)."""
    x = k = 0
    for rows, venue in ((rows_h, "H"), (rows_a, "A")):
        if rows.empty:
            continue
        hg, ag = (rows["gf"], rows["ga"]) if venue == "H" else (rows["ga"], rows["gf"])
        corners = (rows["corners_for"] + rows["corners_against"]).to_numpy()
        for h, a, c in zip(hg, ag, corners):
            if market_group(market) == "CORNERS" and c != c:
                continue
            st = settle_selection(market, selection, h, a, c)
            if st is None or st == "PUSH":
                continue
            k += 1
            x += st == "WON"
    return (x, k) if k else None


def e4(data: Data) -> None:
    t0 = time.time()
    sm = sample(data, N_SAMPLE, SEED)
    feats = data.features
    recs, prematch = [], []
    for snap in sm.itertuples(index=False):
        row = data.by_id.loc[snap.match_id]
        hist_h = team_matches(feats, snap.home_team, row["country"], before=row["date"])
        hist_a = team_matches(feats, snap.away_team, row["country"], before=row["date"])
        last = max([d for d in (hist_h["date"].max(), hist_a["date"].max()) if pd.notna(d)], default=pd.NaT)
        # pre-match: the facts use only earlier matches, and the stored form features equal the form
        # recomputed from those matches
        rec5 = {}
        for side, hist in (("h", hist_h), ("a", hist_a)):
            for k in (5, 10):
                if f"{side}_form{k}_gf" in row.index and pd.notna(row[f"{side}_form{k}_gf"]) and len(hist) >= k:
                    rec5[f"{side}_form{k}_gf"] = abs(float(row[f"{side}_form{k}_gf"]) - hist.head(k)["gf"].mean())
        prematch.append({"match_id": snap.match_id, "history_before_kickoff": bool(pd.isna(last) or last < row["date"]),
                         "form_max_abs_diff": max(rec5.values()) if rec5 else np.nan, "form_checked": len(rec5)})
        h_rows = hist_h.loc[(hist_h["venue"] == "H") & hist_h["gf"].notna()].head(10)
        a_rows = hist_a.loc[(hist_a["venue"] == "A") & hist_a["gf"].notna()].head(10)
        for role in ("main", "risk"):
            market, sel = getattr(snap, f"{role}_market"), getattr(snap, f"{role}_selection")
            if not isinstance(market, str):
                continue
            facts = market_facts(market, sel, snap.home_team, snap.away_team, hist_h, hist_a, row)
            rate = _fact_rate(market, sel, h_rows, a_rows)
            p = data.markets.loc[(data.markets["match_id"] == snap.match_id) & (data.markets["market"] == market)
                                 & (data.markets["selection"] == sel), "probability"]
            recs.append({"match_id": snap.match_id, "role": role, "market": market, "selection": sel,
                         "group": market_group(market), "p": float(p.iloc[0]) if len(p) else np.nan,
                         "n_facts": len(facts), "k_home": len(h_rows), "k_away": len(a_rows),
                         "fact_rate": rate[0] / rate[1] if rate else np.nan, "fact_k": rate[1] if rate else 0})
    df = pd.DataFrame(recs)
    pm = pd.DataFrame(prematch)
    df.to_csv(RESULTS / "e4_facts.csv", index=False)
    pm.to_csv(RESULTS / "e4_prematch.csv", index=False)
    ok = df.dropna(subset=["fact_rate", "p"])
    out = {"picks": int(len(df)), "with_facts": float((df["n_facts"] > 0).mean()),
           "short_history_k_below_10": float(((df["k_home"] < 10) | (df["k_away"] < 10)).mean()),
           "rate_vs_p_corr": float(ok["fact_rate"].corr(ok["p"])),
           "agree_direction": float(((ok["fact_rate"] >= 0.5) == (ok["p"] >= 0.5)).mean()),
           "mean_abs_gap_pp": float((ok["fact_rate"] - ok["p"]).abs().mean() * 100),
           "by_group": ok.groupby("group").agg(n=("p", "size"), p=("p", "mean"), rate=("fact_rate", "mean"),
                                               corr=("fact_rate", lambda s: float(s.corr(ok.loc[s.index, "p"])))
                                               ).round(4).reset_index().to_dict("records"),
           "prematch": {"matches": int(len(pm)), "history_before_kickoff": float(pm["history_before_kickoff"].mean()),
                        "form_features_checked": int(pm["form_checked"].sum()),
                        "form_max_abs_diff": float(pm["form_max_abs_diff"].max())},
           "model_inputs_post_match": sorted(set(raw_columns(data.bundles["prod_20261012"]))
                                             & {"home_goals", "away_goals", "result", "total_corners", "home_shots",
                                                "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
                                                "home_yellows", "away_yellows", "home_reds", "away_reds", "played"}),
           "seconds": time.time() - t0}
    live = data.snaps.loc[data.snaps["source"] == "live"]
    kick = pd.to_datetime(live["kickoff"], errors="coerce")
    gen = pd.to_datetime(live["first_generated_at"].fillna(live["generated_at"]), errors="coerce")
    known = kick.notna() & gen.notna()
    out["live_snapshots"] = {"n": int(len(live)), "with_kickoff": int(known.sum()),
                             "first_saved_before_kickoff": float((gen[known] < kick[known]).mean()) if known.any() else None}
    save_json("e4_summary.json", out)
    log.info("E4 done %s", out)


# ----------------------------------------------------------------------------- E5
def e5(data: Data) -> None:
    out = {}
    for mid in EXAMPLES:
        snap = data.snaps.loc[data.snaps["match_id"] == mid].iloc[0]
        bundles = data.bundles[snap["model_version"]]
        row = data.row(mid)
        cols = raw_columns(bundles)
        groups = factor_groups(cols)
        sels = [(snap["main_market"], snap["main_selection"], "main"), (snap["risk_market"], snap["risk_selection"], "risk")]
        bg = _background(data, snap["model_version"], N_BG, SEED)
        ref = permutation_reference(row, bg, bundles, groups, [(m, s) for m, s, _ in sels], N_PERM, SEED)
        items = []
        for i, (market, sel, role) in enumerate(sels):
            ex = chain.explain_market(row, market, sel, bundles)
            old = old_factors(row, market, sel, bundles)
            new = to_factors(ex.contributions)
            R = pd.Series(ref["phi"][i], index=ref["keys"])
            items.append({
                "role": role, "market": market, "selection": sel, "p": ex.probability, "p_base": ex.base_probability,
                "output_value": dict(zip(ex.weights.index, (chain.chain_inputs(row, bundles, chain.outputs_for(market)).u).tolist())),
                "output_effect": ex.output_effect.to_dict(), "model_effect": ex.model_effect().to_dict(),
                "old_top": [(k, float(old[k]), float(abs(old[k]) / old.abs().sum())) for k in top(old, 6)],
                "new_top": [(k, float(new[k]), float(abs(new[k]) / new.abs().sum())) for k in top(new, 6)],
                "ref_top": [(k, float(R[k]), float(abs(R[k]) / R.abs().sum())) for k in top(R, 6)],
                "old_on_new_top": {k: float(old.get(k, 0.0)) for k in top(new, 6)},
                "ref_on_new_top": {k: float(R.get(k, 0.0)) for k in top(new, 6)},
            })
        out[mid] = {"home": snap["home_team"], "away": snap["away_team"], "date": str(snap["date"].date()),
                    "version": snap["model_version"], "picks": items}
    save_json("e5_examples.json", out)
    log.info("E5 done")


def run_info(times: dict) -> None:
    import shap
    import sklearn
    import xgboost
    from scripts.report_kit import git
    prev = {}
    path = RESULTS / "run_info.json"
    if path.exists():
        import json
        prev = json.loads(path.read_text())
    prev.update({"git_sha": git("rev-parse", "HEAD"), "python": platform.python_version(),
                 "shap": shap.__version__, "xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
                 "versions": {v: d.get("trained_to") if d else None for v, d in Data.meta_cache.items()},
                 "seconds": {**prev.get("seconds", {}), **times},
                 "run_at": pd.Timestamp.now().isoformat(timespec="seconds")})
    save_json("run_info.json", prev)


def main(args: list[str]) -> None:
    setup_logging("run_research")
    steps = {"e0": e0, "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}
    chosen = list(steps) if not args or args == ["all"] else args
    data = Data()
    Data.meta_cache = data.meta
    times = {}
    for name in chosen:
        t = time.time()
        steps[name](data)
        times[name] = round(time.time() - t, 1)
        log.info("%s finished in %.0fs", name, times[name])
    run_info(times)
    (CACHE / "run_research.done").write_text("done\n")


if __name__ == "__main__":
    main(sys.argv[1:])
