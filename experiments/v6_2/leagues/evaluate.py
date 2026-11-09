"""Evaluates every league-specialisation variant on the same matches.

Per season s (EVAL) and variant v: per-match 1X2 log loss, Poisson deviance of the goal rates (home + away) and the
production Main (hit, claimed probability, profit at the Bet365 opening price). References:
    post-hoc variants (C*, G*)  ->  B0  (production, seed 42): they are applied to its predictions
    refitted variants (M*)      ->  B0̄  (production setup averaged over the same three seeds)
Selection = 2023/24 + 2024/25 pooled; 2025/26 for information; 2026/27 current (already looked at).
Holm correction across all variants on the pooled-selection primary metric (1X2 log loss; goal deviance for G*).

Decision rule (fixed before looking at the results):
    A  primary metric better on the pooled selection with Holm p < 0.05, better in BOTH selection seasons, not worse
       on 2025/26 (point estimate), and Main ROI not significantly worse (95 % CI of ΔROI not entirely < 0)
    B  better in both selection seasons and unadjusted 95 % CI entirely < 0 (fails Holm or 2025/26)
    C  otherwise
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calib as CB
from . import common as C
from . import goalpool as GP
from . import models as MD
from . import profiles as PR
from . import wf

POSTHOC = {**CB.VARIANTS, **GP.VARIANTS}
REFIT = dict(MD.VARIANTS)


def _aligned(fr: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    return fr[["match_id"]].merge(pred, on="match_id", how="left")


def history(s: int) -> pd.DataFrame:
    """Walk-forward B0 predictions of 2022/23 ... s-1 with results (calibration data for season s)."""
    rows = []
    for t in range(C.FIRST_OOS, s):
        fr = wf.frame(t)
        p = _aligned(fr, wf.base_predictions(t, C.PROD_SEED))
        rows.append(pd.concat([fr[["match_id", "league", "season", "y", "home_goals", "away_goals"]],
                               p[["pH", "pD", "pA", "lh", "la"]]], axis=1))
    return pd.concat(rows, ignore_index=True)


def season_variants(s: int) -> tuple[pd.DataFrame, dict]:
    """Per-match rows of every variant for season s (evaluation matches only)."""
    path = C.CACHE / f"eval_rows_{s}.parquet"
    pinfo = C.CACHE / f"eval_info_{s}.json"
    if path.exists() and pinfo.exists():
        import json
        return pd.read_parquet(path), json.loads(pinfo.read_text())
    fr = wf.frame(s)
    ev_mask = fr["in_eval"].to_numpy()
    ev = fr.loc[ev_mask].reset_index(drop=True)
    y = ev["y"].to_numpy(int)
    groups = PR.groups_for(s)
    preds: dict[str, pd.DataFrame] = {}
    b0 = _aligned(fr, wf.base_predictions(s, C.PROD_SEED))
    preds["B0"] = b0
    for seed in C.SEEDS[1:]:
        preds[f"B0_seed{seed}"] = _aligned(fr, wf.base_predictions(s, seed))
    preds["B0bar"] = _aligned(fr, MD.base_averaged(s))
    hist = history(s)
    cal = CB.Calibrator(hist[["pH", "pD", "pA"]].to_numpy(), hist["y"].to_numpy(), hist["league"].to_numpy(),
                        hist["league"].map(groups).to_numpy()).fit_all()
    gp = GP.GoalPool(hist)
    lg_all, gr_all = fr["league"].to_numpy(), fr["league"].map(groups).to_numpy()
    for v in CB.VARIANTS:
        p = b0.copy()
        p[["pH", "pD", "pA"]] = cal.transform(wf.P(b0), lg_all, gr_all, v)
        preds[v] = p
    for v in GP.VARIANTS:
        p = b0.copy()
        p["lh"], p["la"] = gp.transform(b0["lh"], b0["la"], lg_all, v)
        preds[v] = p
    for v in MD.VARIANTS:
        preds[v] = _aligned(fr, MD.averaged(v, s))
    out = []
    for v, p in preds.items():
        pe = p.loc[ev_mask].reset_index(drop=True)
        r = pd.DataFrame({"match_id": ev["match_id"], "league": ev["league"], "season": s, "variant": v,
                          "ll": C.logloss_rows(wf.P(pe), y),
                          "dev": C.poisson_dev_rows(ev["home_goals"], pe["lh"]) + C.poisson_dev_rows(ev["away_goals"], pe["la"]),
                          "ll_mkt": C.logloss_rows(wf.Q(ev), y)})
        if not v.startswith("B0_seed"):
            o = wf.main_outcome(fr, p, s)
            r = r.merge(o[["match_id", "p", "won", "profit", "decided", "priced", "group"]].rename(
                columns={"p": "m_p", "won": "m_won", "profit": "m_profit", "decided": "m_decided",
                         "priced": "m_priced", "group": "m_group"}), on="match_id", how="left")
        out.append(r)
        C.log.info("season %s variant %s evaluated", s, v)
    rows = pd.concat(out, ignore_index=True)
    info = {"calibration": cal.info, "goals": gp.info(), "groups": groups, "hist_rows": int(len(hist))}
    rows.to_parquet(path)
    import json
    pinfo.write_text(json.dumps(C.clean(info)))
    return rows, info


def _roi_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """ROI(b) − ROI(a) on the same matches, bootstrap over matches (profit NaN = no bet)."""
    pa, pb = a["m_profit"].to_numpy(float), b["m_profit"].to_numpy(float)
    rng = np.random.default_rng(C.SEED)
    n, ds = len(pa), []
    for _ in range(C.N_BOOT // 250):
        idx = rng.integers(0, n, size=(250, n))
        with np.errstate(all="ignore"):
            ds.append(np.nanmean(pb[idx], 1) - np.nanmean(pa[idx], 1))
    d = np.concatenate(ds)
    return {"diff": float(np.nanmean(pb) - np.nanmean(pa)), "ci95": [float(np.nanquantile(d, 0.025)),
                                                                    float(np.nanquantile(d, 0.975))]}


def _gap(r: pd.DataFrame) -> dict:
    dec = r.loc[r["m_decided"].fillna(False).astype(bool)]
    return {"n": int(len(dec)), "gap": float((dec["m_p"] - dec["m_won"]).mean()), "hit": float(dec["m_won"].mean()),
            "p": float(dec["m_p"].mean())}


def _main(r: pd.DataFrame) -> dict:
    bets = r.loc[r["m_profit"].notna()]
    return {**_gap(r), "bets": int(len(bets)), "roi": C.mean_ci(bets["m_profit"])}


def summarise(rows: pd.DataFrame) -> dict:
    res = {"variants": {}, "reference": {}, "n_matches": {}}
    seasons = sorted(rows["season"].unique())
    by = {(s, v): g.sort_values("match_id").reset_index(drop=True) for (s, v), g in rows.groupby(["season", "variant"])}
    for s in seasons:
        res["n_matches"][str(s)] = int(len(by[(s, "B0")]))
    # reference rows: B0 and B0bar vs market, Main
    for ref in ("B0", "B0bar"):
        rr = {}
        for s in seasons:
            g = by[(s, ref)]
            rr[str(s)] = {"ll": float(g["ll"].mean()), "dev": float(g["dev"].mean()),
                          "vs_market": C.paired_diff_ci(g["ll_mkt"], g["ll"]), "main": _main(g)}
        sel = pd.concat([by[(s, ref)] for s in C.SELECT])
        rr["select"] = {"ll": float(sel["ll"].mean()), "dev": float(sel["dev"].mean()),
                        "vs_market": C.paired_diff_ci(sel["ll_mkt"], sel["ll"]), "main": _main(sel)}
        res["reference"][ref] = rr
    # seed noise of the refits: single seeds vs seed 42, and the 3-seed mean vs seed 42
    noise = {}
    for v in [f"B0_seed{x}" for x in C.SEEDS[1:]] + ["B0bar"]:
        noise[v] = {str(s): C.paired_diff_ci(by[(s, "B0")]["ll"], by[(s, v)]["ll"]) for s in seasons}
        noise[v]["dev"] = {str(s): C.paired_diff_ci(by[(s, "B0")]["dev"], by[(s, v)]["dev"]) for s in seasons}
    res["seed_noise"] = noise
    pvals = {}
    for v in list(POSTHOC) + list(REFIT):
        ref = "B0" if v in POSTHOC else "B0bar"
        metric = "dev" if v.startswith("G") else "ll"
        out = {"ref": ref, "metric": metric, "name": {**POSTHOC, **REFIT}[v], "seasons": {}}
        for s in seasons:
            a, b = by[(s, ref)], by[(s, v)]
            out["seasons"][str(s)] = {"d_ll": C.paired_diff_ci(a["ll"], b["ll"]),
                                      "d_dev": C.paired_diff_ci(a["dev"], b["dev"]),
                                      "vs_market": C.paired_diff_ci(b["ll_mkt"], b["ll"]),
                                      "main": _main(b), "d_roi": _roi_diff(a, b),
                                      "d_gap": _gap(b)["gap"] - _gap(a)["gap"]}
        a = pd.concat([by[(s, ref)] for s in C.SELECT], ignore_index=True)
        b = pd.concat([by[(s, v)] for s in C.SELECT], ignore_index=True)
        sel = {"d_ll": C.paired_diff_ci(a["ll"], b["ll"]), "d_dev": C.paired_diff_ci(a["dev"], b["dev"]),
               "vs_market": C.paired_diff_ci(b["ll_mkt"], b["ll"]), "main": _main(b), "d_roi": _roi_diff(a, b),
               "d_gap": _gap(b)["gap"] - _gap(a)["gap"]}
        # per league (pooled selection)
        d = (b[metric] - a[metric]).groupby(a["league"])
        sel["leagues"] = {lg: {"n": int(len(x)), "mean": float(x.mean()), "ci95": C.mean_ci(x)["ci95"]} for lg, x in d}
        out["select"] = sel
        pvals[v] = sel[f"d_{metric}"]["p_two_sided"]
        res["variants"][v] = out
    adj = C.holm(pvals)
    for v, out in res["variants"].items():
        out["select"]["p_holm"] = adj.get(v)
        out["decision"] = decide(out)
    res["n_variants_tested"] = len(pvals)
    C.save_json("variants.json", res)
    return res


def decide(out: dict) -> str:
    m = "d_" + out["metric"]
    sel = out["select"][m]
    each = [out["seasons"][str(s)][m]["diff"] for s in C.SELECT]
    info = out["seasons"].get(str(C.INFO), {}).get(m, {}).get("diff")
    roi_ci = out["select"]["d_roi"]["ci95"]
    both = all(x is not None and x < 0 for x in each)
    if (sel["diff"] < 0 and (out["select"].get("p_holm") or 1) < 0.05 and both and info is not None and info <= 0
            and not (roi_ci[1] is not None and roi_ci[1] < 0)):
        return "A"
    if both and sel["ci95"][1] is not None and sel["ci95"][1] < 0:
        return "B"
    return "C"


def run() -> dict:
    rows = []
    infos = {}
    for s in C.EVAL:
        r, info = season_variants(s)
        rows.append(r)
        infos[str(s)] = info
    rows = pd.concat(rows, ignore_index=True)
    res = summarise(rows)
    C.save_json("variant_fits.json", infos)
    return res


if __name__ == "__main__":
    C.setup_logging("evaluate")
    run()
    print("EVAL_DONE")
