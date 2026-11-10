"""Evaluates the temporal feature-set variants on the same matches (reference T0 = production features, 3 seeds).

Per season and variant: 1X2 log loss, goal-rate Poisson deviance (home + away), the production Main (hit, gap,
ROI at the Bet365 opening price). Selection = 2023/24 + 2024/25 pooled; Holm across all variants on the 1X2 log
loss (primary) and, separately, on the goal deviance. Breakdowns: per league, by phase of the season (both teams
with ≤ 5 league games this season = "early"), by tier. The decision rule is the league study's
(experiments/v6_2/leagues/evaluate.py: A / B / C), applied to the 1X2 log loss, or to the goal deviance when the
variant improves only the goals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_2.leagues import evaluate as LE
from experiments.v6_2.leagues import wf
from experiments.v6_2.leagues.common import tier

from . import common as C
from . import variants as V


def season_rows(s: int) -> pd.DataFrame:
    path = C.CACHE / f"eval_rows_{s}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fr = wf.frame(s)
    ev_mask = fr["in_eval"].to_numpy()
    ev = fr.loc[ev_mask].reset_index(drop=True)
    y = ev["y"].to_numpy(int)
    feats = wf.season(wf.features(), s).set_index("match_id")
    early = ((feats["h_season_games"] <= 5) & (feats["a_season_games"] <= 5)).reindex(ev["match_id"]).to_numpy()
    preds = {"B0": wf.base_predictions(s, C.PROD_SEED), "T0": V.averaged("T0", s)}
    for v in V.VARIANTS:
        preds[v] = V.averaged(v, s)
    out = []
    for v, p in preds.items():
        p = fr[["match_id"]].merge(p, on="match_id", how="left")
        pe = p.loc[ev_mask].reset_index(drop=True)
        r = pd.DataFrame({"match_id": ev["match_id"], "league": ev["league"], "season": s, "variant": v,
                          "early": early, "ll": C.logloss_rows(wf.P(pe), y),
                          "dev": C.poisson_dev_rows(ev["home_goals"], pe["lh"]) + C.poisson_dev_rows(ev["away_goals"], pe["la"]),
                          "ll_mkt": C.logloss_rows(wf.Q(ev), y)})
        o = wf.main_outcome(fr, p, s)
        r = r.merge(o[["match_id", "p", "won", "profit", "decided", "priced", "group"]].rename(
            columns={"p": "m_p", "won": "m_won", "profit": "m_profit", "decided": "m_decided", "priced": "m_priced",
                     "group": "m_group"}), on="match_id", how="left")
        out.append(r)
        C.log.info("season %s variant %s evaluated", s, v)
    rows = pd.concat(out, ignore_index=True)
    rows.to_parquet(path)
    return rows


def summarise(rows: pd.DataFrame) -> dict:
    seasons = sorted(rows["season"].unique())
    by = {(s, v): g.sort_values("match_id").reset_index(drop=True) for (s, v), g in rows.groupby(["season", "variant"])}
    res = {"n_matches": {str(s): int(len(by[(s, "T0")])) for s in seasons}, "reference": {}, "variants": {}}
    for ref in ("B0", "T0"):
        rr = {}
        for s in seasons:
            g = by[(s, ref)]
            rr[str(s)] = {"ll": float(g["ll"].mean()), "dev": float(g["dev"].mean()),
                          "vs_market": C.paired_diff_ci(g["ll_mkt"], g["ll"]), "main": LE._main(g)}
        sel = pd.concat([by[(s, ref)] for s in C.SELECT])
        rr["select"] = {"ll": float(sel["ll"].mean()), "dev": float(sel["dev"].mean()),
                        "vs_market": C.paired_diff_ci(sel["ll_mkt"], sel["ll"]), "main": LE._main(sel)}
        res["reference"][ref] = rr
    res["seed_noise"] = {str(s): {"d_ll": C.paired_diff_ci(by[(s, "B0")]["ll"], by[(s, "T0")]["ll"]),
                                  "d_dev": C.paired_diff_ci(by[(s, "B0")]["dev"], by[(s, "T0")]["dev"])} for s in seasons}
    p_ll, p_dev = {}, {}
    for v, name in V.VARIANTS.items():
        out = {"name": name, "seasons": {}}
        for s in seasons:
            a, b = by[(s, "T0")], by[(s, v)]
            out["seasons"][str(s)] = {"d_ll": C.paired_diff_ci(a["ll"], b["ll"]), "d_dev": C.paired_diff_ci(a["dev"], b["dev"]),
                                      "vs_market": C.paired_diff_ci(b["ll_mkt"], b["ll"]), "main": LE._main(b),
                                      "d_roi": LE._roi_diff(a, b), "d_gap": LE._gap(b)["gap"] - LE._gap(a)["gap"]}
        a = pd.concat([by[(s, "T0")] for s in C.SELECT], ignore_index=True)
        b = pd.concat([by[(s, v)] for s in C.SELECT], ignore_index=True)
        sel = {"d_ll": C.paired_diff_ci(a["ll"], b["ll"]), "d_dev": C.paired_diff_ci(a["dev"], b["dev"]),
               "vs_market": C.paired_diff_ci(b["ll_mkt"], b["ll"]), "main": LE._main(b), "d_roi": LE._roi_diff(a, b),
               "d_gap": LE._gap(b)["gap"] - LE._gap(a)["gap"]}
        for flag, lab in ((True, "early"), (False, "late")):
            m = (a["early"] == flag).to_numpy()
            sel[f"d_ll_{lab}"] = C.paired_diff_ci(a.loc[m, "ll"], b.loc[m, "ll"])
            sel[f"d_dev_{lab}"] = C.paired_diff_ci(a.loc[m, "dev"], b.loc[m, "dev"])
        tiers = a["league"].map(tier)
        sel["d_ll_tier"] = {str(t): C.paired_diff_ci(a.loc[(tiers == t).to_numpy(), "ll"], b.loc[(tiers == t).to_numpy(), "ll"])
                            for t in sorted(tiers.unique())}
        d = (b["ll"] - a["ll"]).groupby(a["league"])
        sel["leagues"] = {lg: {"n": int(len(x)), "mean": float(x.mean()), "ci95": C.mean_ci(x)["ci95"]} for lg, x in d}
        out["select"] = sel
        p_ll[v], p_dev[v] = sel["d_ll"]["p_two_sided"], sel["d_dev"]["p_two_sided"]
        res["variants"][v] = out
    adj_ll, adj_dev = C.holm(p_ll), C.holm(p_dev)
    for v, out in res["variants"].items():
        out["select"]["p_holm_ll"], out["select"]["p_holm_dev"] = adj_ll.get(v), adj_dev.get(v)
        dec_ll = LE.decide({**out, "metric": "ll", "select": {**out["select"], "p_holm": adj_ll.get(v)}})
        dec_dev = LE.decide({**out, "metric": "dev", "select": {**out["select"], "p_holm": adj_dev.get(v)}})
        order = {"A": 0, "B": 1, "C": 2}
        out["decision_ll"], out["decision_dev"] = dec_ll, dec_dev
        out["decision"] = min(dec_ll, dec_dev, key=order.get)
    res["n_variants_tested"] = len(p_ll)
    res["feature_counts"] = V.feature_counts()
    C.save_json("variants.json", res)
    return res


def run() -> dict:
    rows = pd.concat([season_rows(s) for s in C.EVAL], ignore_index=True)
    return summarise(rows)


if __name__ == "__main__":
    C.setup_logging("evaluate")
    run()
    print("EVAL_DONE")
