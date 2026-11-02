"""Clean training-time benchmark (run alone, sequentially): one fit per model kind on fold 2024
(training rows = seasons < 2024), the tree counts / settings found by the studies.
The per-fold times inside the study JSONs were measured while other studies ran in parallel."""
from __future__ import annotations

import re
import time

import numpy as np

from experiments.model_comparison.models import NegBinGLMCandidate, PoissonGLMCandidate

from . import common as C
from . import corners as K
from . import draw as D

log = C.setup_logging("timing")
SEASON = 2024


def timed(fn, repeat=1):
    out = []
    for _ in range(repeat):
        t0 = time.time()
        fn()
        out.append(time.time() - t0)
    return float(np.median(out))


def run() -> dict:
    df = C.features()
    tr = C.before(df, SEASON)
    y = C.target("result", tr)
    dres, cres = C.load_json("draw.json"), C.load_json("corners.json")
    res = {"season_fold": SEASON, "train_rows": int(len(tr)), "seconds": {}}
    S = res["seconds"]
    S["result: production XGBoost"] = timed(lambda: C.production_model("result").fit(tr, y), 2)
    S["result: draw-weighted refit (b)"] = timed(lambda: C.production_model("result").fit(
        tr, y, sample_weight=np.where(y == 1, 1.2, 1.0)), 2)
    full_abs, cols = D.add_abs_features(df)
    tra = C.before(full_abs, SEASON)
    n = dres["folds"][str(SEASON)]["params"]["d_draw_model_blend"]["n_trees"]
    rp, _ = C.prod_params("result")
    S["result: binary draw model (d), one fit"] = timed(lambda: C.XGBFit(
        C.task_numeric("result") + cols, rp, n, "binary:logistic", "logloss", True).fit(tra, tra["result"] == "D"), 2)
    inner, outer, _ = C.base_predictions(SEASON)
    p = inner[["pH", "pD", "pA"]].to_numpy()
    yi = inner["y"].to_numpy()
    S["calibration: temperature fit"] = timed(lambda: D.fit_temperature(p, yi))
    S["calibration: vector scaling fit"] = timed(lambda: D.fit_vector(p, yi))
    for task in ("home_goals", "away_goals"):
        yt = tr[task].to_numpy(float)
        S[f"{task}: production XGBoost"] = timed(lambda: C.production_model(task).fit(tr, yt), 2)
        g = PoissonGLMCandidate("count", C.task_numeric(task))
        g.params = {"regularisation": 1.0}
        S[f"{task}: Poisson GLM"] = timed(lambda: g.fit(tr, yt))
    dfx, extra = K.extra_features(df)
    S["corners: extra feature computation (all seasons)"] = timed(lambda: K.extra_features(df))
    cl = C.fold_corner_leagues(tr)
    ct = dfx.loc[(dfx["season"] < SEASON) & dfx["total_corners"].notna() & dfx["league"].isin(cl)]
    yc = ct["total_corners"].to_numpy(float)
    params, _ = C.prod_params("corners")
    prm = cres["folds"][str(SEASON)]["params"]
    S["corners: production XGBoost"] = timed(lambda: C.production_model("corners").fit(ct, yc))
    S["corners: XGBoost enriched"] = timed(lambda: C.XGBFit(C.task_numeric("corners") + extra, params,
                                                            prm["xgb_enriched"]["n_trees"], "count:poisson",
                                                            "poisson-nloglik", False).fit(ct, yc))
    S["corners: home + away XGBoost (2 fits)"] = timed(lambda: [C.XGBFit(
        C.task_numeric("corners") + extra, params, prm["xgb_home_away"][f"{side}_n_trees"], "count:poisson",
        "poisson-nloglik", False).fit(ct.loc[ct[side].notna()], ct.loc[ct[side].notna(), side])
        for side in ("home_corners", "away_corners")])
    g = PoissonGLMCandidate("count", C.task_numeric("corners"))
    g.params = {"regularisation": prm["poisson_glm"]["l2"]}
    S["corners: Poisson GLM"] = timed(lambda: g.fit(ct, yc))
    clean = K.numeric_from_groups(K.CLEAN_GROUPS, list(dfx.columns)) + [
        c for c in dfx.columns if re.search(r"^[ha]_form10_(shots_for|shots_against|sot_for|sot_against)$", c)] + extra
    nb = NegBinGLMCandidate("count", clean)
    nb.params = {"regularisation": 1e-2}
    S["corners: negative-binomial GLM"] = timed(lambda: nb.fit(ct, yc))
    C.save_json("timing.json", res)
    for k, v in S.items():
        log.info("%-50s %.1f s", k, v)
    return res


if __name__ == "__main__":
    run()
