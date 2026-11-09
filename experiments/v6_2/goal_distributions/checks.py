"""Checks of the production score-matrix code paths (results/production_checks.json).

1. Dixon-Coles as the pipeline would fit it if DIXON_COLES were switched on: src/modelling/pipeline.py fits rho with
   fit_dixon_coles_rho on the RAW goal rates of the validation season, while src/prediction/engine.py applies it
   BEFORE reconcile_with_1x2. The rho that maximises the likelihood of the reconciled matrix differs; this
   measures what the pipeline's rho would do to the reconciled matrix (latent: the switch is off).
2. Tail handling: production folds P(goals > 10) into the 10-goal cell; the mass involved and the effect on the
   score log loss compared with renormalising the truncated matrix.
"""
from __future__ import annotations

import numpy as np

from experiments.v6_2.market_models import data as MD
from experiments.v6_research import scores as SC
from src.markets.distributions import fit_dixon_coles_rho

from . import common as C
from . import dists as D
from .run import fit


def main() -> None:
    C.setup_logging()
    out = {"dc_pipeline_rho": {}, "tail": {}}
    for s in C.SEASONS:
        inner, _ = MD.base(s - 1)
        b, _ = MD.base(s)
        hg, ag = b["home_goals"].to_numpy(), b["away_goals"].to_numpy()
        rho_pipe = fit_dixon_coles_rho(inner["lh"].to_numpy(), inner["la"].to_numpy(),
                                       inner["home_goals"].to_numpy(), inner["away_goals"].to_numpy())
        rho_rec = fit("DC", inner, True)["rho"]
        p1x2 = b[["pH", "pD", "pA"]].to_numpy()
        ref = D.score_loglik(SC.reconcile(SC.matrices(b["lh"], b["la"]), p1x2), hg, ag)
        pipe = D.score_loglik(SC.reconcile(SC.matrices(b["lh"], b["la"], rho=rho_pipe), p1x2), hg, ag)
        rec = D.score_loglik(SC.reconcile(SC.matrices(b["lh"], b["la"], rho=rho_rec), p1x2), hg, ag)
        out["dc_pipeline_rho"][str(s)] = {
            "rho_pipeline_raw_fit": rho_pipe, "rho_ml_reconciled": rho_rec,
            "score_ll_diff_pipeline_vs_production": C.paired(-pipe, -ref, n_boot=1000),
            "score_ll_diff_reconciled_rho_vs_production": C.paired(-rec, -ref, n_boot=1000)}
        lh, la = b["lh"].to_numpy(), b["la"].to_numpy()
        from scipy import stats
        tail = 1 - stats.poisson.cdf(10, lh) + 1 - stats.poisson.cdf(10, la)
        fold = SC.matrices(lh, la)
        ph = stats.poisson.pmf(np.arange(11)[None, :], lh[:, None])
        pa = stats.poisson.pmf(np.arange(11)[None, :], la[:, None])
        renorm = ph[:, :, None] * pa[:, None, :]
        renorm /= renorm.sum((1, 2), keepdims=True)
        out["tail"][str(s)] = {"max_tail_mass": float(tail.max()), "mean_tail_mass": float(tail.mean()),
                               "max_abs_cell_diff": float(np.abs(fold - renorm).max()),
                               "score_ll_diff_renorm_vs_fold": C.paired(-D.score_loglik(renorm, hg, ag),
                                                                        -D.score_loglik(fold, hg, ag), n_boot=1000)}
    C.save_json("production_checks.json", out)


if __name__ == "__main__":
    main()
