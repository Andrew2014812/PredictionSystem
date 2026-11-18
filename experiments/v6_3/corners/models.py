"""Light mean models (no tree fitting) and the assembly of every model's expected corners.

    rating     dynamic team corner ratings (data.dynamic_ratings): μ = μ_h + μ_a, updated after every match day
    glm        Poisson GLM (sklearn PoissonRegressor, log link) on a compact standardised feature set
    naive      league average of earlier matches (feature league_avg_corners)
    ens_*      convex combinations of model means, weights chosen on 2025/26 (select stage) only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import PoissonRegressor

from experiments.v6_3.corners import data as D

GLM_COLS = ["league_avg_corners", "league_avg_home_corners", "league_avg_away_corners", "rc_mu_h", "rc_mu_a",
            "rs_mu_h", "rs_mu_a", "h_x15_cf", "h_x15_ca", "a_x15_cf", "a_x15_ca", "h_xv5_cf", "a_xv5_ca",
            "h_xv5_ca", "a_xv5_cf", "h_x10_sf", "a_x10_sf", "h_x10_sa", "a_x10_sa", "h_x10_tf", "a_x10_tf",
            "h_x10_ff", "a_x10_ff", "n_exp_total_add", "n_exp_total_mult", "n_tot_hist", "n_abs_elo",
            "m_abs_diff", "m_over25", "m_exp_goals", "m_diff", "h_form10_corner_share", "a_form10_corner_share"]


class GLM:
    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    def _X(self, rows: pd.DataFrame) -> np.ndarray:
        X = rows.reindex(columns=GLM_COLS).astype(float)
        X["log_rc"] = np.log(np.clip(rows["rc_mu_h"] + rows["rc_mu_a"], 1, None))
        X["log_add"] = np.log(np.clip(rows["n_exp_total_add"].astype(float), 1, None))
        miss = X.isna().astype(float).add_suffix("_na")
        X = X.fillna(self.med)
        Z = (X - self.mean) / self.sd
        return np.hstack([Z.to_numpy(), miss[self.miss_cols].to_numpy()])

    def fit(self, rows: pd.DataFrame, y) -> "GLM":
        X = rows.reindex(columns=GLM_COLS).astype(float)
        X["log_rc"] = np.log(np.clip(rows["rc_mu_h"] + rows["rc_mu_a"], 1, None))
        X["log_add"] = np.log(np.clip(rows["n_exp_total_add"].astype(float), 1, None))
        self.med = X.median()
        miss = X.isna().astype(float).add_suffix("_na")
        self.miss_cols = [c for c in miss.columns if miss[c].sum() > 50]
        Xf = X.fillna(self.med)
        self.mean, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        from src.modelling.training import recency_weights
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(
            self._X(rows), np.asarray(y, float), sample_weight=recency_weights(rows["date"], 2))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._X(rows))


def trainable(df: pd.DataFrame, leagues) -> pd.DataFrame:
    return df.loc[df["played"].astype(bool) & df["total_corners"].notna() & df["league"].isin(leagues)]


def glm_predictions(df: pd.DataFrame, leagues, cut: pd.Timestamp, target: pd.DataFrame) -> np.ndarray:
    tr = trainable(df, leagues)
    tr = tr.loc[tr["date"] < cut]
    return GLM().fit(tr, tr["total_corners"]).predict(target)


def convex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights on the simplex minimising the Poisson deviance of Σ w_k μ_k."""
    k = P.shape[1]

    def f(z):
        w = np.exp(z) / np.exp(z).sum()
        return D.poisson_dev(y, P @ w)
    r = optimize.minimize(f, np.zeros(k), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
    return np.exp(r.x) / np.exp(r.x).sum()
