"""Variants compared with the v6.1 baseline (step 3-4 of the study).

All in-season models are refitted every REFIT_DAYS days on the 2026/27 matches played BEFORE the first day
of the week they predict (expanding window); the long-history models are fitted once on 2021/22-2025/26.
Every hyper-parameter below was fixed before the evaluation was run; nothing is tuned on the evaluation weeks.

    B0   production v6.1 (prod_20261012) as is
    R0   control: B0 recalibrated in-season WITHOUT xG (goals: log mu = a + b log lambda_B0;
         1X2: multinomial logit with offset log p_B0 on the B0 log-odds)
    X1   pure xG rating model (opponent-adjusted xG ratings, Poisson matrix; no B0)
    G1   control: the same rating model on goals instead of xG
    X2   Poisson GLM correction of B0 with xG features (ridge), 1X2 by propagating the goal change
    X3   blend of B0 with the xG rating: log mu = a + b log lambda_B0 + c log E[xG]
    X4   XGBoost residual model on xG features with base margin log lambda_B0 (small, fixed trees)
    M1   1X2 correction with xG features (multinomial logit with offset log p_B0), goals of B0
    X2M  X2 goals + M1 1X2 (the full "xG on top of v6.1" candidate)
    S0   short history: production XGBoost settings fitted on 2026/27 rows only, v6.1 features
    S1   S0 + xG features
    T0   control: production XGBoost refitted on 2021/22-2025/26 (should reproduce B0)
    T1   T0 + shot-based proxy of xG ("transfer": xG ~ shots + shots on target fitted on August 2026,
         applied to the existing form features of all seasons) — a proxy, not xG

Added after the first pass (the first specification left the intercept and the B0 slope unpenalised and
was noisier than B0 even without xG); they are counted in the number of variants tried:

    R0a, X2a, X3a, M1a, X2Ma   "anchored" versions: offset = B0, EVERY coefficient penalised, i.e. shrunk
                               towards the unchanged baseline
    X5   one-parameter geometric blend: log mu = (1 - w) log lambda_B0 + w log E[xG], w in [0, 1]

1X2 of a goal-only variant: p_k = p_B0,k * P_k(lambda_new) / P_k(lambda_B0), renormalised, where P_k is the
Poisson outcome probability; this keeps the classifier and moves it in the direction of the goal change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import common as C
from . import data as D
from . import xg_features as F

GLM_RIDGE = 10.0
MNL_RIDGE = 10.0
XGB_RESID = dict(max_depth=2, learning_rate=0.05, n_estimators=150, min_child_weight=30, subsample=0.8,
                 colsample_bytree=1.0, reg_lambda=5.0, objective="count:poisson", tree_method="hist")
MAXG = 10

IN_SEASON = ("R0", "R0a", "X1", "G1", "X2", "X2a", "X3", "X3a", "X4", "X5", "M1", "M1a", "X2M", "X2Ma")
VARIANTS = ["B0", *IN_SEASON, "S0", "S1", "T0", "T1"]
CONTROLS = {"R0", "R0a", "G1", "S0", "T0"}
LABELS = {
    "B0": "v6.1 (база)", "R0": "контроль: перекалибровка без xG", "X1": "рейтинги xG (чистая модель)",
    "G1": "контроль: рейтинги по голам", "X2": "GLM-коррекция xG", "X3": "смесь v6.1 и рейтинга xG",
    "X4": "XGBoost-остаток по xG", "M1": "коррекция 1X2 по xG", "X2M": "X2 + M1 (полный xG)",
    "S0": "короткая история без xG", "S1": "короткая история + xG", "T0": "контроль: переобучение v6.1",
    "T1": "перенос: прокси xG по ударам", "R0a": "контроль: перекалибровка без xG (якорь)",
    "X2a": "GLM-коррекция xG (якорь)", "X3a": "смесь с рейтингом xG (якорь)", "M1a": "коррекция 1X2 по xG (якорь)",
    "X2Ma": "X2a + M1a (полный xG, якорь)", "X5": "геометрическая смесь с рейтингом xG (1 параметр)"}


# ------------------------------------------------------------------------------------------------ helpers
def pois_1x2(lh, la) -> np.ndarray:
    """Independent-Poisson H/D/A probabilities, vectorised."""
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], np.asarray(lh)[:, None])
    pa = poisson.pmf(k[None, :], np.asarray(la)[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    i, j = np.indices((MAXG + 1, MAXG + 1))
    out = np.stack([m[:, i > j].sum(1), m[:, i == j].sum(1), m[:, i < j].sum(1)], 1)
    return out / out.sum(1, keepdims=True)


def propagate(p0: np.ndarray, lh0, la0, lh1, la1) -> np.ndarray:
    r = pois_1x2(lh1, la1) / np.clip(pois_1x2(lh0, la0), 1e-12, None)
    p = p0 * r
    return p / p.sum(1, keepdims=True)


def fit_poisson_glm(X: np.ndarray, y: np.ndarray, offset: np.ndarray, penalize: np.ndarray, ridge: float) -> np.ndarray:
    """min sum(mu - y log mu) + ridge/2 * sum(beta[penalize]^2), log mu = offset + X beta."""
    X = np.asarray(X, float)

    def f(beta):
        eta = np.clip(offset + X @ beta, -10, 5)
        mu = np.exp(eta)
        loss = np.sum(mu - y * eta) + 0.5 * ridge * np.sum((beta * penalize) ** 2)
        return loss, X.T @ (mu - y) + ridge * beta * penalize

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def fit_mnl(X: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float, penalize: np.ndarray) -> np.ndarray:
    """Multinomial logit with offset: eta_k = offset_k + X w_k for k in (H, A); D is the reference (w_D = 0).
    Returns W of shape (2, n_features)."""
    X = np.asarray(X, float)
    n, p = X.shape
    Y = np.eye(3)[y]

    def f(w):
        W = w.reshape(2, p)
        eta = offset.copy()
        eta[:, 0] += X @ W[0]
        eta[:, 2] += X @ W[1]
        eta -= eta.max(1, keepdims=True)
        P = np.exp(eta)
        P /= P.sum(1, keepdims=True)
        loss = -np.sum(Y * np.log(np.clip(P, 1e-15, 1))) + 0.5 * ridge * np.sum((W * penalize) ** 2)
        R = P - Y
        g = np.stack([X.T @ R[:, 0], X.T @ R[:, 2]]) + ridge * W * penalize
        return loss, g.ravel()

    return minimize(f, np.zeros(2 * p), jac=True, method="L-BFGS-B").x.reshape(2, p)


def apply_mnl(W: np.ndarray, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    eta = offset.copy()
    eta[:, 0] += X @ W[0]
    eta[:, 2] += X @ W[1]
    eta -= eta.max(1, keepdims=True)
    P = np.exp(eta)
    return P / P.sum(1, keepdims=True)


def _log(x):
    return np.log(np.clip(np.asarray(x, float), 1e-3, None))


# ------------------------------------------------------------------------------------------------ design
def frame(lag_days: int = 0) -> pd.DataFrame:
    """B0 + xG features for the 2026/27 matches of the xG leagues, with the transformed inputs."""
    b = D.baseline()
    f = F.build(lag_days)
    df = b.merge(f.drop(columns=["league", "date"]), on="match_id", how="inner")
    prior = df[["lg_xg_h", "lg_xg_a"]].mean(axis=1)
    df["zh1"] = _log(df["exp_xg_h"]) - _log(df["lg_xg_h"])
    df["za1"] = _log(df["exp_xg_a"]) - _log(df["lg_xg_a"])
    df["zh2"] = _log(df["h_xgf_s"]) - _log(prior)
    df["za2"] = _log(df["a_xgf_s"]) - _log(prior)
    df["zh3"] = _log(df["a_xga_s"]) - _log(prior)
    df["za3"] = _log(df["h_xga_s"]) - _log(prior)
    df["zh4"], df["za4"] = df["h_gmx_s"], df["a_gmx_s"]
    df["d1"] = _log(df["exp_xg_h"]) - _log(df["exp_xg_a"])
    df["d2"] = (_log(df["h_xgf_s"]) - _log(df["h_xga_s"])) - (_log(df["a_xgf_s"]) - _log(df["a_xga_s"]))
    df["d3"] = _log(df["exp_xg_h"]) + _log(df["exp_xg_a"]) - _log(df["lg_xg_h"]) - _log(df["lg_xg_a"])
    df["d4"] = df["h_gmx_s"] - df["a_gmx_s"]
    df["lb_h"], df["lb_a"] = _log(df["lh"]), _log(df["la"])
    df["lo_H"] = np.log(df["pH"] / df["pD"])
    df["lo_A"] = np.log(df["pA"] / df["pD"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


ZH = ["zh1", "zh2", "zh3", "zh4"]
ZA = ["za1", "za2", "za3", "za4"]
D1X2 = ["d1", "d2", "d3", "d4"]


def weeks(df: pd.DataFrame) -> list[pd.Timestamp]:
    last = df.loc[df["played"].astype(bool), "date"].max()
    out, w = [], C.EVAL_START
    while w <= last:
        out.append(w)
        w += pd.Timedelta(days=C.REFIT_DAYS)
    return out


# ------------------------------------------------------------------------------------------------ in-season variants
def _goal_glm(tr, te, side: str, feats: list[str], with_base: bool, penal_feats: bool,
              anchored: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Poisson GLM of the goals of one side.

    free:      log mu = a + b log lambda_B0 + c z        (a, b unpenalised; c penalised if penal_feats)
    anchored:  log mu = log lambda_B0 + a + (b - 1) log lambda_B0 + c z, every coefficient penalised,
               i.e. shrunk towards the unchanged baseline (a = 0, b = 1, c = 0)."""
    y = tr["home_goals" if side == "h" else "away_goals"].to_numpy(float)
    base = "lb_h" if side == "h" else "lb_a"
    cols = ([base] if with_base else []) + feats
    Xtr = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)]
    Xte = np.c_[np.ones(len(te)), te[cols].to_numpy(float)]
    if anchored:
        off_tr, off_te = tr[base].to_numpy(float), te[base].to_numpy(float)
        pen = np.ones(Xtr.shape[1])
    else:
        off_tr, off_te = np.zeros(len(tr)), np.zeros(len(te))
        pen = np.r_[0.0, [0.0] * int(with_base), [1.0 if penal_feats else 0.0] * len(feats)]
    beta = fit_poisson_glm(Xtr, y, off_tr, pen, GLM_RIDGE)
    return np.exp(np.clip(off_te + Xte @ beta, -10, 3)), beta


def _mnl(tr, te, feats: list[str], anchored: bool = False):
    """Multinomial logit with offset log p_B0. free: intercepts and the B0 log-odds slopes unpenalised;
    anchored: every coefficient penalised (shrunk towards the unchanged B0 probabilities)."""
    cols = ["lo_H", "lo_A"] + feats
    off_tr = np.log(tr[["pH", "pD", "pA"]].to_numpy(float))
    off_te = np.log(te[["pH", "pD", "pA"]].to_numpy(float))
    Xtr = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)]
    Xte = np.c_[np.ones(len(te)), te[cols].to_numpy(float)]
    pen = np.ones(Xtr.shape[1]) if anchored else np.r_[0.0, 0.0, 0.0, [1.0] * len(feats)]
    W = fit_mnl(Xtr, tr["y"].to_numpy(int), off_tr, MNL_RIDGE, pen)
    return apply_mnl(W, Xte, off_te), W


def _geo_blend(tr, te) -> tuple[np.ndarray, np.ndarray, float]:
    """X5: log mu = (1 - w) log lambda_B0 + w log E[xG], one weight w in [0, 1] for both sides (Poisson ML)."""
    from scipy.optimize import minimize_scalar
    lb = np.r_[tr["lb_h"], tr["lb_a"]]
    lx = np.r_[_log(tr["exp_xg_h"]), _log(tr["exp_xg_a"])]
    y = np.r_[tr["home_goals"], tr["away_goals"]].astype(float)

    def nll(w):
        eta = (1 - w) * lb + w * lx
        return np.sum(np.exp(eta) - y * eta)

    w = float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)
    lh = np.exp((1 - w) * te["lb_h"].to_numpy(float) + w * _log(te["exp_xg_h"]))
    la = np.exp((1 - w) * te["lb_a"].to_numpy(float) + w * _log(te["exp_xg_a"]))
    return lh, la, w


def _xgb_resid(tr, te, side: str) -> np.ndarray:
    from xgboost import DMatrix, train
    y = tr["home_goals" if side == "h" else "away_goals"].to_numpy(float)
    base = "lb_h" if side == "h" else "lb_a"
    cols = (ZH if side == "h" else ZA) + [base, "min_xg_n"]
    params = {k: v for k, v in XGB_RESID.items() if k != "n_estimators"}
    params.update({"nthread": C.N_JOBS, "seed": C.SEED})
    dtr = DMatrix(tr[cols].to_numpy(float), label=y, base_margin=tr[base].to_numpy(float))
    dte = DMatrix(te[cols].to_numpy(float), base_margin=te[base].to_numpy(float))
    bst = train(params, dtr, num_boost_round=XGB_RESID["n_estimators"])
    return bst.predict(dte)


def in_season(df: pd.DataFrame, only: tuple[str, ...] | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Weekly expanding-window predictions of the in-season variants on the evaluation weeks."""
    played = df["played"].astype(bool)
    names = only or IN_SEASON
    out = {v: [] for v in names}
    coefs = {k: [] for k in ("R0_h", "R0a_h", "X2_h", "X2_a", "X2a_h", "X2a_a", "X3_h", "X3_a", "X3a_h", "X3a_a",
                             "M1", "M1a", "R0a_1x2", "X5_w")}
    log = []
    for w in weeks(df):
        te = df.loc[(df["date"] >= w) & (df["date"] < w + pd.Timedelta(days=C.REFIT_DAYS))]
        tr = df.loc[played & (df["date"] < w) & df["xg_ok"]]
        if te.empty:
            continue
        te_ok = te.loc[te["xg_ok"]]
        res = {}
        lh_b, la_b = te_ok["lh"].to_numpy(), te_ok["la"].to_numpy()
        p_b = te_ok[["pH", "pD", "pA"]].to_numpy(float)
        t3 = tr.assign(_fh=_log(tr["exp_xg_h"]), _fa=_log(tr["exp_xg_a"]))
        e3 = te_ok.assign(_fh=_log(te_ok["exp_xg_h"]), _fa=_log(te_ok["exp_xg_a"]))
        for anch, sfx in ((False, ""), (True, "a")):
            # R0 / R0a: recalibration without xG
            lh, bh = _goal_glm(tr, te_ok, "h", [], True, False, anch)
            la, _ = _goal_glm(tr, te_ok, "a", [], True, False, anch)
            p, W0 = _mnl(tr, te_ok, [], anch)
            res["R0" + sfx] = (lh, la, p)
            coefs[f"R0{sfx}_h"].append(bh.tolist())
            if anch:
                coefs["R0a_1x2"].append(W0.tolist())
            # X2 / X2a: GLM correction with xG features
            lh2, bh = _goal_glm(tr, te_ok, "h", ZH, True, True, anch)
            la2, ba = _goal_glm(tr, te_ok, "a", ZA, True, True, anch)
            res["X2" + sfx] = (lh2, la2, propagate(p_b, lh_b, la_b, lh2, la2))
            coefs[f"X2{sfx}_h"].append(bh.tolist()), coefs[f"X2{sfx}_a"].append(ba.tolist())
            # X3 / X3a: blend with the rating expectation
            lh, bh = _goal_glm(t3, e3, "h", ["_fh"], True, anch, anch)
            la, ba = _goal_glm(t3, e3, "a", ["_fa"], True, anch, anch)
            res["X3" + sfx] = (lh, la, propagate(p_b, lh_b, la_b, lh, la))
            coefs[f"X3{sfx}_h"].append(bh.tolist()), coefs[f"X3{sfx}_a"].append(ba.tolist())
            # M1 / M1a: 1X2 correction with xG
            p, W = _mnl(tr, te_ok, D1X2, anch)
            res["M1" + sfx] = (lh_b, la_b, p)
            coefs["M1" + sfx].append(W.tolist())
            res["X2M" + sfx] = (lh2, la2, p)
        # X1 / G1: pure rating models (scale calibrated in-season)
        for name, eh, ea in (("X1", "exp_xg_h", "exp_xg_a"), ("G1", "exp_g_h", "exp_g_a")):
            t2, e2 = tr.assign(_fh=_log(tr[eh]), _fa=_log(tr[ea])), te_ok.assign(_fh=_log(te_ok[eh]), _fa=_log(te_ok[ea]))
            lh, _ = _goal_glm(t2, e2, "h", ["_fh"], False, False)
            la, _ = _goal_glm(t2, e2, "a", ["_fa"], False, False)
            res[name] = (lh, la, pois_1x2(lh, la))
        # X5: one-parameter geometric blend
        lh, la, wgt = _geo_blend(tr, te_ok)
        res["X5"] = (lh, la, propagate(p_b, lh_b, la_b, lh, la))
        coefs["X5_w"].append(wgt)
        # X4: XGBoost residual
        if "X4" in names:
            lh, la = _xgb_resid(tr, te_ok, "h"), _xgb_resid(tr, te_ok, "a")
            res["X4"] = (lh, la, propagate(p_b, lh_b, la_b, lh, la))
        for v in names:
            lh, la, p = res[v]
            out[v].append(pd.DataFrame({"match_id": te_ok["match_id"].to_numpy(), "lh": lh, "la": la,
                                        "pH": p[:, 0], "pD": p[:, 1], "pA": p[:, 2], "week": w}))
        log.append({"week": str(w.date()), "train": int(len(tr)), "test": int(len(te)), "test_xg_ok": int(len(te_ok))})
    return {v: pd.concat(parts, ignore_index=True) for v, parts in out.items()}, {"weeks": log, "coefs": coefs}


# ------------------------------------------------------------------------------------------------ XGBoost variants
def _prod_params(task: str) -> tuple[dict, int]:
    from src.modelling.registry import version_metadata
    info = version_metadata(C.PRODUCTION_VERSION)["tasks"][task]
    return dict(info["params"]), int(info["n_estimators"])


def _xgb_fit_predict(task: str, train: pd.DataFrame, score: pd.DataFrame, numeric: list[str]) -> np.ndarray:
    from xgboost import XGBClassifier, XGBRegressor

    from src import config
    from src.modelling.preprocessing import FeaturePreprocessor
    from src.modelling.tasks import TASKS
    from src.modelling.training import recency_weights
    t = TASKS[task]
    params, n = _prod_params(task)
    rows = train.loc[t.trainable(train)]
    prep = FeaturePreprocessor(numeric, ["league"]).fit(rows)
    cls = XGBClassifier if t.kind == "classifier" else XGBRegressor
    est = cls(n_estimators=n, objective=t.objective, eval_metric=t.eval_metric, tree_method="hist",
              random_state=config.RANDOM_STATE, n_jobs=C.N_JOBS,
              **{k: v for k, v in params.items() if k != "half_life_years"})
    est.fit(prep.transform(rows), t.target_values(rows).to_numpy(),
            sample_weight=recency_weights(rows["date"], params.get("half_life_years", 0)))
    X = prep.transform(score)
    return est.predict_proba(X) if t.kind == "classifier" else est.predict(X)


XG_FEATURES_S1 = ["h_xg_n", "a_xg_n", "h_xgf_s", "h_xga_s", "a_xgf_s", "a_xga_s", "h_gmx_s", "a_gmx_s", "h_att",
                  "h_dfn", "a_att", "a_dfn", "exp_xg_h", "exp_xg_a", "h_xgf_v5", "a_xgf_v5", "h_xgf_ewm", "a_xgf_ewm",
                  "h_xga_ewm", "a_xga_ewm"]


def short_history(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """S0 / S1: production XGBoost settings, fitted weekly on 2026/27 rows only."""
    from src.features.sets import model_features
    full = D.season_features()
    full = full.merge(df[["match_id"] + [c for c in XG_FEATURES_S1 if c in df]], on="match_id", how="inner")
    full = full.sort_values(["date", "match_id"]).reset_index(drop=True)
    cols = list(full.columns)
    out = {"S0": [], "S1": []}
    for w in weeks(df):
        te = full.loc[(full["date"] >= w) & (full["date"] < w + pd.Timedelta(days=C.REFIT_DAYS))]
        tr = full.loc[full["played"].astype(bool) & (full["date"] < w)]
        if te.empty:
            continue
        for v in ("S0", "S1"):
            preds = {}
            for task in ("result", "home_goals", "away_goals"):
                numeric = model_features(task, cols)[0]
                numeric = [c for c in numeric if c not in XG_FEATURES_S1]
                if v == "S1":
                    numeric = numeric + XG_FEATURES_S1
                preds[task] = _xgb_fit_predict(task, tr, te, numeric)
            p = preds["result"]
            out[v].append(pd.DataFrame({"match_id": te["match_id"].to_numpy(), "lh": preds["home_goals"],
                                        "la": preds["away_goals"], "pH": p[:, 0], "pD": p[:, 1], "pA": p[:, 2],
                                        "week": w}))
        C.log.info("short history week %s: train %d", w.date(), len(tr))
    return {v: pd.concat(parts, ignore_index=True) for v, parts in out.items()}


def proxy_mapping(df_raw_xg: pd.DataFrame) -> dict:
    """xG ~ a + b shots + c shots-on-target, fitted on xG matches played before EVAL_START only."""
    x = df_raw_xg.loc[df_raw_xg["date"] < C.EVAL_START].dropna(subset=["home_shots", "away_shots", "home_sot", "away_sot"])
    shots = np.r_[x["home_shots"], x["away_shots"]]
    sot = np.r_[x["home_sot"], x["away_sot"]]
    y = np.r_[x["xg_home"], x["xg_away"]]
    A = np.c_[np.ones_like(shots), shots, sot]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"a": float(coef[0]), "b_shots": float(coef[1]), "c_sot": float(coef[2]), "r2": float(r2), "n": int(len(y))}


def add_proxy(df: pd.DataFrame, mp: dict) -> pd.DataFrame:
    out = df.copy()
    for s in ("h", "a"):
        for win in (5, 10):
            out[f"{s}_pxg_f{win}"] = mp["a"] + mp["b_shots"] * out[f"{s}_form{win}_shots_for"] + mp["c_sot"] * out[f"{s}_form{win}_sot_for"]
            out[f"{s}_pxg_a{win}"] = mp["a"] + mp["b_shots"] * out[f"{s}_form{win}_shots_against"] + mp["c_sot"] * out[f"{s}_form{win}_sot_against"]
    out["diff_pxg10"] = (out["h_pxg_f10"] - out["h_pxg_a10"]) - (out["a_pxg_f10"] - out["a_pxg_a10"])
    return out


PROXY_COLS = [f"{s}_pxg_{k}{w}" for s in ("h", "a") for k in ("f", "a") for w in (5, 10)] + ["diff_pxg10"]


def long_history(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """T0 (refit of v6.1 on 2021/22-2025/26) and T1 (+ shot-based xG proxy), predicting all of 2026/27."""
    from src.features.sets import model_features
    mp = proxy_mapping(D.clean_xg(D.raw_xg("2627")))
    hist = pd.read_parquet(C.ROOT / "local_data/processed/features.parquet",
                           filters=[("season", "<", C.SEASON), ("season", ">=", 2021)])
    hist = hist.loc[hist["played"].astype(bool)]
    cur = D.season_features()
    cur = cur.loc[cur["match_id"].isin(df["match_id"])]
    hist, cur = add_proxy(hist, mp), add_proxy(cur, mp)
    cols = list(hist.columns)
    out = {}
    for v in ("T0", "T1"):
        preds = {}
        for task in ("result", "home_goals", "away_goals"):
            numeric = [c for c in model_features(task, cols)[0] if c not in PROXY_COLS]
            if v == "T1":
                numeric += PROXY_COLS
            preds[task] = _xgb_fit_predict(task, hist, cur, numeric)
            C.log.info("long history %s %s fitted", v, task)
        p = preds["result"]
        out[v] = pd.DataFrame({"match_id": cur["match_id"].to_numpy(), "lh": preds["home_goals"],
                               "la": preds["away_goals"], "pH": p[:, 0], "pD": p[:, 1], "pA": p[:, 2]})
    return out, mp


def corners_check(df: pd.DataFrame) -> dict:
    """Do xG features explain the B0 corner residual? Weekly Poisson GLM correction of the corner mean."""
    d = df.loc[df["cmu"].notna() & df["total_corners"].notna()].copy()
    d["lcmu"] = _log(d["cmu"])
    d["xg_sum"] = d["d3"]
    d["xg_gap"] = d["d1"].abs()
    feats = ["xg_sum", "xg_gap", "zh2", "za2"]
    res_all, base_all, ids = [], [], []
    for w in weeks(d):
        te = d.loc[(d["date"] >= w) & (d["date"] < w + pd.Timedelta(days=C.REFIT_DAYS)) & d["xg_ok"]]
        tr = d.loc[(d["date"] < w) & d["xg_ok"]]
        if te.empty or len(tr) < 50:
            continue
        Xtr = np.c_[np.ones(len(tr)), tr[["lcmu"] + feats].to_numpy(float)]
        Xte = np.c_[np.ones(len(te)), te[["lcmu"] + feats].to_numpy(float)]
        beta = fit_poisson_glm(Xtr, tr["total_corners"].to_numpy(float), np.zeros(len(tr)),
                               np.r_[0, 0, [1.0] * len(feats)], GLM_RIDGE)
        Xr = np.c_[np.ones(len(tr)), tr[["lcmu"]].to_numpy(float)]
        br = fit_poisson_glm(Xr, tr["total_corners"].to_numpy(float), np.zeros(len(tr)), np.r_[0, 0], GLM_RIDGE)
        res_all.append(np.exp(Xte @ beta))
        base_all.append(np.exp(np.c_[np.ones(len(te)), te[["lcmu"]].to_numpy(float)] @ br))
        ids.append(te["match_id"].to_numpy())
    ids = np.concatenate(ids)
    e = d.set_index("match_id").loc[ids]
    y = e["total_corners"].to_numpy(float)
    mu_x, mu_r, mu_b = np.concatenate(res_all), np.concatenate(base_all), e["cmu"].to_numpy(float)
    ev = d.loc[d["xg_ok"] & (d["date"] >= C.EVAL_START)]
    resid = ev["total_corners"] - ev["cmu"]
    corr = {f: float(np.corrcoef(ev[f], resid)[0, 1]) for f in feats}
    return {"n": int(len(y)), "dev_B0": float(C.poisson_dev(y, mu_b).mean()),
            "dev_recal": float(C.poisson_dev(y, mu_r).mean()), "dev_xg": float(C.poisson_dev(y, mu_x).mean()),
            "diff_xg_vs_recal": C.boot_mean_ci(C.poisson_dev(y, mu_x) - C.poisson_dev(y, mu_r)),
            "diff_xg_vs_B0": C.boot_mean_ci(C.poisson_dev(y, mu_x) - C.poisson_dev(y, mu_b)),
            "corr_with_residual": corr}
