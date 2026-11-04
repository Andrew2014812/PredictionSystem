"""Step 2 (question 3) — new ways to get draw probabilities, judged by log loss / Brier.

Extra models per block b (fitted on seasons < b, predicting b; cached):
    parity_xgb   production 1X2 XGBoost (same parameters and trees) + 22 parity / low-scoring features
    draw_xgb     binary draw-vs-not XGBoost on the same features; trees by early stopping 2021 -> 2022
    draw_lr      logistic regression draw-vs-not on a compact parity set (+ league one-hot)
    mnl          multinomial logistic regression 1X2 on the same compact set

Variants per fold s (OUTER = block s; every constant is fitted on the INNER block s-1):
    a_baseline            production XGBoost (what FootPredict uses)
    c1_temperature        one temperature
    c2_vector             vector scaling z_k' = a_k log p_k + b_k
    c3_draw_bias          log p_D + b only
    s1_poisson            1X2 of the independent-Poisson score matrix (goal models)
    s2_dixon_coles        the same with the Dixon–Coles low-score correction (rho fitted on inner)
    s3_low_score_infl     low-scoring draw inflation: cells 0-0 and 1-1 multiplied by (1 + k00), (1 + k11)
    b_poisson / b_dc / b_low_score   (1 - w) XGBoost + w score model, w on inner
    p_parity_xgb          parity XGBoost
    d_draw_xgb_blend      P(D) := (1 - w) P(D) + w q_xgb, H / A rescaled
    d_draw_lr_blend       the same with the logistic draw model
    m_mnl                 multinomial logistic regression alone
    e_avg_parity          (baseline + parity XGBoost) / 2
    e_xgb_mnl             (1 - w) baseline + w MNL, w on inner
    e_stack               multinomial LR stacking of [log p_xgb, log p_parity, logit q_xgb, log P_DC(D), |pH - pA|]
    bk_proportional       Bet365 de-margined proportionally (rows with odds; pre-match prices)
    bk_power              Bet365 de-margined with the power method
    e_book_blend          (1 - w) baseline + w bookmaker (w on inner; uses prices, so not a model-only forecast)
    naive_league          training-season outcome frequencies of the league

Selection rule (fixed before looking at results): a variant is eligible when its log loss AND Brier are
lower than the baseline on BOTH 2023/24 and 2024/25; among eligible variants the lowest pooled log loss
wins; it counts as an improvement only if the pooled paired bootstrap 95 % CI of the log-loss difference is
below zero and the 2026/27 check also has a lower log loss. Variants that use bookmaker prices are reported
but not eligible for production (prices are not model inputs and are missing for some matches).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from experiments.v6_research import scores as S
from src import config
from src.features.sets import model_features
from src.markets.distributions import fit_dixon_coles_rho

from . import blocks as B
from . import common as C

log = C.setup_logging("draw61.probs")
BOOK = ("bk_proportional", "bk_power", "e_book_blend")
COMPACT = ["elo_diff", "abs_elo_diff", "elo_exp_home", "abs_elo_exp_gap", "diff_season_ppg", "abs_diff_season_ppg",
           "diff_form10_ppg", "abs_diff_form10_ppg", "diff_form10_gd", "abs_diff_form10_gd", "diff_venue_ppg",
           "abs_venue_form_gap", "diff_position", "abs_diff_position", "diff_form10_shots", "abs_diff_form10_shots",
           "sum_form10_draw_rate", "sum_form15_draw_rate", "sum_form10_goals", "sum_venue_attack", "low_score_index",
           "league_draw_rate", "h2h_draw_rate", "h2h_n"]


# ---------------------------------------------------------------------------------------------- maps
def temperature(p, T):
    q = np.power(np.clip(p, 1e-12, 1), 1 / T)
    return q / q.sum(1, keepdims=True)


def fit_temperature(p, y):
    return float(minimize_scalar(lambda t: log_loss(y, temperature(p, t), labels=[0, 1, 2]),
                                 bounds=(0.5, 2.5), method="bounded").x)


def vector_scale(p, a, b):
    z = np.log(np.clip(p, 1e-12, 1)) * a + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_vector(p, y):
    x = minimize(lambda x: log_loss(y, vector_scale(p, x[:3], np.r_[0.0, x[3:]]), labels=[0, 1, 2]),
                 np.r_[1, 1, 1, 0, 0], method="L-BFGS-B").x
    return x[:3].tolist(), [0.0, *x[3:].tolist()]


def fit_draw_bias(p, y):
    return float(minimize_scalar(lambda b: log_loss(y, vector_scale(p, np.ones(3), np.array([0, b, 0])),
                                                    labels=[0, 1, 2]), bounds=(-1, 1), method="bounded").x)


def blend_draw(p, q, w):
    pd_ = (1 - w) * p[:, 1] + w * q
    rest = np.clip(p[:, 0] + p[:, 2], 1e-12, None)
    return np.stack([p[:, 0] * (1 - pd_) / rest, pd_, p[:, 2] * (1 - pd_) / rest], 1)


GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def best_weight(fn, y):
    scores = {float(w): log_loss(y, fn(w), labels=[0, 1, 2]) for w in GRID}
    return min(scores, key=scores.get)


# low-scoring draw inflation of the Poisson score matrix
def inflate(m, k00, k11):
    m = m.copy()
    m[:, 0, 0] *= 1 + k00
    m[:, 1, 1] *= 1 + k11
    return m / m.sum((1, 2), keepdims=True)


def fit_inflation(m, hg, ag):
    g = m.shape[1] - 1
    hg, ag = np.minimum(np.asarray(hg, int), g), np.minimum(np.asarray(ag, int), g)
    cell = m[np.arange(len(hg)), hg, ag]
    m00, m11 = m[:, 0, 0], m[:, 1, 1]

    def nll(k):
        mult = np.where((hg == 0) & (ag == 0), 1 + k[0], np.where((hg == 1) & (ag == 1), 1 + k[1], 1.0))
        z = 1 + k[0] * m00 + k[1] * m11
        return -np.sum(np.log(cell * mult / z))
    x = minimize(nll, np.zeros(2), method="L-BFGS-B", bounds=[(-0.9, 2.0), (-0.9, 2.0)]).x
    return float(x[0]), float(x[1])


# ---------------------------------------------------------------------------------------------- extra fits
def _lr_pipe(multinomial: bool):
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), COMPACT),
                             ("lg", OneHotEncoder(handle_unknown="ignore"), ["league"])])
    return make_pipeline(pre, LogisticRegression(C=0.5, max_iter=3000))


_DRAW_TREES: int | None = None


def draw_trees(full: pd.DataFrame, numeric: list[str], params: dict) -> int:
    """Tree count of the binary draw model: early stopping, fitted on 2021, validated on 2022 (before every fold)."""
    global _DRAW_TREES
    path = C.CACHE / "draw_xgb_trees.txt"
    if _DRAW_TREES is None:
        if path.exists():
            _DRAW_TREES = int(path.read_text())
        else:
            tr, va = full.loc[full["season"] == 2021], full.loc[full["season"] == 2022]
            m = C.XGBFit(numeric, params, 1, "binary:logistic", "logloss")
            _DRAW_TREES = m.early_stopped_trees(tr, (tr["y"] == 1).astype(int), va, (va["y"] == 1).astype(int))
            path.write_text(str(_DRAW_TREES))
    return _DRAW_TREES


def extra_block(b: int, refresh: bool = False) -> pd.DataFrame:
    path = C.CACHE / f"extra_{b}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    t0 = time.time()
    full, parity = C.add_parity(C.features())
    full["y"] = C.y_index(full["result"])
    base_num = model_features("result", list(C.features().columns))[0]
    numeric = base_num + parity
    info = C.prod_meta()["tasks"]["result"]
    params, n = dict(info["params"]), int(info["n_estimators"])
    tr, ev = full.loc[full["season"] < b], full.loc[full["season"] == b]
    out = pd.DataFrame({"match_id": ev["match_id"].to_numpy()})
    secs = {}
    m = C.XGBFit(numeric, params, n, "multi:softprob", "mlogloss").fit(tr, tr["y"])
    p = m.predict(ev)
    out["par_H"], out["par_D"], out["par_A"] = p[:, 0], p[:, 1], p[:, 2]
    secs["parity_xgb"] = m.seconds
    nt = draw_trees(full, numeric, params)
    m = C.XGBFit(numeric, params, nt, "binary:logistic", "logloss").fit(tr, (tr["y"] == 1).astype(int))
    out["q_draw_xgb"] = m.predict(ev)
    secs["draw_xgb"] = m.seconds
    t1 = time.time()
    lr = _lr_pipe(False).fit(tr[COMPACT + ["league"]], (tr["y"] == 1).astype(int))
    out["q_draw_lr"] = lr.predict_proba(ev[COMPACT + ["league"]])[:, 1]
    secs["draw_lr"] = time.time() - t1
    t1 = time.time()
    mnl = _lr_pipe(True).fit(tr[COMPACT + ["league"]], tr["y"])
    p = mnl.predict_proba(ev[COMPACT + ["league"]])
    out["mnl_H"], out["mnl_D"], out["mnl_A"] = p[:, 0], p[:, 1], p[:, 2]
    secs["mnl"] = time.time() - t1
    out["abs_gap_feature_count"] = len(parity)
    out.to_parquet(path)
    (C.CACHE / f"extra_{b}.json").write_text(pd.Series({**secs, "draw_trees": nt, "total": time.time() - t0}).to_json())
    log.info("extra block %s done in %.0fs (%s)", b, time.time() - t0, secs)
    return out


def block_frame(b: int) -> pd.DataFrame:
    pred = B.block(b)[0]
    return pred.merge(extra_block(b), on="match_id", how="left", validate="1:1")


# ---------------------------------------------------------------------------------------------- variants
def _stack_x(fr, dc):
    p = fr[["pH", "pD", "pA"]].to_numpy()
    q = np.clip(fr["q_draw_xgb"].to_numpy(), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(np.clip(p, 1e-9, 1)), np.log(np.clip(fr[["par_H", "par_D", "par_A"]].to_numpy(), 1e-9, 1)),
                            np.log(q / (1 - q)), np.log(np.clip(dc[:, 1], 1e-9, 1)), np.abs(p[:, 0] - p[:, 2])])


def fold_variants(s: int) -> tuple[dict, dict, pd.DataFrame]:
    """(probabilities by variant on the OUTER block, fitted parameters, the outer frame)."""
    fin, fout = block_frame(s - 1), block_frame(s)
    y_in = fin["y"].to_numpy()
    P = {}
    prm = {}
    p_in, p_out = fin[["pH", "pD", "pA"]].to_numpy(), fout[["pH", "pD", "pA"]].to_numpy()
    P["a_baseline"] = p_out
    T = fit_temperature(p_in, y_in)
    P["c1_temperature"], prm["c1_temperature"] = temperature(p_out, T), {"T": T}
    a, b = fit_vector(p_in, y_in)
    P["c2_vector"], prm["c2_vector"] = vector_scale(p_out, np.array(a), np.array(b)), {"a": a, "b": b}
    bd = fit_draw_bias(p_in, y_in)
    P["c3_draw_bias"], prm["c3_draw_bias"] = vector_scale(p_out, np.ones(3), np.array([0, bd, 0])), {"b_draw": bd}

    m_in, m_out = S.matrices(fin["lh"], fin["la"]), S.matrices(fout["lh"], fout["la"])
    pois_in, pois_out = S.outcome_1x2(m_in), S.outcome_1x2(m_out)
    rho = fit_dixon_coles_rho(fin["lh"].to_numpy(), fin["la"].to_numpy(), fin["home_goals"].to_numpy(),
                              fin["away_goals"].to_numpy())
    dc_in = S.outcome_1x2(S.matrices(fin["lh"], fin["la"], rho))
    dc_out = S.outcome_1x2(S.matrices(fout["lh"], fout["la"], rho))
    k00, k11 = fit_inflation(m_in, fin["home_goals"], fin["away_goals"])
    li_in, li_out = S.outcome_1x2(inflate(m_in, k00, k11)), S.outcome_1x2(inflate(m_out, k00, k11))
    P["s1_poisson"], P["s2_dixon_coles"], P["s3_low_score_infl"] = pois_out, dc_out, li_out
    prm["s2_dixon_coles"], prm["s3_low_score_infl"] = {"rho": rho}, {"k00": k00, "k11": k11}
    for name, qi, qo in (("b_poisson", pois_in, pois_out), ("b_dc", dc_in, dc_out), ("b_low_score", li_in, li_out)):
        w = best_weight(lambda w: (1 - w) * p_in + w * qi, y_in)
        P[name], prm[name] = (1 - w) * p_out + w * qo, {"weight": w}

    par_in, par_out = fin[["par_H", "par_D", "par_A"]].to_numpy(), fout[["par_H", "par_D", "par_A"]].to_numpy()
    P["p_parity_xgb"] = par_out
    for name, col in (("d_draw_xgb_blend", "q_draw_xgb"), ("d_draw_lr_blend", "q_draw_lr")):
        w = best_weight(lambda w: blend_draw(p_in, fin[col].to_numpy(), w), y_in)
        P[name], prm[name] = blend_draw(p_out, fout[col].to_numpy(), w), {"weight": w}
    mnl_in, mnl_out = fin[["mnl_H", "mnl_D", "mnl_A"]].to_numpy(), fout[["mnl_H", "mnl_D", "mnl_A"]].to_numpy()
    P["m_mnl"] = mnl_out
    P["e_avg_parity"] = (p_out + par_out) / 2
    w = best_weight(lambda w: (1 - w) * p_in + w * mnl_in, y_in)
    P["e_xgb_mnl"], prm["e_xgb_mnl"] = (1 - w) * p_out + w * mnl_out, {"weight": w}
    st = LogisticRegression(C=1.0, max_iter=3000).fit(_stack_x(fin, dc_in), y_in)
    P["e_stack"] = st.predict_proba(_stack_x(fout, dc_out))
    prm["e_stack"] = {"coef": st.coef_.tolist(), "intercept": st.intercept_.tolist()}

    odds_in, odds_out = fin[["odds_home", "odds_draw", "odds_away"]].to_numpy(float), fout[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    P["bk_proportional"], P["bk_power"] = C.demargin(odds_out, "proportional"), C.demargin(odds_out, "power")
    bk_in = C.demargin(odds_in, "proportional")
    ok_in = np.isfinite(bk_in).all(1)
    w = best_weight(lambda w: (1 - w) * p_in[ok_in] + w * bk_in[ok_in], y_in[ok_in])
    bko = P["bk_proportional"]
    P["e_book_blend"] = np.where(np.isfinite(bko).all(1, keepdims=True), (1 - w) * p_out + w * np.nan_to_num(bko), np.nan)
    prm["e_book_blend"] = {"weight": w}

    df = C.features()
    tr = df.loc[df["season"] < s]
    yt = C.y_index(tr["result"])
    freq = np.bincount(yt, minlength=3) / len(yt)
    lf = {lg: np.bincount(g, minlength=3) / len(g) for lg, g in pd.Series(yt, index=tr["league"].to_numpy()).groupby(level=0)}
    P["naive_league"] = np.stack([lf.get(lg, freq) for lg in fout["league"]])
    return P, prm, fout


TITLES = {
    "a_baseline": "базовая XGBoost (продакшн)", "c1_temperature": "температура", "c2_vector": "vector scaling",
    "c3_draw_bias": "сдвиг ничьей", "s1_poisson": "Пуассон (модели голов)", "s2_dixon_coles": "Диксон–Коулс",
    "s3_low_score_infl": "инфляция 0-0/1-1", "b_poisson": "бленд XGB + Пуассон", "b_dc": "бленд XGB + Д–К",
    "b_low_score": "бленд XGB + инфляция", "p_parity_xgb": "XGB + признаки паритета",
    "d_draw_xgb_blend": "бленд: XGB-классификатор ничьей", "d_draw_lr_blend": "бленд: логит-классификатор ничьей",
    "m_mnl": "мультиномиальная логрегрессия", "e_avg_parity": "среднее XGB и XGB-паритет",
    "e_xgb_mnl": "бленд XGB + логрегрессия", "e_stack": "стекинг (логрегрессия)",
    "bk_proportional": "букмекер (пропорц.)", "bk_power": "букмекер (степенной)",
    "e_book_blend": "бленд XGB + букмекер", "naive_league": "наивная частота лиги",
}


def run() -> dict:
    t0 = time.time()
    res = {"folds": {}, "titles": TITLES}
    allp = {}
    for s in C.FOLDS:
        P, prm, fout = fold_variants(s)
        y = fout["y"].to_numpy()
        ok = np.isfinite(P["bk_proportional"]).all(1)
        base_ll, base_br = C.loss_per_match(y, P["a_baseline"]), C.brier_per_match(y, P["a_baseline"])
        fold = {"n": int(len(y)), "n_with_odds": int(ok.sum()), "params": prm, "all_rows": {}, "odds_rows": {}}
        for name, p in P.items():
            if name not in BOOK:
                m = C.prob_metrics(y, p)
                m["delta_log_loss"] = C.paired_bootstrap(C.loss_per_match(y, p), base_ll)
                m["delta_brier"] = C.paired_bootstrap(C.brier_per_match(y, p), base_br)
                m["reliability_draw"] = C.reliability(y == 1, p[:, 1])
                fold["all_rows"][name] = m
            m = C.prob_metrics(y[ok], p[ok])
            m["delta_log_loss"] = C.paired_bootstrap(C.loss_per_match(y[ok], p[ok]), base_ll[ok])
            if name in ("a_baseline", "bk_proportional"):
                m["reliability_draw"] = C.reliability(y[ok] == 1, p[ok, 1])
            fold["odds_rows"][name] = m
        res["folds"][str(s)] = fold
        allp[s] = (P, y, ok)
        pd.DataFrame({f"{n}|{c}": p[:, i] for n, p in P.items() for i, c in enumerate(C.CLASSES)}
                     ).assign(match_id=fout["match_id"].to_numpy(), y=y, league=fout["league"].to_numpy()
                              ).to_parquet(C.CACHE / f"probs_{s}.parquet")
        log.info("fold %s: %s", s, {n: round(v["log_loss"], 4) for n, v in fold["all_rows"].items()})

    names = [n for n in allp[C.SELECT[0]][0] if n not in BOOK]
    sel = {}
    for n in names:
        f = [res["folds"][str(s)]["all_rows"] for s in C.SELECT]
        ys = np.concatenate([allp[s][1] for s in C.SELECT])
        ps = np.concatenate([allp[s][0][n] for s in C.SELECT])
        pb = np.concatenate([allp[s][0]["a_baseline"] for s in C.SELECT])
        sel[n] = {"pooled_log_loss": float(log_loss(ys, ps, labels=[0, 1, 2])),
                  "pooled_brier": float(C.brier_per_match(ys, ps).mean()),
                  "pooled_delta_log_loss": C.paired_bootstrap(C.loss_per_match(ys, ps), C.loss_per_match(ys, pb)),
                  "better_log_loss_both": bool(all(x[n]["log_loss"] < x["a_baseline"]["log_loss"] for x in f)),
                  "better_brier_both": bool(all(x[n]["brier"] < x["a_baseline"]["brier"] for x in f)),
                  "check_2026_delta_log_loss": res["folds"][str(C.CHECK)]["all_rows"][n]["delta_log_loss"],
                  "info_2025_delta_log_loss": res["folds"][str(C.INFO)]["all_rows"][n]["delta_log_loss"]}
    eligible = [n for n in names if n not in ("a_baseline", "naive_league") and sel[n]["better_log_loss_both"]
                and sel[n]["better_brier_both"]]
    best = min(eligible, key=lambda n: sel[n]["pooled_log_loss"]) if eligible else None
    convincing = bool(best and sel[best]["pooled_delta_log_loss"]["ci95"][1] < 0
                      and sel[best]["check_2026_delta_log_loss"]["mean_diff"] < 0)
    # bookmaker comparison on the matches with prices (pooled selection folds)
    book = {}
    for n in list(allp[C.SELECT[0]][0]):
        ys = np.concatenate([allp[s][1][allp[s][2]] for s in C.SELECT])
        ps = np.concatenate([allp[s][0][n][allp[s][2]] for s in C.SELECT])
        book[n] = {"pooled_log_loss": float(log_loss(ys, ps, labels=[0, 1, 2])),
                   "pooled_brier": float(C.brier_per_match(ys, ps).mean()),
                   "pooled_draw_log_loss": C.prob_metrics(ys, ps)["draw_log_loss"]}
    res["selection"] = {"rule": "eligible: log loss AND Brier below the baseline on both 2023/24 and 2024/25; best: lowest "
                                "pooled log loss; convincing: pooled paired-bootstrap CI of the log-loss difference < 0 "
                                "and lower log loss on 2026/27; bookmaker-based variants excluded",
                        "table": sel, "eligible": eligible, "best": best, "convincing": convincing,
                        "odds_rows_pooled_selection": book}
    res["minutes"] = (time.time() - t0) / 60
    res["extra_fit_seconds"] = {str(b): pd.read_json(C.CACHE / f"extra_{b}.json", typ="series").to_dict()
                                for b in C.BLOCKS if (C.CACHE / f"extra_{b}.json").exists()}
    C.save_json("probabilities.json", res)
    return res


def load_probs(s: int) -> pd.DataFrame:
    return pd.read_parquet(C.CACHE / f"probs_{s}.parquet")


def get(fr: pd.DataFrame, name: str) -> np.ndarray:
    return fr[[f"{name}|{c}" for c in C.CLASSES]].to_numpy()


if __name__ == "__main__":
    run()
