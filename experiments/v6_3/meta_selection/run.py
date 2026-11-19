"""Study 9 (v6.3) — a meta-selection model that picks the Main among the computed candidates.

    experiments/v6_3/heavy.sh 3600 experiments/v6_3/meta_selection/logs/run.log \
        .venv/bin/python -m experiments.v6_3.meta_selection.run

Needs the calibration study's cache (``experiments.v6_3.calibration.run``).

Walk-forward by month inside 2026/27: the meta-model that picks the Main of month M is trained on the 2025/26
backtest pool (out-of-sample predictions of a model trained to 2025-06-01, Football-Data Bet365 prices) plus the
2026/27 pool of the same universe settled before the 1st of M. It never sees the matches it picks for.

Meta variants (hyper-parameters fixed in advance, nothing tuned on 2026/27; the primary one is fixed in advance as
``meta_lr_ev`` — the smallest model):
    meta_lr_ev     logistic regression, P(win); Main = max EV_meta = P_meta (1 − P(push)) odds − (1 − P(push))
    meta_gbm_ev    small gradient boosting (XGBoost, depth 3, 300 trees, 2 threads), P(win); max EV_meta
    meta_gbm_ret   small gradient boosting regression of the unit profit; max predicted return
    meta_lr_rule   production rule with the model probability replaced by P_meta of meta_lr_ev
Simple rules on the same pool: production rule restricted to the priced pool (``rule_priced``: separates the effect
of choosing only priced candidates, which matters at t09, from the effect of the meta-model), most likely (max p), max EV, most likely by the
bookmaker (max q), production rule on calibrated probabilities (study 2 primary map); random pick (null).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from experiments.v6_3 import common as C
from experiments.v6_3.calibration import data as D
from experiments.v6_3.calibration import maps as MP
from experiments.v6_3.calibration import metrics as MT
from experiments.v6_3.calibration import rule as R
from experiments.v6_3.replay import evaluate as EV

from . import features as F

RES, CACHE = F.DIRS["results"], F.DIRS["cache"]
META = ["meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret", "meta_lr_rule"]
SIMPLE = ["rule_priced", "max_p", "max_ev", "max_q", "rule_calibrated"]
PRIMARY = "meta_lr_ev"
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
N_PERM = {"meta_lr_ev": 20, "meta_gbm_ev": 8, "meta_gbm_ret": 8}
N_RANDOM = 300


# ------------------------------------------------------------------------------------------------ data
def load_pools(universe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training-history pool = 2025/26 backtest, season pool of the universe) with all features."""
    primary = C.read_json(F.CAL_CACHE / "primary.json")["primary"]
    bt = D.backtest()
    lomo = pd.read_parquet(F.CAL_CACHE / "backtest_cal_lomo.parquet")
    bt_pc = bt[["match_id", "market", "selection"]].merge(lomo, on=["match_id", "market", "selection"],
                                                         how="left")["pc_cal_lomo"].to_numpy()
    c = D.season(universe)
    cal = pd.read_parquet(F.CAL_CACHE / "calibrated_2026.parquet", columns=["match_id", "market", "selection", primary])
    c_pc = c[["match_id", "market", "selection"]].merge(cal, on=["match_id", "market", "selection"],
                                                       how="left")[primary].to_numpy(float)
    pb = F.build(bt, bt_pc, "2025-08").assign(season=2025)
    ps = F.build(c, c_pc, "2026-08").assign(season=2026)
    both = F.add_reliability(pd.concat([pb, ps], ignore_index=True))
    return both.loc[both["season"] == 2025].reset_index(drop=True), both.loc[both["season"] == 2026].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ models
class LR:
    def __init__(self):
        self.sc, self.m = StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)

    def fit(self, X, y):
        self.m.fit(self.sc.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(self.sc.transform(X))[:, 1]


def gbm_cls():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, reg_lambda=5.0, n_jobs=C.N_JOBS, random_state=C.SEED, verbosity=0)


def gbm_reg():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=50, reg_lambda=5.0, n_jobs=C.N_JOBS, random_state=C.SEED, verbosity=0)


class GBMWrap:
    def __init__(self, kind):
        self.kind, self.m = kind, gbm_cls() if kind == "cls" else gbm_reg()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if self.kind == "cls" else self.m.predict(X)


def X_of(d: pd.DataFrame) -> np.ndarray:
    return d[F.FEATURES].to_numpy(float)


def fit_models(train: pd.DataFrame, which=("lr", "gbm_cls", "gbm_reg"), shuffle_rng=None) -> dict:
    cls = train.loc[train["y"].notna()]
    reg = train.loc[train["profit_u"].notna()]
    ycls, yreg = cls["y"].to_numpy(), reg["profit_u"].to_numpy()
    if shuffle_rng is not None:
        ycls, yreg = shuffle_rng.permutation(ycls), shuffle_rng.permutation(yreg)
    out = {}
    if "lr" in which:
        out["lr"] = LR().fit(X_of(cls), ycls)
    if "gbm_cls" in which:
        out["gbm_cls"] = GBMWrap("cls").fit(X_of(cls), ycls)
    if "gbm_reg" in which:
        out["gbm_reg"] = GBMWrap("reg").fit(X_of(reg), yreg)
    return out


def scores(test: pd.DataFrame, models: dict) -> pd.DataFrame:
    s = pd.DataFrame(index=test.index)
    X = X_of(test)
    push, odds = test["push"].to_numpy(float), test["odds"].to_numpy(float)
    if "lr" in models:
        p = np.clip(models["lr"].predict(X), 1e-4, 1 - 1e-4)
        s["p_lr"] = p
        s["meta_lr_ev"] = p * (1 - push) * odds - (1 - push)
        rule = test.assign(probability=p * (1 - push))
        s["meta_lr_rule"] = R.rule_scores(rule)["score"].to_numpy()
    if "gbm_cls" in models:
        p = np.clip(models["gbm_cls"].predict(X), 1e-4, 1 - 1e-4)
        s["p_gbm"] = p
        s["meta_gbm_ev"] = p * (1 - push) * odds - (1 - push)
    if "gbm_reg" in models:
        s["meta_gbm_ret"] = models["gbm_reg"].predict(X)
    return s


def walk_forward(hist: pd.DataFrame, season: pd.DataFrame, which=("lr", "gbm_cls", "gbm_reg"), frac: float = 1.0,
                 rng=None, shuffle: bool = False, train_season: bool = True) -> pd.DataFrame:
    """Out-of-sample meta scores of every season pool row (month M scored by a model fitted before the 1st of M)."""
    out = []
    for month in sorted(season["month"].unique()):
        train = pd.concat([hist, season.loc[season["month"] < month]] if train_season else [hist], ignore_index=True)
        train = train.loc[train["profit_u"].notna()]
        if frac < 1.0:
            ids = train["match_id"].unique()
            keep = set(rng.choice(ids, size=max(50, int(len(ids) * frac)), replace=False))
            train = train.loc[train["match_id"].isin(keep)]
        models = fit_models(train, which, rng if shuffle else None)
        test = season.loc[season["month"] == month]
        out.append(scores(test, models))
    return pd.concat(out).reindex(season.index)


# ------------------------------------------------------------------------------------------------ selection
def pick(pool: pd.DataFrame, score: pd.Series, cands: pd.DataFrame, fallback: pd.DataFrame, name: str) -> pd.DataFrame:
    """Main per match = pool row with the highest score; matches without a pool row keep the production Main."""
    p = pool.assign(_s=score.to_numpy()).sort_values(["match_id", "_s", "probability"], ascending=[True, False, False])
    best = p.drop_duplicates("match_id")[["match_id", "market", "selection"]]
    rows = cands.merge(best, on=["match_id", "market", "selection"])
    rows["note"] = ""
    miss = fallback.loc[~fallback["match_id"].isin(set(rows["match_id"]))].assign(note="fallback: production Main")
    out = pd.concat([rows, miss], ignore_index=True)
    out = out.loc[out["match_id"].isin(set(fallback["match_id"]))]
    out["method"] = name
    out["stated_p"] = out["probability"].astype(float)
    out["ev_stated"] = pd.to_numeric(out["ev"], errors="coerce")
    return out.sort_values("match_id").reset_index(drop=True)


def roi_of(main: pd.DataFrame) -> float:
    b = main.loc[main["bet"] == 1]
    return float(b["profit"].sum() / len(b)) if len(b) else float("nan")


def simple_scores(season: pd.DataFrame) -> dict[str, pd.Series]:
    return {"rule_priced": R.rule_scores(season)["score"], "max_p": season["probability"], "max_ev": season["ev_raw"], "max_q": season["q"],
            "rule_calibrated": R.rule_scores(season.assign(probability=season["pc_cal"] * (1 - season["push"])))["score"]}


def prob_quality(season: pd.DataFrame, s: pd.DataFrame) -> dict:
    d = season.loc[season["y"].notna()]
    out = {"raw": MT.prob_metrics(d["pc"], d["y"]), "calibrated": MT.prob_metrics(d["pc_cal"], d["y"]),
           "market_q": MT.prob_metrics(d["q"], d["y"]),
           "market_q_full_book": MT.prob_metrics(d.loc[d["q_full"] == 1, "q"], d.loc[d["q_full"] == 1, "y"])}
    for col in ("p_lr", "p_gbm"):
        if col in s:
            out[col] = MT.prob_metrics(s.loc[d.index, col], d["y"])
    return out


def run_universe(u: str, log) -> tuple[dict, dict]:
    hist, season = load_pools(u)
    cands = D.season(u)
    base_b = D.baseline(f"baseline_b_{u}")
    base_a = D.baseline("baseline_a")
    played = set(base_b["match_id"])
    log.info("%s: history pool %d rows / %d matches, season pool %d rows / %d matches", u, len(hist),
             hist["match_id"].nunique(), len(season), season["match_id"].nunique())
    info = {"history_rows": int(len(hist)), "history_matches": int(hist["match_id"].nunique()),
            "history_groups": hist["group"].value_counts().to_dict(), "season_rows": int(len(season)),
            "season_matches": int(season["match_id"].nunique()),
            "season_groups": season["group"].value_counts().to_dict(),
            "season_played_matches_without_pool": int(len(played - set(season["match_id"])))}
    s = walk_forward(hist, season)
    allscores = {**{m: s[m] for m in META}, **simple_scores(season)}
    b = base_b.assign(stated_p=base_b["probability"].astype(float), ev_stated=pd.to_numeric(base_b["ev"], errors="coerce"))
    mains = {"baseline_b": b}
    res = {"baseline_b": MT.main_eval(b, base_b, base_a), "baseline_a": {"summary": EV.summary(base_a),
                                                                          "roi_ci": EV.roi_with_ci(base_a)}}
    for name, sc in allscores.items():
        mains[name] = pick(season, sc, cands, b, name)
        res[name] = MT.main_eval(mains[name], base_b, base_a)
        log.info("%s %s: %s", u, name, MT.short(res[name]))
    res["holm"] = C.holm({n: res[n]["vs_b"]["roi_diff"]["p"] for n in META + SIMPLE})
    # random pick (null distribution)
    rng = np.random.default_rng(C.SEED)
    rr = []
    for _ in range(N_RANDOM):
        rr.append(roi_of(pick(season, pd.Series(rng.random(len(season)), index=season.index), cands, b, "random")))
    res["random_pick"] = {"n": N_RANDOM, "mean": float(np.mean(rr)), "q05": float(np.quantile(rr, 0.05)),
                          "q95": float(np.quantile(rr, 0.95)), "values": rr}
    res["prob_quality"] = prob_quality(season, s)
    res["robustness"] = {n: MT.robustness(mains[n], base_b) for n in ["baseline_b"] + META}
    res["info"] = info
    return res, {"mains": mains, "hist": hist, "season": season, "cands": cands, "b": b, "scores": s}


def checks(ctx: dict, log) -> dict:
    """Learning curves, permutation sanity check, in-sample optimism, source of training, coefficients."""
    hist, season, cands, b = ctx["hist"], ctx["season"], ctx["cands"], ctx["b"]
    out = {}
    rng = np.random.default_rng(C.SEED + 1)
    d = season.loc[season["y"].notna()]
    lc = []
    for frac in FRACTIONS:
        for rep in range(3 if frac < 1 else 1):
            s = walk_forward(hist, season, ("lr", "gbm_cls"), frac=frac, rng=rng)
            row = {"frac": frac, "rep": rep}
            for name, col in (("meta_lr_ev", "p_lr"), ("meta_gbm_ev", "p_gbm")):
                row[f"{name}_roi"] = roi_of(pick(season, s[name], cands, b, name))
                row[f"{name}_log_loss"] = MT.log_loss(s.loc[d.index, col], d["y"])
            lc.append(row)
        log.info("learning curve %.2f done", frac)
    out["learning_curve"] = lc
    # trained on 2025/26 only (no 2026/27 rows) and on 2026/27 only
    s_hist = walk_forward(hist, season, ("lr", "gbm_cls"), train_season=False)
    out["history_only"] = {n: roi_of(pick(season, s_hist[n], cands, b, n)) for n in ("meta_lr_ev", "meta_gbm_ev")}
    out["history_only_log_loss"] = {"p_lr": MT.log_loss(s_hist.loc[d.index, "p_lr"], d["y"]),
                                    "p_gbm": MT.log_loss(s_hist.loc[d.index, "p_gbm"], d["y"])}
    # permutation sanity check: targets shuffled inside every training set
    perm = {}
    which = {"meta_lr_ev": ("lr",), "meta_gbm_ev": ("gbm_cls",), "meta_gbm_ret": ("gbm_reg",)}
    for name, n in N_PERM.items():
        vals = []
        for _ in range(n):
            s = walk_forward(hist, season, which[name], rng=rng, shuffle=True)
            vals.append(roi_of(pick(season, s[name], cands, b, name)))
        perm[name] = {"n": n, "values": vals, "mean": float(np.mean(vals)), "max": float(np.max(vals))}
        log.info("permutation %s: %s", name, perm[name]["mean"])
    out["permutation"] = perm
    # in-sample optimism: fitted on the whole season (and 2025/26), evaluated on the same matches
    allt = pd.concat([hist, season], ignore_index=True)
    models = fit_models(allt.loc[allt["profit_u"].notna()])
    s_in = scores(season, models)
    out["in_sample"] = {n: roi_of(pick(season, s_in[n], cands, b, n)) for n in ("meta_lr_ev", "meta_gbm_ev", "meta_gbm_ret")}
    out["in_sample_log_loss"] = {"p_lr": MT.log_loss(s_in.loc[d.index, "p_lr"], d["y"]),
                                 "p_gbm": MT.log_loss(s_in.loc[d.index, "p_gbm"], d["y"])}
    # coefficients (standardised) of the final LR and gain importance of the final GBM
    lr = models["lr"].m
    out["lr_coef"] = dict(sorted(zip(F.FEATURES, lr.coef_[0].tolist()), key=lambda kv: -abs(kv[1])))
    imp = models["gbm_cls"].m.get_booster().get_score(importance_type="gain")
    out["gbm_gain"] = dict(sorted(((F.FEATURES[int(k[1:])], v) for k, v in imp.items()), key=lambda kv: -kv[1]))
    return out


def main() -> None:
    started = time.time()
    log = C.setup_logging("meta_selection", F.DIRS["logs"] / "run_python.log")
    out = {"primary": PRIMARY, "meta": META, "simple": SIMPLE, "features": F.FEATURES}
    m = D.matches()
    res = D.results(m)
    ctxs = {}
    for u in ("close", "t09"):
        out[u], ctxs[u] = run_universe(u, log)
    out["checks"] = {"close": checks(ctxs["close"], log)}
    for u in ("close", "t09"):
        mains = ctxs[u]["mains"]
        for name in ("baseline_b", PRIMARY, "meta_gbm_ev"):
            EV.per_match_table(MT.with_teams(mains[name], m), res, name, "prod_20261012 + " + (name if name != "baseline_b" else "rule")
                               ).to_csv(RES / f"per_match_{name}_{u}.csv", index=False)
        out[u]["coverage"] = MT.coverage(m, {n: mains[n] for n in ("baseline_b", PRIMARY, "meta_gbm_ev")})
    C.write_json(RES / "meta_selection.json", out)
    C.write_json(RES / "run_info.json", C.run_info("v6_3/meta_selection", started, {
        "band": F.BAND, "primary": PRIMARY, "lr": "StandardScaler + LogisticRegression(C=0.1)",
        "gbm": "XGBoost depth 3, 300 trees, lr 0.03, subsample 0.8, colsample 0.8, n_jobs 2",
        "training": "2025/26 backtest pool + 2026/27 pool of the universe settled before the 1st of the month",
        "fractions": FRACTIONS, "n_perm": N_PERM, "n_random": N_RANDOM}))
    log.info("done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
