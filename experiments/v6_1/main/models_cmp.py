"""Analysis 8 — are the probability models the problem? Alternative learners plugged into the same Main rule.

No model is retrained here: the walk-forward out-of-sample predictions of experiments/model_comparison
(results/fold_<season>.parquet: XGBoost, improved XGBoost, CatBoost, LightGBM, logistic / Poisson regression and
their ensembles for 1X2 and both goal counts; fitted on seasons < s with that study's own tuning) replace the
production probabilities; corners stay the production XGBoost (they are almost never the Main). Also a pure
Poisson goal model (1X2 from the goal rates without the classifier) and the production XGBoost.

For each model and fold: 1X2 log loss / Brier against the de-margined Bet365 price, the information test
y ~ logit q + (logit p - logit q), the inner-fitted blend weight (folds with a previous fold), and the Main of
strategy A on identical bettable matches: ROI, hit rate, over-confidence of the chosen pick.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

from . import base
from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE
from . import transforms as T

MC = config.ROOT_DIR / "experiments" / "model_comparison" / "results"
SYSTEMS = {"XGBoost (сравнение)": ("XGBoost", "XGBoost", "XGBoost"),
           "Improved XGBoost": ("Improved XGBoost", "Improved XGBoost", "Improved XGBoost"),
           "CatBoost": ("CatBoost", "CatBoost", "CatBoost"), "LightGBM": ("LightGBM", "LightGBM", "LightGBM"),
           "Логистическая + Пуассон GLM": ("Logistic regression", "Poisson regression", "Poisson regression"),
           "Ансамбль": ("Ensemble", "Ensemble", "Ensemble")}


def _col(frame: pd.DataFrame, task: str, model: str, sel: str = "") -> str:
    cands = [c for c in frame.columns if c.startswith(f"{task}|{model}") and c.endswith(f"|{sel}")]
    if model == "XGBoost":
        cands = [c for c in cands if c == f"{task}|XGBoost|{sel}"]
    if not cands:
        raise KeyError(f"{task}|{model}|{sel}")
    return cands[0]


def predictions(season: int) -> tuple[dict[str, pd.DataFrame], dict]:
    prod, info = base.outer(season)
    out = {"XGBoost (продакшн)": prod}
    path = MC / f"fold_{season}.parquet"
    poisson = prod.copy()
    from experiments.v6_research import scores as SC
    m = SC.matrices(prod["lh"].to_numpy(), prod["la"].to_numpy())
    p1 = SC.outcome_1x2(m)
    poisson["pH"], poisson["pD"], poisson["pA"] = p1[:, 0], p1[:, 1], p1[:, 2]
    out["Пуассон голов (без классификатора)"] = poisson
    if not path.exists():
        return out, info
    mc = pd.read_parquet(path)
    mc = mc.reindex(prod["match_id"].to_numpy())
    for name, (res, hg, ag) in SYSTEMS.items():
        try:
            x = prod.copy()
            x["pH"] = mc[_col(mc, "result", res, "H")].to_numpy()
            x["pD"] = mc[_col(mc, "result", res, "D")].to_numpy()
            x["pA"] = mc[_col(mc, "result", res, "A")].to_numpy()
            x["lh"] = mc[_col(mc, "home_goals", hg)].to_numpy()
            x["la"] = mc[_col(mc, "away_goals", ag)].to_numpy()
            if x[["pH", "lh", "la"]].isna().mean().max() > 0.01:
                continue
            out[name] = x
        except KeyError as exc:
            C.log.info("model comparison column missing: %s", exc)
    return out, info


def one(long: pd.DataFrame, inner: pd.DataFrame | None) -> dict:
    u = SE.universe(long)
    y1 = u.loc[u["market"] == "1X2"]
    res = {}
    for name, d in (("1X2", y1), ("TOTAL_2.5", u.loc[u["market"] == "TOTAL_2.5"])):
        ok = d["status"].isin([0, 1]) & d["q_prod_shin"].notna()
        dd = d.loc[ok]
        res[f"{name} model"] = C.binary_scores((dd["status"] == 1).astype(float), dd["p"])
        res[f"{name} market"] = C.binary_scores((dd["status"] == 1).astype(float), dd["q_prod_shin"])
    # multiclass 1X2 log loss
    w = y1.pivot_table(index="match_id", columns="selection", values=["p", "q_prod_shin", "status"], aggfunc="first")
    st = w["status"][["H", "D", "A"]].to_numpy()
    ok = (st == 1).sum(1) == 1
    for k, col in (("model", "p"), ("market", "q_prod_shin")):
        pr = np.clip(w[col][["H", "D", "A"]].to_numpy()[ok], 1e-9, 1)
        res[f"1X2 multiclass log loss {k}"] = float(-np.mean(np.log((pr * (st[ok] == 1)).sum(1))))
    res["disagreement"] = T.disagreement_fit(u).get("ALL")
    if inner is not None:
        res["blend_weight_inner"] = T.Blend("prod", "shin").fit(SE.universe(inner)).w
    P = ST.CURRENT.with_(priced_only=True)
    o = ST.outcome(u, ST.choose(u, P))
    res["main_bettable"] = ST.summary(o, int(u["mi"].max() + 1))
    o2 = ST.outcome(long, ST.choose(long, ST.CURRENT))
    res["main_production"] = ST.summary(o2, int(long["mi"].max() + 1))
    return res


def run() -> dict:
    out = {}
    prev: dict[str, pd.DataFrame] = {}
    for s in C.FOLDS:
        preds, info = predictions(s)
        fd = CD.fd_prices.prices(s)
        out[str(s)] = {}
        cur = {}
        for name, pred in preds.items():
            long = CD.long_from_pred(pred, info["nb_size"], s, fd)
            out[str(s)][name] = one(long, prev.get(name))
            cur[name] = long
            C.log.info("model %s %s: ROI %.4f", s, name, out[str(s)][name]["main_bettable"]["roi"])
        prev = cur
    C.save_json("models_cmp.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    names = list(out[str(C.FOLDS[-1])].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, s in enumerate(C.FOLDS):
        r = out[str(s)]
        ns = [n for n in names if n in r]
        axes[0].bar(np.arange(len(ns)) + (k - 1) * 0.27,
                    [(r[n]["1X2 multiclass log loss model"] - r[n]["1X2 multiclass log loss market"]) for n in ns],
                    0.27, color=PALETTE[k], label=C.period_label(s))
        axes[1].bar(np.arange(len(ns)) + (k - 1) * 0.27, [r[n]["main_bettable"]["roi"] * 100 for n in ns], 0.27,
                    color=PALETTE[k], label=C.period_label(s))
        for i, n in enumerate(ns):
            lo, hi = r[n]["main_bettable"]["roi_ci95"]
            axes[1].plot([i + (k - 1) * 0.27] * 2, [lo * 100, hi * 100], color="#333", lw=0.6)
    for ax in axes:
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="#555", lw=0.6)
    axes[0].set_title("Log loss 1X2: модель − рынок (Bet365 без маржи)", fontsize=9.5)
    axes[1].set_title("ROI Main (A) на одних матчах, %", fontsize=9.5)
    axes[0].legend(fontsize=7)
    save_chart(fig, C.FIGURES / "models_cmp.png")


if __name__ == "__main__":
    C.setup_logging("models")
    run()
