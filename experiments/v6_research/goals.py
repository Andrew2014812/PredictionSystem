"""Q2 — home / away goal models: production XGBoost (count:poisson) vs Poisson GLM vs a blend.

GLM: sklearn PoissonRegressor on the production feature set (median-imputed, standardised,
missing indicators — experiments/model_comparison/data.Matrix); the L2 strength is chosen on
the inner validation season. Blend: (1-w) XGBoost + w GLM, w chosen on the inner season.
An alternative replaces production only if its Poisson deviance is lower on BOTH 2023 and 2024.
"""
from __future__ import annotations

import time

import numpy as np

from src import config
from experiments.model_comparison.models import PoissonGLMCandidate

from . import common as C

log = C.setup_logging("goals")
TARGETS = {"home_goals": "lh", "away_goals": "la"}


def run() -> dict:
    t_start = time.time()
    df = C.features()
    out = {"folds": {}}
    preds = {}
    for s in C.FOLDS:
        inner, outer, info = C.base_predictions(s)
        fold = {}
        for task, col in TARGETS.items():
            tr_in, va_in = C.before(df, s - 1), C.seasons(df, [s - 1])
            tr_out, te = C.before(df, s), C.seasons(df, [s])
            y_in, y = va_in[task].to_numpy(float), te[task].to_numpy(float)
            t0 = time.time()
            glm = PoissonGLMCandidate("count", C.task_numeric(task))
            glm.tune(tr_in, va_in, tr_in[task].to_numpy(float), y_in)
            tune_sec = time.time() - t0
            t0 = time.time()
            glm.fit(tr_out, tr_out[task].to_numpy(float))
            glm_out = glm.predict(te)
            fit_sec = time.time() - t0
            xgb_in, xgb_out = inner[col].to_numpy(), outer[col].to_numpy()
            grid = np.round(np.arange(0, 1.0001, 0.1), 1)
            dev = {float(w): float(C.poisson_dev(y_in, (1 - w) * xgb_in + w * glm.valid_pred).mean()) for w in grid}
            w = min(dev, key=dev.get)
            blend = (1 - w) * xgb_out + w * glm_out
            league_mean = tr_out.groupby("league")[task].mean()
            naive = te["league"].map(league_mean).fillna(tr_out[task].mean()).to_numpy()
            variants = {"xgb_production": xgb_out, "poisson_glm": glm_out, "blend": blend, "naive_league_mean": naive}
            base_dev = C.poisson_dev(y, xgb_out)
            fold[task] = {
                "variants": {n: {**C.count_metrics(y, mu),
                                 "delta_deviance_vs_xgb": C.paired_bootstrap(C.poisson_dev(y, mu), base_dev)}
                             for n, mu in variants.items()},
                "params": {"glm_l2": glm.params["regularisation"], "blend_weight_glm": w, "blend_grid_inner": dev},
                "train_seconds": {"xgb_production": info["outer"]["seconds"][task], "poisson_glm": fit_sec,
                                  "poisson_glm_tuning": tune_sec,
                                  "blend": info["outer"]["seconds"][task] + info["inner"]["seconds"][task]
                                  + fit_sec + tune_sec, "naive_league_mean": 0.0},
            }
            preds[(s, task)] = (y, variants)
            log.info("fold %s %s: %s", s, task, {n: round(v["poisson_deviance"], 4)
                                                  for n, v in fold[task]["variants"].items()})
        out["folds"][str(s)] = fold

    out["decision"] = decide(out)
    out["minutes"] = (time.time() - t_start) / 60
    C.save_json("goals.json", out)
    figures(out, preds)
    return out


def decide(out: dict) -> dict:
    """An alternative replaces production only if it is CLEARLY better on both selection folds:
    lower Poisson deviance and the 95 % paired-bootstrap interval of the difference below 0."""
    decision = {}
    for task in TARGETS:
        base = [out["folds"][str(s)][task]["variants"]["xgb_production"]["poisson_deviance"] for s in C.SELECT]
        better = {}
        for n in ("poisson_glm", "blend"):
            v = [out["folds"][str(s)][task]["variants"][n] for s in C.SELECT]
            alt = [x["poisson_deviance"] for x in v]
            better[n] = {"deviance_by_fold": dict(zip(map(str, C.SELECT), alt)),
                         "better_both": bool(all(a < b for a, b in zip(alt, base))),
                         "ci95_upper_by_fold": [x["delta_deviance_vs_xgb"]["ci95"][1] for x in v],
                         "clearly_better_both": bool(all(a < b for a, b in zip(alt, base))
                                                     and all(x["delta_deviance_vs_xgb"]["ci95"][1] < 0 for x in v))}
        ok = [n for n, b in better.items() if b["clearly_better_both"]]
        chosen = min(ok, key=lambda n: sum(better[n]["deviance_by_fold"].values())) if ok else "xgb_production"
        decision[task] = {"baseline_deviance_by_fold": dict(zip(map(str, C.SELECT), base)),
                          "alternatives": better, "chosen": chosen,
                          "rule": "lower deviance on both 2023 and 2024 AND bootstrap 95 % CI of the difference "
                                  "entirely below 0 on both"}
    return decision


def figures(out, preds):
    plt = C.plot_style()
    names = {"xgb_production": "XGBoost (прод.)", "poisson_glm": "Poisson GLM", "blend": "Бленд"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for r, task in enumerate(TARGETS):
        for c, s in enumerate(C.FOLDS):
            ax = axes[r, c]
            y, var = preds[(s, task)]
            bins = np.linspace(-3, 6, 37)
            for k, n in enumerate(names):
                ax.hist(y - var[n], bins=bins, histtype="step", lw=1.4, color=C.PALETTE[k],
                        label=f"{names[n]} (среднее {np.mean(y - var[n]):+.3f})")
            ax.set_title(f"{'Голы хозяев' if task == 'home_goals' else 'Голы гостей'}, "
                         f"{config.season_label(s)}{'' if s in C.SELECT else ' (только для информации)'}", fontsize=9)
            ax.set_xlabel("Остаток: факт − прогноз"), ax.set_ylabel("Матчей")
            ax.legend(fontsize=7)
    fig.suptitle("Распределение остатков моделей голов")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "goals_residuals.png"), plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for r, task in enumerate(TARGETS):
        for c, s in enumerate(C.FOLDS):
            ax = axes[r, c]
            for k, n in enumerate(names):
                cal = out["folds"][str(s)][task]["variants"][n]["calibration_by_decile"]
                ax.plot([b["predicted"] for b in cal], [b["observed"] for b in cal], "o-", ms=3,
                        color=C.PALETTE[k], label=names[n])
            lo, hi = (0.6, 2.6) if task == "home_goals" else (0.4, 2.2)
            ax.plot([lo, hi], [lo, hi], ":", color="grey")
            ax.set_title(f"{'Голы хозяев' if task == 'home_goals' else 'Голы гостей'}, "
                         f"{config.season_label(s)}{'' if s in C.SELECT else ' (только для информации)'}", fontsize=9)
            ax.set_xlabel("Средний прогноз в децили (ожидаемые голы)"), ax.set_ylabel("Средний факт")
            ax.legend(fontsize=7)
    fig.suptitle("Калибровка ожидаемых голов по децилям прогноза")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "goals_calibration.png"), plt.close(fig)


if __name__ == "__main__":
    run()
