"""Build the HTML research report from the saved fold results (no training).

    python -m experiments.model_comparison.report            # -> report/model_comparison.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.features.store import load_features

from .data import REPORT_SEASON, SELECTION_SEASONS

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "model_comparison.html"
RNG = np.random.default_rng(11)
TASK_LABEL = {"result": "Match result (1X2)", "home_goals": "Home goals", "away_goals": "Away goals",
              "corners": "Total corners"}
PRIMARY = {"result": ("log_loss", True), "home_goals": ("poisson_deviance", True),
           "away_goals": ("poisson_deviance", True), "corners": ("poisson_deviance", True)}
_plotly_loaded = False


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, digits=4, pct=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def table(df: pd.DataFrame, pct_cols=(), digits: int = 4, best_low=(), best_high=()) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            text = fmt(v, digits, c in pct_cols) if isinstance(v, (float, np.floating)) else esc(v)
            best = ((c in best_low and v == df[c].min()) or (c in best_high and v == df[c].max()))
            cells.append(f"<td class='{'best' if best else ''}'>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def chart(fig: go.Figure, height: int = 360) -> str:
    global _plotly_loaded
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=50, b=40), template="plotly_white")
    out = fig.to_html(full_html=False, include_plotlyjs="cdn" if not _plotly_loaded else False)
    _plotly_loaded = True
    return out


def load_folds(quick: bool = False) -> dict[int, dict]:
    out = {}
    for path in sorted(RESULTS.glob("fold_*.json")):
        if ("_quick" in path.stem) != quick:
            continue
        season = int(path.stem.split("_")[1])
        out[season] = {"res": json.loads(path.read_text()),
                       "preds": pd.read_parquet(path.with_suffix(".parquet"))}
    return out


# -- per-match losses for the bootstrap -----------------------------------------------------------
def per_match_loss(task: str, preds: pd.DataFrame, name: str, truth: pd.DataFrame) -> pd.Series:
    if task == "result":
        cols = [f"result|{name}|{c}" for c in ("H", "D", "A")]
        if not all(c in preds for c in cols):
            return pd.Series(dtype=float)
        p = preds[cols].dropna()
        y = truth.loc[p.index, "result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        return pd.Series(-np.log(np.clip(p.to_numpy()[np.arange(len(p)), y], 1e-12, 1)), index=p.index)
    col = f"{task}|{name}|"
    if col not in preds:
        return pd.Series(dtype=float)
    mu = np.clip(preds[col].dropna(), 1e-6, None)
    target = {"home_goals": "home_goals", "away_goals": "away_goals", "corners": "total_corners"}[task]
    y = truth.loc[mu.index, target].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * (np.where(y > 0, y * np.log(y / mu.to_numpy()), 0.0) - (y - mu.to_numpy()))
    return pd.Series(dev, index=mu.index)


def bootstrap(a: pd.Series, b: pd.Series, n: int = 2000) -> tuple[float, float, float]:
    """Mean of (b - a) over shared matches with a paired 95 % interval."""
    idx = a.index.intersection(b.index)
    d = (b.loc[idx] - a.loc[idx]).to_numpy()
    draws = RNG.integers(0, len(d), (n, len(d)))
    means = d[draws].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    import sys
    quick = "--quick" in sys.argv
    folds = load_folds(quick)
    if not folds:
        raise SystemExit("no results: run python -m experiments.model_comparison.run first")
    truth = load_features().set_index("match_id")[["result", "home_goals", "away_goals", "total_corners"]]
    sel = [s for s in SELECTION_SEASONS if s in folds] or sorted(folds)     # smoke test: whatever exists
    parts = ["<h1>FootPredict v5 — comparison of alternative ML models</h1>"
             "<p class='muted'>Research report, not part of the website. Generated from "
             "<code>experiments/model_comparison</code>. Nothing here changes the production models.</p>"]

    # -- methodology ------------------------------------------------------------------------------------
    rows = [{"Evaluated season": folds[s]["res"]["season"], "Tuning train rows": folds[s]["res"]["rows"]["tune_train"],
             "Tuning validation rows": folds[s]["res"]["rows"]["tune_valid"],
             "Fit rows (all earlier seasons)": folds[s]["res"]["rows"]["fit"],
             "Evaluated matches": folds[s]["res"]["rows"]["test"],
             "Role": "selection" if s in SELECTION_SEASONS else "reported only (looked at in v1–v4)"}
            for s in sorted(folds)]
    parts.append("<h2>Method</h2><ul>"
                 "<li>Data: Football-Data results and statistics of 21 leagues, the same leak-free pre-match "
                 "features as FootPredict (each match only sees matches played before its day).</li>"
                 "<li>Walk-forward folds: for an evaluated season <i>s</i>, every algorithm is tuned on "
                 "train = seasons before <i>s</i>−1 and validation = <i>s</i>−1, then refitted on all seasons before "
                 "<i>s</i> and scored on <i>s</i>. Identical rows, targets, features (one-hot league; linear models "
                 "additionally median-imputed and standardised with missing-value flags) and metrics for every "
                 "algorithm.</li>"
                 f"<li>Winners are chosen on the folds {', '.join(config.season_label(s) for s in sel)}. "
                 f"{config.season_label(REPORT_SEASON)} was the test season of v1–v4 and has been looked at; it is "
                 "reported but not used to choose.</li>"
                 "<li>Differences are tested with a paired bootstrap over matches (95 % interval of the mean "
                 "per-match loss difference against the current XGBoost). A model is called better only when the "
                 "whole interval is on the good side.</li>"
                 "<li>ROI uses real Bet365 prices (Football-Data) only, the same matches and the same rule for "
                 "every model; such samples are small relative to the noise of betting returns.</li></ul>")
    parts.append(table(pd.DataFrame(rows)))

    # -- per task tables, bootstrap and winners ---------------------------------------------------------------
    winners = {}
    for task in ("result", "home_goals", "away_goals", "corners"):
        metric, low = PRIMARY[task]
        parts.append(f"<h2>{TASK_LABEL[task]}</h2>")
        names = list(folds[max(folds)]["res"][task]["metrics"])
        rows = []
        for name in names:
            r = {"Model": name}
            for s in sorted(folds):
                m = folds[s]["res"][task]["metrics"].get(name)
                r[f"{metric} {folds[s]['res']['season']}"] = m[metric] if m else np.nan
            r[f"{metric} mean (selection folds)"] = float(np.nanmean([r[f"{metric} {folds[s]['res']['season']}"]
                                                                       for s in sel])) if sel else np.nan
            rows.append(r)
        df = pd.DataFrame(rows).sort_values(f"{metric} mean (selection folds)")
        parts.append(table(df, best_low=[c for c in df.columns if c != "Model"]))
        # bootstrap vs XGBoost on the pooled selection folds
        boot_rows = []
        for name in names:
            if name == "XGBoost":
                continue
            a = pd.concat([per_match_loss(task, folds[s]["preds"], "XGBoost", truth) for s in sel])
            b = pd.concat([per_match_loss(task, folds[s]["preds"], name, truth) for s in sel])
            if a.empty or b.empty:
                continue
            d, lo, hi = bootstrap(a, b)
            verdict = "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
            boot_rows.append({"Model": name, "Mean loss difference vs XGBoost": d, "95% low": lo, "95% high": hi,
                              "Verdict": verdict})
        if boot_rows:
            bdf = pd.DataFrame(boot_rows)
            parts.append("<p class='muted'>Paired bootstrap on the selection folds (negative = lower loss than "
                         "the current XGBoost).</p>" + table(bdf, digits=5))
            better = bdf.loc[bdf["Verdict"] == "better"].sort_values("Mean loss difference vs XGBoost")
            winners[task] = (better.iloc[0]["Model"] + " (clearly better than XGBoost)" if not better.empty
                             else df.iloc[0]["Model"] + " (best mean, but not clearly better than XGBoost)"
                             if df.iloc[0]["Model"] != "XGBoost" else "XGBoost")
        last = folds[max(folds)]["res"][task]["metrics"]
        if task == "result":
            rows = [{"Model": n, "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"],
                     "Balanced accuracy": m["balanced_accuracy"], "Log loss": m["log_loss"], "Brier": m["brier"],
                     "Calibration error": m["ece"], "Home recall": m["recall"]["H"], "Draw recall": m["recall"]["D"],
                     "Away recall": m["recall"]["A"], "Draw precision": m["draw_precision"],
                     "Predicted draw share": m["predicted_share"]["D"], "Actual draw share": m["actual_share"]["D"]}
                    for n, m in last.items()]
            parts.append(f"<h3>All 1X2 metrics, {folds[max(folds)]['res']['season']}</h3>" + table(
                pd.DataFrame(rows), pct_cols=("Accuracy", "Balanced accuracy", "Home recall", "Draw recall",
                                              "Away recall", "Draw precision", "Predicted draw share",
                                              "Actual draw share", "Calibration error"),
                best_low=("Log loss", "Brier", "Calibration error"), best_high=("Accuracy", "Macro F1")))
            figs = []
            for cls in ("H", "D", "A"):
                fig = go.Figure()
                fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="perfect")
                for n, m in last.items():
                    rel = pd.DataFrame(m["reliability"][cls])
                    if not rel.empty:
                        fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers", name=n)
                fig.update_layout(title=f"Calibration — {cls}", xaxis_title="predicted", yaxis_title="observed")
                figs.append(chart(fig, 380))
            parts.append("<div class='grid3'>" + "".join(f"<div>{f}</div>" for f in figs) + "</div>")
            cms = []
            for n, m in last.items():
                z = np.array(m["confusion"])
                fig = go.Figure(go.Heatmap(z=z, x=["predicted H", "predicted D", "predicted A"],
                                           y=["actual H", "actual D", "actual A"], text=z, texttemplate="%{text}",
                                           showscale=False, colorscale="Greens"))
                fig.update_layout(title=f"Confusion matrix — {n}", yaxis=dict(autorange="reversed"))
                cms.append(chart(fig, 300))
            parts.append("<h3>Confusion matrices (rows = actual, columns = predicted)</h3><div class='grid3'>"
                         + "".join(f"<div>{c}</div>" for c in cms) + "</div>")
        else:
            rows = [{"Model": n, "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"], "Poisson deviance":
                     m["poisson_deviance"], "Mean actual": m["mean_actual"], "Mean predicted": m["mean_predicted"],
                     **{f"P({k}) pred / obs": f"{v['predicted']:.3f} / {v['observed']:.3f}"
                        for k, v in m["count_distribution"].items()}} for n, m in last.items()]
            parts.append(f"<h3>All metrics, {folds[max(folds)]['res']['season']}</h3>"
                         + table(pd.DataFrame(rows), best_low=("MAE", "RMSE", "Poisson deviance"), best_high=("R²",)))
        params = {n: v["params"] for n, v in folds[max(folds)]["res"][task]["candidates"].items()}
        parts.append("<details><summary>Hyperparameters chosen in the last fold</summary><pre>"
                     + esc(json.dumps(params, indent=1, default=str)) + "</pre></details>")

    # -- corners lines --------------------------------------------------------------------------------------------
    parts.append("<h2>Corners over 9.5 (from each count model, negative-binomial lines)</h2>")
    rows = []
    for name in folds[max(folds)]["res"]["corners_lines"]:
        r = {"Model": name}
        for s in sorted(folds):
            m = folds[s]["res"]["corners_lines"].get(name)
            r[f"log loss {folds[s]['res']['season']}"] = m["log_loss"] if m else np.nan
            r[f"ECE {folds[s]['res']['season']}"] = m["ece"] if m else np.nan
        rows.append(r)
    parts.append(table(pd.DataFrame(rows), best_low=[c for c in pd.DataFrame(rows).columns if c != "Model"]))

    # -- derived markets ---------------------------------------------------------------------------------------------
    parts.append("<h2>Markets derived from each system's score matrix</h2><p class='muted'>System = the 1X2 model "
                 "and the two goal models of the same algorithm family; markets come from the 1X2-reconciled "
                 "Poisson score matrix exactly as in FootPredict. Log loss, lower is better; naive = always the "
                 "training base rate.</p>")
    keys = ["over15", "over25", "over35", "btts", "dc_1x", "dc_x2", "dc_12", "asian_handicap_-0.5",
            "asian_handicap_-1", "asian_handicap_+1", "asian_handicap_-1.5"]
    for s in sorted(folds):
        dm = folds[s]["res"]["derived_markets"]
        rows = []
        for system, v in dm.items():
            r = {"System": system}
            for k in keys:
                if k in v:
                    r[k] = v[k]["log_loss"]
            r["exact score log loss"] = v["exact_score"]["log_loss"]
            rows.append(r)
        if rows:
            any_v = next(iter(dm.values()))
            rows.append({"System": "naive base rate", **{k: any_v[k]["naive_log_loss"] for k in keys if k in any_v}})
            df = pd.DataFrame(rows)
            parts.append(f"<h3>{folds[s]['res']['season']}</h3>"
                         + table(df, best_low=[c for c in df.columns if c != "System"]))
    # calibration of over 2.5 / BTTS in the last fold
    last = folds[max(folds)]["res"]["derived_markets"]
    figs = []
    for key, title in (("over25", "Over 2.5"), ("btts", "BTTS yes")):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="perfect")
        for system, v in last.items():
            rel = pd.DataFrame(v[key]["reliability"])
            if not rel.empty:
                fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers", name=system)
        fig.update_layout(title=f"Calibration — {title}", xaxis_title="predicted", yaxis_title="observed")
        figs.append(chart(fig, 380))
    parts.append("<div class='grid2'>" + "".join(f"<div>{f}</div>" for f in figs) + "</div>")

    # -- direct market models ------------------------------------------------------------------------------------------
    parts.append("<h2>Direct market models vs derived probabilities</h2><p class='muted'>A direct model is a "
                 "binary classifier trained on the market outcome itself.</p>")
    for s in sorted(folds):
        rows = []
        for target, v in folds[s]["res"]["direct_markets"].items():
            for name, m in v["metrics"].items():
                rows.append({"Market": target, "Model": f"direct · {name}", "Log loss": m["log_loss"],
                             "Brier": m["brier"], "ECE": m["ece"], "Naive log loss": m["naive_log_loss"]})
            key = {"btts": "btts", "over15": "over15", "over25": "over25", "over35": "over35"}.get(target)
            for system, dv in folds[s]["res"]["derived_markets"].items():
                if key and key in dv:
                    rows.append({"Market": target, "Model": f"derived · {system}", "Log loss": dv[key]["log_loss"],
                                 "Brier": dv[key]["brier"], "ECE": dv[key]["ece"],
                                 "Naive log loss": dv[key]["naive_log_loss"]})
            if target == "corners_over95":
                for name, m in folds[s]["res"]["corners_lines"].items():
                    rows.append({"Market": target, "Model": f"count model · {name}", "Log loss": m["log_loss"],
                                 "Brier": m["brier"], "ECE": m["ece"], "Naive log loss": m["naive_log_loss"]})
        df = pd.DataFrame(rows)
        if not df.empty:
            parts.append(f"<h3>{folds[s]['res']['season']}</h3>" + table(df, pct_cols=("ECE",)))
    parts.append("<h3>Consistency of separately trained market models</h3>")
    for s in sorted(folds):
        for model, v in folds[s]["res"].get("consistency", {}).items():
            row = {"Season": folds[s]["res"]["season"], "Model": model,
                   **{f"violations: {k}": val for k, val in v["violation_rate"].items()},
                   **{f"{k} log loss before → after repair": f"{v['log_loss_before_repair'][k]:.4f} → "
                                                              f"{v['log_loss_after_repair'][k]:.4f}"
                      for k in v["log_loss_before_repair"]}}
            parts.append(table(pd.DataFrame([row]), pct_cols=tuple(c for c in row if c.startswith("violations"))))
    parts.append("<p>Derived markets come from one score matrix, so they can never contradict each other "
                 "(P(over 3.5) ≤ P(over 2.5), double chance = sums of 1X2, totals and exact scores agree). Separate "
                 "direct models can; the repair above (pool-adjacent averaging across lines) restores the order. A "
                 "cleaner fix is to keep the score matrix and only tilt it towards a direct model's totals.</p>")

    # -- ROI --------------------------------------------------------------------------------------------------------
    parts.append("<h2>Return at real Bet365 prices (same matches, same rule for every system)</h2>"
                 "<p class='muted'>1X2 value: the outcome with the highest EV when EV &gt; 0. 1X2 most likely: always "
                 "the most probable outcome. Over/under 2.5 value: the side with positive EV. Stake 1.</p>")
    rows = []
    for s in sorted(folds):
        for system, v in folds[s]["res"]["derived_markets"].items():
            for key, label in (("roi_1x2_value", "1X2 value"), ("roi_1x2_most_likely", "1X2 most likely"),
                               ("roi_over_under_25_value", "O/U 2.5 value")):
                r = v.get(key)
                if r:
                    rows.append({"Season": folds[s]["res"]["season"], "System": system, "Strategy": label,
                                 "Bets": r["bets"], "Hit rate": r["hit_rate"], "Profit": r["profit"],
                                 "ROI": r["roi"], "Average odds": r["avg_odds"]})
    if rows:
        parts.append(table(pd.DataFrame(rows), pct_cols=("Hit rate", "ROI"), digits=2))
    parts.append("<p class='muted'>With a few thousand bets a season the standard error of ROI is about 2 "
                 "percentage points (1X2), so differences of 1–3 points between systems are noise.</p>")

    # -- summary ----------------------------------------------------------------------------------------------------
    direct_best = {}
    for target in ("btts", "over25"):
        cands = []
        for s in sel:
            for name, m in folds[s]["res"]["direct_markets"][target]["metrics"].items():
                cands.append((f"direct · {name}", m["log_loss"]))
            for system, dv in folds[s]["res"]["derived_markets"].items():
                cands.append((f"derived · {system}", dv[target]["log_loss"]))
        if cands:
            mean = pd.DataFrame(cands, columns=["m", "ll"]).groupby("m")["ll"].mean().sort_values()
            direct_best[target] = f"{mean.index[0]} (mean log loss {mean.iloc[0]:.4f}; next {mean.index[1]} " \
                                  f"{mean.iloc[1]:.4f})"
    summary = "".join(f"<li><b>{TASK_LABEL[t]}</b>: {esc(w)}</li>" for t, w in winners.items())
    summary += "".join(f"<li><b>{'BTTS' if t == 'btts' else 'Totals (over 2.5)'}</b>: {esc(w)}</li>"
                       for t, w in direct_best.items())
    parts.insert(1, "<h2>Summary</h2><ul>" + summary + "</ul><p>Results that are close or not significant are "
                 "reported as such; no experimental model replaces the FootPredict production model without a "
                 "separate decision.</p>")

    style = ("<style>body{font-family:Inter,Arial,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;"
             "color:#1b2636;background:#f4f6f9} h1{font-size:1.7rem} h2{margin-top:2.2rem;border-bottom:2px solid "
             "#16a34a;padding-bottom:.3rem} table{border-collapse:collapse;margin:.6rem 0 1.2rem;background:#fff;"
             "font-size:.85rem} th,td{border:1px solid #d5dce6;padding:.35rem .55rem;text-align:right} "
             "th:first-child,td:first-child{text-align:left} th{background:#eef2f7} td.best{background:#dcfce7;"
             "font-weight:700} .muted{color:#4d5b6e} .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}"
             " .grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem} pre{font-size:.75rem;"
             "background:#fff;padding:.6rem;overflow:auto}</style>")
    out = OUT.with_name("model_comparison_quick.html") if quick else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!doctype html><html><head><meta charset='utf-8'><title>FootPredict model comparison</title>"
                   + style + "</head><body>" + "".join(parts) + "</body></html>", encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
