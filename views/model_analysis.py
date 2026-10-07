"""Model Analysis (technical / diploma page): periods, metrics, draws, calibration, SHAP."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.features.names import feature_label
from src.features.sets import MODEL_GROUPS
from src.ui import data
from src.ui.components import (empty_state, esc, guard, html_table, kpi_row, num, page_header, pct, plotly,
                               section)
from src.ui.i18n import current_language, t
from src.ui.theme import colors

METRIC_HELP = {
    "Accuracy": "Share of matches where the predicted outcome was the actual one. Higher is better.",
    "Macro F1": "Average quality over the three outcomes (home, draw, away); draws count as much as home wins. "
                "Higher is better.",
    "Balanced accuracy": "Average recall per outcome: how often each actual outcome was recognised. Higher is better.",
    "Log loss": "Measures the quality of predicted probabilities; punishes confident wrong forecasts. "
                "Lower is better.",
    "Brier score": "Measures probability forecast error (mean squared error of the probabilities). Lower is better.",
    "RMSE": "Typical prediction error, with larger mistakes penalised more heavily. Lower is better.",
    "MAE": "Average absolute prediction error (in goals or corners). Lower is better.",
    "R²": "Shows how much of the variation in the target is explained by the model. Higher is better.",
    "Poisson deviance": "Proper error measure for count forecasts (goals, corners). Lower is better.",
}

GROUP_NAMES = {
    "table": "League table", "season_attack": "Season statistics", "season_corners": "Corner statistics",
    "discipline": "Discipline (cards, fouls)", "venue_result": "Home / away", "venue_goals": "Home / away",
    "venue_corners": "Corner statistics", "form_result": "Recent form 5 / 10 / 15",
    "form_goals": "Attack / defence", "form_corners": "Corner statistics", "frequencies": "Event frequencies",
    "trends_goals": "Trends", "trends_corners": "Trends", "rest": "Rest days", "h2h_result": "Head-to-head",
    "h2h_goals": "Head-to-head", "diff_result": "Strength differences", "diff_attack": "Attack / defence",
    "diff_corners": "Corner statistics", "league_result": "League context", "league_goals": "League context",
    "league_corners": "League context", "venue_form_goals": "Home / away", "venue_form_corners": "Corner statistics",
    "corner_pressure": "Attacking pressure", "market": "Bookmaker probabilities",
}
TASK_TITLES = {"result": "Result model (1X2)", "home_goals": "Home goals model", "away_goals": "Away goals model",
               "corners": "Corners model"}


def _k(metric: str, value: str, sub: str = "") -> tuple:
    """KPI tuple with an (i) tooltip explaining the metric."""
    return (t(metric), value, sub, "", t(METRIC_HELP[metric]) if metric in METRIC_HELP else "")


def _versions() -> tuple[str | None, str | None]:
    versions = data.model_versions()
    active = data.active_version()
    backtests = sorted(v for v, m in versions.items() if m.get("role") == "backtest")
    return active, (backtests[-1] if backtests else None)


def _model_cards(prod: dict | None, bt: dict | None) -> None:
    cols = st.columns(2)
    if prod:
        cols[0].html(f"<div class='kpi'><div class='label'>{t('Active production model')}</div>"
                     f"<div class='value'>{t('Trained through')} {esc(', '.join(prod['periods']['test']))}</div>"
                     f"<div class='sub'>{t('Used for the current season')} {esc(prod['periods']['current'])}</div></div>")
    if bt:
        cols[1].html(f"<div class='kpi'><div class='label'>{t('Evaluation model')}</div>"
                     f"<div class='value'>{t('Test season')} {esc(', '.join(bt['periods']['test']))}</div>"
                     f"<div class='sub'>{t('Trained on')} {esc(', '.join(bt['periods']['train'] + bt['periods']['validation']))}"
                     f"</div></div>")


def _periods(meta: dict) -> None:
    p = meta["periods"]
    rows = []
    for name, task in meta["tasks"].items():
        rows.append({t("Model"): t(TASK_TITLES.get(name, task["title"])), t("Target"): task["target"],
                     t("Training"): ", ".join(p["train"]), t("Validation"): ", ".join(p["validation"]),
                     t("Test"): ", ".join(p["test"]), t("Features"): task["n_features"]})
    st.html(html_table(pd.DataFrame(rows), align_right=(t("Features"),)))


def _class_metrics(m: dict) -> None:
    kpi_row([_k("Accuracy", pct(m["accuracy"], 1), ""), _k("Macro F1", num(m["macro_f1"], 3), ""),
             _k("Balanced accuracy", pct(m["balanced_accuracy"], 1), ""),
             _k("Log loss", num(m["log_loss"], 4), ""), _k("Brier score", num(m["brier"], 4), "")])


def _draw_section(task: dict) -> None:
    section(t("Draw predictions"))
    test, val = task["test_metrics"], task.get("validation_metrics") or {}
    rows = []
    for cls, name in (("H", "Home"), ("D", "Draw"), ("A", "Away")):
        rows.append({t("Outcome"): t(name), t("Actual share"): test["actual_share"][cls],
                     t("Predicted share"): test["predicted_share"][cls], t("Recall"): test.get("recall", {}).get(cls),
                     t("Precision"): test.get("precision", {}).get(cls),
                     t("Average probability"): test.get("mean_probability", {}).get(cls)})
    fmt = {k: (lambda v: pct(v, 1)) for k in (t("Actual share"), t("Predicted share"), t("Recall"), t("Precision"),
                                              t("Average probability"))}
    st.html(html_table(pd.DataFrame(rows), formats=fmt, align_right=tuple(fmt)))
    multiplier = task.get("extra", {}).get("draw_multiplier")
    st.html(f'<div class="note">{t("A draw is rarely the single most likely outcome, so a plain arg-max almost never "
            "predicts one. FootPredict keeps the probabilities and names an outcome with argmax(P(H), m·P(D), P(A)); "
            "the multiplier m is chosen on the validation season only.")}'
            + (f" m = {multiplier:.2f}." if multiplier else "") + "</div>")
    compare = []
    for label, metrics in ((t("Validation · arg-max"), val.get("argmax")),
                           (t("Validation · draw-aware rule"), val.get("draw_rule")),
                           (t("Test · arg-max"), task.get("test_metrics_argmax")),
                           (t("Test · draw-aware rule"), test)):
        if not metrics:
            continue
        compare.append({t("Evaluation"): label, "Accuracy": metrics["accuracy"], "Macro F1": metrics["macro_f1"],
                        t("Balanced accuracy"): metrics["balanced_accuracy"], "Log loss": metrics["log_loss"],
                        t("Draw recall"): metrics.get("recall", {}).get("D"),
                        t("Draw precision"): metrics.get("precision", {}).get("D"),
                        t("Predicted draw share"): metrics["predicted_share"]["D"],
                        t("Actual draw share"): metrics["actual_share"]["D"]})
    if compare:
        f2 = {"Accuracy": lambda v: pct(v, 1), "Macro F1": lambda v: num(v, 3),
              t("Balanced accuracy"): lambda v: pct(v, 1), "Log loss": lambda v: num(v, 4),
              t("Draw recall"): lambda v: pct(v, 1), t("Draw precision"): lambda v: pct(v, 1),
              t("Predicted draw share"): lambda v: pct(v, 1), t("Actual draw share"): lambda v: pct(v, 1)}
        st.html(html_table(pd.DataFrame(compare), formats=f2, align_right=tuple(f2)))


def _classifier(task: dict) -> None:
    m = task["test_metrics"]
    _class_metrics(m)
    _draw_section(task)
    bench = task.get("benchmarks", {})
    rows = [{t("Forecast"): t("FootPredict result model"), "Log loss": m["log_loss"], "Brier": m["brier"],
             "Accuracy": m["accuracy"]}]
    if bench.get("prior"):
        b = bench["prior"]
        rows.append({t("Forecast"): t("Naive baseline (always the historical outcome frequencies)"),
                     "Log loss": b["log_loss"], "Brier": b["brier"], "Accuracy": b["accuracy"]})
    if bench.get("bookmaker"):
        b = bench["bookmaker"]
        rows.append({t("Forecast"): t("Bookmaker probabilities (reference)"), "Log loss": b["log_loss"],
                     "Brier": b["brier"], "Accuracy": b["accuracy"]})
    c = colors()
    c1, c2 = st.columns([1, 1.1], gap="medium")
    with c1:
        section(t("Confusion matrix"), t("test season, {n} matches", n=f"{m['n']:,}"))
        cm = m["confusion_matrix"]
        labels = [t("Home"), t("Draw"), t("Away")]
        z = [[cm[a][p] for a in range(3)] for p in range(3)]
        fig = go.Figure(go.Heatmap(z=z, x=[f"{t('Actual')} {x}" for x in labels],
                                   y=[f"{t('Predicted')} {x}" for x in labels],
                                   colorscale=[[0, c["surface"]], [1, c["accent"]]], showscale=False,
                                   text=z, texttemplate="%{text}", textfont=dict(size=15)))
        fig.update_yaxes(autorange="reversed")
        plotly(fig, height=320)
    with c2:
        section(t("Model vs naive baseline"))
        f3 = {"Log loss": lambda v: num(v, 4), "Brier": lambda v: num(v, 4), "Accuracy": lambda v: pct(v, 1)}
        st.html(html_table(pd.DataFrame(rows), formats=f3, align_right=tuple(f3)))
        st.html(f'<div class="note">{t("The naive baseline does not analyse the specific match. "
                "The bookmaker row is only a reference — odds are not model inputs.")}</div>')
    if task.get("calibration"):
        section(t("Calibration"), t("predicted probability vs observed frequency"))
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=c["faint"]), name=t("Perfect"))
        for cls, color, name in (("H", c["home"], "Home"), ("D", c["muted"], "Draw"), ("A", c["away"], "Away")):
            cal = pd.DataFrame(task["calibration"][cls])
            fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name=t(name),
                            line=dict(color=color))
        fig.update_layout(xaxis_title=t("Predicted probability"), yaxis_title=t("Observed frequency"),
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        plotly(fig, height=330)


def _regressor(task: dict) -> None:
    m = task["test_metrics"]
    base = task.get("benchmarks", {}).get("mean", {})
    kpi_row([_k("MAE", num(m["mae"], 3), t("baseline {v}", v=num(base.get("mae"), 3))),
             _k("RMSE", num(m["rmse"], 3), t("baseline {v}", v=num(base.get("rmse"), 3))),
             _k("Poisson deviance", num(m["poisson_deviance"], 4),
              t("baseline {v}", v=num(base.get("poisson_deviance"), 4))),
             _k("R²", num(m["r2"], 3), ""),
             (t("Mean predicted / actual"), f"{num(m['mean_predicted'])} / {num(m['mean_actual'])}", "")])
    st.html(f'<div class="note">{t("Baseline = always predicting the training average. Football is very random, so "
            "even good models explain only a small part of the variance (low R²); what matters is that the expected "
            "values and resulting probabilities are well calibrated.")}</div>')


def _features(name: str, task: dict) -> None:
    groups = sorted({t(GROUP_NAMES.get(g, g)) for g in MODEL_GROUPS.get(name, [])})
    st.html("<div class='card'>" + " · ".join(esc(g) for g in groups)
            + f" <span class='muted small'>({t('{n} features', n=task['n_features'])})</span></div>")
    lang = current_language()
    with st.expander(t("Show technical features")):
        st.write(", ".join(task["features"]))
    n = st.segmented_control(t("Features"), [10, 20, 30], default=20, key=f"imp_{name}",
                             format_func=lambda x: t("Top {n}", n=x), label_visibility="collapsed") or 20
    c = colors()
    c1, c2 = st.columns(2, gap="medium")
    shap_rows = pd.DataFrame(task.get("shap_global", [])).head(n)
    gain_rows = pd.DataFrame(task.get("gain_importance", [])).head(n)
    with c1:
        if not shap_rows.empty:
            d = shap_rows.iloc[::-1]
            fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=[feature_label(f, lang) for f in d["feature"]],
                                   orientation="h", marker_color=c["accent"]))
            fig.update_layout(title=t("Global SHAP (mean |SHAP|, test season)"), yaxis=dict(automargin=True))
            plotly(fig, height=max(320, 24 * len(d) + 60))
    with c2:
        if not gain_rows.empty:
            d = gain_rows.iloc[::-1]
            fig = go.Figure(go.Bar(x=d["gain_share"], y=[feature_label(f, lang) for f in d["feature"]],
                                   orientation="h", marker_color=c["home"]))
            fig.update_layout(title=t("Feature importance (XGBoost gain)"), xaxis_tickformat=".0%",
                              yaxis=dict(automargin=True))
            plotly(fig, height=max(320, 24 * len(d) + 60))
    st.html(f'<div class="note">{t("Global importance shows what matters on average across many matches; the "
            "explanation on a match page describes one particular prediction.")}</div>')


def _advanced(name: str, task: dict, version: str) -> None:
    with st.expander(t("Advanced: hyperparameter search"), icon=":material/tune:"):
        trials = pd.DataFrame(task.get("trials", []))
        if not trials.empty:
            trials["trial"] = range(1, len(trials) + 1)
            trials["best"] = trials["score"].cummin()
            c = colors()
            fig = go.Figure()
            fig.add_scatter(x=trials["trial"], y=trials["score"], mode="markers", name=t("Trial"),
                            marker=dict(color=c["faint"]))
            fig.add_scatter(x=trials["trial"], y=trials["best"], mode="lines", name=t("Best so far"),
                            line=dict(color=c["accent"]))
            fig.update_layout(title=f"Hyperopt · {task['hyperopt_metric']} ({t('validation')})")
            plotly(fig, height=260)
        st.json({"model_id": version, "objective": task["objective"], "params": task["params"],
                 "trees": task["n_estimators"], "extra": task.get("extra", {})}, expanded=False)


def _derived(meta: dict) -> None:
    rows = []
    for name, v in meta.get("derived_markets", {}).items():
        if "hit_rate" in v:
            rows.append({t("Market"): name, "Accuracy": v["hit_rate"], "Brier": None, t("Base rate"): None})
            continue
        r = v.get("reconciled", {})
        rows.append({t("Market"): name, "Accuracy": r.get("accuracy"), "Brier": r.get("brier"),
                     t("Base rate"): r.get("base_rate")})
    f = {"Accuracy": lambda v: pct(v, 1), "Brier": lambda v: num(v, 4), t("Base rate"): lambda v: pct(v, 1)}
    st.html(html_table(pd.DataFrame(rows), formats=f, align_right=tuple(f)))
    st.html(f'<div class="note">{t("Totals, BTTS and exact scores come from one score matrix built from the two goal "
            "models (Poisson) and aligned with the 1X2 probabilities, so all markets agree with each other.")}</div>')


@guard
def render() -> None:
    page_header(t("Model Analysis"))
    active, backtest = _versions()
    meta = data.metadata(active) if active else None
    if not meta:
        empty_state(t("No trained models available."))
        return
    bt_meta = data.metadata(backtest) if backtest else None
    _model_cards(meta, bt_meta)
    st.html(f'<div class="note">{t("All test-season metrics come from the evaluation model, which never saw the "
            "test season; the production model is then refitted on all completed seasons with the same settings.")}'
            "</div>")
    section(t("Models and data periods"))
    _periods(meta)
    with st.expander(t("What do these metrics mean?"), icon=":material/help:"):
        st.html("<div class='card'>" + "".join(f"<div class='mx'><div class='ic'>•</div><div><b>{esc(t(k))}</b>"
                                               f"<span>{esc(t(v))}</span></div></div>"
                                               for k, v in METRIC_HELP.items()) + "</div>")
    tasks = meta["tasks"]
    tabs = st.tabs([t(TASK_TITLES.get(k, k)) for k in tasks] + [t("Derived markets")])
    for tab, (name, task) in zip(tabs, tasks.items()):
        with tab:
            section(t("Test-season metrics"))
            if task["kind"] == "classifier":
                _classifier(task)
            else:
                _regressor(task)
            section(t("Features and importance"))
            _features(name, task)
            _advanced(name, task, active)
    with tabs[-1]:
        section(t("Markets derived from the goal and corner models"), t("test season"))
        _derived(meta)
    st.caption(t("Model settings: {a} Hyperopt trials for the result model, {b} for each regression model.",
                 a=config.HYPEROPT_EVALS["result"], b=config.HYPEROPT_EVALS["home_goals"]))


render()
