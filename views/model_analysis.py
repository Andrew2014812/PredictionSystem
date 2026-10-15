"""Model Analysis (technical / diploma page): periods, metrics, draws, calibration, SHAP."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.features.catalog import catalogue
from src.features.names import feature_label
from src.ui import data
from src.ui.components import (empty_state, esc, guard, html_table, kpi_row, num, page_header, pct, plotly, signed,
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
    "elo": "Elo strength rating", "opp_adjusted": "Opponent-adjusted form",
    "opp_adjusted_corners": "Opponent-adjusted corners",
}
EVALUATION_SOURCES = {"backtest": "Test season (evaluation model)", "backfill": "Current season, predicted afterwards"}
TASK_TITLES = {"result": "Result model (1X2)", "home_goals": "Home goals model", "away_goals": "Away goals model",
               "corners": "Corners model"}


def _k(metric: str, value: str, sub: str = "") -> tuple:
    """KPI tuple with an (i) tooltip explaining the metric (which also says which way is better)."""
    return (t(metric), value, sub, "", t(METRIC_HELP[metric]) if metric in METRIC_HELP else "")


def _versions() -> tuple[str | None, str | None]:
    versions = data.model_versions()
    active = data.active_version()
    backtests = sorted(v for v, m in versions.items() if m.get("role") == "backtest")
    return active, (backtests[-1] if backtests else None)


def _model_note(prod: dict | None, bt: dict | None) -> None:
    """One short line about the active model instead of two technical cards."""
    if not prod:
        return
    text = t("Active model: trained on {seasons}, used for {current}.",
             seasons=f"{prod['periods']['train'][0]}–{prod['periods']['test'][-1]}", current=prod["periods"]["current"])
    if bt:
        text += " " + t("Test-season metrics below come from a separate evaluation model that never saw {test}.",
                        test=", ".join(bt["periods"]["test"]))
    st.html(f'<div class="note">{esc(text)}</div>')


def _periods(meta: dict) -> None:
    p = meta["periods"]
    rows = []
    for name, task in meta["tasks"].items():
        rows.append({t("Model"): t(TASK_TITLES.get(name, task["title"])), t("Training"): ", ".join(p["train"]), t("Validation"): ", ".join(p["validation"]),
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
    multiplier = task.get("extra", {}).get("draw_multiplier") or 1.0
    if multiplier > 1.0:
        st.html(f'<div class="note">{t("A draw is rarely the single most likely outcome, so a plain arg-max almost "
                "never predicts one. FootPredict keeps the probabilities and names an outcome with "
                "argmax(P(H), m·P(D), P(A)); the multiplier m is chosen on the validation season only.")}'
                f" m = {multiplier:.2f}.</div>")
    else:
        st.html(f'<div class="note">{t("FootPredict names the most probable outcome (arg-max). A draw-aware rule "
                "that boosts draws was tested on the validation season and not adopted: it predicts more draws but "
                "lowers accuracy, and the return at real odds stays practically the same. Draw probabilities "
                "themselves are unchanged and are used by every market.")}</div>')
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
    """Model Features: one importance chart (SHAP), gain in an expander, then the feature dictionary."""
    lang = current_language()
    n = st.segmented_control(t("Features"), [10, 20, 30], default=20, key=f"imp_{name}",
                             format_func=lambda x: t("Top {n}", n=x), label_visibility="collapsed") or 20
    c = colors()
    shap_all = pd.DataFrame(task.get("shap_global", []))
    if not shap_all.empty:
        shap_all["share"] = shap_all["mean_abs_shap"] / shap_all["mean_abs_shap"].sum()
        d = shap_all.head(n).iloc[::-1]
        fig = go.Figure(go.Bar(x=d["share"], y=[feature_label(f, lang) for f in d["feature"]],
                               orientation="h", marker_color=c["accent"]))
        fig.update_layout(title=t("Share of influence on the predictions (SHAP, test season)"), xaxis_tickformat=".0%",
                          yaxis=dict(automargin=True))
        plotly(fig, height=max(320, 24 * len(d) + 60))
    st.html(f'<div class="note">{t("Each bar is a feature\'s share of the model\'s total influence on its test-season "
            "predictions (all features together = 100%). It shows what matters on average; a match page explains "
            "one particular prediction.")}</div>')
    gain_rows = pd.DataFrame(task.get("gain_importance", [])).head(n)
    if not gain_rows.empty:
        with st.expander(t("XGBoost gain (technical)"), icon=":material/bar_chart:"):
            st.caption(t("Gain measures how much a feature improved the training objective when the trees were "
                         "built; SHAP measures how much it moves actual predictions. They usually agree on the main "
                         "features but can differ for correlated ones."))
            d = gain_rows.iloc[::-1]
            fig = go.Figure(go.Bar(x=d["gain_share"], y=[feature_label(f, lang) for f in d["feature"]],
                                   orientation="h", marker_color=c["home"]))
            fig.update_layout(title=t("Feature importance (XGBoost gain)"), xaxis_tickformat=".0%",
                              yaxis=dict(automargin=True))
            plotly(fig, height=max(320, 24 * len(d) + 60))
    st.html(f"<div class='why-h'>{t('Feature dictionary')}</div>")
    _dictionary(name, task, lang)


def _dictionary(name: str, task: dict, lang: str) -> None:
    query = st.text_input(t("Search features"), key=f"dict_{name}", placeholder=t("e.g. corners, Elo, form"))
    entries = catalogue({name: task["features"]}, lang)
    q = (query or "").strip().lower()
    if q:
        entries = [e for e in entries if q in e["name"].lower() or q in e["description"].lower()
                   or any(q in c.lower() for c in e["columns"])]
    if not entries:
        st.caption(t("No features match the search."))
        return
    st.html("<div class='card'>" + "".join(
        f"<div class='mx'><div class='ic'>•</div><div><b>{esc(e['name'])}</b> "
        f"<span class='muted small'>({t('{n} features', n=len(e['columns']))})</span>"
        f"<span>{esc(e['description'])}</span></div></div>" for e in entries) + "</div>")
    with st.popover(t("Technical column names")):
        for e in entries:
            st.caption(f"{e['name']}: " + ", ".join(e["columns"]))


def _advanced(name: str, task: dict, version: str) -> None:
    with st.expander(t("Advanced: hyperparameter search"), icon=":material/tune:"):
        st.caption(t("Hyperopt tries different model settings (tree depth, learning rate, regularisation …). Each "
                     "dot is one try, scored on the validation season — lower is better. The line is the best score "
                     "found so far. A line that flattens after the first tries means good settings were found early; "
                     "that is a normal result, not a problem."))
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


def _evaluation() -> None:
    """How Main / Risk would have done on past matches (test season, current season afterwards).
    Kept here, separate from the Prediction History of predictions recorded before kick-off."""
    from src.analytics.performance import select_dataset, summarize
    picks = select_dataset(data.picks(), "evaluation")
    if picks.empty:
        st.caption(t("No historical evaluation available."))
        return
    st.html(f'<div class="note">{t("Past matches predicted afterwards from pre-match data by a model trained only on "
            "earlier seasons. This checks the model; it is not a record of predictions published before the "
            "matches, so it is never mixed into Prediction History.")}</div>')
    rows = []
    for source, label in EVALUATION_SOURCES.items():
        for role in ("main", "risk"):
            s_ = summarize(picks.loc[(picks["source"] == source) & (picks["role"] == role)])
            if s_["settled"]:
                rows.append({t("Data"): t(label), t("Type"): t("Main") if role == "main" else t("Risk"),
                             t("Predictions"): s_["settled"], t("Hit rate"): s_["hit_rate"],
                             t("With real odds"): s_["bets"], "ROI": s_["roi"], t("Average odds"): s_["avg_odds"]})
    f = {t("Hit rate"): lambda v: pct(v, 1), "ROI": lambda v: signed(v, 1, "%"), t("Average odds"): lambda v: num(v)}
    st.html(html_table(pd.DataFrame(rows), formats=f,
                       align_right=(t("Predictions"), t("Hit rate"), t("With real odds"), "ROI", t("Average odds"))))
    with st.expander(t("How is the Main prediction chosen?"), icon=":material/help:"):
        st.html(f'<div class="card">{t("Among the selections with a probability of about 40–85 %, FootPredict ranks "
                "each one by its probability, how reliable the market is for the models, a soft preference for odds "
                "around 1.40–2.50 and its expected value at a real bookmaker price. The Risk prediction is the best "
                "higher-odds alternative (about 3.0 and up) that is not the same bet as Main. The weights were chosen "
                "on the validation season only.")}</div>')


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
    _model_note(meta, bt_meta)
    section(t("Models and data periods"))
    _periods(meta)
    with st.expander(t("What do these metrics mean?"), icon=":material/help:"):
        st.html("<div class='card'>" + "".join(
            f"<div class='mx'><div class='ic'>•</div><div><b>{esc(t(k))}</b>"
            f"<span>{esc(t(v))}</span></div></div>" for k, v in METRIC_HELP.items()) + "</div>")
    tasks = meta["tasks"]
    tabs = st.tabs([t(TASK_TITLES.get(k, k)) for k in tasks] + [t("Derived markets"), t("Historical evaluation")])
    for tab, (name, task) in zip(tabs, tasks.items()):
        with tab:
            section(t("Test-season metrics"))
            if task["kind"] == "classifier":
                _classifier(task)
            else:
                _regressor(task)
            _advanced(name, task, active)
            section(t("Model Features"))
            _features(name, task)
    with tabs[-2]:
        section(t("Markets derived from the goal and corner models"), t("test season"))
        _derived(meta)
    with tabs[-1]:
        _evaluation()
    st.caption(t("Model settings: {a} Hyperopt trials for the result model, {b} for each regression model.",
                 a=config.HYPEROPT_EVALS["result"], b=config.HYPEROPT_EVALS["home_goals"]))


render()
