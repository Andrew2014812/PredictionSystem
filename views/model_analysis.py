"""Model Analysis: periods, metrics, confusion matrix, calibration, SHAP."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.ui import data
from src.ui.components import empty_state, esc, guard, kpi_row, num, page_header, pct, plotly, section
from src.ui.theme import COLORS

METRIC_HELP = [
    ("🎯", "Accuracy", "Share of matches where the most likely outcome was the actual one."),
    ("⚖️", "Macro F1", "Average quality over the three outcomes (home, draw, away) — draws count as much as home wins."),
    ("📐", "Balanced accuracy", "Average recall per outcome: how often each actual outcome was recognised."),
    ("📉", "Log loss", "Quality of the probabilities: punishes confident wrong forecasts. Lower is better; "
                      "it is the metric Hyperopt optimises for the result model."),
    ("🎲", "Brier score", "Mean squared error of the probabilities. 0 is perfect; lower is better."),
    ("🧮", "Confusion matrix", "Where the model makes mistakes: rows = predicted outcome, columns = actual outcome."),
    ("📏", "RMSE / MAE", "Typical size of the error of a goals or corners forecast (in goals / corners)."),
    ("📊", "Poisson deviance", "Proper error measure for count forecasts; the metric Hyperopt optimises for "
                               "the goals and corners models. Lower is better."),
    ("🔍", "Feature importance", "Which variables matter most on average over many matches (global view). "
                                 "The SHAP chart on a match page explains one specific prediction."),
]


def _periods(meta: dict) -> None:
    p = meta["periods"]
    tasks = meta["tasks"]
    rows = []
    for name, t in tasks.items():
        tr = t["trained_on"]
        rows.append({
            "Model": t["title"], "Algorithm": "XGBoost " + ("classifier" if t["kind"] == "classifier" else "regressor"),
            "Target": t["target"], "Objective": t["objective"],
            "Train": ", ".join(p["train"]), "Validation (Hyperopt)": ", ".join(p["validation"]),
            "Test": ", ".join(p["test"]), "Features": t["n_features"],
            "Rows (final fit)": tr["production"]["rows"] if meta.get("role") == "production" else tr["backtest"]["rows"],
            "Trees": t["n_estimators"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _classifier(t: dict, meta: dict) -> None:
    m = t["test_metrics"]
    kpi_row([("Accuracy", pct(m["accuracy"], 1), ""), ("Macro F1", num(m["macro_f1"], 3), ""),
             ("Balanced accuracy", pct(m["balanced_accuracy"], 1), ""),
             ("Log loss", num(m["log_loss"], 4), "lower is better"), ("Brier score", num(m["brier"], 4), "lower is better")])
    bench = t.get("benchmarks", {})
    rows = [{"Forecast": "XGBoost result model", "Log loss": m["log_loss"], "Brier": m["brier"],
             "Accuracy": m["accuracy"]}]
    if bench.get("prior"):
        b = bench["prior"]
        rows.append({"Forecast": "Naive: training class frequencies", "Log loss": b["log_loss"], "Brier": b["brier"],
                     "Accuracy": b["accuracy"]})
    if bench.get("bookmaker"):
        b = bench["bookmaker"]
        rows.append({"Forecast": "Reference: bookmaker implied probabilities", "Log loss": b["log_loss"],
                     "Brier": b["brier"], "Accuracy": b["accuracy"]})
    c1, c2 = st.columns([1.05, 1], gap="medium")
    with c1:
        section("Confusion matrix", f"test season, {m['n']:,} matches")
        cm = m["confusion_matrix"]
        labels = ["Home", "Draw", "Away"]
        # sklearn: rows = actual, columns = predicted -> show predicted on rows
        z = [[cm[a][p] for a in range(3)] for p in range(3)]
        fig = go.Figure(go.Heatmap(z=z, x=[f"Actual {x}" for x in labels], y=[f"Pred {x}" for x in labels],
                                   colorscale=[[0, "#131c2e"], [1, COLORS["positive"]]], showscale=False,
                                   text=z, texttemplate="%{text}", textfont=dict(size=16)))
        fig.update_yaxes(autorange="reversed")
        plotly(fig, height=320)
    with c2:
        section("Comparison", "same test season")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Brier": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="percent")})
        st.caption("The bookmaker row is only a reference point — odds are not model inputs.")
        share = pd.DataFrame({"Predicted share": m["predicted_share"], "Actual share": m["actual_share"]})
        st.dataframe(share.style.format("{:.1%}"), width="stretch")
    if t.get("calibration"):
        section("Calibration", "predicted probability vs observed frequency")
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=COLORS["muted"]),
                        name="Perfect")
        for cls, color in (("H", COLORS["home"]), ("D", COLORS["draw"]), ("A", COLORS["away"])):
            cal = pd.DataFrame(t["calibration"][cls])
            fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name=cls,
                            line=dict(color=color))
        fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        plotly(fig, height=340)


def _regressor(t: dict) -> None:
    m = t["test_metrics"]
    base = t.get("benchmarks", {}).get("mean", {})
    kpi_row([("RMSE", num(m["rmse"], 3), f"baseline {num(base.get('rmse'), 3)}"),
             ("MAE", num(m["mae"], 3), f"baseline {num(base.get('mae'), 3)}"),
             ("R²", num(m["r2"], 3), "share of variance explained"),
             ("Poisson deviance", num(m["poisson_deviance"], 4), f"baseline {num(base.get('poisson_deviance'), 4)}"),
             ("Mean predicted / actual", f"{num(m['mean_predicted'])} / {num(m['mean_actual'])}", "")])
    st.caption("Baseline = always predicting the training average. Football scores are very random, so even "
               "good models explain only a small part of the variance (low R²) — what matters is that the "
               "expected values and the resulting probabilities are well calibrated.")
    extra = t.get("extra", {})
    if extra.get("leagues"):
        st.caption(f"Corner model available for {len(extra['leagues'])} leagues with ≥ "
                   f"{config.MIN_STAT_COVERAGE:.0%} corner coverage. Negative-binomial size: "
                   f"{num(extra.get('nb_size'), 1) if extra.get('nb_size') else 'Poisson (no over-dispersion)'}.")


def _importance(t: dict) -> None:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        n = st.segmented_control("Features", [10, 20, 30], default=20, key=f"imp_{t['target']}",
                                 format_func=lambda x: f"Top {x}") or 20
    shap_rows = pd.DataFrame(t.get("shap_global", [])).head(n)
    gain_rows = pd.DataFrame(t.get("gain_importance", [])).head(n)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if not shap_rows.empty:
            d = shap_rows.iloc[::-1]
            fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=d["label"], orientation="h", marker_color=COLORS["positive"]))
            fig.update_layout(title="Global SHAP (mean |SHAP| on the test season)", yaxis=dict(automargin=True))
            plotly(fig, height=max(320, 24 * len(d) + 60))
    with c2:
        if not gain_rows.empty:
            d = gain_rows.iloc[::-1]
            fig = go.Figure(go.Bar(x=d["gain_share"], y=d["label"], orientation="h", marker_color=COLORS["home"]))
            fig.update_layout(title="XGBoost gain importance (share)", xaxis_tickformat=".0%",
                              yaxis=dict(automargin=True))
            plotly(fig, height=max(320, 24 * len(d) + 60))
    st.caption("Global importance shows what matters on average across many matches. The SHAP chart on a "
               "match page explains one particular prediction.")


def _trials(t: dict) -> None:
    trials = pd.DataFrame(t.get("trials", []))
    if trials.empty:
        return
    trials["trial"] = range(1, len(trials) + 1)
    trials["best so far"] = trials["score"].cummin()
    fig = go.Figure()
    fig.add_scatter(x=trials["trial"], y=trials["score"], mode="markers", name="trial",
                    marker=dict(color=COLORS["muted"]))
    fig.add_scatter(x=trials["trial"], y=trials["best so far"], mode="lines", name="best so far",
                    line=dict(color=COLORS["positive"]))
    fig.update_layout(title=f"Hyperopt search ({t['hyperopt_metric']} on validation)", xaxis_title="Trial")
    plotly(fig, height=280)
    st.json({"best parameters": t["params"], "trees": t["n_estimators"]}, expanded=False)


def _derived(meta: dict) -> None:
    rows = []
    for name, v in meta.get("derived_markets", {}).items():
        if "hit_rate" in v:
            rows.append({"Market": name, "Accuracy / hit rate": v["hit_rate"], "Brier": None,
                         "Log loss": None, "Base rate": None, "Brier (plain Poisson)": None})
            continue
        r, p = v.get("reconciled", {}), v.get("poisson", {})
        rows.append({"Market": name, "Accuracy / hit rate": r.get("accuracy"), "Brier": r.get("brier"),
                     "Log loss": r.get("log_loss"), "Base rate": r.get("base_rate"),
                     "Brier (plain Poisson)": p.get("brier")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Accuracy / hit rate": st.column_config.NumberColumn(format="percent"),
        "Base rate": st.column_config.NumberColumn(format="percent"),
        "Brier": st.column_config.NumberColumn(format="%.4f"),
        "Log loss": st.column_config.NumberColumn(format="%.4f"),
        "Brier (plain Poisson)": st.column_config.NumberColumn(format="%.4f")})
    st.caption("Totals, BTTS and exact score are not separate models: they are read from the score matrix "
               "built from the two goal models (Poisson) and rescaled to the 1X2 classifier probabilities. "
               "'Plain Poisson' shows the same market without that rescaling.")


@guard
def render() -> None:
    page_header("Model Analysis", "How the four XGBoost models were trained and how well they perform")
    versions = data.model_versions()
    if not versions:
        empty_state("No trained models", "Run python scripts/retrain_models.py")
        return
    active = data.active_version()
    keys = sorted(versions, key=lambda v: (v != active, v))
    version = st.selectbox("Model version", keys, format_func=lambda v: f"{v} ({versions[v].get('role')}"
                           + (", active)" if v == active else ")"))
    meta = data.metadata(version)
    if not meta:
        empty_state("Metadata missing for this version")
        return
    st.caption(f"Created {meta['created']} · training data up to {meta['trained_to']} · "
               f"odds used as features: {'yes' if meta.get('use_odds_features') else 'no'}")

    section("Models and data periods")
    _periods(meta)
    if meta.get("role") == "production":
        st.caption("Test metrics below were measured by the backtest model (trained on train + validation with "
                   "the same hyperparameters); the production model is then refitted on all completed seasons.")

    with st.expander("What do these metrics mean?", icon=":material/help:"):
        st.html("<div class='card'>" + "".join(
            f"<div class='mx'><div class='ic'>{i}</div><div><b>{esc(n)}</b><span>{esc(d)}</span></div></div>"
            for i, n, d in METRIC_HELP) + "</div>")

    tasks = meta["tasks"]
    labels = {"result": "Result model (1X2)", "home_goals": "Home goals model",
              "away_goals": "Away goals model", "corners": "Corners model"}
    tabs = st.tabs([labels.get(k, k) for k in tasks] + ["Derived markets"])
    for tab, (name, t) in zip(tabs, tasks.items()):
        with tab:
            st.caption(t["description"])
            section("Test-season metrics")
            if t["kind"] == "classifier":
                _classifier(t, meta)
            else:
                _regressor(t)
            section("Feature importance")
            _importance(t)
            with st.expander("Hyperparameter search"):
                _trials(t)
            with st.expander(f"All {t['n_features']} features"):
                st.write(", ".join(t["features"]))
    with tabs[-1]:
        section("Markets derived from the goal and corner models", "test season")
        _derived(meta)


render()
