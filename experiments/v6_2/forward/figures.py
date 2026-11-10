"""Charts of the forward-validation study (figures/*.png), from results/*.json only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import report_kit as K

from . import common as C

SHORT = {"maincal_ps_platt_monthly@v1": "Main: PS-Platt по месяцам", "maincal_ps_group@v1": "Main: PS-Group",
         "risk_stacked@v1": "Risk: стек с рынком", "risk_market_probability@v1": "Risk: вероятность рынка",
         "bookmaker_h2_pool@v1": "Пул H2 (тотал 2.5)", "xg_x5@v1": "xG X5"}


def architecture(plt) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#4b5a63", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, wrap=True)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color="#4b5a63", lw=1.0))

    box(0.1, 4.4, 2.2, 1.2, "Ежедневный прогон\nrun_pipeline.py\n(данные, расчёт, прогнозы)", "#e3ebe7")
    box(3.0, 4.4, 2.6, 1.2, "Таблицы продакшна\nlocal_data/predictions/\nsnapshots, markets, picks", "#e3ebe7")
    box(6.4, 4.4, 3.4, 1.2, "Сайт (Streamlit)\nчитает только таблицы продакшна", "#e3ebe7")
    box(3.0, 2.3, 2.6, 1.3, "record_shadow_predictions()\nsrc/research/shadow.py\n(только чтение продакшна)", "#f6ecd6")
    box(0.1, 2.3, 2.2, 1.3, "Реестр теней\nregistry.py + замороженные\npreregistration.json (SHA-256)", "#f6ecd6")
    box(6.4, 2.3, 3.4, 1.3, "local_data/research/\nshadow_predictions.parquet (append-only,\nцепочка хэшей), shadow_results.parquet,\nmanifest.json", "#f6ecd6")
    box(6.4, 0.3, 3.4, 1.3, "evaluate_shadow()\nпарное сравнение с продакшном;\nнет вывода до N; решение замораживается", "#f6ecd6")
    box(3.0, 0.3, 2.6, 1.3, "shadow_report.json\n(для отчёта / диплома,\nне для сайта)", "#f6ecd6")
    arrow(2.3, 5.0, 3.0, 5.0)
    arrow(5.6, 5.0, 6.4, 5.0)
    arrow(4.3, 4.4, 4.3, 3.6)
    arrow(2.3, 2.95, 3.0, 2.95)
    arrow(5.6, 2.95, 6.4, 2.95)
    arrow(8.1, 2.3, 8.1, 1.6)
    arrow(6.4, 0.95, 5.6, 0.95)
    ax.text(5.0, 3.85, "после прогнозов продакшна", fontsize=7, color="#4b5a63", ha="center")
    ax.plot([0.05, 9.95], [4.1, 4.1], color="#c0392b", lw=0.8, ls="--")
    ax.text(9.9, 4.15, "граница: тени ничего не пишут выше", fontsize=7, color="#c0392b", ha="right", va="bottom")
    K.save_chart(fig, C.FIGURES / "architecture.png")


def replay_costs(plt, sim: dict) -> None:
    runs = pd.DataFrame(sim["runs"])
    sizes = pd.DataFrame(sim["storage"]["size_by_run"])
    t = pd.to_datetime(runs["run_uk"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    axes[0].bar(t, runs["seconds"], width=0.6, color=K.PALETTE[0], label="первый вызов")
    axes[0].plot(t, runs["rerun_seconds"], "o", ms=3, color=K.PALETTE[1], label="повтор (0 новых строк)")
    axes[0].set_ylabel("секунд на прогон")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].set_title("Время записи теней в реплее", fontsize=9)
    axes[1].plot(pd.to_datetime(sizes["run_uk"]), sizes["bytes"] / 1024, "o-", ms=3, color=K.PALETTE[2])
    ax2 = axes[1].twinx()
    ax2.plot(pd.to_datetime(sizes["run_uk"]), sizes["rows_total"], "s--", ms=2.5, color=K.PALETTE[3])
    ax2.set_ylabel("строк в журнале", color=K.PALETTE[3])
    axes[1].set_ylabel("КБ shadow_predictions.parquet", color=K.PALETTE[2])
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1].set_title("Рост журнала", fontsize=9)
    K.save_chart(fig, C.FIGURES / "replay_costs.png")


def information(plt, sim: dict, eta: dict) -> None:
    ev = sim["evaluations"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    cut = pd.to_datetime([e["cutoff"] for e in ev])
    for i, key in enumerate(SHORT):
        n = [e.get(key, {}).get("n") or 0 for e in ev]
        min_n = eta["shadows"].get(key, {}).get("min_n") or 1
        axes[0].plot(cut, np.array(n) / min_n, "o-", ms=3, color=K.PALETTE[i % len(K.PALETTE)], label=SHORT[key])
    axes[0].axhline(1, color=K.NEG, lw=0.9, ls="--")
    axes[0].set_ylabel("n / N (доля информации)")
    axes[0].set_title("Реплей: накопление до предрегистрированного N", fontsize=9)
    axes[0].legend(fontsize=6.8, frameon=False)
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    keys = [k for k in SHORT if eta["shadows"].get(k, {}).get("weeks_to_n")]
    weeks = [eta["shadows"][k]["weeks_to_n"] for k in keys]
    y = np.arange(len(keys))[::-1]
    axes[1].barh(y, weeks, color=[K.PALETTE[list(SHORT).index(k) % len(K.PALETTE)] for k in keys])
    axes[1].set_yticks(y, [SHORT[k] for k in keys], fontsize=7.5)
    axes[1].set_xlabel("недель до N после 10.11.2026 (оценка)")
    axes[1].set_title("Когда возможен первый вывод", fontsize=9)
    K.save_chart(fig, C.FIGURES / "information.png")


def run() -> None:
    plt = C.plt() if hasattr(C, "plt") else None
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F811
    sim = C.load_json("simulation.json")
    eta = C.load_json("eta.json")
    architecture(plt)
    replay_costs(plt, sim)
    information(plt, sim, eta)


if __name__ == "__main__":
    run()
