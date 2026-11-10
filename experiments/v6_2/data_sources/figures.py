"""Charts of the data-sources report (read from results/, written to figures/)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.report_kit import ACCENT, AMBER, MUTED, PALETTE, save_chart

from . import common as C


def lead_hist() -> None:
    m = pd.read_csv(C.RESULTS / "af_vs_fd_matches.csv")
    led = m.loc[m["category"] == "API-Football раньше", "lead_h"] / 24
    days = np.floor(led).astype(int)
    counts = days.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(counts.index, counts.values, color=ACCENT, width=0.7)
    for x, y in counts.items():
        ax.text(x, y + 2, str(y), ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("на сколько суток API-Football опередил Football-Data (целые сутки, вниз)")
    ax.set_ylabel("матчей")
    ax.set_xticks(range(0, int(days.max()) + 1))
    save_chart(fig, C.FIGURES / "af_lead_days.png")


def delay_ecdf() -> None:
    m = pd.read_csv(C.RESULTS / "af_vs_fd_matches.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for col, label, color in (("af_after_ko_h", "API-Football", ACCENT), ("fd_after_ko_h", "Football-Data", AMBER)):
        s = np.sort(m[col].dropna().to_numpy() / 24)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=color, lw=2, label=label)
        med = np.median(s)
        ax.annotate(f"{label}: медиана {med:.1f} сут", (med, 0.5), xytext=(1.6, 0.92) if color == ACCENT else (5.4, 0.3),
                    fontsize=8, color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_xlabel("сутки от начала матча до коммита, в котором появился результат")
    ax.set_ylabel("доля матчей")
    ax.set_xlim(0, 10)
    ax.legend(loc="lower right", frameon=False)
    save_chart(fig, C.FIGURES / "result_delay_ecdf.png")


def oddspapi_runs() -> None:
    r = pd.read_csv(C.RESULTS / "oddspapi_runs.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = [AMBER if a else ACCENT for a in r["archive_import"]]
    ax.bar(range(len(r)), r["billable"], color=colors, width=0.7)
    for i, v in enumerate(r["billable"]):
        ax.text(i, v + 0.4, str(v), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([s[5:10].replace("-", ".")[3:] + "." + s[5:7] for s in r["run_hour"]], rotation=90, fontsize=7)
    ax.set_ylabel("платных запросов")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="ежедневный запуск"), Patch(color=AMBER, label="загрузка архива")],
              frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, r["billable"].max() * 1.3)
    save_chart(fig, C.FIGURES / "oddspapi_billable.png")


def schedule() -> None:
    d = pd.read_csv(C.RESULTS / "daily_runs.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = [ACCENT if c == "03:17" else PALETTE[2] for c in d["cron_kyiv"]]
    ax.bar(range(len(d)), d["delay_h"], color=colors, width=0.7)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(pd.to_datetime(d["kyiv"]).dt.strftime("%d.%m"), rotation=90, fontsize=7)
    ax.set_ylabel("часов после расписания")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[2], label="расписание 11:17 Киев"),
                       Patch(color=ACCENT, label="расписание 03:17 Киев")], frameon=False, loc="upper left")
    save_chart(fig, C.FIGURES / "schedule_delay.png")


def roi_class() -> None:
    d = pd.read_csv(C.RESULTS / "bets_by_price_class.csv")
    d = d.loc[d["bets"] > 0]
    classes = [c for c in ("A1", "A2", "B", "C") if c in set(d["price_class"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.38
    for k, (role, color) in enumerate((("main", ACCENT), ("risk", AMBER))):
        g = d.loc[d["role"] == role].set_index("price_class").reindex(classes)
        xs = np.arange(len(classes)) + (k - 0.5) * w
        ax.bar(xs, g["roi"], width=w, color=color, label=role.capitalize())
        for x, v, n in zip(xs, g["roi"], g["bets"]):
            if v == v:
                ax.text(x, v + (1 if v >= 0 else -3.5), f"{v:+.1f}%\n(n={int(n)})", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("ROI, %")
    ax.set_ylim(min(-40, ax.get_ylim()[0]), max(10, ax.get_ylim()[1] + 4))
    ax.legend(frameon=False, loc="lower left")
    save_chart(fig, C.FIGURES / "roi_by_price_class.png")


def main() -> None:
    lead_hist()
    delay_ecdf()
    oddspapi_runs()
    schedule()


if __name__ == "__main__":
    main()
