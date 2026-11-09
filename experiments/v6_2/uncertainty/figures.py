"""Charts of the uncertainty study (figures/*.png), drawn from results/*.json only."""
from __future__ import annotations

import numpy as np

from scripts import report_kit as K

from . import common as C
from .run import MEASURES

SEASONS = [str(s) for s in C.SEASONS]
COL = dict(zip(SEASONS, K.PALETTE))


def spread_vs_gap(plt, iv: dict, ms: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(SEASONS))
    w = 0.16
    series = [("медиана σ сидов", [ms[s]["u_seed"]["median"] for s in SEASONS]),
              ("медиана σ бутстрепа", [ms[s]["u_boot"]["median"] for s in SEASONS]),
              ("медиана σ подвыборок", [ms[s]["u_sub"]["median"] for s in SEASONS]),
              ("медиана ширины Венна–Аберса", [iv[s]["va_median_width"] for s in SEASONS]),
              ("|gap| выбранного Main", [abs(iv[s]["gap"]) for s in SEASONS])]
    for i, (name, vals) in enumerate(series):
        ax.bar(x + (i - 2) * w, np.array(vals) * 100, w, label=name, color=K.PALETTE[i % len(K.PALETTE)])
    ax.set_xticks(x, [C.period_label(int(s)) for s in SEASONS])
    ax.set_ylabel("п.п. вероятности")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    ax.set_title("Разброс оценки вероятности Main и систематическая переоценка", fontsize=9.5)
    K.save_chart(fig, C.FIGURES / "spread_vs_gap.png")


def quintiles(plt, pred: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, col in zip(axes, ("u_boot", "u_1x2", "u_va")):
        for s in SEASONS:
            q = pred["quintiles"][s].get(col) or []
            if not q:
                continue
            xs = [r["q"] + (SEASONS.index(s) - 1.5) * 0.08 for r in q]
            g = np.array([r["gap"] for r in q]) * 100
            lo = g - np.array([r["gap_ci95"][0] for r in q]) * 100
            hi = np.array([r["gap_ci95"][1] for r in q]) * 100 - g
            ax.errorbar(xs, g, yerr=[lo, hi], marker="o", ms=3.5, lw=1.1, capsize=2, color=COL[s],
                        label=C.period_label(int(s)))
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(1, 6), ["1 (мин.)", "2", "3", "4", "5 (макс.)"])
        ax.set_xlabel("квинтиль меры")
        ax.set_title(MEASURES[col], fontsize=9)
    axes[0].set_ylabel("gap = p − доля выигрышей, п.п.")
    axes[0].legend(fontsize=7.5, frameon=False)
    K.save_chart(fig, C.FIGURES / "gap_by_quintile.png")


def betas(plt, pred: dict) -> None:
    pool = pred["pooled_select"]
    names = [c for c in MEASURES if pool.get(c, {}).get("beta_u") is not None]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = np.arange(len(names))[::-1]
    for yi, c in zip(y, names):
        r = pool[c]
        ax.errorbar(r["beta_u"], yi, xerr=[[r["beta_u"] - r["beta_u_ci95"][0]], [r["beta_u_ci95"][1] - r["beta_u"]]],
                    fmt="o", color=K.PALETTE[0] if (r.get("p_holm") or 1) >= 0.05 else K.NEG, capsize=3)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y, [MEASURES[c] for c in names])
    ax.set_xlabel("β при z(u) в логит-модели «выиграл» (при фиксированном logit p); < 0 — чаще проигрывает")
    ax.set_title("Предсказывает ли неопределённость проигрыш сверх p? (2023/24 + 2024/25)", fontsize=9.5)
    K.save_chart(fig, C.FIGURES / "beta_u.png")


def filters(plt, filt: dict) -> None:
    keys = list(filt["variants"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, max(3.2, 0.2 * len(keys) + 1)), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, metric, title, scale in ((axes[0], "pooled_select_roi", "ΔROI (оставленные − все), п.п.", 100),
                                     (axes[1], "pooled_select_abs_gap", "Δ|gap| (оставленные − все), п.п.", 100)):
        for yi, k in zip(y, keys):
            r = filt["variants"][k][metric]
            ax.errorbar(r["diff"] * scale, yi, xerr=[[(r["diff"] - r["ci95"][0]) * scale],
                                                     [(r["ci95"][1] - r["diff"]) * scale]],
                        fmt="o", ms=3, capsize=2, color=K.PALETTE[2])
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(title, fontsize=9)
    labels = [f"{MEASURES[filt['variants'][k]['measure']]}, −{int(filt['variants'][k]['drop'] * 100)} %"
              for k in keys]
    axes[0].set_yticks(y, labels, fontsize=7)
    K.save_chart(fig, C.FIGURES / "filters.png")


def calibration(plt, cb: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for ax, s in zip(axes, SEASONS):
        rows = cb[s]
        if not rows:
            continue
        p = np.array([r["predicted"] for r in rows])
        o = np.array([r["observed"] for r in rows])
        lo = np.array([r["ci95"][0] for r in rows])
        hi = np.array([r["ci95"][1] for r in rows])
        ax.fill_between(p, lo, hi, color=COL[s], alpha=0.25, lw=0)
        ax.plot(p, o, "o-", color=COL[s], ms=3)
        ax.plot([0.35, 0.9], [0.35, 0.9], color="#888", lw=0.8, ls="--")
        ax.set_title(C.period_label(int(s)), fontsize=9)
        ax.set_xlabel("заявленная p Main")
    axes[0].set_ylabel("доля выигрышей [95 % ДИ]")
    K.save_chart(fig, C.FIGURES / "calibration_band.png")


def run() -> None:
    plt = C.plt()
    ms = C.load_json("measures.json")
    pred = C.load_json("predictive.json")
    filt = C.load_json("filters.json")
    iv = C.load_json("intervals.json")
    cb = C.load_json("calibration_band.json")
    spread_vs_gap(plt, iv, ms)
    quintiles(plt, pred)
    betas(plt, pred)
    filters(plt, filt)
    calibration(plt, cb)


if __name__ == "__main__":
    run()
