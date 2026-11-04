"""Analysis 6 — market mix of the Main and sensitivity of the rule's constants (walk-forward folds).

* which markets / lines / sides the current rule chooses, hit rate and ROI by them, by odds band and league;
* one-at-a-time sensitivity of MARKET_RELIABILITY, MAIN_EV_WEIGHT, MAIN_PROBABILITY_POWER, the preferred odds
  window and its softness, in both universes. This is a description of the rule's behaviour, not a selection:
  the adoption rule (preregistration.json) applies to the strategies of strategies_eval.py only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calibration as CAL
from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE


def _rel(P, **kw):
    r = P.rel()
    r.update(kw)
    return tuple(sorted(r.items()))


def grid(P: ST.Params) -> dict[str, list[tuple[str, ST.Params]]]:
    return {
        "reliability TOTAL": [(f"{v:g}", P.with_(reliability=_rel(P, TOTAL=v))) for v in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05)],
        "reliability HANDICAP": [(f"{v:g}", P.with_(reliability=_rel(P, HANDICAP=v))) for v in (0.8, 0.9, 1.0, 1.1)],
        "reliability 1X2": [(f"{v:g}", P.with_(reliability=_rel(P, **{"1X2": v}))) for v in (0.9, 1.0, 1.1, 1.2)],
        "EV weight": [(f"{v:g}", P.with_(ev_weight=v)) for v in (0.0, 0.25, 0.5, 1.0, 2.0)],
        "probability power": [(f"{v:g}", P.with_(power=v)) for v in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
        "preferred odds": [(f"{a:.2f}–{b:.2f}", P.with_(preferred=(a, b))) for a, b in
                           ((1.20, 1.80), (1.30, 2.00), (1.40, 2.50), (1.60, 3.00), (1.80, 3.50))],
        "softness": [(f"{v:g}", P.with_(softness=v)) for v in (0.05, 0.15, 0.30, 0.60)],
        "probability band min": [(f"{v:g}", P.with_(min_p=v)) for v in (0.30, 0.40, 0.50, 0.60)],
    }


def mix_tables(o: pd.DataFrame) -> dict:
    o = CAL.annotate(o)
    out = {}
    for key in ("group", "line_key", "side", "odds_band", "league"):
        rows = []
        dec = o.loc[o["decided"]]
        for k, g in dec.groupby(key, observed=True):
            bets = g.loc[g["priced"]]
            lo, hi = C.roi_ci(bets["profit"].to_numpy(), 1000) if len(bets) > 1 else (np.nan, np.nan)
            rows.append({"key": k, "n": int(len(g)), "share": float(len(g) / len(dec)), "hit": float(g["won"].mean()),
                         "stated": float((g["p"] / (1 - g["p_push"])).mean()), "bets": int(len(bets)),
                         "avg_odds": float(bets["odds"].mean()) if len(bets) else None,
                         "roi": float(bets["profit"].mean()) if len(bets) else None, "roi_ci95": [lo, hi]})
        out[key] = sorted(rows, key=lambda r: -r["n"])
    return out


def run() -> dict:
    out = {"mix": {}, "sensitivity": {}}
    for mode in ("production", "bettable"):
        out["mix"][mode], out["sensitivity"][mode] = {}, {}
        for s in C.FOLDS:
            d = CD.historical(s)
            priced_only = mode == "bettable"
            if priced_only:
                d = SE.universe(d)
            P = ST.CURRENT.with_(priced_only=priced_only)
            o = ST.outcome(d, ST.choose(d, P))
            out["mix"][mode][str(s)] = mix_tables(o)
            sens = {}
            for pname, variants in grid(P).items():
                rows = []
                for label, params in variants:
                    oo = ST.outcome(d, ST.choose(d, params))
                    sm = ST.summary(oo, boot=False)
                    sm["share_total"] = float((oo["group"] == "TOTAL").mean())
                    sm["label"] = label
                    rows.append(sm)
                sens[pname] = rows
            out["sensitivity"][mode][str(s)] = sens
            C.log.info("market mix %s %s", mode, s)
    C.save_json("market_mix.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    for mode in ("production", "bettable"):
        sens = out["sensitivity"][mode]
        names = list(sens[str(C.FOLDS[0])].keys())
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharey=True)
        for ax, pname in zip(axes.ravel(), names):
            for k, s in enumerate(C.FOLDS):
                rows = sens[str(s)][pname]
                ax.plot(range(len(rows)), [r["roi"] * 100 for r in rows], marker="o", ms=3, color=PALETTE[k],
                        label=C.period_label(s), ls="-" if s in C.SELECT else ":")
                ax.set_xticks(range(len(rows)), [r["label"] for r in rows], fontsize=6.5, rotation=30)
            ax.axhline(0, color="#555", lw=0.6)
            ax.set_title(pname, fontsize=8.5)
        axes[0, 0].set_ylabel("ROI, %"), axes[1, 0].set_ylabel("ROI, %")
        axes[0, 0].legend(fontsize=6.5)
        fig.suptitle(f"Чувствительность ROI текущего правила к его константам ({'все кандидаты' if mode == 'production' else 'только с реальной ценой'})",
                     fontsize=10)
        save_chart(fig, C.FIGURES / f"mix_sensitivity_{mode}.png")
    # share of groups / lines (production)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, s in enumerate(C.FOLDS):
        rows = out["mix"]["production"][str(s)]["line_key"][:9]
        axes[0].barh(np.arange(len(rows)) + k * 0.27, [r["share"] * 100 for r in rows], 0.27, color=PALETTE[k],
                     label=C.period_label(s))
        if k == 0:
            axes[0].set_yticks(np.arange(len(rows)) + 0.27, [r["key"] for r in rows], fontsize=7.5)
    axes[0].invert_yaxis(), axes[0].set_xlabel("Доля Main, %"), axes[0].legend(fontsize=7)
    axes[0].set_title("Какие линии выбирает Main (все кандидаты)", fontsize=9.5)
    for k, s in enumerate(C.FOLDS):
        rows = [r for r in out["mix"]["bettable"][str(s)]["odds_band"] if r["roi"] is not None]
        rows = sorted(rows, key=lambda r: r["key"])
        axes[1].errorbar(np.arange(len(rows)) + (k - 1) * 0.1, [r["roi"] * 100 for r in rows],
                         yerr=[[(r["roi"] - r["roi_ci95"][0]) * 100 for r in rows],
                               [(r["roi_ci95"][1] - r["roi"]) * 100 for r in rows]],
                         marker="o", ms=3.5, capsize=2, ls="none", color=PALETTE[k], label=C.period_label(s))
        axes[1].set_xticks(range(len(rows)), [r["key"] for r in rows], fontsize=7.5)
    axes[1].axhline(0, color="#555", lw=0.6), axes[1].set_ylabel("ROI, %")
    axes[1].set_title("ROI Main по диапазонам коэффициента (только с ценой)", fontsize=9.5)
    save_chart(fig, C.FIGURES / "mix_lines_odds.png")


if __name__ == "__main__":
    C.setup_logging("mix")
    run()
