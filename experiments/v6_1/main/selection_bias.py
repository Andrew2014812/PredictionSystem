"""Analysis 2 — selection bias (winner's curse) of the Main pick, and analysis 5 (information in the market).

* chosen vs not-chosen candidates at the same stated probability: stated - observed;
* dependence on the number of candidates in the band;
* EV inflation: stated EV vs realised return, by EV band, for all priced candidates and for the chosen picks;
* market information: y ~ logit q + (logit p - logit q) per group (how much of the model's disagreement with the
  de-margined Bet365 price is signal), Shin vs proportional, log loss of model / market / blend;
* correlation of residuals between markets of the same match;
* a simulation: if the true probability were the market's and the model = truth + noise of the observed size,
  how over-confident would the chosen pick look?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as CD
from . import common as C
from . import strategies as ST
from . import strategies_eval as SE
from . import transforms as T

EV_BANDS = [-1, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 10]
NCAND_BANDS = [0, 2, 4, 6, 8, 11, 100]


def _cond(p, push):
    return np.asarray(p, float) / np.clip(1 - np.asarray(push, float), 1e-6, None)


def chosen_vs_rest(d: pd.DataFrame, idx: np.ndarray, priced_only: bool) -> dict:
    p = d["p"].to_numpy(float)
    band = (p >= 0.40) & (p <= 0.85) & ~d["group"].isin(ST.EXCLUDED).to_numpy() & np.isin(d["status"], [0, 1])
    if priced_only:
        band &= d["odds"].notna().to_numpy()
    chosen = np.zeros(len(d), bool)
    chosen[idx] = True
    y = (d["status"].to_numpy() == 1).astype(float)
    pc = _cond(p, d["p_push"])
    bins = np.digitize(pc, np.arange(0.40, 0.90, 0.05))
    rows = []
    for b in np.unique(bins[band]):
        for flag in (True, False):
            m = band & (bins == b) & (chosen == flag)
            if m.sum() >= 30:
                rows.append({"bin": float(0.40 + 0.05 * (b - 1)), "chosen": flag, "n": int(m.sum()),
                             "stated": float(pc[m].mean()), "observed": float(y[m].mean())})
    # same-probability comparison: reweight the non-chosen to the chosen probability distribution
    ch, rest = band & chosen, band & ~chosen
    w = np.zeros(len(d))
    for b in np.unique(bins[ch]):
        nc, nr = (ch & (bins == b)).sum(), (rest & (bins == b)).sum()
        if nr:
            w[rest & (bins == b)] = nc / nr
    gap_rest_rw = float(np.sum(w[rest] * (pc[rest] - y[rest])) / max(w[rest].sum(), 1e-9))
    return {"bins": rows, "gap_chosen": float(np.mean(pc[ch] - y[ch])), "gap_rest": float(np.mean(pc[rest] - y[rest])),
            "gap_rest_reweighted": gap_rest_rw, "n_chosen": int(ch.sum()), "n_rest": int(rest.sum())}


def by_ncand(o: pd.DataFrame, ncand: np.ndarray) -> list[dict]:
    q = pd.qcut(pd.Series(ncand).rank(method="first"), 5, labels=False).to_numpy()
    o = o.assign(ncand=ncand, nb=q)
    rows = []
    for k, g in o.loc[o["decided"]].groupby("nb", observed=True):
        bets = g.loc[g["priced"]]
        pc = _cond(g["p"], g["p_push"])
        rows.append({"band": int(k), "range": f"{int(g['ncand'].min())}–{int(g['ncand'].max())}", "n": int(len(g)),
                     "mean_ncand": float(g["ncand"].mean()),
                     "stated": float(pc.mean()), "observed": float(g["won"].mean()),
                     "gap": float(pc.mean() - g["won"].mean()),
                     "se": float(np.sqrt(max(g["won"].mean() * (1 - g["won"].mean()), 1e-9) / len(g))),
                     "bets": int(len(bets)), "roi": float(bets["profit"].mean()) if len(bets) else None})
    return rows


def ev_inflation(d: pd.DataFrame, rows_mask: np.ndarray, p_col: str = "p") -> list[dict]:
    p = d[p_col].to_numpy(float)
    odds = d["odds"].to_numpy(float)
    st = d["status"].to_numpy()
    m = rows_mask & np.isfinite(odds) & np.isin(st, [0, 1])
    e = p * odds - (1 - d["p_push"].to_numpy(float))
    prof = np.where(st == 1, odds - 1, -1.0)
    b = np.digitize(e, EV_BANDS) - 1
    out = []
    for k in range(len(EV_BANDS) - 1):
        mm = m & (b == k)
        if mm.sum() >= 30:
            lo, hi = C.roi_ci(prof[mm], 1000)
            out.append({"band": f"[{EV_BANDS[k]:+.2f}; {EV_BANDS[k + 1]:+.2f})", "n": int(mm.sum()),
                        "stated_ev": float(e[mm].mean()), "realised_roi": float(prof[mm].mean()), "ci95": [lo, hi],
                        "avg_odds": float(odds[mm].mean())})
    return out


def loss_table(d: pd.DataFrame, fit_blend: T.Blend) -> dict:
    """Log loss / Brier of model, market (Shin, proportional) and the inner-fitted blend on all priced candidates."""
    st = d["status"].to_numpy()
    ok = np.isin(st, [0, 1]) & np.isfinite(d["q_prod_shin"].to_numpy(float))
    x = d.loc[ok]
    y = (x["status"].to_numpy() == 1).astype(float)
    out = {}
    cands = {"model": x["p"].to_numpy(float), "market_shin": x["q_prod_shin"].to_numpy(float),
             "market_prop": x["q_prod_prop"].to_numpy(float), "blend_inner": fit_blend.transform(x)}
    if "q_ps_shin" in x and np.isfinite(x["q_ps_shin"]).mean() > 0.5:
        cands["pinnacle_shin"] = x["q_ps_shin"].to_numpy(float)
    for g in ["ALL"] + sorted(x["group"].unique()):
        m = np.ones(len(x), bool) if g == "ALL" else x["group"].to_numpy() == g
        both = m & np.all([np.isfinite(v) for v in cands.values()], axis=0)
        out[g] = {k: C.binary_scores(y[both], _cond(v[both], x["p_push"].to_numpy()[both])) for k, v in cands.items()}
    return out


def residual_correlation(d: pd.DataFrame) -> dict:
    sels = [("1X2", "H"), ("1X2", "D"), ("1X2", "A"), ("TOTAL_1.5", "OVER"), ("TOTAL_2.5", "OVER"),
            ("TOTAL_3.5", "OVER"), ("BTTS", "YES"), ("DOUBLE_CHANCE", "1X"), ("HANDICAP_-1.5", "H"),
            ("HANDICAP_+1.5", "A"), ("CORNERS_9.5", "OVER")]
    cols = {}
    for mk, s in sels:
        x = d.loc[(d["market"] == mk) & (d["selection"] == s) & d["status"].isin([0, 1])]
        cols[f"{mk} {s}"] = pd.Series((x["status"] == 1).astype(float).to_numpy() - _cond(x["p"], x["p_push"]),
                                      index=x["match_id"].to_numpy())
    frame = pd.DataFrame(cols)
    corr = frame.corr()
    return {"labels": list(corr.columns), "matrix": corr.round(3).to_numpy().tolist(),
            "mean_abs_offdiag": float(np.nanmean(np.abs(corr.to_numpy()[~np.eye(len(corr), dtype=bool)])))}


def simulate(d: pd.DataFrame, params: ST.Params, reps: int = 20, seed: int = C.SEED) -> dict:
    """Truth = the de-margined market probability; model = truth + noise (logit scale) of the observed size.

    The noise s.d. per group is sqrt(max(var(logit p - logit q) - 0, 0)) — an upper bound of pure noise. The
    chosen pick (production score on the simulated model probability) is then compared with the truth."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(d["q_prod_shin"].to_numpy(float))
    x = d.loc[ok].copy()
    x["mi"] = pd.factorize(x["match_id"])[0]
    q = np.clip(_cond(x["q_prod_shin"], x["p_push"]), 1e-4, 1 - 1e-4)
    p = np.clip(_cond(x["p"], x["p_push"]), 1e-4, 1 - 1e-4)
    diff = T.logit(p) - T.logit(q)
    sd = x.assign(dd=diff).groupby("group")["dd"].std().to_dict()
    gaps, rois = [], []
    for _ in range(reps):
        noise = rng.normal(0, x["group"].map(sd).to_numpy(float))
        ps = T.sigmoid(T.logit(q) + noise) * (1 - x["p_push"].to_numpy(float))
        truth = q * (1 - x["p_push"].to_numpy(float))
        sim = x.assign(p=ps)
        idx = ST.choose(sim, params)
        gaps.append(float(np.mean(ps[idx] - truth[idx])))
        e_true = truth[idx] * sim["odds"].to_numpy(float)[idx] - (1 - x["p_push"].to_numpy(float)[idx])
        rois.append(float(np.nanmean(e_true)))
    return {"noise_sd_logit": sd, "expected_gap_chosen": float(np.mean(gaps)), "gap_sd": float(np.std(gaps)),
            "expected_true_ev_chosen": float(np.mean(rois)), "reps": reps}


def run() -> dict:
    out = {"bettable": {}, "production": {}}
    for s in C.FOLDS:
        full = CD.historical(s)
        for mode in ("production", "bettable"):
            priced_only = mode == "bettable"
            d = SE.universe(full) if priced_only else full.copy()
            d = SE.add_ncand(d, priced_only)
            P = ST.CURRENT.with_(priced_only=priced_only)
            idx = ST.choose(d, P)
            o = ST.outcome(d, idx)
            res = {"chosen_vs_rest": chosen_vs_rest(d, idx, priced_only),
                   "by_ncand": by_ncand(o, d["ncand"].to_numpy()[idx])}
            p = d["p"].to_numpy(float)
            band = (p >= 0.40) & (p <= 0.85) & ~d["group"].isin(ST.EXCLUDED).to_numpy()
            chosen = np.zeros(len(d), bool)
            chosen[idx] = True
            res["ev_inflation_all_priced"] = ev_inflation(d, band)
            res["ev_inflation_chosen"] = ev_inflation(d, chosen)
            if priced_only:
                inner = SE.universe(CD.historical(s - 1))
                blend = T.Blend("prod", "shin").fit(inner)
                res["losses"] = loss_table(d, blend)
                res["blend_weights_inner"] = blend.w
                res["disagreement"] = T.disagreement_fit(d)
                res["disagreement_pinnacle"] = T.disagreement_fit(d, book="ps") if np.isfinite(
                    d["q_ps_shin"]).mean() > 0.5 else None
                res["simulation"] = simulate(d, P)
                # market-implied EV of the chosen picks (what the bookmaker's own probability says)
                qq = d["q_prod_shin"].to_numpy(float)[idx]
                odds = d["odds"].to_numpy(float)[idx]
                res["market_ev_chosen"] = float(np.nanmean(qq * odds - (1 - d["p_push"].to_numpy(float)[idx])))
                res["margin_chosen"] = float(np.nanmean(1 / odds - qq))
            else:
                res["residual_correlation"] = residual_correlation(d)
            out[mode][str(s)] = res
            C.log.info("selection bias %s %s", s, mode)
    C.save_json("selection_bias.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    plt = C.plt()
    from scripts.report_kit import PALETTE, save_chart
    # chosen vs rest at equal stated probability
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, mode, title in zip(axes, ("production", "bettable"), ("Все кандидаты", "Кандидаты с реальной ценой")):
        for k, s in enumerate(C.SELECT):
            rows = out[mode][str(s)]["chosen_vs_rest"]["bins"]
            for flag, ls in ((True, "-"), (False, ":")):
                r = [x for x in rows if x["chosen"] == flag]
                ax.plot([x["stated"] for x in r], [x["observed"] for x in r], marker="o" if flag else "s", ms=3,
                        ls=ls, color=PALETTE[k], label=f"{'выбран' if flag else 'не выбран'} {C.period_label(s)}")
        ax.plot([0.38, 0.88], [0.38, 0.88], color="#888", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9.5), ax.set_xlabel("Заявленная вероятность")
    axes[0].set_ylabel("Доля выигрышей"), axes[0].legend(fontsize=7)
    fig.suptitle("Проклятие победителя: выбранный кандидат против невыбранных той же вероятности", fontsize=10)
    save_chart(fig, C.FIGURES / "bias_chosen_vs_rest.png")
    # EV inflation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key, title in zip(axes, ("ev_inflation_all_priced", "ev_inflation_chosen"),
                              ("Все кандидаты с ценой (0.40–0.85)", "Выбранные Main (A)")):
        for k, s in enumerate(C.FOLDS):
            r = out["bettable"][str(s)][key]
            ax.errorbar([x["stated_ev"] * 100 for x in r], [x["realised_roi"] * 100 for x in r],
                        yerr=[[(x["realised_roi"] - x["ci95"][0]) * 100 for x in r],
                              [(x["ci95"][1] - x["realised_roi"]) * 100 for x in r]],
                        marker="o", ms=3.5, capsize=2, lw=1, color=PALETTE[k], label=C.period_label(s))
        ax.plot([-15, 25], [-15, 25], color="#888", ls="--", lw=0.8)
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_title(title, fontsize=9.5), ax.set_xlabel("Заявленный EV, %")
    axes[0].set_ylabel("Фактический ROI, %"), axes[0].legend(fontsize=7)
    fig.suptitle("Инфляция EV: заявленное преимущество против фактической доходности (Bet365)", fontsize=10)
    save_chart(fig, C.FIGURES / "bias_ev_inflation.png")
    # number of candidates
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode, title in zip(axes, ("production", "bettable"), ("Все кандидаты", "Кандидаты с реальной ценой")):
        for k, s in enumerate(C.FOLDS):
            r = out[mode][str(s)]["by_ncand"]
            ax.errorbar([x["mean_ncand"] for x in r], [x["gap"] * 100 for x in r],
                        yerr=[1.96 * x["se"] * 100 for x in r], marker="o", ms=3.5, capsize=2, color=PALETTE[k],
                        label=C.period_label(s))
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_xlabel("Число кандидатов в диапазоне 0.40–0.85 (квинтили)"), ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("Заявлено − факт, п.п."), axes[0].legend(fontsize=8)
    fig.suptitle("Переоценка выбранного Main в зависимости от числа кандидатов", fontsize=10)
    save_chart(fig, C.FIGURES / "bias_ncand.png")
    # residual correlation heat map (2024)
    rc = out["production"]["2024"]["residual_correlation"]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    m = np.array(rc["matrix"], float)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rc["labels"])), rc["labels"], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(rc["labels"])), rc["labels"], fontsize=7)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(m[i, j]) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляция остатков (исход − p) между рынками одного матча, 2024/25", fontsize=9.5)
    save_chart(fig, C.FIGURES / "bias_residual_corr.png")


if __name__ == "__main__":
    C.setup_logging("bias")
    run()
