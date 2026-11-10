"""Risk Selection 2.0 — runs every experiment and writes results/.

    python -m experiments.v6_2.risk.run             # everything
    python -m experiments.v6_2.risk.run --figures   # figures only

Stages: per season (2022 ... 2026) load + prepare (pool rows only are kept), check of the vectorised production
rule against src.markets.recommendation.risk_prediction, diagnostics (EV decomposition, calibration of rare
outcomes, favourite-longshot, margins, selection effect, odds dependence, breakdowns), post-selection calibration
of the Risk probability, alternative rankings and fewer-bets rules with paired bootstrap and Holm, EV with refunds.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.v6_1.risk import strategies as RS
from experiments.v6_2.main_calibration import methods as M

from . import analysis as A
from . import common as C
from . import data as D
from . import strategies as S

PRICED_COLS = ["season", "match_id", "group", "market", "selection", "odds", "p", "pc", "qc", "p_mkt", "p_mkt_u",
               "overround", "decided", "y", "profit", "priced", "push_p"]


def season_table(s: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(pool rows of the production Risk rule, priced complete-book rows, info)."""
    d = S.prepare(D.load(s))
    info = {"matches": int(d["match_id"].nunique()), "rows": int(len(d))}
    if s in (2024, C.CURRENT):
        chk = RS.check_current(d, n_max=400)
        info["check_vs_production"] = chk
    priced = d.loc[d["priced"] & d["p_mkt"].notna(), PRICED_COLS].copy()
    pool = d.loc[d["cand"]].copy()
    if "stored_risk" in d:
        shown = d.loc[d["stored_risk"]]
        info["stored_risk_in_pool"] = float(shown["cand"].mean())
        pool["stored_risk"] = pool["stored_risk"].astype(bool)
        pool = pd.concat([pool, shown.loc[~shown["cand"]]], ignore_index=True)    # keep shown picks outside the pool
    del d
    return pool.reset_index(drop=True), priced, info


def per_match(picks: pd.DataFrame, n: int, col: str) -> np.ndarray:
    a = np.full(n, np.nan)
    a[picks["mi"].to_numpy()] = picks[col].to_numpy(float)
    return a


def metrics(f: pd.DataFrame, n: int, boot, base: dict | None) -> tuple[dict, dict]:
    dec = f.loc[f["status"].isin(["WON", "LOST"])]
    bets = f.loc[np.isfinite(f["profit"].to_numpy(float))].sort_values("date")
    arr = {"profit": per_match(f, n, "profit"),
           "err": np.where(f.set_index("mi").reindex(range(n))["status"].isin(["WON", "LOST"]).to_numpy(),
                           per_match(f, n, "pc") - per_match(f, n, "y"), np.nan)}
    r = boot.dist(lambda x: np.nanmean(x, 1), arr["profit"])
    out = {"picks": int(len(f)), "settled": int(len(dec)), "bets": int(len(bets)),
           "priced_share": float(len(bets) / max(len(dec), 1)),
           "hit": float(dec["y"].mean()) if len(dec) else None, "mean_p": float(dec["pc"].mean()) if len(dec) else None,
           "gap": float(np.nanmean(arr["err"])), "bet_hit": float(bets["y"].mean()) if len(bets) else None,
           "avg_odds": float(bets["odds"].mean()) if len(bets) else None,
           "roi": float(bets["profit"].mean()) if len(bets) else None, "roi_ci": C.ci(r),
           "model_ev": float(bets["ev"].mean()) if len(bets) else None,
           "max_drawdown": C.max_drawdown(bets["profit"].to_numpy(float)),
           "longest_losing": C.longest_losing(bets["y"].to_numpy(float) == 1)}
    if base is not None:
        dr = boot.dist(lambda x, b: np.nanmean(x, 1) - np.nanmean(b, 1), arr["profit"], base["profit"])
        out.update({"d_roi": float(np.nanmean(arr["profit"]) - np.nanmean(base["profit"])), "d_roi_ci": C.ci(dr),
                    "d_roi_p": C.p_two_sided(dr)})
    return out, arr


class Boot:
    def __init__(self, n, seed=C.SEED):
        self.chunks = [i.astype(np.int32) for i in C.boot_idx(n, C.N_BOOT, seed)]

    def dist(self, fn, *arrays):
        out = []
        for idx in self.chunks:
            with np.errstate(all="ignore"):
                out.append(fn(*[a[idx] for a in arrays]))
        return np.concatenate(out)


def fewer_bets(d: pd.DataFrame, chosen: dict, n: int, boot, base_arr) -> dict:
    """Thresholds on the ranking key (fixed, chosen before the run) and the top N matches of a calendar day."""
    out = {}
    for name, (col, ths) in S.THRESHOLDS.items():
        f = chosen[name]
        for th in ths:
            g = f.loc[(f[col] >= th) & f["priced"]] if col in f else f.iloc[0:0]
            m, _ = metrics(g, n, boot, base_arr)
            out[f"{name} | ключ ≥ {th:+.2f}"] = m
        for k in S.TOP_PER_DAY:
            g = f.loc[f["priced"] & f[col].notna()].copy()
            g["day"] = pd.to_datetime(g["date"]).dt.date
            g = g.loc[g.groupby("day")[col].rank(ascending=False, method="first") <= k]
            m, _ = metrics(g, n, boot, base_arr)
            out[f"{name} | топ-{k} за день"] = m
    return out


def post_calibration(picks: pd.DataFrame, hist_picks: pd.DataFrame, s: int) -> dict:
    """Stated probability of the production Risk after maps fitted on the history Risk picks (no-refund scale)."""
    def frame(x):
        dec = x.loc[x["status"].isin(["WON", "LOST"])]
        return pd.DataFrame({"pc": dec["pc"].to_numpy(float), "y": dec["y"].to_numpy(float),
                             "group": dec["group"].to_numpy(), "qc": (dec["p_mkt_u"] / (1 - dec["push_p"].fillna(0))).to_numpy(float),
                             "ncand": dec["ncand"].to_numpy(), "priced": dec["priced"].to_numpy(),
                             "league": dec["league"].to_numpy(), "date": dec["date"].to_numpy(),
                             "odds": dec["odds"].to_numpy(float), "profit": dec["profit"].to_numpy(float)})
    f, h = frame(picks), frame(hist_picks)
    out = {}
    probs = {"сырая p": f["pc"].to_numpy(float)}
    for nm in ("PS-Platt", "PS-Group", "PS-Market", "PS-OS"):
        probs[nm] = M.POST[nm]().fit(h).predict(f)
    if s == C.CURRENT:
        from experiments.v6_2.main_calibration.study import _monthly
        for nm in ("PS-Platt", "PS-Market"):
            probs[f"{nm} по месяцам"] = _monthly(f, nm, h)
    q = f["qc"].to_numpy(float)
    probs["рынок (Шин→пропорц.)"] = np.where(np.isfinite(q), q, np.nan)
    y = f["y"].to_numpy(float)
    for nm, p in probs.items():
        ok = np.isfinite(p)
        err = p[ok] - y[ok]
        ll = -np.mean(y[ok] * np.log(np.clip(p[ok], 1e-6, 1)) + (1 - y[ok]) * np.log(np.clip(1 - p[ok], 1e-6, 1)))
        out[nm] = {"n": int(ok.sum()), "mean_p": float(p[ok].mean()), "hit": float(y[ok].mean()),
                   "gap": float(err.mean()), "gap_ci": C.mean_ci(err), "log_loss": float(ll),
                   "reliability": C.reliability_bins(y[ok], p[ok], edges=np.arange(0.10, 0.60, 0.05), min_n=30)}
        bets = np.isfinite(f["profit"].to_numpy(float)) & ok
        if bets.any():
            ev = p[bets] * f["odds"].to_numpy(float)[bets] - 1
            out[nm]["mean_ev_decided"] = float(ev.mean())
            out[nm]["roi"] = float(f["profit"].to_numpy(float)[bets].mean())
    return out


def run_all() -> dict:
    res = {"info": {}, "strategies": {}, "fewer": {}, "pooled": {}, "diagnostics": {}, "post": {},
           "breakdowns": {}}
    pools, priced_all, cur_picks = {}, {}, {}
    keep_cols = ["season", "match_id", "date", "league", "market", "selection", "group", "key", "p", "pc", "qc",
                 "push_p", "odds", "priced", "decided", "y", "status", "profit", "stake_profit", "ev", "p_mkt",
                 "p_mkt_u", "ev_mkt", "cand", "rel", "eligible", "is_main", "mi", "ncand", "ncand_priced"]
    for s in C.HISTORY + (C.CURRENT,):
        t0 = time.time()
        pool, priced, info = season_table(s)
        pools[s] = pool[[c for c in keep_cols + ["stored_risk"] if c in pool]].copy()
        priced_all[s] = priced
        res["info"][str(s)] = info | {"seconds": round(time.time() - t0, 1)}
        C.log.info("loaded %s: %s", s, info)
    keep = {}
    for s in C.SEASONS:
        H = [h for h in C.HISTORY if h < s]
        hist = pd.concat([pools[h] for h in H], ignore_index=True)
        fitted = S.Fitted(hist)
        d = fitted.apply(pools[s].loc[pools[s]["cand"]])
        n = int(d["mi"].max()) + 1
        boot = Boot(n)
        chosen, sres, arrs = {}, {}, {}
        base = None
        for name, (fam, fn) in S.CATALOGUE.items():
            ch = S.picks_frame(d, fn(d))
            chosen[name] = ch
            m, a = metrics(ch, n, boot, None if base is None else base)
            if base is None:
                base = a
            m["family"] = fam
            sres[name] = m
            arrs[name] = a
        if s == C.CURRENT:
            sh = pools[s].loc[pools[s].get("stored_risk", False) == True]          # noqa: E712
            sh = S.picks_frame(sh, sh.drop_duplicates("mi"))
            m, _ = metrics(sh, n, boot, base)
            m["family"] = "shown"
            sres["Risk как показан"] = m
            cur = chosen["Текущий Risk (v6.1)"]
            res["info"][str(s)]["recomputed_equals_shown"] = float(
                pd.Series(list(zip(cur["match_id"], cur["market"], cur["selection"]))).isin(
                    list(zip(sh["match_id"], sh["market"], sh["selection"]))).mean())
            chosen["Risk как показан"] = sh
        res["strategies"][str(s)] = sres
        res["fewer"][str(s)] = fewer_bets(d, chosen, n, boot, base)
        keep[s] = arrs
        cur = chosen["Текущий Risk (v6.1)"]
        cur_picks[s] = cur.assign(season=s)
        # diagnostics on the production Risk of this season
        diag = {"ev_decomposition": A.ev_decomposition(cur), "rare_pool": A.calib_by_band(d),
                "rare_pool_priced": A.calib_by_band(d.loc[d["priced"]]),
                "rare_picks": A.calib_by_band(cur.assign(decided=cur["status"].isin(["WON", "LOST"]),
                                                         qc=cur["p_mkt_u"] / (1 - cur["push_p"].fillna(0)))),
                "favourite_longshot": A.favourite_longshot(priced_all[s]),
                "margins": A.margins(priced_all[s]), "margin_by_band": A.margin_by_band(priced_all[s]),
                "selection": A.selection_effect(d, cur), "odds_dependence": A.odds_dependence(cur),
                "fitted": fitted.cal.params()}
        if s == C.CURRENT:
            diag["ev_decomposition_shown"] = A.ev_decomposition(chosen["Risk как показан"])
            diag["odds_dependence_shown"] = A.odds_dependence(chosen["Risk как показан"])
        res["diagnostics"][str(s)] = diag
        res["breakdowns"][str(s)] = {"group": A.breakdown(cur, "group"), "league": A.breakdown(cur, "league")}
        # post-selection calibration: maps fitted on the production Risk of the history seasons
        hp = []
        for h in H:
            dh = pools[h].loc[pools[h]["cand"]]
            hp.append(S.picks_frame(dh, RS.current(dh)))
        res["post"][str(s)] = post_calibration(cur, pd.concat(hp, ignore_index=True), s)
        if s == C.CURRENT:
            res["post"]["2026_shown"] = post_calibration(chosen["Risk как показан"], pd.concat(hp, ignore_index=True), s)
        C.log.info("season %s done", s)
    # pooled selection seasons, Holm over the non-baseline rules
    names = [nm for nm in keep[C.SELECT[0]] if nm != "Текущий Risk (v6.1)"]
    cat = {nm: np.concatenate([keep[s][nm]["profit"] for s in C.SELECT]) for nm in list(names) + ["Текущий Risk (v6.1)"]}
    boot = Boot(len(cat["Текущий Risk (v6.1)"]), seed=C.SEED + 1)
    b = cat["Текущий Risk (v6.1)"]
    pv, rows = {}, {}
    for nm in names:
        dr = boot.dist(lambda x, y: np.nanmean(x, 1) - np.nanmean(y, 1), cat[nm], b)
        rows[nm] = {"roi": float(np.nanmean(cat[nm])), "d_roi": float(np.nanmean(cat[nm]) - np.nanmean(b)),
                    "d_roi_ci": C.ci(dr), "p": C.p_two_sided(dr)}
        pv[nm] = rows[nm]["p"]
    hol = C.holm(pv)
    for nm in rows:
        rows[nm]["p_holm"] = hol[nm]
    res["pooled"] = {"base_roi": float(np.nanmean(b)), "rules": rows, "n_rules": len(rows)}
    # pooled diagnostics 2023-2025 (favourite-longshot / margins on all priced rows of 2022-2025)
    pr = pd.concat([priced_all[s] for s in C.HISTORY], ignore_index=True)
    res["diagnostics"]["pooled_history"] = {"favourite_longshot": A.favourite_longshot(pr), "margins": A.margins(pr),
                                            "margin_by_band": A.margin_by_band(pr)}
    cp = pd.concat([cur_picks[s] for s in (2023, 2024, 2025)], ignore_index=True)
    res["breakdowns"]["2023-2025"] = {"group": A.breakdown(cp, "group"), "league": A.breakdown(cp, "league"),
                                      "odds": A.odds_dependence(cp)}
    C.save_json("risk.json", res)
    for s, f in cur_picks.items():
        f.to_parquet(C.CACHE / f"risk_picks_{s}.parquet")
    return res


SELECTION_RULE = ("Среди правил, отличных от текущего (12 правил каталога), на 2023/24 и 2024/25: доля ставок с ценой "
                  "≥ 85 % и средний коэффициент ≥ 2.80 в обоих сезонах; выбирается правило с наибольшим min(ROI 2023/24, "
                  "ROI 2024/25). Перспективно, если ΔROI к текущему > 0 в обоих сезонах и Holm-p объединённого теста < 0.05.")


def select(res: dict) -> dict:
    st = res["strategies"]
    ok = []
    for nm in st[str(C.SELECT[0])]:
        if nm in ("Текущий Risk (v6.1)", "Risk как показан"):
            continue
        if all(st[str(s)][nm]["priced_share"] >= 0.85 and (st[str(s)][nm]["avg_odds"] or 0) >= 2.80 for s in C.SELECT):
            ok.append(nm)
    ranked = sorted(ok, key=lambda nm: -min(st[str(s)][nm]["roi"] for s in C.SELECT))
    best = ranked[0] if ranked else None
    out = {"rule": SELECTION_RULE, "eligible": ok,
           "ranking": [{"rule": nm, "worst_roi": min(st[str(s)][nm]["roi"] for s in C.SELECT)} for nm in ranked],
           "chosen": best}
    if best:
        pooled = res["pooled"]["rules"][best]
        out.update({"d_roi_each": [st[str(s)][best]["d_roi"] for s in C.SELECT], "pooled": pooled,
                    "promising": bool(all(st[str(s)][best]["d_roi"] > 0 for s in C.SELECT) and pooled["p_holm"] < 0.05)})
    C.save_json("selection.json", out)
    return out


def ev_stage() -> dict:
    from experiments.v6_2.main_calibration import data as MD
    from experiments.v6_2.main_calibration import study as MS
    from . import ev_refunds

    def main_tables():
        for s in C.SEASONS:
            yield s, MS.prepare(MD.universe(MD.load(s), "site"))

    def risk_tables():
        for s in C.SEASONS:
            yield s, S.prepare(D.load(s))
    return ev_refunds.run(main_tables(), risk_tables())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--ev", action="store_true", help="only the EV-with-refunds stage")
    a = ap.parse_args()
    C.setup_logging()
    if a.figures:
        from . import figures
        figures.all_figures()
        return
    t = {}
    if not a.ev:
        t0 = time.time()
        res = run_all()
        select(res)
        t["main"] = round(time.time() - t0, 1)
    t0 = time.time()
    ev_stage()
    t["ev"] = round(time.time() - t0, 1)
    from . import figures
    figures.all_figures()
    C.save_json("run_info.json", {"git_sha": C.git_sha(), "data": C.data_version(), "environment": C.environment(),
                                  "timings_seconds": t, "finished": C.stamp(),
                                  "seasons": {"select": C.SELECT, "info": C.INFO, "current": C.CURRENT},
                                  "risk_band": S.BAND, "z_lcb": S.Z_LCB})


if __name__ == "__main__":
    main()
