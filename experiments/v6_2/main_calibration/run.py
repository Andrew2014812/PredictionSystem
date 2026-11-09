"""Main Calibration 2.0 — runs every experiment and writes results/.

    python -m experiments.v6_2.main_calibration.run            # everything
    python -m experiments.v6_2.main_calibration.run --figures  # figures only (from results/)

Stages: variants (all seasons, both universes, paired bootstrap), pooled selection-season tests with Holm,
breakdowns (group, league, number of candidates), league transfer (leave-one-league-out), EV calibration,
frozen selection rule, run_info.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import common as C
from . import evaluate as E
from . import methods as M
from . import study as S

KINDS = ("site", "bet")
POST_NAMES = list(M.POST)
DETAIL = ["A базовый", "PS-Platt", "PS-Group", "PS-OS", "PS-Market", "PS-EB-League", "Q3 Q2 + Платт после выбора",
          "Q4 Q2 + смесь с рынком", "PS-Group по месяцам", "показано + PS-Group", "показано + PS-Group по месяцам",
          "показано + PS-Platt по месяцам"]


def slug(name: str) -> str:
    return {"A базовый": "A", "A как показано": "Astored"}.get(name, name.split()[0])


def run_variants() -> dict:
    res = {"variants": {}, "info": {}, "families": {}, "breakdowns": {}, "ncand": {}, "reliability": {},
           "ev": {}, "pooled": {}}
    keep = {}                                   # (kind, season) -> {variant: per-match arrays} for pooled tests
    for kind in KINDS:
        hist = S.History(kind)
        for s in C.SEASONS:
            meta, out, info = S.season_variants(s, kind, hist)
            n = len(meta)
            boot = E.Boot(n)
            base_f = out["A базовый"][1]
            base_a = E.arrays(base_f, n)
            vres, arrs = {}, {}
            for name, (fam, f) in out.items():
                a = E.arrays(f, n)
                shown = name.startswith("показано")
                ref = (E.arrays(out["A как показано"][1], n) if shown else base_a)
                same = fam in ("post", "post_month", "post_shown") or name == "A базовый"
                vres[name] = E.metrics(f, a, boot, None if name == "A базовый" else ref, same)
                vres[name]["reference"] = "A как показано" if shown else "A базовый"
                vres[name]["family"] = fam
                res["families"][name] = fam
                arrs[name] = {k: a[k] for k in ("err", "profit", "ll")}
                if name in DETAIL or name == "A как показано":
                    key = f"{kind}|{s}|{name}"
                    res["breakdowns"][key] = {"group": E.breakdown(f, "group"), "league": E.breakdown(f, "league")}
                    res["ncand"][key] = E.by_ncand(f, kind)
                    dec = f.loc[np.isfinite(f["y"].to_numpy(float))]
                    res["reliability"][key] = C.reliability_bins(dec["y"], dec["pc"])
                    if kind == "bet":
                        res["ev"][key] = {"bins": E.ev_bins(f, "ev"), "slope": E.ev_slope(f, "ev")}
                    f.assign(season=s, kind=kind, variant=name).to_parquet(
                        C.CACHE / f"picks_{kind}_{s}_{slug(name)}.parquet")
            res["variants"].setdefault(kind, {})[str(s)] = vres
            res["info"].setdefault(kind, {})[str(s)] = info
            keep[(kind, s)] = arrs
            C.log.info("variants %s %s: %d matches, %.0fs", kind, s, n, info["seconds"])
            del boot
    # pooled selection seasons: paired bootstrap over the concatenated matches, Holm across variants
    for kind in KINDS:
        names = list(keep[(kind, C.SELECT[0])])
        cat = {nm: {k: np.concatenate([keep[(kind, s)][nm][k] for s in C.SELECT]) for k in ("err", "profit", "ll")}
               for nm in names if all(nm in keep[(kind, s)] for s in C.SELECT)}
        boot = E.Boot(len(cat["A базовый"]["err"]), seed=C.SEED + 1)
        b = cat["A базовый"]
        rows, pg, pr = {}, {}, {}
        for nm, a in cat.items():
            if nm == "A базовый":
                continue
            g = boot.dist(lambda x: np.nanmean(x, 1), a["err"])
            dg = boot.dist(lambda x, y: np.abs(np.nanmean(x, 1)) - np.abs(np.nanmean(y, 1)), a["err"], b["err"])
            dr = boot.dist(lambda x, y: np.nanmean(x, 1) - np.nanmean(y, 1), a["profit"], b["profit"])
            rows[nm] = {"gap": float(np.nanmean(a["err"])), "gap_ci": C.ci(g),
                        "d_absgap": float(abs(np.nanmean(a["err"])) - abs(np.nanmean(b["err"]))), "d_absgap_ci": C.ci(dg),
                        "d_absgap_p": C.p_two_sided(dg), "roi": float(np.nanmean(a["profit"])),
                        "d_roi": float(np.nanmean(a["profit"]) - np.nanmean(b["profit"])), "d_roi_ci": C.ci(dr),
                        "d_roi_p": C.p_two_sided(dr), "family": res["families"][nm]}
            pg[nm] = rows[nm]["d_absgap_p"]
            if res["families"][nm] != "post":
                pr[nm] = rows[nm]["d_roi_p"]
        hg, hr = C.holm(pg), C.holm(pr)
        for nm in rows:
            rows[nm]["d_absgap_p_holm"] = hg.get(nm)
            rows[nm]["d_roi_p_holm"] = hr.get(nm)
        g0 = boot.dist(lambda x: np.nanmean(x, 1), b["err"])
        res["pooled"][kind] = {"variants": rows, "base_gap": float(np.nanmean(b["err"])), "base_gap_ci": C.ci(g0),
                               "base_roi": float(np.nanmean(b["profit"])), "n_variants_gap": len(pg),
                               "n_variants_roi": len(pr), "matches": int(len(b["err"]))}
    C.save_json("variants.json", res)
    return res


def ncand_summary() -> dict:
    """Number of Main candidates (probability 0.40-0.85; in ``bet`` with a real price) of the baseline picks."""
    out = {}
    for kind in KINDS:
        for s in C.SEASONS:
            f = pd.read_parquet(C.CACHE / f"picks_{kind}_{s}_A.parquet", columns=["ncand"])
            q = f["ncand"].quantile([0.1, 0.5, 0.9]).tolist()
            out.setdefault(kind, {})[str(s)] = {"mean": float(f["ncand"].mean()), "p10": q[0], "median": q[1], "p90": q[2]}
    C.save_json("ncand_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ league transfer
def league_transfer() -> dict:
    """PS-Platt fitted on the history picks of the OTHER leagues vs on all leagues, applied to each league."""
    out = {}
    for kind in KINDS:
        hist = S.History(kind)
        rows = []
        for s in C.SEASONS:
            H = [h for h in C.HISTORY if h < s]
            hb = S._decided(hist.base_picks(H))
            f = S._decided(pd.read_parquet(C.CACHE / f"picks_{kind}_{s}_A.parquet"))
            pooled = M.Platt().fit(hb["pc"], hb["y"])
            for lg, g in f.groupby("league"):
                other = hb.loc[hb["league"] != lg]
                lolo = M.Platt().fit(other["pc"], other["y"])
                hit = float(g["y"].mean())
                rows.append({"season": s, "league": lg, "n": int(len(g)), "hit": hit,
                             "raw": float(g["pc"].mean()), "pooled": float(pooled.predict(g["pc"]).mean()),
                             "lolo": float(lolo.predict(g["pc"]).mean()),
                             "se": float(np.sqrt(max(hit * (1 - hit), 1e-9) / len(g)))})
        df = pd.DataFrame(rows)
        summ = {}
        for per, sub in (("2023-2025", df.loc[df["season"] < C.CURRENT]), ("2026", df.loc[df["season"] == C.CURRENT])):
            agg = sub.groupby("league").apply(lambda g: pd.Series({
                "n": g["n"].sum(), "hit": np.average(g["hit"], weights=g["n"]),
                "raw": np.average(g["raw"], weights=g["n"]), "pooled": np.average(g["pooled"], weights=g["n"]),
                "lolo": np.average(g["lolo"], weights=g["n"])}), include_groups=False).reset_index()
            agg["se"] = np.sqrt(agg["hit"] * (1 - agg["hit"]) / agg["n"])
            res = {}
            for col in ("raw", "pooled", "lolo"):
                z = (agg[col] - agg["hit"]) / agg["se"]
                pv = 2 * (1 - __import__("scipy").stats.norm.cdf(np.abs(z)))
                hol = C.holm(dict(zip(agg["league"], pv)))
                res[col] = {"mean_abs_gap": float(np.average(np.abs(agg[col] - agg["hit"]), weights=agg["n"])),
                            "n_sig_05": int((pv < 0.05).sum()), "n_sig_holm": int(sum(v < 0.05 for v in hol.values())),
                            "chi2": float(np.sum(z ** 2)), "leagues": int(len(agg))}
            summ[per] = {"summary": res, "table": agg.to_dict("records")}
        out[kind] = summ
    C.save_json("league_transfer.json", out)
    return out


# ------------------------------------------------------------------------------------------------ EV calibration
def ev_calibration() -> dict:
    """Stated EV of the baseline bets under several probabilities vs the realised per-stake return."""
    out = {}
    hist = S.History("bet")
    for s in C.SEASONS:
        H = [h for h in C.HISTORY if h < s]
        hb = S._decided(hist.base_picks(H))
        f = pd.read_parquet(C.CACHE / f"picks_bet_{s}_A.parquet")
        push = f["p_push"].to_numpy(float)
        odds = f["odds"].to_numpy(float)
        probs = {"сырая p": f["pc_raw"].to_numpy(float)}
        for nm in ("PS-Platt", "PS-OS", "PS-Market"):
            m = M.POST[nm]().fit(hb)
            probs[nm] = m.predict(f)
        q = f["qc"].to_numpy(float)
        probs["рынок (Шин)"] = q
        res = {}
        for nm, pc in probs.items():
            g = f.assign(ev_x=np.where(f["priced"], pc * (1 - push) * odds - (1 - push), np.nan))
            sel = g.loc[g["ev_x"] > 0]
            bets = sel.loc[np.isfinite(sel["profit"].to_numpy(float))]
            pr = bets["profit"].to_numpy(float)
            res[nm] = {"slope": E.ev_slope(g, "ev_x"), "bins": E.ev_bins(g, "ev_x"),
                       "positive_ev": {"bets": int(len(pr)), "roi": float(pr.mean()) if len(pr) else None,
                                       "roi_ci": C.ci(C.boot_stat(lambda x: np.nanmean(x, 1), pr)) if len(pr) > 1 else None,
                                       "share": float(len(pr) / max(np.isfinite(g["profit"]).sum(), 1))}}
        out[str(s)] = res
        C.log.info("ev calibration %s", s)
    C.save_json("ev_calibration.json", out)
    return out


# ------------------------------------------------------------------------------------------------ frozen selection
SELECTION_RULE = {
    "calibration": "Среди всех вариантов (кроме базового) на универсуме site: минимум max(|gap 2023/24|, |gap 2024/25|); "
                   "при равенстве — меньший log loss 2024/25. Перспективен, если 95 % ДИ gap содержит 0 в обоих сезонах "
                   "отбора на обоих универсумах, а |gap| < |gap базового| в объединённых сезонах отбора (Holm p < 0.05).",
    "roi": "Среди вариантов, меняющих выбор (pre/rank/few), на универсуме bet: максимум min(ROI 2023/24, ROI 2024/25). "
           "Перспективен, если ΔROI к базовому > 0 в обоих сезонах отбора и Holm-p объединённого теста < 0.05.",
}


def select(res: dict) -> dict:
    v = res["variants"]
    cands = [nm for nm in v["site"][str(C.SELECT[0])] if nm not in ("A базовый", "A как показано")]

    def worst_gap(kind, nm):
        return max(abs(v[kind][str(s)][nm]["gap"]) for s in C.SELECT)
    ranked = sorted(cands, key=lambda nm: (worst_gap("site", nm), v["site"][str(C.SELECT[1])][nm]["log_loss"]))
    best = ranked[0]
    ok_ci = all(v[k][str(s)][best]["gap_ci"][0] <= 0 <= v[k][str(s)][best]["gap_ci"][1]
                for k in KINDS for s in C.SELECT if best in v[k][str(s)])
    pooled = res["pooled"]["site"]["variants"][best]
    cal = {"chosen": best, "family": res["families"][best], "worst_gap_site": worst_gap("site", best),
           "ranking": [{"variant": nm, "worst_abs_gap_site": worst_gap("site", nm),
                        "worst_abs_gap_bet": max(abs(v["bet"][str(s)][nm]["gap"]) for s in C.SELECT)
                        if nm in v["bet"][str(C.SELECT[0])] else None} for nm in ranked],
           "ci_contains_zero_all": ok_ci, "pooled_d_absgap": pooled["d_absgap"],
           "pooled_d_absgap_p_holm": pooled["d_absgap_p_holm"],
           "promising": bool(ok_ci and pooled["d_absgap"] < 0 and (pooled["d_absgap_p_holm"] or 1) < 0.05)}
    rv = v["bet"]
    rc = [nm for nm in rv[str(C.SELECT[0])] if res["families"].get(nm) in ("pre", "rank", "few")]
    rranked = sorted(rc, key=lambda nm: -min(rv[str(s)][nm]["roi"] for s in C.SELECT))
    rb = rranked[0]
    rp = res["pooled"]["bet"]["variants"][rb]
    roi = {"chosen": rb, "family": res["families"][rb], "worst_roi": min(rv[str(s)][rb]["roi"] for s in C.SELECT),
           "base_worst_roi": min(rv[str(s)]["A базовый"]["roi"] for s in C.SELECT),
           "d_roi_each": [rv[str(s)][rb]["d_roi"] for s in C.SELECT], "pooled_d_roi": rp["d_roi"],
           "pooled_d_roi_ci": rp["d_roi_ci"], "pooled_p_holm": rp["d_roi_p_holm"],
           "ranking": [{"variant": nm, "worst_roi": min(rv[str(s)][nm]["roi"] for s in C.SELECT)} for nm in rranked],
           "promising": bool(all(rv[str(s)][rb]["d_roi"] > 0 for s in C.SELECT) and (rp["d_roi_p_holm"] or 1) < 0.05)}
    out = {"rule": SELECTION_RULE, "calibration": cal, "roi": roi,
           "n_variants_tried": {k: len(v[k][str(C.SELECT[0])]) - 1 for k in KINDS}}
    C.save_json("selection.json", out)
    return out


def run_info(timings: dict) -> None:
    from experiments.v6_2.main_calibration import data as D
    C.save_json("run_info.json", {"git_sha": C.git_sha(), "data": C.data_version(), "environment": C.environment(),
                                  "timings_seconds": timings, "seasons": {"select": C.SELECT, "info": C.INFO,
                                                                          "current": C.CURRENT},
                                  "band": D.BAND, "min_fit": M.MIN_FIT, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--ncand", action="store_true", help="only the candidate-count summary (needs the picks cache)")
    a = ap.parse_args()
    C.setup_logging()
    if a.ncand:
        ncand_summary()
        return
    if a.figures:
        from . import figures
        figures.all_figures()
        return
    t = {}
    t0 = time.time()
    res = run_variants()
    t["variants"] = round(time.time() - t0, 1)
    ncand_summary()
    t0 = time.time()
    league_transfer()
    t["league_transfer"] = round(time.time() - t0, 1)
    t0 = time.time()
    ev_calibration()
    t["ev_calibration"] = round(time.time() - t0, 1)
    select(res)
    from . import figures
    figures.all_figures()
    run_info(t)


if __name__ == "__main__":
    main()
