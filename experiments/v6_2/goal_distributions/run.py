"""Goal-distribution study: every family on the same matches, the same XGBoost goal means, walk-forward.

    python -m experiments.v6_2.goal_distributions.run

Season s (2023/24 ... 2026/27) is predicted with the production walk-forward goal means lh, la and 1X2
probabilities of fold s (models fitted on seasons < s); the shape parameters of every family are fitted by
maximum likelihood of the observed scores on season s-1 predicted by fold s-1 (models fitted on seasons < s-1),
separately for the raw matrix and for the matrix reconciled with the 1X2 classifier (``+R``, as production).
Production = ``P+R`` (independent Poisson, rho = 0, reconciled).

Results: results/fits.json, tests.json, metrics.json, comparisons.json, calibration.json, leagues.json,
main_risk.json, decisions.json, run_info.json.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy import stats

from experiments.v6_2.market_models import data as MD
from experiments.v6_2.market_models import probs as PR
from experiments.v6_2.market_models import selection as SEL
from experiments.v6_research import scores as SC

from . import common as C
from . import dists as D
from . import maher

ALL_LINES = PR.ALL_TOTAL_LINES
FAMILIES = ["P", "DC", "BP", "NB", "CMP", "FR", "FRCMP", "ZI", "MH1", "MH2", "HY", "XC"]
NAMES = {"P": "Пуассон (независимый)", "DC": "Диксон–Коулс", "BP": "Бивариантный Пуассон", "NB": "Отр. биномиальное",
         "CMP": "Конвей–Максвелл–Пуассон", "FR": "Копула Франка", "FRCMP": "Франк + КМП", "ZI": "Инфляция 0-0",
         "MH1": "Махер (атака/защита)", "MH2": "Махер (дом/выезд раздельно)", "HY": "Смесь XGB + Махер",
         "XC": "Контроль: перекалибровка средних XGB"}
REF = "P+R"


# ------------------------------------------------------------------------------------------------ inputs
def season_frame(season: int, mh: pd.DataFrame) -> pd.DataFrame:
    b, _ = MD.base(season)
    m = mh.set_index("match_id").reindex(b["match_id"])
    for c in ("lh_m1", "la_m1", "lh_m2", "la_m2"):
        b[c] = m[c].to_numpy()
    return b


def _means(fam: str, b: pd.DataFrame, par: dict) -> tuple[np.ndarray, np.ndarray]:
    lh, la = b["lh"].to_numpy(float), b["la"].to_numpy(float)
    if fam in ("MH1", "MH2"):
        k = fam[-1]
        mh, ma = b[f"lh_m{k}"].to_numpy(float), b[f"la_m{k}"].to_numpy(float)
        return np.where(np.isfinite(mh), mh, lh), np.where(np.isfinite(ma), ma, la)
    if fam == "XC":                                    # control: log-linear recalibration of the XGB means
        return (np.exp(par["ah"] + par["bh"] * np.log(lh)), np.exp(par["aa"] + par["ba"] * np.log(la)))
    if fam == "HY":
        k = par["maher"]
        mh, ma = b[f"lh_m{k}"].to_numpy(float), b[f"la_m{k}"].to_numpy(float)
        mh, ma = np.where(np.isfinite(mh), mh, lh), np.where(np.isfinite(ma), ma, la)
        w = par["w"]
        return np.exp(w * np.log(lh) + (1 - w) * np.log(mh)), np.exp(w * np.log(la) + (1 - w) * np.log(ma))
    return lh, la


def matrices(fam: str, b: pd.DataFrame, par: dict, reconcile: bool) -> np.ndarray:
    lh, la = _means(fam, b, par)
    if fam == "DC":
        M = D.dc(lh, la, par["rho"])
    elif fam == "BP":
        M = D.bp(lh, la, par["l3"])
    elif fam == "NB":
        M = D.nb(lh, la, par["rh"], par["ra"])
    elif fam == "CMP":
        M = D.cmp(lh, la, par["nu"])
    elif fam == "FR":
        M = D.frank(lh, la, par["theta"])
    elif fam == "FRCMP":
        M = D.frank_cmp(lh, la, par["theta"], par["nu"])
    elif fam == "ZI":
        M = D.zero_inflate(D.poisson(lh, la), par["w"])
    else:
        M = D.poisson(lh, la)
    if reconcile:
        M = SC.reconcile(M, b[["pH", "pD", "pA"]].to_numpy())
    return M


# ------------------------------------------------------------------------------------------------ fitting
def fit(fam: str, b: pd.DataFrame, reconcile: bool) -> dict:
    hg, ag = b["home_goals"].to_numpy(), b["away_goals"].to_numpy()

    def nll(par):
        return -D.score_loglik(matrices(fam, b, par, reconcile), hg, ag).mean()

    def scalar(name, lo, hi):
        r = minimize_scalar(lambda x: nll({name: x}), bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        return {name: float(r.x)}

    if fam in ("P", "MH1", "MH2"):
        par = {}
    elif fam == "DC":
        par = scalar("rho", -0.3, 0.3)                  # the production bounds (fit_dixon_coles_rho)
    elif fam == "BP":
        par = scalar("l3", 0.0, 0.4)
    elif fam == "CMP":
        par = scalar("nu", 0.6, 2.0)
    elif fam == "FR":
        par = scalar("theta", -3.0, 3.0)
    elif fam == "ZI":
        par = scalar("w", 0.0, 0.1)
    elif fam == "NB":
        f = lambda x: nll({"rh": float(np.exp(x[0])), "ra": float(np.exp(x[1]))})   # noqa: E731
        best = min((minimize(f, x0=[x0, x0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-8,
                                                                            "maxiter": 300}) for x0 in (2.0, 6.0)),
                   key=lambda r: r.fun)
        x = np.clip(best.x, 0.0, 9.2)
        par = {"rh": float(np.exp(x[0])), "ra": float(np.exp(x[1]))}
    elif fam == "FRCMP":
        f = lambda x: nll({"theta": float(np.clip(x[0], -3, 3)), "nu": float(np.clip(x[1], 0.6, 2.0))})  # noqa
        r = minimize(f, x0=[-0.3, 1.1], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
        par = {"theta": float(np.clip(r.x[0], -3, 3)), "nu": float(np.clip(r.x[1], 0.6, 2.0))}
    elif fam == "XC":
        # per side Poisson GLM log mu = a + b log(lambda_XGB) on season s-1 (the raw score likelihood factorises);
        # the same calibration is used for the reconciled matrix
        par = {}
        for side, lam, y in (("h", b["lh"].to_numpy(float), hg), ("a", b["la"].to_numpy(float), ag)):
            x = np.log(lam)
            f = lambda t: float(np.mean(np.exp(t[0] + t[1] * x) - y * (t[0] + t[1] * x)))   # noqa: E731
            r = minimize(f, x0=[0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
            par[f"a{side}"], par[f"b{side}"] = float(r.x[0]), float(r.x[1])
    elif fam == "HY":
        best = None
        for k in ("1", "2"):
            r = minimize_scalar(lambda w: nll({"maher": k, "w": w}), bounds=(0.0, 1.0), method="bounded",
                                options={"xatol": 1e-4})
            if best is None or r.fun < best[0]:
                best = (r.fun, {"maher": k, "w": float(r.x)})
        par = best[1]
    else:
        raise ValueError(fam)
    par["inner_nll"] = float(nll(par))
    return par


# ------------------------------------------------------------------------------------------------ tests
def excess_tests(b: pd.DataFrame) -> dict:
    """Observed vs expected counts of 0-0, 1-1, draws and 0 goals per side under independent Poisson (raw)."""
    M = D.poisson(b["lh"].to_numpy(), b["la"].to_numpy())
    hg, ag = b["home_goals"].to_numpy(), b["away_goals"].to_numpy()
    cells = {"0-0": (M[:, 0, 0], (hg == 0) & (ag == 0)), "1-1": (M[:, 1, 1], (hg == 1) & (ag == 1)),
             "draw": (np.einsum("nii->n", M), hg == ag), "home_0": (M[:, 0, :].sum(1), hg == 0),
             "away_0": (M[:, :, 0].sum(1), ag == 0)}
    out = {}
    for k, (p, o) in cells.items():
        e, v = p.sum(), (p * (1 - p)).sum()
        z = (o.sum() - e) / np.sqrt(v)
        out[k] = {"observed": int(o.sum()), "expected": float(e), "ratio": float(o.sum() / e), "z": float(z),
                  "p_one_sided": float(1 - stats.norm.cdf(z))}
    return out


def dispersion_test(y, mu) -> dict:
    """Cameron-Trivedi: ((y - mu)^2 - y) / mu = alpha * mu + e; alpha > 0 over-, < 0 under-dispersion."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    z = ((y - mu) ** 2 - y) / mu
    alpha = float((z * mu).sum() / (mu ** 2).sum())
    res = z - alpha * mu
    se = float(np.sqrt((res ** 2 * mu ** 2).sum()) / (mu ** 2).sum())          # robust (HC0)
    return {"alpha": alpha, "se": se, "t": alpha / se, "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(alpha / se)))),
            "var_over_mean": float(np.mean((y - mu) ** 2) / np.mean(mu))}


# ------------------------------------------------------------------------------------------------ evaluation
LOSS_COLS = (["SCORE", "ES17", "TOT7_RPS", "GD7_RPS", "TOTALS", "BTTS", "AH", "AH_HALF", "AH_WHOLE", "1X2", "DC"]
             + [f"T{l}" for l in ALL_LINES] + [f"AH{l:+g}" for l in PR.AH_HALF] + [f"AHW{l:+g}" for l in PR.AH_WHOLE])
CAL_KEYS = ["TOTAL_0.5", "TOTAL_1.5", "TOTAL_2.5", "TOTAL_3.5", "TOTAL_4.5", "BTTS", "DRAW", "ZERO_ZERO"]


def evaluate(M: np.ndarray, b: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    hg, ag = b["home_goals"].to_numpy(), b["away_goals"].to_numpy()
    book = PR.from_matrix(M, hg, ag, total_lines=ALL_LINES)
    out = PR.outcomes(hg, ag, total_lines=ALL_LINES)
    L = {"SCORE": PR.key_loss("SCORE", book, out), "ES17": PR.key_loss("ES17", book, out),
         "TOT7_RPS": C.rps(out["TOT7"], book["TOT7"]), "GD7_RPS": C.rps(out["GD7"], book["GD7"]),
         "BTTS": PR.key_loss("BTTS", book, out), "1X2": PR.key_loss("1X2", book, out)}
    for l in ALL_LINES:
        L[f"T{l}"] = PR.key_loss(PR.key_total(l), book, out)
    for l in PR.AH_HALF:
        L[f"AH{l:+g}"] = PR.key_loss(PR.key_ah(l), book, out)
    for l in PR.AH_WHOLE:
        L[f"AHW{l:+g}"] = PR.key_loss(PR.key_ahw(l), book, out)
    L["TOTALS"] = np.mean([L[f"T{l}"] for l in ALL_LINES], 0)
    L["AH_HALF"] = np.mean([L[f"AH{l:+g}"] for l in PR.AH_HALF], 0)
    L["AH_WHOLE"] = np.mean([L[f"AHW{l:+g}"] for l in PR.AH_WHOLE], 0)
    L["AH"] = np.mean([L[f"AH{l:+g}"] for l in PR.AH_HALF] + [L[f"AHW{l:+g}"] for l in PR.AH_WHOLE], 0)
    L["DC"] = np.mean([PR.key_loss(k, book, out) for k in ("DC_1X", "DC_X2", "DC_12")], 0)
    loss = pd.DataFrame({k: np.asarray(L[k], np.float32) for k in LOSS_COLS})
    # calibration inputs: (outcome, probability) per key; push: predicted refund probability vs observed
    cal = {k: (out[k], book[k]) for k in CAL_KEYS if k.startswith("TOTAL_") or k == "BTTS"}
    cal["DRAW"] = ((hg == ag).astype(float), book["1X2"][:, 1])
    cal["ZERO_ZERO"] = (((hg == 0) & (ag == 0)).astype(float), M[:, 0, 0])
    push = {f"{l:+g}": (out[PR.key_ahw(l)] == 1, book[PR.key_ahw(l)][:, 1]) for l in PR.AH_WHOLE}
    return loss, cal, push


def main() -> None:
    C.setup_logging()
    T = C.Timer()
    mh, mh_info = maher.build()
    frames = {s: season_frame(s, mh) for s in (2022,) + tuple(C.SEASONS)}

    # --- tests that decide whether a zero-inflated family is justified (selection seasons only)
    tests = {"excess": {}, "dispersion": {}}
    for s in C.SEASONS:
        b = frames[s]
        tests["excess"][str(s)] = excess_tests(b)
        tests["dispersion"][str(s)] = {"home": dispersion_test(b["home_goals"], b["lh"]),
                                       "away": dispersion_test(b["away_goals"], b["la"])}
    sel = pd.concat([frames[s] for s in C.SELECT])
    tests["excess"]["select"] = excess_tests(sel)
    tests["dispersion"]["select"] = {"home": dispersion_test(sel["home_goals"], sel["lh"]),
                                     "away": dispersion_test(sel["away_goals"], sel["la"])}
    zz = [tests["excess"][str(s)]["0-0"] for s in C.SELECT]
    zi_ok = bool(tests["excess"]["select"]["0-0"]["p_one_sided"] < 0.05 and all(x["ratio"] > 1 for x in zz))
    tests["zero_inflation_justified"] = zi_ok
    tests["rule"] = ("0-0 inflation is evaluated only if the observed 0-0 count of 2023/24+2024/25 exceeds the "
                     "independent-Poisson expectation with one-sided p < 0.05 and the ratio is > 1 in both seasons.")
    C.save_json("tests.json", tests)
    fams = [f for f in FAMILIES if f != "ZI" or zi_ok]
    variants = [f + m for f in fams for m in ("", "+R")]
    C.log.info("variants: %s", variants)

    # --- fit on s-1, evaluate on s
    fits, losses, cals, pushes, mats_meta = {}, {}, {}, {}, {}
    for s in C.SEASONS:
        inner, outer = frames[s - 1], frames[s]
        fits[str(s)] = {}
        for v in variants:
            fam, rec = v.replace("+R", ""), v.endswith("+R")
            par = fit(fam, inner, rec)
            fits[str(s)][v] = par
            M = matrices(fam, outer, par, rec)
            loss, cal, push = evaluate(M, outer)
            losses[(v, s)] = loss
            cals[(v, s)] = cal
            pushes[(v, s)] = push
            C.log.info("season %s %s: %s  score LL %.5f", s, v, {k: round(x, 4) for k, x in par.items()
                                                                 if isinstance(x, float)}, loss["SCORE"].mean())
    C.save_json("fits.json", {"fits": fits, "maher": mh_info, "variants": variants, "names": NAMES})

    # --- metrics per variant and season (+ pooled selection)
    def pooled(v, seasons):
        return pd.concat([losses[(v, s)] for s in seasons], ignore_index=True)

    periods = {str(s): (s,) for s in C.SEASONS}
    periods["select"] = C.SELECT
    metrics = {v: {p: pooled(v, ss).mean().to_dict() for p, ss in periods.items()} for v in variants}
    ecev = {}
    for v in variants:
        ecev[v] = {}
        for p, ss in periods.items():
            ecev[v][p] = {k: C.ece(np.concatenate([cals[(v, s)][k][0] for s in ss]),
                                   np.concatenate([cals[(v, s)][k][1] for s in ss])) for k in CAL_KEYS}
    C.save_json("metrics.json", {"loss": metrics, "ece": ecev, "n": {p: int(sum(len(frames[s]) for s in ss))
                                                                       for p, ss in periods.items()}})

    # --- paired comparisons with production (P+R); Holm over variants on the primary metric (selection, pooled)
    comp = {}
    for v in variants:
        if v == REF:
            continue
        comp[v] = {}
        for p, ss in periods.items():
            a, r = pooled(v, ss), pooled(REF, ss)
            comp[v][p] = {k: C.paired(a[k].to_numpy(float), r[k].to_numpy(float), n_boot=1000)
                          for k in ["SCORE", "ES17", "TOT7_RPS", "GD7_RPS", "TOTALS", "BTTS", "AH", "AH_WHOLE", "1X2",
                                    "T1.5", "T2.5", "T3.5"]}
    holm = C.holm({v: comp[v]["select"]["SCORE"]["p_t"] for v in comp})          # normal-approx. p (bootstrap
    #                                                                                 p is limited to 1/1000)
    for v in comp:
        comp[v]["holm_p_score_select"] = holm[v]
    # Holm within each market group (selection, pooled)
    holm_groups = {}
    for k in ["TOTALS", "BTTS", "AH", "AH_WHOLE", "GD7_RPS", "TOT7_RPS", "ES17"]:
        holm_groups[k] = C.holm({v: comp[v]["select"][k]["p_t"] for v in comp})
    C.save_json("comparisons.json", {"reference": REF, "comparisons": comp, "holm_groups": holm_groups,
                                     "n_variants_compared": len(comp)})

    # --- calibration curves and refund calibration (selection pooled and 2026/27)
    calib = {}
    for v in variants:
        calib[v] = {}
        for p, ss in (("select", C.SELECT), ("2025", (2025,)), ("2026", (2026,))):
            calib[v][p] = {k: C.calib_curve(np.concatenate([cals[(v, s)][k][0] for s in ss]),
                                            np.concatenate([cals[(v, s)][k][1] for s in ss]), bins=10)
                           for k in ("TOTAL_2.5", "BTTS", "DRAW", "ZERO_ZERO", "TOTAL_1.5", "TOTAL_3.5")}
            calib[v][p]["push"] = {}
            for l in [f"{x:+g}" for x in PR.AH_WHOLE]:
                o = np.concatenate([pushes[(v, s)][l][0] for s in ss]).astype(float)
                q = np.concatenate([pushes[(v, s)][l][1] for s in ss])
                d = C.boot_stat(lambda a: a.mean(1), o - q, n_boot=1000)
                calib[v][p]["push"][l] = {"predicted": float(q.mean()), "observed": float(o.mean()),
                                          "obs_minus_pred_ci": C.ci(d), "n": int(len(o))}
    C.save_json("calibration.json", calib)

    # --- best variant (frozen rule: lowest pooled selection score log loss) and league breakdown
    best = min((v for v in variants if v != REF), key=lambda v: metrics[v]["select"]["SCORE"])
    sel_meta = pd.concat([frames[s][["league"]] for s in C.SELECT], ignore_index=True)
    a, r = pooled(best, C.SELECT), pooled(REF, C.SELECT)
    leagues = {}
    for lg, idx in sel_meta.groupby("league").groups.items():
        idx = np.asarray(idx)
        leagues[lg] = {"n": int(len(idx)), **C.paired(a["SCORE"].to_numpy(float)[idx],
                                                     r["SCORE"].to_numpy(float)[idx], n_boot=1000)}
    C.save_json("leagues.json", {"variant": best, "reference": REF, "metric": "SCORE", "leagues": leagues})

    # --- Main / Risk with every reconciled family (production selection, real prices)
    mr = main_risk_all([v for v in variants if v.endswith("+R")], frames, fits)
    C.save_json("main_risk.json", mr)

    C.save_json("run_info.json", {"git_sha": C.git_sha(), "environment": C.environment(),
                                  "data_version": C.data_version(), "seconds": T(),
                                  "seasons": {"select": C.SELECT, "info": C.INFO, "current": C.CURRENT},
                                  "variants": variants, "best_by_rule": best,
                                  "matches": {str(s): int(len(frames[s])) for s in frames}})
    decide()
    C.log.info("done in %.0fs", T())


def main_risk_all(variants: list[str], frames: dict, fits: dict) -> dict:
    out = {"variants": {}, "self_check": {}}
    picks = {}
    for s in C.SEASONS:
        b = frames[s]
        _, info = MD.base(s)
        corners = PR.corners_from_nb(b["cmu"].to_numpy(), info["nb_size"])
        prices = MD.prices(s)
        for v in [REF] + [x for x in variants if x != REF]:
            fam = v.replace("+R", "")
            M = matrices(fam, b, fits[str(s)][v], True)
            book = {**PR.from_matrix(M), **corners}
            long = SEL.attach_prices(SEL.candidates(b, book, SEL.top_scores(M)), prices)
            if v == REF:
                out["self_check"][str(s)] = SEL.self_check(long, 200)
            main, risk = SEL.main_risk(long)
            res = {"main": SEL.summary(main), "risk": SEL.summary(risk)}
            picks[(v, s)] = (main, risk)
            if v != REF:
                res["main_vs_ref"] = SEL.paired_roi(picks[(REF, s)][0], main)
                res["risk_vs_ref"] = SEL.paired_roi(picks[(REF, s)][1], risk)
                res["main_hit_vs_ref"] = SEL.paired_hit(picks[(REF, s)][0], main)
            out["variants"].setdefault(v, {})[str(s)] = res
    for v in variants:                                   # pooled selection seasons
        cat = lambda k, vv: pd.concat([picks[(vv, s)][k] for s in C.SELECT], ignore_index=True)  # noqa: E731
        res = {"main": SEL.summary(cat(0, v)), "risk": SEL.summary(cat(1, v))}
        if v != REF:
            res["main_vs_ref"] = SEL.paired_roi(cat(0, REF), cat(0, v))
            res["risk_vs_ref"] = SEL.paired_roi(cat(1, REF), cat(1, v))
            res["main_hit_vs_ref"] = SEL.paired_hit(cat(0, REF), cat(0, v))
        out["variants"][v]["select"] = res
    return out


RULE = ("Primary metric: log loss of the exact score (log score of the whole matrix). A: Holm-adjusted p < 0.05 "
        "(pooled 2023/24+2024/25, all variants vs P+R), better in both selection seasons, not worse by point in "
        "2025/26 and 2026/27, no market group significantly worse on the selection seasons (95 % CI above 0 for "
        "totals, BTTS, handicaps, 1X2) and Main ROI not significantly worse (pooled selection CI not below 0). "
        "B: better in both selection seasons and the pooled 95 % CI below 0 without the Holm correction, or A-level "
        "evidence failing 2025/26 / 2026/27. C: everything else. Added after the first run (BP+R): a family whose "
        "fitted parameter sits in every fold at the value where it equals production (BP l3 = 0, NB r at the bound) "
        "is C — the remaining difference is numerical (tail renormalised instead of folded into 10 goals).")


BOUNDARY = {"BP": ("l3", 1e-3), "NB": ("rh", 5000.0)}


def reduces_to_production(v: str, fits: dict) -> bool:
    """True when the fitted shape parameter sits, in every fold, at the value where the family IS the production
    distribution (BP: l3 = 0; NB: r at the upper bound); any remaining difference is numerical (tail handling)."""
    fam = v.replace("+R", "")
    if fam not in BOUNDARY:
        return False
    k, lim = BOUNDARY[fam]
    vals = [fits[s][v][k] for s in fits]
    return all(x <= lim for x in vals) if fam == "BP" else all(x >= lim and fits[s][v]["ra"] >= lim
                                                                   for s, x in zip(fits, vals))


def decide() -> dict:
    comp = C.load_json("comparisons.json")["comparisons"]
    fits = C.load_json("fits.json")["fits"]
    mr = C.load_json("main_risk.json")["variants"]
    out = {"rule": RULE, "decisions": {}}
    for v, c in comp.items():
        sc = c["select"]["SCORE"]
        both = all(c[str(s)]["SCORE"]["diff"] < 0 for s in C.SELECT)
        later = all(c[str(s)]["SCORE"]["diff"] <= 0 for s in (C.INFO, C.CURRENT))
        worse = [k for k in ("TOTALS", "BTTS", "AH", "1X2") if c["select"][k]["ci95"][0] > 0]
        rk = v if v.endswith("+R") else v + "+R"
        mvr = mr.get(rk, {}).get("select", {}).get("main_vs_ref")
        roi_bad = bool(mvr and mvr["ci95"][1] < 0)
        sig = c["holm_p_score_select"] < 0.05 and sc["diff"] < 0
        reduces = reduces_to_production(v, fits)
        if reduces:
            d = "C"
        elif sig and both and later and not worse and not roi_bad:
            d = "A"
        elif (both and sc["ci95"][1] < 0) or (sig and both):
            d = "B"
        else:
            d = "C"
        out["decisions"][v] = {"decision": d, "reduces_to_production": reduces,
                               "score_diff_select": sc["diff"], "ci95": sc["ci95"],
                               "holm_p": c["holm_p_score_select"], "better_both_selection": both,
                               "not_worse_later": later, "groups_significantly_worse": worse,
                               "main_roi_significantly_worse": roi_bad}
    C.save_json("decisions.json", out)
    return out


if __name__ == "__main__":
    main()
