"""Q4 — consistency and probability quality of every derived market on the fold predictions.

The production chain is reproduced with the production functions: score_matrix (independent
Poisson, rho = 0 as in config.DIXON_COLES = False) -> reconcile_with_1x2 (classifier 1X2) ->
goal_markets(); corners: corners_distribution(mu, r) with r from the inner season -> corner_markets().

Checks per fold: every probability in [0, 1]; complementary selections sum to 1; Asian whole lines
win + push + lose = 1; European 3-way handicaps sum to 1; identities between markets (AH -0.5 H =
1X2 H, AH +0.5 H = 1X 1X2, EH -1 D = AH -1 push, ...); monotonicity over lines.
Quality: log loss / Brier of each market against a base-rate baseline of the training seasons.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src import config
from src.markets.distributions import (corners_distribution, empirical_score_matrix, reconcile_with_1x2,
                                       score_matrix)
from src.markets.markets import corner_markets, goal_markets
from src.modelling.training import negative_binomial_size

from . import common as C
from . import scores as S

log = C.setup_logging("markets")
TOL = 1e-6                     # classifier probabilities are float32 (sum off by ~1e-7)


def consistency(outer: pd.DataFrame, r: float | None) -> dict:
    """Run the production market functions match by match and record every violation."""
    worst = defaultdict(float)
    count = defaultdict(int)
    n = 0

    def chk(name, value):
        worst[name] = max(worst[name], abs(value))
        count[name] += int(abs(value) > TOL)

    for row in outer.itertuples():
        m = reconcile_with_1x2(score_matrix(row.lh, row.la), np.array([row.pH, row.pD, row.pA]))
        sels = goal_markets(m)
        if row.cmu == row.cmu:
            sels += corner_markets(corners_distribution(row.cmu, r))
        by = defaultdict(dict)
        for s in sels:
            by[s.market][s.selection] = (s.probability, s.push_probability)
            chk("probability outside [0,1]", min(0.0, s.probability) + max(0.0, s.probability - 1))
            chk("push outside [0,1]", min(0.0, s.push_probability) + max(0.0, s.push_probability - 1))
        p = {k: v[0] for k, v in by["1X2"].items()}
        chk("1X2 sum = 1", sum(p.values()) - 1)
        chk("1X2 = classifier", max(abs(p["H"] - row.pH), abs(p["D"] - row.pD), abs(p["A"] - row.pA)))
        dc = {k: v[0] for k, v in by["DOUBLE_CHANCE"].items()}
        chk("DC 1X = H + D", dc["1X"] - p["H"] - p["D"])
        chk("DC X2 = D + A", dc["X2"] - p["D"] - p["A"])
        chk("DC 12 = H + A", dc["12"] - p["H"] - p["A"])
        overs = []
        for line in config.TOTAL_GOAL_LINES:
            t = by[f"TOTAL_{line}"]
            chk("totals over + under = 1", t["OVER"][0] + t["UNDER"][0] - 1)
            overs.append(t["OVER"][0])
        chk("totals monotone in line", max(0.0, max(np.diff(overs))))
        chk("BTTS yes + no = 1", by["BTTS"]["YES"][0] + by["BTTS"]["NO"][0] - 1)
        chk("score matrix sums to 1", m.sum() - 1)
        ex = [s.probability for s in sels if s.market == "EXACT_SCORE"]
        chk("top-5 exact scores sum <= 1", max(0.0, sum(ex) - 1))
        for line in config.HANDICAP_LINES:
            h = by[f"HANDICAP_{line:+g}"]
            (ph, pp), (pa, pp2) = h["H"], h["A"]
            chk("AH push equal on both sides", pp - pp2)
            if (abs(line * 2) % 2) == 1:
                chk("AH half line: H + A = 1, no push", ph + pa - 1 + pp)
            else:
                chk("AH whole line: win + push + lose = 1", ph + pp + pa - 1)
                e = by[f"EHANDICAP_{line:+g}"]
                chk("EH 3-way sum = 1", e["H"][0] + e["D"][0] + e["A"][0] - 1)
                chk("EH draw = AH push", e["D"][0] - pp)
                chk("EH H = AH H win", e["H"][0] - ph)
        chk("AH -0.5 H = 1X2 H", by["HANDICAP_-0.5"]["H"][0] - p["H"])
        chk("AH +0.5 H = DC 1X", by["HANDICAP_+0.5"]["H"][0] - dc["1X"])
        chk("AH -1 push = P(home wins by exactly 1)", by["HANDICAP_-1"]["H"][1]
            - sum(m[i + 1, i] for i in range(m.shape[0] - 1)))
        if row.cmu == row.cmu:
            co = []
            for line in config.CORNER_LINES:
                c = by[f"CORNERS_{line}"]
                chk("corners over + under = 1", c["OVER"][0] + c["UNDER"][0] - 1)
                co.append(c["OVER"][0])
            chk("corner lines monotone", max(0.0, max(np.diff(co))))
        n += 1
    return {"matches": n, "tolerance": TOL,
            "checks": {k: {"max_abs_violation": worst[k], "violations": count[k]} for k in worst}}


def multi_ll(y, p):
    return float(log_loss(y, np.clip(p, 1e-12, 1), labels=list(range(p.shape[1]))))


def quality(outer: pd.DataFrame, train: pd.DataFrame, r: float | None, raw_too: bool = True) -> dict:
    p1x2 = outer[["pH", "pD", "pA"]].to_numpy()
    raw = S.matrices(outer["lh"], outer["la"])
    rec = S.reconcile(raw, p1x2)
    hg, ag = outer["home_goals"].to_numpy(int), outer["away_goals"].to_numpy(int)
    thg, tag = train["home_goals"].to_numpy(int), train["away_goals"].to_numpy(int)
    I, J = S._I, S._J
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    ty = np.where(thg > tag, 0, np.where(thg == tag, 1, 2))
    tfreq = np.bincount(ty, minlength=3) / len(ty)
    out = {"1X2": {"model_log_loss": multi_ll(y, p1x2), "baseline_log_loss": multi_ll(y, np.tile(tfreq, (len(y), 1))),
                   "model_brier": float(np.mean(np.sum((p1x2 - np.eye(3)[y]) ** 2, 1))),
                   "baseline_brier": float(np.mean(np.sum((tfreq - np.eye(3)[y]) ** 2, 1)))}}
    out["1X2"]["poisson_matrix_log_loss"] = multi_ll(y, S.outcome_1x2(raw))

    def binary_market(name, mask_fn, outcome, train_outcome):
        for tag_, mat in (("reconciled", rec), ("poisson_raw", raw)):
            if tag_ == "poisson_raw" and not raw_too:
                continue
            p = (mat * mask_fn(I, J)).sum((1, 2))
            out.setdefault(name, {})[tag_] = C.binary_metrics(outcome, p, float(np.mean(train_outcome)))

    binary_market("DC 1X", lambda i, j: i >= j, y != 2, ty != 2)
    binary_market("DC X2", lambda i, j: i <= j, y != 0, ty != 0)
    binary_market("DC 12", lambda i, j: i != j, y != 1, ty != 1)
    for line in config.TOTAL_GOAL_LINES:
        binary_market(f"Total over {line}", lambda i, j, l=line: (i + j) > l, hg + ag > line, thg + tag > line)
    binary_market("BTTS yes", lambda i, j: (i > 0) & (j > 0), (hg > 0) & (ag > 0), (thg > 0) & (tag > 0))
    # exact score
    emp = empirical_score_matrix(thg, tag)
    h, a = np.minimum(hg, S.G - 1), np.minimum(ag, S.G - 1)
    out["Exact score"] = {"model_log_loss": float(-np.log(np.clip(rec[np.arange(len(h)), h, a], 1e-12, 1)).mean()),
                          "poisson_raw_log_loss": float(-np.log(np.clip(raw[np.arange(len(h)), h, a], 1e-12, 1)).mean()),
                          "baseline_log_loss": float(-np.log(emp[h, a]).mean()),
                          "top1_hit_rate": float(np.mean(rec.reshape(len(h), -1).argmax(1) == h * S.G + a))}
    # Asian / European handicaps
    margin, tmargin = hg - ag, thg - tag
    for line in config.HANDICAP_LINES:
        win = (rec * ((I + line - J) > 0)).sum((1, 2))
        push = (rec * ((I + line - J) == 0)).sum((1, 2))
        lose = 1 - win - push
        oc = np.where(margin + line > 0, 0, np.where(margin + line == 0, 1, 2))
        toc = np.where(tmargin + line > 0, 0, np.where(tmargin + line == 0, 1, 2))
        if abs(line * 2) % 2 == 1:
            out[f"AH {line:+g} (H)"] = C.binary_metrics(oc == 0, win, float(np.mean(toc == 0)))
        else:
            P = np.clip(np.stack([win, push, lose], 1), 1e-12, 1)
            tf = np.bincount(toc, minlength=3) / len(toc)
            out[f"AH {line:+g} (win/push/lose) = EH {line:+g}"] = {
                "model_log_loss": multi_ll(oc, P), "baseline_log_loss": multi_ll(oc, np.tile(tf, (len(oc), 1))),
                "push_predicted": float(push.mean()), "push_observed": float((oc == 1).mean())}
    # corners
    if True:                                      # corners (NB with the inner-season size r)
        ok = outer["cmu"].notna() & outer["total_corners"].notna()
        tc = train["total_corners"].dropna()
        for line in config.CORNER_LINES:
            pmf_over = np.array([corners_distribution(mu, r)[int(np.floor(line)) + 1:].sum()
                                 for mu in outer.loc[ok, "cmu"]])
            out[f"Corners over {line}"] = C.binary_metrics(outer.loc[ok, "total_corners"].to_numpy() > line,
                                                           pmf_over, float((tc > line).mean()))
    # bookmaker references on the same rows (normalised Bet365)
    o = outer[["odds_over25", "odds_under25"]].to_numpy(float)
    ok = np.all(o > 1, 1)
    inv = 1 / o[ok]
    pb = inv[:, 0] / inv.sum(1)
    p_model = (rec[ok] * ((I + J) > 2.5)).sum((1, 2))
    out["Total over 2.5"]["bookmaker_same_rows"] = {
        "n": int(ok.sum()), "bookmaker_log_loss": float(log_loss((hg + ag > 2.5)[ok], pb, labels=[0, 1])),
        "model_log_loss": float(log_loss((hg + ag > 2.5)[ok], np.clip(p_model, 1e-6, 1 - 1e-6), labels=[0, 1]))}
    o = outer[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    ok = np.all(o > 1, 1)
    inv = 1 / o[ok]
    out["1X2"]["bookmaker_same_rows"] = {"n": int(ok.sum()),
                                         "bookmaker_log_loss": multi_ll(y[ok], inv / inv.sum(1, keepdims=True)),
                                         "model_log_loss": multi_ll(y[ok], p1x2[ok])}
    return out


def gap_1x2(outer: pd.DataFrame) -> dict:
    p = outer[["pH", "pD", "pA"]].to_numpy()
    q = S.outcome_1x2(S.matrices(outer["lh"], outer["la"]))
    d = q - p
    return {"mean_abs_gap": {c: float(np.abs(d[:, i]).mean()) for i, c in enumerate(C.CLASSES)},
            "max_abs_gap": {c: float(np.abs(d[:, i]).max()) for i, c in enumerate(C.CLASSES)},
            "mean_gap_poisson_minus_classifier": {c: float(d[:, i].mean()) for i, c in enumerate(C.CLASSES)},
            "share_abs_gap_gt_0.05": {c: float((np.abs(d[:, i]) > 0.05).mean()) for i, c in enumerate(C.CLASSES)},
            "argmax_disagreement": float((p.argmax(1) != q.argmax(1)).mean()),
            "mean_p_draw_classifier": float(p[:, 1].mean()), "mean_p_draw_poisson": float(q[:, 1].mean())}


def run() -> dict:
    t0 = time.time()
    df = C.features()
    out = {"folds": {}}
    for s in C.FOLDS:
        inner, outer, info = C.base_predictions(s)
        ok = inner["cmu"].notna() & inner["total_corners"].notna()
        r = negative_binomial_size(inner.loc[ok, "total_corners"].to_numpy(float), inner.loc[ok, "cmu"].to_numpy())
        out["folds"][str(s)] = {"nb_size_inner": r, "consistency": consistency(outer, r),
                                "gap_1x2_score_matrix_vs_classifier": gap_1x2(outer),
                                "quality": quality(outer, C.before(df, s), r)}
        log.info("fold %s done (%.0fs)", s, time.time() - t0)
    # does every market beat its base rate on both selection folds?
    verdict = {}
    for mk, v in out["folds"]["2023"]["quality"].items():
        def ll(x):
            x = x.get("reconciled", x)
            return x.get("log_loss", x.get("model_log_loss")), x.get("baseline_log_loss")
        res = [ll(out["folds"][str(s)]["quality"][mk]) for s in C.SELECT]
        verdict[mk] = {"beats_base_rate_both": bool(all(a < b for a, b in res)),
                       "skill_by_fold": {str(s): float(1 - a / b) for s, (a, b) in zip(C.SELECT, res)}}
    out["verdict"] = verdict
    out["minutes"] = (time.time() - t0) / 60
    C.save_json("markets.json", out)
    return out


if __name__ == "__main__":
    run()
