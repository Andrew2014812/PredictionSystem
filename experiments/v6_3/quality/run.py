"""ИССЛЕДОВАНИЕ 6 — quality vs coverage. Runs every experiment of the study and writes results/ and figures/.

    .venv/bin/python -m experiments.v6_3.quality.oos        # heavy (through heavy.sh): out-of-sample 2024/25, 2025/26
    .venv/bin/python -m experiments.v6_3.quality.run        # light: strata, mitigations, identification, groups
    .venv/bin/python -m experiments.v6_3.quality.build_pdf

Steps
1. Frames: every played 2026/27 match (production outputs, Baseline B Main in both universes) and the selection
   seasons 2024/25 + 2025/26 (out-of-sample refits, Football-Data prices).
2. Strata (strata.py): model losses and Main ROI by completeness / club history / season progress / league /
   price availability.
3. Mitigations (mitigate.py): weights chosen on 2024/25 + 2025/26, then evaluated on 2026/27 in both universes,
   paired with Baseline B; Holm over every variant tried.
4. Identification (ident.py): name audit + controlled misidentification with and without mitigation.
5. Feature groups (groups.py): permutation importance in 2025/26 and 2026/27, missingness by league.
6. Per-match tables, coverage, preregistration, run info, figures.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.v6_3 import common as C
from experiments.v6_3.alt_ml import mainroi as R
from experiments.v6_3.replay import engine as EN
from experiments.v6_3.replay import evaluate as EV

from . import data as Q
from . import groups as G
from . import ident as I
from . import mitigate as MT
from . import strata as S

D = Q.D
RES = D["results"]
HIST = (2024, 2025)
PRIMARY = "mkt_both"
QCOLS = {"close": ["q_home_close", "q_draw_close", "q_away_close"],
         "t09": ["q_home_t09", "q_draw_t09", "q_away_t09"], "fd": ["q_home_fd", "q_draw_fd", "q_away_fd"]}


def hist_frames(log) -> tuple[pd.DataFrame, dict]:
    f = Q.features()
    frames, store = [], {}
    for s in HIST:
        out = pd.read_parquet(D["cache"] / f"oos_{s}.parquet")
        fr = MT.attach_priors(Q.frame_hist(s, out))
        fs = f.loc[f["season"] == s]
        main, cands = R.hist_main(fs, out, f"oos_{s}", keep_candidates=True)
        b = main.set_index("match_id")
        fr["main_fd_profit"] = fr["match_id"].map(b["profit"])
        fr["main_fd_bet"] = fr["match_id"].map(b["bet"]).fillna(0).astype(int)
        fr["main_fd_won"] = fr["match_id"].map((b["status"] == "WON").astype(float).where(b["status"].isin(["WON", "LOST"])))
        fr["season"] = s
        frames.append(fr)
        store[s] = {"outputs": out, "frame": fr, "features": fs, "main": main, "cands": cands}
        log.info("history %s: %d matches, Main bets %d, ROI %.4f", s, len(fr), int(main["bet"].sum()),
                 EV.summary(main)["roi"])
    return pd.concat(frames, ignore_index=True), store


def select_on_history(store: dict, log) -> tuple[dict, pd.DataFrame]:
    rows = []
    base_ll = pd.concat([MT.ll_1x2(store[s]["outputs"], store[s]["frame"]) for s in HIST])
    base_gd = pd.concat([MT.goal_dev(store[s]["outputs"], store[s]["frame"]) for s in HIST])
    base_main = pd.concat([store[s]["main"] for s in HIST])
    for label, fam, w in MT.variant_list():
        lls, gds, mains = [], [], []
        for s in HIST:
            st = store[s]
            if fam == "lowdata_rank":
                mains.append(MT.lowdata_main(st["main"], st["cands"], st["frame"], label))
                lls.append(MT.ll_1x2(st["outputs"], st["frame"]))
                gds.append(MT.goal_dev(st["outputs"], st["frame"]))
                continue
            o = MT.apply(st["outputs"], st["frame"], QCOLS["fd"], fam, w)
            lls.append(MT.ll_1x2(o, st["frame"]))
            gds.append(MT.goal_dev(o, st["frame"]))
            mains.append(R.hist_main(st["features"], o, label))
        ll, gd, mn = pd.concat(lls), pd.concat(gds), pd.concat(mains)
        d_ll = C.paired_mean_diff(ll.to_numpy(), base_ll.reindex(ll.index).to_numpy())
        d_gd = C.paired_mean_diff(gd.to_numpy(), base_gd.reindex(gd.index).to_numpy())
        cmp = EV.compare(mn, base_main)
        sm = EV.summary(mn)
        rows.append({"variant": label, "family": fam, "w": None if w is None else str(w), "log_loss": float(ll.mean()),
                     "d_log_loss": d_ll["diff"], "d_ll_lo": d_ll["ci95"][0], "d_ll_hi": d_ll["ci95"][1],
                     "d_goal_dev": d_gd["diff"], "roi": sm["roi"], "bets": sm["bets"], "hit": sm["hit_rate"],
                     "d_roi": cmp["roi_diff"]["diff"], "d_roi_lo": cmp["roi_diff"]["ci95"][0],
                     "d_roi_hi": cmp["roi_diff"]["ci95"][1], "same_main": cmp["same_main_share"]})
        log.info("history %s: dLL %.5f dGoalDev %.5f dROI %.4f", label, d_ll["diff"], d_gd["diff"], cmp["roi_diff"]["diff"])
    tab = pd.DataFrame(rows)
    chosen = {}
    for fam in ("mkt_season", "mkt_hist"):
        t = tab.loc[tab["family"] == fam]
        chosen[fam] = float(t.sort_values("d_log_loss").iloc[0]["w"])
    t = tab.loc[tab["family"] == "prior"]
    chosen["prior"] = float(t.sort_values("d_goal_dev").iloc[0]["w"])
    # control weight: the average weight the chosen combination gives on the selection seasons
    fr_all = pd.concat([store[s]["frame"] for s in HIST])
    chosen["uniform_w"] = float(MT.weight(fr_all, "mkt_both", (chosen["mkt_season"], chosen["mkt_hist"])).mean())
    # the combined / control variants on history too (reported, not selected on)
    extra = []
    for label, fam, w in MT.variant_list(chosen)[-2:]:
        lls, mains = [], []
        for s in HIST:
            st = store[s]
            o = MT.apply(st["outputs"], st["frame"], QCOLS["fd"], fam, w)
            lls.append(MT.ll_1x2(o, st["frame"]))
            mains.append(R.hist_main(st["features"], o, label))
        ll, mn = pd.concat(lls), pd.concat(mains)
        d_ll = C.paired_mean_diff(ll.to_numpy(), base_ll.reindex(ll.index).to_numpy())
        cmp = EV.compare(mn, base_main)
        sm = EV.summary(mn)
        extra.append({"variant": label, "family": fam, "w": str(w), "log_loss": float(ll.mean()),
                      "d_log_loss": d_ll["diff"], "d_ll_lo": d_ll["ci95"][0], "d_ll_hi": d_ll["ci95"][1],
                      "d_goal_dev": 0.0, "roi": sm["roi"], "bets": sm["bets"], "hit": sm["hit_rate"],
                      "d_roi": cmp["roi_diff"]["diff"], "d_roi_lo": cmp["roi_diff"]["ci95"][0],
                      "d_roi_hi": cmp["roi_diff"]["ci95"][1], "same_main": cmp["same_main_share"]})
    tab = pd.concat([tab, pd.DataFrame(extra)], ignore_index=True)
    bm = EV.summary(base_main)
    chosen["history_baseline"] = {"log_loss": float(base_ll.mean()), "roi": bm["roi"], "bets": bm["bets"],
                                  "hit": bm["hit_rate"]}
    return chosen, tab


def evaluate_current(x: pd.DataFrame, chosen: dict, log) -> tuple[dict, dict]:
    out = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    base_ll = MT.ll_1x2(out, x)
    base_gd = MT.goal_dev(out, x)
    base_cd = MT.corner_dev(out, x)
    low = MT.lowdata_ids(x)
    res, mains = {}, {}
    for label, fam, w in MT.variant_list(chosen):
        sel = fam in ("mkt_both", "mkt_uniform", "lowdata_rank") or (w is not None and w == chosen.get(fam))
        r = {"family": fam, "w": None if w is None else (list(w) if isinstance(w, tuple) else w), "chosen": bool(sel),
             "universes": {}}
        for u in R.UNIVERSES:
            if fam == "lowdata_rank":
                cands = R.candidates_b(u)
                main = MT.lowdata_main(R.baseline_b(u), cands.loc[cands["match_id"].isin(R.results().index)], x, label)
                o = out
            else:
                o = MT.apply(out, x, QCOLS[u], fam, w)
                main = R.run_main(o, u, label)
            ev = R.evaluate(main, u)
            b = R.baseline_b(u)
            sub_new, sub_b = main.loc[main["match_id"].isin(low)], b.loc[b["match_id"].isin(low)]
            ev["lowdata"] = {"matches": int(len(sub_new)), "summary": EV.summary(sub_new),
                             "summary_b": EV.summary(sub_b), "vs_b": EV.compare(sub_new, sub_b)}
            ev["monthly"] = R.monthly_diff(main, b).to_dict("records")
            ev["league"] = R.league_diff(main, b).to_dict("records")
            ev["top_wins"] = R.top_win_dependence(main)
            ev["odds_sensitivity"] = R.odds_sensitivity(main)
            ll = MT.ll_1x2(o, x)
            ev["d_log_loss"] = C.paired_mean_diff(ll.to_numpy(), base_ll.reindex(ll.index).to_numpy())
            ev["d_log_loss_lowdata"] = C.paired_mean_diff(ll.loc[ll.index.isin(low)].to_numpy(),
                                                          base_ll.loc[base_ll.index.isin(low)].to_numpy())
            gd = MT.goal_dev(o, x)
            cd = MT.corner_dev(o, x)
            ev["d_goal_dev"] = C.paired_mean_diff(gd.to_numpy(), base_gd.reindex(gd.index).to_numpy())
            ev["d_corner_dev"] = C.paired_mean_diff(cd.to_numpy(), base_cd.reindex(cd.index).to_numpy())
            r["universes"][u] = ev
            mains[(label, u)] = main
            log.info("2026/27 %s %s: ROI %.4f (B %.4f) dROI %.4f p %.3f dLL %.5f", label, u, ev["summary"]["roi"],
                     EV.summary(b)["roi"], ev["vs_b"]["roi_diff"]["diff"], ev["vs_b"]["roi_diff"]["p"],
                     ev["d_log_loss"]["diff"])
        res[label] = r
    for u in R.UNIVERSES:
        h = C.holm({k: v["universes"][u]["vs_b"]["roi_diff"]["p"] for k, v in res.items()})
        for k in res:
            res[k]["universes"][u]["holm_p"] = h.get(k)
    return res, mains


def identification(chosen: dict, log) -> dict:
    out_p, xp, side = I.perturbed_outputs()
    fr = Q.frame_current()
    fr_p = fr.drop(columns=["n_min", "hist_min"]).merge(
        pd.DataFrame({"match_id": xp["match_id"].to_numpy(),
                      "n_min": np.minimum(xp["h_season_games"], xp["a_season_games"]).to_numpy(),
                      "hist_min": np.minimum(xp["h_form10_n"], xp["a_form10_n"]).to_numpy()}), on="match_id")
    base_out = pd.read_parquet(C.SHARED_CACHE / "outputs_production.parquet")
    base_ll = MT.ll_1x2(base_out, fr)
    res = {"matches": int(len(xp)), "home_side": int((side == "h").sum()), "away_side": int((side == "a").sum())}
    cases = {"misidentified": (out_p, None, None),
             "misidentified + mkt_hist": (out_p, "mkt_hist", chosen["mkt_hist"]),
             "misidentified + mkt_both": (out_p, "mkt_both", (chosen["mkt_season"], chosen["mkt_hist"])),
             "correct + mkt_both": (base_out, "mkt_both", (chosen["mkt_season"], chosen["mkt_hist"]))}
    for name, (o0, fam, w) in cases.items():
        r = {}
        frx = fr if name.startswith("correct") else fr_p
        for u in R.UNIVERSES:
            o = o0 if fam is None else MT.apply(o0, frx, QCOLS[u], fam, w)
            main = R.run_main(o, u, name)
            ev = R.evaluate(main, u)
            ll = MT.ll_1x2(o, fr)
            r[u] = {"summary": ev["summary"], "vs_b": ev["vs_b"], "roi_ci": ev["roi_ci"],
                    "d_log_loss": C.paired_mean_diff(ll.to_numpy(), base_ll.reindex(ll.index).to_numpy())}
            log.info("ident %s %s: ROI %.4f dROI %.4f dLL %.5f same %.3f", name, u, ev["summary"]["roi"],
                     ev["vs_b"]["roi_diff"]["diff"], r[u]["d_log_loss"]["diff"], ev["vs_b"]["same_main_share"])
        res[name] = r
    res["audit"] = I.name_audit()
    return res


def coverage(x: pd.DataFrame, mains: dict) -> pd.DataFrame:
    m = R.matches()
    rows = []
    for u in R.UNIVERSES:
        b = R.baseline_b(u)
        for label, main in [("baseline_b", b)] + [(k[0], v) for k, v in mains.items() if k[1] == u]:
            rows.append({"universe": u, "method": label, "season_matches": int(len(m)),
                         "played": int(m["played"].astype(bool).sum()), "processed": int(main["match_id"].nunique()),
                         "with_real_price": int(main["odds"].notna().sum()), "bets": int(main["bet"].sum()),
                         "push_or_void": int(main["status"].isin(["PUSH", "VOID"]).sum()),
                         "no_price_not_a_bet": int((main["odds"].isna() & main["status"].isin(["WON", "LOST"])).sum()),
                         "unknown_result": int((main["status"] == "UNKNOWN").sum()),
                         "low_data_matches": int(len(MT.lowdata_ids(x))),
                         "newcomer_like_matches": int((x["hist_min"] < 10).sum()),
                         "matches_partial_stats": int((x["stats_missing"] > 0).sum())})
    return pd.DataFrame(rows).drop_duplicates(["universe", "method"])


def figures(x: pd.DataFrame, st: dict, cur: dict, hist_tab: pd.DataFrame, grp: dict) -> None:
    import matplotlib.pyplot as plt
    from scripts import report_kit as K
    K.plot_style()
    F = D["figures"]
    # 1. excess log loss and Main ROI by season progress (n_min bucket): current vs history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (key, col, lab) in enumerate((("current", "excess_close", "2026/27 (рынок closing)"),
                                         ("history", "excess_fd", "2024/25–2025/26 (Bet365)"))):
        d = st[key]["n_bucket"]
        ks = [k for k in Q.NMIN_LABELS if k in d]
        m = [d[k][col]["mean"] * 1000 for k in ks]
        lo = [(d[k][col]["mean"] - d[k][col]["ci95"][0]) * 1000 for k in ks]
        hi = [(d[k][col]["ci95"][1] - d[k][col]["mean"]) * 1000 for k in ks]
        ax[0].errorbar(np.arange(len(ks)) + (j - 0.5) * 0.15, m, yerr=[lo, hi], fmt="o-", capsize=3,
                       color=K.PALETTE[j], label=lab)
        ax[0].set_xticks(range(len(ks)), ks)
    ax[0].axhline(0, color="#888", lw=0.8)
    ax[0].set_xlabel("min. число матчей сезона у двух команд до матча")
    ax[0].set_ylabel("log loss модели − рынка, ·10⁻³")
    ax[0].legend(fontsize=7)
    d = st["current"]["n_bucket"]
    ks = [k for k in Q.NMIN_LABELS if k in d]
    for j, u in enumerate(("close", "t09")):
        r = [d[k][f"main_{u}"]["roi"] * 100 for k in ks]
        lo = [(d[k][f"main_{u}"]["roi"] - d[k][f"main_{u}"]["ci95"][0]) * 100 for k in ks]
        hi = [(d[k][f"main_{u}"]["ci95"][1] - d[k][f"main_{u}"]["roi"]) * 100 for k in ks]
        ax[1].errorbar(np.arange(len(ks)) + (j - 0.5) * 0.15, r, yerr=[lo, hi], fmt="s-", capsize=3,
                       color=K.PALETTE[j + 2], label=f"Baseline B, {u}")
    ax[1].set_xticks(range(len(ks)), ks)
    ax[1].axhline(0, color="#888", lw=0.8)
    ax[1].set_xlabel("min. число матчей сезона у двух команд до матча")
    ax[1].set_ylabel("ROI Main, %")
    ax[1].legend(fontsize=7)
    K.save_chart(fig, F / "season_progress.png")
    # 2. club status groups: excess log loss, current and history
    fig, ax = plt.subplots(figsize=(9, 3.6))
    keys = ["established | 6+", "established | 0–5", "moved_within | 6+", "moved_within | 0–5", "returner | 6+",
            "returner | 0–5", "outside | 6+", "outside | 0–5"]
    for j, (key, col, lab) in enumerate((("current", "excess_close", "2026/27"), ("history", "excess_fd", "2024/25–2025/26"))):
        d = st[key]["club_early"]
        ks = [k for k in keys if k in d]
        m = [d[k][col]["mean"] * 1000 for k in ks]
        lo = [(d[k][col]["mean"] - d[k][col]["ci95"][0]) * 1000 for k in ks]
        hi = [(d[k][col]["ci95"][1] - d[k][col]["mean"]) * 1000 for k in ks]
        ax.errorbar(np.arange(len(ks)) + (j - 0.5) * 0.2, m, yerr=[lo, hi], fmt="o", capsize=3, color=K.PALETTE[j], label=lab)
    ax.set_xticks(range(len(keys)), [k.replace(" | ", "\n") for k in keys], fontsize=7)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("log loss модели − рынка, ·10⁻³")
    ax.legend(fontsize=7)
    K.save_chart(fig, F / "club_status.png")
    # 3. mitigation results on 2026/27: dROI with CI (close, t09) and dLL
    labs = list(cur)
    fig, ax = plt.subplots(1, 2, figsize=(10, 0.45 * len(labs) + 1.2))
    for j, u in enumerate(("close", "t09")):
        dv = [cur[k]["universes"][u]["vs_b"]["roi_diff"] for k in labs]
        y = np.arange(len(labs)) + (j - 0.5) * 0.25
        ax[0].errorbar([v["diff"] * 100 for v in dv], y,
                       xerr=[[(v["diff"] - v["ci95"][0]) * 100 for v in dv], [(v["ci95"][1] - v["diff"]) * 100 for v in dv]],
                       fmt="o", capsize=2, color=K.PALETTE[j], label=u)
    ax[0].set_yticks(range(len(labs)), labs, fontsize=7)
    ax[0].axvline(0, color="#888", lw=0.8)
    ax[0].set_xlabel("ΔROI Main к Baseline B, п.п.")
    ax[0].legend(fontsize=7)
    dl = [cur[k]["universes"]["close"]["d_log_loss"] for k in labs]
    ax[1].errorbar([v["diff"] * 1000 for v in dl], range(len(labs)),
                   xerr=[[(v["diff"] - v["ci95"][0]) * 1000 for v in dl], [(v["ci95"][1] - v["diff"]) * 1000 for v in dl]],
                   fmt="o", capsize=2, color=K.PALETTE[3])
    ax[1].set_yticks(range(len(labs)), [""] * len(labs))
    ax[1].axvline(0, color="#888", lw=0.8)
    ax[1].set_xlabel("Δ log loss 1X2 (close), ·10⁻³")
    K.save_chart(fig, F / "mitigations_2026.png")
    # 4. feature-group importance in two seasons (result task)
    imp = grp["importance"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, task in zip(axs, ("result", "corners")):
        t = imp.loc[imp["task"] == task].pivot_table(index="group", columns="season", values="importance")
        t = t.sort_values(t.columns[-1])
        yy = np.arange(len(t))
        for j, c in enumerate(t.columns):
            ax.barh(yy + (j - 0.5) * 0.38, t[c] * 1000, height=0.36, color=K.PALETTE[j], label=c)
        ax.set_yticks(yy, t.index, fontsize=7)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("рост потерь при перестановке, ·10⁻³")
        ax.set_title("1X2 (log loss)" if task == "result" else "угловые (девиация Пуассона)", fontsize=9)
        ax.legend(fontsize=7)
    K.save_chart(fig, F / "feature_groups.png")
    # 5. history selection curve
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for j, fam in enumerate(("mkt_season", "mkt_hist")):
        t = hist_tab.loc[hist_tab["family"] == fam]
        ax.errorbar(t["w"].astype(float), t["d_log_loss"] * 1000,
                    yerr=[(t["d_log_loss"] - t["d_ll_lo"]) * 1000, (t["d_ll_hi"] - t["d_log_loss"]) * 1000],
                    fmt="o-", capsize=3, color=K.PALETTE[j], label=fam)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("вес w")
    ax.set_ylabel("Δ log loss 1X2, ·10⁻³ (2024/25 + 2025/26)")
    ax.legend(fontsize=7)
    K.save_chart(fig, F / "selection_history.png")


def decide(cur: dict) -> tuple[dict, dict]:
    """Decision per variant (rule fixed before looking at 2026/27):
    * only variants whose parameter was chosen on 2024/25 + 2025/26 (or that have none) can be A or B; the other
      grid points are C ("not selected on the earlier seasons");
    * A: ΔROI > 0 in both universes, Holm p < 0.05 (close), no model metric significantly worse;
    * B: a model metric of the changed outputs (1X2 log loss, goal / corner deviance) significantly better and ROI
      not lower in both universes, or ΔROI > 0 in both universes; and no model metric significantly worse;
    * C: otherwise."""
    out, why = {}, {}
    for k, v in cur.items():
        c, t = v["universes"]["close"], v["universes"]["t09"]
        dc, dt = c["vs_b"]["roi_diff"]["diff"], t["vs_b"]["roi_diff"]["diff"]
        mets = [c["d_log_loss"], c["d_goal_dev"], c["d_corner_dev"]]
        better = any(m["ci95"][1] is not None and m["ci95"][1] < 0 for m in mets)
        worse = any(m["ci95"][0] is not None and m["ci95"][0] > 0 for m in mets)
        if not v.get("chosen"):
            out[k], why[k] = "C", "not selected on 2024/25 + 2025/26"
        elif dc > 0 and dt > 0 and (c.get("holm_p") or 1) < 0.05 and not worse:
            out[k], why[k] = "A", "ROI better in both universes (Holm), metrics not worse"
        elif worse:
            out[k], why[k] = "C", "a model metric significantly worse"
        elif (better and not (dc < 0 and dt < 0)) or (dc > 0 and dt > 0):
            out[k], why[k] = "B", "metric better / ROI not lower; not significant for ROI"
        else:
            out[k], why[k] = "C", "no improvement"
    return out, why


def preregister(cur: dict, dec: dict) -> dict:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    pre = {}
    for k, d in dec.items():
        if d != "B":
            continue
        v = cur[k]
        pre[k] = {"hypothesis": f"Variant {k} lowers the 1X2 log loss (or goal / corner deviance) versus production and "
                                f"does not lower Main ROI; the gain concentrates in sparse-data matches.",
                  "rule": {"family": v["family"], "w": v["w"],
                           "s_season": "clip((10 - min(h_season_games, a_season_games))/10, 0, 1)",
                           "s_hist": "clip((10 - min(h_form10_n, a_form10_n))/10, 0, 1)",
                           "q": "de-margined (proportional) 1X2 price at the decision moment",
                           "low_data_threshold": MT.LOWDATA},
                  "primary_metric": "paired 1X2 log loss difference on all matches (prior: goal deviance); "
                                    "secondary: paired Main ROI difference (close)",
                  "minimum_forward_sample": "600 played matches including >= 150 low-data matches (start of 2027/28)",
                  "frozen_at": now}
    return pre


def redecide() -> None:
    """Re-apply the decision rule to the stored results (no recomputation)."""
    path = RES / "mitigations_2026.json"
    m = C.read_json(path)
    m["decisions"], m["decision_reasons"] = decide(m["variants"])
    C.write_json(path, m)
    C.write_json(RES / "preregistration.json", preregister(m["variants"], m["decisions"]))


def main() -> None:
    started = time.time()
    log = C.setup_logging("quality", D["logs"] / "run.log")
    x = MT.attach_priors(Q.frame_current())
    x.to_parquet(D["cache"] / "frame_current.parquet")
    xh, store = hist_frames(log)
    xh.to_parquet(D["cache"] / "frame_history.parquet")
    st = S.run(x, xh)
    C.write_json(RES / "strata.json", st)
    log.info("strata done")
    chosen, hist_tab = select_on_history(store, log)
    hist_tab.to_csv(RES / "selection_history.csv", index=False)
    C.write_json(RES / "selection.json", chosen)
    cur, mains = evaluate_current(x, chosen, log)
    dec, why = decide(cur)
    C.write_json(RES / "mitigations_2026.json", {"variants": cur, "decisions": dec, "decision_reasons": why,
                                                 "primary": PRIMARY,
                                                 "n_variants_tried": int(len(hist_tab)),
                                                 "n_variants_on_2026": int(len(cur))})
    idn = identification(chosen, log)
    C.write_json(RES / "identification.json", idn)
    grp = G.run()
    grp["importance"].to_csv(RES / "feature_groups_importance.csv", index=False)
    grp["missing"].to_csv(RES / "feature_groups_missing.csv", index=False)
    C.write_json(RES / "feature_groups.json", {"stability": grp["stability"]})
    # per-match tables and coverage
    res = R.results()
    for u in R.UNIVERSES:
        EV.per_match_table(R.baseline_b(u), res, f"baseline_b_{u}", EN.PRODUCTION_VERSION).to_csv(
            RES / f"per_match_baseline_b_{u}.csv", index=False)
        EV.per_match_table(mains[(PRIMARY, u)], res, f"{PRIMARY}_{u}", EN.PRODUCTION_VERSION + "+mkt_both").to_csv(
            RES / f"per_match_{PRIMARY}_{u}.csv", index=False)
    cov = coverage(x, mains)
    cov.to_csv(RES / "coverage.csv", index=False)
    strata_cols = ["match_id", "league", "date", "home_team", "away_team", "n_min", "hist_min", "dq_missing",
                   "stats_missing", "club_group", "h_status", "a_status", "tier", "oddspapi_at_t09", "n_priced_t09",
                   "ll", "ll_close", "excess_close", "dev_home", "dev_away", "dev_corners"]
    x[strata_cols].to_csv(RES / "match_strata.csv", index=False)
    figures(x, st, cur, hist_tab, grp)
    C.write_json(RES / "preregistration.json", preregister(cur, dec))
    self_check()
    C.write_json(RES / "run_info.json", C.run_info("quality", started, {
        "history_seasons": list(HIST), "grid": MT.GRID, "lowdata_threshold": MT.LOWDATA, "primary": PRIMARY,
        "chosen": chosen}, {"bootstrap": C.N_BOOT, "seed": C.SEED}))
    log.info("done in %.0fs", time.time() - started)


def self_check() -> dict:
    """oos.py 2025/26 predictions vs the xgb:prod configuration of the alternative-ML study (same model, same data)."""
    o = pd.read_parquet(D["cache"] / "oos_2025.parquet").set_index("match_id")
    path = C.V63 / "alt_ml" / "cache" / "valid_result.parquet"
    if not path.exists():
        return {"available": False}
    v = pd.read_parquet(path).set_index("match_id").join(o, how="inner")
    out = {"available": True, "matches": int(len(v)),
           "max_abs_diff_p_home": float(np.abs(v["xgb:prod|H"] - v["p_home"]).max())}
    C.write_json(RES / "self_check.json", out)
    return out


def groups_only() -> None:
    """Recompute the feature-group part only (after oos.py has saved the 2025/26 bundles)."""
    import sys as _s
    log = C.setup_logging("quality", D["logs"] / "run.log")
    grp = G.run()
    grp["importance"].to_csv(RES / "feature_groups_importance.csv", index=False)
    grp["missing"].to_csv(RES / "feature_groups_missing.csv", index=False)
    C.write_json(RES / "feature_groups.json", {"stability": grp["stability"]})
    x = pd.read_parquet(D["cache"] / "frame_current.parquet")
    st = C.read_json(RES / "strata.json")
    cur = C.read_json(RES / "mitigations_2026.json")["variants"]
    hist_tab = pd.read_csv(RES / "selection_history.csv")
    figures(x, st, cur, hist_tab, grp)
    log.info("feature groups recomputed: %s", grp["stability"])


if __name__ == "__main__":
    import sys
    {"groups": groups_only, "redecide": redecide, "selfcheck": self_check}.get((sys.argv[1:] or ["main"])[0], main)()
