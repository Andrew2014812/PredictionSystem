"""Shadow predictions (src/research): append-only, frozen at kick-off, isolated from production, settled once,
idempotent on a re-run, interim-safe evaluation, and faithful to the frozen preregistrations."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.markets.distributions import reconcile_with_1x2, score_matrix
from src.markets.markets import goal_markets
from src.markets.odds import price
from src.markets.recommendation import main_prediction, risk_prediction
from src.research import evaluate as EV
from src.research import registry as R
from src.research import shadow as SH
from src.research import variants as V

NOW = pd.Timestamp("2026-11-20 08:00:00")            # UK time of the "daily run"
CHEAP = ["maincal_ps_platt_monthly", "maincal_ps_group", "risk_stacked", "risk_market_probability",
         "bookmaker_h2_pool"]


# ------------------------------------------------------------------------------------------------ synthetic store
def _match(i, kickoff, source="live", lam=(1.5, 1.1), p=(0.47, 0.27, 0.26), priced=True, gen=None):
    mid = f"E0-{kickoff:%Y%m%d}-home{i}-away{i}"
    matrix = reconcile_with_1x2(score_matrix(*lam), np.array(p))
    sels = goal_markets(matrix)
    quotes = {}
    if priced:
        for s in sels:
            if s.market in ("1X2", "TOTAL_2.5", "TOTAL_1.5", "TOTAL_3.5", "BTTS", "HANDICAP_-0.5", "HANDICAP_-1",
                            "HANDICAP_+0.5"):
                quotes[(s.market, s.selection)] = {"odds": round(0.95 / max(s.probability, 0.05), 2),
                                                   "bookmaker": "Bet365", "provider": "oddspapi",
                                                   "priced_at": kickoff - pd.Timedelta(hours=20)}
    priced_sels = price(sels, quotes)
    main = main_prediction(priced_sels)
    risk = risk_prediction(priced_sels, main.pick)
    gen = gen or (NOW - pd.Timedelta(minutes=5))
    snap = {"match_id": mid, "league": "E0", "country": "England", "season": 2026, "date": kickoff.normalize(),
            "time": f"{kickoff:%H:%M}", "kickoff": kickoff, "home_team": f"Home{i}", "away_team": f"Away{i}",
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "xg_home": lam[0], "xg_away": lam[1],
            "model_version": "prod_test", "source": source, "generated_at": gen, "version": 1}
    mk = [{"match_id": mid, **s.to_dict()} for s in priced_sels]
    picks = []
    for role, rec in (("main", main), ("risk", risk)):
        s = rec.pick
        picks.append({"pick_id": f"{mid}|{role}|{s.market}|{s.selection}", "match_id": mid, "league": "E0",
                      "season": 2026, "date": kickoff.normalize(), "time": f"{kickoff:%H:%M}",
                      "home_team": f"Home{i}", "away_team": f"Away{i}", "role": role, "kind": rec.kind,
                      "market": s.market, "group": s.group, "selection": s.selection, "probability": s.probability,
                      "odds": s.odds if s.has_odds else np.nan, "bookmaker": s.bookmaker, "provider": s.provider,
                      "odds_priced_at": pd.Timestamp(s.priced_at) if s.has_odds else pd.NaT, "status": "UPCOMING",
                      "home_goals": np.nan, "away_goals": np.nan, "total_corners": np.nan, "profit": np.nan,
                      "model_version": "prod_test", "source": source, "generated_at": gen})
    return snap, mk, picks


def _history(n=120, month="2026-10"):
    """Settled Mains of an earlier month (for the monthly Platt fit)."""
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        day = pd.Timestamp(f"{month}-01") + pd.Timedelta(days=int(k % 28))
        p = float(rng.uniform(0.45, 0.8))
        won = rng.uniform() < p - 0.05
        mid = f"E1-{day:%Y%m%d}-h{k}-a{k}"
        rows.append({"pick_id": f"{mid}|main|1X2|H", "match_id": mid, "league": "E1", "season": 2026, "date": day,
                     "time": "15:00", "home_team": f"H{k}", "away_team": f"A{k}", "role": "main", "kind": "priced",
                     "market": "1X2", "group": "1X2", "selection": "H", "probability": p, "odds": 1 / p,
                     "bookmaker": "Bet365", "provider": "oddspapi", "odds_priced_at": day, "status":
                     "WON" if won else "LOST", "home_goals": 1.0 if won else 0.0, "away_goals": 0.0 if won else 1.0,
                     "total_corners": np.nan, "profit": (1 / p - 1) if won else -1.0, "model_version": "prod_test",
                     "source": "live", "generated_at": day - pd.Timedelta(hours=8)})
    return rows


def _write_store(folder: Path, matches, history=()):
    folder.mkdir(parents=True, exist_ok=True)
    snaps = pd.DataFrame([m[0] for m in matches])
    mk = pd.DataFrame([r for m in matches for r in m[1]])
    mk["priced_at"] = pd.to_datetime(mk["priced_at"])
    picks = pd.DataFrame([r for m in matches for r in m[2]] + list(history))
    snaps.to_parquet(folder / "snapshots.parquet", index=False)
    mk.to_parquet(folder / "markets.parquet", index=False)
    picks.to_parquet(folder / "picks.parquet", index=False)
    return snaps, mk, picks


def _features(matches, results=None):
    results = results or {}
    rows = []
    for snap, _, _ in matches:
        hg, ag = results.get(snap["match_id"], (np.nan, np.nan))
        rows.append({"match_id": snap["match_id"], "league": "E0", "date": snap["date"], "kickoff": snap["kickoff"],
                     "home_team": snap["home_team"], "away_team": snap["away_team"],
                     "played": not np.isnan(hg), "home_goals": hg, "away_goals": ag, "total_corners": np.nan,
                     "odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.6, "odds_over25": 1.9, "odds_under25": 1.95})
    return pd.DataFrame(rows)


def _hashes(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*"))}


@pytest.fixture
def store(tmp_path):
    matches = [_match(1, NOW + pd.Timedelta(days=1)),
               _match(2, NOW + pd.Timedelta(hours=3), lam=(1.2, 1.4), p=(0.33, 0.29, 0.38)),
               _match(3, NOW - pd.Timedelta(hours=1)),                           # already started
               _match(4, NOW + pd.Timedelta(days=2), source="backfill")]           # not a pre-match record
    prod = tmp_path / "prod"
    _write_store(prod, matches, _history())
    return {"prod": prod, "root": tmp_path / "research", "matches": matches}


def _record(store, now=NOW, features=None, **kw):
    feats = _features(store["matches"]) if features is None else features
    return SH.record_shadow_predictions(feats, store["prod"], now=now, root=store["root"], shadows=CHEAP, **kw)


# ------------------------------------------------------------------------------------------------ tests
def test_records_only_upcoming_live_predictions(store):
    s = _record(store)
    assert not s["errors"], s["errors"]
    log = SH.load_predictions(store["root"])
    ids = set(log["match_id"])
    assert ids == {store["matches"][0][0]["match_id"], store["matches"][1][0]["match_id"]}
    assert (pd.to_datetime(log["recorded_at"]) < pd.to_datetime(log["kickoff"])).all()
    assert set(log["shadow_id"]) == set(CHEAP)
    # every row names the frozen preregistration it implements
    for key, g in log.groupby("shadow_key"):
        spec = R.get(g["shadow_id"].iloc[0])
        assert (g["prereg_sha256"] == spec.prereg_sha256).all() and (g["freeze_sha"] == spec.freeze_sha).all()


def test_rerun_on_the_same_day_appends_nothing(store):
    first = _record(store)
    second = _record(store, now=NOW + pd.Timedelta(minutes=30))
    assert first["rows_appended"] > 0 and second["rows_appended"] == 0
    manifest = SH.load_manifest(store["root"])
    assert len(manifest["runs"]) == 2 and manifest["runs"][1]["rows_appended"] == 0
    assert SH.verify_log(store["root"])["ok"]


def test_append_only_and_frozen_after_kickoff(store):
    _record(store)
    before = SH.load_predictions(store["root"])
    # production refreshes both upcoming matches with new probabilities; match 2 then kicks off
    m1, m2 = store["matches"][0][0]["match_id"], store["matches"][1][0]["match_id"]
    refreshed = [_match(1, NOW + pd.Timedelta(days=1), p=(0.55, 0.25, 0.20), gen=NOW + pd.Timedelta(hours=4)),
                 _match(2, NOW + pd.Timedelta(hours=3), lam=(1.2, 1.4), p=(0.20, 0.25, 0.55),
                        gen=NOW + pd.Timedelta(hours=4))]
    _write_store(store["prod"], refreshed + store["matches"][2:], _history())
    _record(store, now=NOW + pd.Timedelta(hours=5))                     # after match 2's kick-off
    after = SH.load_predictions(store["root"])
    pd.testing.assert_frame_equal(after.iloc[:len(before)].reset_index(drop=True), before.reset_index(drop=True))
    new = after.iloc[len(before):]
    assert m2 not in set(new["match_id"])                              # never re-recorded after kick-off
    assert m1 in set(new["match_id"])                                  # refreshed before its kick-off
    final = EV.final_records(after, R.get("maincal_ps_group"), "main")
    assert set(final["match_id"]) <= {m1, m2} or final.empty            # (forward rule may exclude test dates)
    assert SH.verify_log(store["root"])["ok"]


def test_production_tables_are_never_written(store):
    before = _hashes(store["prod"])
    _record(store)
    feats = _features(store["matches"], {store["matches"][2][0]["match_id"]: (2.0, 1.0)})
    _record(store, now=NOW + pd.Timedelta(days=3), features=feats)
    assert _hashes(store["prod"]) == before
    written = {p.relative_to(store["root"]).parts[0] for p in store["root"].rglob("*") if p.is_file()}
    assert written <= {SH.PREDICTIONS, SH.RESULTS, SH.MANIFEST, SH.REPORT}
    # every run writes NEW files only: the files of the first run are byte-identical afterwards
    assert SH.verify_log(store["root"])["ok"]


def test_settlement_is_appended_once_and_corrections_are_new_rows(store):
    _record(store)
    m1 = store["matches"][0][0]["match_id"]
    later = NOW + pd.Timedelta(days=2)
    feats = _features(store["matches"], {m1: (2.0, 1.0)})
    assert SH.settle_shadow(feats, store["root"], later)["results_appended"] == 1
    assert SH.settle_shadow(feats, store["root"], later)["results_appended"] == 0
    fixed = _features(store["matches"], {m1: (2.0, 2.0)})
    res = SH.settle_shadow(fixed, store["root"], later + pd.Timedelta(days=1))
    assert (res["results_appended"], res["corrections"]) == (1, 1)
    assert SH.verify_log(store["root"])["ok"]
    results = SH.load_results(store["root"])
    assert len(results) == 2
    merged = EV.with_results(SH.load_predictions(store["root"]).query("match_id == @m1"), results)
    assert set(merged["away_goals"]) == {2.0}                          # the latest result counts
    # a result before kick-off is never taken
    early = _features(store["matches"], {store["matches"][1][0]["match_id"]: (1.0, 0.0)})
    assert SH.settle_shadow(early, store["root"], NOW)["results_appended"] == 0


def test_tampering_breaks_the_chain(store):
    _record(store)
    path = sorted((store["root"] / SH.PREDICTIONS).glob("*.parquet"))[0]
    df = pd.read_parquet(path)
    df.loc[0, "p"] = 0.99
    df.to_parquet(path, index=False)
    check = SH.verify_log(store["root"])
    assert not check["ok"] and check["problems"]


def test_a_deleted_or_extra_file_is_detected(store):
    _record(store)
    files = sorted((store["root"] / SH.PREDICTIONS).glob("*.parquet"))
    extra = store["root"] / SH.PREDICTIONS / "20990101T000000.000000Z.parquet"
    pd.read_parquet(files[0]).to_parquet(extra, index=False)
    assert not SH.verify_log(store["root"])["ok"]
    extra.unlink()
    files[0].unlink()
    assert not SH.verify_log(store["root"])["ok"]


def test_changed_preregistration_is_refused(store, monkeypatch):
    bad = replace(R.get("maincal_ps_group"), prereg_sha256="0" * 64)
    monkeypatch.setattr(R, "SHADOWS", tuple(bad if s.id == "maincal_ps_group" else s for s in R.SHADOWS))
    s = _record(store)
    assert "maincal_ps_group@v1" in s["errors"]
    assert "maincal_ps_group" not in set(SH.load_predictions(store["root"])["shadow_id"])


def test_registry_entry_cannot_change_after_registration(store, monkeypatch):
    _record(store)
    changed = replace(R.get("risk_stacked"), min_n=10)
    monkeypatch.setattr(R, "SHADOWS", tuple(changed if s.id == "risk_stacked" else s for s in R.SHADOWS))
    s = _record(store, now=NOW + pd.Timedelta(hours=1))
    assert "risk_stacked@v1" in s["errors"]


def test_never_raises_into_the_pipeline(store):
    s = SH.record_shadow_predictions(None, store["root"] / "missing", now=NOW, root=store["root"])
    assert "run" in s["errors"] or s["rows_appended"] == 0


def test_preregistered_values(store):
    _record(store)
    log = SH.load_predictions(store["root"])
    doc = R.get("maincal_ps_group").document()
    params = doc["hypotheses"]["H2_challenger"]["frozen_parameters"]
    for r in log.loc[log["shadow_id"] == "maincal_ps_group"].itertuples():
        pl = json.loads(r.payload)
        g = pl["group"] if pl["group"] in params else "pool"
        expect = V.sigmoid(params[g]["a"] + params[g]["b"] * V.logit(pl["pc"]))
        assert r.p == pytest.approx(float(expect)) and r.market == r.base_market
    platt = log.loc[log["shadow_id"] == "maincal_ps_platt_monthly"]
    assert platt["p"].notna().all()
    assert all(json.loads(p)["n"] == 120 for p in platt["payload"])      # fitted on October only
    risk = log.loc[log["shadow_id"] == "risk_stacked"]
    for r in risk.itertuples():
        pl = json.loads(r.payload)
        if pl["tier"] == 2:
            assert (r.market, r.selection) == (r.base_market, r.base_selection)
        else:
            assert pl["ev_stacked"] >= 0 and r.odds > 1
    mkt = log.loc[log["shadow_id"] == "risk_market_probability"]
    assert ((mkt["p"] > 0) & (mkt["p"] < 1) | mkt["p"].isna()).all()


def test_interim_reports_hide_tests_and_decisions_are_frozen(store, monkeypatch):
    forward = {"kind": "date_from", "date": "2026-01-01"}
    monkeypatch.setattr(R, "SHADOWS", tuple(replace(s, forward_rule=forward) for s in R.SHADOWS))
    _record(store)
    key = "maincal_ps_group@v1"
    m1, m2 = store["matches"][0][0]["match_id"], store["matches"][1][0]["match_id"]
    rep = EV.evaluate_shadow(store["root"], now=NOW + pd.Timedelta(hours=1), write=True)
    assert rep["shadows"][key]["status"] == "no_forward_data" and "test" not in rep["shadows"][key]
    feats = _features(store["matches"], {m2: (0.0, 5.0), m1: (5.0, 0.0)})
    SH.settle_shadow(feats, store["root"], NOW + pd.Timedelta(days=2))
    g = EV.evaluate_shadow(store["root"], now=NOW + pd.Timedelta(days=2), write=True)["shadows"][key]
    assert g["n"] >= 1 and g["status"] == "accumulating"
    assert "test" not in g and "point" in g and "p_for_holm" not in g     # interim: no interval, no p-value
    # the same data once the preregistered N is (artificially) reached: tested once, then frozen
    monkeypatch.setattr(R, "SHADOWS", tuple(replace(s, forward_rule=forward, min_n=1) for s in R.SHADOWS))
    g = EV.evaluate_shadow(store["root"], now=NOW + pd.Timedelta(days=2), write=True)["shadows"][key]
    assert g["ready"] and "test" in g
    assert key in SH.load_manifest(store["root"])["decisions"]
    feats = _features(store["matches"], {m2: (5.0, 0.0), m1: (0.0, 5.0)})      # later data would flip results
    SH.settle_shadow(feats, store["root"], NOW + pd.Timedelta(days=3))
    again = EV.evaluate_shadow(store["root"], now=NOW + pd.Timedelta(days=3), write=True)["shadows"][key]
    assert again["frozen"] and again["test"] == g["test"]


def test_bookmaker_pool_uses_frozen_theta():
    doc = R.get("bookmaker_h2_pool").document()
    theta = doc["frozen_variant"]["theta_ou25"]
    q = V.power_demargin(np.array([[1.9, 1.95]]))
    assert q.sum() == pytest.approx(1.0)
    p = np.array([[0.55, 0.45]])
    h2 = V.pool_predict(theta, q, p)
    z = np.array([0.0, theta[0]]) + theta[1] * np.log(q) + theta[2] * np.log(p)
    assert h2 == pytest.approx(np.exp(z) / np.exp(z).sum())


def test_x5_falls_back_to_production_without_xg(store):
    feats = _features(store["matches"])
    xg = pd.DataFrame(columns=["match_id", "league", "date", "home_team", "away_team", "home_goals", "away_goals",
                               "xg_home", "xg_away"])
    s = SH.record_shadow_predictions(feats, store["prod"], now=NOW, root=store["root"], shadows=["xg_x5"],
                                     context_extra={"xg_table": xg})
    assert not s["errors"]
    assert s["recorded"]["xg_x5@v1"]["rows"] == 0                        # no xG league: nothing to shadow


def test_x5_with_xg_keeps_production_main_when_w_is_zero(store):
    rng = np.random.default_rng(3)
    rows = []
    for k in range(40):
        day = NOW.normalize() - pd.Timedelta(days=40 - k)
        h, a = f"Home{k % 4 + 1}", f"Away{(k + 1) % 4 + 1}"
        rows.append({"match_id": f"E0-x{k}", "league": "E0", "date": day, "home_team": h, "away_team": a,
                     "home_goals": float(rng.poisson(1.4)), "away_goals": float(rng.poisson(1.1)),
                     "xg_home": float(rng.uniform(0.5, 2.5)), "xg_away": float(rng.uniform(0.4, 2.0))})
    xg = pd.DataFrame(rows)
    from src.research import xg_x5 as X
    e = X.expected_xg(xg, pd.DataFrame([{k: m[0][k] for k in ("match_id", "league", "date", "home_team",
                                                              "away_team")} for m in store["matches"][:2]]))
    assert e["exp_xg_h"].notna().all() and (e["league_xg_matches"] == 40).all()
    s = SH.record_shadow_predictions(_features(store["matches"]), store["prod"], now=NOW, root=store["root"],
                                     shadows=["xg_x5"], context_extra={"xg_table": xg})
    assert not s["errors"]
    log = SH.load_predictions(store["root"])
    goals = log.loc[log["item"] == "goals"]
    for pl in map(json.loads, goals["payload"]):
        assert pl["fallback"]                     # lambdas of earlier matches are unknown here: w cannot be fitted
        assert pl["mu"] == pytest.approx(pl["lam"])
    main = log.loc[log["item"] == "main"]
    assert all(json.loads(p)["same_as_production"] for p in main["payload"])


# ------------------------------------------------------------------------------------------------ real data checks
REAL = config.PREDICTIONS_DIR / "picks.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="no production history")
def test_monthly_platt_reproduces_the_preregistered_fit():
    doc = R.get("maincal_ps_platt_monthly").document()
    info = doc["hypotheses"]["H1_primary"]["current_fit_for_information"]
    picks = pd.read_parquet(REAL)
    if len(picks) != doc["data_snapshot"]["data_version"]["picks"]["rows"]:
        pytest.skip("production history changed since the preregistration")
    mk = pd.read_parquet(config.PREDICTIONS_DIR / "markets.parquet",
                         columns=["match_id", "market", "selection", "push_probability"])
    mk = mk.drop_duplicates(["match_id", "market", "selection"], keep="last")
    fit = V.monthly_platt_fit(picks, mk, pd.Timestamp(info["last_match_date"]) + pd.Timedelta(days=1), 2026)
    # the population is production's unified_history; commit 466aab5 (a match moved to an earlier day keeps its
    # played date) changed it by one Main after the study computed this fit, so allow that one match
    assert abs(fit["n"] - info["fitted_on_settled_shown_mains"]) <= 1
    assert fit["a"] == pytest.approx(info["a"], abs=0.01) and fit["b"] == pytest.approx(info["b"], abs=0.01)


@pytest.mark.skipif(not (config.RESULTS_DIR / "2627").exists(), reason="no 2026/27 result files")
def test_x5_weekly_w_reproduces_the_preregistration():
    from src.research import xg_x5 as X
    doc = R.get("xg_x5").document()
    xg = X.load_xg()
    snaps = pd.read_parquet(config.PREDICTIONS_DIR / "snapshots.parquet", columns=["match_id", "xg_home", "xg_away"])
    lam = snaps.rename(columns={"xg_home": "lam_h", "xg_away": "lam_a"})
    w = X.fit_w(xg, lam, pd.Timestamp("2026-09-08"))["w"]
    assert w == pytest.approx(doc["primary"]["w_history"][1], abs=1e-6)


def test_an_unchanged_production_refresh_adds_no_row(store):
    _record(store)
    later = NOW + pd.Timedelta(hours=1)
    same = [_match(1, NOW + pd.Timedelta(days=1), gen=later),
            _match(2, NOW + pd.Timedelta(hours=3), lam=(1.2, 1.4), p=(0.33, 0.29, 0.38), gen=later)]
    _write_store(store["prod"], same + store["matches"][2:], _history())
    assert _record(store, now=later + pd.Timedelta(minutes=1))["rows_appended"] == 0


def test_every_registered_preregistration_is_the_frozen_file():
    for spec in R.SHADOWS:
        assert spec.check() is None, spec.key
        assert spec.implemented or spec.not_implemented_reason
        assert len({s.key for s in R.SHADOWS}) == len(R.SHADOWS)
