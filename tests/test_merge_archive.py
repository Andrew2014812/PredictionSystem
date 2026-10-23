import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("merge_history_archive",
                                              Path(__file__).parent.parent / "scripts/merge_history_archive.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _odds(rows):
    base = {"provider": "oddspapi", "bookmaker": "bet365", "line": None, "provider_fixture_id": "f",
            "priced_at": pd.NaT, "fetched_at": pd.NaT, "kickoff": pd.NaT}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    return df


def test_archive_is_added_to_newer_data_without_losing_either(tmp_path, monkeypatch):
    files = {k: tmp_path / "data" / v.name for k, v in mod.FILES.items()}
    monkeypatch.setattr(mod, "FILES", files)
    (tmp_path / "data").mkdir()
    # the download's view: one archived price, its ledger and a learned id
    _odds([{"match_id": "old", "market": "1X2", "selection": "H", "odds": 2.0, "collected_at": "2026-09-01"}]
          ).to_parquet(files["odds"])
    files["ledger"].write_text(json.dumps({"requests": {"historical|f1": {"status": 200}}, "month": "2026-10"}))
    files["ids"].write_text(json.dumps({"oddspapi": {"1": "Spain|Espanol"}}))
    files["progress"].write_text(json.dumps({"fixtures": {"SP1": {}}}))
    mod.save(tmp_path / "saved")
    # meanwhile the daily run committed newer data
    _odds([{"match_id": "new", "market": "1X2", "selection": "A", "odds": 3.1, "collected_at": "2026-10-23"},
           {"match_id": "old", "market": "1X2", "selection": "H", "odds": 2.0, "collected_at": "2026-09-01"}]
          ).to_parquet(files["odds"])
    files["ledger"].write_text(json.dumps({"requests": {"fixtures|all": {"status": 200}}, "month": "2026-10"}))
    files["ids"].write_text(json.dumps({"oddspapi": {"1": "Spain|Espanol", "2": "Spain|Barcelona"}}))
    files["progress"].unlink()
    mod.merge(tmp_path / "saved")
    odds = pd.read_parquet(files["odds"])
    assert sorted(odds["match_id"]) == ["new", "old"]                      # union, the shared quote once
    ledger = json.loads(files["ledger"].read_text())
    assert set(ledger["requests"]) == {"historical|f1", "fixtures|all"}
    assert json.loads(files["ids"].read_text())["oddspapi"] == {"1": "Spain|Espanol", "2": "Spain|Barcelona"}
    assert json.loads(files["progress"].read_text()) == {"fixtures": {"SP1": {}}}
