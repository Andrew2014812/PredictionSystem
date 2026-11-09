"""Inputs of the xG study: raw xG from the Football-Data files, the 2026/27 matches with the production
(v6.1) predictions, and the real pre-match prices of every selection.

* xG: columns ``HxG`` / ``AxG`` of ``local_data/results/<season>/<league>.csv``. They are joined to the match
  table by the production ``match_id`` (league + date + teams, ``src.data.cleaning``). Earlier seasons are
  scanned only to confirm that they have no xG columns.
* Baseline B0: the active production model set (``prod_20261012``, trained on 2021/22-2025/26, i.e. before
  2026/27 started) applied to the stored pre-match features of every 2026/27 match. Nothing is refitted.
* Prices: the price stored with the production markets of the match (``markets.parquet``, latest version;
  Football-Data "Market average" is not a bookmaker price and is dropped), completed by the last OddsPapi
  quote strictly before kick-off (``src.prediction.history_repair.archive_prices``) — the v6.1 rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

RAW_COLUMNS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HxG", "AxG", "HS", "AS", "HST", "AST"]
ID_COLS = ["match_id", "league", "season", "date", "time", "kickoff", "home_team", "away_team", "played",
           "home_goals", "away_goals", "total_corners", "home_shots", "away_shots", "home_sot", "away_sot"]


# ------------------------------------------------------------------------------------------------ raw xG
def xg_columns_by_season() -> pd.DataFrame:
    """Which season files carry HxG / AxG at all (header scan of every local file)."""
    from src.data.dataset import _read_raw_csv
    rows = []
    for path in sorted((C.ROOT / "local_data/results").glob("*/*.csv")):
        head = _read_raw_csv(path).columns
        rows.append({"season_code": path.parent.name, "league": path.stem,
                     "has_hxg": "HxG" in head, "has_axg": "AxG" in head})
    return pd.DataFrame(rows)


def raw_xg(season_code: str = "2627") -> pd.DataFrame:
    """One row per match of the season files with the production match_id and the raw xG values."""
    from src.data.cleaning import _parse_dates, normalise
    from src.data.dataset import _read_raw_csv
    frames = []
    for path in sorted((C.ROOT / "local_data/results" / season_code).glob("*.csv")):
        raw = _read_raw_csv(path)
        raw = raw.loc[raw["HomeTeam"].notna() & raw["AwayTeam"].notna() & raw["Date"].notna()].reset_index(drop=True)
        norm = normalise(raw)
        if len(norm) != len(raw):            # normalise drops undated rows; keep the alignment explicit
            raw = raw.loc[_parse_dates(raw["Date"].astype(str).str.strip()).notna()].reset_index(drop=True)
        assert len(norm) == len(raw), path
        out = norm[["match_id", "league", "season", "date", "time", "kickoff", "home_team", "away_team", "played",
                    "home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot"]].copy()
        out["has_xg_column"] = "HxG" in raw.columns
        out["xg_home"] = pd.to_numeric(raw["HxG"], errors="coerce").to_numpy() if "HxG" in raw else np.nan
        out["xg_away"] = pd.to_numeric(raw["AxG"], errors="coerce").to_numpy() if "AxG" in raw else np.nan
        out["xg_home_raw"] = raw["HxG"].astype(str).to_numpy() if "HxG" in raw else None
        out["xg_away_raw"] = raw["AxG"].astype(str).to_numpy() if "AxG" in raw else None
        out["file"] = path.relative_to(C.ROOT).as_posix()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def enabled_leagues() -> list[str]:
    from src.leagues import league_codes
    return list(league_codes())


def clean_xg(raw: pd.DataFrame) -> pd.DataFrame:
    """xG usable as data: enabled league, both values present, finite, 0 <= xG <= 8, goals present."""
    ok = (raw["league"].isin(enabled_leagues()) & raw["xg_home"].notna() & raw["xg_away"].notna()
          & raw["xg_home"].between(0, 8) & raw["xg_away"].between(0, 8) & raw["played"].astype(bool))
    return raw.loc[ok, ["match_id", "league", "date", "home_team", "away_team", "home_goals", "away_goals",
                        "home_shots", "away_shots", "home_sot", "away_sot", "xg_home", "xg_away"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ matches + B0
def season_features(season: int = C.SEASON, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(C.ROOT / "local_data/processed/features.parquet", columns=columns,
                         filters=[("season", "==", season)])
    return df.reset_index(drop=True)


def baseline(refresh: bool = False) -> pd.DataFrame:
    """B0 = production v6.1 predictions for every 2026/27 match (cached)."""
    path = C.CACHE / "baseline_2026.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from src.modelling.registry import load_bundles
    b = load_bundles(C.PRODUCTION_VERSION)
    df = season_features()
    out = df[ID_COLS].copy()
    p = b["result"].predict(df)
    out["pH"], out["pD"], out["pA"] = p[:, 0], p[:, 1], p[:, 2]
    out["lh"] = b["home_goals"].predict(df)
    out["la"] = b["away_goals"].predict(df)
    cl = set(b["corners"].extra.get("leagues", []))
    out["cmu"] = np.where(df["league"].isin(cl), b["corners"].predict(df), np.nan)
    out["nb_size"] = b["corners"].extra.get("nb_size")
    out["y"] = np.select([out["home_goals"] > out["away_goals"], out["home_goals"] == out["away_goals"]], [0, 1], 2)
    out.loc[~out["played"].astype(bool), "y"] = -1
    out.to_parquet(path)
    return out


def stored_snapshots() -> pd.DataFrame:
    s = pd.read_parquet(C.ROOT / "local_data/predictions/snapshots.parquet",
                        columns=["match_id", "season", "p_home", "p_draw", "p_away", "xg_home", "xg_away",
                                 "model_version", "source", "main_market", "main_selection"])
    return s.loc[s["season"] == C.SEASON].drop_duplicates("match_id", keep="last")


# ------------------------------------------------------------------------------------------------ prices
def prices(match_ids, refresh: bool = False) -> pd.DataFrame:
    """Real pre-match price per (match_id, market, selection) for 2026/27 (cached)."""
    path = C.CACHE / "prices_2026.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from src.prediction.history_repair import archive_prices
    ids = set(match_ids)
    mk = pd.read_parquet(C.ROOT / "local_data/predictions/markets.parquet",
                         columns=["match_id", "market", "selection", "odds", "bookmaker", "provider", "priced_at"])
    mk = mk.loc[mk["match_id"].isin(ids)].drop_duplicates(["match_id", "market", "selection"], keep="last").copy()
    avg = mk["bookmaker"] == "Market average"
    mk.loc[avg, ["odds", "bookmaker", "provider"]] = [np.nan, None, None]
    mk["odds"] = mk["odds"].astype(float)
    mk["origin"] = np.where(mk["odds"].notna(), "recorded", None)
    feats = season_features(columns=["match_id", "kickoff"])
    need = mk.loc[mk["odds"].isna(), ["match_id", "market", "selection"]].assign(source="live", odds=np.nan)
    arch = archive_prices(need, feats.loc[feats["match_id"].isin(ids)])
    if not arch.empty:
        a = arch.set_index(["match_id", "market", "selection"])
        key = pd.MultiIndex.from_frame(mk[["match_id", "market", "selection"]])
        pos = a.index.get_indexer(key)
        fill = (pos >= 0) & mk["odds"].isna().to_numpy()
        mk.loc[fill, "odds"] = a["odds"].to_numpy(float)[pos[fill]]
        mk.loc[fill, "bookmaker"] = a["bookmaker"].to_numpy()[pos[fill]]
        mk.loc[fill, "provider"] = "oddspapi"
        mk.loc[fill, "priced_at"] = pd.to_datetime(a["collected_at"].to_numpy()[pos[fill]])
        mk.loc[fill, "origin"] = "archive"
    mk = mk.loc[mk["odds"].notna() & (mk["odds"] > 1.0)]
    out = mk[["match_id", "market", "selection", "odds", "bookmaker", "provider", "origin"]].reset_index(drop=True)
    out["avg_dropped"] = False
    out.to_parquet(path)
    C.save_json("prices_info.json", {"rows": int(len(out)), "matches": int(out["match_id"].nunique()),
                                     "market_average_dropped": int(avg.sum()),
                                     "origin": out["origin"].value_counts().to_dict(),
                                     "provider": out["provider"].fillna("?").value_counts().to_dict()})
    return out


def price_lookup(px: pd.DataFrame) -> dict:
    """match_id -> {(market, selection): {"odds", "bookmaker", "provider"}}."""
    out: dict[str, dict] = {}
    for r in px.itertuples(index=False):
        out.setdefault(r.match_id, {})[(r.market, r.selection)] = {"odds": float(r.odds), "bookmaker": r.bookmaker,
                                                                   "provider": r.provider}
    return out


def history_features(columns: list[str], seasons=None) -> pd.DataFrame:
    """Feature rows of earlier seasons (only the needed columns)."""
    filt = [("season", "in", list(seasons))] if seasons else None
    return pd.read_parquet(C.ROOT / "local_data/processed/features.parquet", columns=columns, filters=filt)


def exists(path: Path) -> bool:
    return Path(path).exists()
