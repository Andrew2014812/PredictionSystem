"""Local file storage (CSV, Parquet, joblib, JSON) under ``local_data/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import config


class LocalStorage:
    """Thin wrapper over a directory that reads and writes typed files.

    Paths are relative to ``root`` and may include sub-directories, e.g.
    ``storage.read_csv("2526/E0.csv")``. Missing parent directories are
    created on write.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def writable_path(self, name: str) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # -- tables -----------------------------------------------------------
    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.path(name), **kwargs)

    def write_csv(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.writable_path(name), index=False)

    def read_parquet(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(name))

    def write_parquet(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.writable_path(name), index=False)

    def read_parquet_or_empty(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        if self.exists(name):
            return self.read_parquet(name)
        return pd.DataFrame(columns=columns or [])

    # -- objects ----------------------------------------------------------
    def read_object(self, name: str) -> Any:
        return joblib.load(self.path(name))

    def write_object(self, name: str, obj: Any) -> None:
        joblib.dump(obj, self.writable_path(name))

    def read_json(self, name: str) -> Any:
        with open(self.path(name), encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, name: str, obj: Any) -> None:
        with open(self.writable_path(name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)

    def list(self, pattern: str = "*") -> list[Path]:
        return sorted(self.root.glob(pattern)) if self.root.exists() else []


results_storage = LocalStorage(config.RESULTS_DIR)
fixtures_storage = LocalStorage(config.FIXTURES_DIR)
processed_storage = LocalStorage(config.PROCESSED_DIR)
models_storage = LocalStorage(config.MODELS_DIR)
predictions_storage = LocalStorage(config.PREDICTIONS_DIR)
