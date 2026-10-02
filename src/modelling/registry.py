"""Saving and loading versioned model sets.

Layout::

    local_data/models/
        metadata.json                 # {"active": <version>, "versions": {...}}
        <version>/
            result_model.pkl
            home_goals_model.pkl
            away_goals_model.pkl
            corners_model.pkl
            metadata.json             # periods, params, metrics, features, SHAP summary

Older versions stay on disk because stored predictions reference the model
version that produced them (and the UI recomputes SHAP with that model).
"""
from __future__ import annotations

from functools import lru_cache

from ..storage import models_storage
from .tasks import TASKS
from .training import ModelBundle

INDEX_FILE = "metadata.json"


def _index() -> dict:
    if models_storage.exists(INDEX_FILE):
        return models_storage.read_json(INDEX_FILE)
    return {"active": None, "versions": {}}


def save_version(version: str, bundles: dict[str, ModelBundle], metadata: dict,
                 activate: bool) -> None:
    for name, bundle in bundles.items():
        models_storage.write_object(f"{version}/{bundle.task.file_name}", bundle)
    models_storage.write_json(f"{version}/{INDEX_FILE}", metadata)
    index = _index()
    index["versions"][version] = {"role": metadata.get("role"), "created": metadata.get("created"),
                                  "trained_to": metadata.get("trained_to")}
    if activate:
        index["active"] = version
    models_storage.write_json(INDEX_FILE, index)
    load_bundles.cache_clear()


def active_version() -> str | None:
    return _index().get("active")


def list_versions() -> dict:
    return _index().get("versions", {})


def version_metadata(version: str | None = None) -> dict | None:
    version = version or active_version()
    if not version or not models_storage.exists(f"{version}/{INDEX_FILE}"):
        return None
    return models_storage.read_json(f"{version}/{INDEX_FILE}")


@lru_cache(maxsize=4)
def load_bundles(version: str | None = None) -> dict[str, ModelBundle]:
    """All available task bundles of a version (missing files are skipped)."""
    version = version or active_version()
    if not version:
        return {}
    bundles = {}
    for name, task in TASKS.items():
        path = f"{version}/{task.file_name}"
        if models_storage.exists(path):
            bundles[name] = models_storage.read_object(path)
    return bundles
