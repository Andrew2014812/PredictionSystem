"""The ML research (experiments/) must never change production models, predictions or site data."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = {"save_version", "save_batch", "write_parquet", "write_json", "write_object", "drop_source",
                   "settle_predictions", "generate_live_predictions", "train_all"}
FORBIDDEN_NAMES = {"models_storage", "predictions_storage", "processed_storage", "live_storage", "odds_storage"}


def _calls(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        if isinstance(node, ast.Name):
            out.add(node.id)
        if isinstance(node, ast.alias):
            out.add(node.name.split(".")[-1])
    return out


def test_research_code_cannot_write_production_data():
    files = sorted((ROOT / "experiments").rglob("*.py"))
    assert files
    for path in files:
        used = _calls(path)
        assert not (used & FORBIDDEN_CALLS), (path, used & FORBIDDEN_CALLS)
        assert not (used & FORBIDDEN_NAMES), (path, used & FORBIDDEN_NAMES)


def test_research_outputs_stay_inside_experiments():
    from experiments.model_comparison import report, run
    from experiments.xgb_v5 import study
    for out in (run.RESULTS, report.OUT, study.OUT):
        assert (ROOT / "experiments") in Path(out).resolve().parents
