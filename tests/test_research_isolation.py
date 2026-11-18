"""The ML research (experiments/) must never change production models, predictions or site data."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = {"save_version", "save_batch", "write_parquet", "write_json", "write_object", "drop_source",
                   "settle_predictions", "generate_live_predictions", "train_all"}
FORBIDDEN_NAMES = {"models_storage", "predictions_storage", "processed_storage", "live_storage", "odds_storage"}


# the v6.3 research helper ``experiments.v6_3.common.write_json`` writes into the study's own folder
RESEARCH_HELPERS = {"experiments.v6_3.common", "experiments.v6_3"}


def _helper_aliases(tree: ast.AST) -> set[str]:
    """Names under which a file imported the research helper module (``from experiments.v6_3 import common as C``)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in RESEARCH_HELPERS:
            out |= {a.asname or a.name for a in node.names if a.name == "common"}
        if isinstance(node, ast.Import):
            out |= {a.asname or a.name for a in node.names if a.name == "experiments.v6_3.common"}
    return out


def _calls(path: Path) -> set[str]:
    out = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helpers = _helper_aliases(tree)
    direct = {a.asname or a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              and node.module == "experiments.v6_3.common" for a in node.names if a.name == "write_json"}
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and (node.asname or node.name) in direct:
            continue
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in helpers
                    and f.attr == "write_json") or (isinstance(f, ast.Name) and f.id in direct):
                continue                                   # the research helper: experiments/ only
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        if isinstance(node, ast.Name) and node.id not in direct:
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


def test_the_v63_json_helper_writes_inside_experiments(tmp_path):
    from experiments.v6_3 import common as C
    assert (ROOT / "experiments") in C.SHARED_RESULTS.resolve().parents
    for name in ("audit", "replay"):
        assert (ROOT / "experiments") in C.study_dirs(name)["results"].resolve().parents
