"""Records the outcome of the test runs (logs in cache/) into results/e2e_runs.json for the report.

    .venv/bin/python -m pytest tests -q --ignore=tests/e2e > experiments/v6_1/explain/cache/unit_tests.log
    .venv/bin/python -m pytest tests/e2e/test_pages.py --e2e -k why_card > .../cache/e2e_without_change.log   # v6 card
    .venv/bin/python -m pytest tests/e2e/test_pages.py --e2e > .../cache/e2e_with_change.log                  # v6.1 card
    .venv/bin/python experiments/v6_1/explain/record_tests.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


def last(name: str) -> str:
    lines = [ln for ln in (CACHE / name).read_text().splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    return lines[-1].strip("= ") if lines else "no result"


def first_error(name: str) -> str:
    lines = [ln[1:].strip() for ln in (CACHE / name).read_text().splitlines() if ln.startswith("E   ")]
    return lines[0] if lines else ""


out = {"unit": "pytest tests -q --ignore=tests/e2e: " + last("unit_tests.log"),
       "without_change": "the new spec on the v6 card: " + last("e2e_without_change.log") + " — "
                         + first_error("e2e_without_change.log"),
       "with_change": "the whole tests/e2e/test_pages.py on the v6.1 card: " + last("e2e_with_change.log")}
(HERE / "results" / "e2e_runs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
