"""Builds the per-season Main tables with every uncertainty measure (cache/mains_<s>.parquet).

    python -m experiments.v6_2.uncertainty.prepare 2023 2024 2025
"""
from __future__ import annotations

import sys
import time

from . import common as C
from . import data as D

if __name__ == "__main__":
    C.setup_logging("unc-prepare")
    for s in [int(x) for x in sys.argv[1:]] or list(C.SEASONS):
        t = time.time()
        m = D.build(s)
        C.log.info("season %s: %d Mains, %.0f s", s, len(m), time.time() - t)
