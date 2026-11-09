"""Builds the cached walk-forward predictions of the production setup (all seasons x seeds, corners, prices).

    python -m experiments.v6_2.leagues.base
"""
from __future__ import annotations

from . import common as C
from . import wf


def build() -> None:
    for s in C.SEASONS:
        wf.fd_table(s)
        wf.corners(s)
        for seed in C.SEEDS:
            wf.base_predictions(s, seed)
    C.log.info("base predictions done")


if __name__ == "__main__":
    C.setup_logging("base")
    build()
    print("BASE_DONE")
