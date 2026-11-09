"""Runs every part of the data-sources study and writes results/ (python -m experiments.v6_2.data_sources.run).

Local data and the git history of this checkout only: no network, no API call, nothing in local_data/ changes.
"""
from __future__ import annotations

import logging
import time

from . import api_football, bets, common as C, football_data, oddspapi, runs, the_odds_api

log = logging.getLogger("data_sources")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    started = time.time()
    for name, mod in (("football_data", football_data), ("api_football", api_football), ("oddspapi", oddspapi),
                      ("the_odds_api", the_odds_api), ("runs", runs), ("bets", bets)):
        t = time.time()
        mod.run()
        log.info("%s done in %.1f s", name, time.time() - t)
    from . import figures
    figures.main()
    C.save_json("run_info.json", C.run_info({"season": C.SEASON, "timing_audit": str(C.TIMING_AUDIT.relative_to(C.ROOT))},
                                            started))
    log.info("all done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
