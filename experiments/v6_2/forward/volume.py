"""Recomputes only results/volume.json (expected daily volume of the shadow log) from simulation.json.

    python -m experiments.v6_2.forward.volume
"""
from __future__ import annotations

from . import common as C
from .run import volume

if __name__ == "__main__":
    C.setup_logging("forward-volume")
    C.save_json("volume.json", volume(C.load_json("simulation.json", {})))
