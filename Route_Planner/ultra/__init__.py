"""ULTRA — UnLimited TRAnsfers (single-criterion projection of Baum et al.).

This package implements a single-criterion earliest-arrival ULTRA-shortcut
variant of the Baum et al. (ESA 2019; Transportation Science 2023) algorithm.
See `Experiments/results/cycle_6/stage4_scope.md` for the scope statement and
the Stage 2 / Stage 3 ship gates for the design lineage.

Public surface:

    from Route_Planner.ultra import (
        ultra_raptor_earliest_arrival,   # the query
        build_ultra_shortcuts,           # the preprocessing
        save_ultra_shortcuts_npz,        # cache I/O
        load_ultra_shortcuts_npz,
        UltraShortcuts,                  # the dataclass returned by build / load
    )
"""
from .preprocess import build_ultra_shortcuts, UltraShortcuts
from .cache import save_ultra_shortcuts_npz, load_ultra_shortcuts_npz
from .query import ultra_raptor_earliest_arrival

__all__ = [
    "build_ultra_shortcuts",
    "UltraShortcuts",
    "save_ultra_shortcuts_npz",
    "load_ultra_shortcuts_npz",
    "ultra_raptor_earliest_arrival",
]
