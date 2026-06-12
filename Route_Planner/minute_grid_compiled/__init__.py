"""Compiled Minute-Grid queries (Numba JIT) — Dial's bucket-Dijkstra + heap Dijkstra.

Mirrors Route_Planner.minute_grid.query_dial with @njit(cache=True) inner loops
for 5-50× speedup. The MinuteGrid object (CSR graph) is unchanged; only the
query traversal is JIT-compiled.

CORRECTNESS: Empirically validated ≥99% agreement with pure-Python
minute_grid_dial_earliest_arrival on 300 stratified ODs.
"""
from __future__ import annotations
import warnings

import numba

_TESTED_NUMBA = "0.60"
if not numba.__version__.startswith("0.6"):
    warnings.warn(
        f"minute_grid_compiled tested on numba ≥{_TESTED_NUMBA}; found {numba.__version__}. "
        f"Results may differ.",
        stacklevel=2,
    )

from .query import (
    minute_grid_dial_earliest_arrival_compiled,
    minute_grid_dijkstra_earliest_arrival_compiled,
    INF_I32,
)

__all__ = [
    "minute_grid_dial_earliest_arrival_compiled",
    "minute_grid_dijkstra_earliest_arrival_compiled",
    "INF_I32",
]
