"""Compiled CH (Contraction Hierarchies) query — Numba JIT.

Drop-in replacement for Route_Planner.ch.query.ch_earliest_arrival, with
the bidirectional upward Dijkstra compiled via @njit(cache=True).

The contraction/build step is NOT JITed (reuse Route_Planner.ch.contraction).
Only the query hot path is compiled.
"""
from __future__ import annotations
import warnings
import numba

_TESTED_NUMBA = "0.65"
if not numba.__version__.startswith(_TESTED_NUMBA):
    warnings.warn(
        f"ch_compiled tested on numba {_TESTED_NUMBA}; found {numba.__version__}. "
        f"Bit-identical results not guaranteed.",
        stacklevel=2,
    )

from .query import ch_earliest_arrival_compiled, ch_shortest_path_compiled, INF_I32

__all__ = ["ch_earliest_arrival_compiled", "ch_shortest_path_compiled", "INF_I32"]
