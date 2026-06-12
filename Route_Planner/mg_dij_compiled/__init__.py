"""Numba-JIT heap Dijkstra for the state-augmented minute-grid."""
from __future__ import annotations
import warnings

import numba

_TESTED_NUMBA_MAJOR_MINOR = "0.6"
if not numba.__version__.startswith(_TESTED_NUMBA_MAJOR_MINOR):
    warnings.warn(
        f"mg_dij_compiled tested on numba {_TESTED_NUMBA_MAJOR_MINOR}x; found {numba.__version__}. ",
        stacklevel=2,
    )

from .query import INF_I32, minute_grid_dijkstra_earliest_arrival_compiled

__all__ = ["INF_I32", "minute_grid_dijkstra_earliest_arrival_compiled"]
