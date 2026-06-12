"""Compiled TEEG-B: Numba-JIT bidirectional Dijkstra on the TEEG event graph.

Mirrors `Route_Planner.teeg.query.teeg_bidirectional` with:
- Manual binary min-heaps on numpy int32 arrays (forward + backward)
- Dense int32 dist arrays (no Python dicts)
- INT32 sentinel (no float('inf'))
- @njit(cache=True) hot core
- Dtype assertions in the Python wrapper
- Standard bidirectional termination: top_f + top_b >= best

Exported:
    teeg_bidirectional_compiled(g, source, dest, t_dep) -> int | None
"""
from __future__ import annotations
import warnings

import numba

_TESTED_NUMBA_MAJOR_MINOR = "0.65"
if not numba.__version__.startswith(_TESTED_NUMBA_MAJOR_MINOR):
    warnings.warn(
        f"teeg_b_compiled tested on numba {_TESTED_NUMBA_MAJOR_MINOR}.x; "
        f"found {numba.__version__}. Behaviour not formally guaranteed.",
        stacklevel=2,
    )

from .query import teeg_bidirectional_compiled, INF_I32

__all__ = ["teeg_bidirectional_compiled", "INF_I32"]
