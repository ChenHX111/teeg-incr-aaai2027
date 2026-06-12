"""Compiled RAPTOR (Numba JIT) — P0-D AAAI 2027 baseline.

Per multi-LLM design audit convergence:
- Pre-computed `(route, stop) → position` CSR table (eliminates O(n_st) find_pos scan)
- NumPy queue array (not numba.typed.Dict)
- Manual scalar early-exit loop (not np.argmin) for trip selection
- @njit(cache=True) to amortise compile cost
- INF = int32 sentinel 2_147_483_647 (NOT float('inf'))
- Dtype assertions in Python wrapper
- INT32 overflow guards in walk relaxation (ship-gate audit-mandated)
- O(1) queue-slot lookup (ship-gate audit-mandated)

Empirical agreement: 100/100 vs pure-Python RAPTOR on subset 100 SHORT ODs +
100/100 on Full London 100-OD parity sample. NOT formally proven bit-identical;
known divergence risks documented in core.py (trip-selection tie-breaking,
loop-route multi-occurrence handling).
"""
from __future__ import annotations
import warnings

import numpy as np
import numba

# Audit-mandated version check
_TESTED_NUMBA = "0.65.1"
_TESTED_LLVMLITE = "0.47.0"
if not numba.__version__.startswith(_TESTED_NUMBA.split(".")[0] + "." + _TESTED_NUMBA.split(".")[1]):
    warnings.warn(
        f"raptor_compiled tested on numba {_TESTED_NUMBA}; found {numba.__version__}. "
        f"Bit-identical results not guaranteed.",
        stacklevel=2,
    )

from .core import (
    raptor_earliest_arrival_compiled,
    build_stop_route_position_csr,
    INF_I32,
)
from .preprocess import build_position_table

__all__ = [
    "raptor_earliest_arrival_compiled",
    "build_stop_route_position_csr",
    "build_position_table",
    "INF_I32",
]
