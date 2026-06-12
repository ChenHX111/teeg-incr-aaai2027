"""Compiled CSA (Numba JIT) — earliest arrival.

Mirrors `Route_Planner.csa.csa.csa_earliest_arrival` with:
- @njit(cache=True) inner scan loop
- INT32 sentinel (2_147_483_647)
- Closed-walks CSR (use Route_Planner.csa.csa._build_transitively_closed_transfers
  then convert to (indptr, indices, wmins) — see `build_closed_walk_csr_from_dict`).
- Dtype assertions in the Python wrapper
- INT32-overflow guards on walk additions (defence-in-depth)
"""
from __future__ import annotations
from .core import (
    csa_earliest_arrival_compiled,
    build_closed_walk_csr_from_dict,
    INF_I32,
)

__all__ = [
    "csa_earliest_arrival_compiled",
    "build_closed_walk_csr_from_dict",
    "INF_I32",
]
