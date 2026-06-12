"""Compiled ULTRA-RAPTOR (Numba JIT) — Cycle 9 paper baseline.

ULTRA-RAPTOR is structurally RAPTOR with a (pruned) shortcut CSR substituted
in for the closed-walks CSR (per `Route_Planner/ultra/preprocess.py` and the
Stage 2 wrapper-depth Option (a) decision). The compiled implementation
therefore reuses every JIT'd hot function from `Route_Planner.raptor_compiled`
unchanged and only adds a thin wrapper that:

1. unpacks an UltraShortcuts / UltraShortcutsCSR / raw 3-tuple, and
2. forwards the call to `raptor_earliest_arrival_compiled` with that CSR
   bound to the `closed_walks_csr=` keyword argument.

Public surface mirrors `Route_Planner.ultra`:

    from Route_Planner.ultra_compiled import ultra_raptor_earliest_arrival_compiled

Empirical agreement: the compiled query is bit-for-bit equivalent to
`Route_Planner.ultra.ultra_raptor_earliest_arrival` whenever the underlying
compiled RAPTOR agrees with the pure-Python RAPTOR (730/730 ODs on Cycle 9
P0-D). See `tests/ultra_compiled/test_ultra_compiled_smoke.py` for parity.
"""
from __future__ import annotations

from .query import ultra_raptor_earliest_arrival_compiled

__all__ = ["ultra_raptor_earliest_arrival_compiled"]
