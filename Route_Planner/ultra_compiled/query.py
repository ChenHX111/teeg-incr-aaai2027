"""Compiled ULTRA-RAPTOR query — JIT wrapper.

ULTRA-RAPTOR == RAPTOR(closed_walks_csr := ULTRA shortcut CSR). The compiled
RAPTOR core in `Route_Planner.raptor_compiled.core` already exposes
`closed_walks_csr` as a tuple of `(indptr, indices, wmins)` int32 CSR arrays,
which is precisely the layout of `UltraShortcutsCSR.as_tuple()`. So this
module is a thin unpack-and-forward shim.

Why this is correct WITHOUT a separate JIT core
------------------------------------------------
The JIT hot loops (`_scan_round_with_start_njit`, `_collect_marked_routes_njit`,
`_relax_walks_njit`, `_raptor_core_njit`) consume the closed-walks CSR as
opaque (indptr, indices, wmins) int32 arrays — they do NOT depend on whether
those edges came from transitive closure of footpaths or from ULTRA's pruned
shortcut graph. The walk-relaxation step (`_relax_walks_njit`) does the same
thing for both. Therefore, swapping ULTRA shortcuts in produces an algorithm
that is *both* (a) the published ULTRA-RAPTOR (by construction of the
shortcuts) and (b) JIT-compiled at the same speed as raptor_compiled. No new
@njit kernel is needed.

Dtype safety: an assert chain coerces the ULTRA CSR to int32 before it
reaches the JIT layer (raptor_compiled also asserts this). The
UltraShortcutsCSR loader (`Route_Planner.ultra.cache.load_ultra_shortcuts_npz`)
also asserts int32, but we re-assert here defensively to support callers who
build the CSR by hand (raw 3-tuple path).
"""
from __future__ import annotations

import numpy as np

from ..raptor_compiled import raptor_earliest_arrival_compiled
from ..ultra.cache import UltraShortcutsCSR
from ..ultra.preprocess import UltraShortcuts


def _coerce_ultra_csr(ultra_shortcuts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack any accepted ULTRA-shortcut container to the JIT-ready CSR triple."""
    if isinstance(ultra_shortcuts, (UltraShortcuts, UltraShortcutsCSR)):
        csr = ultra_shortcuts.as_tuple()
    elif isinstance(ultra_shortcuts, tuple) and len(ultra_shortcuts) == 3:
        csr = ultra_shortcuts
    else:
        raise TypeError(
            f"ultra_shortcuts must be UltraShortcuts/UltraShortcutsCSR or a "
            f"3-tuple (indptr, indices, wmins); got {type(ultra_shortcuts)!r}"
        )
    indptr, indices, wmins = csr
    assert indptr.dtype == np.int32, f"indptr dtype {indptr.dtype} (expected int32)"
    assert indices.dtype == np.int32, f"indices dtype {indices.dtype} (expected int32)"
    assert wmins.dtype == np.int32, f"wmins dtype {wmins.dtype} (expected int32)"
    assert len(indices) == len(wmins), \
        f"indices/wmins length mismatch: {len(indices)} vs {len(wmins)}"
    return indptr, indices, wmins


def ultra_raptor_earliest_arrival_compiled(
    b, source: int, dest: int, t_dep: int,
    ultra_shortcuts,
    *, max_rounds: int = 8, srp_csr=None,
) -> int | None:
    """Compiled ULTRA-RAPTOR earliest arrival (single-criterion).

    Parameters
    ----------
    b : TimetableBundle.
    source, dest, t_dep : query triple (ints, t_dep in minutes since epoch).
    ultra_shortcuts : one of
        - UltraShortcuts (returned by `build_ultra_shortcuts`)
        - UltraShortcutsCSR (returned by `load_ultra_shortcuts_npz`)
        - raw 3-tuple (indptr, indices, wmins) int32 arrays.
    max_rounds : RAPTOR round cap; defaults to 8 (matches pure-Python ULTRA).
    srp_csr : optional precomputed (stop, route) -> position CSR. If None,
        built on the fly (expensive — pass it in for benchmarks).

    Returns
    -------
    int (earliest arrival minute) or None if unreachable.

    Semantics
    ---------
    Bit-identical (up to RAPTOR vs raptor_compiled empirical-agreement bounds)
    to `Route_Planner.ultra.ultra_raptor_earliest_arrival(b, source, dest,
    t_dep, ultra_shortcuts, max_rounds=max_rounds)`.
    """
    csr = _coerce_ultra_csr(ultra_shortcuts)
    return raptor_earliest_arrival_compiled(
        b, source, dest, t_dep,
        max_rounds=max_rounds,
        closed_walks_csr=csr,
        srp_csr=srp_csr,
    )
