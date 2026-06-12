"""Multi-pass CSA (Dibbelt et al. 2018 Algorithm 2) — INDEPENDENT reference oracle.

Algorithm 2 (multi-pass / fixpoint CSA) differs from Algorithm 1 (the textbook
single-pass scan) on exactly one point:

    Alg. 1 only walk-relaxes from ``c_to`` when ``c_arr`` strictly improves
    ``tau[c_to]``. If a faster src-walk already set ``tau[c_to] <= c_arr``,
    the post-arrival walk relaxation is suppressed — the journey "ride to
    c_to, walk to Y, board at Y" is therefore invisible to Alg. 1.

    Alg. 2 walk-relaxes from EVERY usable connection's arrival, regardless of
    whether ``tau[c_to]`` strictly improved. It then iterates the connection
    scan until no ``tau`` value changes (fixpoint). This catches the "src-walk
    suppresses transit walk-relax" blind spot that Alg. 1 has.

What Alg. 2 does NOT do
-----------------------
- It does **not** chain walks across multiple walk legs without a board in
  between. The F_15 transitive closure already captures every walk *leg* up
  to 15 minutes through any number of intermediate hops; the per-leg cap is
  fundamental to the connection-scan model. Chaining walk-fixpoint via the
  closure would mean tau[X] propagates to tau[Y] via closure[X][Y'] and then
  tau[Z] via closure[Y'][Z] — i.e. walking 30+ minutes without boarding —
  which violates the standard CSA walking semantics.
- It does **not** add early-termination beyond Alg. 1's ``c.dep >= tau[dst]``
  check.

Walks per leg are exactly what ``transitive_closure_f15`` produces (multi-
source Dijkstra capped at ``max_walk_min`` per source). One closure lookup
from ``c_to`` is therefore equivalent to "any walk leg ≤ ``max_walk_min``".

Independence
------------
- Imports only ``numpy`` + ``numba`` + this package's parser + footpath_closure.
- Does NOT import from ``Route_Planner.csa``, ``Route_Planner.csa_compiled``,
  ``Route_Planner.teeg``, ``Route_Planner.raptor``, etc.

Correctness
-----------
Each tau value is the length of a real (src → ... → s) journey, so tau is an
upper bound on the EA. Tau values are int32 bounded below by 0 and can only
decrease, so the outer fixpoint loop terminates in finitely many iterations.
Empirically (Full London, 17 residual queries) Alg. 2 converges in 2-3 passes.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from typing import Any

from .parser import RefBundle
from .footpath_closure import transitive_closure_f15


INF_I32 = np.int32(2_147_483_647)


def _closure_to_csr(closed: dict[int, dict[int, int]], n_stops: int):
    """Convert {src: {dst: wmin}} dict-of-dicts to CSR (indptr, indices, wmins).

    The closure already encodes every walk leg ≤ ``max_walk_min`` (multi-source
    Dijkstra in ``transitive_closure_f15``). One lookup from a stop is
    therefore sufficient to relax every walk-reachable target within the cap.
    """
    sizes = np.zeros(n_stops + 1, dtype=np.int64)
    for s, m in closed.items():
        if s < n_stops:
            sizes[s + 1] = len(m)
    indptr = np.cumsum(sizes).astype(np.int32)
    n_edges = int(indptr[-1])
    indices = np.empty(n_edges, dtype=np.int32)
    wmins = np.empty(n_edges, dtype=np.int32)
    cursor = indptr[:-1].copy()
    for s, m in closed.items():
        if s >= n_stops:
            continue
        i = cursor[s]
        for v, w in m.items():
            indices[i] = v
            wmins[i] = w
            i += 1
        cursor[s] = i
    return indptr, indices, wmins


@njit(cache=True)
def _initial_walk_from_src(
    tau: np.ndarray,
    src: np.int32,
    t_dep: np.int32,
    cw_indptr: np.ndarray,
    cw_indices: np.ndarray,
    cw_wmins: np.ndarray,
) -> None:
    """Relax tau via the F_15 closure from src using time t_dep (single hop)."""
    lo = cw_indptr[src]
    hi = cw_indptr[src + 1]
    for k in range(lo, hi):
        v = cw_indices[k]
        w = cw_wmins[k]
        if t_dep > INF_I32 - w:
            continue
        nt = t_dep + w
        if nt < tau[v]:
            tau[v] = nt


@njit(cache=True)
def _connection_pass(
    tau: np.ndarray,
    trip_reached: np.ndarray,
    dep: np.ndarray, arr: np.ndarray,
    fr: np.ndarray, to: np.ndarray, tr: np.ndarray,
    start: int,
    cw_indptr: np.ndarray,
    cw_indices: np.ndarray,
    cw_wmins: np.ndarray,
    dst: int,
) -> int:
    """One Alg. 2 forward connection scan.

    Differences vs Alg. 1 (Dibbelt 2018) inside this loop:

    (i) When a connection is usable (its trip is reached, or boardable at
        ``c.from`` because ``dep >= tau[c.from]``), we ALWAYS walk-relax from
        ``c.to`` using time ``c.arr`` — even if ``c.arr`` does not strictly
        improve ``tau[c.to]``. This is the unconditional walk-relax that
        fixes Alg. 1's "src-walk suppresses transit walk-relax" blind spot.

    (ii) The walk-relax is a single hop over the F_15 closure of ``c.to``.
         No Bellman-Ford chaining: chaining the closure would correspond to
         walking 30+ minutes without boarding, which exceeds the per-leg
         walking cap and is not part of the standard CSA model.

    Early termination ``d >= tau[dst]``: a connection departing at time
    ``d`` no earlier than the current best arrival at dst cannot directly
    improve dst (its arrival is at or later than d, which is at or later than
    tau[dst]). Walk-relaxes from later arrivals likewise cannot improve dst.
    Since connections are sorted by dep_min, we can break.

    Returns
    -------
    n_updates : total tau-decreases this pass (rides + 1-hop walks).
    """
    n_updates = 0
    n = dep.shape[0]
    for i in range(start, n):
        d = dep[i]
        if d >= tau[dst]:
            break
        c_from = fr[i]
        c_to = to[i]
        c_trip = tr[i]
        c_arr = arr[i]
        if trip_reached[c_trip] == 1 or d >= tau[c_from]:
            trip_reached[c_trip] = 1
            # (i.a) Ride relax: arrival at c_to via this connection
            if c_arr < tau[c_to]:
                tau[c_to] = c_arr
                n_updates += 1
            # (i.b) UNCONDITIONAL 1-hop walk relax from c_to using c_arr.
            #       Closure[c_to] = all stops reachable from c_to within the
            #       per-leg walk cap (transitive closure of raw walks within
            #       max_walk_min). One lookup covers every walk-reachable v.
            lo = cw_indptr[c_to]
            hi = cw_indptr[c_to + 1]
            for k in range(lo, hi):
                v = cw_indices[k]
                w = cw_wmins[k]
                if c_arr > INF_I32 - w:
                    continue
                nt = c_arr + w
                if nt < tau[v]:
                    tau[v] = nt
                    n_updates += 1
    return n_updates


def csa_multipass_earliest_arrival(
    bundle: RefBundle,
    src: int,
    dst: int,
    t_dep: int,
    *,
    closed_walks_csr: tuple | None = None,
    closed_walks: dict | None = None,
    max_walk_min: int = 15,
    max_outer_passes: int = 32,
    conn_arrays: tuple | None = None,
) -> int | None:
    """Multi-pass (Dibbelt 2018 Alg. 2) earliest-arrival.

    Returns int minute or None if unreachable.

    Parameters
    ----------
    closed_walks_csr : pre-built (indptr, indices, wmins) tuple. If None,
        will be built from ``closed_walks`` or recomputed from bundle.walks.
    conn_arrays : pre-extracted (dep, arr, fr, to, tr) int32 arrays. If None,
        extracted from bundle.connections (slow if called per-query).
    max_outer_passes : safety cap on conn-scan iterations. Empirically Alg. 2
        converges in 2-3 passes on Full London; 32 is a generous cap.
    """
    n_stops = max(bundle.n_stops, src + 1, dst + 1)
    n_trips = bundle.n_trips

    if closed_walks_csr is None:
        if closed_walks is None:
            closed_walks = transitive_closure_f15(
                bundle.walks, n_stops, max_walk_min=max_walk_min
            )
        cw_indptr, cw_indices, cw_wmins = _closure_to_csr(closed_walks, n_stops)
    else:
        cw_indptr, cw_indices, cw_wmins = closed_walks_csr
    if cw_indptr.shape[0] < n_stops + 1:
        pad = np.full(n_stops + 1 - cw_indptr.shape[0], cw_indptr[-1], dtype=np.int32)
        cw_indptr = np.concatenate([cw_indptr, pad]).astype(np.int32)

    if conn_arrays is None:
        conns = bundle.connections
        dep = np.ascontiguousarray(conns["dep_min"], dtype=np.int32)
        arr = np.ascontiguousarray(conns["arr_min"], dtype=np.int32)
        fr = np.ascontiguousarray(conns["from_stop"], dtype=np.int32)
        to = np.ascontiguousarray(conns["to_stop"], dtype=np.int32)
        tr = np.ascontiguousarray(conns["trip_id_int"], dtype=np.int32)
    else:
        dep, arr, fr, to, tr = conn_arrays

    tau = np.full(n_stops, INF_I32, dtype=np.int32)
    trip_reached = np.zeros(n_trips, dtype=np.uint8)

    tau[src] = np.int32(t_dep)
    _initial_walk_from_src(
        tau, np.int32(src), np.int32(t_dep),
        cw_indptr, cw_indices, cw_wmins,
    )

    start = int(np.searchsorted(dep, t_dep, side="left"))

    # IMPORTANT: ``trip_reached`` is RESET at the start of every outer pass.
    # In Alg. 1 (single pass) this flag is set when we first board a trip and
    # then enables the "stay on trip" shortcut for downstream connections in
    # the same scan. Carrying it across passes is unsound for Alg. 2 because
    # an upstream connection of the trip (earlier in scan order) would be
    # processed before any actual boarding event in the new pass, causing
    # the algorithm to "teleport" onto the trip at its first stop and emit
    # spurious arrivals. Resetting each pass forces a fresh boarding-point
    # determination using the latest ``tau`` values and remains correct
    # because all useful boardings from prior passes are re-discovered (or
    # superseded by an earlier-reachable boarding stop that pass-N's walk
    # relaxations have now made finite).
    for outer in range(max_outer_passes):
        trip_reached.fill(0)
        n_upd = _connection_pass(
            tau, trip_reached, dep, arr, fr, to, tr, start,
            cw_indptr, cw_indices, cw_wmins, np.int32(dst),
        )
        if n_upd == 0:
            break

    result = int(tau[dst])
    return None if result >= int(INF_I32) else result


__all__ = [
    "csa_multipass_earliest_arrival",
    "_closure_to_csr",
]
