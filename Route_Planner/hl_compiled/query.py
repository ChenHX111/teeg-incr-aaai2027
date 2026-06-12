"""Numba-JIT compiled Hub Labels query functions.

This package intentionally does not modify ``Route_Planner.hl``.  The hot path
mirrors ``Route_Planner.hl.query._hl_stop_level_query``: a stop-level APSP/HL
lower-bound lookup followed by the CSA-style connection scan.  The public
wrapper keeps Python-only validation and object access at the boundary while the
per-connection scan and label-intersection loops run under Numba.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF_TIME_I32 = np.int32(32_000)
INF_LABEL_I32 = np.int32(2_147_483_647)

_EMPTY_I32 = np.empty(0, dtype=np.int32)
_EMPTY_INDPTR = np.zeros(1, dtype=np.int32)


@njit(cache=True)
def label_intersection_distance_compiled(fwd_hubs, fwd_dist, bwd_hubs, bwd_dist):
    """Return min dist over shared hubs for sorted int32 label arrays."""
    best = INF_LABEL_I32
    i = 0
    j = 0
    lu = fwd_hubs.shape[0]
    lv = bwd_hubs.shape[0]
    while i < lu and j < lv:
        hu = fwd_hubs[i]
        hv = bwd_hubs[j]
        if hu == hv:
            c = fwd_dist[i] + bwd_dist[j]
            if c < best:
                best = c
            i += 1
            j += 1
        elif hu < hv:
            i += 1
        else:
            j += 1
    return best


@njit(cache=True)
def _direct_walk_arrival_njit(source, dest, t_dep, walk_indptr, walk_indices, walk_weights):
    best = INF_TIME_I32
    lo = walk_indptr[source]
    hi = walk_indptr[source + 1]
    for wk in range(lo, hi):
        if walk_indices[wk] == dest:
            nt = t_dep + walk_weights[wk]
            if nt < best:
                best = nt
    return best


@njit(cache=True)
def _hl_stop_level_query_njit(
    source, dest, t_dep, n_stops, n_trips,
    stop_dist,
    conn_from, conn_to, conn_dep, conn_arr, conn_trip,
    walk_indptr, walk_indices, walk_weights,
):
    """Compiled clone of Route_Planner.hl.query._hl_stop_level_query."""
    lb = np.int32(stop_dist[source, dest])
    if lb >= INF_TIME_I32:
        return _direct_walk_arrival_njit(source, dest, t_dep, walk_indptr, walk_indices, walk_weights)

    n_conn = conn_from.shape[0]
    if n_conn == 0:
        return t_dep + lb

    tau = np.full(n_stops, INF_TIME_I32, dtype=np.int32)
    trip_reached = np.zeros(n_trips, dtype=np.uint8)

    tau[source] = t_dep
    wlo = walk_indptr[source]
    whi = walk_indptr[source + 1]
    for wk in range(wlo, whi):
        sp = walk_indices[wk]
        nt = t_dep + walk_weights[wk]
        if nt < tau[sp]:
            tau[sp] = nt

    best_dest = tau[dest]
    start = np.searchsorted(conn_dep, t_dep, side="left")

    for i in range(start, n_conn):
        d = np.int32(conn_dep[i])
        if d >= best_dest:
            break
        fs = conn_from[i]
        ts = conn_to[i]
        tr = conn_trip[i]
        a = np.int32(conn_arr[i])

        if trip_reached[tr] != 0 or d >= tau[fs]:
            if a < tau[ts]:
                tau[ts] = a
                trip_reached[tr] = 1
                if ts == dest and a < best_dest:
                    best_dest = a
                wlo2 = walk_indptr[ts]
                whi2 = walk_indptr[ts + 1]
                for wk in range(wlo2, whi2):
                    sp2 = walk_indices[wk]
                    nt2 = a + walk_weights[wk]
                    if nt2 < tau[sp2]:
                        tau[sp2] = nt2
                        if sp2 == dest and nt2 < best_dest:
                            best_dest = nt2

    if best_dest < INF_TIME_I32:
        return best_dest
    return INF_TIME_I32


def _as_i32_contig(a: np.ndarray, name: str) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.int32)
    assert out.dtype == np.int32, name
    return out


def _as_time_contig(a: np.ndarray, name: str) -> np.ndarray:
    assert a.dtype in (np.int16, np.int32), name
    return np.ascontiguousarray(a)


def _normalise_walk_csr(hl, n_stops: int):
    if hl.walk_indptr is None:
        return np.zeros(n_stops + 1, dtype=np.int32), _EMPTY_I32, _EMPTY_I32
    walk_indptr = _as_i32_contig(hl.walk_indptr, "walk_indptr")
    walk_indices = _as_i32_contig(hl.walk_indices, "walk_indices")
    walk_weights = _as_i32_contig(hl.walk_weights, "walk_weights")
    assert walk_indptr.shape[0] >= n_stops + 1
    assert walk_indices.shape == walk_weights.shape
    return walk_indptr, walk_indices, walk_weights


def _compiled_arrays(hl, n_stops: int):
    cached = getattr(hl, "_hl_compiled_arrays", None)
    if cached is not None:
        return cached

    stop_dist = np.ascontiguousarray(hl.stop_dist)
    assert stop_dist.dtype in (np.int16, np.int32)
    walk_indptr, walk_indices, walk_weights = _normalise_walk_csr(hl, n_stops)

    if hl.conn_from is None:
        conn_from = _EMPTY_I32
        conn_to = _EMPTY_I32
        conn_dep = _EMPTY_I32
        conn_arr = _EMPTY_I32
        conn_trip = _EMPTY_I32
        n_trips = 0
    else:
        conn_from = _as_i32_contig(hl.conn_from, "conn_from")
        conn_to = _as_i32_contig(hl.conn_to, "conn_to")
        conn_dep = _as_time_contig(hl.conn_dep, "conn_dep")
        conn_arr = _as_time_contig(hl.conn_arr, "conn_arr")
        conn_trip = _as_i32_contig(hl.conn_trip, "conn_trip")
        assert conn_from.shape == conn_to.shape == conn_dep.shape == conn_arr.shape == conn_trip.shape
        n_trips = int(conn_trip.max()) + 1 if conn_trip.size else 0

    cached = (
        stop_dist,
        conn_from, conn_to, conn_dep, conn_arr, conn_trip,
        walk_indptr, walk_indices, walk_weights,
        n_trips,
    )
    setattr(hl, "_hl_compiled_arrays", cached)
    return cached


def hl_earliest_arrival_compiled(hl, source: int, dest: int, t_dep: int) -> int | None:
    """Earliest arrival using the compiled stop-level HL/CSA query path.

    The function accepts the existing ``Route_Planner.hl.labelling.HLIndex`` and
    returns the same values as ``Route_Planner.hl.query.hl_earliest_arrival`` for
    stop-level indexes.  TEEG list-label mode is intentionally not routed through
    this wrapper because its labels are Python lists; use
    ``label_intersection_distance_compiled`` with flat int32 label arrays for the
    compiled label-intersection primitive.
    """
    if source == dest:
        return int(t_dep)
    if hl.stop_dist is None:
        raise ValueError("compiled HL query requires stop-level hl.stop_dist")

    n_stops = int(hl.n_stops)
    if source < 0 or dest < 0 or source >= n_stops or dest >= n_stops:
        return None

    (
        stop_dist,
        conn_from, conn_to, conn_dep, conn_arr, conn_trip,
        walk_indptr, walk_indices, walk_weights,
        n_trips,
    ) = _compiled_arrays(hl, n_stops)

    result = _hl_stop_level_query_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep), np.int32(n_stops), np.int32(n_trips),
        stop_dist,
        conn_from, conn_to, conn_dep, conn_arr, conn_trip,
        walk_indptr, walk_indices, walk_weights,
    )
    if int(result) >= int(INF_TIME_I32):
        return None
    return int(result)
