"""Compiled CSA core — Numba JIT.

Implements `csa_earliest_arrival_compiled`, the JIT-compiled equivalent of
`Route_Planner.csa.csa.csa_earliest_arrival`. The inner connection-scan loop
is @njit-compiled; preprocessing (closed-walks CSR build, dtype coercion)
stays in Python.

SEMANTIC PARITY WITH PURE-PYTHON CSA:
The pure-Python reference (Route_Planner/csa/csa.py:147-161):

    for i in range(start, n):
        d = int(dep[i])
        if d >= best_dest:
            break
        c_from, c_to, c_trip, c_arr = ...
        if trip_reached[c_trip] or d >= tau[c_from]:
            trip_reached[c_trip] = True
            if c_arr < tau[c_to]:
                tau[c_to] = c_arr
                for (v, wmin) in transfers.get(c_to, []):
                    nt = c_arr + wmin
                    if nt < tau[v]:
                        tau[v] = nt
                if c_to == dest and c_arr < best_dest:
                    best_dest = c_arr

We preserve:
- Iteration order over connections (sorted by dep_min, same start).
- Early termination on `d >= best_dest`.
- Trip-reached flag semantics.
- Walk relaxation immediately after a strict improvement (strict `<`).
- best_dest update only when c_to == dest and strict improvement.
- Source-init: tau[source]=t_dep, then one walk hop out from source.

INT32 OVERFLOW GUARDS (defence-in-depth):
- t_dep + wmin and c_arr + wmin are bounded (typical t<=2000, wmin<=15) but we
  still guard via `INF_I32 - wmin` checks.

KNOWN DIVERGENCE RISKS (each empirically tested but not formally proven):
- closed-walks CSR ordering: the dict iteration order in pure-Python can be
  insertion order; CSR iterates `indices[lo:hi]` in stored order. Since walk
  relaxation uses strict `<`, ordering only affects which equal-value writes
  happen first, not the final tau values.
- The reference `_build_transfer_index` uses RAW transfers; `csa_earliest_arrival`
  with `transfers_idx=None` falls back to raw, not closed, walks. Our compiled
  wrapper REQUIRES closed-walks CSR for parity with the closed-walk semantics
  documented in `_build_transitively_closed_transfers`. Callers comparing parity
  must pass the SAME walks to both implementations.
"""
from __future__ import annotations
import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)


@njit(cache=True)
def _csa_core_njit(
    source, dest, t_dep,
    n_stops, n_trips,
    dep, arr, fr, to, tr,
    start,
    cw_indptr, cw_indices, cw_wmins,
):
    """Inner CSA scan loop. All inputs are int32 numpy arrays.

    Returns int32 tau[dest] (== INF_I32 if unreachable).
    """
    tau = np.full(n_stops, INF_I32, dtype=np.int32)
    trip_reached = np.zeros(n_trips, dtype=np.uint8)

    # Source init + one walk hop out
    tau[source] = t_dep
    lo = cw_indptr[source]
    hi = cw_indptr[source + 1]
    for k in range(lo, hi):
        v = cw_indices[k]
        wmin = cw_wmins[k]
        if t_dep > INF_I32 - wmin:
            continue
        nt = t_dep + wmin
        if nt < tau[v]:
            tau[v] = nt

    best_dest = tau[dest]
    n = dep.shape[0]

    for i in range(start, n):
        d = dep[i]
        if d >= best_dest:
            break
        c_from = fr[i]
        c_to = to[i]
        c_trip = tr[i]
        c_arr = arr[i]
        if trip_reached[c_trip] == 1 or d >= tau[c_from]:
            trip_reached[c_trip] = 1
            if c_arr < tau[c_to]:
                tau[c_to] = c_arr
                lo2 = cw_indptr[c_to]
                hi2 = cw_indptr[c_to + 1]
                for k in range(lo2, hi2):
                    v = cw_indices[k]
                    wmin = cw_wmins[k]
                    if c_arr > INF_I32 - wmin:
                        continue
                    nt = c_arr + wmin
                    if nt < tau[v]:
                        tau[v] = nt
                if c_to == dest and c_arr < best_dest:
                    best_dest = c_arr

    return tau[dest]


def build_closed_walk_csr_from_dict(closed_dict: dict, n_stops: int):
    """Convert {src_stop -> [(dst_stop, walk_min)]} to (indptr, indices, wmins).

    Matches the format used by raptor_compiled. n_stops must be >= max stop id + 1.
    """
    n_rows = sum(len(vs) for vs in closed_dict.values())
    srcs = np.empty(n_rows, dtype=np.int32)
    dsts = np.empty(n_rows, dtype=np.int32)
    wmins = np.empty(n_rows, dtype=np.int32)
    i = 0
    for s, vs in closed_dict.items():
        for v, w in vs:
            srcs[i] = s
            dsts[i] = v
            wmins[i] = w
            i += 1
    order = np.argsort(srcs, kind="stable")
    srcs = srcs[order]
    dsts = dsts[order]
    wmins = wmins[order]
    indptr = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(indptr, srcs + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dsts, wmins


def csa_earliest_arrival_compiled(
    b, source: int, dest: int, t_dep: int,
    *, closed_walks_csr=None,
    conn_arrays=None,
):
    """Top-level wrapper.

    Parameters
    ----------
    b : TimetableBundle
    closed_walks_csr : (indptr, indices, wmins), REQUIRED. Use
        `build_closed_walk_csr_from_dict(_build_transitively_closed_transfers(...))`.
    conn_arrays : optional precomputed (dep_i32, arr_i32, fr_i32, to_i32, tr_i32).
        If None, extracted from `b.connections` each call (slower; pass it for benches).

    Returns int earliest arrival minute, or None if unreachable.
    """
    if closed_walks_csr is None:
        raise ValueError("closed_walks_csr is required for compiled CSA")
    cw_indptr, cw_indices, cw_wmins = closed_walks_csr
    assert cw_indptr.dtype == np.int32
    assert cw_indices.dtype == np.int32
    assert cw_wmins.dtype == np.int32

    if conn_arrays is None:
        conns = b.connections
        dep_i32 = np.ascontiguousarray(conns["dep_min"], dtype=np.int32)
        arr_i32 = np.ascontiguousarray(conns["arr_min"], dtype=np.int32)
        fr_i32 = np.ascontiguousarray(conns["from_stop"], dtype=np.int32)
        to_i32 = np.ascontiguousarray(conns["to_stop"], dtype=np.int32)
        tr_i32 = np.ascontiguousarray(conns["trip_id"], dtype=np.int32)
    else:
        dep_i32, arr_i32, fr_i32, to_i32, tr_i32 = conn_arrays
        assert dep_i32.dtype == np.int32
        assert arr_i32.dtype == np.int32
        assert fr_i32.dtype == np.int32
        assert to_i32.dtype == np.int32
        assert tr_i32.dtype == np.int32

    n_stops = int(max(b.n_stops, source + 1, dest + 1))
    # cw_indptr was built for some n_stops; ensure it covers source/dest.
    if cw_indptr.shape[0] < n_stops + 1:
        # Pad indptr (no extra edges) so source/dest indexing stays in-bounds.
        pad = np.full(n_stops + 1 - cw_indptr.shape[0], cw_indptr[-1], dtype=np.int32)
        cw_indptr = np.concatenate([cw_indptr, pad]).astype(np.int32)

    n_trips = int(len(b.trip_id_str))
    start = int(np.searchsorted(dep_i32, t_dep, side="left"))

    result = _csa_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep),
        np.int32(n_stops), np.int32(n_trips),
        dep_i32, arr_i32, fr_i32, to_i32, tr_i32,
        np.int32(start),
        cw_indptr, cw_indices, cw_wmins,
    )
    if int(result) >= int(INF_I32) or int(result) >= 32_000:
        # csa.py uses INF=32_000 as the pure-Python sentinel; mirror that "None" gate.
        return None
    return int(result)
