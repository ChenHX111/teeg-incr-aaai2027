"""Compiled RAPTOR core — Numba JIT functions.

Implements `raptor_earliest_arrival_compiled` which mirrors
`Route_Planner.raptor.raptor.raptor_earliest_arrival` but with:
- @njit-compiled scan/relax loops
- precomputed (stop, route) → position CSR (avoids O(n_st) find_pos)
- numpy queue array (avoids typed.Dict)
- O(1) queue-slot lookup via `queue_slot_per_route` dense array
- manual scalar early-exit trip selection (avoids np.argmin in inner loop)
- INF = int32 sentinel (no float)
- Int32-overflow guards on walk additions (defence-in-depth)

CORRECTNESS RELATIONSHIP TO PURE-PYTHON:
This implementation EMPIRICALLY AGREES with `Route_Planner.raptor.raptor` on:
- 100/100 SHORT-subset ODs (Cycle 9 P0-D smoke test)
- 100/100 Full London parity sample (Cycle 9 P0-D bench)
- 30/30 mixed-stratum bench (Cycle 9 P0-D bench integration)
- 500/500 stratified subset ODs (170 SHORT + 170 MEDIUM + 160 LONG, ALL agree)
Cumulative: 730/730 (100.0%) agreement across all parity tests.

It is NOT formally proven bit-identical. Known divergence-risk axes (each
empirically agreed on the 730-OD cumulative sample but not formally ruled out):
- Trip-selection tie-breaking (manual scalar uses strict `<`, picks lowest
  trip index among same-departure ties; matches numpy `argmin` behaviour
  but assumes same iteration order).
- Loop routes (stops appearing multiple times in one route): both ref AND
  compiled record only first occurrence; agree but neither is loop-aware.
- `tau_prev := tau_best` happens at end-of-round AFTER walk relaxation in
  both implementations (verified at code-review time, not asserted).
- int16 `st_min` NaT sentinel is `< 0` (verified by inspection).
"""
from __future__ import annotations
import numpy as np
from numba import njit

# Sentinel (int32 max — see audit-mandated change 4)
INF_I32 = np.int32(2_147_483_647)


def build_stop_route_position_csr(b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper for preprocess.build_position_table; takes a TimetableBundle."""
    from .preprocess import build_position_table
    return build_position_table(b.routes_array, b.route_stops,
                                 b.stops_array, b.stop_routes, b.n_stops)


@njit(cache=True)
def _scan_round_with_start_njit(
    queue_arr, queue_start, queue_n,
    routes_array, route_stops, st_min,
    tau_best, tau_prev, marked, dest_idx,
):
    """Step 3: scan each queued route from its earliest marked position.

    queue_arr[i] : int32 — i-th route to scan
    queue_start[i] : int32 — earliest marked position in route queue_arr[i]
    queue_n : int — actual queue length
    """
    for qi in range(queue_n):
        r = queue_arr[qi]
        start_pos = queue_start[qi]
        n_tr = routes_array[r, 0]
        n_st = routes_array[r, 1]
        rs_off = routes_array[r, 2]
        st_off = routes_array[r, 3]

        t_idx = np.int32(-1)
        # Vectorized: trip_offsets = st_off + arange(n_tr) * n_st (avoided in JIT)
        for i in range(start_pos, n_st):
            stop_i = route_stops[rs_off + i]

            # If on a trip, update tau_best at stop_i using arrival time
            if t_idx >= 0:
                idx = st_off + t_idx * n_st + i
                a = st_min[idx, 0]
                d = st_min[idx, 1]
                if a >= 0 and a < tau_best[stop_i] and a < tau_best[dest_idx]:
                    tau_best[stop_i] = a
                    marked[stop_i] = 1
                # Trip discontinuity (NaT) — drop trip
                if a < 0 or d < 0:
                    t_idx = -1

            # Can we catch an earlier trip here?
            bound = tau_prev[stop_i]
            if bound >= INF_I32:
                continue

            # Manual scalar early-exit scan over trips (replaces np.argmin per audit)
            new_t = np.int32(-1)
            new_d = np.int32(INF_I32)
            for tt in range(n_tr):
                d_tt = st_min[st_off + tt * n_st + i, 1]
                if d_tt >= 0 and d_tt >= bound and d_tt < new_d:
                    new_d = d_tt
                    new_t = tt

            if new_t == -1:
                continue

            if t_idx == -1:
                t_idx = new_t
            else:
                cur_d = st_min[st_off + t_idx * n_st + i, 1]
                if cur_d < 0 or new_d < cur_d:
                    t_idx = new_t


@njit(cache=True)
def _collect_marked_routes_njit(
    marked, srp_indptr, srp_route_ids, srp_positions,
    queue_arr, queue_start, n_routes,
):
    """Step 2: collect routes-to-scan + earliest-marked-position per route.

    Uses TWO dense arrays for O(1) update (audit-mandated change):
    - queue_pos_per_route[r] = current best earliest position (or -1 if not queued).
    - queue_slot_per_route[r] = index of route r in queue_arr (or -1 if not queued).

    Returns queue_n (the number of routes queued).
    """
    queue_pos_per_route = np.full(n_routes, np.int32(-1), dtype=np.int32)
    queue_slot_per_route = np.full(n_routes, np.int32(-1), dtype=np.int32)
    queue_n = np.int32(0)

    n_stops = marked.shape[0]
    for s in range(n_stops):
        if marked[s] == 0:
            continue
        lo = srp_indptr[s]
        hi = srp_indptr[s + 1]
        for k in range(lo, hi):
            r = srp_route_ids[k]
            pos = srp_positions[k]
            if pos < 0:
                continue
            cur = queue_pos_per_route[r]
            if cur < 0:
                # First time queueing route r
                queue_pos_per_route[r] = pos
                queue_slot_per_route[r] = queue_n
                queue_arr[queue_n] = r
                queue_start[queue_n] = pos
                queue_n += 1
            elif pos < cur:
                # Update queued start position via O(1) slot lookup (was O(queue_n) scan)
                queue_pos_per_route[r] = pos
                queue_start[queue_slot_per_route[r]] = pos
    return queue_n


@njit(cache=True)
def _relax_walks_njit(
    marked_now, tau_snapshot, n_marked,
    cw_indptr, cw_indices, cw_wmins,
    tau_best, marked,
):
    """Step 4: footpath relaxation from snapshotted tau values.

    INT32 OVERFLOW GUARD (audit-mandated): skip if tau_s is sentinel value or
    if addition would wrap negative. Defends against future callers passing
    un-snapshotted data, even though our snapshot logic guarantees safety.
    """
    for mi in range(n_marked):
        s = marked_now[mi]
        tau_s = tau_snapshot[mi]
        if tau_s >= INF_I32:
            continue
        lo = cw_indptr[s]
        hi = cw_indptr[s + 1]
        for k in range(lo, hi):
            v = cw_indices[k]
            wmin = cw_wmins[k]
            nt = tau_s + wmin
            if nt < 0:  # overflow defence (cannot happen if tau_s < INF, but cheap)
                continue
            if nt < tau_best[v]:
                tau_best[v] = nt
                marked[v] = 1


@njit(cache=True)
def _raptor_core_njit(
    source, dest, t_dep, max_rounds,
    n_stops, n_routes,
    routes_array, route_stops, st_min,
    srp_indptr, srp_route_ids, srp_positions,
    cw_indptr, cw_indices, cw_wmins,
):
    """Full RAPTOR earliest-arrival, JIT-compiled.

    Returns int32 result; INF_I32 if unreachable.
    """
    tau_best = np.full(n_stops, INF_I32, dtype=np.int32)
    tau_prev = np.full(n_stops, INF_I32, dtype=np.int32)
    marked = np.zeros(n_stops, dtype=np.uint8)

    # Init from source — source-walk overflow defence (audit-mandated):
    # Skip walks that would overflow int32 when added to t_dep.
    tau_best[source] = t_dep
    tau_prev[source] = t_dep
    marked[source] = 1
    lo = cw_indptr[source]
    hi = cw_indptr[source + 1]
    for k in range(lo, hi):
        v = cw_indices[k]
        wmin = cw_wmins[k]
        # Overflow guard: ensure t_dep + wmin does not wrap (t_dep typically < 1500
        # and wmin <= 15, so headroom is enormous, but be explicit for AAAI correctness)
        if t_dep > INF_I32 - wmin:
            continue
        nt = t_dep + wmin
        if nt < tau_best[v]:
            tau_best[v] = nt
            tau_prev[v] = nt
            marked[v] = 1

    # Preallocated queues (max n_routes entries possible)
    queue_arr = np.zeros(n_routes, dtype=np.int32)
    queue_start = np.zeros(n_routes, dtype=np.int32)

    for round_no in range(1, max_rounds + 1):
        queue_n = _collect_marked_routes_njit(
            marked, srp_indptr, srp_route_ids, srp_positions,
            queue_arr, queue_start, n_routes,
        )
        if queue_n == 0:
            break
        # Clear marked for this round
        for i in range(n_stops):
            marked[i] = 0

        _scan_round_with_start_njit(
            queue_arr, queue_start, queue_n,
            routes_array, route_stops, st_min,
            tau_best, tau_prev, marked, dest,
        )

        # Snapshot marked-now tau values BEFORE walk relaxation (per pure-Python comment)
        # to avoid chaining two walks within a single round.
        marked_now_count = 0
        for i in range(n_stops):
            if marked[i]:
                marked_now_count += 1
        marked_now = np.zeros(marked_now_count, dtype=np.int32)
        tau_snapshot = np.zeros(marked_now_count, dtype=np.int32)
        mi = 0
        for i in range(n_stops):
            if marked[i]:
                marked_now[mi] = i
                tau_snapshot[mi] = tau_best[i]
                mi += 1

        _relax_walks_njit(
            marked_now, tau_snapshot, marked_now_count,
            cw_indptr, cw_indices, cw_wmins,
            tau_best, marked,
        )

        # Advance round: tau_prev := tau_best (for next round's bound)
        for i in range(n_stops):
            tau_prev[i] = tau_best[i]

    return tau_best[dest]


def raptor_earliest_arrival_compiled(
    b, source: int, dest: int, t_dep: int,
    *, max_rounds: int = 8, closed_walks_csr=None,
    srp_csr=None,
) -> int | None:
    """Top-level wrapper.

    Parameters
    ----------
    b : TimetableBundle (from Route_Planner.common.data_layer)
    closed_walks_csr : tuple (cw_indptr, cw_indices, cw_wmins) — REQUIRED.
        Must be transitively-closed walks (use Route_Planner.csa.csa._build_transitively_closed_transfers
        then convert to CSR).
    srp_csr : tuple (srp_indptr, srp_route_ids, srp_positions) — optional.
        If None, built from `b` (expensive — pass it in for benchmark runs).

    Returns
    -------
    int earliest arrival minute, or None if unreachable.
    """
    if closed_walks_csr is None:
        raise ValueError("closed_walks_csr is required for compiled RAPTOR")
    cw_indptr, cw_indices, cw_wmins = closed_walks_csr

    if srp_csr is None:
        srp_csr = build_stop_route_position_csr(b)
    srp_indptr, srp_route_ids, srp_positions = srp_csr

    # Dtype guards (audit-mandated change 4) — coerce inputs from native types
    routes_array_i32 = np.ascontiguousarray(b.routes_array, dtype=np.int32)
    route_stops_i32 = np.ascontiguousarray(b.route_stops, dtype=np.int32)
    st_min = b.stop_times_min  # int16 is fine for JIT
    assert st_min.dtype in (np.int16, np.int32)
    assert cw_indptr.dtype == np.int32
    assert cw_indices.dtype == np.int32
    assert cw_wmins.dtype == np.int32
    assert srp_indptr.dtype == np.int32
    assert srp_route_ids.dtype == np.int32
    assert srp_positions.dtype == np.int32

    result = _raptor_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep), np.int32(max_rounds),
        np.int32(b.n_stops), np.int32(len(routes_array_i32)),
        routes_array_i32, route_stops_i32, st_min,
        srp_indptr, srp_route_ids, srp_positions,
        cw_indptr, cw_indices, cw_wmins,
    )
    if int(result) >= INF_I32:
        return None
    return int(result)
