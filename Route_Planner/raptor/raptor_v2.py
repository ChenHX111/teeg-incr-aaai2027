"""Round-based RAPTOR algorithm (FIFO-safe version using vectorised search).

Delling, Pajor, Werneck. *Round-based Public Transit Routing.* Trans. Sci. 2015.

This implementation does NOT assume trips within a route are FIFO-sorted —
it uses a vectorised min(dep | dep>=bound) per (route, position), which is
correct regardless of the trip ordering in the underlying RAPTOR arrays.
This matters because a small number of routes in our dataset are non-FIFO
(overnight wraparound creates ordering breaks).
"""
from __future__ import annotations
import numpy as np
from ..common import TimetableBundle

INF = 32_000


def raptor_earliest_arrival(b: TimetableBundle, source: int, dest: int, t_dep: int,
                            max_rounds: int = 8) -> int | None:
    routes_array  = b.routes_array
    route_stops   = b.route_stops
    stop_routes   = b.stop_routes
    stops_array   = b.stops_array
    transfer_arr  = b.transfer_array
    st_min        = b.stop_times_min            # (Nst, 2): arr_min, dep_min, -1 = NaT

    S = b.n_stops

    tau_best = np.full(S, INF, dtype=np.int32)
    tau_prev = np.full(S, INF, dtype=np.int32)
    marked   = np.zeros(S, dtype=bool)

    # --- initialisation
    tau_best[source] = t_dep
    tau_prev[source] = t_dep
    marked[source]   = True
    lo, hi = int(stops_array[source, 2]), int(stops_array[source, 3])
    for k in range(lo, hi):
        v   = int(transfer_arr[k, 0])
        wsec= int(transfer_arr[k, 1])
        wmin= max(1, (wsec + 59) // 60) if wsec > 0 else 0
        nt  = t_dep + wmin
        if nt < tau_best[v]:
            tau_best[v] = nt
            tau_prev[v] = nt
            marked[v]   = True

    for round_no in range(1, max_rounds + 1):
        # --- step 1: collect (route, earliest_marked_position)
        queue: dict[int, int] = {}
        marked_stops = np.where(marked)[0]
        if len(marked_stops) == 0:
            break
        for s in marked_stops:
            srlo, srhi = int(stops_array[s, 0]), int(stops_array[s, 1])
            for j in range(srlo, srhi):
                r = int(stop_routes[j])
                rs_off = int(routes_array[r, 2])
                n_st   = int(routes_array[r, 1])
                pos = -1
                for i in range(n_st):
                    if int(route_stops[rs_off + i]) == int(s):
                        pos = i; break
                if pos == -1:
                    continue
                if (r not in queue) or (pos < queue[r]):
                    queue[r] = pos
        marked.fill(False)

        # --- step 2: scan each route from its earliest marked position
        for r, start_pos in queue.items():
            n_st  = int(routes_array[r, 1])
            n_tr  = int(routes_array[r, 0])
            rs_off = int(routes_array[r, 2])
            st_off = int(routes_array[r, 3])
            trip_offsets = st_off + np.arange(n_tr, dtype=np.int64) * n_st   # base offsets for all trips
            t_idx = -1
            for i in range(start_pos, n_st):
                stop_i = int(route_stops[rs_off + i])
                # if on a trip, update tau_best at stop_i using arrival time
                if t_idx >= 0:
                    a = int(st_min[st_off + t_idx * n_st + i, 0])
                    d = int(st_min[st_off + t_idx * n_st + i, 1])
                    if a >= 0 and a < tau_best[stop_i] and a < tau_best[dest]:
                        tau_best[stop_i] = a
                        marked[stop_i] = True
                    # CRITICAL: drop trip when arr or dep is NaT — this position is
                    # outside the trip's actual run; some "trips" in the data are
                    # discontinuous (terminator at pos i with dep=NaT, then a new
                    # physical run resumes at pos i+1 under the same trip_id).
                    if a < 0 or d < 0:
                        t_idx = -1
                # can we catch an earlier trip here?
                bound = int(tau_prev[stop_i])
                if bound >= INF:
                    continue
                # vectorised: deps of every trip at this position
                deps_col = st_min[trip_offsets + i, 1]
                valid = (deps_col >= 0) & (deps_col >= bound)
                if not valid.any():
                    continue
                # min-dep among valid trips, FIFO-safe
                masked = np.where(valid, deps_col, np.int16(INF))
                new_t = int(np.argmin(masked))
                new_d = int(deps_col[new_t])
                if t_idx == -1:
                    t_idx = new_t
                else:
                    cur_d = int(st_min[st_off + t_idx * n_st + i, 1])
                    if cur_d < 0 or new_d < cur_d:
                        t_idx = new_t

        # --- step 3: footpath relaxation
        for s in np.where(marked)[0].copy():
            tau_s = int(tau_best[s])
            lo, hi = int(stops_array[s, 2]), int(stops_array[s, 3])
            for k in range(lo, hi):
                v    = int(transfer_arr[k, 0])
                wsec = int(transfer_arr[k, 1])
                wmin = max(1, (wsec + 59) // 60) if wsec > 0 else 0
                nt   = tau_s + wmin
                if nt < tau_best[v]:
                    tau_best[v] = nt
                    marked[v]   = True

        # snapshot for next round's boarding bound
        tau_prev = tau_best.copy()

    return int(tau_best[dest]) if tau_best[dest] < INF else None
