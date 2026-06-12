"""Reference-CSA* core: vanilla textbook Connection Scan Algorithm.

Dibbelt et al. 2018, single-criterion earliest-arrival. Reads RefBundle.

Independence:
- Only imports `numpy` + `.parser.RefBundle` + `.footpath_closure.transitive_closure_f15`.
- Does NOT import from `Route_Planner.csa.csa` or any other Route_Planner module.

Algorithm (vanilla, per Dibbelt 2018 Alg. 1):
    initialize tau_best[s] = INF for all stops s
    tau_best[src] = t_dep
    for each (s, dst, w) in closed_walks_from[src]: tau_best[dst] = t_dep + w
    trip_reached[t] = False for all trips t
    for each connection c sorted by dep_min:
        if c.dep_min >= tau_best[dst]: break  # early termination, sound for EA
        if trip_reached[c.trip_id_int]:
            pass  # stay-on-trip
        elif c.dep_min >= tau_best[c.from_stop]:
            trip_reached[c.trip_id_int] = True  # first-time boarding
        else:
            continue  # cannot use this connection
        if c.arr_min < tau_best[c.to_stop]:
            tau_best[c.to_stop] = c.arr_min
            for v, w in closed_walks_from[c.to_stop]: tau_best[v] = min(...)
    return tau_best[dst] if reachable else None
"""
from __future__ import annotations
import numpy as np
from typing import Any

from .parser import RefBundle
from .footpath_closure import transitive_closure_f15


INF = np.iinfo(np.int32).max


def csa_earliest_arrival(
    bundle: RefBundle,
    src: int,
    dst: int,
    t_dep: int,
    *,
    closed_walks: dict[int, dict[int, int]] | None = None,
    max_walk_min: int = 15,
) -> int | None:
    """Return earliest arrival minute at dst from src departing at t_dep, or None."""
    if closed_walks is None:
        closed_walks = transitive_closure_f15(bundle.walks, bundle.n_stops, max_walk_min=max_walk_min)

    tau_best = np.full(bundle.n_stops, INF, dtype=np.int32)
    tau_best[src] = t_dep
    # Initial walk relaxation from src
    for v, w in closed_walks.get(src, {}).items():
        nt = t_dep + w
        if nt < tau_best[v]:
            tau_best[v] = nt

    trip_reached = np.zeros(bundle.n_trips, dtype=bool)

    conns = bundle.connections
    # Vanilla Dibbelt-et-al 2018 CSA (Alg. 1), single-criterion EA.
    # Connections are pre-sorted by dep_min (ascending) — verified in parser.py via lexsort.
    # Early termination is sound because future connections have c.dep >= current c.dep,
    # so they cannot improve tau_best[dst] (already <= current c.dep).
    for i in range(len(conns)):
        c_from = int(conns["from_stop"][i])
        c_to = int(conns["to_stop"][i])
        c_dep = int(conns["dep_min"][i])
        c_arr = int(conns["arr_min"][i])
        c_trip = int(conns["trip_id_int"][i])

        # Early termination: if c_dep >= tau_best[dst], no future connection can improve dst.
        if c_dep >= tau_best[dst]:
            break

        # Boarding predicate (Dibbelt 2018 Alg. 1): connection c is "usable" if either
        #   (a) trip is already reached at an earlier stop (stay-on-trip), OR
        #   (b) we can reach c.from_stop by time c.dep_min (board here for the first time).
        # Rewritten as explicit two-branch form per audit feedback (was confusing one-line OR).
        if trip_reached[c_trip]:
            # Stay-on-trip: vehicle is already moving along this trip, no boarding check needed.
            pass
        elif c_dep >= tau_best[c_from]:
            # First-time boarding: passenger reaches c_from by departure time of c.
            trip_reached[c_trip] = True
        else:
            # Cannot use this connection: trip not reached AND cannot board at c_from in time.
            continue

        # Relax arrival at c_to and propagate via footpath closure from c_to.
        if c_arr < tau_best[c_to]:
            tau_best[c_to] = c_arr
            walks_from_to = closed_walks.get(c_to)
            if walks_from_to:
                for v, w in walks_from_to.items():
                    nt = c_arr + w
                    if nt < tau_best[v]:
                        tau_best[v] = nt

    result = int(tau_best[dst])
    return None if result == INF else result
