"""Connection Scan Algorithm (CSA) — earliest arrival and latest departure.

Reference algorithm. Treated as ground truth for correctness validation.

Dibbelt, Pajor, Strasser, Wagner. *Intriguingly Simple and Fast Transit
Routing.* SEA 2013.

We use the "basic CSA" variant with iterated walking footpaths: every time we
improve tau[v] via a trip arrival, we also relax walking transfers out of v.
Footpaths are also used at the source and at the destination (one walking hop
between transit segments).
"""
from __future__ import annotations
import numpy as np
from ..common import TimetableBundle

INF = 32_000


def _build_transfer_index(transfers_from: np.ndarray):
    """Group transfer rows by from_stop. Returns dict s -> list[(to, walk_min)]."""
    out: dict[int, list[tuple[int, int]]] = {}
    for from_s, to_s, walk_s in transfers_from:
        wmin = max(1, (int(walk_s) + 59) // 60) if walk_s > 0 else 0
        out.setdefault(int(from_s), []).append((int(to_s), wmin))
    return out


def _build_transitively_closed_transfers(transfers_from: np.ndarray, n_stops: int,
                                          max_walk_min: int = 10):
    """Transitively close the walking footpath graph.

    Standard CSA/RAPTOR assume footpaths are closed under transitivity
    (Dibbelt et al. 2018, §2). When the raw footpath set is not closed
    (as in our dataset, where walks are within a fixed radius and not
    re-densified for multi-hop), single-hop relaxation misses arrivals
    of the form trip -> walk -> walk -> dest.

    We close by running multi-source Dijkstra on the walking-only graph,
    capping each search at max_walk_min minutes (10 min ~ ~840 m, two
    400 m hops, which is the practically useful regime; further hops are
    extremely rare in passenger journeys).

    Returns
    -------
    dict : {src_stop_int -> list[(dst_stop_int, walk_min)]} for CSA use
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    if len(transfers_from) == 0:
        return {}
    src = transfers_from[:, 0].astype(np.int32)
    dst = transfers_from[:, 1].astype(np.int32)
    secs = transfers_from[:, 2].astype(np.int32)
    wmin = np.where(secs > 0, np.maximum(1, (secs + 59) // 60), 0).astype(np.int32)
    n = int(max(n_stops, src.max() + 1, dst.max() + 1))
    # dedup-min by lexsort
    key = src.astype(np.int64) * n + dst.astype(np.int64)
    order = np.lexsort((wmin, key))
    key_s = key[order]; src_s = src[order]; dst_s = dst[order]; wmin_s = wmin[order]
    first = np.empty(len(key_s), dtype=bool); first[0] = True
    first[1:] = key_s[1:] != key_s[:-1]
    src_u = src_s[first]; dst_u = dst_s[first]; wmin_u = wmin_s[first]
    # Build CSR manually (avoiding sum_duplicates)
    order2 = np.argsort(src_u, kind="stable")
    rs = src_u[order2]; cs = dst_u[order2]; ws = wmin_u[order2]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, rs + 1, 1)
    np.cumsum(indptr, out=indptr)
    mat = sp.csr_matrix((ws, cs, indptr), shape=(n, n))
    sources_with_walks = np.unique(src_u)
    dists = dijkstra(mat, directed=True, indices=sources_with_walks,
                     limit=float(max_walk_min), return_predecessors=False)
    out: dict[int, list[tuple[int, int]]] = {}
    for row_i, s in enumerate(sources_with_walks):
        d_row = dists[row_i]
        reachable = np.where(np.isfinite(d_row) & (d_row > 0))[0]
        out[int(s)] = [(int(v), int(d_row[v])) for v in reachable]
    return out


def closed_transfers_to_array(closed_dict: dict, n_stops: int) -> np.ndarray:
    """Flatten {s -> [(v, wmin)]} back to an Nx3 array (src, dst, walk_seconds)
    compatible with the original transfers_from format. Walk minutes are
    converted back to seconds (wmin * 60) so that downstream code that does
    `(wsec + 59) // 60` round-trips correctly.
    """
    n_rows = sum(len(vs) for vs in closed_dict.values())
    out = np.zeros((n_rows, 3), dtype=np.int32)
    i = 0
    for s, vs in closed_dict.items():
        for v, wmin in vs:
            out[i, 0] = s
            out[i, 1] = v
            out[i, 2] = wmin * 60  # back to seconds for downstream consumers
            i += 1
    return out


def build_closed_walk_csr(closed_dict: dict, n_stops: int):
    """Build CSR layout (indptr, indices, walk_min) from the closed-walks dict.
    Used by RAPTOR's transfer step for O(1) neighbour iteration.
    """
    n_rows = sum(len(vs) for vs in closed_dict.values())
    srcs = np.empty(n_rows, dtype=np.int32)
    dsts = np.empty(n_rows, dtype=np.int32)
    wmins = np.empty(n_rows, dtype=np.int32)
    i = 0
    for s, vs in closed_dict.items():
        for v, wmin in vs:
            srcs[i] = s
            dsts[i] = v
            wmins[i] = wmin
            i += 1
    order = np.argsort(srcs, kind="stable")
    srcs = srcs[order]; dsts = dsts[order]; wmins = wmins[order]
    indptr = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(indptr, srcs + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dsts, wmins


def csa_earliest_arrival(b: TimetableBundle, source: int, dest: int, t_dep: int,
                         transfers_idx: dict | None = None):
    """Earliest arrival at `dest` if leaving `source` at minute t_dep.

    Returns (best_arrival_min or None, tau_array).
    """
    nstops = max(b.n_stops, source + 1, dest + 1)
    tau = np.full(nstops, INF, dtype=np.int32)
    trip_reached = np.zeros(len(b.trip_id_str), dtype=bool)
    transfers = transfers_idx if transfers_idx is not None else _build_transfer_index(b.transfers_from)

    tau[source] = t_dep
    for (v, wmin) in transfers.get(source, []):
        nt = t_dep + wmin
        if nt < tau[v]:
            tau[v] = nt

    conns = b.connections
    dep = conns["dep_min"]; arr = conns["arr_min"]
    fr  = conns["from_stop"]; to = conns["to_stop"]; tr = conns["trip_id"]

    best_dest = int(tau[dest])
    start = int(np.searchsorted(dep, t_dep, side="left"))
    n = len(conns)
    for i in range(start, n):
        d = int(dep[i])
        if d >= best_dest:
            break
        c_from = int(fr[i]); c_to = int(to[i]); c_trip = int(tr[i]); c_arr = int(arr[i])
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
    return (int(tau[dest]) if tau[dest] < INF else None), tau


def csa_latest_departure(b: TimetableBundle, source: int, dest: int,
                         t_arr_deadline: int,
                         transfers_idx: dict | None = None):
    """Latest minute to leave `source` and still arrive at `dest` by `t_arr_deadline`."""
    nstops = max(b.n_stops, source + 1, dest + 1)
    sigma = np.full(nstops, -1, dtype=np.int32)
    trip_reached = np.zeros(len(b.trip_id_str), dtype=bool)
    transfers = transfers_idx if transfers_idx is not None else _build_transfer_index(b.transfers_from)

    sigma[dest] = t_arr_deadline
    for (v, wmin) in transfers.get(dest, []):
        nt = t_arr_deadline - wmin
        if nt > sigma[v]:
            sigma[v] = nt

    conns = b.connections
    dep = conns["dep_min"]; arr = conns["arr_min"]
    fr  = conns["from_stop"]; to = conns["to_stop"]; tr = conns["trip_id"]

    order = np.argsort(arr, kind="stable")[::-1]
    for i in order:
        a = int(arr[i])
        if a > t_arr_deadline:
            continue
        c_from = int(fr[i]); c_to = int(to[i]); c_trip = int(tr[i]); d = int(dep[i])
        if trip_reached[c_trip] or a <= sigma[c_to]:
            trip_reached[c_trip] = True
            if d > sigma[c_from]:
                sigma[c_from] = d
                for (v, wmin) in transfers.get(c_from, []):
                    nt = d - wmin
                    if nt > sigma[v]:
                        sigma[v] = nt
    return (int(sigma[source]) if sigma[source] > -1 else None), sigma
