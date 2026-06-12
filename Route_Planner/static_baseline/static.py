"""Static (timetable-ignoring) Dijkstra baseline — known incorrect.

Builds a stop-graph whose edge weight is min(arr - dep) across all
observed connections, and a separate walking weight in minutes. Runs
plain Dijkstra. The resulting "earliest arrival" claim is
t_dep + sum(edge_weights) along the shortest path, which ignores all
waiting and trip-coupling constraints — and is therefore systematically
under-estimating.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from ..common import TimetableBundle


def build_static_index(b: TimetableBundle):
    """Per-pair (u,v) min ride time across all connections; plus walking."""
    c = b.connections
    n = b.n_stops
    # per (u,v) min ride
    agg: dict[tuple[int, int], int] = {}
    for i in range(len(c)):
        u = int(c["from_stop"][i]); v = int(c["to_stop"][i])
        w = int(c["arr_min"][i]) - int(c["dep_min"][i])
        k = (u, v)
        if k not in agg or w < agg[k]:
            agg[k] = w
    # walking
    for from_s, to_s, walk_s in b.transfers_from:
        wmin = max(1, (int(walk_s) + 59) // 60) if int(walk_s) > 0 else 0
        k = (int(from_s), int(to_s))
        if k not in agg or wmin < agg[k]:
            agg[k] = wmin
    if not agg:
        return sp.csr_matrix((n, n), dtype=np.int32)
    rows = np.fromiter((k[0] for k in agg.keys()), dtype=np.int32, count=len(agg))
    cols = np.fromiter((k[1] for k in agg.keys()), dtype=np.int32, count=len(agg))
    vals = np.fromiter(agg.values(), dtype=np.int32, count=len(agg))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def static_earliest_arrival(b: TimetableBundle, source: int, dest: int,
                             t_dep: int, idx=None):
    if idx is None:
        idx = build_static_index(b)
    d = dijkstra(idx, directed=True, indices=int(source),
                 return_predecessors=False)
    val = d[dest]
    if np.isinf(val):
        return None
    return int(t_dep + int(val))
