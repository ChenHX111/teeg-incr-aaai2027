"""Earliest-arrival queries on the state-augmented (R/W) minute-grid.

Three implementations:

  * minute_grid_dial_earliest_arrival   -- Dial's bucket-Dijkstra (no heap).
                                            The headline "BFS-like" method.
  * minute_grid_bfs_earliest_arrival    -- alias of Dial's; named separately
                                            for the paper narrative
                                            ("shortest hop on unit-expanded
                                            minute-grid").
  * minute_grid_dijkstra_earliest_arrival -- plain heap Dijkstra (reference).

All three return the earliest arrival minute at `dest` (or None if
unreachable), given departure at minute `t_dep` from `source`.

Source initialisation (matches CSA / RAPTOR / TEEG)
---------------------------------------------------
  dist[R(src, t_dep)]                 = 0
  dist[W(s', t_dep + w)]              = w     for every closed walk
                                              (src -> s') with walk-min w.

The destination is reached when ANY R(dst, t) or W(dst, t) is popped from
the priority queue.
"""
from __future__ import annotations
import heapq
from typing import Optional

import numpy as np

from .build_minute_grid import MinuteGrid

INF16 = np.int32(32_000)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _init_dist_and_seeds(g: MinuteGrid, source: int, dest: int, t_dep: int,
                        closed_walks_csr):
    """Initial dist[] and seed list (node, elapsed) for the query."""
    T = g.T
    dist = np.full(g.n_nodes, INF16, dtype=np.int32)
    seeds = []                              # list[(elapsed, node)]
    src_R = g.node_R(source, t_dep)
    dist[src_R] = 0
    seeds.append((0, src_R))
    # source walks land in W-plane
    if closed_walks_csr is not None:
        cw_indptr, cw_indices, cw_wmins = closed_walks_csr
        lo, hi = int(cw_indptr[source]), int(cw_indptr[source + 1])
        for k in range(lo, hi):
            sp = int(cw_indices[k]); wmin = int(cw_wmins[k])
            if wmin <= 0 or sp == source:
                continue
            t_arr = t_dep + wmin
            if t_arr > g.t_hi:
                continue
            v = g.node_W(sp, t_arr)
            if wmin < dist[v]:
                dist[v] = wmin
                seeds.append((wmin, v))
    # destination check helpers
    dst_R_lo = dest * T
    dst_R_hi = (dest + 1) * T
    dst_W_lo = g.RW_OFFSET + dst_R_lo
    dst_W_hi = g.RW_OFFSET + dst_R_hi
    return dist, seeds, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi


def _is_dest(u: int, dst_R_lo: int, dst_R_hi: int,
            dst_W_lo: int, dst_W_hi: int) -> bool:
    return (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi)


# ----------------------------------------------------------------------
# Dial's bucket Dijkstra
# ----------------------------------------------------------------------
def minute_grid_dial_earliest_arrival(
    g: MinuteGrid,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
) -> Optional[int]:
    """Dial's algorithm (priority-queue-free Dijkstra) on the minute-grid."""
    if t_dep < g.t_lo or t_dep > g.t_hi:
        return None
    dist, seeds, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = (
        _init_dist_and_seeds(g, source, dest, t_dep, closed_walks_csr)
    )

    max_buckets = g.T + 1
    buckets: list[list[int]] = [[] for _ in range(max_buckets)]
    for d, node in seeds:
        if d < max_buckets:
            buckets[d].append(node)

    indptr  = g.indptr
    indices = g.indices
    weights = g.weights

    cur = 0
    while cur < max_buckets:
        bucket = buckets[cur]
        while bucket:
            u = bucket.pop()
            if cur > dist[u]:
                continue
            if _is_dest(u, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi):
                return t_dep + int(cur)
            lo, hi = int(indptr[u]), int(indptr[u + 1])
            for k in range(lo, hi):
                v = int(indices[k])
                w = int(weights[k])
                nd = cur + w
                if nd < dist[v]:
                    dist[v] = nd
                    if nd < max_buckets:
                        buckets[nd].append(v)
        cur += 1
    return None


# ----------------------------------------------------------------------
# Pure BFS (semantic equivalent on the unit-expanded minute-grid)
# ----------------------------------------------------------------------
def minute_grid_bfs_earliest_arrival(
    g: MinuteGrid,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
) -> Optional[int]:
    """Pure BFS on the *implicit* unit-expanded minute-grid.

    Mathematically identical to Dial's on this graph because all edge
    weights are integers in [0, 30] and elapsed time bounds the BFS depth;
    presented as a named entry point for the paper narrative.
    """
    return minute_grid_dial_earliest_arrival(
        g, source, dest, t_dep, closed_walks_csr=closed_walks_csr,
    )


# ----------------------------------------------------------------------
# Plain heap Dijkstra (reference)
# ----------------------------------------------------------------------
def minute_grid_dijkstra_earliest_arrival(
    g: MinuteGrid,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
) -> Optional[int]:
    """Heap-based Dijkstra on the state-augmented minute-grid (reference)."""
    if t_dep < g.t_lo or t_dep > g.t_hi:
        return None
    dist, seeds, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = (
        _init_dist_and_seeds(g, source, dest, t_dep, closed_walks_csr)
    )

    pq: list[tuple[int, int]] = []
    for d, node in seeds:
        heapq.heappush(pq, (d, node))

    indptr  = g.indptr
    indices = g.indices
    weights = g.weights

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if _is_dest(u, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi):
            return t_dep + int(d)
        lo, hi = int(indptr[u]), int(indptr[u + 1])
        for k in range(lo, hi):
            v = int(indices[k]); w = int(weights[k])
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None
