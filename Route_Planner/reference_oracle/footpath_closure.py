"""Independent transitively-closed F_15 footpath builder for Reference-CSA*.

Multi-source Dijkstra from every stop on the walking subgraph with per-source
cap of `max_walk_min` minutes. Returns a dict-of-dicts {src: {dst: walk_min}}.

Independence: imports only `numpy` + `heapq` + this package's own types.
Does NOT import from `Route_Planner.common.data_layer._build_transitively_closed_transfers`.
"""
from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Iterable

import numpy as np


def build_walk_adjacency(walks: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """walks: (W, 3) int32 (from, to, walk_min). Returns adj_list[from] = [(to, walk_min), ...]."""
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in walks:
        f, t, w = int(row[0]), int(row[1]), int(row[2])
        if w > 0:
            adj[f].append((t, w))
    return adj


def transitive_closure_f15(
    walks: np.ndarray,
    n_stops: int,
    *,
    max_walk_min: int = 15,
) -> dict[int, dict[int, int]]:
    """Multi-source Dijkstra from every stop; cap each source's reach at max_walk_min.

    Cap boundary semantics: paths of EXACTLY `max_walk_min` minutes are INCLUDED
    in the closure (`nd <= max_walk_min` passes the `nd > max_walk_min` filter).
    Paths exceeding `max_walk_min` are excluded. This matches the F_15 convention
    that 15-min walks are usable, 16-min walks are not.
    """
    adj = build_walk_adjacency(walks)
    closed: dict[int, dict[int, int]] = {}
    for src in range(n_stops):
        if src not in adj:
            continue
        # Dijkstra from src, cap at max_walk_min
        dist = {src: 0}
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, 0):
                continue
            if d >= max_walk_min:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd > max_walk_min:
                    continue
                if nd < dist.get(v, max_walk_min + 1):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        # Exclude src itself; only positive-walk targets
        out = {v: w for v, w in dist.items() if v != src and w > 0}
        if out:
            closed[src] = out
    return closed
