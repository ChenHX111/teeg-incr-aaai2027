"""CH query: bidirectional upward Dijkstra on the CHGraph.

Two entrypoints:

  * `ch_shortest_path(ch, src_node, dst_node)`  — plain bidirectional CH on
    arbitrary node ids (used by the toy test).
  * `ch_earliest_arrival(ch, teeg, source_stop, dest_stop, t_dep)` — full
    transit query: builds the same per-query source/destination attachments
    as the TEEG layer, then runs bidirectional upward Dijkstra and converts
    the elapsed-minutes answer back to arrival minute.
"""
from __future__ import annotations
import heapq
from typing import Optional

import numpy as np

from .contraction import CHGraph
from ..teeg.query import (
    _source_attachments, _target_attachments, _pure_walk_arrival, INF_TIME,
)


# ---------------------------------------------------------------------------
# Core bidirectional CH
# ---------------------------------------------------------------------------

def _bidirectional_upward(ch: CHGraph, src_seeds, dst_seeds) -> int:
    """Bidirectional upward Dijkstra.

    src_seeds : iterable of (node_id, initial_distance)
        Forward search starts with these (node, dist) pairs already in its
        heap (no upward restriction on the seed itself — the restriction
        applies to edges traversed during expansion).
    dst_seeds : iterable of (node_id, initial_distance)
        Backward search seeds (`initial_distance` represents the extra cost
        from `node_id` to the actual destination, in original-graph terms).

    Returns the shortest combined distance, or INF_TIME if unreachable.
    """
    src_seeds = list(src_seeds)
    dst_seeds = list(dst_seeds)
    if not src_seeds or not dst_seeds:
        return INF_TIME

    dist_f: dict = {}
    dist_b: dict = {}
    pq_f: list = []
    pq_b: list = []
    for (n, d) in src_seeds:
        if d < dist_f.get(n, INF_TIME):
            dist_f[n] = d
            heapq.heappush(pq_f, (d, n))
    for (n, d) in dst_seeds:
        if d < dist_b.get(n, INF_TIME):
            dist_b[n] = d
            heapq.heappush(pq_b, (d, n))

    f_indptr = ch.up_out_indptr; f_indices = ch.up_out_indices; f_w = ch.up_out_weights
    b_indptr = ch.up_in_indptr;  b_indices = ch.up_in_indices;  b_w = ch.up_in_weights
    # f_*: outgoing upward edges by source — forward search expansion.
    # b_*: incoming upward edges by target — backward search at node v
    #      iterates u such that (u→v) is upward, and "expands" to u with
    #      weight w(u→v). This is bidirectional Dijkstra on the upward graph.

    mu = INF_TIME
    while pq_f or pq_b:
        # stopping condition: when both heap-tops sum to >= mu, no improvement possible
        top_f = pq_f[0][0] if pq_f else INF_TIME
        top_b = pq_b[0][0] if pq_b else INF_TIME
        if min(top_f, top_b) >= mu:
            break

        if top_f <= top_b and pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f.get(u, INF_TIME):
                continue
            # meeting check
            if u in dist_b:
                tot = d + dist_b[u]
                if tot < mu:
                    mu = tot
            if d >= mu:
                continue
            for k in range(f_indptr[u], f_indptr[u + 1]):
                v = int(f_indices[k])
                nd = d + int(f_w[k])
                if nd < dist_f.get(v, INF_TIME):
                    dist_f[v] = nd
                    heapq.heappush(pq_f, (nd, v))
        elif pq_b:
            d, v = heapq.heappop(pq_b)
            if d > dist_b.get(v, INF_TIME):
                continue
            if v in dist_f:
                tot = d + dist_f[v]
                if tot < mu:
                    mu = tot
            if d >= mu:
                continue
            for k in range(b_indptr[v], b_indptr[v + 1]):
                u = int(b_indices[k])
                nd = d + int(b_w[k])
                if nd < dist_b.get(u, INF_TIME):
                    dist_b[u] = nd
                    heapq.heappush(pq_b, (nd, u))
        else:
            break
    return int(mu)


def ch_shortest_path(ch: CHGraph, src: int, dst: int) -> Optional[int]:
    """Single-source single-target CH query on raw node ids (toy-test API)."""
    if src == dst:
        return 0
    d = _bidirectional_upward(ch, [(src, 0)], [(dst, 0)])
    return int(d) if d < INF_TIME else None


# ---------------------------------------------------------------------------
# Transit query
# ---------------------------------------------------------------------------

def ch_earliest_arrival(ch: CHGraph, teeg, source: int, dest: int,
                        t_dep: int) -> Optional[int]:
    """Earliest-arrival query via CH-on-TEEG.

    Parameters
    ----------
    ch    : CHGraph produced by `build_ch_from_teeg(teeg, ...)`
    teeg  : TEEG, same instance used to build `ch`
    source, dest, t_dep : stop ints and departure minute (TEEG semantics)
    """
    if ch.n_nodes != teeg.n_nodes:
        raise ValueError("CHGraph and TEEG node counts disagree")

    src_attach = _source_attachments(teeg, source, t_dep)
    tgt_attach = _target_attachments(teeg, dest)
    pure_walk = _pure_walk_arrival(teeg, source, dest, t_dep)

    if (not src_attach or not tgt_attach) and pure_walk is None:
        return None
    if not src_attach or not tgt_attach:
        return pure_walk

    # Seeds: source side is (D_node, elapsed_to_board); target side is
    # (A_node, walk_to_dest_minutes).
    src_seeds = [(int(n), int(d)) for (n, d) in src_attach]
    dst_seeds = [(int(n), int(d)) for (n, d) in tgt_attach.items()]

    elapsed = _bidirectional_upward(ch, src_seeds, dst_seeds)

    best_arrival = INF_TIME if pure_walk is None else int(pure_walk)
    if elapsed < INF_TIME:
        cand = int(t_dep + elapsed)
        if cand < best_arrival:
            best_arrival = cand
    return int(best_arrival) if best_arrival < INF_TIME else None
