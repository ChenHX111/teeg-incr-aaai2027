"""Compiled heap-Dijkstra query for :mod:`Route_Planner.minute_grid`.

The MinuteGrid CSR is built and loaded by the existing pure-Python package.  This
module only JIT-compiles the hot query traversal: seed preparation stays in
Python because it is tiny, while the full distance scan and binary heap live in
Numba nopython mode.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)


@njit(cache=True)
def _push_heap(heap_d, heap_n, heap_size, d, node):
    pos = heap_size
    heap_d[pos] = d
    heap_n[pos] = node
    heap_size += 1
    while pos > 0:
        parent = (pos - 1) // 2
        if heap_d[pos] < heap_d[parent]:
            td = heap_d[parent]
            tn = heap_n[parent]
            heap_d[parent] = heap_d[pos]
            heap_n[parent] = heap_n[pos]
            heap_d[pos] = td
            heap_n[pos] = tn
            pos = parent
        else:
            break
    return heap_size


@njit(cache=True)
def _pop_heap(heap_d, heap_n, heap_size):
    d = heap_d[0]
    node = heap_n[0]
    heap_size -= 1
    if heap_size > 0:
        heap_d[0] = heap_d[heap_size]
        heap_n[0] = heap_n[heap_size]
        pos = 0
        while True:
            left = 2 * pos + 1
            right = left + 1
            smallest = pos
            if left < heap_size and heap_d[left] < heap_d[smallest]:
                smallest = left
            if right < heap_size and heap_d[right] < heap_d[smallest]:
                smallest = right
            if smallest == pos:
                break
            td = heap_d[pos]
            tn = heap_n[pos]
            heap_d[pos] = heap_d[smallest]
            heap_n[pos] = heap_n[smallest]
            heap_d[smallest] = td
            heap_n[smallest] = tn
            pos = smallest
    return d, node, heap_size


@njit(cache=True)
def _grow_i32_pair(a, b):
    old = a.shape[0]
    new = old * 2
    aa = np.empty(new, dtype=np.int32)
    bb = np.empty(new, dtype=np.int32)
    for i in range(old):
        aa[i] = a[i]
        bb[i] = b[i]
    return aa, bb


@njit(cache=True)
def _mg_dijkstra_core_njit(
    indptr,
    indices,
    weights,
    n_nodes,
    max_dist,
    seed_nodes,
    seed_dists,
    dst_R_lo,
    dst_R_hi,
    dst_W_lo,
    dst_W_hi,
):
    """Return elapsed minutes to destination, or -1 if unreachable."""
    dist = np.full(n_nodes, INF_I32, dtype=np.int32)

    heap_cap = 1024
    heap_d = np.empty(heap_cap, dtype=np.int32)
    heap_n = np.empty(heap_cap, dtype=np.int32)
    heap_size = 0

    for i in range(seed_nodes.shape[0]):
        node = seed_nodes[i]
        d = seed_dists[i]
        if d >= 0 and node >= 0 and node < n_nodes and d < dist[node] and d < max_dist:
            dist[node] = d
            if heap_size >= heap_cap:
                heap_d, heap_n = _grow_i32_pair(heap_d, heap_n)
                heap_cap = heap_d.shape[0]
            heap_size = _push_heap(heap_d, heap_n, heap_size, d, node)

    while heap_size > 0:
        d_u, u, heap_size = _pop_heap(heap_d, heap_n, heap_size)
        if d_u != dist[u]:
            continue
        if (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi):
            return d_u
        if d_u >= max_dist:
            continue

        lo = indptr[u]
        hi = indptr[u + 1]
        for k in range(lo, hi):
            v = indices[k]
            w = np.int32(weights[k])
            if d_u > INF_I32 - w:
                continue
            nd = d_u + w
            if nd < dist[v] and nd < max_dist:
                dist[v] = nd
                if heap_size >= heap_cap:
                    heap_d, heap_n = _grow_i32_pair(heap_d, heap_n)
                    heap_cap = heap_d.shape[0]
                heap_size = _push_heap(heap_d, heap_n, heap_size, nd, v)

    return np.int32(-1)


def _prepare_seeds(g, source: int, dest: int, t_dep: int, closed_walks_csr):
    if t_dep < g.t_lo or t_dep > g.t_hi:
        return None

    T = int(g.T)
    t_off = int(t_dep) - int(g.t_lo)
    seed_nodes = [int(source) * T + t_off]
    seed_dists = [0]

    if closed_walks_csr is not None:
        cw_indptr, cw_indices, cw_wmins = closed_walks_csr
        lo = int(cw_indptr[source])
        hi = int(cw_indptr[source + 1])
        for k in range(lo, hi):
            sp = int(cw_indices[k])
            wmin = int(cw_wmins[k])
            if wmin <= 0 or sp == source:
                continue
            t_arr = int(t_dep) + wmin
            if t_arr > g.t_hi:
                continue
            seed_nodes.append(int(g.RW_OFFSET) + sp * T + (t_arr - int(g.t_lo)))
            seed_dists.append(wmin)

    dst_R_lo = int(dest) * T
    dst_R_hi = (int(dest) + 1) * T
    dst_W_lo = int(g.RW_OFFSET) + dst_R_lo
    dst_W_hi = int(g.RW_OFFSET) + dst_R_hi
    return (
        np.asarray(seed_nodes, dtype=np.int32),
        np.asarray(seed_dists, dtype=np.int32),
        np.int32(dst_R_lo),
        np.int32(dst_R_hi),
        np.int32(dst_W_lo),
        np.int32(dst_W_hi),
    )


def _assert_csr_dtypes(g) -> None:
    if g.indptr.dtype != np.int64:
        raise TypeError(f"MinuteGrid.indptr must be int64, got {g.indptr.dtype}")
    if g.indices.dtype != np.int32:
        raise TypeError(f"MinuteGrid.indices must be int32, got {g.indices.dtype}")
    if g.weights.dtype != np.int16:
        raise TypeError(f"MinuteGrid.weights must be int16, got {g.weights.dtype}")
    if not (g.indptr.flags.c_contiguous and g.indices.flags.c_contiguous and g.weights.flags.c_contiguous):
        raise TypeError("MinuteGrid CSR arrays must be C-contiguous")
    if int(g.n_nodes) > np.iinfo(np.int32).max:
        raise OverflowError("MinuteGrid node ids exceed int32 range")


def minute_grid_dijkstra_earliest_arrival_compiled(
    g,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
) -> Optional[int]:
    """Numba-compiled heap Dijkstra on an existing MinuteGrid.

    Drop-in counterpart to
    :func:`Route_Planner.minute_grid.minute_grid_dijkstra_earliest_arrival`.
    Returns the earliest arrival minute, or ``None`` when unreachable.
    """
    prep = _prepare_seeds(g, source, dest, t_dep, closed_walks_csr)
    if prep is None:
        return None
    _assert_csr_dtypes(g)
    seed_nodes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = prep

    # No valid target state can be more than T elapsed minutes from t_dep.
    max_dist = np.int32(int(g.T) + 1)
    elapsed = _mg_dijkstra_core_njit(
        g.indptr,
        g.indices,
        g.weights,
        np.int64(g.n_nodes),
        max_dist,
        seed_nodes,
        seed_dists,
        dst_R_lo,
        dst_R_hi,
        dst_W_lo,
        dst_W_hi,
    )
    if elapsed < 0:
        return None
    return int(t_dep) + int(elapsed)
