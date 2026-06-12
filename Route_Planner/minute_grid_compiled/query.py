"""Numba-JIT compiled Dial's bucket-Dijkstra on the MinuteGrid.

Mirrors Route_Planner.minute_grid.query_dial but with @njit(cache=True)
hot loops for 5-50× speedup on the inner Dijkstra scan.

CORRECTNESS: Must agree ≥99% with pure-Python minute_grid_dial_earliest_arrival
on 300 stratified ODs (SHORT/MEDIUM/LONG).
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Sentinel matching pure-Python INF16 = 32_000, but we use int32 max for safety
INF_I32 = np.int32(2_147_483_647)


@njit(cache=True)
def _dial_core_njit(
    indptr,        # int64[n_nodes+1]  CSR row pointers
    indices,       # int32[n_edges]    CSR column indices
    weights,       # int16[n_edges]    CSR edge weights
    n_nodes,       # int64             total nodes in graph
    max_buckets,   # int32             = T + 1 (max distance bound)
    seed_nodes,    # int32[n_seeds]    initial nodes
    seed_dists,    # int32[n_seeds]    initial distances
    dst_R_lo,      # int32             destination R-plane lower bound (inclusive)
    dst_R_hi,      # int32             destination R-plane upper bound (exclusive)
    dst_W_lo,      # int32             destination W-plane lower bound (inclusive)
    dst_W_hi,      # int32             destination W-plane upper bound (exclusive)
):
    """Dial's bucket-Dijkstra inner loop, JIT-compiled.

    Uses an entry-pool linked list to avoid corruption when a node is
    re-inserted into a different bucket (each insertion gets its own entry).

    Returns the distance (elapsed minutes) to destination, or -1 if unreachable.
    """
    # Distance array
    dist = np.full(n_nodes, INF_I32, dtype=np.int32)

    # Entry-pool bucket structure:
    # pool_node[i] = which graph node entry i refers to
    # pool_next[i] = next entry in same bucket chain (-1 = end)
    # bucket_head[b] = first entry index in bucket b (-1 = empty)
    # This avoids the bug where reusing node-id as linked-list pointer
    # corrupts chains when a node moves between buckets.
    n_edges = indices.shape[0]
    pool_cap = n_edges + seed_nodes.shape[0]  # upper bound on insertions
    pool_node = np.empty(pool_cap, dtype=np.int32)
    pool_next = np.full(pool_cap, np.int32(-1), dtype=np.int32)
    pool_size = np.int64(0)

    bucket_head = np.full(max_buckets, np.int32(-1), dtype=np.int32)

    # Seed initialization
    n_seeds = seed_nodes.shape[0]
    for i in range(n_seeds):
        node = seed_nodes[i]
        d = seed_dists[i]
        if d < max_buckets and d < dist[node]:
            dist[node] = d
            # Allocate entry
            eidx = pool_size
            pool_size += 1
            pool_node[eidx] = node
            pool_next[eidx] = bucket_head[d]
            bucket_head[d] = np.int32(eidx)

    # Main Dial loop
    # Key: when weight-0 edges insert into the CURRENT bucket, we must
    # re-drain the bucket (Python's `while bucket: pop()` does this naturally).
    # We drain by repeatedly taking bucket_head until -1.
    cur = np.int32(0)
    while cur < max_buckets:
        # Drain bucket completely (including entries added during processing)
        while bucket_head[cur] >= 0:
            # Pop from head
            eidx = bucket_head[cur]
            bucket_head[cur] = pool_next[eidx]
            node = pool_node[eidx]
            # Stale check
            if dist[node] != cur:
                continue
            # Destination check
            if ((dst_R_lo <= node < dst_R_hi) or
                    (dst_W_lo <= node < dst_W_hi)):
                return cur
            # Relax neighbors
            lo = indptr[node]
            hi = indptr[node + 1]
            for k in range(lo, hi):
                v = indices[k]
                w = weights[k]
                nd = cur + w
                if nd < dist[v] and nd < max_buckets:
                    dist[v] = nd
                    # Allocate new entry in bucket nd
                    new_eidx = pool_size
                    pool_size += 1
                    pool_node[new_eidx] = v
                    pool_next[new_eidx] = bucket_head[nd]
                    bucket_head[nd] = np.int32(new_eidx)
        cur += 1
    return np.int32(-1)


@njit(cache=True)
def _dijkstra_core_njit(
    indptr,        # int64[n_nodes+1]
    indices,       # int32[n_edges]
    weights,       # int16[n_edges]
    n_nodes,       # int64
    max_dist,      # int32  upper bound on useful distance
    seed_nodes,    # int32[n_seeds]
    seed_dists,    # int32[n_seeds]
    dst_R_lo,      # int32
    dst_R_hi,      # int32
    dst_W_lo,      # int32
    dst_W_hi,      # int32
):
    """Heap-based Dijkstra, JIT-compiled.

    Uses a simple binary heap (manual implementation for Numba).
    Returns elapsed minutes to destination, or -1 if unreachable.
    """
    dist = np.full(n_nodes, INF_I32, dtype=np.int32)

    # Manual min-heap: heap_d[i] = distance, heap_n[i] = node
    heap_cap = np.int64(1024)
    heap_d = np.zeros(heap_cap, dtype=np.int32)
    heap_n = np.zeros(heap_cap, dtype=np.int32)
    heap_size = np.int64(0)

    # Seed
    n_seeds = seed_nodes.shape[0]
    for i in range(n_seeds):
        node = seed_nodes[i]
        d = seed_dists[i]
        if d < dist[node]:
            dist[node] = d
            # Push to heap
            if heap_size >= heap_cap:
                new_cap = heap_cap * 2
                new_d = np.zeros(new_cap, dtype=np.int32)
                new_n = np.zeros(new_cap, dtype=np.int32)
                for j in range(heap_size):
                    new_d[j] = heap_d[j]
                    new_n[j] = heap_n[j]
                heap_d = new_d
                heap_n = new_n
                heap_cap = new_cap
            # Sift up
            pos = heap_size
            heap_d[pos] = d
            heap_n[pos] = node
            heap_size += 1
            while pos > 0:
                parent = (pos - 1) // 2
                if heap_d[pos] < heap_d[parent]:
                    heap_d[pos], heap_d[parent] = heap_d[parent], heap_d[pos]
                    heap_n[pos], heap_n[parent] = heap_n[parent], heap_n[pos]
                    pos = parent
                else:
                    break

    while heap_size > 0:
        # Pop min
        d_u = heap_d[0]
        u = heap_n[0]
        heap_size -= 1
        heap_d[0] = heap_d[heap_size]
        heap_n[0] = heap_n[heap_size]
        # Sift down
        pos = np.int64(0)
        while True:
            left = 2 * pos + 1
            right = 2 * pos + 2
            smallest = pos
            if left < heap_size and heap_d[left] < heap_d[smallest]:
                smallest = left
            if right < heap_size and heap_d[right] < heap_d[smallest]:
                smallest = right
            if smallest != pos:
                heap_d[pos], heap_d[smallest] = heap_d[smallest], heap_d[pos]
                heap_n[pos], heap_n[smallest] = heap_n[smallest], heap_n[pos]
                pos = smallest
            else:
                break

        if d_u > dist[u]:
            continue
        # Destination check
        if (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi):
            return d_u
        if d_u >= max_dist:
            continue

        lo = indptr[u]
        hi = indptr[u + 1]
        for k in range(lo, hi):
            v = indices[k]
            w = weights[k]
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                # Push to heap (with resize if needed)
                if heap_size >= heap_cap:
                    new_cap = heap_cap * 2
                    new_d = np.zeros(new_cap, dtype=np.int32)
                    new_n = np.zeros(new_cap, dtype=np.int32)
                    for j in range(heap_size):
                        new_d[j] = heap_d[j]
                        new_n[j] = heap_n[j]
                    heap_d = new_d
                    heap_n = new_n
                    heap_cap = new_cap
                pos = heap_size
                heap_d[pos] = nd
                heap_n[pos] = v
                heap_size += 1
                while pos > 0:
                    parent = (pos - 1) // 2
                    if heap_d[pos] < heap_d[parent]:
                        heap_d[pos], heap_d[parent] = heap_d[parent], heap_d[pos]
                        heap_n[pos], heap_n[parent] = heap_n[parent], heap_n[pos]
                        pos = parent
                    else:
                        break

    return np.int32(-1)


def _prepare_seeds(g, source: int, dest: int, t_dep: int, closed_walks_csr):
    """Prepare seed nodes and distances for the compiled query.

    Returns (seed_nodes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi)
    or None if t_dep is out of range.
    """
    if t_dep < g.t_lo or t_dep > g.t_hi:
        return None

    T = g.T
    seeds_nodes = []
    seeds_dists = []

    # Source R-node
    src_R = source * T + (t_dep - g.t_lo)
    seeds_nodes.append(src_R)
    seeds_dists.append(0)

    # Source walks -> W-plane
    if closed_walks_csr is not None:
        cw_indptr, cw_indices, cw_wmins = closed_walks_csr
        lo = int(cw_indptr[source])
        hi = int(cw_indptr[source + 1])
        for k in range(lo, hi):
            sp = int(cw_indices[k])
            wmin = int(cw_wmins[k])
            if wmin <= 0 or sp == source:
                continue
            t_arr = t_dep + wmin
            if t_arr > g.t_hi:
                continue
            v = g.RW_OFFSET + sp * T + (t_arr - g.t_lo)
            seeds_nodes.append(v)
            seeds_dists.append(wmin)

    seed_nodes = np.array(seeds_nodes, dtype=np.int32)
    seed_dists = np.array(seeds_dists, dtype=np.int32)

    # Destination bounds
    dst_R_lo = np.int32(dest * T)
    dst_R_hi = np.int32((dest + 1) * T)
    dst_W_lo = np.int32(g.RW_OFFSET + dest * T)
    dst_W_hi = np.int32(g.RW_OFFSET + (dest + 1) * T)

    return seed_nodes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi


def minute_grid_dial_earliest_arrival_compiled(
    g,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
):
    """Compiled Dial's bucket-Dijkstra on MinuteGrid.

    Drop-in replacement for minute_grid_dial_earliest_arrival with same
    signature and semantics.

    Parameters
    ----------
    g : MinuteGrid
    source, dest : int (stop indices)
    t_dep : int (departure minute)
    closed_walks_csr : tuple (indptr, indices, wmins) or None

    Returns
    -------
    int (earliest arrival minute) or None if unreachable.
    """
    prep = _prepare_seeds(g, source, dest, t_dep, closed_walks_csr)
    if prep is None:
        return None
    seed_nodes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = prep

    # Dtype assertions (audit-mandated)
    indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(g.indices, dtype=np.int32)
    weights = np.ascontiguousarray(g.weights, dtype=np.int16)

    max_buckets = np.int32(g.T + 1)
    n_nodes = np.int64(g.n_nodes)

    result = _dial_core_njit(
        indptr, indices, weights, n_nodes, max_buckets,
        seed_nodes, seed_dists,
        dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi,
    )
    if result < 0:
        return None
    return t_dep + int(result)


def minute_grid_dijkstra_earliest_arrival_compiled(
    g,
    source: int,
    dest: int,
    t_dep: int,
    *,
    closed_walks_csr=None,
):
    """Compiled heap-Dijkstra on MinuteGrid (lower priority reference).

    Same interface as minute_grid_dijkstra_earliest_arrival.
    """
    prep = _prepare_seeds(g, source, dest, t_dep, closed_walks_csr)
    if prep is None:
        return None
    seed_nodes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = prep

    indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(g.indices, dtype=np.int32)
    weights = np.ascontiguousarray(g.weights, dtype=np.int16)

    max_dist = np.int32(g.T + 1)
    n_nodes = np.int64(g.n_nodes)

    result = _dijkstra_core_njit(
        indptr, indices, weights, n_nodes, max_dist,
        seed_nodes, seed_dists,
        dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi,
    )
    if result < 0:
        return None
    return t_dep + int(result)
