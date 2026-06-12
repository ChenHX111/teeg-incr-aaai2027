"""Numba-JIT compiled CH query: bidirectional upward Dijkstra.

Implements the same algorithm as Route_Planner.ch.query._bidirectional_upward
but using @njit(cache=True) compiled functions with manual min-heaps on
numpy arrays (Numba-friendly, no Python objects).

Sentinel: INT32 max (2_147_483_647) — consistent with raptor_compiled.
"""
from __future__ import annotations
import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)


# ---------------------------------------------------------------------------
# Manual min-heap on numpy arrays (Numba-friendly)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _heap_push(heap_dist, heap_node, heap_size, dist, node):
    """Push (dist, node) onto the min-heap. Returns new heap_size."""
    i = heap_size
    heap_dist[i] = dist
    heap_node[i] = node
    # sift up
    while i > 0:
        parent = (i - 1) >> 1
        if heap_dist[parent] > heap_dist[i]:
            # swap
            heap_dist[parent], heap_dist[i] = heap_dist[i], heap_dist[parent]
            heap_node[parent], heap_node[i] = heap_node[i], heap_node[parent]
            i = parent
        else:
            break
    return heap_size + 1


@njit(cache=True)
def _heap_pop(heap_dist, heap_node, heap_size):
    """Pop min element. Returns (dist, node, new_heap_size)."""
    d = heap_dist[0]
    n = heap_node[0]
    heap_size -= 1
    # move last to root
    heap_dist[0] = heap_dist[heap_size]
    heap_node[0] = heap_node[heap_size]
    # sift down
    i = np.int32(0)
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < heap_size and heap_dist[left] < heap_dist[smallest]:
            smallest = left
        if right < heap_size and heap_dist[right] < heap_dist[smallest]:
            smallest = right
        if smallest != i:
            heap_dist[i], heap_dist[smallest] = heap_dist[smallest], heap_dist[i]
            heap_node[i], heap_node[smallest] = heap_node[smallest], heap_node[i]
            i = smallest
        else:
            break
    return d, n, heap_size


# ---------------------------------------------------------------------------
# Core bidirectional upward Dijkstra (JIT)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bidirectional_upward_jit(
    src_nodes, src_dists, n_src,
    dst_nodes, dst_dists, n_dst,
    f_indptr, f_indices, f_weights,
    b_indptr, b_indices, b_weights,
    n_nodes,
):
    """Bidirectional upward Dijkstra, fully compiled.

    Parameters
    ----------
    src_nodes, src_dists : int32 arrays of seed nodes/distances (forward)
    dst_nodes, dst_dists : int32 arrays of seed nodes/distances (backward)
    f_indptr/f_indices/f_weights : forward (outgoing upward) CSR
    b_indptr/b_indices/b_weights : backward (incoming upward) CSR
    n_nodes : total node count

    Returns int32 shortest combined distance, or INF_I32 if unreachable.
    """
    if n_src == 0 or n_dst == 0:
        return INF_I32

    # Distance arrays (dense — fast lookup)
    dist_f = np.full(n_nodes, INF_I32, dtype=np.int32)
    dist_b = np.full(n_nodes, INF_I32, dtype=np.int32)

    # Heap capacity: in lazy-deletion Dijkstra, a node can be pushed multiple
    # times. Upward-only CH limits expansion, so n_nodes is a safe bound for
    # the number of pushes (each node settled once, pushed at most degree times).
    # Use 2*n_nodes as generous bound to cover seed duplicates.
    max_heap = 2 * n_nodes + n_src + n_dst
    heap_f_dist = np.empty(max_heap, dtype=np.int32)
    heap_f_node = np.empty(max_heap, dtype=np.int32)
    heap_f_size = np.int32(0)
    heap_b_dist = np.empty(max_heap, dtype=np.int32)
    heap_b_node = np.empty(max_heap, dtype=np.int32)
    heap_b_size = np.int32(0)

    # Seed forward heap
    for i in range(n_src):
        node = src_nodes[i]
        d = src_dists[i]
        if d < dist_f[node]:
            dist_f[node] = d
            heap_f_size = _heap_push(heap_f_dist, heap_f_node, heap_f_size, d, node)

    # Seed backward heap
    for i in range(n_dst):
        node = dst_nodes[i]
        d = dst_dists[i]
        if d < dist_b[node]:
            dist_b[node] = d
            heap_b_size = _heap_push(heap_b_dist, heap_b_node, heap_b_size, d, node)

    mu = INF_I32

    while heap_f_size > 0 or heap_b_size > 0:
        # Stopping condition
        top_f = heap_f_dist[0] if heap_f_size > 0 else INF_I32
        top_b = heap_b_dist[0] if heap_b_size > 0 else INF_I32
        min_top = top_f if top_f <= top_b else top_b
        if min_top >= mu:
            break

        if top_f <= top_b and heap_f_size > 0:
            d, u, heap_f_size = _heap_pop(heap_f_dist, heap_f_node, heap_f_size)
            if d > dist_f[u]:
                continue
            # Meeting check
            if dist_b[u] < INF_I32:
                tot = d + dist_b[u]
                if tot < mu:
                    mu = tot
            if d >= mu:
                continue
            # Expand forward (outgoing upward edges from u)
            for k in range(f_indptr[u], f_indptr[u + 1]):
                v = f_indices[k]
                nd = d + f_weights[k]
                if nd < dist_f[v]:
                    dist_f[v] = nd
                    heap_f_size = _heap_push(heap_f_dist, heap_f_node, heap_f_size, nd, v)
        elif heap_b_size > 0:
            d, v, heap_b_size = _heap_pop(heap_b_dist, heap_b_node, heap_b_size)
            if d > dist_b[v]:
                continue
            # Meeting check
            if dist_f[v] < INF_I32:
                tot = d + dist_f[v]
                if tot < mu:
                    mu = tot
            if d >= mu:
                continue
            # Expand backward (incoming upward edges to v)
            for k in range(b_indptr[v], b_indptr[v + 1]):
                u = b_indices[k]
                nd = d + b_weights[k]
                if nd < dist_b[u]:
                    dist_b[u] = nd
                    heap_b_size = _heap_push(heap_b_dist, heap_b_node, heap_b_size, nd, u)
        else:
            break

    return mu


# ---------------------------------------------------------------------------
# Python wrapper (matches ch_earliest_arrival signature)
# ---------------------------------------------------------------------------

def ch_earliest_arrival_compiled(ch, teeg, source: int, dest: int,
                                 t_dep: int) -> int | None:
    """Earliest-arrival query via CH-on-TEEG, Numba-JIT compiled.

    Drop-in replacement for Route_Planner.ch.query.ch_earliest_arrival.
    """
    from ..teeg.query import _source_attachments, _target_attachments, _pure_walk_arrival, INF_TIME

    if ch.n_nodes != teeg.n_nodes:
        raise ValueError("CHGraph and TEEG node counts disagree")

    src_attach = _source_attachments(teeg, source, t_dep)
    tgt_attach = _target_attachments(teeg, dest)
    pure_walk = _pure_walk_arrival(teeg, source, dest, t_dep)

    if (not src_attach or not tgt_attach) and pure_walk is None:
        return None
    if not src_attach or not tgt_attach:
        return pure_walk

    # Build seed arrays
    n_src = len(src_attach)
    src_nodes = np.empty(n_src, dtype=np.int32)
    src_dists = np.empty(n_src, dtype=np.int32)
    for i, (n, d) in enumerate(src_attach):
        src_nodes[i] = np.int32(n)
        src_dists[i] = np.int32(d)

    tgt_items = list(tgt_attach.items())
    n_dst = len(tgt_items)
    dst_nodes = np.empty(n_dst, dtype=np.int32)
    dst_dists = np.empty(n_dst, dtype=np.int32)
    for i, (n, d) in enumerate(tgt_items):
        dst_nodes[i] = np.int32(n)
        dst_dists[i] = np.int32(d)

    # Dtype assertions
    assert ch.up_out_indptr.dtype == np.int32
    assert ch.up_out_indices.dtype == np.int32
    assert ch.up_out_weights.dtype == np.int32
    assert ch.up_in_indptr.dtype == np.int32
    assert ch.up_in_indices.dtype == np.int32
    assert ch.up_in_weights.dtype == np.int32

    elapsed = _bidirectional_upward_jit(
        src_nodes, src_dists, np.int32(n_src),
        dst_nodes, dst_dists, np.int32(n_dst),
        ch.up_out_indptr, ch.up_out_indices, ch.up_out_weights,
        ch.up_in_indptr, ch.up_in_indices, ch.up_in_weights,
        np.int32(ch.n_nodes),
    )

    best_arrival = INF_TIME if pure_walk is None else int(pure_walk)
    if int(elapsed) < INF_I32:
        cand = int(t_dep + int(elapsed))
        if cand < best_arrival:
            best_arrival = cand
    return int(best_arrival) if best_arrival < INF_TIME else None


def ch_shortest_path_compiled(ch, src: int, dst: int) -> int | None:
    """Single-source single-target CH query (toy-test API), compiled."""
    if src == dst:
        return 0
    src_nodes = np.array([src], dtype=np.int32)
    src_dists = np.array([0], dtype=np.int32)
    dst_nodes = np.array([dst], dtype=np.int32)
    dst_dists = np.array([0], dtype=np.int32)

    assert ch.up_out_indptr.dtype == np.int32
    assert ch.up_out_indices.dtype == np.int32
    assert ch.up_out_weights.dtype == np.int32
    assert ch.up_in_indptr.dtype == np.int32
    assert ch.up_in_indices.dtype == np.int32
    assert ch.up_in_weights.dtype == np.int32

    d = _bidirectional_upward_jit(
        src_nodes, src_dists, np.int32(1),
        dst_nodes, dst_dists, np.int32(1),
        ch.up_out_indptr, ch.up_out_indices, ch.up_out_weights,
        ch.up_in_indptr, ch.up_in_indices, ch.up_in_weights,
        np.int32(ch.n_nodes),
    )
    return int(d) if int(d) < INF_I32 else None
