"""Numba-JIT compiled forward Dijkstra on TEEG event graph.

Mirrors Route_Planner.teeg.query.teeg_dijkstra but with:
- @njit(cache=True) compiled hot Dijkstra loop
- Manual binary min-heap on numpy arrays (Numba-compatible)
- INT32 sentinel, no float
- Early-termination on destination nodes
- Source/target attachment computed in Python wrapper, hot loop in Numba
"""
from __future__ import annotations
import numpy as np
from numba import njit, types
from numba.core.types import int32, int64, boolean

# Sentinel: int32 max
INF_I32 = np.int32(2_147_483_647)


# ============================================================================
# Manual min-heap on numpy arrays (Numba-compatible)
# ============================================================================

@njit(cache=True)
def _heap_push(heap_keys, heap_vals, heap_size, key, val):
    """Push (key, val) onto min-heap. Returns new heap_size."""
    pos = heap_size
    heap_keys[pos] = key
    heap_vals[pos] = val
    # sift up
    while pos > 0:
        parent = (pos - 1) >> 1
        if heap_keys[parent] > heap_keys[pos]:
            # swap
            heap_keys[parent], heap_keys[pos] = heap_keys[pos], heap_keys[parent]
            heap_vals[parent], heap_vals[pos] = heap_vals[pos], heap_vals[parent]
            pos = parent
        else:
            break
    return heap_size + 1


@njit(cache=True)
def _heap_pop(heap_keys, heap_vals, heap_size):
    """Pop min element. Returns (key, val, new_heap_size)."""
    key = heap_keys[0]
    val = heap_vals[0]
    heap_size -= 1
    # move last to root
    heap_keys[0] = heap_keys[heap_size]
    heap_vals[0] = heap_vals[heap_size]
    # sift down
    pos = 0
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos
        if left < heap_size and heap_keys[left] < heap_keys[smallest]:
            smallest = left
        if right < heap_size and heap_keys[right] < heap_keys[smallest]:
            smallest = right
        if smallest != pos:
            heap_keys[pos], heap_keys[smallest] = heap_keys[smallest], heap_keys[pos]
            heap_vals[pos], heap_vals[smallest] = heap_vals[smallest], heap_vals[pos]
            pos = smallest
        else:
            break
    return key, val, heap_size


# ============================================================================
# Core Dijkstra (njit)
# ============================================================================

@njit(cache=True)
def _dijkstra_core(
    indptr,        # int32[n_nodes+1]
    indices,       # int32[n_edges]
    weights,       # int32[n_edges]
    n_nodes,       # int32
    src_nodes,     # int32[n_src] — source-attached node IDs
    src_dists,     # int32[n_src] — initial distances (elapsed from t_dep)
    tgt_nodes,     # int32[n_tgt] — target-attached node IDs
    tgt_extras,    # int32[n_tgt] — extra walking cost to reach dest
    tgt_mask,      # bool[n_nodes] — True if node is a target attachment
    tgt_extra_arr, # int32[n_nodes] — per-node extra (only valid where tgt_mask=True)
    best_init,     # int32 — initial best (pure walk or INF_I32)
):
    """Dijkstra with early termination. Returns best elapsed time (int32).

    Returns INF_I32 if no path found.
    """
    # dist array
    dist = np.full(n_nodes, INF_I32, dtype=np.int32)

    # Estimate heap capacity: n_edges is upper bound on pushes
    # Use a generous pre-allocation
    heap_cap = len(indices) + len(src_nodes) + 1
    if heap_cap > 4_000_000:
        heap_cap = 4_000_000
    heap_keys = np.empty(heap_cap, dtype=np.int32)
    heap_vals = np.empty(heap_cap, dtype=np.int32)
    heap_size = np.int32(0)

    # Initialize with source attachments
    for i in range(len(src_nodes)):
        n = src_nodes[i]
        d = src_dists[i]
        if d < dist[n]:
            dist[n] = d
            if heap_size < heap_cap:
                heap_size = _heap_push(heap_keys, heap_vals, heap_size, d, n)

    best = best_init

    while heap_size > 0:
        d, u, heap_size = _heap_pop(heap_keys, heap_vals, heap_size)

        if d > dist[u]:
            continue

        # Check if u is a target node
        if tgt_mask[u]:
            cand = d + tgt_extra_arr[u]
            if cand >= 0 and cand < best:  # overflow guard
                best = cand

        # Early termination: if d >= best, no further improvement possible
        if d >= best:
            break

        # Relax neighbors
        lo = indptr[u]
        hi = indptr[u + 1]
        for k in range(lo, hi):
            v = indices[k]
            w = weights[k]
            nd = d + w
            if nd >= 0 and nd < dist[v]:  # overflow guard
                dist[v] = nd
                if heap_size < heap_cap:
                    heap_size = _heap_push(heap_keys, heap_vals, heap_size, nd, v)

    return best


# ============================================================================
# Python wrapper (source/target attachment + dtype enforcement)
# ============================================================================

def _source_attachments(g, source: int, t_dep: int):
    """Compute (node_id, elapsed) pairs for source-attached D-events."""
    res_nodes = []
    res_dists = []

    # Source stop itself
    lo = int(g.stop_dep_indptr[source])
    hi = int(g.stop_dep_indptr[source + 1])
    if hi > lo:
        times = g.stop_dep_time[lo:hi]
        k = int(np.searchsorted(times, t_dep, side="left"))
        for j in range(lo + k, hi):
            res_nodes.append(int(g.stop_dep_node[j]))
            res_dists.append(int(g.stop_dep_time[j]) - t_dep)

    # Walking neighbours
    wlo = int(g.walk_indptr[source])
    whi = int(g.walk_indptr[source + 1])
    for wk in range(wlo, whi):
        sp = int(g.walk_indices[wk])
        wmin = int(g.walk_weights[wk])
        if sp == source:
            continue
        lo2 = int(g.stop_dep_indptr[sp])
        hi2 = int(g.stop_dep_indptr[sp + 1])
        if hi2 <= lo2:
            continue
        t_threshold = t_dep + wmin
        times = g.stop_dep_time[lo2:hi2]
        k = int(np.searchsorted(times, t_threshold, side="left"))
        if lo2 + k < hi2:
            res_nodes.append(int(g.stop_dep_node[lo2 + k]))
            res_dists.append(int(g.stop_dep_time[lo2 + k]) - t_dep)

    return res_nodes, res_dists


def _target_attachments(g, dest: int):
    """Compute target-attached A-event nodes and their walk-to-dest extras."""
    out_nodes = []
    out_extras = []
    seen = {}

    # A-events at dest
    lo = int(g.stop_arr_indptr[dest])
    hi = int(g.stop_arr_indptr[dest + 1])
    for j in range(lo, hi):
        n = int(g.stop_arr_node[j])
        seen[n] = 0
        out_nodes.append(n)
        out_extras.append(0)

    # A-events at walking-neighbours of dest (reverse walk)
    rwlo = int(g.rev_walk_indptr[dest])
    rwhi = int(g.rev_walk_indptr[dest + 1])
    for wk in range(rwlo, rwhi):
        sp = int(g.rev_walk_indices[wk])
        wmin = int(g.rev_walk_weights[wk])
        if sp == dest:
            continue
        lo2 = int(g.stop_arr_indptr[sp])
        hi2 = int(g.stop_arr_indptr[sp + 1])
        for j in range(lo2, hi2):
            n = int(g.stop_arr_node[j])
            if n not in seen or wmin < seen[n]:
                seen[n] = wmin
                out_nodes.append(n)
                out_extras.append(wmin)

    # Deduplicate keeping smallest extra per node
    final_map = {}
    for i in range(len(out_nodes)):
        n = out_nodes[i]
        e = out_extras[i]
        if n not in final_map or e < final_map[n]:
            final_map[n] = e

    fn = np.array(list(final_map.keys()), dtype=np.int32)
    fe = np.array(list(final_map.values()), dtype=np.int32)
    return fn, fe


def _pure_walk_arrival(g, source: int, dest: int, t_dep: int):
    """Return earliest arrival via pure walking (no transit), or None."""
    if source == dest:
        return t_dep
    wlo = int(g.walk_indptr[source])
    whi = int(g.walk_indptr[source + 1])
    if whi <= wlo:
        return None
    targets = g.walk_indices[wlo:whi]
    weights_w = g.walk_weights[wlo:whi]
    mask = targets == dest
    if not mask.any():
        return None
    wmin = int(weights_w[mask].min())
    return t_dep + wmin


def teeg_dijkstra_compiled(g, source: int, dest: int, t_dep: int):
    """Numba-JIT compiled forward Dijkstra on TEEG. Returns earliest arrival minute or None.

    Parameters
    ----------
    g : TEEG dataclass (from Route_Planner.teeg.builder)
    source, dest : int stop IDs
    t_dep : int departure time in minutes
    """
    # Source/target attachment (Python)
    src_nodes_list, src_dists_list = _source_attachments(g, source, t_dep)
    tgt_nodes_arr, tgt_extras_arr = _target_attachments(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)

    if not src_nodes_list and pure_walk is None:
        return None

    if not src_nodes_list:
        return pure_walk

    # Prepare numpy arrays for JIT
    src_nodes = np.array(src_nodes_list, dtype=np.int32)
    src_dists = np.array(src_dists_list, dtype=np.int32)

    n_nodes = np.int32(g.n_nodes)

    # Build target mask (dense bool array for O(1) lookup in JIT)
    tgt_mask = np.zeros(n_nodes, dtype=np.bool_)
    tgt_extra_dense = np.full(n_nodes, INF_I32, dtype=np.int32)
    for i in range(len(tgt_nodes_arr)):
        nd = tgt_nodes_arr[i]
        tgt_mask[nd] = True
        tgt_extra_dense[nd] = tgt_extras_arr[i]

    # Dtype enforcement
    indptr = g.indptr
    indices = g.indices
    weights = g.weights
    assert indptr.dtype == np.int32, f"indptr dtype {indptr.dtype}"
    assert indices.dtype == np.int32, f"indices dtype {indices.dtype}"
    assert weights.dtype == np.int32, f"weights dtype {weights.dtype}"

    # Initial best from pure walk
    if pure_walk is not None:
        best_init = np.int32(pure_walk - t_dep)
    else:
        best_init = INF_I32

    # Call JIT core
    best_elapsed = _dijkstra_core(
        indptr, indices, weights, n_nodes,
        src_nodes, src_dists,
        tgt_nodes_arr, tgt_extras_arr,
        tgt_mask, tgt_extra_dense,
        best_init,
    )

    if best_elapsed >= INF_I32:
        return None
    return int(t_dep + best_elapsed)
