"""Numba-compiled TEEG Dijkstra/ALT query kernels.

This module mirrors Route_Planner.teeg.query without modifying that package.
The hot shortest-path loop, manual min-heap, per-query attachments, and ALT
heuristic table are @njit(cache=True) compiled.  Public wrappers coerce CSR and
label arrays to contiguous int32 arrays and return the same Python contract as
the reference functions: earliest-arrival minute, or None if unreachable.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)
LABEL_INF = np.int32(32_000)


@njit(cache=True)
def _lower_bound_i16(arr, lo, hi, value):
    left = lo
    right = hi
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def _heap_push(heap_key, heap_dist, heap_node, size, key, dist, node):
    i = size
    heap_key[i] = key
    heap_dist[i] = dist
    heap_node[i] = node
    size += 1
    while i > 0:
        p = (i - 1) // 2
        pk = heap_key[p]
        if pk < key or (pk == key and heap_node[p] <= node):
            break
        heap_key[i] = heap_key[p]
        heap_dist[i] = heap_dist[p]
        heap_node[i] = heap_node[p]
        i = p
    heap_key[i] = key
    heap_dist[i] = dist
    heap_node[i] = node
    return size


@njit(cache=True)
def _heap_pop(heap_key, heap_dist, heap_node, size):
    out_key = heap_key[0]
    out_dist = heap_dist[0]
    out_node = heap_node[0]
    size -= 1
    if size > 0:
        key = heap_key[size]
        dist = heap_dist[size]
        node = heap_node[size]
        i = 0
        while True:
            left = 2 * i + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if right < size:
                lk = heap_key[left]
                rk = heap_key[right]
                if rk < lk or (rk == lk and heap_node[right] < heap_node[left]):
                    child = right
            ck = heap_key[child]
            if ck > key or (ck == key and heap_node[child] >= node):
                break
            heap_key[i] = heap_key[child]
            heap_dist[i] = heap_dist[child]
            heap_node[i] = heap_node[child]
            i = child
        heap_key[i] = key
        heap_dist[i] = dist
        heap_node[i] = node
    return out_key, out_dist, out_node, size


@njit(cache=True)
def _direct_walk_arrival(source, dest, t_dep, walk_indptr, walk_indices, walk_weights):
    if source == dest:
        return t_dep
    best = INF_I32
    lo = walk_indptr[source]
    hi = walk_indptr[source + 1]
    for k in range(lo, hi):
        if walk_indices[k] == dest and walk_weights[k] < best:
            best = walk_weights[k]
    if best >= INF_I32:
        return INF_I32
    return t_dep + best


@njit(cache=True)
def _fill_target_extra(dest, target_extra, target_seen, query_stamp, stop_arr_indptr, stop_arr_node,
                       rev_walk_indptr, rev_walk_indices, rev_walk_weights):
    any_target = False
    lo = stop_arr_indptr[dest]
    hi = stop_arr_indptr[dest + 1]
    for j in range(lo, hi):
        n = stop_arr_node[j]
        if target_seen[n] != query_stamp or target_extra[n] > 0:
            target_seen[n] = query_stamp
            target_extra[n] = 0
        any_target = True

    wlo = rev_walk_indptr[dest]
    whi = rev_walk_indptr[dest + 1]
    for wk in range(wlo, whi):
        sp = rev_walk_indices[wk]
        wmin = rev_walk_weights[wk]
        if sp == dest:
            continue
        alo = stop_arr_indptr[sp]
        ahi = stop_arr_indptr[sp + 1]
        for j in range(alo, ahi):
            n = stop_arr_node[j]
            if target_seen[n] != query_stamp or wmin < target_extra[n]:
                target_seen[n] = query_stamp
                target_extra[n] = wmin
                any_target = True
    return any_target


@njit(cache=True)
def _compute_h_stop(dest, h_stop, d_to, d_from):
    n_stops = h_stop.shape[0]
    n_landmarks = d_from.shape[0]
    for s in range(n_stops):
        best = np.int32(0)
        for lm in range(n_landmarks):
            term1 = d_to[s, lm] - d_to[dest, lm]
            if term1 > best:
                best = term1
            term2 = d_from[lm, dest] - d_from[lm, s]
            if term2 > best:
                best = term2
        h_stop[s] = best
    h_stop[dest] = 0


@njit(cache=True)
def _teeg_alt_core_njit(
    source, dest, t_dep,
    indptr, indices, weights, node_stop,
    stop_dep_indptr, stop_dep_time, stop_dep_node,
    stop_arr_indptr, stop_arr_node,
    walk_indptr, walk_indices, walk_weights,
    rev_walk_indptr, rev_walk_indices, rev_walk_weights,
    d_to, d_from,
    dist, target_extra, dist_seen, target_seen, query_stamp,
    h_stop, heap_key, heap_dist, heap_node,
):

    pure_walk = _direct_walk_arrival(source, dest, t_dep, walk_indptr, walk_indices, walk_weights)
    best_elapsed = LABEL_INF
    if pure_walk < INF_I32:
        best_elapsed = pure_walk - t_dep

    any_target = _fill_target_extra(dest, target_extra, target_seen, query_stamp,
                                    stop_arr_indptr, stop_arr_node,
                                    rev_walk_indptr, rev_walk_indices, rev_walk_weights)
    if not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    _compute_h_stop(dest, h_stop, d_to, d_from)

    heap_size = np.int32(0)
    any_source = False

    # Source stop departure events at or after t_dep.
    lo = stop_dep_indptr[source]
    hi = stop_dep_indptr[source + 1]
    j = _lower_bound_i16(stop_dep_time, lo, hi, t_dep)
    for p in range(j, hi):
        n = stop_dep_node[p]
        elapsed = np.int32(stop_dep_time[p]) - t_dep
        if dist_seen[n] != query_stamp or elapsed < dist[n]:
            dist_seen[n] = query_stamp
            dist[n] = elapsed
            key = elapsed + h_stop[node_stop[n]]
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n)
            any_source = True

    # First feasible departure at each walk-neighbour.
    wlo = walk_indptr[source]
    whi = walk_indptr[source + 1]
    for wk in range(wlo, whi):
        sp = walk_indices[wk]
        wmin = walk_weights[wk]
        if sp == source:
            continue
        slo = stop_dep_indptr[sp]
        shi = stop_dep_indptr[sp + 1]
        if shi <= slo:
            continue
        threshold = t_dep + wmin
        p = _lower_bound_i16(stop_dep_time, slo, shi, threshold)
        if p < shi:
            n = stop_dep_node[p]
            elapsed = np.int32(stop_dep_time[p]) - t_dep
            if dist_seen[n] != query_stamp or elapsed < dist[n]:
                dist_seen[n] = query_stamp
                dist[n] = elapsed
                key = elapsed + h_stop[node_stop[n]]
                if heap_size >= heap_key.shape[0]:
                    return np.int32(-2)
                heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n)
                any_source = True

    if (not any_source or not any_target) and best_elapsed >= LABEL_INF:
        return INF_I32
    if not any_source or not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    while heap_size > 0:
        f, d, u, heap_size = _heap_pop(heap_key, heap_dist, heap_node, heap_size)
        if dist_seen[u] != query_stamp or d > dist[u]:
            continue
        if target_seen[u] == query_stamp:
            extra = target_extra[u]
        else:
            extra = INF_I32
        if extra < INF_I32:
            cand_elapsed = d + extra
            if cand_elapsed < best_elapsed:
                best_elapsed = cand_elapsed
        if f >= best_elapsed:
            continue
        elo = indptr[u]
        ehi = indptr[u + 1]
        for k in range(elo, ehi):
            v = indices[k]
            nd = d + weights[k]
            if dist_seen[v] != query_stamp or nd < dist[v]:
                dist_seen[v] = query_stamp
                dist[v] = nd
                key = nd + h_stop[node_stop[v]]
                if heap_size >= heap_key.shape[0]:
                    return np.int32(-2)
                heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, key, nd, v)

    if best_elapsed >= LABEL_INF:
        return INF_I32
    return t_dep + best_elapsed


@njit(cache=True)
def _teeg_dijkstra_core_njit(
    source, dest, t_dep,
    indptr, indices, weights,
    stop_dep_indptr, stop_dep_time, stop_dep_node,
    stop_arr_indptr, stop_arr_node,
    walk_indptr, walk_indices, walk_weights,
    rev_walk_indptr, rev_walk_indices, rev_walk_weights,
    dist, target_extra, dist_seen, target_seen, query_stamp,
    heap_key, heap_dist, heap_node,
):

    pure_walk = _direct_walk_arrival(source, dest, t_dep, walk_indptr, walk_indices, walk_weights)
    best_elapsed = LABEL_INF
    if pure_walk < INF_I32:
        best_elapsed = pure_walk - t_dep

    any_target = _fill_target_extra(dest, target_extra, target_seen, query_stamp,
                                    stop_arr_indptr, stop_arr_node,
                                    rev_walk_indptr, rev_walk_indices, rev_walk_weights)
    if not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed
    heap_size = np.int32(0)
    any_source = False

    lo = stop_dep_indptr[source]
    hi = stop_dep_indptr[source + 1]
    j = _lower_bound_i16(stop_dep_time, lo, hi, t_dep)
    for p in range(j, hi):
        n = stop_dep_node[p]
        elapsed = np.int32(stop_dep_time[p]) - t_dep
        if dist_seen[n] != query_stamp or elapsed < dist[n]:
            dist_seen[n] = query_stamp
            dist[n] = elapsed
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n)
            any_source = True

    wlo = walk_indptr[source]
    whi = walk_indptr[source + 1]
    for wk in range(wlo, whi):
        sp = walk_indices[wk]
        wmin = walk_weights[wk]
        if sp == source:
            continue
        slo = stop_dep_indptr[sp]
        shi = stop_dep_indptr[sp + 1]
        if shi <= slo:
            continue
        threshold = t_dep + wmin
        p = _lower_bound_i16(stop_dep_time, slo, shi, threshold)
        if p < shi:
            n = stop_dep_node[p]
            elapsed = np.int32(stop_dep_time[p]) - t_dep
            if dist_seen[n] != query_stamp or elapsed < dist[n]:
                dist_seen[n] = query_stamp
                dist[n] = elapsed
                if heap_size >= heap_key.shape[0]:
                    return np.int32(-2)
                heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n)
                any_source = True

    if (not any_source or not any_target) and best_elapsed >= LABEL_INF:
        return INF_I32
    if not any_source or not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    while heap_size > 0:
        f, d, u, heap_size = _heap_pop(heap_key, heap_dist, heap_node, heap_size)
        if dist_seen[u] != query_stamp or d > dist[u]:
            continue
        if target_seen[u] == query_stamp:
            extra = target_extra[u]
        else:
            extra = INF_I32
        if extra < INF_I32:
            cand_elapsed = d + extra
            if cand_elapsed < best_elapsed:
                best_elapsed = cand_elapsed
        if d >= best_elapsed:
            continue
        elo = indptr[u]
        ehi = indptr[u + 1]
        for k in range(elo, ehi):
            v = indices[k]
            nd = d + weights[k]
            if dist_seen[v] != query_stamp or nd < dist[v]:
                dist_seen[v] = query_stamp
                dist[v] = nd
                if heap_size >= heap_key.shape[0]:
                    return np.int32(-2)
                heap_size = _heap_push(heap_key, heap_dist, heap_node, heap_size, nd, nd, v)

    if best_elapsed >= LABEL_INF:
        return INF_I32
    return t_dep + best_elapsed


def _as_i32_c(a, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype != np.int32 and arr.size:
        assert int(arr.min()) >= np.iinfo(np.int32).min, name
        assert int(arr.max()) <= np.iinfo(np.int32).max, name
    out = np.ascontiguousarray(arr, dtype=np.int32)
    assert out.dtype == np.int32, name
    return out


def _as_i16_c(a, name: str) -> np.ndarray:
    out = np.ascontiguousarray(a)
    assert out.dtype in (np.int16, np.int32), name
    return out


_ALT_WORK_CACHE = None
_DIJK_WORK_CACHE = None
_ALT_QUERY_STAMP = 0
_DIJK_QUERY_STAMP = 0

def _alloc_work(n_nodes: int, n_stops: int, heap_capacity: int):
    dist = np.empty(n_nodes, dtype=np.int32)
    target_extra = np.empty(n_nodes, dtype=np.int32)
    dist_seen = np.zeros(n_nodes, dtype=np.int32)
    target_seen = np.zeros(n_nodes, dtype=np.int32)
    h_stop = np.empty(n_stops, dtype=np.int32)
    heap_key = np.empty(heap_capacity, dtype=np.int32)
    heap_dist = np.empty(heap_capacity, dtype=np.int32)
    heap_node = np.empty(heap_capacity, dtype=np.int32)
    return dist, target_extra, dist_seen, target_seen, h_stop, heap_key, heap_dist, heap_node


def teeg_alt_compiled(g, source: int, dest: int, t_dep: int, landmark_labels):
    """Compiled ALT (A*) on a TEEG graph.

    landmark_labels must expose int32-compatible d_to (n_stops, L) and d_from
    (L, n_stops), as Route_Planner.teeg.query.LandmarkLabels does.
    """
    assert 0 <= int(source) < int(g.n_stops)
    assert 0 <= int(dest) < int(g.n_stops)
    assert 0 <= int(source) < int(g.n_stops)
    assert 0 <= int(dest) < int(g.n_stops)
    indptr = _as_i32_c(g.indptr, "indptr")
    indices = _as_i32_c(g.indices, "indices")
    weights = _as_i32_c(g.weights, "weights")
    node_stop = _as_i32_c(g.node_stop, "node_stop")
    stop_dep_indptr = _as_i32_c(g.stop_dep_indptr, "stop_dep_indptr")
    stop_dep_time = _as_i16_c(g.stop_dep_time, "stop_dep_time")
    stop_dep_node = _as_i32_c(g.stop_dep_node, "stop_dep_node")
    stop_arr_indptr = _as_i32_c(g.stop_arr_indptr, "stop_arr_indptr")
    stop_arr_node = _as_i32_c(g.stop_arr_node, "stop_arr_node")
    walk_indptr = _as_i32_c(g.walk_indptr, "walk_indptr")
    walk_indices = _as_i32_c(g.walk_indices, "walk_indices")
    walk_weights = _as_i32_c(g.walk_weights, "walk_weights")
    rev_walk_indptr = _as_i32_c(g.rev_walk_indptr, "rev_walk_indptr")
    rev_walk_indices = _as_i32_c(g.rev_walk_indices, "rev_walk_indices")
    rev_walk_weights = _as_i32_c(g.rev_walk_weights, "rev_walk_weights")
    d_to = _as_i32_c(landmark_labels.d_to, "landmark_labels.d_to")
    d_from = _as_i32_c(landmark_labels.d_from, "landmark_labels.d_from")
    assert d_to.shape[0] == int(g.n_stops)
    assert d_from.shape[1] == int(g.n_stops)
    assert d_to.shape[1] == d_from.shape[0]

    heap_capacity = int(len(indices) + len(stop_dep_node) + 1)
    global _ALT_WORK_CACHE, _ALT_QUERY_STAMP
    cache_key = (id(g), int(g.n_nodes), int(g.n_stops), heap_capacity)
    if _ALT_WORK_CACHE is None or _ALT_WORK_CACHE[0] != cache_key:
        _ALT_WORK_CACHE = (cache_key, _alloc_work(int(g.n_nodes), int(g.n_stops), heap_capacity))
    _ALT_QUERY_STAMP += 1
    if _ALT_QUERY_STAMP >= np.iinfo(np.int32).max - 1:
        _ALT_WORK_CACHE[1][2].fill(0)
        _ALT_WORK_CACHE[1][3].fill(0)
        _ALT_QUERY_STAMP = 1
    dist, target_extra, dist_seen, target_seen, h_stop, heap_key, heap_dist, heap_node = _ALT_WORK_CACHE[1]
    result = _teeg_alt_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep),
        indptr, indices, weights, node_stop,
        stop_dep_indptr, stop_dep_time, stop_dep_node,
        stop_arr_indptr, stop_arr_node,
        walk_indptr, walk_indices, walk_weights,
        rev_walk_indptr, rev_walk_indices, rev_walk_weights,
        d_to, d_from,
        dist, target_extra, dist_seen, target_seen, np.int32(_ALT_QUERY_STAMP),
        h_stop, heap_key, heap_dist, heap_node,
    )
    if int(result) == -2:
        raise MemoryError("TEEG compiled ALT heap capacity exhausted")
    if int(result) >= int(INF_I32):
        return None
    return int(result)


def teeg_dijkstra_compiled(g, source: int, dest: int, t_dep: int):
    """Compiled forward Dijkstra on a TEEG graph."""
    indptr = _as_i32_c(g.indptr, "indptr")
    indices = _as_i32_c(g.indices, "indices")
    weights = _as_i32_c(g.weights, "weights")
    stop_dep_indptr = _as_i32_c(g.stop_dep_indptr, "stop_dep_indptr")
    stop_dep_time = _as_i16_c(g.stop_dep_time, "stop_dep_time")
    stop_dep_node = _as_i32_c(g.stop_dep_node, "stop_dep_node")
    stop_arr_indptr = _as_i32_c(g.stop_arr_indptr, "stop_arr_indptr")
    stop_arr_node = _as_i32_c(g.stop_arr_node, "stop_arr_node")
    walk_indptr = _as_i32_c(g.walk_indptr, "walk_indptr")
    walk_indices = _as_i32_c(g.walk_indices, "walk_indices")
    walk_weights = _as_i32_c(g.walk_weights, "walk_weights")
    rev_walk_indptr = _as_i32_c(g.rev_walk_indptr, "rev_walk_indptr")
    rev_walk_indices = _as_i32_c(g.rev_walk_indices, "rev_walk_indices")
    rev_walk_weights = _as_i32_c(g.rev_walk_weights, "rev_walk_weights")

    heap_capacity = int(len(indices) + len(stop_dep_node) + 1)
    global _DIJK_WORK_CACHE, _DIJK_QUERY_STAMP
    cache_key = (id(g), int(g.n_nodes), heap_capacity)
    if _DIJK_WORK_CACHE is None or _DIJK_WORK_CACHE[0] != cache_key:
        dist = np.empty(int(g.n_nodes), dtype=np.int32)
        target_extra = np.empty(int(g.n_nodes), dtype=np.int32)
        dist_seen = np.zeros(int(g.n_nodes), dtype=np.int32)
        target_seen = np.zeros(int(g.n_nodes), dtype=np.int32)
        heap_key = np.empty(heap_capacity, dtype=np.int32)
        heap_dist = np.empty(heap_capacity, dtype=np.int32)
        heap_node = np.empty(heap_capacity, dtype=np.int32)
        _DIJK_WORK_CACHE = (cache_key, (dist, target_extra, dist_seen, target_seen, heap_key, heap_dist, heap_node))
    _DIJK_QUERY_STAMP += 1
    if _DIJK_QUERY_STAMP >= np.iinfo(np.int32).max - 1:
        _DIJK_WORK_CACHE[1][2].fill(0)
        _DIJK_WORK_CACHE[1][3].fill(0)
        _DIJK_QUERY_STAMP = 1
    dist, target_extra, dist_seen, target_seen, heap_key, heap_dist, heap_node = _DIJK_WORK_CACHE[1]
    result = _teeg_dijkstra_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep),
        indptr, indices, weights,
        stop_dep_indptr, stop_dep_time, stop_dep_node,
        stop_arr_indptr, stop_arr_node,
        walk_indptr, walk_indices, walk_weights,
        rev_walk_indptr, rev_walk_indices, rev_walk_weights,
        dist, target_extra, dist_seen, target_seen, np.int32(_DIJK_QUERY_STAMP),
        heap_key, heap_dist, heap_node,
    )
    if int(result) == -2:
        raise MemoryError("TEEG compiled Dijkstra heap capacity exhausted")
    if int(result) >= int(INF_I32):
        return None
    return int(result)





