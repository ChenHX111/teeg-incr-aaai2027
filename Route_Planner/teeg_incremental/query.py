"""Tombstone- and overlay-aware ALT/Dijkstra query kernels for TEEG-INCR.

These are the modified `_teeg_alt_core_njit` / `_teeg_dijkstra_core_njit` that
DESIGN.md §3.5 / §6.4.3 describe.  The static counterparts live in
`Route_Planner/teeg_compiled/query.py`; this file is intentionally a near-clone
with three additions per relaxation:

  1. `if edge_alive[k] == 0: continue`
  2. `if node_alive[v] == 0: continue`
  3. main→overlay and overlay→{overlay,main} edge enumeration

Overlay node ids are encoded as -(ov+1) in indices arrays and PQ entries.
The `dist` array is sized `V + overlay_capacity_nodes`; for is_ov nodes we
index into `dist[V + ov_id]`.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)
LABEL_INF = np.int32(32_000)


# ---------- helpers (mirror of teeg_compiled.query) ----------

@njit(cache=True)
def _lb_i16(arr, lo, hi, v):
    left = lo; right = hi
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < v:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def _lb_i32(arr, lo, hi, v):
    left = lo; right = hi
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < v:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def _hpush(hk, hd, hn, sz, k, d, n):
    i = sz
    hk[i] = k; hd[i] = d; hn[i] = n
    sz += 1
    while i > 0:
        p = (i - 1) // 2
        pk = hk[p]
        if pk < k or (pk == k and hn[p] <= n):
            break
        hk[i] = hk[p]; hd[i] = hd[p]; hn[i] = hn[p]
        i = p
    hk[i] = k; hd[i] = d; hn[i] = n
    return sz


@njit(cache=True)
def _hpop(hk, hd, hn, sz):
    ok = hk[0]; od = hd[0]; on = hn[0]
    sz -= 1
    if sz > 0:
        k = hk[sz]; d = hd[sz]; n = hn[sz]
        i = 0
        while True:
            left = 2 * i + 1
            if left >= sz:
                break
            right = left + 1
            child = left
            if right < sz:
                lk = hk[left]; rk = hk[right]
                if rk < lk or (rk == lk and hn[right] < hn[left]):
                    child = right
            ck = hk[child]
            if ck > k or (ck == k and hn[child] >= n):
                break
            hk[i] = hk[child]; hd[i] = hd[child]; hn[i] = hn[child]
            i = child
        hk[i] = k; hd[i] = d; hn[i] = n
    return ok, od, on, sz


@njit(cache=True)
def _direct_walk_arr(source, dest, t_dep, walk_indptr, walk_indices, walk_weights):
    if source == dest:
        return t_dep
    best = INF_I32
    lo = walk_indptr[source]; hi = walk_indptr[source + 1]
    for k in range(lo, hi):
        if walk_indices[k] == dest and walk_weights[k] < best:
            best = walk_weights[k]
    if best >= INF_I32:
        return INF_I32
    return t_dep + best


@njit(cache=True)
def _fill_target(dest, target_extra, target_seen, query_stamp,
                 stop_arr_indptr, stop_arr_node,
                 rev_walk_indptr, rev_walk_indices, rev_walk_weights,
                 node_alive):
    """Mark target A-events (with required final-walk extra). Tombstoned A-events
    are not eligible targets — but stop-level walking reachability is preserved
    via the rev-walk pass since walking is tombstone-immune (I-5)."""
    any_target = False
    lo = stop_arr_indptr[dest]; hi = stop_arr_indptr[dest + 1]
    for j in range(lo, hi):
        n = stop_arr_node[j]
        if not node_alive[n]:
            continue
        if target_seen[n] != query_stamp or target_extra[n] > 0:
            target_seen[n] = query_stamp
            target_extra[n] = 0
        any_target = True
    wlo = rev_walk_indptr[dest]; whi = rev_walk_indptr[dest + 1]
    for wk in range(wlo, whi):
        sp = rev_walk_indices[wk]; wmin = rev_walk_weights[wk]
        if sp == dest:
            continue
        alo = stop_arr_indptr[sp]; ahi = stop_arr_indptr[sp + 1]
        for j in range(alo, ahi):
            n = stop_arr_node[j]
            if not node_alive[n]:
                continue
            if target_seen[n] != query_stamp or wmin < target_extra[n]:
                target_seen[n] = query_stamp
                target_extra[n] = wmin
                any_target = True
    return any_target


@njit(cache=True)
def _h_stop(dest, h_stop, d_to, d_from):
    n_stops = h_stop.shape[0]
    L = d_from.shape[0]
    for s in range(n_stops):
        best = np.int32(0)
        for lm in range(L):
            # Both term1 and term2 are inadmissible whenever any operand is the
            # LABEL_INF sentinel: e.g. d_to[s,lm] = INF and d_to[dest,lm] finite
            # gives term1 ~ +INF.  We skip such landmarks rather than relying on
            # the original code's implicit clamp (which is only safe when the
            # unreachable side is the destination).  This closes a pre-existing
            # admissibility hole exposed by post-compaction queries on a graph
            # whose connected component differs from the LB-graph's (DESIGN
            # §6.1 / §6.4.1).
            dts = d_to[s, lm]
            dtd = d_to[dest, lm]
            if dts < LABEL_INF and dtd < LABEL_INF:
                t1 = dts - dtd
                if t1 > best:
                    best = t1
            dfd = d_from[lm, dest]
            dfs = d_from[lm, s]
            if dfd < LABEL_INF and dfs < LABEL_INF:
                t2 = dfd - dfs
                if t2 > best:
                    best = t2
        h_stop[s] = best
    h_stop[dest] = 0


# ---------- core ALT-INCR ----------

@njit(cache=True)
def _teeg_alt_incr_core_njit(
    source, dest, t_dep,
    indptr, indices, weights, node_stop,
    stop_dep_indptr, stop_dep_time, stop_dep_node,
    stop_arr_indptr, stop_arr_node,
    walk_indptr, walk_indices, walk_weights,
    rev_walk_indptr, rev_walk_indices, rev_walk_weights,
    d_to, d_from,
    # incremental:
    node_alive, edge_alive,
    overlay_n_nodes,
    overlay_node_stop, overlay_node_alive,
    overlay_indptr, overlay_indices, overlay_weights,
    overlay_fwd_from_main_indptr, overlay_fwd_from_main_indices, overlay_fwd_from_main_weights,
    overlay_stop_dep_indptr, overlay_stop_dep_time, overlay_stop_dep_node,
    # work arrays:
    dist, target_extra, dist_seen, target_seen, query_stamp,
    h_stop, heap_key, heap_dist, heap_node,
    V,
):
    pure_walk = _direct_walk_arr(source, dest, t_dep, walk_indptr, walk_indices, walk_weights)
    best_elapsed = LABEL_INF
    if pure_walk < INF_I32:
        best_elapsed = pure_walk - t_dep

    any_target = _fill_target(dest, target_extra, target_seen, query_stamp,
                              stop_arr_indptr, stop_arr_node,
                              rev_walk_indptr, rev_walk_indices, rev_walk_weights,
                              node_alive)
    if not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    _h_stop(dest, h_stop, d_to, d_from)

    heap_size = np.int32(0)
    any_source = False

    # ---- source attachment: main D-events at src ----
    lo = stop_dep_indptr[source]; hi = stop_dep_indptr[source + 1]
    j = _lb_i16(stop_dep_time, lo, hi, t_dep)
    for p in range(j, hi):
        n = stop_dep_node[p]
        if not node_alive[n]:
            continue
        elapsed = np.int32(stop_dep_time[p]) - t_dep
        if dist_seen[n] != query_stamp or elapsed < dist[n]:
            dist_seen[n] = query_stamp
            dist[n] = elapsed
            key = elapsed + h_stop[node_stop[n]]
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n)
            any_source = True

    # overlay D-events at src
    olo = overlay_stop_dep_indptr[source]; ohi = overlay_stop_dep_indptr[source + 1]
    op = _lb_i32(overlay_stop_dep_time, olo, ohi, t_dep)
    for p in range(op, ohi):
        n_raw = overlay_stop_dep_node[p]   # already negative-encoded
        ov_idx = -n_raw - 1
        if not overlay_node_alive[ov_idx]:
            continue
        elapsed = overlay_stop_dep_time[p] - t_dep
        slot = V + ov_idx
        if dist_seen[slot] != query_stamp or elapsed < dist[slot]:
            dist_seen[slot] = query_stamp
            dist[slot] = elapsed
            key = elapsed + h_stop[overlay_node_stop[ov_idx]]
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n_raw)
            any_source = True

    # walk-neighbours: main + overlay D-events at each neighbour
    wlo = walk_indptr[source]; whi = walk_indptr[source + 1]
    for wk in range(wlo, whi):
        sp = walk_indices[wk]; wmin = walk_weights[wk]
        if sp == source:
            continue
        threshold = t_dep + wmin
        # main
        slo = stop_dep_indptr[sp]; shi = stop_dep_indptr[sp + 1]
        if shi > slo:
            p = _lb_i16(stop_dep_time, slo, shi, threshold)
            # advance over tombstoned events
            while p < shi and not node_alive[stop_dep_node[p]]:
                p += 1
            if p < shi:
                n = stop_dep_node[p]
                elapsed = np.int32(stop_dep_time[p]) - t_dep
                if dist_seen[n] != query_stamp or elapsed < dist[n]:
                    dist_seen[n] = query_stamp
                    dist[n] = elapsed
                    key = elapsed + h_stop[node_stop[n]]
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n)
                    any_source = True
        # overlay
        oslo = overlay_stop_dep_indptr[sp]; oshi = overlay_stop_dep_indptr[sp + 1]
        if oshi > oslo:
            p = _lb_i32(overlay_stop_dep_time, oslo, oshi, threshold)
            while p < oshi:
                n_raw = overlay_stop_dep_node[p]
                ov_idx = -n_raw - 1
                if overlay_node_alive[ov_idx]:
                    elapsed = overlay_stop_dep_time[p] - t_dep
                    slot = V + ov_idx
                    if dist_seen[slot] != query_stamp or elapsed < dist[slot]:
                        dist_seen[slot] = query_stamp
                        dist[slot] = elapsed
                        key = elapsed + h_stop[overlay_node_stop[ov_idx]]
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, elapsed, n_raw)
                        any_source = True
                    break
                p += 1

    if (not any_source or not any_target) and best_elapsed >= LABEL_INF:
        return INF_I32
    if not any_source or not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    while heap_size > 0:
        f, d, u_raw, heap_size = _hpop(heap_key, heap_dist, heap_node, heap_size)
        is_ov_u = u_raw < 0
        if is_ov_u:
            u_idx = -u_raw - 1
            slot_u = V + u_idx
            u_stop = overlay_node_stop[u_idx]
        else:
            u_idx = u_raw
            slot_u = u_raw
            u_stop = node_stop[u_raw]
        if dist_seen[slot_u] != query_stamp or d > dist[slot_u]:
            continue
        # target check (overlay nodes are never targets — reachable target main A-events
        # are still in target_seen; overlay A-events re-board into main and then become
        # targets via the static A-event)
        if not is_ov_u:
            if target_seen[u_idx] == query_stamp:
                extra = target_extra[u_idx]
                if extra < INF_I32:
                    cand = d + extra
                    if cand < best_elapsed:
                        best_elapsed = cand
        if f >= best_elapsed:
            continue
        # ---- enumerate out-edges ----
        if not is_ov_u:
            # case A: main → main
            elo = indptr[u_idx]; ehi = indptr[u_idx + 1]
            for k in range(elo, ehi):
                if not edge_alive[k]:
                    continue
                v = indices[k]
                if not node_alive[v]:
                    continue
                nd = d + weights[k]
                if dist_seen[v] != query_stamp or nd < dist[v]:
                    dist_seen[v] = query_stamp
                    dist[v] = nd
                    key = nd + h_stop[node_stop[v]]
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, nd, v)
            # case B: main → overlay
            olo = overlay_fwd_from_main_indptr[u_idx]; ohi = overlay_fwd_from_main_indptr[u_idx + 1]
            for k in range(olo, ohi):
                v_raw = overlay_fwd_from_main_indices[k]
                # always negative-encoded
                v_ov = -v_raw - 1
                if not overlay_node_alive[v_ov]:
                    continue
                nd = d + overlay_fwd_from_main_weights[k]
                slot_v = V + v_ov
                if dist_seen[slot_v] != query_stamp or nd < dist[slot_v]:
                    dist_seen[slot_v] = query_stamp
                    dist[slot_v] = nd
                    key = nd + h_stop[overlay_node_stop[v_ov]]
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, nd, v_raw)
        else:
            # cases C+D: overlay → {overlay,main}
            elo = overlay_indptr[u_idx]; ehi = overlay_indptr[u_idx + 1]
            for k in range(elo, ehi):
                v_raw = overlay_indices[k]
                if v_raw < 0:
                    v_ov = -v_raw - 1
                    if not overlay_node_alive[v_ov]:
                        continue
                    nd = d + overlay_weights[k]
                    slot_v = V + v_ov
                    if dist_seen[slot_v] != query_stamp or nd < dist[slot_v]:
                        dist_seen[slot_v] = query_stamp
                        dist[slot_v] = nd
                        key = nd + h_stop[overlay_node_stop[v_ov]]
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, nd, v_raw)
                else:
                    if not node_alive[v_raw]:
                        continue
                    nd = d + overlay_weights[k]
                    if dist_seen[v_raw] != query_stamp or nd < dist[v_raw]:
                        dist_seen[v_raw] = query_stamp
                        dist[v_raw] = nd
                        key = nd + h_stop[node_stop[v_raw]]
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, key, nd, v_raw)

    if best_elapsed >= LABEL_INF:
        return INF_I32
    return t_dep + best_elapsed


# ---------- core Dijkstra-INCR (oracle) ----------

@njit(cache=True)
def _teeg_dij_incr_core_njit(
    source, dest, t_dep,
    indptr, indices, weights,
    stop_dep_indptr, stop_dep_time, stop_dep_node,
    stop_arr_indptr, stop_arr_node,
    walk_indptr, walk_indices, walk_weights,
    rev_walk_indptr, rev_walk_indices, rev_walk_weights,
    node_alive, edge_alive,
    overlay_n_nodes,
    overlay_node_stop, overlay_node_alive,
    overlay_indptr, overlay_indices, overlay_weights,
    overlay_fwd_from_main_indptr, overlay_fwd_from_main_indices, overlay_fwd_from_main_weights,
    overlay_stop_dep_indptr, overlay_stop_dep_time, overlay_stop_dep_node,
    dist, target_extra, dist_seen, target_seen, query_stamp,
    heap_key, heap_dist, heap_node, V,
):
    pure_walk = _direct_walk_arr(source, dest, t_dep, walk_indptr, walk_indices, walk_weights)
    best_elapsed = LABEL_INF
    if pure_walk < INF_I32:
        best_elapsed = pure_walk - t_dep
    any_target = _fill_target(dest, target_extra, target_seen, query_stamp,
                              stop_arr_indptr, stop_arr_node,
                              rev_walk_indptr, rev_walk_indices, rev_walk_weights,
                              node_alive)
    if not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    heap_size = np.int32(0); any_source = False
    lo = stop_dep_indptr[source]; hi = stop_dep_indptr[source + 1]
    j = _lb_i16(stop_dep_time, lo, hi, t_dep)
    for p in range(j, hi):
        n = stop_dep_node[p]
        if not node_alive[n]:
            continue
        elapsed = np.int32(stop_dep_time[p]) - t_dep
        if dist_seen[n] != query_stamp or elapsed < dist[n]:
            dist_seen[n] = query_stamp; dist[n] = elapsed
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n)
            any_source = True
    olo = overlay_stop_dep_indptr[source]; ohi = overlay_stop_dep_indptr[source + 1]
    op = _lb_i32(overlay_stop_dep_time, olo, ohi, t_dep)
    for p in range(op, ohi):
        n_raw = overlay_stop_dep_node[p]
        ov_idx = -n_raw - 1
        if not overlay_node_alive[ov_idx]:
            continue
        elapsed = overlay_stop_dep_time[p] - t_dep
        slot = V + ov_idx
        if dist_seen[slot] != query_stamp or elapsed < dist[slot]:
            dist_seen[slot] = query_stamp; dist[slot] = elapsed
            if heap_size >= heap_key.shape[0]:
                return np.int32(-2)
            heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n_raw)
            any_source = True
    wlo = walk_indptr[source]; whi = walk_indptr[source + 1]
    for wk in range(wlo, whi):
        sp = walk_indices[wk]; wmin = walk_weights[wk]
        if sp == source:
            continue
        threshold = t_dep + wmin
        slo = stop_dep_indptr[sp]; shi = stop_dep_indptr[sp + 1]
        if shi > slo:
            p = _lb_i16(stop_dep_time, slo, shi, threshold)
            while p < shi and not node_alive[stop_dep_node[p]]:
                p += 1
            if p < shi:
                n = stop_dep_node[p]
                elapsed = np.int32(stop_dep_time[p]) - t_dep
                if dist_seen[n] != query_stamp or elapsed < dist[n]:
                    dist_seen[n] = query_stamp; dist[n] = elapsed
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n)
                    any_source = True
        oslo = overlay_stop_dep_indptr[sp]; oshi = overlay_stop_dep_indptr[sp + 1]
        if oshi > oslo:
            p = _lb_i32(overlay_stop_dep_time, oslo, oshi, threshold)
            while p < oshi:
                n_raw = overlay_stop_dep_node[p]
                ov_idx = -n_raw - 1
                if overlay_node_alive[ov_idx]:
                    elapsed = overlay_stop_dep_time[p] - t_dep
                    slot = V + ov_idx
                    if dist_seen[slot] != query_stamp or elapsed < dist[slot]:
                        dist_seen[slot] = query_stamp; dist[slot] = elapsed
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, elapsed, elapsed, n_raw)
                        any_source = True
                    break
                p += 1

    if (not any_source or not any_target) and best_elapsed >= LABEL_INF:
        return INF_I32
    if not any_source or not any_target:
        if best_elapsed >= LABEL_INF:
            return INF_I32
        return t_dep + best_elapsed

    while heap_size > 0:
        f, d, u_raw, heap_size = _hpop(heap_key, heap_dist, heap_node, heap_size)
        is_ov_u = u_raw < 0
        if is_ov_u:
            u_idx = -u_raw - 1; slot_u = V + u_idx
        else:
            u_idx = u_raw; slot_u = u_raw
        if dist_seen[slot_u] != query_stamp or d > dist[slot_u]:
            continue
        if not is_ov_u:
            if target_seen[u_idx] == query_stamp:
                extra = target_extra[u_idx]
                if extra < INF_I32:
                    cand = d + extra
                    if cand < best_elapsed:
                        best_elapsed = cand
        if d >= best_elapsed:
            continue
        if not is_ov_u:
            elo = indptr[u_idx]; ehi = indptr[u_idx + 1]
            for k in range(elo, ehi):
                if not edge_alive[k]:
                    continue
                v = indices[k]
                if not node_alive[v]:
                    continue
                nd = d + weights[k]
                if dist_seen[v] != query_stamp or nd < dist[v]:
                    dist_seen[v] = query_stamp; dist[v] = nd
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, nd, nd, v)
            olo = overlay_fwd_from_main_indptr[u_idx]; ohi = overlay_fwd_from_main_indptr[u_idx + 1]
            for k in range(olo, ohi):
                v_raw = overlay_fwd_from_main_indices[k]
                v_ov = -v_raw - 1
                if not overlay_node_alive[v_ov]:
                    continue
                nd = d + overlay_fwd_from_main_weights[k]
                slot_v = V + v_ov
                if dist_seen[slot_v] != query_stamp or nd < dist[slot_v]:
                    dist_seen[slot_v] = query_stamp; dist[slot_v] = nd
                    if heap_size >= heap_key.shape[0]:
                        return np.int32(-2)
                    heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, nd, nd, v_raw)
        else:
            elo = overlay_indptr[u_idx]; ehi = overlay_indptr[u_idx + 1]
            for k in range(elo, ehi):
                v_raw = overlay_indices[k]
                if v_raw < 0:
                    v_ov = -v_raw - 1
                    if not overlay_node_alive[v_ov]:
                        continue
                    nd = d + overlay_weights[k]
                    slot_v = V + v_ov
                    if dist_seen[slot_v] != query_stamp or nd < dist[slot_v]:
                        dist_seen[slot_v] = query_stamp; dist[slot_v] = nd
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, nd, nd, v_raw)
                else:
                    if not node_alive[v_raw]:
                        continue
                    nd = d + overlay_weights[k]
                    if dist_seen[v_raw] != query_stamp or nd < dist[v_raw]:
                        dist_seen[v_raw] = query_stamp; dist[v_raw] = nd
                        if heap_size >= heap_key.shape[0]:
                            return np.int32(-2)
                        heap_size = _hpush(heap_key, heap_dist, heap_node, heap_size, nd, nd, v_raw)

    if best_elapsed >= LABEL_INF:
        return INF_I32
    return t_dep + best_elapsed


# ---------- Python wrappers ----------

def _as_i32(a):  return np.ascontiguousarray(a, dtype=np.int32)
def _as_i16(a):
    out = np.ascontiguousarray(a)
    if out.dtype not in (np.int16, np.int32):
        out = out.astype(np.int32, copy=False)
    return out


_WORK = {}


def _alloc_work(state, kind: str, heap_capacity: int):
    V = state.n_nodes
    Vov = state.overlay_capacity_nodes
    n_total = V + Vov
    dist = np.empty(n_total, dtype=np.int32)
    target_extra = np.empty(V, dtype=np.int32)
    dist_seen = np.zeros(n_total, dtype=np.int32)
    target_seen = np.zeros(V, dtype=np.int32)
    h_stop = np.empty(state.n_stops, dtype=np.int32)
    heap_key = np.empty(heap_capacity, dtype=np.int32)
    heap_dist = np.empty(heap_capacity, dtype=np.int32)
    heap_node = np.empty(heap_capacity, dtype=np.int32)
    return dist, target_extra, dist_seen, target_seen, h_stop, heap_key, heap_dist, heap_node


def query_compiled(state, g, source: int, dest: int, t_dep: int, labels) -> int | None:
    """ALT-INCR query: tombstone- and overlay-aware compiled ALT.

    Returns earliest-arrival minute or None. Mirrors teeg_alt_compiled API.
    """
    indptr = _as_i32(g.indptr); indices = _as_i32(g.indices); weights = _as_i32(g.weights)
    node_stop = _as_i32(g.node_stop)
    stop_dep_indptr = _as_i32(g.stop_dep_indptr)
    stop_dep_time = _as_i16(g.stop_dep_time)
    stop_dep_node = _as_i32(g.stop_dep_node)
    stop_arr_indptr = _as_i32(g.stop_arr_indptr)
    stop_arr_node = _as_i32(g.stop_arr_node)
    walk_indptr = _as_i32(g.walk_indptr); walk_indices = _as_i32(g.walk_indices); walk_weights = _as_i32(g.walk_weights)
    rev_walk_indptr = _as_i32(g.rev_walk_indptr); rev_walk_indices = _as_i32(g.rev_walk_indices); rev_walk_weights = _as_i32(g.rev_walk_weights)
    d_to = _as_i32(labels.d_to); d_from = _as_i32(labels.d_from)

    V = int(g.n_nodes)
    heap_capacity = int(len(indices) + len(stop_dep_node) + state.overlay_n_edges + state.overlay_n_nodes + 64)
    cache_key = ('alt', id(state), id(g), V, state.overlay_capacity_nodes, heap_capacity)
    cached = _WORK.get(cache_key)
    if cached is None:
        cached = _alloc_work(state, 'alt', heap_capacity)
        _WORK[cache_key] = cached
    dist, target_extra, dist_seen, target_seen, h_stop, heap_key, heap_dist, heap_node = cached

    # query stamp lives globally per-cache-key
    stamp_key = ('stamp_alt', cache_key)
    stamp = _WORK.get(stamp_key, 0) + 1
    if stamp >= np.iinfo(np.int32).max - 1:
        dist_seen.fill(0); target_seen.fill(0); stamp = 1
    _WORK[stamp_key] = stamp

    res = _teeg_alt_incr_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep),
        indptr, indices, weights, node_stop,
        stop_dep_indptr, stop_dep_time, stop_dep_node,
        stop_arr_indptr, stop_arr_node,
        walk_indptr, walk_indices, walk_weights,
        rev_walk_indptr, rev_walk_indices, rev_walk_weights,
        d_to, d_from,
        state.node_alive, state.edge_alive,
        np.int32(state.overlay_n_nodes),
        state.overlay_node_stop, state.overlay_node_alive,
        state.overlay_indptr, state.overlay_indices, state.overlay_weights,
        state.overlay_fwd_from_main_indptr,
        state.overlay_fwd_from_main_indices,
        state.overlay_fwd_from_main_weights,
        state.overlay_stop_dep_indptr,
        state.overlay_stop_dep_time,
        state.overlay_stop_dep_node,
        dist, target_extra, dist_seen, target_seen, np.int32(stamp),
        h_stop, heap_key, heap_dist, heap_node, np.int32(V),
    )
    if int(res) == -2:
        raise MemoryError("TEEG-ALT-INCR heap exhausted")
    if int(res) >= int(INF_I32):
        return None
    return int(res)


def query_dijkstra_compiled(state, g, source: int, dest: int, t_dep: int) -> int | None:
    """Reference Dijkstra-INCR oracle (no heuristic) on the patched graph."""
    indptr = _as_i32(g.indptr); indices = _as_i32(g.indices); weights = _as_i32(g.weights)
    stop_dep_indptr = _as_i32(g.stop_dep_indptr)
    stop_dep_time = _as_i16(g.stop_dep_time)
    stop_dep_node = _as_i32(g.stop_dep_node)
    stop_arr_indptr = _as_i32(g.stop_arr_indptr)
    stop_arr_node = _as_i32(g.stop_arr_node)
    walk_indptr = _as_i32(g.walk_indptr); walk_indices = _as_i32(g.walk_indices); walk_weights = _as_i32(g.walk_weights)
    rev_walk_indptr = _as_i32(g.rev_walk_indptr); rev_walk_indices = _as_i32(g.rev_walk_indices); rev_walk_weights = _as_i32(g.rev_walk_weights)

    V = int(g.n_nodes)
    heap_capacity = int(len(indices) + len(stop_dep_node) + state.overlay_n_edges + state.overlay_n_nodes + 64)
    cache_key = ('dij', id(state), id(g), V, state.overlay_capacity_nodes, heap_capacity)
    cached = _WORK.get(cache_key)
    if cached is None:
        cached = _alloc_work(state, 'dij', heap_capacity)
        _WORK[cache_key] = cached
    dist, target_extra, dist_seen, target_seen, _h_stop_unused, heap_key, heap_dist, heap_node = cached
    stamp_key = ('stamp_dij', cache_key)
    stamp = _WORK.get(stamp_key, 0) + 1
    if stamp >= np.iinfo(np.int32).max - 1:
        dist_seen.fill(0); target_seen.fill(0); stamp = 1
    _WORK[stamp_key] = stamp

    res = _teeg_dij_incr_core_njit(
        np.int32(source), np.int32(dest), np.int32(t_dep),
        indptr, indices, weights,
        stop_dep_indptr, stop_dep_time, stop_dep_node,
        stop_arr_indptr, stop_arr_node,
        walk_indptr, walk_indices, walk_weights,
        rev_walk_indptr, rev_walk_indices, rev_walk_weights,
        state.node_alive, state.edge_alive,
        np.int32(state.overlay_n_nodes),
        state.overlay_node_stop, state.overlay_node_alive,
        state.overlay_indptr, state.overlay_indices, state.overlay_weights,
        state.overlay_fwd_from_main_indptr,
        state.overlay_fwd_from_main_indices,
        state.overlay_fwd_from_main_weights,
        state.overlay_stop_dep_indptr,
        state.overlay_stop_dep_time,
        state.overlay_stop_dep_node,
        dist, target_extra, dist_seen, target_seen, np.int32(stamp),
        heap_key, heap_dist, heap_node, np.int32(V),
    )
    if int(res) == -2:
        raise MemoryError("TEEG-Dij-INCR heap exhausted")
    if int(res) >= int(INF_I32):
        return None
    return int(res)
