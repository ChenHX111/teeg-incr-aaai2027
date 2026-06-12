"""Lane-aware Numba-JIT Dial kernel for MG-Dial v2.

Two reset strategies are provided so the Day-1 perf gate (F11) can pick
the better one without rebuilding the package:

* ``_dial_K_core_genA`` — Plan A: per-(lane, node) generation counter.
  Memory: ``dist[K+1, n_nodes]`` int16 + ``gen[K+1, n_nodes]`` uint32.
  Reset cost per query: O(1) ``current_gen += 1``.

* ``_dial_K_core_touchedC`` — Plan C: per-query touched-list reset.
  Memory: ``dist[K+1, n_nodes]`` int16 + ``touched_node`` + ``touched_lane``
  (sized to pool capacity).  Reset cost per query: O(#unique writes).

Both kernels share the same lane-aware relaxation rule (RIDE bumps the
lane index by one, all other edge types preserve it) and the same
tombstone + overlay support.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit


# Edge-type codes (must match build.py)
ET_RIDE = 2

# Sentinels
INF16 = np.int16(32_000)
POOL_EXHAUSTED = np.int32(-2)
DIST_OVERFLOW = np.int32(-3)


# -------------------------------------------------------------------------
# Plan A: generation-counter reset
# -------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, fastmath=False)
def _dial_K_core_genA(
    indptr,        # int64[n_nodes+1]
    indices,       # int32[n_edges]
    weights,       # int16[n_edges]
    etype,         # int8[n_edges]
    ov_indptr,     # int64[n_nodes+1] or shape (0,) sentinel
    ov_indices,    # int32[ov_n_edges]
    ov_weights,    # int16[ov_n_edges]
    ov_etype,      # int8[ov_n_edges]
    tomb,          # uint8[ceil(n_edges/8)]
    have_tomb,     # int32 (0/1)
    dist,          # int16[K+1, n_nodes]
    gen,           # uint32[K+1, n_nodes]
    current_gen,   # uint32
    pool_node,     # int32[pool_cap]
    pool_lane,     # int8[pool_cap]
    pool_next,     # int32[pool_cap]
    bucket_head,   # int32[max_buckets]
    pool_cap,      # int64
    n_nodes,       # int64
    max_buckets,   # int32
    K,             # int32
    seed_nodes,    # int32[n_seeds]
    seed_lanes,    # int8[n_seeds]
    seed_dists,    # int16[n_seeds]
    dst_R_lo,
    dst_R_hi,
    dst_W_lo,
    dst_W_hi,
):
    INF16_loc = np.int16(32_000)
    # init bucket heads
    for b in range(max_buckets):
        bucket_head[b] = -1

    pool_size = np.int64(0)

    # seeds
    for i in range(seed_nodes.shape[0]):
        node = seed_nodes[i]
        lane = seed_lanes[i]
        d = seed_dists[i]
        if d < max_buckets and lane <= K:
            cur_d = INF16_loc
            if gen[lane, node] == current_gen:
                cur_d = dist[lane, node]
            if d < cur_d:
                dist[lane, node] = d
                gen[lane, node] = current_gen
                if pool_size >= pool_cap:
                    return POOL_EXHAUSTED
                eidx = pool_size
                pool_size += 1
                pool_node[eidx] = node
                pool_lane[eidx] = lane
                pool_next[eidx] = bucket_head[d]
                bucket_head[d] = np.int32(eidx)

    have_overlay = ov_indptr.shape[0] > 0

    for cur in range(max_buckets):
        while bucket_head[cur] >= 0:
            eidx = bucket_head[cur]
            bucket_head[cur] = pool_next[eidx]
            u = pool_node[eidx]
            k_u = pool_lane[eidx]
            # stale check
            if gen[k_u, u] != current_gen or dist[k_u, u] != cur:
                continue
            # destination check
            if (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi):
                return np.int32(cur)
            # base CSR
            lo = indptr[u]
            hi = indptr[u + 1]
            for e in range(lo, hi):
                if have_tomb == 1:
                    if (tomb[e >> 3] & (np.uint8(1) << np.uint8(e & 7))) != 0:
                        continue
                v = indices[e]
                w = weights[e]
                et = etype[e]
                dk = 1 if et == ET_RIDE else 0
                k_v = k_u + dk
                if k_v > K:
                    continue
                d_v = cur + w
                if d_v >= max_buckets:
                    continue
                cur_d = INF16_loc
                if gen[k_v, v] == current_gen:
                    cur_d = dist[k_v, v]
                if d_v < cur_d:
                    if d_v >= INF16_loc:
                        return DIST_OVERFLOW
                    dist[k_v, v] = np.int16(d_v)
                    gen[k_v, v] = current_gen
                    if pool_size >= pool_cap:
                        return POOL_EXHAUSTED
                    new_eidx = pool_size
                    pool_size += 1
                    pool_node[new_eidx] = v
                    pool_lane[new_eidx] = np.int8(k_v)
                    pool_next[new_eidx] = bucket_head[d_v]
                    bucket_head[d_v] = np.int32(new_eidx)
            # overlay CSR (no tombstones; freshly added)
            if have_overlay:
                ov_lo = ov_indptr[u]
                ov_hi = ov_indptr[u + 1]
                for e in range(ov_lo, ov_hi):
                    v = ov_indices[e]
                    w = ov_weights[e]
                    et = ov_etype[e]
                    dk = 1 if et == ET_RIDE else 0
                    k_v = k_u + dk
                    if k_v > K:
                        continue
                    d_v = cur + w
                    if d_v >= max_buckets:
                        continue
                    cur_d = INF16_loc
                    if gen[k_v, v] == current_gen:
                        cur_d = dist[k_v, v]
                    if d_v < cur_d:
                        if d_v >= INF16_loc:
                            return DIST_OVERFLOW
                        dist[k_v, v] = np.int16(d_v)
                        gen[k_v, v] = current_gen
                        if pool_size >= pool_cap:
                            return POOL_EXHAUSTED
                        new_eidx = pool_size
                        pool_size += 1
                        pool_node[new_eidx] = v
                        pool_lane[new_eidx] = np.int8(k_v)
                        pool_next[new_eidx] = bucket_head[d_v]
                        bucket_head[d_v] = np.int32(new_eidx)

    return np.int32(-1)


# -------------------------------------------------------------------------
# Plan C: touched-list reset
# -------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, fastmath=False)
def _dial_K_core_touchedC(
    indptr,
    indices,
    weights,
    etype,
    ov_indptr,
    ov_indices,
    ov_weights,
    ov_etype,
    tomb,
    have_tomb,
    dist,            # int16[K+1, n_nodes]   pre-cleared to INF16
    touched_node,    # int32[max_touched]
    touched_lane,    # int8[max_touched]
    pool_node,
    pool_lane,
    pool_next,
    bucket_head,
    pool_cap,
    n_nodes,
    max_buckets,
    K,
    seed_nodes,
    seed_lanes,
    seed_dists,
    dst_R_lo,
    dst_R_hi,
    dst_W_lo,
    dst_W_hi,
):
    INF16_loc = np.int16(32_000)
    for b in range(max_buckets):
        bucket_head[b] = -1
    pool_size = np.int64(0)
    n_touched = np.int64(0)

    # seeds
    for i in range(seed_nodes.shape[0]):
        node = seed_nodes[i]
        lane = seed_lanes[i]
        d = seed_dists[i]
        if d < max_buckets and lane <= K:
            if d < dist[lane, node]:
                if dist[lane, node] >= INF16_loc:
                    # first touch; record
                    touched_node[n_touched] = node
                    touched_lane[n_touched] = lane
                    n_touched += 1
                dist[lane, node] = d
                if pool_size >= pool_cap:
                    return POOL_EXHAUSTED, n_touched
                eidx = pool_size
                pool_size += 1
                pool_node[eidx] = node
                pool_lane[eidx] = lane
                pool_next[eidx] = bucket_head[d]
                bucket_head[d] = np.int32(eidx)

    have_overlay = ov_indptr.shape[0] > 0

    for cur in range(max_buckets):
        while bucket_head[cur] >= 0:
            eidx = bucket_head[cur]
            bucket_head[cur] = pool_next[eidx]
            u = pool_node[eidx]
            k_u = pool_lane[eidx]
            if dist[k_u, u] != cur:
                continue
            if (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi):
                return np.int32(cur), n_touched
            lo = indptr[u]
            hi = indptr[u + 1]
            for e in range(lo, hi):
                if have_tomb == 1:
                    if (tomb[e >> 3] & (np.uint8(1) << np.uint8(e & 7))) != 0:
                        continue
                v = indices[e]
                w = weights[e]
                et = etype[e]
                dk = 1 if et == ET_RIDE else 0
                k_v = k_u + dk
                if k_v > K:
                    continue
                d_v = cur + w
                if d_v >= max_buckets:
                    continue
                if d_v < dist[k_v, v]:
                    if d_v >= INF16_loc:
                        return DIST_OVERFLOW, n_touched
                    if dist[k_v, v] >= INF16_loc:
                        if n_touched >= touched_node.shape[0]:
                            return POOL_EXHAUSTED, n_touched
                        touched_node[n_touched] = v
                        touched_lane[n_touched] = np.int8(k_v)
                        n_touched += 1
                    dist[k_v, v] = np.int16(d_v)
                    if pool_size >= pool_cap:
                        return POOL_EXHAUSTED, n_touched
                    new_eidx = pool_size
                    pool_size += 1
                    pool_node[new_eidx] = v
                    pool_lane[new_eidx] = np.int8(k_v)
                    pool_next[new_eidx] = bucket_head[d_v]
                    bucket_head[d_v] = np.int32(new_eidx)
            if have_overlay:
                ov_lo = ov_indptr[u]
                ov_hi = ov_indptr[u + 1]
                for e in range(ov_lo, ov_hi):
                    v = ov_indices[e]
                    w = ov_weights[e]
                    et = ov_etype[e]
                    dk = 1 if et == ET_RIDE else 0
                    k_v = k_u + dk
                    if k_v > K:
                        continue
                    d_v = cur + w
                    if d_v >= max_buckets:
                        continue
                    if d_v < dist[k_v, v]:
                        if d_v >= INF16_loc:
                            return DIST_OVERFLOW, n_touched
                        if dist[k_v, v] >= INF16_loc:
                            if n_touched >= touched_node.shape[0]:
                                return POOL_EXHAUSTED, n_touched
                            touched_node[n_touched] = v
                            touched_lane[n_touched] = np.int8(k_v)
                            n_touched += 1
                        dist[k_v, v] = np.int16(d_v)
                        if pool_size >= pool_cap:
                            return POOL_EXHAUSTED, n_touched
                        new_eidx = pool_size
                        pool_size += 1
                        pool_node[new_eidx] = v
                        pool_lane[new_eidx] = np.int8(k_v)
                        pool_next[new_eidx] = bucket_head[d_v]
                        bucket_head[d_v] = np.int32(new_eidx)

    return np.int32(-1), n_touched


@njit(cache=True, boundscheck=False)
def _reset_touched(dist, touched_node, touched_lane, n_touched):
    INF16_loc = np.int16(32_000)
    for i in range(n_touched):
        dist[touched_lane[i], touched_node[i]] = INF16_loc


# -------------------------------------------------------------------------
# Fast path: single-lane Dial for K=None (unbounded transfers)
# Bypasses all lane logic — equivalent to v1 _dial_core_njit but supports
# tombstones and overlay CSR.
# -------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, fastmath=False)
def _dial_single_lane(
    indptr,        # int64[n_nodes+1]
    indices,       # int32[n_edges]
    weights,       # int16[n_edges]
    ov_indptr,     # int64[n_nodes+1] or shape (0,) sentinel
    ov_indices,    # int32[ov_n_edges]
    ov_weights,    # int16[ov_n_edges]
    tomb,          # uint8[ceil(n_edges/8)]
    have_tomb,     # int32 (0/1)
    dist_1d,       # int32[n_nodes]  (flat, no lanes)
    pool_node,     # int32[pool_cap]
    pool_next,     # int32[pool_cap]
    bucket_head,   # int32[max_buckets]
    pool_cap,      # int64
    n_nodes,       # int64
    max_buckets,   # int32
    seed_nodes,    # int32[n_seeds]
    seed_dists,    # int32[n_seeds]
    dst_R_lo,
    dst_R_hi,
    dst_W_lo,
    dst_W_hi,
):
    """Single-lane Dial's bucket-Dijkstra — no K-lane overhead.

    Used when K=None (unbounded transfers). Matches v1 performance while
    supporting tombstones and overlay CSR for dynamic updates.
    """
    INF_I32 = np.int32(2_147_483_647)

    # init
    for i in range(n_nodes):
        dist_1d[i] = INF_I32
    for b in range(max_buckets):
        bucket_head[b] = -1

    pool_size = np.int64(0)

    # seeds
    for i in range(seed_nodes.shape[0]):
        node = seed_nodes[i]
        d = seed_dists[i]
        if d < max_buckets and d < dist_1d[node]:
            dist_1d[node] = d
            if pool_size >= pool_cap:
                return POOL_EXHAUSTED
            eidx = pool_size
            pool_size += 1
            pool_node[eidx] = node
            pool_next[eidx] = bucket_head[d]
            bucket_head[d] = np.int32(eidx)

    have_overlay = ov_indptr.shape[0] > 0

    for cur in range(max_buckets):
        while bucket_head[cur] >= 0:
            eidx = bucket_head[cur]
            bucket_head[cur] = pool_next[eidx]
            u = pool_node[eidx]
            if dist_1d[u] != cur:
                continue
            # destination check
            if (dst_R_lo <= u < dst_R_hi) or (dst_W_lo <= u < dst_W_hi):
                return np.int32(cur)
            # base CSR
            lo = indptr[u]
            hi = indptr[u + 1]
            for e in range(lo, hi):
                if have_tomb == 1:
                    if (tomb[e >> 3] & (np.uint8(1) << np.uint8(e & 7))) != 0:
                        continue
                v = indices[e]
                w = weights[e]
                nd = np.int32(cur + w)
                if nd < dist_1d[v] and nd < max_buckets:
                    dist_1d[v] = nd
                    if pool_size >= pool_cap:
                        return POOL_EXHAUSTED
                    new_eidx = pool_size
                    pool_size += 1
                    pool_node[new_eidx] = v
                    pool_next[new_eidx] = bucket_head[nd]
                    bucket_head[nd] = np.int32(new_eidx)
            # overlay CSR
            if have_overlay:
                ov_lo = ov_indptr[u]
                ov_hi = ov_indptr[u + 1]
                for e in range(ov_lo, ov_hi):
                    v = ov_indices[e]
                    w = ov_weights[e]
                    nd = np.int32(cur + w)
                    if nd < dist_1d[v] and nd < max_buckets:
                        dist_1d[v] = nd
                        if pool_size >= pool_cap:
                            return POOL_EXHAUSTED
                        new_eidx = pool_size
                        pool_size += 1
                        pool_node[new_eidx] = v
                        pool_next[new_eidx] = bucket_head[nd]
                        bucket_head[nd] = np.int32(new_eidx)

    return np.int32(-1)


# -------------------------------------------------------------------------
# Python-side wrappers
# -------------------------------------------------------------------------
def _prepare_seeds(g, source: int, dest: int, t_dep: int, closed_walks_csr):
    if t_dep < g.t_lo or t_dep > g.t_hi:
        return None
    T = g.T
    nodes = [source * T + (t_dep - g.t_lo)]
    lanes = [0]
    dists = [0]
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
            nodes.append(v)
            lanes.append(0)
            dists.append(wmin)
    seed_nodes = np.array(nodes, dtype=np.int32)
    seed_lanes = np.array(lanes, dtype=np.int8)
    seed_dists = np.array(dists, dtype=np.int16)
    dst_R_lo = np.int32(dest * T)
    dst_R_hi = np.int32((dest + 1) * T)
    dst_W_lo = np.int32(g.RW_OFFSET + dest * T)
    dst_W_hi = np.int32(g.RW_OFFSET + (dest + 1) * T)
    return seed_nodes, seed_lanes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi


class MGDialV2:
    """Stateful runner for MG-Dial V2 — owns the dist / pool / overlay buffers.

    Parameters
    ----------
    g : MinuteGridV2
    K : int (default 15)
    plan : {'A', 'C'}  reset strategy
    pool_cap : int     initial pool size (auto-grown to 2x then 4x on overflow)
    """

    def __init__(self, g, K: int = 15, plan: str = "A",
                 pool_cap: int = 32_000_000):
        if plan not in ("A", "C"):
            raise ValueError(f"plan must be 'A' or 'C', got {plan}")
        self.g = g
        self.K = int(K)
        self.plan = plan
        self.pool_cap = int(pool_cap)
        n = g.n_nodes
        # base CSR (already correct dtype but ensure contiguity)
        self.indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(g.indices, dtype=np.int32)
        self.weights = np.ascontiguousarray(g.weights, dtype=np.int16)
        self.etype = np.ascontiguousarray(g.etype, dtype=np.int8)
        # tombstone bitmap
        self.tomb = np.zeros((g.n_edges + 7) // 8, dtype=np.uint8)
        # overlay CSR (empty by default)
        self._reset_overlay()
        # dist + reset state
        self.dist = np.full((self.K + 1, n), INF16, dtype=np.int16)
        if plan == "A":
            self.gen = np.zeros((self.K + 1, n), dtype=np.uint32)
            self.current_gen = np.uint32(0)
            self.touched_node = np.zeros(0, dtype=np.int32)
            self.touched_lane = np.zeros(0, dtype=np.int8)
        else:
            self.gen = np.zeros((0, 0), dtype=np.uint32)
            self.current_gen = np.uint32(0)
            self.touched_node = np.zeros(self.pool_cap, dtype=np.int32)
            self.touched_lane = np.zeros(self.pool_cap, dtype=np.int8)
        # pool
        self.pool_node = np.zeros(self.pool_cap, dtype=np.int32)
        self.pool_lane = np.zeros(self.pool_cap, dtype=np.int8)
        self.pool_next = np.zeros(self.pool_cap, dtype=np.int32)
        self.bucket_head = np.zeros(g.T + 1, dtype=np.int32)
        # Single-lane buffer for K=None fast path
        self.dist_1d = np.zeros(n, dtype=np.int32)

    # ------------------------------------------------------------------
    def _reset_overlay(self):
        self.ov_indptr = np.zeros(0, dtype=np.int64)
        self.ov_indices = np.zeros(0, dtype=np.int32)
        self.ov_weights = np.zeros(0, dtype=np.int16)
        self.ov_etype = np.zeros(0, dtype=np.int8)
        self._ov_buf_src = []
        self._ov_buf_dst = []
        self._ov_buf_w = []
        self._ov_buf_et = []
        self._ov_dirty = False
        self._ov_added_trips: list = []  # list of dicts (so restore can find them)

    def _grow_pool(self, new_cap: int):
        self.pool_cap = int(new_cap)
        self.pool_node = np.zeros(self.pool_cap, dtype=np.int32)
        self.pool_lane = np.zeros(self.pool_cap, dtype=np.int8)
        self.pool_next = np.zeros(self.pool_cap, dtype=np.int32)
        if self.plan == "C":
            self.touched_node = np.zeros(self.pool_cap, dtype=np.int32)
            self.touched_lane = np.zeros(self.pool_cap, dtype=np.int8)

    def _compile_overlay(self):
        if not self._ov_dirty:
            return
        n_nodes = self.g.n_nodes
        if not self._ov_buf_src:
            self.ov_indptr = np.zeros(0, dtype=np.int64)
            self.ov_indices = np.zeros(0, dtype=np.int32)
            self.ov_weights = np.zeros(0, dtype=np.int16)
            self.ov_etype = np.zeros(0, dtype=np.int8)
            self._ov_dirty = False
            return
        src = np.concatenate(self._ov_buf_src)
        dst = np.concatenate(self._ov_buf_dst)
        w = np.concatenate(self._ov_buf_w)
        et = np.concatenate(self._ov_buf_et)
        order = np.argsort(src, kind="stable")
        self.ov_indices = dst[order].astype(np.int32)
        self.ov_weights = w[order].astype(np.int16)
        self.ov_etype = et[order].astype(np.int8)
        ov_indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(ov_indptr, src[order] + 1, 1)
        np.cumsum(ov_indptr, out=ov_indptr)
        self.ov_indptr = ov_indptr
        self._ov_dirty = False

    # ------------------------------------------------------------------
    def query(self, source: int, dest: int, t_dep: int, *,
              closed_walks_csr=None, K: Optional[int] = None) -> Optional[int]:
        """Earliest-arrival minute (or None) for (source -> dest) at t_dep,
        capped at K transfers (RIDE edges).

        When K is None, uses single-lane fast path (no lane overhead) for
        performance parity with MG-Dial v1.
        """
        unbounded = K is None
        if unbounded:
            # Fast path: single-lane Dial, no K tracking
            prep = _prepare_seeds(self.g, source, dest, t_dep, closed_walks_csr)
            if prep is None:
                return None
            seed_nodes, seed_lanes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = prep

            self._compile_overlay()
            max_buckets = np.int32(self.g.T + 1)
            n_nodes = np.int64(self.g.n_nodes)
            # Convert seed_dists to int32 for single-lane kernel
            seed_dists_i32 = seed_dists.astype(np.int32)

            for attempt in range(5):
                result = _dial_single_lane(
                    self.indptr, self.indices, self.weights,
                    self.ov_indptr, self.ov_indices, self.ov_weights,
                    self.tomb, np.int32(1),
                    self.dist_1d,
                    self.pool_node, self.pool_next,
                    self.bucket_head,
                    np.int64(self.pool_cap), n_nodes, max_buckets,
                    seed_nodes, seed_dists_i32,
                    dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi,
                )
                if result == int(POOL_EXHAUSTED):
                    self._grow_pool(self.pool_cap * 2)
                    continue
                if int(result) < 0:
                    return None
                return int(t_dep) + int(result)
            raise RuntimeError(
                f"pool exhausted at cap={self.pool_cap}; investigate")

        # Bounded K path: use lane-aware kernel
        if K > self.K:
            raise ValueError(f"K={K} exceeds dist allocation K={self.K}")
        prep = _prepare_seeds(self.g, source, dest, t_dep, closed_walks_csr)
        if prep is None:
            return None
        seed_nodes, seed_lanes, seed_dists, dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi = prep

        self._compile_overlay()
        max_buckets = np.int32(self.g.T + 1)
        n_nodes = np.int64(self.g.n_nodes)

        for attempt in range(5):
            if self.plan == "A":
                self.current_gen = np.uint32(int(self.current_gen) + 1)
                if int(self.current_gen) == 0:
                    self.gen.fill(0)
                    self.current_gen = np.uint32(1)
                result = _dial_K_core_genA(
                    self.indptr, self.indices, self.weights, self.etype,
                    self.ov_indptr, self.ov_indices, self.ov_weights, self.ov_etype,
                    self.tomb, np.int32(1),
                    self.dist, self.gen, self.current_gen,
                    self.pool_node, self.pool_lane, self.pool_next,
                    self.bucket_head,
                    np.int64(self.pool_cap), n_nodes, max_buckets,
                    np.int32(K),
                    seed_nodes, seed_lanes, seed_dists,
                    dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi,
                )
            else:
                result, n_touched = _dial_K_core_touchedC(
                    self.indptr, self.indices, self.weights, self.etype,
                    self.ov_indptr, self.ov_indices, self.ov_weights, self.ov_etype,
                    self.tomb, np.int32(1),
                    self.dist, self.touched_node, self.touched_lane,
                    self.pool_node, self.pool_lane, self.pool_next,
                    self.bucket_head,
                    np.int64(self.pool_cap), n_nodes, max_buckets,
                    np.int32(K),
                    seed_nodes, seed_lanes, seed_dists,
                    dst_R_lo, dst_R_hi, dst_W_lo, dst_W_hi,
                )
                _reset_touched(self.dist, self.touched_node, self.touched_lane,
                               np.int64(n_touched))

            if result == int(POOL_EXHAUSTED):
                self._grow_pool(self.pool_cap * 2)
                continue
            if result == int(DIST_OVERFLOW):
                raise RuntimeError(
                    "dist overflow — distance exceeded INF16 sentinel")
            if int(result) < 0:
                return None
            return int(t_dep) + int(result)
        raise RuntimeError(
            f"pool exhausted twice at cap={self.pool_cap}; investigate")
