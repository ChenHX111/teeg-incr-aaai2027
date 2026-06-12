"""Numba-JIT bidirectional Dijkstra on the TEEG event graph.

Reference: Route_Planner.teeg.query.teeg_bidirectional (pure Python).

Hot path is `_bidir_core_njit`, which runs forward and backward Dijkstra
searches on parallel manual min-heaps and stops as soon as
``top_f + top_b >= mu``, the standard bidirectional Dijkstra termination
condition (Goldberg & Harrelson, 2005).

CSR layout: the TEEG already stores both forward (``indptr``/``indices``/
``weights``) and reverse (``rev_indptr``/``rev_indices``/``rev_weights``)
adjacencies, so no reverse-CSR build is needed at query time.

Per-query attachments (source D-events / target A-events / pure-walk
fall-back) are computed in numpy in the Python wrapper — they are tiny
relative to the inner Dijkstra and dominated by ``np.searchsorted``.

CORRECTNESS NOTES (relative to pure-Python teeg_bidirectional)
--------------------------------------------------------------
* Source-attachment dedup: pure-Python uses ``dict[node] = min_w``; we
  use a dense int32 array sized n_nodes initialised to ``INF_I32`` and
  push only when ``w < dist_f[node]`` — equivalent.
* Same for target attachments / backward init.
* Termination uses ``>=`` matching pure-Python (``top_f + top_b >= mu``).
* Alternation: pop from the heap with the smaller (or equal) top first,
  matching pure-Python (``if top_f <= top_b``).
* Lazy deletion (``d > dist[u]`` skip) is preserved.

INTEGER OVERFLOW
----------------
All weights are minute counts (typically <= a few hours). ``mu`` and
all distances stay well below ``2^31``; adding one edge weight to an
in-range distance cannot overflow int32. We still skip the relaxation
if ``dist_u >= INF_I32`` defensively.
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF_I32 = np.int32(2_147_483_647)
# Matches pure-Python Route_Planner.teeg.query.INF_TIME.  Used as the mu
# sentinel so bidirectional termination (top_f + top_b >= mu) behaves
# identically to the reference; dist_f / dist_b still use INF_I32.
INF_TIME = np.int32(32_000)

# ---------------------------------------------------------------------------
# Manual int32 binary min-heap on parallel arrays (key, val).
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _heap_push(hk, hv, n, key, val):
    hk[n] = key
    hv[n] = val
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        tk = hk[p]; hk[p] = hk[i]; hk[i] = tk
        tv = hv[p]; hv[p] = hv[i]; hv[i] = tv
        i = p
    return n + 1


@njit(cache=True, inline="always")
def _heap_pop(hk, hv, n):
    rk = hk[0]
    rv = hv[0]
    n -= 1
    if n > 0:
        hk[0] = hk[n]
        hv[0] = hv[n]
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            sm = i
            if l < n and hk[l] < hk[sm]:
                sm = l
            if r < n and hk[r] < hk[sm]:
                sm = r
            if sm == i:
                break
            tk = hk[sm]; hk[sm] = hk[i]; hk[i] = tk
            tv = hv[sm]; hv[sm] = hv[i]; hv[i] = tv
            i = sm
    return rk, rv, n


# ---------------------------------------------------------------------------
# Core bidirectional Dijkstra (single query).
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bidir_core_njit(
    n_nodes,
    f_indptr, f_indices, f_weights,
    b_indptr, b_indices, b_weights,
    src_nodes, src_w,
    tgt_nodes, tgt_w,
    mu_init,
):
    """Run bidirectional Dijkstra. Returns mu (int32 elapsed minutes)."""
    dist_f = np.full(n_nodes, INF_I32, dtype=np.int32)
    dist_b = np.full(n_nodes, INF_I32, dtype=np.int32)

    # Heap capacity bound: every edge relaxation can push one entry, plus
    # the initial pushes. Bidirectional searches don't share heaps so each
    # has its own bound.
    cap_f = f_indices.shape[0] + src_nodes.shape[0] + 16
    cap_b = b_indices.shape[0] + tgt_nodes.shape[0] + 16
    hk_f = np.empty(cap_f, dtype=np.int32)
    hv_f = np.empty(cap_f, dtype=np.int32)
    hk_b = np.empty(cap_b, dtype=np.int32)
    hv_b = np.empty(cap_b, dtype=np.int32)
    n_f = np.int32(0)
    n_b = np.int32(0)

    # Init forward
    for i in range(src_nodes.shape[0]):
        u = src_nodes[i]
        w = src_w[i]
        if w < dist_f[u]:
            dist_f[u] = w
            n_f = _heap_push(hk_f, hv_f, n_f, w, u)
    # Init backward
    for i in range(tgt_nodes.shape[0]):
        u = tgt_nodes[i]
        w = tgt_w[i]
        if w < dist_b[u]:
            dist_b[u] = w
            n_b = _heap_push(hk_b, hv_b, n_b, w, u)

    mu = mu_init

    while n_f > 0 and n_b > 0:
        top_f = hk_f[0]
        top_b = hk_b[0]
        if top_f + top_b >= mu:
            break

        if top_f <= top_b:
            d, u, n_f = _heap_pop(hk_f, hv_f, n_f)
            if d > dist_f[u]:
                continue
            db = dist_b[u]
            if db < INF_I32:
                tot = d + db
                if tot < mu:
                    mu = tot
            lo = f_indptr[u]
            hi = f_indptr[u + 1]
            for k in range(lo, hi):
                v = f_indices[k]
                w = f_weights[k]
                nd = d + w
                if nd < dist_f[v]:
                    dist_f[v] = nd
                    n_f = _heap_push(hk_f, hv_f, n_f, nd, v)
        else:
            d, u, n_b = _heap_pop(hk_b, hv_b, n_b)
            if d > dist_b[u]:
                continue
            df = dist_f[u]
            if df < INF_I32:
                tot = d + df
                if tot < mu:
                    mu = tot
            lo = b_indptr[u]
            hi = b_indptr[u + 1]
            for k in range(lo, hi):
                v = b_indices[k]
                w = b_weights[k]
                nd = d + w
                if nd < dist_b[v]:
                    dist_b[v] = nd
                    n_b = _heap_push(hk_b, hv_b, n_b, nd, v)

    return mu


# ---------------------------------------------------------------------------
# Per-query attachment builders (numpy; tiny compared to core).
# ---------------------------------------------------------------------------

def _source_attachments_np(g, source, t_dep):
    """Return (nodes int32[K], elapsed int32[K]) for boarding events.

    Matches Route_Planner.teeg.query._source_attachments semantics exactly.
    """
    nodes_out = []
    w_out = []

    # Source stop itself: all D-events at dep >= t_dep
    lo = int(g.stop_dep_indptr[source]); hi = int(g.stop_dep_indptr[source + 1])
    if hi > lo:
        times = g.stop_dep_time[lo:hi]
        k = int(np.searchsorted(times, t_dep, side="left"))
        for j in range(lo + k, hi):
            nodes_out.append(int(g.stop_dep_node[j]))
            w_out.append(int(g.stop_dep_time[j]) - t_dep)

    # Walking neighbours: only FIRST feasible D at each neighbour
    wlo = int(g.walk_indptr[source]); whi = int(g.walk_indptr[source + 1])
    for wk in range(wlo, whi):
        sp = int(g.walk_indices[wk])
        wmin = int(g.walk_weights[wk])
        if sp == source:
            continue
        lo = int(g.stop_dep_indptr[sp]); hi = int(g.stop_dep_indptr[sp + 1])
        if hi <= lo:
            continue
        t_thr = t_dep + wmin
        times = g.stop_dep_time[lo:hi]
        k = int(np.searchsorted(times, t_thr, side="left"))
        if lo + k < hi:
            nodes_out.append(int(g.stop_dep_node[lo + k]))
            w_out.append(int(g.stop_dep_time[lo + k]) - t_dep)

    return (np.asarray(nodes_out, dtype=np.int32),
            np.asarray(w_out, dtype=np.int32))


def _target_attachments_np(g, dest):
    """Return (nodes int32[K], extra int32[K]) for sink A-events.

    Mirrors Route_Planner.teeg.query._target_attachments (dict -> two arrays,
    keeping the SMALLER walk-tail when the same A-node would appear twice).
    """
    out = {}
    lo = int(g.stop_arr_indptr[dest]); hi = int(g.stop_arr_indptr[dest + 1])
    for j in range(lo, hi):
        out[int(g.stop_arr_node[j])] = 0

    rwlo = int(g.rev_walk_indptr[dest]); rwhi = int(g.rev_walk_indptr[dest + 1])
    for wk in range(rwlo, rwhi):
        sp = int(g.rev_walk_indices[wk])
        wmin = int(g.rev_walk_weights[wk])
        if sp == dest:
            continue
        lo = int(g.stop_arr_indptr[sp]); hi = int(g.stop_arr_indptr[sp + 1])
        for j in range(lo, hi):
            n = int(g.stop_arr_node[j])
            if (n not in out) or (wmin < out[n]):
                out[n] = wmin

    if not out:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
    nodes = np.fromiter(out.keys(), dtype=np.int32, count=len(out))
    weights = np.fromiter(out.values(), dtype=np.int32, count=len(out))
    return nodes, weights


def _pure_walk_arrival(g, source, dest, t_dep):
    """Earliest arrival via pure walking (no transit). None if not reachable."""
    if source == dest:
        return t_dep
    wlo = int(g.walk_indptr[source]); whi = int(g.walk_indptr[source + 1])
    if whi <= wlo:
        return None
    targets = g.walk_indices[wlo:whi]
    weights = g.walk_weights[wlo:whi]
    mask = targets == dest
    if not mask.any():
        return None
    return t_dep + int(weights[mask].min())


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def teeg_bidirectional_compiled(g, source: int, dest: int, t_dep: int):
    """Compiled bidirectional Dijkstra on TEEG. Earliest-arrival minute or None.

    Parameters
    ----------
    g : Route_Planner.teeg.TEEG
        Built TEEG with forward + reverse CSR adjacencies.
    source, dest : int
        Stop ids (must satisfy ``0 <= s < g.n_stops``).
    t_dep : int
        Departure minute from ``source`` (>= g.t_lo).

    Returns
    -------
    int earliest-arrival minute, or None if unreachable within the TEEG.
    """
    src_nodes, src_w = _source_attachments_np(g, source, t_dep)
    tgt_nodes, tgt_w = _target_attachments_np(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)

    if (src_nodes.size == 0 or tgt_nodes.size == 0) and pure_walk is None:
        return None
    if src_nodes.size == 0 or tgt_nodes.size == 0:
        return int(pure_walk)

    if pure_walk is None:
        mu_init = INF_TIME
    else:
        mu_init = np.int32(pure_walk - t_dep)

    # Dtype guards (audit-mandated, matches raptor_compiled pattern)
    assert g.indptr.dtype == np.int32
    assert g.indices.dtype == np.int32
    assert g.weights.dtype == np.int32
    assert g.rev_indptr.dtype == np.int32
    assert g.rev_indices.dtype == np.int32
    assert g.rev_weights.dtype == np.int32

    mu = _bidir_core_njit(
        np.int32(g.n_nodes),
        g.indptr, g.indices, g.weights,
        g.rev_indptr, g.rev_indices, g.rev_weights,
        src_nodes, src_w,
        tgt_nodes, tgt_w,
        mu_init,
    )
    if int(mu) >= int(INF_TIME):
        return None
    return int(t_dep) + int(mu)
