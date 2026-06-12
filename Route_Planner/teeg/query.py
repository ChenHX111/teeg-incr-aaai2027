"""TEEG query algorithms.

Three flavours, all returning earliest-arrival minute (or None):
  - teeg_dijkstra            : forward Dijkstra (scipy C-speed)
  - teeg_dijkstra_pure       : pure-Python Dijkstra (used by ALT)
  - teeg_bidirectional       : bidirectional Dijkstra meeting in the middle
  - teeg_alt                 : A* with landmark heuristic (precomputed)

All use the same TEEG produced by builder.build_teeg().
The source / destination attachment is done per-query: we identify the set
of D-events reachable from `source` (at or via walking) at time >= t_dep,
and the set of A-events that can deliver us to `dest` (with optional final walk).
"""
from __future__ import annotations
import numpy as np
import heapq
from .builder import TEEG

INF_TIME = 32_000


def _source_attachments(g: TEEG, source: int, t_dep: int):
    """Yield (D_node, initial_elapsed) for every feasible boarding event.

    Considers both the source stop itself and walking-neighbours.
    Initial elapsed = (dep_time - t_dep) for source-stop events, or
    (dep_time - t_dep) for walk-neighbour events (the walk time is already
    accounted for by requiring dep_time >= t_dep + walk).
    """
    res = []
    # Source stop itself
    lo, hi = int(g.stop_dep_indptr[source]), int(g.stop_dep_indptr[source + 1])
    if hi > lo:
        times = g.stop_dep_time[lo:hi]
        k = int(np.searchsorted(times, t_dep, side="left"))
        for j in range(lo + k, hi):
            res.append((int(g.stop_dep_node[j]), int(g.stop_dep_time[j]) - t_dep))
    # Walking neighbours
    wlo, whi = int(g.walk_indptr[source]), int(g.walk_indptr[source + 1])
    for wk in range(wlo, whi):
        sp = int(g.walk_indices[wk]); wmin = int(g.walk_weights[wk])
        if sp == source:
            continue
        lo, hi = int(g.stop_dep_indptr[sp]), int(g.stop_dep_indptr[sp + 1])
        if hi <= lo:
            continue
        t_threshold = t_dep + wmin
        times = g.stop_dep_time[lo:hi]
        k = int(np.searchsorted(times, t_threshold, side="left"))
        if lo + k < hi:
            # only the FIRST feasible D-event at this walk-neighbour;
            # subsequent ones can be reached via the wait-chain.
            res.append((int(g.stop_dep_node[lo + k]),
                        int(g.stop_dep_time[lo + k]) - t_dep))
    return res


def _pure_walk_arrival(g: TEEG, source: int, dest: int, t_dep: int):
    """Return earliest arrival at dest via pure walking from source (no transit).

    Handles src == dest (arrival = t_dep) and src is a walking neighbour of dest.
    """
    if source == dest:
        return t_dep
    wlo, whi = int(g.walk_indptr[source]), int(g.walk_indptr[source + 1])
    if whi <= wlo:
        return None
    targets = g.walk_indices[wlo:whi]
    weights = g.walk_weights[wlo:whi]
    mask = targets == dest
    if not mask.any():
        return None
    wmin = int(weights[mask].min())
    return t_dep + wmin


def _target_attachments(g: TEEG, dest: int):
    """Return dict: A_node -> extra_minutes (walking-to-dest tail).

    Includes A-events at `dest` itself (extra=0) and at walking-neighbours of dest.
    """
    out = {}
    # A-events at dest
    lo, hi = int(g.stop_arr_indptr[dest]), int(g.stop_arr_indptr[dest + 1])
    for j in range(lo, hi):
        out[int(g.stop_arr_node[j])] = 0
    # A-events at walking-neighbours of dest (reverse walk: s -> dest)
    rwlo, rwhi = int(g.rev_walk_indptr[dest]), int(g.rev_walk_indptr[dest + 1])
    for wk in range(rwlo, rwhi):
        sp = int(g.rev_walk_indices[wk]); wmin = int(g.rev_walk_weights[wk])
        if sp == dest:
            continue
        lo, hi = int(g.stop_arr_indptr[sp]), int(g.stop_arr_indptr[sp + 1])
        for j in range(lo, hi):
            n = int(g.stop_arr_node[j])
            # keep the SMALLER walk tail if same A-node appears
            if (n not in out) or (wmin < out[n]):
                out[n] = wmin
    return out


def teeg_dijkstra(g: TEEG, source: int, dest: int, t_dep: int):
    """Forward Dijkstra on TEEG (pure-Python heapq). Earliest-arrival minute."""
    src_attach = _source_attachments(g, source, t_dep)
    tgt_attach = _target_attachments(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)
    if not src_attach and pure_walk is None:
        return None

    dist = {}
    pq = []
    for (n, w) in src_attach:
        if (n not in dist) or (w < dist[n]):
            dist[n] = w
            heapq.heappush(pq, (w, n))

    indptr = g.indptr; indices = g.indices; weights = g.weights
    best_arrival = INF_TIME if pure_walk is None else pure_walk
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF_TIME):
            continue
        # target-extra check
        if u in tgt_attach:
            cand = t_dep + d + tgt_attach[u]
            if cand < best_arrival:
                best_arrival = cand
        if d >= best_arrival - t_dep:
            continue
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k]); w = int(weights[k])
            nd = d + w
            if nd < dist.get(v, INF_TIME):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return int(best_arrival) if best_arrival < INF_TIME else None


def teeg_dijkstra_scipy(g: TEEG, source: int, dest: int, t_dep: int):
    """Forward Dijkstra using scipy.sparse.csgraph.dijkstra (C-speed).

    We add a virtual super-source node n that points to each feasible boarding
    event. The augmented CSR is assembled DIRECTLY from edge lists (not via
    sparse `+`, which would drop zero-weight edges).
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    src_attach = _source_attachments(g, source, t_dep)
    tgt_attach = _target_attachments(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)
    if not src_attach and pure_walk is None:
        return None
    if not src_attach:
        return pure_walk

    n = g.n_nodes
    super_src = n
    # Original edges (src, dst, w)
    base_src = np.repeat(np.arange(n, dtype=np.int32), np.diff(g.indptr))
    base_dst = g.indices
    base_w   = g.weights
    # Extra edges from super_src
    n_extra = len(src_attach)
    ex_src = np.full(n_extra, super_src, dtype=np.int32)
    ex_dst = np.array([x[0] for x in src_attach], dtype=np.int32)
    ex_w   = np.array([x[1] for x in src_attach], dtype=np.int32)
    # Concatenate and build CSR via lexsort + manual indptr (no sum_duplicates)
    all_src = np.concatenate([base_src, ex_src])
    all_dst = np.concatenate([base_dst, ex_dst])
    all_w   = np.concatenate([base_w,   ex_w])
    order = np.argsort(all_src, kind="stable")
    s_sorted = all_src[order]
    d_sorted = all_dst[order]
    w_sorted = all_w[order]
    aug_indptr = np.zeros(n + 2, dtype=np.int32)
    np.add.at(aug_indptr, s_sorted + 1, 1)
    np.cumsum(aug_indptr, out=aug_indptr)
    aug = sp.csr_matrix((w_sorted, d_sorted, aug_indptr), shape=(n + 1, n + 1))

    dists = dijkstra(aug, directed=True, indices=super_src,
                     return_predecessors=False, limit=np.inf)
    best_arrival = INF_TIME if pure_walk is None else pure_walk
    for (a_node, extra) in tgt_attach.items():
        d = dists[a_node]
        if np.isinf(d):
            continue
        cand = t_dep + int(d) + extra
        if cand < best_arrival:
            best_arrival = cand
    return int(best_arrival) if best_arrival < INF_TIME else None


def teeg_bidirectional(g: TEEG, source: int, dest: int, t_dep: int):
    """Bidirectional Dijkstra on TEEG.

    Forward search from each source-attached D-event.
    Backward search from each target-attached A-event.
    Stop when the sum of minimum heap-tops exceeds the best so far.
    """
    src_attach = _source_attachments(g, source, t_dep)
    tgt_attach = _target_attachments(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)
    if (not src_attach or not tgt_attach) and pure_walk is None:
        return None
    if not src_attach or not tgt_attach:
        return pure_walk

    dist_f = {}
    dist_b = {}
    pq_f = []
    pq_b = []
    for (n, w) in src_attach:
        if (n not in dist_f) or (w < dist_f[n]):
            dist_f[n] = w
            heapq.heappush(pq_f, (w, n))
    for (n, extra) in tgt_attach.items():
        if (n not in dist_b) or (extra < dist_b[n]):
            dist_b[n] = extra
            heapq.heappush(pq_b, (extra, n))

    f_indptr = g.indptr; f_indices = g.indices; f_weights = g.weights
    b_indptr = g.rev_indptr; b_indices = g.rev_indices; b_weights = g.rev_weights

    mu = INF_TIME if pure_walk is None else (pure_walk - t_dep)
    while pq_f and pq_b:
        top_f = pq_f[0][0]
        top_b = pq_b[0][0]
        if top_f + top_b >= mu:
            break
        # alternate, pop from smaller heap-top
        if top_f <= top_b:
            d, u = heapq.heappop(pq_f)
            if d > dist_f.get(u, INF_TIME):
                continue
            # meeting check
            if u in dist_b:
                tot = d + dist_b[u]
                if tot < mu:
                    mu = tot
            for k in range(f_indptr[u], f_indptr[u + 1]):
                v = int(f_indices[k]); w = int(f_weights[k])
                nd = d + w
                if nd < dist_f.get(v, INF_TIME):
                    dist_f[v] = nd
                    heapq.heappush(pq_f, (nd, v))
        else:
            d, u = heapq.heappop(pq_b)
            if d > dist_b.get(u, INF_TIME):
                continue
            if u in dist_f:
                tot = d + dist_f[u]
                if tot < mu:
                    mu = tot
            for k in range(b_indptr[u], b_indptr[u + 1]):
                v = int(b_indices[k]); w = int(b_weights[k])
                nd = d + w
                if nd < dist_b.get(v, INF_TIME):
                    dist_b[v] = nd
                    heapq.heappush(pq_b, (nd, v))
    return int(t_dep + mu) if mu < INF_TIME else None


# ============================================================================
# ALT (A* with Landmark heuristic)
# ============================================================================

def build_lower_bound_graph(g: TEEG):
    """Per-stop minimum travel-time graph (waiting ignored).

    For each transit edge (u,v) appearing in any connection, we set the
    LB weight = min over all connections of (arr - dep).
    Walking edges are added with their walk_min weight.
    Result is a CSR matrix over stops (n_stops, n_stops) with int32 weights.

    This is an admissible lower bound for the elapsed-time-from-now metric:
    even with zero waiting, you cannot beat the per-edge minimum ride time.
    """
    import scipy.sparse as sp
    n = g.n_stops
    # extract ride edges from the TEEG
    n_conn = (g.node_kind == 0).sum()
    # for D-events, the unique outgoing RIDE edge has weight = arr - dep at the
    # A-event with same trip+route. We can recover (u,v,w) from node_stop/time
    # but it's simpler to use the TEEG structure: D-node i (index 2i in original)
    # always has its corresponding A-node at index 2i+1 OR we can index by D-events.
    # However during build we used scipy sum_duplicates which scrambled order.
    # So we walk the forward CSR and pick only edges D->A at the same stop pair.
    mins = np.full((n, n), 32000, dtype=np.int32)
    # Use COO for efficiency
    src_stops = []
    dst_stops = []
    wts       = []
    indptr = g.indptr; indices = g.indices; weights = g.weights
    node_kind = g.node_kind; node_stop = g.node_stop
    for u in range(g.n_nodes):
        if node_kind[u] != 0:   # only D-events
            continue
        s = int(node_stop[u])
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if node_kind[v] == 1:  # ride edge to an A-event
                t = int(node_stop[v])
                w = int(weights[k])
                if t != s:
                    src_stops.append(s)
                    dst_stops.append(t)
                    wts.append(w)
    src_stops = np.asarray(src_stops, dtype=np.int32)
    dst_stops = np.asarray(dst_stops, dtype=np.int32)
    wts       = np.asarray(wts,       dtype=np.int32)
    # min-aggregate per (s,t) pair
    # use sparse, then convert to per-pair min via group-by
    # Simpler: build sparse with sum_duplicates -> sum; not what we want.
    # Use a dict aggregator.
    agg: dict[tuple[int, int], int] = {}
    for s, t, w in zip(src_stops, dst_stops, wts):
        k = (int(s), int(t))
        if k not in agg or w < agg[k]:
            agg[k] = w
    # add walking edges
    for s in range(n):
        wlo, whi = int(g.walk_indptr[s]), int(g.walk_indptr[s + 1])
        for k in range(wlo, whi):
            t = int(g.walk_indices[k]); w = int(g.walk_weights[k])
            kk = (s, t)
            if kk not in agg or w < agg[kk]:
                agg[kk] = w
    if not agg:
        return sp.csr_matrix((n, n), dtype=np.int32)
    rows = np.fromiter((k[0] for k in agg.keys()), dtype=np.int32, count=len(agg))
    cols = np.fromiter((k[1] for k in agg.keys()), dtype=np.int32, count=len(agg))
    vals = np.fromiter(agg.values(), dtype=np.int32, count=len(agg))
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return mat


def select_landmarks(lb_graph, n_landmarks: int = 16, seed: int = 42):
    """Furthest-point sampling on the lower-bound graph.

    Start from a random stop; iteratively add the stop farthest from the
    current landmark set in the LB metric.
    """
    from scipy.sparse.csgraph import dijkstra
    n = lb_graph.shape[0]
    rng = np.random.default_rng(seed)
    # start from a non-isolated stop
    deg = np.asarray((lb_graph != 0).sum(axis=1)).ravel()
    cand = np.where(deg > 0)[0]
    if len(cand) == 0:
        return np.zeros(0, dtype=np.int32)
    first = int(rng.choice(cand))
    landmarks = [first]
    # distance from existing landmarks
    dist = dijkstra(lb_graph, directed=True, indices=first, return_predecessors=False)
    dist[np.isinf(dist)] = -1
    for _ in range(n_landmarks - 1):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break
        landmarks.append(nxt)
        d2 = dijkstra(lb_graph, directed=True, indices=nxt, return_predecessors=False)
        d2[np.isinf(d2)] = -1
        # update by elementwise max only where d2 > 0 -- we keep MIN distance to any landmark
        # for furthest-point: want max-min
        dist = np.minimum(dist, d2)
        dist[dist < 0] = -1
    return np.asarray(landmarks, dtype=np.int32)


class LandmarkLabels:
    """Precomputed lower-bound distances from each landmark to all stops,
    both forward (landmark -> stop) and backward (stop -> landmark)."""
    def __init__(self, lb_graph, landmarks: np.ndarray):
        from scipy.sparse.csgraph import dijkstra
        self.landmarks = landmarks.astype(np.int32)
        n = lb_graph.shape[0]
        L = len(landmarks)
        # d_from[i, s] = LB distance from landmark i to stop s (minutes)
        # d_to[s, i]   = LB distance from stop s to landmark i (minutes)
        self.d_from = np.zeros((L, n), dtype=np.int32)
        self.d_to   = np.zeros((n, L), dtype=np.int32)
        for i, lm in enumerate(landmarks):
            d = dijkstra(lb_graph, directed=True, indices=int(lm), return_predecessors=False)
            d = np.where(np.isinf(d), 32000, d).astype(np.int32)
            self.d_from[i] = d
            d2 = dijkstra(lb_graph, directed=True, indices=int(lm),
                          return_predecessors=False, min_only=False, unweighted=False)
        # also compute d_to: dijkstra on the TRANSPOSE for each landmark
        lb_T = lb_graph.T.tocsr()
        for i, lm in enumerate(landmarks):
            d = dijkstra(lb_T, directed=True, indices=int(lm), return_predecessors=False)
            d = np.where(np.isinf(d), 32000, d).astype(np.int32)
            self.d_to[:, i] = d

    def heuristic(self, stop_from: int, stop_to: int) -> int:
        """Admissible lower bound on minimum elapsed time from stop_from to stop_to."""
        if stop_from == stop_to:
            return 0
        # h(u, v) >= max_i | d(u, lm_i) - d(v, lm_i) |
        a = self.d_to[stop_from]      # u -> lm
        b = self.d_to[stop_to]        # v -> lm
        c = self.d_from[:, stop_from] # lm -> u
        d = self.d_from[:, stop_to]   # lm -> v
        # lower bound = max(d_to[u, lm] - d_to[v, lm], d_from[lm, v] - d_from[lm, u])
        # both expressions are admissible lower bounds.
        lb1 = np.max(np.abs(c - d))   # use abs for robustness; pick a single expression
        return int(max(0, lb1))


def teeg_alt(g: TEEG, source: int, dest: int, t_dep: int,
             landmark_labels: LandmarkLabels):
    """A* on TEEG with landmark-based heuristic (admissible).
    Precomputes h(stop) for all stops once per query, then heuristic is O(1) lookup.
    """
    src_attach = _source_attachments(g, source, t_dep)
    tgt_attach = _target_attachments(g, dest)
    pure_walk = _pure_walk_arrival(g, source, dest, t_dep)
    if (not src_attach or not tgt_attach) and pure_walk is None:
        return None
    if not src_attach or not tgt_attach:
        return pure_walk

    node_stop = g.node_stop
    indptr = g.indptr; indices = g.indices; weights = g.weights

    # Precompute h(stop) = max over landmarks of max(d_to[u,lm] - d_to[dest,lm],
    #                                                d_from[lm,dest] - d_from[lm,u], 0)
    # Both terms are admissible LBs on min-travel-time(u -> dest) by triangle inequality
    # on a DIRECTED graph. The ABSOLUTE value would be inadmissible.
    d_to = landmark_labels.d_to            # (n_stops, L)
    d_from = landmark_labels.d_from        # (L, n_stops)
    # term1[u, lm] = d_to[u, lm] - d_to[dest, lm]
    term1 = d_to - d_to[dest]              # (n_stops, L)
    # term2[u, lm] = d_from[lm, dest] - d_from[lm, u]  (broadcast)
    term2 = (d_from[:, dest][:, None] - d_from).T   # (n_stops, L)
    both = np.maximum(term1, term2)
    h_stop = np.maximum(0, np.max(both, axis=1)).astype(np.int32)
    h_stop[dest] = 0
    # for each TEEG node, its heuristic is h_stop[node_stop[node]]
    h_node = h_stop[node_stop]             # (n_nodes,) int32

    dist = {}
    pq = []
    for (nd_id, w) in src_attach:
        if (nd_id not in dist) or (w < dist[nd_id]):
            dist[nd_id] = w
            heapq.heappush(pq, (w + int(h_node[nd_id]), w, nd_id))

    best_arrival = INF_TIME if pure_walk is None else pure_walk
    while pq:
        f, d, u = heapq.heappop(pq)
        if d > dist.get(u, INF_TIME):
            continue
        if u in tgt_attach:
            cand = t_dep + d + tgt_attach[u]
            if cand < best_arrival:
                best_arrival = cand
        if f >= best_arrival - t_dep:
            continue
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k]); w = int(weights[k])
            nd = d + w
            if nd < dist.get(v, INF_TIME):
                dist[v] = nd
                heapq.heappush(pq, (nd + int(h_node[v]), nd, v))
    return int(best_arrival) if best_arrival < INF_TIME else None
