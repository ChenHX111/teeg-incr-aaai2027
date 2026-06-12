"""Hub Labels construction via pruned BFS/Dijkstra (Akiba et al. 2013 simplified).

Design choice: We build HL on the TEEG event graph (parallel to CH-on-TEEG).
Two modes are supported:
  1. FULL TEEG HL (max_hubs > 0): Labels on the 493K-node TEEG. Requires pruning
     for correctness; too slow in pure Python for full-scale. Research-grade.
  2. STOP-LEVEL HL (stop_level=True): All-pairs shortest path on the stop graph
     (~2300 nodes). Exact within the stop-level model. Combined with time-
     dependent source/destination attachment for the query. FAST to build and query.

For the MINIMAL VIABLE baseline, we use mode (2): stop-level HL.

Implementation note: scipy.sparse.csgraph.dijkstra provides C-speed APSP.
With 2300 stops, APSP completes in <5 seconds.
"""
from __future__ import annotations
import heapq
import time as _time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as sp_dijkstra, shortest_path


@dataclass
class HLIndex:
    """Hub Label index for stop-level HL.

    For the stop-level approach, we store the full stop-to-stop distance matrix
    (APSP). This is equivalent to HL where every stop is a hub — the degenerate
    but exact case. Query = matrix lookup + time-dependent attachment.
    """
    # Stop-level distance matrix (n_stops × n_stops), INF for unreachable
    stop_dist: Optional[np.ndarray] = None
    n_stops: int = 0
    # TEEG node-level labels (for full TEEG HL mode)
    fwd_labels: Optional[list] = None
    bwd_labels: Optional[list] = None
    n_nodes: int = 0
    # TEEG metadata for time-dependent attachment
    node_stop: Optional[np.ndarray] = None
    node_time: Optional[np.ndarray] = None
    node_kind: Optional[np.ndarray] = None
    stop_dep_indptr: Optional[np.ndarray] = None
    stop_dep_time: Optional[np.ndarray] = None
    stop_dep_node: Optional[np.ndarray] = None
    stop_arr_indptr: Optional[np.ndarray] = None
    stop_arr_time: Optional[np.ndarray] = None
    stop_arr_node: Optional[np.ndarray] = None
    walk_indptr: Optional[np.ndarray] = None
    walk_indices: Optional[np.ndarray] = None
    walk_weights: Optional[np.ndarray] = None
    rev_walk_indptr: Optional[np.ndarray] = None
    rev_walk_indices: Optional[np.ndarray] = None
    rev_walk_weights: Optional[np.ndarray] = None
    # Connection data for time-dependent queries
    conn_from: Optional[np.ndarray] = None
    conn_to: Optional[np.ndarray] = None
    conn_dep: Optional[np.ndarray] = None
    conn_arr: Optional[np.ndarray] = None
    conn_trip: Optional[np.ndarray] = None

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if self.stop_dist is not None:
            return self.stop_dist.nbytes / (1024 * 1024)
        if self.fwd_labels is not None:
            total_entries = sum(len(l) for l in self.fwd_labels) + sum(len(l) for l in self.bwd_labels)
            return total_entries * 16 / (1024 * 1024)
        return 0.0

    def avg_label_size(self) -> float:
        if self.stop_dist is not None:
            # In stop-level mode, "label size" = n_stops (full APSP)
            return float(self.n_stops)
        if self.fwd_labels is not None:
            total = sum(len(l) for l in self.fwd_labels) + sum(len(l) for l in self.bwd_labels)
            return total / (2 * self.n_nodes) if self.n_nodes > 0 else 0.0
        return 0.0


def _query_distance(fwd_u: list, bwd_v: list) -> int:
    """Label intersection: min over shared hubs of dist(u,h) + dist(h,v)."""
    INF = 2**31
    best = INF
    i, j = 0, 0
    lu, lv = len(fwd_u), len(bwd_v)
    while i < lu and j < lv:
        hu, du = fwd_u[i]
        hv, dv = bwd_v[j]
        if hu == hv:
            c = du + dv
            if c < best:
                best = c
            i += 1
            j += 1
        elif hu < hv:
            i += 1
        else:
            j += 1
    return best


def build_hl_labels_stop_level(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                               n_stops: int, verbose: bool = False) -> HLIndex:
    """Build stop-level HL via All-Pairs Shortest Path (APSP).

    This is the MINIMAL VIABLE approach: exact APSP on the stop-level graph.
    Equivalent to HL where every stop is a hub.

    Parameters
    ----------
    indptr, indices, weights : stop-level CSR graph (from connections/transfers)
    n_stops : number of stops
    verbose : print progress

    Returns
    -------
    HLIndex with stop_dist matrix.
    """
    t0 = _time.time()
    # Build scipy CSR
    csr = sp.csr_matrix((weights.astype(np.float64), indices, indptr), shape=(n_stops, n_stops))
    # APSP with scipy (C-speed Dijkstra from all nodes)
    dist_matrix = shortest_path(csr, directed=True, method='D')
    # Convert to int32 (minutes), INF -> 32000
    INF_TIME = 32000
    dist_matrix = np.where(np.isinf(dist_matrix), INF_TIME, dist_matrix).astype(np.int16)

    if verbose:
        elapsed = _time.time() - t0
        n_reachable = np.sum(dist_matrix < INF_TIME)
        print(f"  Stop-level HL: {n_stops} stops, APSP {elapsed:.1f}s, "
              f"{n_reachable} reachable pairs, "
              f"memory {dist_matrix.nbytes / 1024 / 1024:.1f} MB")

    return HLIndex(stop_dist=dist_matrix, n_stops=n_stops)


def build_stop_graph_from_connections(connections: np.ndarray, transfers_from: np.ndarray,
                                      n_stops: int, verbose: bool = False):
    """Build a stop-level graph from connection timetable data.

    Edge weight = minimum travel time (arr - dep) across all connections for each
    (from_stop, to_stop) pair. Walking transfers use walk time.

    Returns (indptr, indices, weights) CSR arrays.
    """
    from_s = connections["from_stop"].astype(np.int32)
    to_s = connections["to_stop"].astype(np.int32)
    travel = (connections["arr_min"] - connections["dep_min"]).astype(np.int32)

    # Build edge list: for each (from, to) pair, keep minimum travel time
    # Use a dict for dedup
    edge_min = {}
    for i in range(len(from_s)):
        key = (int(from_s[i]), int(to_s[i]))
        t = int(travel[i])
        if t > 0:
            if key not in edge_min or t < edge_min[key]:
                edge_min[key] = t

    # Add walking transfers
    if len(transfers_from) > 0:
        for row in transfers_from:
            fs, ts, wsec = int(row[0]), int(row[1]), int(row[2])
            wmin = max(1, (wsec + 59) // 60) if wsec > 0 else 1
            key = (fs, ts)
            if key not in edge_min or wmin < edge_min[key]:
                edge_min[key] = wmin

    # Convert to CSR
    if not edge_min:
        indptr = np.zeros(n_stops + 1, dtype=np.int32)
        indices = np.array([], dtype=np.int32)
        weights = np.array([], dtype=np.int32)
        return indptr, indices, weights

    srcs = np.array([k[0] for k in edge_min], dtype=np.int32)
    dsts = np.array([k[1] for k in edge_min], dtype=np.int32)
    wts = np.array([edge_min[k] for k in edge_min], dtype=np.int32)

    order = np.argsort(srcs, kind="stable")
    srcs = srcs[order]; dsts = dsts[order]; wts = wts[order]
    indptr = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(indptr, srcs + 1, 1)
    np.cumsum(indptr, out=indptr)

    if verbose:
        print(f"  Stop graph: {n_stops} nodes, {len(srcs)} edges")

    return indptr, dsts, wts


def build_hl_labels(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                    rev_indptr: np.ndarray, rev_indices: np.ndarray, rev_weights: np.ndarray,
                    n_nodes: int, max_label_size: int = 100,
                    max_hubs: int = 0, max_dist: float = 120.0,
                    verbose: bool = False) -> HLIndex:
    """Build Hub Labels on TEEG via scipy Dijkstra (no pruning, for research reference).

    NOTE: Without pruning, this gives upper-bound distances only.
    For correct results, use build_hl_labels_stop_level() instead.
    """
    INF = 2**31
    fwd_csr = sp.csr_matrix((weights.astype(np.float64), indices, indptr), shape=(n_nodes, n_nodes))
    rev_csr = sp.csr_matrix((rev_weights.astype(np.float64), rev_indices, rev_indptr), shape=(n_nodes, n_nodes))

    out_deg = np.diff(indptr).astype(np.int64)
    in_deg = np.diff(rev_indptr).astype(np.int64)
    importance = out_deg + in_deg
    order = np.argsort(-importance)

    n_process = min(max_hubs, n_nodes) if max_hubs > 0 else n_nodes
    hub_set = order[:n_process]

    fwd_labels: list = [[] for _ in range(n_nodes)]
    bwd_labels: list = [[] for _ in range(n_nodes)]

    t0 = _time.time()
    for rank in range(n_process):
        v = int(hub_set[rank])
        if verbose and rank % 100 == 0:
            elapsed = _time.time() - t0
            print(f"  HL: hub {rank}/{n_process} (elapsed {elapsed:.1f}s)", flush=True)

        fwd_dist = sp_dijkstra(fwd_csr, directed=True, indices=v,
                               limit=max_dist, return_predecessors=False)
        reachable_fwd = np.where(np.isfinite(fwd_dist) & (fwd_dist >= 0))[0]
        dists_fwd = fwd_dist[reachable_fwd].astype(np.int32)
        if len(reachable_fwd) > 10000:
            top_k = np.argpartition(dists_fwd, 10000)[:10000]
            reachable_fwd = reachable_fwd[top_k]
            dists_fwd = dists_fwd[top_k]
        for i in range(len(reachable_fwd)):
            u = int(reachable_fwd[i])
            d = int(dists_fwd[i])
            if len(bwd_labels[u]) < max_label_size:
                bwd_labels[u].append((v, d))

        bwd_dist = sp_dijkstra(rev_csr, directed=True, indices=v,
                               limit=max_dist, return_predecessors=False)
        reachable_bwd = np.where(np.isfinite(bwd_dist) & (bwd_dist >= 0))[0]
        dists_bwd = bwd_dist[reachable_bwd].astype(np.int32)
        if len(reachable_bwd) > 10000:
            top_k = np.argpartition(dists_bwd, 10000)[:10000]
            reachable_bwd = reachable_bwd[top_k]
            dists_bwd = dists_bwd[top_k]
        for i in range(len(reachable_bwd)):
            u = int(reachable_bwd[i])
            d = int(dists_bwd[i])
            if len(fwd_labels[u]) < max_label_size:
                fwd_labels[u].append((v, d))

    for u in range(n_nodes):
        fwd_labels[u].sort(key=lambda x: x[0])
        bwd_labels[u].sort(key=lambda x: x[0])

    if verbose:
        total_entries = sum(len(l) for l in fwd_labels) + sum(len(l) for l in bwd_labels)
        elapsed = _time.time() - t0
        print(f"  HL done: {n_process} hubs, {total_entries} label entries, {elapsed:.1f}s")

    return HLIndex(fwd_labels=fwd_labels, bwd_labels=bwd_labels, n_nodes=n_nodes)
