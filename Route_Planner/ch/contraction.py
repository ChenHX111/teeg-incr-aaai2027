"""CH contraction on a TEEG.

We implement classical Contraction Hierarchies (Geisberger, Sanders, Schultes,
Delling 2008) on top of the TEEG event graph:

  1. Levels (node importance ranking) are computed in two modes:
       - "trivial"  : level = topological rank by (node_time, node_id).
                      Since every TEEG edge is time-respecting, every edge is
                      naturally "upward" and bidirectional upward Dijkstra
                      reduces to ordinary bidirectional Dijkstra (correct).
       - "contract" : greedy edge-difference priority, hop-limited witness
                      Dijkstra, shortcut insertion. Pure-Python; intended for
                      small graphs (≤ ~10⁴ nodes).
  2. Shortcuts: when contracting node v, for every pair (u→v→w) with u,w not
     yet contracted, we add a shortcut u→w of weight w(u→v)+w(v→w) iff no
     witness path of equal-or-less weight exists in the remaining graph.
  3. Storage backend: scipy.sparse-style CSR arrays (no extra dependency).
     `networkit` was *not* installed in the project venv (verified with
     `pip show networkit` — not found); falling back to scipy/numpy keeps the
     dependency footprint minimal.

Output: CHGraph stores the FULL CH edge set (original + shortcuts) split into
two CSRs, both keyed on "upward" edges (rank[v] > rank[u]):
  * up_out  : outgoing-by-source  — used by forward search
  * up_in   : incoming-by-target  — used by backward search (transpose)

Time-budget: contraction terminates early if `time_budget_s` is exceeded; any
non-contracted nodes get assigned the highest remaining levels (their original
edges still participate via the up_in / up_out CSR view).
"""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CHGraph:
    """Contraction-hierarchy view of a TEEG.

    Attributes
    ----------
    n_nodes : int
    level   : int32[n_nodes]                  node ranks (higher = more important)
    up_out_indptr  / up_out_indices / up_out_weights
        CSR of UPWARD outgoing edges per source (level[target] > level[source]).
        Used by forward search.
    up_in_indptr   / up_in_indices  / up_in_weights
        CSR of UPWARD incoming edges per target (i.e., edges (u,v) with
        level[u] < level[v], indexed by v giving list of (u, w)).
        Used by backward search.
    n_shortcuts : int
        Number of shortcut edges introduced by contraction (0 in trivial mode).
    build_seconds : float
    mode : str  ("trivial" or "contract")
    """
    n_nodes: int
    level: np.ndarray
    up_out_indptr: np.ndarray
    up_out_indices: np.ndarray
    up_out_weights: np.ndarray
    up_in_indptr: np.ndarray
    up_in_indices: np.ndarray
    up_in_weights: np.ndarray
    n_shortcuts: int
    build_seconds: float
    mode: str

    @property
    def n_edges(self) -> int:
        return int(len(self.up_out_indices))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csr_from_edges(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray,
                    by: str = "src"):
    """Build CSR (indptr, indices, weights) keyed by `src` or `dst`.

    Deduplicates (key, other) pairs keeping the MIN weight (so multi-edges or
    shortcut-overlay-original collisions resolve correctly).
    """
    if len(src) == 0:
        return (np.zeros(n + 1, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    w = w.astype(np.int32)
    key_a = src if by == "src" else dst
    key_b = dst if by == "src" else src
    # dedup-min by (key_a, key_b)
    flat = key_a * np.int64(n + 1) + key_b
    order = np.lexsort((w, flat))
    flat_s = flat[order]; ka = key_a[order]; kb = key_b[order]; ws = w[order]
    first = np.empty(len(flat_s), dtype=bool); first[0] = True
    first[1:] = flat_s[1:] != flat_s[:-1]
    ka_u = ka[first]; kb_u = kb[first]; ws_u = ws[first]
    # sort by ka_u and build indptr
    order2 = np.argsort(ka_u, kind="stable")
    ka_o = ka_u[order2].astype(np.int32)
    kb_o = kb_u[order2].astype(np.int32)
    ws_o = ws_u[order2]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, ka_o + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, kb_o, ws_o


def _trivial_levels(teeg) -> np.ndarray:
    """Topological rank using (node_time, node_id) as the tiebreaker."""
    n = teeg.n_nodes
    nt = teeg.node_time.astype(np.int64)
    key = nt * np.int64(n + 1) + np.arange(n, dtype=np.int64)
    order = np.argsort(key, kind="stable")
    level = np.empty(n, dtype=np.int32)
    level[order] = np.arange(n, dtype=np.int32)
    return level


def _split_for_search(level: np.ndarray, src: np.ndarray, dst: np.ndarray,
                      w: np.ndarray):
    """Split the edge set into:
      forward edges  E_up = {(u,v) : level[v] > level[u]}  — for forward search
      backward edges E_dn = {(u,v) : level[u] > level[v]}  — for backward search
                     stored in reverse (keyed by `v`) so backward at v can
                     follow them to `u`.
    Equal-level ties are routed to forward (arbitrary; for trivial mode levels
    are unique so this never fires)."""
    fwd_mask = level[dst] > level[src]
    bwd_mask = level[src] > level[dst]
    tie_mask = ~fwd_mask & ~bwd_mask
    if tie_mask.any():
        fwd_mask = fwd_mask | tie_mask
    return (src[fwd_mask], dst[fwd_mask], w[fwd_mask],
            src[bwd_mask], dst[bwd_mask], w[bwd_mask])


# ---------------------------------------------------------------------------
# Genuine contraction (small graphs only)
# ---------------------------------------------------------------------------

def _witness_search(adj: dict, source: int, target: int, via: int,
                    limit: int, hop_limit: int = 3) -> bool:
    """Return True iff a u→...→w path of weight ≤ limit exists in the
    remaining graph WITHOUT passing through `via`. Hop-limited to keep
    pure-Python contraction tractable.
    """
    pq = [(0, source, 0)]
    seen = {source: 0}
    while pq:
        d, u, h = heapq.heappop(pq)
        if d > seen.get(u, 10**18):
            continue
        if d > limit:
            return False
        if u == target:
            return True
        if h >= hop_limit:
            continue
        for v, ww in adj.get(u, ()):
            if v == via:
                continue
            nd = d + ww
            if nd < seen.get(v, 10**18) and nd <= limit:
                seen[v] = nd
                heapq.heappush(pq, (nd, v, h + 1))
    return False


def _contract_node(v: int, fwd: dict, rev: dict, hop_limit: int):
    """Simulate (and apply) the contraction of v. Returns (n_shortcuts_added,
    list_of_shortcut_edges_as_(u,w,weight))."""
    in_edges = list(rev.get(v, ()))      # (u, w(u→v))
    out_edges = list(fwd.get(v, ()))     # (w_, w(v→w_))
    shortcuts = []
    # for each (u, v, w1) and (v, w_, w2): possibly add u→w_
    for (u, w1) in in_edges:
        if u == v:
            continue
        for (w_, w2) in out_edges:
            if w_ == v or w_ == u:
                continue
            limit = w1 + w2
            if _witness_search(fwd, u, w_, via=v, limit=limit, hop_limit=hop_limit):
                continue
            shortcuts.append((u, w_, limit))
    # remove v from graph
    for (u, _w) in in_edges:
        fwd[u] = [(t, ww) for (t, ww) in fwd.get(u, ()) if t != v]
    for (w_, _w) in out_edges:
        rev[w_] = [(t, ww) for (t, ww) in rev.get(w_, ()) if t != v]
    fwd.pop(v, None); rev.pop(v, None)
    # apply shortcuts (keep min weight)
    n_added = 0
    for (u, w_, ww) in shortcuts:
        adj_u = fwd.setdefault(u, [])
        existing = next((i for i, (t, _) in enumerate(adj_u) if t == w_), None)
        if existing is None:
            adj_u.append((w_, ww))
            rev.setdefault(w_, []).append((u, ww))
            n_added += 1
        else:
            old_w = adj_u[existing][1]
            if ww < old_w:
                adj_u[existing] = (w_, ww)
                radj = rev.setdefault(w_, [])
                for i, (t, _) in enumerate(radj):
                    if t == u:
                        radj[i] = (u, ww); break
    return n_added, shortcuts


def _edge_difference_priority(v: int, fwd: dict, rev: dict, hop_limit: int) -> int:
    """Simulate contraction of v WITHOUT modifying graph; return |shortcuts| -
    |original adjacent edges|. Lower = better candidate."""
    in_edges = rev.get(v, ())
    out_edges = fwd.get(v, ())
    n_sh = 0
    for (u, w1) in in_edges:
        if u == v: continue
        for (w_, w2) in out_edges:
            if w_ == v or w_ == u: continue
            if not _witness_search(fwd, u, w_, via=v, limit=w1 + w2, hop_limit=hop_limit):
                n_sh += 1
    return n_sh - (len(in_edges) + len(out_edges))


def _genuine_contraction(n_nodes: int, src: np.ndarray, dst: np.ndarray,
                         w: np.ndarray, time_budget_s: float,
                         hop_limit: int, verbose: bool):
    """Run greedy edge-difference contraction. Returns (level, all_src, all_dst,
    all_w, n_shortcuts)."""
    fwd: dict = {}
    rev: dict = {}
    for s, d, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
        fwd.setdefault(s, []).append((d, ww))
        rev.setdefault(d, []).append((s, ww))
    # original edge list (preserved separately, since fwd/rev get mutated)
    orig_edges = list(zip(src.tolist(), dst.tolist(), w.tolist()))
    shortcut_edges = []

    # priority queue: (priority, version, node)
    versions = [0] * n_nodes
    pq = []
    for v in range(n_nodes):
        p = _edge_difference_priority(v, fwd, rev, hop_limit)
        heapq.heappush(pq, (p, 0, v))

    level = np.full(n_nodes, -1, dtype=np.int32)
    next_level = 0
    n_short = 0
    t0 = time.time()
    n_contracted = 0
    while pq:
        if time.time() - t0 > time_budget_s:
            if verbose:
                print(f"  [CH] time budget {time_budget_s:.1f}s reached after "
                      f"{n_contracted}/{n_nodes} contractions; assigning "
                      f"trivial levels to remainder")
            break
        p, ver, v = heapq.heappop(pq)
        if ver != versions[v] or level[v] != -1:
            continue
        # lazy update: recompute priority; if it's now worse than next, requeue
        p_now = _edge_difference_priority(v, fwd, rev, hop_limit)
        if pq and p_now > pq[0][0]:
            versions[v] += 1
            heapq.heappush(pq, (p_now, versions[v], v))
            continue
        # contract
        added, sc = _contract_node(v, fwd, rev, hop_limit)
        for (a, b, ww) in sc:
            shortcut_edges.append((a, b, ww))
        n_short += added
        level[v] = next_level; next_level += 1
        n_contracted += 1
        # invalidate neighbours' priorities (lazy)
        for (u, _) in rev.get(v, ()):  # already removed but be safe
            versions[u] += 1
        for (w_, _) in fwd.get(v, ()):
            versions[w_] += 1
    # assign remaining levels via trivial ordering of non-contracted nodes
    remaining = np.where(level == -1)[0]
    for v in remaining:
        level[v] = next_level; next_level += 1

    all_src = np.array([e[0] for e in orig_edges + shortcut_edges], dtype=np.int32)
    all_dst = np.array([e[1] for e in orig_edges + shortcut_edges], dtype=np.int32)
    all_w   = np.array([e[2] for e in orig_edges + shortcut_edges], dtype=np.int32)
    return level, all_src, all_dst, all_w, n_short


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ch_from_teeg(teeg, mode: str = "trivial",
                       time_budget_s: float = 30.0,
                       hop_limit: int = 3,
                       verbose: bool = True) -> CHGraph:
    """Build a CHGraph view of a TEEG.

    Parameters
    ----------
    teeg : TEEG produced by Route_Planner.teeg.builder.build_teeg
    mode : "trivial" | "contract"
        See module docstring.
    time_budget_s : float
        Hard cap on contraction time (genuine mode only). After this, the
        remaining nodes are assigned trivial levels.
    hop_limit : int
        Witness-search hop limit for contraction.
    """
    t0 = time.time()
    n = teeg.n_nodes
    # expand TEEG forward CSR back to edge list
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(teeg.indptr))
    dst = teeg.indices.astype(np.int32)
    w   = teeg.weights.astype(np.int32)

    if mode == "trivial":
        level = _trivial_levels(teeg)
        all_src, all_dst, all_w = src, dst, w
        n_short = 0
    elif mode == "contract":
        if verbose:
            print(f"  [CH] genuine contraction on n={n} nodes, "
                  f"m={len(src)} edges, hop_limit={hop_limit}, "
                  f"budget={time_budget_s:.1f}s")
        level, all_src, all_dst, all_w, n_short = _genuine_contraction(
            n, src, dst, w, time_budget_s, hop_limit, verbose)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    f_src, f_dst, f_w, b_src, b_dst, b_w = _split_for_search(
        level, all_src, all_dst, all_w)
    up_out_indptr, up_out_indices, up_out_weights = _csr_from_edges(
        n, f_src, f_dst, f_w, by="src")
    # backward CSR: at node v we read entries dst==v and expand to src
    up_in_indptr, up_in_indices, up_in_weights = _csr_from_edges(
        n, b_src, b_dst, b_w, by="dst")

    dt = time.time() - t0
    if verbose:
        print(f"  [CH] mode={mode} | levels assigned | "
              f"fwd_edges={len(f_src):,} bwd_edges={len(b_src):,} "
              f"(incl {n_short} shortcuts) | {dt:.2f}s")
    return CHGraph(
        n_nodes=n, level=level,
        up_out_indptr=up_out_indptr, up_out_indices=up_out_indices,
        up_out_weights=up_out_weights,
        up_in_indptr=up_in_indptr, up_in_indices=up_in_indices,
        up_in_weights=up_in_weights,
        n_shortcuts=int(n_short), build_seconds=float(dt), mode=mode,
    )


# ---------------------------------------------------------------------------
# Lightweight CH on a plain weighted DAG (used by the 5-node toy test)
# ---------------------------------------------------------------------------

def build_ch_from_edges(n_nodes: int, edges: list,
                        mode: str = "contract",
                        time_budget_s: float = 5.0,
                        hop_limit: int = 4,
                        verbose: bool = False) -> CHGraph:
    """Build a CH from a raw weighted edge list (for unit tests on toy graphs).

    Parameters
    ----------
    n_nodes : int
    edges   : list of (src, dst, weight) tuples, weights non-negative
    """
    src = np.array([e[0] for e in edges], dtype=np.int32)
    dst = np.array([e[1] for e in edges], dtype=np.int32)
    w   = np.array([e[2] for e in edges], dtype=np.int32)
    t0 = time.time()
    if mode == "contract":
        level, all_src, all_dst, all_w, n_short = _genuine_contraction(
            n_nodes, src, dst, w, time_budget_s, hop_limit, verbose)
    else:
        # trivial: rank by node id
        level = np.arange(n_nodes, dtype=np.int32)
        all_src, all_dst, all_w = src, dst, w
        n_short = 0
    f_src, f_dst, f_w, b_src, b_dst, b_w = _split_for_search(
        level, all_src, all_dst, all_w)
    up_out_indptr, up_out_indices, up_out_weights = _csr_from_edges(
        n_nodes, f_src, f_dst, f_w, by="src")
    up_in_indptr, up_in_indices, up_in_weights = _csr_from_edges(
        n_nodes, b_src, b_dst, b_w, by="dst")
    return CHGraph(
        n_nodes=n_nodes, level=level,
        up_out_indptr=up_out_indptr, up_out_indices=up_out_indices,
        up_out_weights=up_out_weights,
        up_in_indptr=up_in_indptr, up_in_indices=up_in_indices,
        up_in_weights=up_in_weights,
        n_shortcuts=int(n_short),
        build_seconds=float(time.time() - t0), mode=mode,
    )
