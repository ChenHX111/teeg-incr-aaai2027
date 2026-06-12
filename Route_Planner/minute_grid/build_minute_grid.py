"""Minute-Grid Stop graph -- state-augmented R/W version (CORRECTED).

History
=======
The v1 minute-grid had a CORRECTNESS BUG: walks could be chained
back-to-back through the (s, t) grid because WALK edges always fired from
any node identified as a "transit arrival event", regardless of whether
the rider got there by RIDE or by WALK.  This effectively bypassed the
15-min walk cap and allowed paths that CSA / RAPTOR / TEEG correctly
reject.  Verified on q=1745 (London): v1 MG returned 593 by chaining
walks 848 -> 2055 (11 min) -> 667 (6 min) = 17 min, while the closed
walking table forbids 848 -> 667 directly because it exceeds 15 min.

Corrected design (state-augmented)
==================================
Every stop has TWO planes of (s, t) nodes:

    R-plane :  R(s, t)  -- rider is at stop s at minute t having just
                            RIDDEN in (or being at the SOURCE).
                            Walks ARE allowed FROM R-state.
    W-plane :  W(s, t)  -- rider is at stop s at minute t having just
                            WALKED in.
                            Walks are NOT allowed FROM W-state.

Edges
-----
    WAIT-R   R(s, t)    --1-->  R(s, t+1)
    WAIT-W   W(s, t)    --1-->  W(s, t+1)
    RIDE     R(u, dep)  --d-->  R(v, arr)    (board from R)
    RIDE     W(u, dep)  --d-->  R(v, arr)    (board from W -- rider
                                              transitions to R after riding)
    WALK     R(s, t)    --w-->  W(s', t+w)   ONLY when (s, t) is a transit
                                              ARRIVAL event at s

Source walks are handled by the QUERY as a prelude:
    dist[R(src, t_dep)] = 0
    dist[W(neighbour, t_dep + walk)] = walk     for each closed walk
                                                 (src -> neighbour).

The target is reached when ANY R(dst, t) or W(dst, t) is popped from the
priority queue.

Cost (London subset, n_stops = 2922, t_hi = 780)
------------------------------------------------
  nodes      = 4,558,320   (~2x v1)
  wait edges = 4,558,320
  ride edges =   492,876   (2 per connection)
  walk edges ~ 5.0 M
  total      ~10.05 M edges; CSR ~110 MB; build ~ 6 s.

Query algorithms (in query_dial.py)
-----------------------------------
  * Dial's bucket-Dijkstra (priority-queue-free):  headline "BFS-like"
  * Plain Dijkstra (heap):                          reference

Both return earliest-arrival minutes that exactly match CSA / RAPTOR / TEEG
under the standard "at most one walk between rides" semantics.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data class
# ----------------------------------------------------------------------
@dataclass
class MinuteGrid:
    """Compressed sparse-row representation of the state-augmented minute-grid.

    Two planes (R = ride/init-state, W = walk-state).  Node id layout:

        R(s, t) = s * T + (t - t_lo)                       in [0, RW_OFFSET)
        W(s, t) = RW_OFFSET + s * T + (t - t_lo)           in [RW_OFFSET, 2*RW_OFFSET)

        RW_OFFSET = n_stops * T

    Fields:
      indptr     : (n_nodes + 1,) int64    CSR row pointers
      indices    : (n_edges,)    int32     destination node id
      weights    : (n_edges,)    int16     edge weight (minutes) in [0, 30]
      n_stops    : int                     |S|
      t_lo, t_hi : int                     minute window [t_lo, t_hi]
      T          : int = t_hi - t_lo + 1   minute slots per stop per plane
      RW_OFFSET  : int = n_stops * T       base of W-plane
      n_nodes    : int = 2 * n_stops * T
    """
    indptr:    np.ndarray
    indices:   np.ndarray
    weights:   np.ndarray
    n_stops:   int
    t_lo:      int
    t_hi:      int
    T:         int
    n_nodes:   int
    RW_OFFSET: int

    def node_R(self, s: int, t: int) -> int:
        return int(s) * self.T + (int(t) - self.t_lo)

    def node_W(self, s: int, t: int) -> int:
        return self.RW_OFFSET + int(s) * self.T + (int(t) - self.t_lo)

    def decode(self, nid: int):
        """Return (stop, minute, plane) where plane in {'R', 'W'}."""
        nid = int(nid)
        if nid < self.RW_OFFSET:
            s, off = divmod(nid, self.T)
            return s, self.t_lo + off, 'R'
        nid2 = nid - self.RW_OFFSET
        s, off = divmod(nid2, self.T)
        return s, self.t_lo + off, 'W'

    @property
    def n_edges(self) -> int:
        return int(len(self.indices))


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build_minute_grid(
    bundle,
    closed_walks_csr,
    t_lo: int = 0,
    t_hi: Optional[int] = None,
    *,
    walk_buffer: int = 60,
    verbose: bool = True,
) -> MinuteGrid:
    """Construct the minute-grid graph.

    Parameters
    ----------
    bundle : TimetableBundle
        From Route_Planner.common.load_bundle().
    closed_walks_csr : tuple (indptr, indices, walk_mins)
        From Route_Planner.csa.csa.build_closed_walk_csr(...).
    t_lo, t_hi : int
        Minute window (inclusive).  When ``t_hi`` is None the default is
        ``max(arr_min) + walk_buffer`` so the W-plane has room for walking
        suffix arrivals (otherwise late-evening queries truncate -> NaN).
    walk_buffer : int
        Extra minutes appended past the last connection's arrival so that
        terminal walks land inside the grid.  Default 60 (>= max closed
        walk minutes which is bounded by 15).

    Returns
    -------
    MinuteGrid
    """
    c = bundle.connections
    if t_hi is None:
        t_hi = int(max(c["arr_min"].max(), c["dep_min"].max())) + int(walk_buffer)
    T = t_hi - t_lo + 1
    n_stops = int(bundle.n_stops)
    RW_OFFSET = n_stops * T
    n_nodes = 2 * RW_OFFSET                        # R + W planes

    if verbose:
        print(f"[minute-grid] building (state-augmented R/W): "
              f"n_stops={n_stops}  T={T}  n_nodes={n_nodes:,}")
    t0 = time.time()

    # ----- WAIT edges in BOTH planes ---------------------------------
    # R-plane: R(s, t) -> R(s, t+1)
    # W-plane: W(s, t) -> W(s, t+1)
    s_idx = np.arange(n_stops, dtype=np.int64).reshape(-1, 1)
    t_idx = np.arange(t_lo, t_hi, dtype=np.int64).reshape(1, -1)
    wait_src_R = (s_idx * T + (t_idx - t_lo)).ravel()
    wait_dst_R = wait_src_R + 1
    wait_src_W = wait_src_R + RW_OFFSET
    wait_dst_W = wait_dst_R + RW_OFFSET
    wait_src = np.concatenate([wait_src_R, wait_src_W])
    wait_dst = np.concatenate([wait_dst_R, wait_dst_W])
    wait_w   = np.ones(wait_src.size, dtype=np.int16)
    if verbose:
        print(f"[minute-grid]   WAIT  {wait_src.size:>12,d} edges "
              f"(R+W planes)")

    # ----- RIDE edges: from R AND from W, landing on R ---------------
    # Keep weight-0 (arr==dep) edges; they encode instantaneous transitions
    # along a multi-stop trip (rider stays on the vehicle).
    dep = c["dep_min"].astype(np.int64)
    arr = c["arr_min"].astype(np.int64)
    mask = (dep >= t_lo) & (arr <= t_hi) & (arr >= dep)
    u = c["from_stop"][mask].astype(np.int64)
    v = c["to_stop"  ][mask].astype(np.int64)
    dep_m = dep[mask]; arr_m = arr[mask]
    ride_src_R = u * T + (dep_m - t_lo)               # R(u, dep)
    ride_src_W = ride_src_R + RW_OFFSET               # W(u, dep)
    ride_dst   = v * T + (arr_m - t_lo)               # R(v, arr) -- always R
    ride_src = np.concatenate([ride_src_R, ride_src_W])
    ride_dst = np.concatenate([ride_dst,   ride_dst])
    w0 = (arr_m - dep_m).astype(np.int16)
    ride_w   = np.concatenate([w0, w0])
    if verbose:
        n0 = int((w0 == 0).sum())
        print(f"[minute-grid]   RIDE  {ride_src.size:>12,d} edges "
              f"(R->R + W->R, {n0} weight-0 each)")

    # ----- STAY_ON_VEHICLE (trip continuation across gaps) ----------
    # The raw stop_times sometimes have NaT arrival/departure at intermediate
    # stops of a trip; data_layer.py drops those rows, producing trips whose
    # consecutive connections satisfy ``prev.to_stop != next.from_stop``.
    # CSA handles this transparently via its ``trip_reached`` marker (once
    # a trip is reached anywhere, every subsequent connection on the trip
    # auto-propagates regardless of the rider's per-stop tau).  TEEG models
    # the same semantics with explicit "stay-on-vehicle" edges
    # A(prev) -> D(next).  Without an analogous edge in MG, the rider falls
    # off the bus at every data gap, which produces 4-10 min delays on a
    # measurable fraction of queries.  These edges restore the missing
    # vehicle-continuity link in R-plane:
    #     R(prev.to, prev.arr_min)  --(dep_next - arr_prev)-->
    #     R(next.from, next.dep_min)
    # The rider must be on the vehicle (R-plane) before the gap; after the
    # gap they are still on the vehicle and ready to board next.
    t_stay = time.time()
    trip_id_col = c["trip_id"]
    trip_order  = np.lexsort((c["dep_min"], trip_id_col))   # sort by trip, then dep
    trip_sorted = trip_id_col[trip_order]
    if trip_sorted.size:
        breaks = np.concatenate(([0],
                                 np.where(np.diff(trip_sorted) != 0)[0] + 1,
                                 [trip_sorted.size]))
    else:
        breaks = np.array([0], dtype=np.int64)
    stay_src_chunks = []
    stay_dst_chunks = []
    stay_w_chunks   = []
    for k in range(len(breaks) - 1):
        seg = trip_order[breaks[k]:breaks[k+1]]
        if seg.size <= 1:
            continue
        prev = seg[:-1]
        nxt  = seg[1:]
        prev_to  = c["to_stop"][prev].astype(np.int64)
        nxt_from = c["from_stop"][nxt].astype(np.int64)
        gap = prev_to != nxt_from
        if not gap.any():
            continue
        prev_g = prev[gap]; nxt_g = nxt[gap]
        p_arr = c["arr_min"][prev_g].astype(np.int64)
        n_dep = c["dep_min"][nxt_g].astype(np.int64)
        p_to  = c["to_stop"][prev_g].astype(np.int64)
        n_from = c["from_stop"][nxt_g].astype(np.int64)
        keep = (p_arr >= t_lo) & (n_dep <= t_hi) & (n_dep >= p_arr)
        if not keep.any():
            continue
        p_to = p_to[keep]; p_arr = p_arr[keep]
        n_from = n_from[keep]; n_dep = n_dep[keep]
        src_R = p_to   * T + (p_arr - t_lo)     # R(prev.to, prev.arr)
        dst_R = n_from * T + (n_dep - t_lo)     # R(next.from, next.dep)
        w = (n_dep - p_arr).astype(np.int16)
        stay_src_chunks.append(src_R)
        stay_dst_chunks.append(dst_R)
        stay_w_chunks  .append(w)
    if stay_src_chunks:
        stay_src = np.concatenate(stay_src_chunks)
        stay_dst = np.concatenate(stay_dst_chunks)
        stay_w   = np.concatenate(stay_w_chunks)
    else:
        stay_src = np.zeros(0, dtype=np.int64)
        stay_dst = np.zeros(0, dtype=np.int64)
        stay_w   = np.zeros(0, dtype=np.int16)
    if verbose:
        print(f"[minute-grid]   STAY  {stay_src.size:>12,d} edges "
              f"(trip continuation across data gaps) "
              f"-- {time.time()-t_stay:.2f}s")

    # ----- WALK edges: R(s, t_arr) -> W(s', t_arr + w) ---------------
    # ONLY emitted from R-plane nodes that ARE actual transit arrival
    # events at (s, t).  Walks land on the W-plane so they cannot be
    # chained (W has no outgoing walks).
    to_arr  = c["to_stop"][mask].astype(np.int64)
    arr_arr = arr[mask]
    arr_keys = to_arr * T + (arr_arr - t_lo)          # R-plane node ids
    arr_nodes_unique = np.unique(arr_keys)
    arr_stops, arr_offs = np.divmod(arr_nodes_unique, T)
    arr_stops = arr_stops.astype(np.int64)
    arr_offs  = arr_offs.astype(np.int64)

    cw_indptr, cw_indices, cw_wmins = closed_walks_csr
    walk_src_chunks, walk_dst_chunks, walk_w_chunks = [], [], []
    for i in range(arr_nodes_unique.size):
        s   = int(arr_stops[i])
        toff = int(arr_offs[i])               # = t - t_lo
        lo, hi = int(cw_indptr[s]), int(cw_indptr[s + 1])
        if hi == lo:
            continue
        sp_arr = cw_indices[lo:hi].astype(np.int64)
        w_arr  = cw_wmins  [lo:hi].astype(np.int64)
        keep = (w_arr >= 1) & (sp_arr != s) & (toff + w_arr <= T - 1)
        if not keep.any():
            continue
        sp_arr = sp_arr[keep]; w_arr = w_arr[keep]
        src_nodes = np.full(sp_arr.size, s * T + toff,
                            dtype=np.int64)             # R-plane
        dst_nodes = RW_OFFSET + sp_arr * T + (toff + w_arr)   # W-plane
        walk_src_chunks.append(src_nodes)
        walk_dst_chunks.append(dst_nodes)
        walk_w_chunks  .append(w_arr.astype(np.int16))
    walk_src = (np.concatenate(walk_src_chunks)
                if walk_src_chunks else np.zeros(0, dtype=np.int64))
    walk_dst = (np.concatenate(walk_dst_chunks)
                if walk_dst_chunks else np.zeros(0, dtype=np.int64))
    walk_w   = (np.concatenate(walk_w_chunks)
                if walk_w_chunks   else np.zeros(0, dtype=np.int16))
    if verbose:
        print(f"[minute-grid]   WALK  {walk_src.size:>12,d} edges "
              f"R->W ({arr_nodes_unique.size} arrival events)")

    # ----- Combine ---------------------------------------------------
    all_src = np.concatenate([wait_src, ride_src, stay_src, walk_src])
    all_dst = np.concatenate([wait_dst, ride_dst, stay_dst, walk_dst])
    all_w   = np.concatenate([wait_w,   ride_w,   stay_w,   walk_w])
    del wait_src, ride_src, stay_src, walk_src
    del wait_dst, ride_dst, stay_dst, walk_dst
    del wait_w,  ride_w,  stay_w, walk_w
    del walk_src_chunks, walk_dst_chunks, walk_w_chunks
    del stay_src_chunks, stay_dst_chunks, stay_w_chunks

    if verbose:
        print(f"[minute-grid]   TOTAL {all_src.size:>12,d} edges; sorting...")

    # ----- Build CSR (sorted by src) --------------------------------
    order = np.argsort(all_src, kind="stable")
    indices = all_dst[order].astype(np.int32)
    weights = all_w  [order]
    del all_dst, all_w
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, all_src[order] + 1, 1)
    np.cumsum(indptr, out=indptr)
    del all_src, order

    g = MinuteGrid(
        indptr=indptr, indices=indices, weights=weights,
        n_stops=n_stops, t_lo=t_lo, t_hi=t_hi, T=T,
        n_nodes=n_nodes, RW_OFFSET=RW_OFFSET,
    )
    if verbose:
        mb = (indptr.nbytes + indices.nbytes + weights.nbytes) / 1e6
        print(f"[minute-grid] built in {time.time() - t0:.1f}s, "
              f"CSR size {mb:.1f} MB")
    return g


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_minute_grid(g: MinuteGrid, path: str) -> None:
    np.savez_compressed(
        path,
        indptr=g.indptr, indices=g.indices, weights=g.weights,
        meta=np.array([g.n_stops, g.t_lo, g.t_hi, g.T, g.n_nodes,
                       g.RW_OFFSET], dtype=np.int64),
    )


def load_minute_grid(path: str) -> MinuteGrid:
    z = np.load(path)
    m = z["meta"]
    return MinuteGrid(
        indptr=z["indptr"], indices=z["indices"], weights=z["weights"],
        n_stops=int(m[0]), t_lo=int(m[1]), t_hi=int(m[2]),
        T=int(m[3]), n_nodes=int(m[4]), RW_OFFSET=int(m[5]),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from Route_Planner.common import load_bundle
    from Route_Planner.csa.csa import (
        _build_transitively_closed_transfers, build_closed_walk_csr,
    )

    b = load_bundle(load_raptor=False)
    cd = _build_transitively_closed_transfers(b.transfers_from, b.n_stops,
                                              max_walk_min=15)
    cwc = build_closed_walk_csr(cd, b.n_stops)
    g = build_minute_grid(b, cwc)
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "Graph_Builder", "minute_grid_full_day.npz")
    save_minute_grid(g, out)
    print(f"[minute-grid] saved -> {out}")
