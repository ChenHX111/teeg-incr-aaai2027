"""Time-Expanded Event Graph (TEEG) builder.

Constructs a directed graph where each node is a single timetable event
(departure or arrival) and each edge corresponds to one elementary passenger
transition with weight = elapsed minutes.

Node layout:
  For each elementary connection c=(u,v,dep,arr,trip,route,mode), we create
  two nodes:
    D(c) — departure event at stop u at time dep
    A(c) — arrival event at stop v at time arr

Edge categories (weight = elapsed minutes):
  1. RIDE             D(c) -> A(c)                    weight = arr - dep
  2. STAY_ON_VEHICLE  A(prev) -> D(next) same trip    weight = dep_next - arr_prev
  3. WAIT_AT_STOP     D_i(s) -> D_{i+1}(s) at s       weight = dep_{i+1} - dep_i
  4. BOARD_AT_STOP    A(c) -> next D(c') at s         weight = dep_{c'} - arr_c
  5. WALK_AND_BOARD   A(c at s) -> first D(c') at s'  weight = dep_{c'} - arr_c
                                                      (only if dep_{c'} >= arr_c + walk(s,s'))

Per-query attachments (not stored in the graph):
  SOURCE -> first feasible D(.) at source or walking-neighbour
  A(.) at dest or walking-neighbour -> SINK   (handled as a "target-extra"
                                                cost in the query)

Output (.npz fields):
  indptr, indices, weights        : scipy.sparse.csr_matrix layout
  rev_indptr, rev_indices, rev_weights : transpose, for reverse queries
  node_kind   int8[n_nodes]       0 = departure, 1 = arrival
  node_stop   int32[n_nodes]
  node_time   int16[n_nodes]      minutes since SERVICE_START
  node_trip   int32[n_nodes]
  node_route  int32[n_nodes]
  first_dep_event_idx int32[n_stops+1]   CSR-like layout of D-events sorted by time
  first_dep_event_time int16[n_dep_events] sorted dep_min per stop
  first_dep_event_node int32[n_dep_events] node id of the D-event
  walking_adj_indptr / indices / weights  walking footpaths CSR (stop->stop, weight = walk_min)
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.sparse as sp
from ..common import TimetableBundle, DAY_MIN


@dataclass
class TEEG:
    """Compiled time-expanded event graph in CSR form."""
    # forward CSR
    indptr:  np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    # reverse CSR (transpose)
    rev_indptr:  np.ndarray
    rev_indices: np.ndarray
    rev_weights: np.ndarray
    # node attributes
    node_kind:  np.ndarray            # 0 = D, 1 = A
    node_stop:  np.ndarray
    node_time:  np.ndarray
    node_trip:  np.ndarray
    node_route: np.ndarray
    # per-stop D-event index (for source attachment in O(log))
    stop_dep_indptr: np.ndarray       # (S+1,)
    stop_dep_time:   np.ndarray       # (n_D,)
    stop_dep_node:   np.ndarray       # (n_D,)
    # per-stop A-event index (for sink attachment in reverse query)
    stop_arr_indptr: np.ndarray
    stop_arr_time:   np.ndarray
    stop_arr_node:   np.ndarray
    # walking footpath CSR (stop -> stop, weight = walk_min)
    walk_indptr:  np.ndarray
    walk_indices: np.ndarray
    walk_weights: np.ndarray
    rev_walk_indptr:  np.ndarray
    rev_walk_indices: np.ndarray
    rev_walk_weights: np.ndarray
    n_stops: int
    t_lo: int
    t_hi: int

    @property
    def n_nodes(self) -> int:
        return len(self.node_kind)

    @property
    def n_edges(self) -> int:
        return len(self.indices)

    def save(self, path: str):
        np.savez_compressed(
            path,
            indptr=self.indptr, indices=self.indices, weights=self.weights,
            rev_indptr=self.rev_indptr, rev_indices=self.rev_indices, rev_weights=self.rev_weights,
            node_kind=self.node_kind, node_stop=self.node_stop, node_time=self.node_time,
            node_trip=self.node_trip, node_route=self.node_route,
            stop_dep_indptr=self.stop_dep_indptr, stop_dep_time=self.stop_dep_time,
            stop_dep_node=self.stop_dep_node,
            stop_arr_indptr=self.stop_arr_indptr, stop_arr_time=self.stop_arr_time,
            stop_arr_node=self.stop_arr_node,
            walk_indptr=self.walk_indptr, walk_indices=self.walk_indices, walk_weights=self.walk_weights,
            rev_walk_indptr=self.rev_walk_indptr, rev_walk_indices=self.rev_walk_indices,
            rev_walk_weights=self.rev_walk_weights,
            n_stops=np.int32(self.n_stops), t_lo=np.int32(self.t_lo), t_hi=np.int32(self.t_hi),
        )

    @classmethod
    def load(cls, path: str) -> "TEEG":
        z = np.load(path)
        return cls(
            indptr=z["indptr"], indices=z["indices"], weights=z["weights"],
            rev_indptr=z["rev_indptr"], rev_indices=z["rev_indices"], rev_weights=z["rev_weights"],
            node_kind=z["node_kind"], node_stop=z["node_stop"], node_time=z["node_time"],
            node_trip=z["node_trip"], node_route=z["node_route"],
            stop_dep_indptr=z["stop_dep_indptr"], stop_dep_time=z["stop_dep_time"],
            stop_dep_node=z["stop_dep_node"],
            stop_arr_indptr=z["stop_arr_indptr"], stop_arr_time=z["stop_arr_time"],
            stop_arr_node=z["stop_arr_node"],
            walk_indptr=z["walk_indptr"], walk_indices=z["walk_indices"], walk_weights=z["walk_weights"],
            rev_walk_indptr=z["rev_walk_indptr"], rev_walk_indices=z["rev_walk_indices"], rev_walk_weights=z["rev_walk_weights"],
            n_stops=int(z["n_stops"]), t_lo=int(z["t_lo"]), t_hi=int(z["t_hi"]),
        )


def _build_walk_csr(transfers_from: np.ndarray, n_stops: int):
    """Walking footpaths in CSR. weight = walk_min (ceil from seconds).
    Deduplicates (from, to) pairs keeping the MIN walk."""
    rows = transfers_from[:, 0].astype(np.int64)
    cols = transfers_from[:, 1].astype(np.int64)
    wsec = transfers_from[:, 2]
    wmin = np.where(wsec > 0, np.maximum(1, (wsec + 59) // 60), 0).astype(np.int32)
    if len(rows) == 0:
        empty = np.zeros(n_stops + 1, dtype=np.int32)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    key = rows * np.int64(n_stops) + cols
    order = np.lexsort((wmin, key))
    key_s = key[order]; rows_s = rows[order]; cols_s = cols[order]; wmin_s = wmin[order]
    first = np.empty(len(key_s), dtype=bool); first[0] = True
    first[1:] = key_s[1:] != key_s[:-1]
    rows_u = rows_s[first].astype(np.int32)
    cols_u = cols_s[first].astype(np.int32)
    wmin_u = wmin_s[first]
    # build CSR manually
    order = np.argsort(rows_u, kind="stable")
    rs = rows_u[order]; cs = cols_u[order]; ws = wmin_u[order]
    indptr = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(indptr, rs + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, cs, ws


def build_teeg(b: TimetableBundle, verbose: bool = True,
               closed_transfers_from: np.ndarray | None = None) -> TEEG:
    """Compile the bundle's connection list into a TEEG.

    Parameters
    ----------
    closed_transfers_from : optional Nx3 array (src, dst, walk_seconds) of
        transitively closed walking footpaths. If provided, this is used
        instead of `b.transfers_from` for the walking CSR. Used to obtain a
        timetable-correct TEEG even when the raw footpath set in the bundle
        is not closed (see csa._build_transitively_closed_transfers).
    """
    t0 = time.time()
    c = b.connections
    n_conn = len(c)
    n_nodes = 2 * n_conn

    # ---- node attributes
    node_kind  = np.empty(n_nodes, dtype=np.int8)
    node_stop  = np.empty(n_nodes, dtype=np.int32)
    node_time  = np.empty(n_nodes, dtype=np.int16)
    node_trip  = np.empty(n_nodes, dtype=np.int32)
    node_route = np.empty(n_nodes, dtype=np.int32)

    # D-event at even indices, A-event at odd indices
    d_ids = np.arange(0, n_nodes, 2, dtype=np.int64)
    a_ids = np.arange(1, n_nodes, 2, dtype=np.int64)
    node_kind[d_ids] = 0
    node_kind[a_ids] = 1
    node_stop[d_ids] = c["from_stop"]
    node_stop[a_ids] = c["to_stop"]
    node_time[d_ids] = c["dep_min"]
    node_time[a_ids] = c["arr_min"]
    node_trip[d_ids] = c["trip_id"]
    node_trip[a_ids] = c["trip_id"]
    node_route[d_ids] = c["route_int"]
    node_route[a_ids] = c["route_int"]

    if verbose:
        print(f"  [TEEG] {n_nodes:,} nodes ({n_conn:,} D + {n_conn:,} A) — {time.time()-t0:.2f}s")

    # ---- edges ----------------------------------------------------------------
    src_list = []
    dst_list = []
    wt_list  = []

    # (1) RIDE: D(c) -> A(c), weight = arr - dep
    t1 = time.time()
    src_list.append(d_ids)
    dst_list.append(a_ids)
    ride_w = c["arr_min"].astype(np.int32) - c["dep_min"].astype(np.int32)
    # CYCLE 9 SAFETY NET: warn (not assert) if any negative RIDE edge slips through.
    # The root-cause fix is in Route_Planner/common/data_layer.py::_collapse_consecutive_same_stop
    # but we keep this as a defense-in-depth check.
    n_neg = int((ride_w < 0).sum())
    if n_neg:
        import warnings
        warnings.warn(
            f"build_teeg: {n_neg:,} RIDE edges with negative weight (arr < dep); "
            f"this should be impossible after _collapse_consecutive_same_stop. "
            f"Did you load a stale cache or skip the data_layer fix?",
            RuntimeWarning,
        )
    wt_list.append(ride_w)
    if verbose: print(f"  [TEEG] ride edges: {n_conn:,} — {time.time()-t1:.2f}s")

    # (2) STAY_ON_VEHICLE: A(prev) -> D(next) same trip
    t1 = time.time()
    # group connections by trip_id; within each trip, the connection order in
    # `c` may not be the route order — but we did `lexsort((arr, dep))`. Trips
    # have monotone-increasing dep within their stops, so sorting connections
    # of the same trip by dep gives the route order.
    trip_id_col = c["trip_id"]
    trip_order  = np.lexsort((c["dep_min"], trip_id_col))   # sort by trip, then dep
    trip_sorted = trip_id_col[trip_order]
    # find boundaries between trips
    breaks = np.concatenate(([0], np.where(np.diff(trip_sorted) != 0)[0] + 1, [len(trip_sorted)]))
    so_src = []
    so_dst = []
    so_wt  = []
    for k in range(len(breaks) - 1):
        seg = trip_order[breaks[k]:breaks[k+1]]   # connection indices for this trip in dep order
        if len(seg) <= 1:
            continue
        prev_a = a_ids[seg[:-1]]
        next_d = d_ids[seg[1:]]
        w = c["dep_min"][seg[1:]].astype(np.int32) - c["arr_min"][seg[:-1]].astype(np.int32)
        so_src.append(prev_a); so_dst.append(next_d); so_wt.append(w)
    if so_src:
        src_list.append(np.concatenate(so_src))
        dst_list.append(np.concatenate(so_dst))
        wt_list.append(np.concatenate(so_wt))
        n_so = len(np.concatenate(so_src))
    else:
        n_so = 0
    if verbose: print(f"  [TEEG] stay-on edges: {n_so:,} — {time.time()-t1:.2f}s")

    # (3) WAIT at stop: D_i(s) -> D_{i+1}(s) at same stop (sorted by dep_min)
    t1 = time.time()
    # group D events by stop
    d_stops = c["from_stop"]
    d_times = c["dep_min"]
    d_order = np.lexsort((d_times, d_stops))     # sort by stop, then by dep_min
    d_stops_s = d_stops[d_order]
    breaks = np.concatenate(([0], np.where(np.diff(d_stops_s) != 0)[0] + 1, [len(d_stops_s)]))
    wt_src = []
    wt_dst = []
    wt_w   = []
    # also build the stop -> sorted-D-event index for source attachment
    stop_dep_indptr = np.zeros(b.n_stops + 1, dtype=np.int32)
    stop_dep_time   = np.empty(n_conn, dtype=np.int16)
    stop_dep_node   = np.empty(n_conn, dtype=np.int32)
    write_pos = 0
    for k in range(len(breaks) - 1):
        seg = d_order[breaks[k]:breaks[k+1]]
        if len(seg) == 0:
            continue
        s = int(d_stops[seg[0]])
        # wait chain
        if len(seg) > 1:
            prev_d = d_ids[seg[:-1]]
            next_d = d_ids[seg[1:]]
            w = c["dep_min"][seg[1:]].astype(np.int32) - c["dep_min"][seg[:-1]].astype(np.int32)
            wt_src.append(prev_d); wt_dst.append(next_d); wt_w.append(w)
        # store per-stop D-event index
        n_seg = len(seg)
        stop_dep_indptr[s+1] = write_pos + n_seg
        stop_dep_time[write_pos:write_pos+n_seg] = c["dep_min"][seg]
        stop_dep_node[write_pos:write_pos+n_seg] = d_ids[seg]
        write_pos += n_seg
    # cumulative
    stop_dep_indptr = np.maximum.accumulate(stop_dep_indptr)
    if wt_src:
        src_list.append(np.concatenate(wt_src))
        dst_list.append(np.concatenate(wt_dst))
        wt_list.append(np.concatenate(wt_w))
        n_wait = len(np.concatenate(wt_src))
    else:
        n_wait = 0
    if verbose: print(f"  [TEEG] wait edges: {n_wait:,} — {time.time()-t1:.2f}s")

    # Per-stop A-event index (for sink attachment in reverse queries)
    t1 = time.time()
    a_stops = c["to_stop"]
    a_times = c["arr_min"]
    a_order = np.lexsort((a_times, a_stops))
    stop_arr_indptr = np.zeros(b.n_stops + 1, dtype=np.int32)
    stop_arr_time   = np.empty(n_conn, dtype=np.int16)
    stop_arr_node   = np.empty(n_conn, dtype=np.int32)
    a_stops_s = a_stops[a_order]
    breaks = np.concatenate(([0], np.where(np.diff(a_stops_s) != 0)[0] + 1, [len(a_stops_s)]))
    write_pos = 0
    for k in range(len(breaks) - 1):
        seg = a_order[breaks[k]:breaks[k+1]]
        if len(seg) == 0:
            continue
        s = int(a_stops[seg[0]])
        n_seg = len(seg)
        stop_arr_indptr[s+1] = write_pos + n_seg
        stop_arr_time[write_pos:write_pos+n_seg] = c["arr_min"][seg]
        stop_arr_node[write_pos:write_pos+n_seg] = a_ids[seg]
        write_pos += n_seg
    stop_arr_indptr = np.maximum.accumulate(stop_arr_indptr)
    if verbose: print(f"  [TEEG] A-event index built — {time.time()-t1:.2f}s")

    # (4) BOARD: A(c) -> next D at same stop with dep >= arr
    # (5) WALK_AND_BOARD: A(c at s) -> first D at neighbour s' with dep >= arr + walk
    t1 = time.time()
    walk_src = b.transfers_from if closed_transfers_from is None else closed_transfers_from
    walk_indptr, walk_indices, walk_weights = _build_walk_csr(walk_src, b.n_stops)

    bs_src = []; bs_dst = []; bs_w = []
    n_board = 0
    n_walk_board = 0
    for ci in range(n_conn):
        a_node = int(a_ids[ci])
        s      = int(c["to_stop"][ci])
        a_t    = int(c["arr_min"][ci])
        # (4) board at same stop s, dep >= a_t
        lo, hi = int(stop_dep_indptr[s]), int(stop_dep_indptr[s+1])
        if hi > lo:
            times = stop_dep_time[lo:hi]
            k = int(np.searchsorted(times, a_t, side="left"))
            if k < (hi - lo):
                d_node = int(stop_dep_node[lo + k])
                w = int(stop_dep_time[lo + k]) - a_t
                bs_src.append(a_node); bs_dst.append(d_node); bs_w.append(w)
                n_board += 1
        # (5) walk + board at neighbours s'
        wlo, whi = int(walk_indptr[s]), int(walk_indptr[s+1])
        for wk in range(wlo, whi):
            sp_ = int(walk_indices[wk])
            wmin = int(walk_weights[wk])
            if sp_ == s and wmin == 0:
                continue   # self-loop covered by (4)
            t_threshold = a_t + wmin
            dlo, dhi = int(stop_dep_indptr[sp_]), int(stop_dep_indptr[sp_+1])
            if dhi > dlo:
                times = stop_dep_time[dlo:dhi]
                k = int(np.searchsorted(times, t_threshold, side="left"))
                if k < (dhi - dlo):
                    d_node = int(stop_dep_node[dlo + k])
                    w = int(stop_dep_time[dlo + k]) - a_t
                    bs_src.append(a_node); bs_dst.append(d_node); bs_w.append(w)
                    n_walk_board += 1
    if bs_src:
        src_list.append(np.asarray(bs_src, dtype=np.int64))
        dst_list.append(np.asarray(bs_dst, dtype=np.int64))
        wt_list .append(np.asarray(bs_w,   dtype=np.int32))
    if verbose:
        print(f"  [TEEG] board edges: {n_board:,}  walk+board edges: {n_walk_board:,} — {time.time()-t1:.2f}s")

    src_all = np.concatenate(src_list).astype(np.int32)
    dst_all = np.concatenate(dst_list).astype(np.int32)
    wt_all  = np.concatenate(wt_list ).astype(np.int32)

    # Deduplicate (src, dst) pairs, keeping the MIN weight.
    # scipy.sparse.csr_matrix would SUM duplicates -- catastrophic for routing.
    # Strategy: encode (u, v) as a single int64 key, find unique-min via lexsort.
    t1 = time.time()
    key = src_all.astype(np.int64) * np.int64(n_nodes) + dst_all.astype(np.int64)
    order = np.lexsort((wt_all, key))     # secondary by weight: smallest weight first per key
    key_s = key[order]
    src_s = src_all[order]; dst_s = dst_all[order]; wt_s = wt_all[order]
    # Keep the FIRST entry per unique key (which has min weight thanks to lexsort)
    first_in_group = np.empty(len(key_s), dtype=bool)
    first_in_group[0] = True
    first_in_group[1:] = key_s[1:] != key_s[:-1]
    src_u = src_s[first_in_group]
    dst_u = dst_s[first_in_group]
    wt_u  = wt_s [first_in_group]
    if verbose:
        print(f"  [TEEG] dedup edges: {len(src_all):,} -> {len(src_u):,} unique — {time.time()-t1:.2f}s")

    # Build forward CSR manually (sort by src, group)
    t1 = time.time()
    order = np.argsort(src_u, kind="stable")
    src_o = src_u[order]; dst_o = dst_u[order]; wt_o = wt_u[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(indptr, src_o + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = dst_o
    weights = wt_o
    if verbose: print(f"  [TEEG] forward CSR: {len(indices):,} edges — {time.time()-t1:.2f}s")

    # Reverse CSR: sort by dst (becomes src in reverse)
    t1 = time.time()
    rorder = np.argsort(dst_u, kind="stable")
    src_r = dst_u[rorder]; dst_r = src_u[rorder]; wt_r = wt_u[rorder]
    rev_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(rev_indptr, src_r + 1, 1)
    np.cumsum(rev_indptr, out=rev_indptr)
    rev_indices = dst_r
    rev_weights = wt_r
    if verbose: print(f"  [TEEG] reverse CSR: {len(rev_indices):,} edges — {time.time()-t1:.2f}s")

    # reverse walk CSR (transpose of dedup'd walk CSR)
    # walk_indices/walk_weights already deduped; just transpose by re-sorting on target
    n = b.n_stops
    src_w = np.repeat(np.arange(n, dtype=np.int32), np.diff(walk_indptr))
    dst_w = walk_indices
    wts_w = walk_weights
    rorder = np.argsort(dst_w, kind="stable")
    rsrc = dst_w[rorder]; rdst = src_w[rorder]; rwts = wts_w[rorder]
    rev_walk_indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(rev_walk_indptr, rsrc + 1, 1)
    np.cumsum(rev_walk_indptr, out=rev_walk_indptr)
    rev_walk_indices = rdst
    rev_walk_weights = rwts

    if verbose: print(f"  [TEEG] TOTAL build: {time.time()-t0:.2f}s")

    return TEEG(
        indptr=indptr, indices=indices, weights=weights,
        rev_indptr=rev_indptr, rev_indices=rev_indices, rev_weights=rev_weights,
        node_kind=node_kind, node_stop=node_stop, node_time=node_time,
        node_trip=node_trip, node_route=node_route,
        stop_dep_indptr=stop_dep_indptr, stop_dep_time=stop_dep_time, stop_dep_node=stop_dep_node,
        stop_arr_indptr=stop_arr_indptr, stop_arr_time=stop_arr_time, stop_arr_node=stop_arr_node,
        walk_indptr=walk_indptr, walk_indices=walk_indices, walk_weights=walk_weights,
        rev_walk_indptr=rev_walk_indptr, rev_walk_indices=rev_walk_indices, rev_walk_weights=rev_walk_weights,
        n_stops=int(b.n_stops), t_lo=int(b.t_lo), t_hi=int(b.t_hi),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Graph_Builder/artifacts/teeg_full_day.npz")
    ap.add_argument("--t_lo", type=int, default=0)
    ap.add_argument("--t_hi", type=int, default=DAY_MIN)
    args = ap.parse_args()

    from ..common import load_bundle
    print(f"Loading bundle t=[{args.t_lo},{args.t_hi}] ...")
    b = load_bundle(t_lo=args.t_lo, t_hi=args.t_hi, load_raptor=False)
    g = build_teeg(b, verbose=True)
    print(f"Saving to {args.out} ...")
    g.save(args.out)
    sz = os.path.getsize(args.out) / 1024 / 1024
    print(f"Saved.  File size: {sz:.1f} MB")
