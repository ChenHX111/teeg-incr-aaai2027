"""Incremental state for TEEG-ALT-INCR (EXT3).

Holds tombstone bitmaps (event + edge), trip→node/edge indexes for O(1)
cancel, and a pre-allocated overlay-CSR for ADD-ed trips.

Memory layout follows DESIGN.md §2.2 / §6.4.5.

Overlay node-id encoding (also used in PQ entries and overlay indices arrays):
    raw = -(ov_id + 1)     # negative ⇒ overlay node
    ov_id = -raw - 1
This is identical to the trick used in `Route_Planner.ultra.sidecar`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

INF_I32 = np.int32(2_147_483_647)


@dataclass
class IncrementalState:
    """All mutable arrays for TEEG-ALT-INCR. Static arrays come from a TEEG `g`."""

    # ---- references back to the static TEEG (read-only) ----
    n_nodes: int
    n_edges: int
    n_stops: int

    # ---- tombstones (uint8 byte form for hot-loop simplicity) ----
    node_alive: np.ndarray   # uint8[V]
    edge_alive: np.ndarray   # uint8[E]

    # ---- per-trip event/edge index for O(1) cancel ----
    trip_first_node: np.ndarray   # int32[T+1]; trip_node_idx[lo:hi] = events of trip t
    trip_node_idx:   np.ndarray   # int32[Σ events per trip]

    # Per-trip-event incident-edge index (forward CSR positions). Used by
    # cancel_trip to tombstone every IN- and OUT-edge of every event on the
    # trip in O(degree) without binary searches.
    trip_edge_off:   np.ndarray   # int32[T+1]
    trip_edge_idx:   np.ndarray   # int32[Σ in/out edges per trip event] (forward-CSR positions)

    # ---- overlay (added trips) ----
    overlay_capacity_nodes: int
    overlay_capacity_edges: int
    overlay_n_nodes: int
    overlay_n_edges: int
    overlay_n_main_fwd_edges: int   # main→overlay BOARD edges currently emitted

    # overlay node attrs (parallel to main node_*; int32 time, not int16)
    overlay_node_kind:  np.ndarray   # int8[cap_nodes]
    overlay_node_stop:  np.ndarray   # int32[cap_nodes]
    overlay_node_time:  np.ndarray   # int32[cap_nodes]   (minutes since SERVICE_START; int32 to allow t0_delta)
    overlay_node_trip:  np.ndarray   # int32[cap_nodes]   (negative synthetic trip ids)
    overlay_node_route: np.ndarray   # int32[cap_nodes]
    overlay_node_alive: np.ndarray   # uint8[cap_nodes]

    # overlay→{overlay,main} CSR  (case C+D in DESIGN §6.4.3)
    overlay_indptr:  np.ndarray   # int32[cap_nodes + 1]
    overlay_indices: np.ndarray   # int32[cap_edges]   (signed; negative ⇒ overlay)
    overlay_weights: np.ndarray   # int32[cap_edges]

    # main→overlay CSR  (case B; rebuilt eagerly per batch)
    overlay_fwd_from_main_indptr:  np.ndarray   # int32[V + 1]
    overlay_fwd_from_main_indices: np.ndarray   # int32[cap_edges]   (always negative-encoded)
    overlay_fwd_from_main_weights: np.ndarray   # int32[cap_edges]

    # Buffer holding (main_src, ov_dst_negencoded, weight) triples until the next rebuild.
    # We append to it during add_trip; the rebuild_main_to_overlay scatters into CSR.
    overlay_main_src_buf:    np.ndarray   # int32[cap_edges]
    overlay_main_dst_buf:    np.ndarray   # int32[cap_edges]   (negative-encoded)
    overlay_main_weight_buf: np.ndarray   # int32[cap_edges]

    # overlay per-stop D-event index (rebuilt eagerly per batch)
    overlay_stop_dep_indptr: np.ndarray   # int32[n_stops + 1]
    overlay_stop_dep_time:   np.ndarray   # int32[cap_nodes]
    overlay_stop_dep_node:   np.ndarray   # int32[cap_nodes]   (negative-encoded)

    # ---- compaction bookkeeping ----
    edits_applied:        int = 0
    cancels_applied:      int = 0
    delays_applied:       int = 0
    adds_applied:         int = 0
    dead_edge_count:      int = 0
    compactions_run:      int = 0

    # ---- thresholds (DESIGN §4 / §7.5b) ----
    overlay_compact_frac: float = 0.75   # trigger when overlay_n_edges/cap > 0.75
    dead_edge_compact_frac: float = 0.25 # trigger when dead/E > 0.25

    def overlay_should_compact(self) -> bool:
        return (self.overlay_n_edges > self.overlay_capacity_edges * self.overlay_compact_frac
                or self.overlay_n_nodes > self.overlay_capacity_nodes * self.overlay_compact_frac
                or self.dead_edge_count > self.n_edges * self.dead_edge_compact_frac)


# ---------- builders -------------------------------------------------------

def _build_trip_indexes(g):
    """Compute trip_first_node, trip_node_idx, trip_edge_off, trip_edge_idx.

    Trip ids are int32 in node_trip; we treat them as a contiguous range
    [0..T-1] (this matches `build_teeg`'s output on the standard fixtures).
    """
    node_trip = np.asarray(g.node_trip, dtype=np.int64)
    node_time = np.asarray(g.node_time, dtype=np.int32)
    node_kind = np.asarray(g.node_kind, dtype=np.int8)
    n_nodes = int(g.n_nodes)
    n_edges = int(g.n_edges)
    indptr = np.asarray(g.indptr, dtype=np.int32)
    indices = np.asarray(g.indices, dtype=np.int32)
    rev_indptr = np.asarray(g.rev_indptr, dtype=np.int32)
    rev_indices = np.asarray(g.rev_indices, dtype=np.int32)

    if n_nodes == 0:
        empty32 = np.zeros(0, dtype=np.int32)
        return (np.zeros(1, dtype=np.int32), empty32,
                np.zeros(1, dtype=np.int32), empty32, 0)

    t_min = int(node_trip.min())
    t_max = int(node_trip.max())
    T = t_max - t_min + 1

    # Group events by trip, ordered by (trip, time, kind) so that within a trip
    # we get D0, A0, D1, A1, ... in route order.  (D=0, A=1; ties by kind
    # mean a D before an A at the same minute, which is the connection order.)
    nodes = np.arange(n_nodes, dtype=np.int32)
    order = np.lexsort((node_kind, node_time, node_trip))   # primary: trip
    node_idx_sorted = nodes[order]
    trip_sorted = node_trip[order]

    trip_first_node = np.zeros(T + 1, dtype=np.int32)
    np.add.at(trip_first_node, (trip_sorted - t_min) + 1, 1)
    np.cumsum(trip_first_node, out=trip_first_node)
    trip_node_idx = node_idx_sorted.astype(np.int32, copy=False)

    # Now for each trip event collect incident forward-CSR edge positions.
    # OUT-edges of node n: indptr[n]..indptr[n+1]
    # IN-edges of node n in forward CSR: rev_indptr[n]..rev_indptr[n+1] gives
    # *reverse* CSR positions whose source-node is the destination of forward edges
    # ending at n.  We need the FORWARD CSR position of those edges to set
    # edge_alive[fwd_pos] = 0.  We materialise this only for trip-incident
    # edges to keep memory at ~80 MB on Full LDN (DESIGN §6.4.5).
    #
    # Strategy: build an inverse map fwd_pos → rev_pos via a single pass that
    # mirrors the rev-CSR construction, but only for nodes that appear in any
    # trip (here that is *every* event node that has node_trip set, which on
    # TEEG-style graphs is all of them).  Cheaper and simpler: for each node
    # n on a trip, scan rev_indptr[n]..rev_indptr[n+1], get the source u, then
    # binary-search for `n` inside indptr[u]..indptr[u+1].  But the dst within
    # that range is not sorted in general — DESIGN §6.4.5 explicitly rejects
    # binary search.
    #
    # We instead invert the forward CSR fully ONCE: rev_to_fwd[r] = forward
    # position whose endpoint-pair matches the r-th reverse entry.  Memory:
    # 4·E bytes (~10 MB on subset, ~590 MB on Full LDN — too big).  We trim
    # by storing only positions for *trip-incident* nodes, but on TEEG this
    # is essentially every node.  For the AAAI subset (≤ 5 M edges) the
    # full inverse fits comfortably (≤ 20 MB), so we use the full inverse
    # here; the production Full-LDN build can switch to per-trip materialisation.

    # Build full rev_to_fwd by sorting each side's edge tuples lexicographically.
    # The forward CSR ordering is by source; edges within a node are in the
    # order originally emitted (RIDE, STAY, WAIT, BOARD…).  We need a stable
    # mapping rev_pos → fwd_pos such that
    #   forward[fwd_pos] = (u, v, w) and reverse[rev_pos] = (v, u, w).
    # Build it by encoding (u, v) as int64 keys and arg-sorting.
    fwd_src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr).astype(np.int64))
    fwd_dst = indices.astype(np.int64)
    fwd_key = fwd_src * np.int64(n_nodes) + fwd_dst
    fwd_pos = np.arange(n_edges, dtype=np.int32)

    rev_src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(rev_indptr).astype(np.int64))
    rev_dst = rev_indices.astype(np.int64)
    # In reverse CSR, rev_src is the original DST and rev_dst is the original SRC.
    rev_key = rev_dst * np.int64(n_nodes) + rev_src

    # Sort both by key, then map.  Within a key duplicates may exist; pair them
    # by stable order on (key, weight) — but both representations preserve the
    # original emission order, so a stable sort by key alone is enough as long
    # as both sides agree on tie-breaking.  We tie-break by weight as well.
    fwd_w = np.asarray(g.weights, dtype=np.int64)
    rev_w = np.asarray(g.rev_weights, dtype=np.int64)
    fwd_order = np.lexsort((fwd_pos.astype(np.int64), fwd_w, fwd_key))
    rev_order = np.lexsort((np.arange(n_edges, dtype=np.int64), rev_w, rev_key))
    rev_to_fwd = np.empty(n_edges, dtype=np.int32)
    rev_to_fwd[rev_order] = fwd_pos[fwd_order]

    # Now collate trip-incident edges.
    # Total per-event edges: out-deg + in-deg.
    out_deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
    in_deg = (rev_indptr[1:] - rev_indptr[:-1]).astype(np.int64)
    per_node = out_deg + in_deg
    total = int(per_node[trip_node_idx].sum())

    trip_edge_off = np.zeros(T + 1, dtype=np.int32)
    # CSR by trip: cumulate per-trip total edge count
    per_trip = np.zeros(T, dtype=np.int64)
    # Each event in trip_node_idx contributes per_node[node] edges
    np.add.at(per_trip, (node_trip[trip_node_idx] - t_min), per_node[trip_node_idx])
    trip_edge_off[1:] = np.cumsum(per_trip).astype(np.int32)

    trip_edge_idx = np.empty(total, dtype=np.int32)
    write = 0
    # iterate trips
    for t in range(T):
        lo = int(trip_first_node[t]); hi = int(trip_first_node[t + 1])
        for k in range(lo, hi):
            n = int(trip_node_idx[k])
            elo = int(indptr[n]); ehi = int(indptr[n + 1])
            ln = ehi - elo
            if ln:
                trip_edge_idx[write:write + ln] = np.arange(elo, ehi, dtype=np.int32)
                write += ln
            rlo = int(rev_indptr[n]); rhi = int(rev_indptr[n + 1])
            ln = rhi - rlo
            if ln:
                trip_edge_idx[write:write + ln] = rev_to_fwd[rlo:rhi]
                write += ln
    assert write == total
    return trip_first_node, trip_node_idx, trip_edge_off, trip_edge_idx, t_min


def build_state(g, *, overlay_capacity_nodes: int | None = None,
                overlay_capacity_edges: int | None = None) -> IncrementalState:
    """Construct an IncrementalState for the static TEEG `g`."""
    n_nodes = int(g.n_nodes)
    n_edges = int(g.n_edges)
    n_stops = int(g.n_stops)

    # Default overlay capacity: 5% of V on subset (DESIGN §2.2 says 5–10%);
    # the 50K-stress test will set this much smaller (e.g. 0.5%) to force
    # compaction firing per §7.5b.
    if overlay_capacity_nodes is None:
        overlay_capacity_nodes = max(1024, int(0.05 * n_nodes))
    if overlay_capacity_edges is None:
        overlay_capacity_edges = max(8192, 8 * overlay_capacity_nodes)

    node_alive = np.ones(n_nodes, dtype=np.uint8)
    edge_alive = np.ones(n_edges, dtype=np.uint8)

    tfn, tni, teo, tei, _t_min = _build_trip_indexes(g)

    cap_n = int(overlay_capacity_nodes)
    cap_e = int(overlay_capacity_edges)

    overlay_node_kind  = np.zeros(cap_n, dtype=np.int8)
    overlay_node_stop  = np.zeros(cap_n, dtype=np.int32)
    overlay_node_time  = np.zeros(cap_n, dtype=np.int32)
    overlay_node_trip  = np.zeros(cap_n, dtype=np.int32)
    overlay_node_route = np.zeros(cap_n, dtype=np.int32)
    overlay_node_alive = np.zeros(cap_n, dtype=np.uint8)

    overlay_indptr  = np.zeros(cap_n + 1, dtype=np.int32)
    overlay_indices = np.zeros(cap_e, dtype=np.int32)
    overlay_weights = np.zeros(cap_e, dtype=np.int32)

    # main→overlay buffer (we append (src, dst, w) tuples)
    overlay_main_src_buf    = np.zeros(cap_e, dtype=np.int32)
    overlay_main_dst_buf    = np.zeros(cap_e, dtype=np.int32)
    overlay_main_weight_buf = np.zeros(cap_e, dtype=np.int32)

    overlay_fwd_from_main_indptr  = np.zeros(n_nodes + 1, dtype=np.int32)
    overlay_fwd_from_main_indices = np.zeros(cap_e, dtype=np.int32)
    overlay_fwd_from_main_weights = np.zeros(cap_e, dtype=np.int32)

    overlay_stop_dep_indptr = np.zeros(n_stops + 1, dtype=np.int32)
    overlay_stop_dep_time   = np.zeros(cap_n, dtype=np.int32)
    overlay_stop_dep_node   = np.zeros(cap_n, dtype=np.int32)

    return IncrementalState(
        n_nodes=n_nodes, n_edges=n_edges, n_stops=n_stops,
        node_alive=node_alive, edge_alive=edge_alive,
        trip_first_node=tfn, trip_node_idx=tni,
        trip_edge_off=teo, trip_edge_idx=tei,
        overlay_capacity_nodes=cap_n,
        overlay_capacity_edges=cap_e,
        overlay_n_nodes=0, overlay_n_edges=0, overlay_n_main_fwd_edges=0,
        overlay_node_kind=overlay_node_kind,
        overlay_node_stop=overlay_node_stop,
        overlay_node_time=overlay_node_time,
        overlay_node_trip=overlay_node_trip,
        overlay_node_route=overlay_node_route,
        overlay_node_alive=overlay_node_alive,
        overlay_indptr=overlay_indptr,
        overlay_indices=overlay_indices,
        overlay_weights=overlay_weights,
        overlay_fwd_from_main_indptr=overlay_fwd_from_main_indptr,
        overlay_fwd_from_main_indices=overlay_fwd_from_main_indices,
        overlay_fwd_from_main_weights=overlay_fwd_from_main_weights,
        overlay_main_src_buf=overlay_main_src_buf,
        overlay_main_dst_buf=overlay_main_dst_buf,
        overlay_main_weight_buf=overlay_main_weight_buf,
        overlay_stop_dep_indptr=overlay_stop_dep_indptr,
        overlay_stop_dep_time=overlay_stop_dep_time,
        overlay_stop_dep_node=overlay_stop_dep_node,
    )
