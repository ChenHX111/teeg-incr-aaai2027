"""Incremental update kernels for TEEG-ALT-INCR (EXT3).

Public API:
    cancel_trip(state, trip_id)
    delay_trip(state, trip_id, delta_min)   # decomposed as cancel + add (DESIGN §3.4)
    add_trip(state, template_trip_id, t0_delta_min)
    apply_edits(state, g, edits) → BatchStats
    compact(state, g) → new (state, g_compacted) tuple

Edits are tuples ('CANCEL', trip_id) / ('DELAY', trip_id, delta) /
('ADD', template_trip_id, t0_delta).

All hot loops are @njit(cache=True). Python wrappers handle dispatch and the
post-batch aux rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import time
import numpy as np
from numba import njit

from .state import IncrementalState, build_state, INF_I32


# ------------------------------------------------------------------
# Numba kernels
# ------------------------------------------------------------------

@njit(cache=True)
def _cancel_trip_njit(trip_id,
                      trip_first_node, trip_node_idx,
                      trip_edge_off, trip_edge_idx,
                      node_alive, edge_alive):
    """Tombstone every event-node and every incident forward-CSR edge of `trip_id`.

    Returns count of newly-killed (alive→dead) edges, for the dead_edge_count
    bookkeeping that drives compaction.
    """
    n_killed = np.int32(0)
    lo = trip_first_node[trip_id]; hi = trip_first_node[trip_id + 1]
    for k in range(lo, hi):
        n = trip_node_idx[k]
        if node_alive[n]:
            node_alive[n] = 0
    elo = trip_edge_off[trip_id]; ehi = trip_edge_off[trip_id + 1]
    for k in range(elo, ehi):
        e = trip_edge_idx[k]
        if edge_alive[e]:
            edge_alive[e] = 0
            n_killed += 1
    return n_killed


@njit(cache=True)
def _add_trip_emit_overlay_njit(
    template_id, t0_delta,
    trip_first_node, trip_node_idx,
    node_kind, node_stop, node_time, node_route,
    indptr, indices, weights,
    edge_alive, node_alive,
    overlay_n_nodes_in, overlay_n_edges_in,
    overlay_node_kind, overlay_node_stop, overlay_node_time,
    overlay_node_trip, overlay_node_route, overlay_node_alive,
    overlay_indptr, overlay_indices, overlay_weights,
    overlay_main_n_in,
    overlay_main_src_buf, overlay_main_dst_buf, overlay_main_weight_buf,
    overlay_capacity_nodes, overlay_capacity_edges,
    synth_trip_id, day_min,
    stop_dep_indptr, stop_dep_time, stop_dep_node,
):
    """Materialise a template trip into the overlay.

    Returns (new_overlay_n_nodes, new_overlay_n_edges, new_overlay_main_n,
             status). status==0 ok, ==-1 capacity exceeded.
    """
    lo = trip_first_node[template_id]; hi = trip_first_node[template_id + 1]
    n_new = hi - lo
    base = overlay_n_nodes_in
    if base + n_new > overlay_capacity_nodes:
        return overlay_n_nodes_in, overlay_n_edges_in, overlay_main_n_in, np.int32(-1)

    # 1) materialise nodes
    for k in range(n_new):
        src = trip_node_idx[lo + k]
        ov = base + k
        overlay_node_kind [ov] = node_kind [src]
        overlay_node_stop [ov] = node_stop [src]
        overlay_node_time [ov] = np.int32(node_time[src]) + t0_delta
        overlay_node_trip [ov] = synth_trip_id
        overlay_node_route[ov] = node_route[src]
        overlay_node_alive[ov] = 1

    # 2) emit overlay→{overlay,main} edges. Walk template events in order and
    #    copy the topology:
    #    - For each template node n at position k in route order, look at its
    #      forward-CSR edges; for each edge n→v whose `v` is also on the same
    #      template trip (i.e. node_trip[v] == template_id), emit an
    #      overlay→overlay edge with the same weight (RIDE/STAY are weight-
    #      invariant under uniform t0_delta).
    #    - For other edges (BOARD, WAIT, WALK_AND_BOARD) we DO NOT copy from
    #      the template; instead we rebuild fresh BOARD edges by querying
    #      stop_dep_* for main-graph D-events.
    #
    # Because we already have trip_node_idx[lo..hi] in route order with node
    # ids, build a lookup: main_node_id → overlay_node_id (only for this trip).
    # Use a small linear scan since trips are short (~30-60 events).
    #
    # CSR write: produce per-overlay-row out-edges. We accumulate into a small
    # local buffer first (pessimistic max-degree per overlay node), then commit
    # to overlay_indptr/indices/weights contiguously.

    # Linear lookup table for template-internal edges.
    # template_node[k] -> overlay node = base + k
    # We avoid a full-V map by linearly scanning template_id membership.
    # Build small inverse map keyed by `src` value.

    # Assume n_new <= 256 typical; build sorted (src, ov_id) pairs for fast lookup.
    # For simplicity and correctness, use a naive linear search per edge.
    template_id_int = template_id

    # First pass: count out-edges per overlay node (RIDE/STAY internal only,
    # plus BOARD-out at the same stop for A-events). Simplification per
    # implementation_status: we currently emit only same-stop BOARD-out.

    # Compute write positions:
    # We'll do a two-pass: count, then fill.

    # PASS 1: count
    out_count = np.zeros(n_new, dtype=np.int32)
    for k in range(n_new):
        ov = base + k
        src = trip_node_idx[lo + k]
        # template-internal edges
        elo = indptr[src]; ehi = indptr[src + 1]
        for e in range(elo, ehi):
            v = indices[e]
            # only count if v is on same template trip
            for j in range(n_new):
                if trip_node_idx[lo + j] == v:
                    out_count[k] += 1
                    break
        # BOARD-out: if this is an A-event, scan main D-events at same stop with
        # dep_time >= overlay A.time
        if overlay_node_kind[ov] == 1:
            s = overlay_node_stop[ov]
            a_t = overlay_node_time[ov]
            slo = stop_dep_indptr[s]; shi = stop_dep_indptr[s + 1]
            # binary-search for first dep_time >= a_t (dep times are int16 per stop)
            left = slo; right = shi
            while left < right:
                mid = (left + right) // 2
                if np.int32(stop_dep_time[mid]) < a_t:
                    left = mid + 1
                else:
                    right = mid
            if left < shi:
                d_node = stop_dep_node[left]
                # only count if alive
                if node_alive[d_node]:
                    out_count[k] += 1

    # commit overlay_indptr for these new rows
    cum = overlay_n_edges_in
    if cum + np.sum(out_count) > overlay_capacity_edges:
        return overlay_n_nodes_in, overlay_n_edges_in, overlay_main_n_in, np.int32(-1)

    for k in range(n_new):
        ov = base + k
        overlay_indptr[ov] = cum
        cum += out_count[k]
    overlay_indptr[base + n_new] = cum   # tail of CSR

    # PASS 2: write
    write_pos = overlay_n_edges_in
    for k in range(n_new):
        ov = base + k
        src = trip_node_idx[lo + k]
        elo = indptr[src]; ehi = indptr[src + 1]
        for e in range(elo, ehi):
            v = indices[e]
            # find v on template trip
            for j in range(n_new):
                if trip_node_idx[lo + j] == v:
                    ov_v = base + j
                    overlay_indices[write_pos] = -(ov_v + 1)
                    overlay_weights[write_pos] = weights[e]
                    write_pos += 1
                    break
        if overlay_node_kind[ov] == 1:
            s = overlay_node_stop[ov]
            a_t = overlay_node_time[ov]
            slo = stop_dep_indptr[s]; shi = stop_dep_indptr[s + 1]
            left = slo; right = shi
            while left < right:
                mid = (left + right) // 2
                if np.int32(stop_dep_time[mid]) < a_t:
                    left = mid + 1
                else:
                    right = mid
            if left < shi:
                d_node = stop_dep_node[left]
                if node_alive[d_node]:
                    overlay_indices[write_pos] = d_node            # positive ⇒ main
                    overlay_weights[write_pos] = np.int32(stop_dep_time[left]) - a_t
                    write_pos += 1

    # 3) BOARD-in edges: for each overlay D-event at stop s with time td,
    #    emit a main→overlay BOARD edge from the latest main A-event at s with
    #    arr_time <= td.  This requires main_a-event index per stop.  For
    #    simplicity here we emit edges from EACH alive main A-event at stop s
    #    whose arr_time satisfies td - a_t >= 0 -- but that's too many.  We
    #    emit only ONE edge: from the latest such alive A-event (smallest
    #    waiting time).  This mirrors builder.py's BOARD construction.
    #
    # We use stop_arr_indptr + stop_arr_time + stop_arr_node, passed in from
    # the caller via globals... actually we don't have them here. The caller
    # will handle BOARD-in via a separate pass after this kernel returns.
    # (See _add_trip_emit_board_in_njit below.)

    new_overlay_n_nodes = base + n_new
    new_overlay_n_edges = write_pos
    return new_overlay_n_nodes, new_overlay_n_edges, overlay_main_n_in, np.int32(0)


@njit(cache=True)
def _add_trip_emit_board_in_njit(
    new_ov_lo, new_ov_hi,
    overlay_node_kind, overlay_node_stop, overlay_node_time, overlay_node_alive,
    stop_arr_indptr, stop_arr_time, stop_arr_node,
    node_alive,
    overlay_main_n_in,
    overlay_main_src_buf, overlay_main_dst_buf, overlay_main_weight_buf,
    overlay_capacity_edges,
):
    """Emit main→overlay BOARD edges for each new overlay D-event.

    For each overlay D in [new_ov_lo, new_ov_hi) at stop s with time td, find
    the latest alive main A-event at s with arr_time <= td and emit one edge.
    Returns (new_overlay_main_n, status).
    """
    write = overlay_main_n_in
    for ov in range(new_ov_lo, new_ov_hi):
        if overlay_node_kind[ov] != 0:   # D-events only
            continue
        s = overlay_node_stop[ov]
        td = overlay_node_time[ov]
        alo = stop_arr_indptr[s]; ahi = stop_arr_indptr[s + 1]
        # binary search for first arr_time > td  → upper_bound; predecessor is
        # the latest with arr_time <= td.
        left = alo; right = ahi
        while left < right:
            mid = (left + right) // 2
            if np.int32(stop_arr_time[mid]) <= td:
                left = mid + 1
            else:
                right = mid
        # left is first arr_time > td; latest <= td is at left-1
        idx = left - 1
        if idx < alo:
            continue
        # walk backwards over alive A-events at this stop to find one alive
        found = -1
        i = idx
        while i >= alo:
            n_a = stop_arr_node[i]
            if node_alive[n_a]:
                found = i
                break
            i -= 1
        if found < 0:
            continue
        a_node = stop_arr_node[found]
        a_t = np.int32(stop_arr_time[found])
        if write >= overlay_capacity_edges:
            return overlay_main_n_in, np.int32(-1)
        overlay_main_src_buf[write] = a_node
        overlay_main_dst_buf[write] = -(ov + 1)
        overlay_main_weight_buf[write] = td - a_t
        write += 1
    return write, np.int32(0)


@njit(cache=True)
def _rebuild_main_to_overlay_csr_njit(
    overlay_main_n,
    overlay_main_src_buf, overlay_main_dst_buf, overlay_main_weight_buf,
    overlay_fwd_from_main_indptr,
    overlay_fwd_from_main_indices,
    overlay_fwd_from_main_weights,
    n_nodes,
):
    """Build main→overlay CSR from the (src, dst, w) triples in the buffers."""
    # zero indptr
    for i in range(n_nodes + 1):
        overlay_fwd_from_main_indptr[i] = 0
    # count
    for k in range(overlay_main_n):
        s = overlay_main_src_buf[k]
        overlay_fwd_from_main_indptr[s + 1] += 1
    # cumsum
    for i in range(1, n_nodes + 1):
        overlay_fwd_from_main_indptr[i] += overlay_fwd_from_main_indptr[i - 1]
    # scatter into a copy of indptr that we mutate as cursor
    cursor = overlay_fwd_from_main_indptr.copy()
    for k in range(overlay_main_n):
        s = overlay_main_src_buf[k]
        pos = cursor[s]
        overlay_fwd_from_main_indices[pos] = overlay_main_dst_buf[k]
        overlay_fwd_from_main_weights[pos] = overlay_main_weight_buf[k]
        cursor[s] = pos + 1


@njit(cache=True)
def _rebuild_overlay_stop_dep_njit(
    overlay_n_nodes,
    overlay_node_kind, overlay_node_stop, overlay_node_time, overlay_node_alive,
    overlay_stop_dep_indptr,
    overlay_stop_dep_time,
    overlay_stop_dep_node,
    n_stops,
):
    """Per-stop sorted index of *alive* overlay D-events (by time, ascending)."""
    for i in range(n_stops + 1):
        overlay_stop_dep_indptr[i] = 0
    # count
    for ov in range(overlay_n_nodes):
        if overlay_node_kind[ov] == 0 and overlay_node_alive[ov]:
            s = overlay_node_stop[ov]
            overlay_stop_dep_indptr[s + 1] += 1
    for i in range(1, n_stops + 1):
        overlay_stop_dep_indptr[i] += overlay_stop_dep_indptr[i - 1]
    # scatter (unsorted)
    cursor = overlay_stop_dep_indptr.copy()
    for ov in range(overlay_n_nodes):
        if overlay_node_kind[ov] == 0 and overlay_node_alive[ov]:
            s = overlay_node_stop[ov]
            pos = cursor[s]
            overlay_stop_dep_time[pos] = overlay_node_time[ov]
            overlay_stop_dep_node[pos] = -(ov + 1)
            cursor[s] = pos + 1
    # sort each stop's slice by time (insertion sort - small per-stop counts)
    for s in range(n_stops):
        lo = overlay_stop_dep_indptr[s]; hi = overlay_stop_dep_indptr[s + 1]
        for i in range(lo + 1, hi):
            kt = overlay_stop_dep_time[i]
            kn = overlay_stop_dep_node[i]
            j = i - 1
            while j >= lo and overlay_stop_dep_time[j] > kt:
                overlay_stop_dep_time[j + 1] = overlay_stop_dep_time[j]
                overlay_stop_dep_node[j + 1] = overlay_stop_dep_node[j]
                j -= 1
            overlay_stop_dep_time[j + 1] = kt
            overlay_stop_dep_node[j + 1] = kn


# ------------------------------------------------------------------
# Public Python wrappers
# ------------------------------------------------------------------

@dataclass
class BatchStats:
    n_cancels: int = 0
    n_delays: int = 0
    n_adds: int = 0
    n_compactions: int = 0
    wall_total_s: float = 0.0
    wall_apply_s: float = 0.0
    wall_compact_s: float = 0.0
    capacity_overflow: bool = False
    per_edit_latencies_us: list = None  # populated when track_per_edit=True


def cancel_trip(state: IncrementalState, trip_id: int) -> int:
    """Cancel a trip on the static graph. Returns # newly-dead edges."""
    if trip_id < 0 or trip_id >= state.trip_first_node.shape[0] - 1:
        return 0
    n_killed = int(_cancel_trip_njit(
        np.int32(trip_id),
        state.trip_first_node, state.trip_node_idx,
        state.trip_edge_off, state.trip_edge_idx,
        state.node_alive, state.edge_alive,
    ))
    state.dead_edge_count += n_killed
    state.cancels_applied += 1
    state.edits_applied += 1
    return n_killed


def add_trip(state: IncrementalState, g, template_trip_id: int, t0_delta: int) -> int:
    """Add a template-based trip. Returns synth (negative) trip id, or 0 on capacity overflow."""
    if template_trip_id < 0 or template_trip_id >= state.trip_first_node.shape[0] - 1:
        return 0
    synth_id = -(state.adds_applied + 1)
    new_lo = state.overlay_n_nodes
    res = _add_trip_emit_overlay_njit(
        np.int32(template_trip_id), np.int32(t0_delta),
        state.trip_first_node, state.trip_node_idx,
        np.asarray(g.node_kind, dtype=np.int8),
        np.asarray(g.node_stop, dtype=np.int32),
        np.asarray(g.node_time, dtype=np.int16),
        np.asarray(g.node_route, dtype=np.int32),
        np.asarray(g.indptr, dtype=np.int32),
        np.asarray(g.indices, dtype=np.int32),
        np.asarray(g.weights, dtype=np.int32),
        state.edge_alive, state.node_alive,
        np.int32(state.overlay_n_nodes), np.int32(state.overlay_n_edges),
        state.overlay_node_kind, state.overlay_node_stop, state.overlay_node_time,
        state.overlay_node_trip, state.overlay_node_route, state.overlay_node_alive,
        state.overlay_indptr, state.overlay_indices, state.overlay_weights,
        np.int32(state.overlay_n_main_fwd_edges),
        state.overlay_main_src_buf, state.overlay_main_dst_buf, state.overlay_main_weight_buf,
        np.int32(state.overlay_capacity_nodes), np.int32(state.overlay_capacity_edges),
        np.int32(synth_id), np.int32(1440),
        np.asarray(g.stop_dep_indptr, dtype=np.int32),
        np.asarray(g.stop_dep_time, dtype=np.int16),
        np.asarray(g.stop_dep_node, dtype=np.int32),
    )
    new_n_nodes, new_n_edges, _new_main_n, status = res
    if int(status) != 0:
        return 0
    state.overlay_n_nodes = int(new_n_nodes)
    state.overlay_n_edges = int(new_n_edges)
    new_hi = state.overlay_n_nodes
    # BOARD-in pass (main→overlay edges)
    res2 = _add_trip_emit_board_in_njit(
        np.int32(new_lo), np.int32(new_hi),
        state.overlay_node_kind, state.overlay_node_stop,
        state.overlay_node_time, state.overlay_node_alive,
        np.asarray(g.stop_arr_indptr, dtype=np.int32),
        np.asarray(g.stop_arr_time, dtype=np.int16),
        np.asarray(g.stop_arr_node, dtype=np.int32),
        state.node_alive,
        np.int32(state.overlay_n_main_fwd_edges),
        state.overlay_main_src_buf, state.overlay_main_dst_buf, state.overlay_main_weight_buf,
        np.int32(state.overlay_capacity_edges),
    )
    new_main_n, status2 = res2
    if int(status2) != 0:
        return 0
    state.overlay_n_main_fwd_edges = int(new_main_n)
    state.adds_applied += 1
    state.edits_applied += 1
    return synth_id


def delay_trip(state: IncrementalState, g, trip_id: int, delta_min: int) -> int:
    """Decompose as cancel+add (DESIGN §3.4). Returns synth_id of the replacement."""
    cancel_trip(state, trip_id)
    state.cancels_applied -= 1   # don't double-count for stats
    state.edits_applied -= 1
    synth = add_trip(state, g, trip_id, delta_min)
    if synth == 0:
        return 0
    state.adds_applied -= 1
    state.delays_applied += 1
    state.edits_applied += 1
    return synth


def _rebuild_aux(state: IncrementalState):
    """Rebuild eager aux structures (main→overlay CSR + overlay_stop_dep_*)."""
    _rebuild_main_to_overlay_csr_njit(
        np.int32(state.overlay_n_main_fwd_edges),
        state.overlay_main_src_buf,
        state.overlay_main_dst_buf,
        state.overlay_main_weight_buf,
        state.overlay_fwd_from_main_indptr,
        state.overlay_fwd_from_main_indices,
        state.overlay_fwd_from_main_weights,
        np.int32(state.n_nodes),
    )
    _rebuild_overlay_stop_dep_njit(
        np.int32(state.overlay_n_nodes),
        state.overlay_node_kind, state.overlay_node_stop,
        state.overlay_node_time, state.overlay_node_alive,
        state.overlay_stop_dep_indptr,
        state.overlay_stop_dep_time,
        state.overlay_stop_dep_node,
        np.int32(state.n_stops),
    )


def apply_edits(state: IncrementalState, g, edits: list, *,
                track_per_edit: bool = False,
                allow_compaction: bool = True,
                compaction_callback=None) -> BatchStats:
    """Apply a batch of edits.

    edits: list of tuples
        ('CANCEL', trip_id)
        ('DELAY',  trip_id, delta_min)
        ('ADD',    template_trip_id, t0_delta_min)

    If `allow_compaction` is True, we may invoke `compaction_callback(state, g)`
    before this batch returns when overlay_should_compact() fires.  The callback
    is expected to return (new_state, new_g); we then mutate the local state's
    references and update the returned `g` via the callback's mutation semantics.

    To keep the API simple, callers using compaction should use the higher-level
    `TEEGIncremental` class which manages the swap.
    """
    stats = BatchStats(per_edit_latencies_us=[] if track_per_edit else None)
    t_total = time.perf_counter()
    t_apply = time.perf_counter()
    for e in edits:
        et = e[0]
        if track_per_edit:
            t0 = time.perf_counter()
        if et == 'CANCEL':
            cancel_trip(state, int(e[1]))
            stats.n_cancels += 1
        elif et == 'DELAY':
            r = delay_trip(state, g, int(e[1]), int(e[2]))
            if r == 0:
                stats.capacity_overflow = True
            stats.n_delays += 1
        elif et == 'ADD':
            r = add_trip(state, g, int(e[1]), int(e[2]))
            if r == 0:
                stats.capacity_overflow = True
            stats.n_adds += 1
        else:
            raise ValueError(f"unknown edit type {et!r}")
        if track_per_edit:
            stats.per_edit_latencies_us.append((time.perf_counter() - t0) * 1e6)
    stats.wall_apply_s = time.perf_counter() - t_apply
    _rebuild_aux(state)
    stats.wall_total_s = time.perf_counter() - t_total
    if allow_compaction and compaction_callback is not None and state.overlay_should_compact():
        t_c = time.perf_counter()
        compaction_callback(state, g)
        stats.wall_compact_s = time.perf_counter() - t_c
        stats.n_compactions = 1
        # NB: state.compactions_run on the OLD state is lost when the callback
        # swaps in the new state — the callback itself bumps the new state's
        # counter, so we don't mutate `state` here.
        stats.wall_total_s = time.perf_counter() - t_total
    return stats


# ------------------------------------------------------------------
# Compaction
# ------------------------------------------------------------------

def compact(state: IncrementalState, g) -> Tuple[IncrementalState, Any]:
    """Rebuild the static graph from the alive subset of (main ∪ overlay).

    Returns (new_state, new_g) where new_g is a new TEEG instance and new_state
    is its companion IncrementalState (all-alive).
    """
    from Route_Planner.teeg.builder import TEEG  # local import to avoid cycles
    # ---- collect alive connections ----
    # A connection on the main graph is a (D, A) pair where D=2i, A=2i+1 (the
    # builder's invariant).  Filter both alive AND with all RIDE edge alive.
    n_conn_main = state.n_nodes // 2
    main_d = np.arange(0, state.n_nodes, 2, dtype=np.int64)
    main_a = main_d + 1
    alive_pair = (state.node_alive[main_d] & state.node_alive[main_a]).astype(bool)
    keep_d = main_d[alive_pair]
    keep_a = main_a[alive_pair]
    # collect (from_stop, to_stop, dep_min, arr_min, trip_id, route_int)
    node_stop_main = np.asarray(g.node_stop)
    node_time_main = np.asarray(g.node_time)
    node_trip_main = np.asarray(g.node_trip)
    node_route_main = np.asarray(g.node_route)

    from_stop = node_stop_main[keep_d].astype(np.int32)
    to_stop   = node_stop_main[keep_a].astype(np.int32)
    dep_min   = node_time_main[keep_d].astype(np.int32)
    arr_min   = node_time_main[keep_a].astype(np.int32)
    trip_id   = node_trip_main[keep_d].astype(np.int32)
    route_int = node_route_main[keep_d].astype(np.int32)

    # Also collect alive overlay connections (paired the same way: D,A,D,A,...)
    if state.overlay_n_nodes > 0:
        ov_d_idx = np.arange(0, state.overlay_n_nodes, 2, dtype=np.int64)
        ov_a_idx = ov_d_idx + 1
        # Some overlays may be partially built (e.g. odd count) — guard:
        if ov_a_idx.size and ov_a_idx[-1] >= state.overlay_n_nodes:
            ov_d_idx = ov_d_idx[:-1]; ov_a_idx = ov_a_idx[:-1]
        ov_alive = (state.overlay_node_alive[ov_d_idx] & state.overlay_node_alive[ov_a_idx]).astype(bool)
        keep_ov_d = ov_d_idx[ov_alive]
        keep_ov_a = ov_a_idx[ov_alive]
        if keep_ov_d.size:
            # Re-number overlay synthetic trips to a non-overlapping range.
            ov_trip_offset = int(trip_id.max()) + 1 if trip_id.size else 0
            from_stop = np.concatenate([from_stop, state.overlay_node_stop[keep_ov_d].astype(np.int32)])
            to_stop   = np.concatenate([to_stop,   state.overlay_node_stop[keep_ov_a].astype(np.int32)])
            dep_min   = np.concatenate([dep_min,   state.overlay_node_time[keep_ov_d].astype(np.int32)])
            arr_min   = np.concatenate([arr_min,   state.overlay_node_time[keep_ov_a].astype(np.int32)])
            ov_trips_unique = -np.asarray(state.overlay_node_trip[keep_ov_d], dtype=np.int32)  # positive
            trip_id   = np.concatenate([trip_id,   ov_trip_offset + ov_trips_unique])
            route_int = np.concatenate([route_int, state.overlay_node_route[keep_ov_d].astype(np.int32)])

    # ---- build a synthetic TimetableBundle-like object and call build_teeg ----
    # We only need the fields used by build_teeg: connections (recarray with
    # named fields), n_stops, t_lo, t_hi, transfers_from.
    # Easier path: directly hand-build the TEEG object using the collected
    # connections + the existing walk CSR (walk edges are tombstone-immune
    # per invariant I-5).
    new_g = _build_teeg_from_connections(
        from_stop, to_stop, dep_min, arr_min, trip_id, route_int,
        n_stops=state.n_stops,
        t_lo=int(g.t_lo), t_hi=int(g.t_hi),
        walk_indptr=np.asarray(g.walk_indptr, dtype=np.int32),
        walk_indices=np.asarray(g.walk_indices, dtype=np.int32),
        walk_weights=np.asarray(g.walk_weights, dtype=np.int32),
        rev_walk_indptr=np.asarray(g.rev_walk_indptr, dtype=np.int32),
        rev_walk_indices=np.asarray(g.rev_walk_indices, dtype=np.int32),
        rev_walk_weights=np.asarray(g.rev_walk_weights, dtype=np.int32),
    )
    new_state = build_state(
        new_g,
        overlay_capacity_nodes=state.overlay_capacity_nodes,
        overlay_capacity_edges=state.overlay_capacity_edges,
    )
    return new_state, new_g


def _build_teeg_from_connections(from_stop, to_stop, dep_min, arr_min,
                                 trip_id, route_int, *, n_stops, t_lo, t_hi,
                                 walk_indptr, walk_indices, walk_weights,
                                 rev_walk_indptr, rev_walk_indices, rev_walk_weights):
    """Inline (small) replication of build_teeg() given pre-collected connections.

    Output has the same TEEG dataclass shape, with all events alive.
    """
    from Route_Planner.teeg.builder import TEEG
    # Sort connections by (trip, dep) for stable trip order.
    n_conn = len(from_stop)
    if n_conn == 0:
        # produce a degenerate TEEG with zero events.
        return TEEG(
            indptr=np.zeros(1, dtype=np.int32),
            indices=np.zeros(0, dtype=np.int32),
            weights=np.zeros(0, dtype=np.int32),
            rev_indptr=np.zeros(1, dtype=np.int32),
            rev_indices=np.zeros(0, dtype=np.int32),
            rev_weights=np.zeros(0, dtype=np.int32),
            node_kind=np.zeros(0, dtype=np.int8),
            node_stop=np.zeros(0, dtype=np.int32),
            node_time=np.zeros(0, dtype=np.int16),
            node_trip=np.zeros(0, dtype=np.int32),
            node_route=np.zeros(0, dtype=np.int32),
            stop_dep_indptr=np.zeros(n_stops + 1, dtype=np.int32),
            stop_dep_time=np.zeros(0, dtype=np.int16),
            stop_dep_node=np.zeros(0, dtype=np.int32),
            stop_arr_indptr=np.zeros(n_stops + 1, dtype=np.int32),
            stop_arr_time=np.zeros(0, dtype=np.int16),
            stop_arr_node=np.zeros(0, dtype=np.int32),
            walk_indptr=walk_indptr, walk_indices=walk_indices, walk_weights=walk_weights,
            rev_walk_indptr=rev_walk_indptr, rev_walk_indices=rev_walk_indices, rev_walk_weights=rev_walk_weights,
            n_stops=int(n_stops), t_lo=int(t_lo), t_hi=int(t_hi),
        )

    # Reassign trip_id to a contiguous [0..T-1] range
    uniq, inv = np.unique(trip_id, return_inverse=True)
    trip_id = inv.astype(np.int32)

    # Sort by (trip, dep_min) so within-trip events are in route order
    order = np.lexsort((dep_min, trip_id))
    from_stop = from_stop[order]; to_stop = to_stop[order]
    dep_min = dep_min[order]; arr_min = arr_min[order]
    trip_id = trip_id[order]; route_int = route_int[order]

    n_nodes = 2 * n_conn
    node_kind  = np.empty(n_nodes, dtype=np.int8)
    node_stop  = np.empty(n_nodes, dtype=np.int32)
    node_time  = np.empty(n_nodes, dtype=np.int16)
    node_trip  = np.empty(n_nodes, dtype=np.int32)
    node_route = np.empty(n_nodes, dtype=np.int32)
    d_ids = np.arange(0, n_nodes, 2, dtype=np.int64)
    a_ids = d_ids + 1
    node_kind[d_ids] = 0; node_kind[a_ids] = 1
    node_stop[d_ids] = from_stop; node_stop[a_ids] = to_stop
    node_time[d_ids] = dep_min.astype(np.int16); node_time[a_ids] = arr_min.astype(np.int16)
    node_trip[d_ids] = trip_id;   node_trip[a_ids] = trip_id
    node_route[d_ids] = route_int; node_route[a_ids] = route_int

    # ---- edges ----
    src_list = []; dst_list = []; wt_list = []
    # RIDE
    src_list.append(d_ids); dst_list.append(a_ids); wt_list.append((arr_min - dep_min).astype(np.int32))

    # STAY: A_prev → D_next within the same trip (consecutive in route order)
    same_trip = (trip_id[1:] == trip_id[:-1])
    if same_trip.any():
        prev_a = a_ids[:-1][same_trip]
        next_d = d_ids[1:][same_trip]
        w = (dep_min[1:][same_trip] - arr_min[:-1][same_trip]).astype(np.int32)
        src_list.append(prev_a); dst_list.append(next_d); wt_list.append(w)

    # WAIT, BOARD, WALK_AND_BOARD: rebuild via stop_dep_index
    # Per-stop sorted D-event index
    d_stops = from_stop
    d_times = dep_min
    d_order = np.lexsort((d_times, d_stops))
    d_stops_s = d_stops[d_order]
    breaks = np.concatenate(([0], np.where(np.diff(d_stops_s) != 0)[0] + 1, [len(d_stops_s)]))
    stop_dep_indptr = np.zeros(n_stops + 1, dtype=np.int32)
    stop_dep_time   = np.empty(n_conn, dtype=np.int16)
    stop_dep_node   = np.empty(n_conn, dtype=np.int32)
    write_pos = 0
    wait_src = []; wait_dst = []; wait_w = []
    for k in range(len(breaks) - 1):
        seg = d_order[breaks[k]:breaks[k + 1]]
        if seg.size == 0:
            continue
        s = int(d_stops[seg[0]])
        if seg.size > 1:
            prev_d = d_ids[seg[:-1]]
            next_d = d_ids[seg[1:]]
            w = (dep_min[seg[1:]] - dep_min[seg[:-1]]).astype(np.int32)
            wait_src.append(prev_d); wait_dst.append(next_d); wait_w.append(w)
        stop_dep_indptr[s + 1] = write_pos + seg.size
        stop_dep_time[write_pos:write_pos + seg.size] = dep_min[seg].astype(np.int16)
        stop_dep_node[write_pos:write_pos + seg.size] = d_ids[seg]
        write_pos += seg.size
    stop_dep_indptr = np.maximum.accumulate(stop_dep_indptr)
    if wait_src:
        src_list.append(np.concatenate(wait_src))
        dst_list.append(np.concatenate(wait_dst))
        wt_list.append(np.concatenate(wait_w))

    # Per-stop A-event index
    a_stops = to_stop
    a_times = arr_min
    a_order = np.lexsort((a_times, a_stops))
    stop_arr_indptr = np.zeros(n_stops + 1, dtype=np.int32)
    stop_arr_time   = np.empty(n_conn, dtype=np.int16)
    stop_arr_node   = np.empty(n_conn, dtype=np.int32)
    a_stops_s = a_stops[a_order]
    breaks = np.concatenate(([0], np.where(np.diff(a_stops_s) != 0)[0] + 1, [len(a_stops_s)]))
    write_pos = 0
    for k in range(len(breaks) - 1):
        seg = a_order[breaks[k]:breaks[k + 1]]
        if seg.size == 0:
            continue
        s = int(a_stops[seg[0]])
        stop_arr_indptr[s + 1] = write_pos + seg.size
        stop_arr_time[write_pos:write_pos + seg.size] = arr_min[seg].astype(np.int16)
        stop_arr_node[write_pos:write_pos + seg.size] = a_ids[seg]
        write_pos += seg.size
    stop_arr_indptr = np.maximum.accumulate(stop_arr_indptr)

    # BOARD + WALK_AND_BOARD
    bs_src = []; bs_dst = []; bs_w = []
    for ci in range(n_conn):
        a_node = int(a_ids[ci])
        s = int(to_stop[ci]); a_t = int(arr_min[ci])
        lo, hi = int(stop_dep_indptr[s]), int(stop_dep_indptr[s + 1])
        if hi > lo:
            times = stop_dep_time[lo:hi]
            k = int(np.searchsorted(times, a_t, side="left"))
            if k < (hi - lo):
                d_node = int(stop_dep_node[lo + k])
                w = int(stop_dep_time[lo + k]) - a_t
                bs_src.append(a_node); bs_dst.append(d_node); bs_w.append(w)
        wlo, whi = int(walk_indptr[s]), int(walk_indptr[s + 1])
        for wk in range(wlo, whi):
            sp_ = int(walk_indices[wk]); wmin = int(walk_weights[wk])
            if sp_ == s and wmin == 0:
                continue
            t_threshold = a_t + wmin
            dlo, dhi = int(stop_dep_indptr[sp_]), int(stop_dep_indptr[sp_ + 1])
            if dhi > dlo:
                times = stop_dep_time[dlo:dhi]
                k = int(np.searchsorted(times, t_threshold, side="left"))
                if k < (dhi - dlo):
                    d_node = int(stop_dep_node[dlo + k])
                    w = int(stop_dep_time[dlo + k]) - a_t
                    bs_src.append(a_node); bs_dst.append(d_node); bs_w.append(w)
    if bs_src:
        src_list.append(np.asarray(bs_src, dtype=np.int64))
        dst_list.append(np.asarray(bs_dst, dtype=np.int64))
        wt_list.append(np.asarray(bs_w, dtype=np.int32))

    src_all = np.concatenate(src_list).astype(np.int32)
    dst_all = np.concatenate(dst_list).astype(np.int32)
    wt_all  = np.concatenate(wt_list).astype(np.int32)

    # Dedup (src,dst) keeping min weight
    key = src_all.astype(np.int64) * np.int64(n_nodes) + dst_all.astype(np.int64)
    order = np.lexsort((wt_all, key))
    key_s = key[order]
    src_s = src_all[order]; dst_s = dst_all[order]; wt_s = wt_all[order]
    first_in_group = np.empty(len(key_s), dtype=bool)
    first_in_group[0] = True
    first_in_group[1:] = key_s[1:] != key_s[:-1]
    src_u = src_s[first_in_group]; dst_u = dst_s[first_in_group]; wt_u = wt_s[first_in_group]

    order = np.argsort(src_u, kind="stable")
    src_o = src_u[order]; dst_o = dst_u[order]; wt_o = wt_u[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(indptr, src_o + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = dst_o; weights = wt_o

    rorder = np.argsort(dst_u, kind="stable")
    src_r = dst_u[rorder]; dst_r = src_u[rorder]; wt_r = wt_u[rorder]
    rev_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(rev_indptr, src_r + 1, 1)
    np.cumsum(rev_indptr, out=rev_indptr)
    rev_indices = dst_r; rev_weights = wt_r

    return TEEG(
        indptr=indptr, indices=indices, weights=weights,
        rev_indptr=rev_indptr, rev_indices=rev_indices, rev_weights=rev_weights,
        node_kind=node_kind, node_stop=node_stop, node_time=node_time,
        node_trip=node_trip, node_route=node_route,
        stop_dep_indptr=stop_dep_indptr, stop_dep_time=stop_dep_time, stop_dep_node=stop_dep_node,
        stop_arr_indptr=stop_arr_indptr, stop_arr_time=stop_arr_time, stop_arr_node=stop_arr_node,
        walk_indptr=walk_indptr, walk_indices=walk_indices, walk_weights=walk_weights,
        rev_walk_indptr=rev_walk_indptr, rev_walk_indices=rev_walk_indices, rev_walk_weights=rev_walk_weights,
        n_stops=int(n_stops), t_lo=int(t_lo), t_hi=int(t_hi),
    )
