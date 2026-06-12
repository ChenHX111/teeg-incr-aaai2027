"""Dynamic update wrappers for MG-Dial v2.

* :func:`cancel_trip` — set tombstone bits for every CSR edge belonging to
  ``trip_id``.
* :func:`restore_trip` — clear tombstone bits set by :func:`cancel_trip`.
* :func:`delay_trip` — uniform whole-trip delay = ``cancel_trip`` followed
  by ``add_trip`` of a shifted copy of the original connections.
* :func:`add_trip` — append a new trip's edges to the runner's overlay CSR
  (R/W RIDE -> A, A -> EXIT R, A -> CONTINUE between consecutive conns).

All operations are O(|edges of trip|) ≈ O(stops_per_trip).  The overlay
CSR is recompiled lazily on the next :meth:`MGDialV2.query` call.

Note: *added* trips live in the overlay CSR, NOT the base CSR.  Cancelling
an *added* trip is supported via a per-overlay-trip toggle list (see
``add_trip`` / ``cancel_added_trip``).  Cancellation of base-graph trips
uses tombstones on ``g.indices``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .build import (
    ET_RIDE, ET_CONTINUE, ET_EXIT, ET_WAIT_R, ET_WAIT_W, ET_WALK,
    MinuteGridV2,
)
from .query import MGDialV2


def cancel_trip(runner: MGDialV2, trip_id: int) -> int:
    """Tombstone every base-CSR edge of ``trip_id``.

    Returns the number of edges tombstoned.
    """
    g = runner.g
    if trip_id < 0 or trip_id >= len(g.trip_edge_index):
        raise ValueError(f"trip_id {trip_id} out of range [0, {g.n_trips})")
    edges = g.trip_edge_index[trip_id]
    tomb = runner.tomb
    n = 0
    for e in edges:
        e_int = int(e)
        if not (tomb[e_int >> 3] & (np.uint8(1) << (e_int & 7))):
            n += 1
        tomb[e_int >> 3] |= (np.uint8(1) << (e_int & 7))
    return n


def restore_trip(runner: MGDialV2, trip_id: int) -> int:
    """Clear tombstone bits for every base-CSR edge of ``trip_id``.

    Returns the number of edges restored.
    """
    g = runner.g
    if trip_id < 0 or trip_id >= len(g.trip_edge_index):
        raise ValueError(f"trip_id {trip_id} out of range [0, {g.n_trips})")
    edges = g.trip_edge_index[trip_id]
    tomb = runner.tomb
    n = 0
    for e in edges:
        e_int = int(e)
        if tomb[e_int >> 3] & (np.uint8(1) << (e_int & 7)):
            n += 1
        tomb[e_int >> 3] &= np.uint8(~(np.uint8(1) << (e_int & 7)))
    return n


def add_trip(runner: MGDialV2, conns: list) -> int:
    """Append a new trip to the runner's overlay CSR.

    Parameters
    ----------
    runner : MGDialV2
    conns  : list of (from_stop, to_stop, dep_min, arr_min) tuples

    The connections are assumed to be already sorted by ``dep_min`` and to
    represent a single trip.  Returns the overlay-trip id (an opaque int
    used by ``cancel_added_trip``).

    The kernel applies overlay edges with no tombstones (overlay is fresh).
    """
    g = runner.g
    if not conns:
        raise ValueError("add_trip: empty connection list")

    T = g.T
    t_lo = g.t_lo
    t_hi = g.t_hi
    A_OFFSET = g.A_OFFSET
    RW_OFFSET = g.RW_OFFSET

    # Allocate a fresh A-node range for this overlay trip OUTSIDE the base
    # graph node space.  Numbering: g.n_nodes + per-overlay-trip offset.
    # We extend the node space implicitly via overlay; the kernel only
    # ever indexes ``dist`` for nodes that appear as destinations of some
    # edge, but we DO need ``ov_indptr`` to be sized to n_nodes.  Since we
    # cannot exceed n_nodes (would require a bigger dist array), we route
    # the overlay through "phantom" A-node ids in the existing A-plane
    # range using *unused* slots beyond the original n_connections.
    # For simplicity and safety, we DO NOT create new A-nodes for overlay
    # trips: we model them with a 2-step path that bypasses the A-plane.
    #
    # Simplification: an added trip is encoded as direct edges
    #   R(from,dep) --(arr-dep, RIDE)--> R(to,arr)
    #   W(from,dep) --(arr-dep, RIDE)--> R(to,arr)
    # plus, between consecutive conns, a CONTINUE edge
    #   R(c_i.to, c_i.arr) --(dep_next - arr, CONTINUE)--> R(c_{i+1}.from, dep_next)
    #
    # This sacrifices the "shadow node" guarantee for OVERLAY trips only.
    # F9 is checked on the BASE graph at build time (the only place where
    # silent shadow-sharing could underestimate transfers); overlay trips
    # are added explicitly by the experimenter and add at most one
    # additional concurrency, which they are responsible for tracking.
    # See DESIGN.md §2.7 (the kernel pseudocode for add_trip already
    # uses this simplified encoding).
    src_chunks = []
    dst_chunks = []
    w_chunks = []
    et_chunks = []
    for i, conn in enumerate(conns):
        from_s, to_s, dep, arr = (int(conn[0]), int(conn[1]),
                                  int(conn[2]), int(conn[3]))
        if dep < t_lo or arr > t_hi or arr < dep:
            raise ValueError(
                f"add_trip: connection {i} out of window ({from_s}->{to_s} "
                f"dep={dep} arr={arr}, window [{t_lo},{t_hi}])")
        if not (0 <= from_s < g.n_stops and 0 <= to_s < g.n_stops):
            raise ValueError(
                f"add_trip: stop id out of range in connection {i}: "
                f"{from_s} -> {to_s}")
        ride_w = arr - dep
        rR = from_s * T + (dep - t_lo)
        wW = RW_OFFSET + from_s * T + (dep - t_lo)
        rDst = to_s * T + (arr - t_lo)
        src_chunks.append(np.array([rR, wW], dtype=np.int64))
        dst_chunks.append(np.array([rDst, rDst], dtype=np.int64))
        w_chunks.append(np.array([ride_w, ride_w], dtype=np.int16))
        et_chunks.append(np.array([ET_RIDE, ET_RIDE], dtype=np.int8))
        if i + 1 < len(conns):
            nxt = conns[i + 1]
            n_from, n_dep = int(nxt[0]), int(nxt[2])
            gap = n_dep - arr
            if gap < 0:
                raise ValueError(
                    f"add_trip: non-monotone trip; conn {i} arr={arr} "
                    f"vs next dep={n_dep}")
            cSrc = to_s * T + (arr - t_lo)
            cDst = n_from * T + (n_dep - t_lo)
            src_chunks.append(np.array([cSrc], dtype=np.int64))
            dst_chunks.append(np.array([cDst], dtype=np.int64))
            w_chunks.append(np.array([gap], dtype=np.int16))
            et_chunks.append(np.array([ET_CONTINUE], dtype=np.int8))
    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    w = np.concatenate(w_chunks)
    et = np.concatenate(et_chunks)

    runner._ov_buf_src.append(src)
    runner._ov_buf_dst.append(dst)
    runner._ov_buf_w.append(w)
    runner._ov_buf_et.append(et)
    runner._ov_dirty = True
    overlay_trip_id = len(runner._ov_added_trips)
    runner._ov_added_trips.append({
        "buf_idx": len(runner._ov_buf_src) - 1,
        "active": True,
        "n_edges": int(src.size),
    })
    return overlay_trip_id


def cancel_added_trip(runner: MGDialV2, overlay_trip_id: int) -> int:
    """Drop an overlay trip previously added via :func:`add_trip`.

    Implemented by zeroing that trip's edges so they cannot relax (we
    rewrite the buffer so the next ``compile_overlay`` skips them).
    Returns the number of edges removed.
    """
    if not (0 <= overlay_trip_id < len(runner._ov_added_trips)):
        raise ValueError(f"overlay_trip_id {overlay_trip_id} out of range")
    rec = runner._ov_added_trips[overlay_trip_id]
    if not rec["active"]:
        return 0
    rec["active"] = False
    n = rec["n_edges"]
    bi = rec["buf_idx"]
    runner._ov_buf_src[bi] = np.zeros(0, dtype=np.int64)
    runner._ov_buf_dst[bi] = np.zeros(0, dtype=np.int64)
    runner._ov_buf_w[bi] = np.zeros(0, dtype=np.int16)
    runner._ov_buf_et[bi] = np.zeros(0, dtype=np.int8)
    runner._ov_dirty = True
    return n


def delay_trip(runner: MGDialV2, trip_id: int, delta_min: int) -> int:
    """Uniformly delay every connection of ``trip_id`` by ``delta_min``
    minutes.

    Implemented as ``cancel_trip(trip_id)`` followed by ``add_trip`` of
    the shifted copy.  Returns the new overlay-trip id.

    Raises ``ValueError`` if any shifted connection falls out of the
    [t_lo, t_hi] window.
    """
    g = runner.g
    if trip_id < 0 or trip_id >= g.n_trips:
        raise ValueError(f"trip_id {trip_id} out of range [0, {g.n_trips})")
    if delta_min == 0:
        # no-op (still cancel/restore? the spec says delay -> cancel+add;
        # zero delay is effectively a redundant rebuild)
        return -1

    # find connections of this trip (using trip_id_of_conn)
    mask = g.trip_id_of_conn == trip_id
    if not mask.any():
        raise ValueError(f"trip_id {trip_id} has no connections")
    idxs = np.where(mask)[0]
    deps = g.conn_dep[idxs].astype(np.int64) + int(delta_min)
    arrs = g.conn_arr[idxs].astype(np.int64) + int(delta_min)
    if (deps < g.t_lo).any() or (arrs > g.t_hi).any():
        raise ValueError(
            f"delay_trip: shifted trip {trip_id} falls outside window "
            f"[{g.t_lo}, {g.t_hi}] for delta={delta_min}")
    order = np.argsort(deps, kind="stable")
    new_conns = [
        (int(g.conn_from[idxs[k]]), int(g.conn_to[idxs[k]]),
         int(deps[k]), int(arrs[k]))
        for k in order
    ]
    cancel_trip(runner, trip_id)
    return add_trip(runner, new_conns)
