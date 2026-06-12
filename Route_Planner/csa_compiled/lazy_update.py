"""Lazy CSA-resort dynamic-update baseline.

This is the realistic "CSA incremental" deployment that hostile reviewer #1
demands we compare TEEG-INCR against. It is a much fairer baseline than
"rebuild the entire TEEG from scratch": when a transit operator applies N
delay/cancel/add edits, the cheapest CSA-side reaction is to mutate the raw
connection array (batching by op-type for efficiency), re-sort by departure
time (CSA's monotonicity invariant) and then run CSA from scratch on the
modified array.

The expensive operation is the lexsort of the (now mutated) connection array,
which is O(M log M) over the *whole* timetable (M ≈ 2.3 M conns for Full
London). TEEG-INCR's incremental graph patching is O(touched edges) and so
should win by 1–2 orders of magnitude — but not by 4–5 orders, which is the
ratio against full TEEG rebuild.

Edit format (int trip indices, identical to TEEG-INCR for parity):
    ("DELAY",  trip_idx, delta_min)
    ("CANCEL", trip_idx)
    ("ADD",    template_trip_idx, start_minute_override)

Within one update we batch all cancellations into a single mask and all
additions into a single concatenate; this is what a competent CSA-only
operator would actually do (per-edit realloc would be silly). Delays are
applied per-trip in place because their cost dominates only at very large N.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np


def lazy_csa_resort_update(bundle, edits, *, copy_bundle: bool = True):
    """Apply N edits to bundle.connections and re-sort by departure time.

    Returns
    -------
    elapsed_s : float
        Wall-clock seconds for the full update (mutation + final lexsort).
    new_bundle : TimetableBundle
        A new bundle whose ``connections`` array satisfies the CSA monotonicity
        invariant (sorted by dep_min, secondary arr_min). The original bundle
        is left untouched when ``copy_bundle=True``.

    Notes
    -----
    This intentionally does NOT include the cost of any per-query CSA work or
    any closed-walks rebuild — only the data-structure mutation that the
    operator MUST do before the next CSA query can be served.
    """
    t0 = time.perf_counter()

    conns = bundle.connections.copy() if copy_bundle else bundle.connections
    trip_id_str = list(bundle.trip_id_str)

    # ---- Pass 1: collect per-op work ----
    cancel_set: set[int] = set()
    delay_ops: list[tuple[int, int]] = []
    add_ops: list[tuple[int, int]] = []
    for e in edits:
        op = e[0]
        if op == "DELAY":
            delay_ops.append((int(e[1]), int(e[2])))
        elif op == "CANCEL":
            cancel_set.add(int(e[1]))
        elif op == "ADD":
            add_ops.append((int(e[1]), int(e[2])))
        else:
            raise ValueError(f"unknown lazy-CSA edit op: {op!r}")

    # ---- Pass 2: apply ----
    # Delays: in-place per trip (no realloc).
    for trip_idx, delta in delay_ops:
        mask = conns["trip_id"] == np.int32(trip_idx)
        if not mask.any():
            continue
        nd = conns["dep_min"][mask].astype(np.int32) + delta
        na = conns["arr_min"][mask].astype(np.int32) + delta
        np.clip(nd, -32768, 32767, out=nd)
        np.clip(na, -32768, 32767, out=na)
        conns["dep_min"][mask] = nd.astype(np.int16)
        conns["arr_min"][mask] = na.astype(np.int16)

    # Cancellations: batch into one mask (realistic operator does this).
    if cancel_set:
        cancel_arr = np.fromiter(cancel_set, dtype=np.int32, count=len(cancel_set))
        keep = ~np.isin(conns["trip_id"], cancel_arr)
        conns = conns[keep]

    # Additions: build all new rows then a single concatenate.
    if add_ops:
        new_parts: list[np.ndarray] = []
        for template_idx, start_min in add_ops:
            tmask = conns["trip_id"] == np.int32(template_idx)
            template = conns[tmask]
            if len(template) == 0:
                continue
            tstart = int(template["dep_min"].min())
            shift = int(start_min) - tstart
            nd = template["dep_min"].astype(np.int32) + shift
            na = template["arr_min"].astype(np.int32) + shift
            if (nd < -32768).any() or (nd > 32767).any() \
                    or (na < -32768).any() or (na > 32767).any():
                continue
            new_conns = template.copy()
            new_conns["dep_min"] = nd.astype(np.int16)
            new_conns["arr_min"] = na.astype(np.int16)
            new_trip_idx = len(trip_id_str)
            trip_id_str.append(f"LAZYCSA_ADDED_{new_trip_idx}")
            new_conns["trip_id"] = new_trip_idx
            new_parts.append(new_conns)
        if new_parts:
            conns = np.concatenate([conns] + new_parts)

    order = np.lexsort((conns["arr_min"], conns["dep_min"]))
    conns = conns[order]

    new_bundle = replace(bundle, connections=conns, trip_id_str=trip_id_str)
    elapsed_s = time.perf_counter() - t0
    return elapsed_s, new_bundle
