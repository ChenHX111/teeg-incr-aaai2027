"""CSA edit operator: apply EditSet to a TimetableBundle (Route_Planner.common).

Edit semantics (per CYCLE_10_PREREGISTRATION.md §3):
- DELAY: shift trip's dep_min and arr_min by +delta_min for ALL its connections.
- CANCELLATION: remove all connections belonging to the trip.
- ADDITION: copy template trip's connection pattern (relative offsets between
  stops preserved); shift dep_min of first stop to start_minute_override.

Returns a NEW TimetableBundle (does not mutate the input). The new bundle has
the same n_stops, transfers_from, stops, routes, etc. — only `connections`
and `trip_id_str` change.

Connections array is re-sorted (lexsort by dep_min then arr_min) to satisfy
CSA's monotonicity invariant.
"""
from __future__ import annotations
import time
from dataclasses import replace
from typing import Iterable

import numpy as np

from ..schemas import EditSet, EditType, TripEdit


def _trip_str_to_idx_map(trip_id_str: list[str]) -> dict[str, int]:
    """Build trip_id (string) → trip_index (int) map."""
    return {tid: i for i, tid in enumerate(trip_id_str)}


def _conns_for_trip(connections: np.ndarray, trip_idx: int) -> np.ndarray:
    """Return mask selecting all connections of trip_idx."""
    return connections["trip_id"] == trip_idx


def apply_delay(connections: np.ndarray, trip_idx: int, delta_min: int) -> np.ndarray:
    """In-place: shift dep_min and arr_min by +delta_min for the trip's conns.

    Returns the MUTATED connections array (assumes caller has a writable copy).
    """
    mask = _conns_for_trip(connections, trip_idx)
    if not mask.any():
        return connections
    # Add delta but respect int16 range
    new_dep = connections["dep_min"][mask].astype(np.int32) + delta_min
    new_arr = connections["arr_min"][mask].astype(np.int32) + delta_min
    # Clip if would overflow int16 (the original dtype)
    new_dep = np.clip(new_dep, -32768, 32767).astype(np.int16)
    new_arr = np.clip(new_arr, -32768, 32767).astype(np.int16)
    connections["dep_min"][mask] = new_dep
    connections["arr_min"][mask] = new_arr
    return connections


def apply_cancellation(connections: np.ndarray, trip_idx: int) -> np.ndarray:
    """Return a NEW connections array with all trip_idx connections removed."""
    keep = ~_conns_for_trip(connections, trip_idx)
    return connections[keep].copy()


def apply_addition(
    connections: np.ndarray,
    template_trip_idx: int,
    new_trip_idx: int,
    start_minute_override: int,
) -> tuple[np.ndarray, int]:
    """Append a NEW trip's connections to the array, copying template's relative
    stop-to-stop offsets but shifting absolute time so dep_min at first stop
    equals start_minute_override.

    Returns (new_connections_array, n_added).
    """
    template_mask = _conns_for_trip(connections, template_trip_idx)
    template_conns = connections[template_mask].copy()
    if len(template_conns) == 0:
        return connections, 0
    # Find the template's first-stop dep_min (= the smallest dep_min in this trip)
    template_start = int(template_conns["dep_min"].min())
    shift = int(start_minute_override) - template_start
    # Apply shift
    new_dep = template_conns["dep_min"].astype(np.int32) + shift
    new_arr = template_conns["arr_min"].astype(np.int32) + shift
    # Reject if any would underflow (don't clip silently)
    if (new_dep < -32768).any() or (new_dep > 32767).any() or (new_arr < -32768).any() or (new_arr > 32767).any():
        # Out of representable range — skip this addition
        return connections, 0
    # Build new conn rows (copy template structure, change dep/arr/trip_id)
    new_conns = template_conns.copy()
    new_conns["dep_min"] = new_dep.astype(np.int16)
    new_conns["arr_min"] = new_arr.astype(np.int16)
    new_conns["trip_id"] = new_trip_idx
    # Append to existing
    appended = np.concatenate([connections, new_conns])
    return appended, len(new_conns)


def apply_edit_set_to_bundle(
    bundle,
    edit_set: EditSet,
    *,
    return_timing: bool = False,
):
    """Apply an EditSet to a Route_Planner.common.TimetableBundle.

    Returns a NEW bundle. Original is not mutated.

    If return_timing: returns (new_bundle, edit_wall_s).
    """
    t0 = time.perf_counter()

    # Build trip-id maps (these mirror b.trip_id_str)
    trip_id_str = list(bundle.trip_id_str)
    trip_to_idx = _trip_str_to_idx_map(trip_id_str)
    # Working copy of connections (mutable)
    conns = bundle.connections.copy()

    n_delay = 0
    n_cancel = 0
    n_addition = 0

    for edit in edit_set.edits:
        if edit.edit_type == EditType.DELAY:
            if edit.trip_id not in trip_to_idx:
                continue
            tidx = trip_to_idx[edit.trip_id]
            apply_delay(conns, tidx, edit.delta_min)
            n_delay += 1
        elif edit.edit_type == EditType.CANCELLATION:
            if edit.trip_id not in trip_to_idx:
                continue
            tidx = trip_to_idx[edit.trip_id]
            conns = apply_cancellation(conns, tidx)
            n_cancel += 1
        elif edit.edit_type == EditType.ADDITION:
            if edit.template_trip_id not in trip_to_idx:
                continue
            template_idx = trip_to_idx[edit.template_trip_id]
            # Assign new trip index = next available
            new_idx = len(trip_id_str)
            trip_id_str.append(edit.trip_id)
            trip_to_idx[edit.trip_id] = new_idx
            conns, n_added = apply_addition(
                conns, template_idx, new_idx, edit.start_minute_override,
            )
            if n_added > 0:
                n_addition += 1
        else:
            raise ValueError(f"unknown edit type: {edit.edit_type}")

    # Re-sort connections (CSA invariant: dep_min ascending, secondary arr_min)
    order = np.lexsort((conns["arr_min"], conns["dep_min"]))
    conns = conns[order]

    # Build the new bundle (replace connections + trip_id_str)
    new_bundle = replace(bundle, connections=conns, trip_id_str=trip_id_str)

    edit_wall_s = time.perf_counter() - t0
    if return_timing:
        return new_bundle, edit_wall_s, {
            "n_delay_applied": n_delay,
            "n_cancel_applied": n_cancel,
            "n_addition_applied": n_addition,
            "n_conns_before": len(bundle.connections),
            "n_conns_after": len(conns),
        }
    return new_bundle
