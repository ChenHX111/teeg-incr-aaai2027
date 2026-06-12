"""Preprocess helpers for compiled RAPTOR (CSR position table).

The position table answers: "given stop s and route r, at what position does s
appear in r's stop sequence?" Without this, the marked-stop-collection loop
must linear-scan each route's stop sequence (O(|marked| × avg_route_length)
per round). With it, lookup is O(degree(s) in stop_routes), which is bounded.

CSR layout (audit-recommended over dense O(R*S) table):
- `srp_indptr[s], srp_indptr[s+1]`: range into `srp_route_ids` / `srp_positions`
  for stop `s`.
- `srp_route_ids[k]`: route index `r` containing stop `s`.
- `srp_positions[k]`: position of `s` within route `r`'s stop sequence.

NOTE on multi-occurrence stops in same route (loop routes):
For stops appearing multiple times in one route (e.g., loop bus routes that
return to the same stop), `build_position_table` records only the FIRST
occurrence. This matches the pure-Python `find_pos` semantics (also `break`s
on first match). Both implementations therefore agree, but neither handles
loop routes optimally per RAPTOR's "each occurrence is a boarding opportunity"
spec. Acceptable for AAAI as long as: (a) we document the limitation, and
(b) the pure-Python reference makes the same simplification (it does).

CSR INVARIANT: the input `stops_array[s, 0:2] = (srlo, srhi)` must describe
a contiguous range in `stop_routes` such that `stop_routes[srlo:srhi]` lists
all routes containing stop `s`. The output `srp_indptr` is a copy of column 0
plus a terminal entry; we assert non-overlap and bound-coverage post-build.
"""
from __future__ import annotations
import numpy as np


def build_position_table(routes_array: np.ndarray, route_stops: np.ndarray,
                          stops_array: np.ndarray, stop_routes: np.ndarray,
                          n_stops: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR-style (stop, route) → position lookup.

    See module docstring for layout + invariants.
    """
    n_entries = len(stop_routes)
    srp_indptr = np.zeros(n_stops + 1, dtype=np.int32)
    srp_route_ids = np.zeros(n_entries, dtype=np.int32)
    srp_positions = np.full(n_entries, -1, dtype=np.int32)

    for s in range(n_stops):
        srlo = int(stops_array[s, 0])
        srhi = int(stops_array[s, 1])
        # Bounds defence (audit-mandated CSR correctness assertion)
        assert 0 <= srlo <= srhi <= n_entries, (
            f"build_position_table: stop {s} bounds ({srlo}, {srhi}) out of "
            f"[0, {n_entries}] — invalid stops_array layout"
        )
        srp_indptr[s] = srlo
        for j in range(srlo, srhi):
            r = int(stop_routes[j])
            srp_route_ids[j] = r
            rs_off = int(routes_array[r, 2])
            n_st = int(routes_array[r, 1])
            pos = -1
            # Multi-occurrence stops: take FIRST occurrence (matches pure-Python find_pos)
            for i in range(n_st):
                if int(route_stops[rs_off + i]) == s:
                    pos = i
                    break
            srp_positions[j] = pos
    srp_indptr[n_stops] = n_entries

    # Final CSR invariant checks (audit-mandated correctness gates)
    # 1. First entry is 0 (or matches stops_array[0, 0]).
    if n_stops > 0:
        assert srp_indptr[0] == int(stops_array[0, 0]), \
            f"srp_indptr[0]={srp_indptr[0]} != stops_array[0,0]={int(stops_array[0, 0])}"
    # 2. Last entry equals total entries.
    assert srp_indptr[n_stops] == n_entries, \
        f"srp_indptr[n_stops]={srp_indptr[n_stops]} != n_entries={n_entries}"
    # 3. Monotone non-decreasing (CSR property).
    diffs = np.diff(srp_indptr)
    assert (diffs >= 0).all(), "srp_indptr not monotone non-decreasing"
    # 4. CONTIGUITY: stops_array[s,1] must equal stops_array[s+1,0] (no gaps/overlaps).
    # This is the strong CSR invariant that the monotone check alone does NOT enforce.
    for s in range(n_stops - 1):
        srhi_s = int(stops_array[s, 1])
        srlo_next = int(stops_array[s + 1, 0])
        assert srhi_s == srlo_next, (
            f"stops_array not contiguous at stop {s}: srhi[{s}]={srhi_s} != "
            f"srlo[{s+1}]={srlo_next} — gap or overlap; CSR malformed"
        )
    if n_stops > 0:
        assert int(stops_array[0, 0]) == 0, \
            f"stops_array[0, 0]={int(stops_array[0, 0])} != 0; CSR must start at 0"
        assert int(stops_array[n_stops - 1, 1]) == n_entries, \
            f"stops_array[-1, 1]={int(stops_array[n_stops - 1, 1])} != n_entries={n_entries}"
    return srp_indptr, srp_route_ids, srp_positions
