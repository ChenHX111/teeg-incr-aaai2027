"""ULTRA FIFO preflight v2 — proper passenger-relevant overtaking classification.

Per Stage 5 §E freeze (3/3 investigator convergence) + Stage 4 L2 known limitation.

**Difference from v1** (`fifo.py`):
v1 detected any "trip A arrives at stop X later than trip B at stop X" pattern
when A was sorted earlier by first valid departure. This produced 189 "data_error"
false positives on the subset, because two trips at very different times of day
(e.g., morning vs evening, both early-departing relative to each other on
different segments) appear as "violations" without any passenger ever facing
this choice.

v2 implements the correct **passenger-relevant overtaking** definition:
For each route, for each (boarding-stop X, target-stop Y) pair with X < Y on
the route:
- Collect trips that have BOTH valid `dep_X` AND valid `arr_Y`.
- Sort them by `dep_X` (the boarding time at X).
- Check if `arr_Y` is monotone non-decreasing in dep-order.
- Any inversion = REAL OVERTAKING (a passenger boarding the later-departing
  trip at X arrives at Y before a passenger on the earlier-departing trip).

This is the definition of FIFO violation that actually breaks ULTRA's
correctness theorem (Baum et al. 2019). NaT-positions are handled by skipping
trips that have invalid dep/arr at the relevant positions.

**Output**: same `FifoReport` dataclass as v1, with `by_class` keys
{overtaking, nat, wraparound, data_error}. The mitigation tree from v1 (iter-2
B6+ classification-only) is unchanged:
- clean → proceed
- nat/wraparound only + synthetic non-FIFO test passes → proceed_with_note
- ANY overtaking OR data_error OR unclassified → block_reopen_stage_2.

Per Stage 5 cross-stage flag: v1 (`fifo.py`) is preserved unchanged; v2 lives
as a separate function in this module for side-by-side comparison.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from .fifo import FifoReport, FifoViolation


def fifo_preflight_v2(b, wrap_threshold_min: int = 720, example_cap: int = 50) -> FifoReport:
    """v2 FIFO preflight: per (route, X<Y) pair, find arr_Y inversions.

    Returns a FifoReport with v1-compatible schema.

    Algorithmic complexity: O(R × S² × T log T) where R = routes,
    S = stops/route, T = trips/route. For our subset (1014 routes, avg ~21
    stops, avg ~9 trips per route): ~1014 × 441 × 32 ≈ 14M ops. Tractable in
    a few seconds.
    """
    routes_array = b.routes_array
    route_stops = b.route_stops
    st_min = b.stop_times_min

    examples: list[FifoViolation] = []
    by_class = {"overtaking": 0, "nat": 0, "wraparound": 0, "data_error": 0}
    n_violations = 0
    n_routes_checked = 0
    n_routes = int(routes_array.shape[0])

    for r in range(n_routes):
        n_tr = int(routes_array[r, 0])
        n_st = int(routes_array[r, 1])
        st_off = int(routes_array[r, 3])
        if n_tr < 2 or n_st < 2:
            continue
        n_routes_checked += 1

        # Materialise the (n_tr, n_st) arr / dep matrices once
        arr_mat = st_min[st_off:st_off + n_tr * n_st, 0].reshape(n_tr, n_st)
        dep_mat = st_min[st_off:st_off + n_tr * n_st, 1].reshape(n_tr, n_st)

        # For each boarding stop position X
        for x in range(n_st - 1):
            dep_x = dep_mat[:, x]
            # For each later target stop position Y > X
            for y in range(x + 1, n_st):
                arr_y = arr_mat[:, y]
                # Trips with both valid dep at X and valid arr at Y
                valid = (dep_x >= 0) & (arr_y >= 0)
                if valid.sum() < 2:
                    continue
                trip_idx = np.where(valid)[0]
                deps = dep_x[trip_idx]
                arrs = arr_y[trip_idx]
                # Sort by dep at X (stable so ties keep original order)
                order = np.argsort(deps, kind="stable")
                trip_sorted = trip_idx[order]
                arrs_sorted = arrs[order]
                deps_sorted = deps[order]
                # Detect non-monotone arr_y: any (i, i+1) where arrs[i+1] < arrs[i]
                # AND it's not a wraparound (|dep gap| > wrap_threshold)
                for i in range(len(trip_sorted) - 1):
                    t_a = int(trip_sorted[i])
                    t_b = int(trip_sorted[i + 1])
                    a_dep = int(deps_sorted[i])
                    b_dep = int(deps_sorted[i + 1])
                    a_arr = int(arrs_sorted[i])
                    b_arr = int(arrs_sorted[i + 1])
                    if b_arr >= a_arr:
                        continue  # FIFO at this (X, Y) for this pair; OK
                    # Inversion detected (b_arr < a_arr while b_dep >= a_dep)
                    # Classify:
                    if abs(b_dep - a_dep) > wrap_threshold_min:
                        klass = "wraparound"
                    elif b_dep > a_dep and b_arr < a_arr:
                        klass = "overtaking"
                    elif b_dep == a_dep:
                        # Two trips with same boarding time but different arrivals
                        # at Y — unusual but not strictly overtaking
                        klass = "data_error"
                    else:
                        klass = "data_error"
                    by_class[klass] += 1
                    n_violations += 1
                    if len(examples) < example_cap:
                        examples.append(FifoViolation(
                            route_int=r, stop_pos=y, trip_a=t_a, trip_b=t_b,
                            a_dep=a_dep, b_dep=b_dep, a_arr=a_arr, b_arr=b_arr,
                            klass=klass,
                        ))
                # Note: we do NOT bump by_class["nat"] because v2 handles NaTs
                # by skipping invalid (dep_X, arr_Y) pairs above — they are
                # not violations, they are missing data.

    # Apply same iter-2 B6+ classification-only mitigation tree as v1
    fifo_clean = (n_violations == 0)
    if fifo_clean:
        decision_code = "proceed"
        rationale = "v2: clean (zero passenger-relevant overtaking, no wraparound, no data_error)."
    elif by_class["overtaking"] == 0 and by_class["data_error"] == 0:
        decision_code = "proceed_with_note"
        rationale = (
            f"v2: wraparound-only ({by_class['wraparound']} cases). Proceed "
            f"conditional on synthetic non-FIFO test passing."
        )
    else:
        decision_code = "block_reopen_stage_2"
        rationale = (
            f"v2: substantive overtaking ({by_class['overtaking']}) or "
            f"data_error ({by_class['data_error']}). BLOCK Cycle 7."
        )

    return FifoReport(
        n_routes=n_routes,
        n_routes_checked=n_routes_checked,
        n_violations=n_violations,
        by_class=by_class,
        examples=examples,
        fifo_clean=fifo_clean,
        decision_code=decision_code,
        decision_rationale=rationale,
    )
