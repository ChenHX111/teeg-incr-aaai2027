"""ULTRA FIFO preflight + violation classification.

Per Stage 2 F.5 freeze (FIFO preflight test) + F.6 (mitigation tree, after
iter-2 B6+ removal of 5%-fraction threshold) + B5 action ledger.

The existing `Route_Planner.raptor.raptor.raptor_earliest_arrival` is
**FIFO-safe** (uses vectorised min-of-valid-departures regardless of
trip ordering in the underlying RAPTOR arrays). Therefore RAPTOR itself
does not require FIFO; but ULTRA's published correctness theorem
(Baum et al. 2019) implicitly assumes the timetable is FIFO-clean within
each route.

This module:
1. Detects FIFO violations in the existing bundle.
2. Classifies each violation into one of:
   - `nat`: a NaT (missing arr/dep) at one stop position of one trip
   - `wraparound`: overnight wraparound (dep > 24h boundary)
   - `overtaking`: trip A overtakes trip B (later dep, earlier arrival at
     same later stop, both within a single service-day window)
   - `data_error`: other unclassified violation
3. Returns a classification report that drives the Stage 6 mitigation
   decision tree (per iter-2 B6+ supplemental synthesis):
   - clean → proceed
   - nat/wraparound only + synthetic non-FIFO test passes → proceed
     with documented note
   - any overtaking | data_error | unclassified → BLOCK Cycle 7;
     reopen Stage 2.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class FifoViolation:
    route_int: int
    stop_pos: int
    trip_a: int            # the "earlier-departing" trip (smaller idx after sorting)
    trip_b: int            # the trip it conflicts with
    a_dep: int             # minutes; -1 for NaT
    b_dep: int
    a_arr: int             # arr_min at the SAME stop pos (or following stop)
    b_arr: int
    klass: str             # one of {"nat", "wraparound", "overtaking", "data_error"}


@dataclass
class FifoReport:
    n_routes: int
    n_routes_checked: int
    n_violations: int
    by_class: dict          # {"nat": int, "wraparound": int, "overtaking": int, "data_error": int}
    examples: list          # list[FifoViolation] (capped at 50 for compactness)
    fifo_clean: bool        # True iff zero violations across all classes
    decision_code: str      # one of:
                            # "proceed"
                            # "proceed_with_note"
                            # "block_reopen_stage_2"
    decision_rationale: str

    def to_json(self) -> dict:
        return {
            "n_routes": self.n_routes,
            "n_routes_checked": self.n_routes_checked,
            "n_violations": self.n_violations,
            "by_class": self.by_class,
            "examples": [v.__dict__ for v in self.examples[:50]],
            "fifo_clean": self.fifo_clean,
            "decision_code": self.decision_code,
            "decision_rationale": self.decision_rationale,
        }


def _classify(a_dep: int, b_dep: int, a_arr: int, b_arr: int,
              wrap_threshold_min: int = 720) -> str:
    """Classify a (trip A, trip B) FIFO violation at one stop pos.

    Definitions:
    - A and B are two distinct trips on the same route, sorted such that
      `a_dep <= b_dep` at the boarding position.
    - Violation: at a later stop on the route, B arrives strictly BEFORE A
      (i.e., the later-boarding trip overtakes the earlier-boarding trip).

    Classes:
    - "nat": a_dep<0 OR b_dep<0 OR a_arr<0 OR b_arr<0 (any -1 marker)
    - "wraparound": |a_dep - b_dep| > wrap_threshold_min (overnight wrap)
    - "overtaking": substantive overtaking within a service-day window
    - "data_error": fall-through (none of the above; e.g., zero-duration trip)
    """
    if a_dep < 0 or b_dep < 0 or a_arr < 0 or b_arr < 0:
        return "nat"
    if abs(a_dep - b_dep) > wrap_threshold_min:
        return "wraparound"
    # Real overtaking: A boards earlier (a_dep <= b_dep) but A arrives later (a_arr > b_arr) at the same later stop.
    if a_dep <= b_dep and a_arr > b_arr:
        return "overtaking"
    return "data_error"


def fifo_preflight(b, wrap_threshold_min: int = 720, example_cap: int = 50) -> FifoReport:
    """Run the FIFO preflight on a TimetableBundle.

    For each route:
      1. Get all trips, sort them by FIRST-VALID departure minute
         (ignoring NaTs at trip head per `routes_array` layout). Stage 2
         F.5 spec: sort by first valid departure.
      2. For each adjacent sorted trip pair (A, B), walk down the stops
         and detect violations.

    Returns a FifoReport with the mitigation decision_code per the F.6
    iter-2 classification-only tree.
    """
    routes_array = b.routes_array       # (R, 4) int64: n_trips, n_stops, rs_off, st_off
    route_stops  = b.route_stops
    st_min       = b.stop_times_min     # (Nst, 2) int16: arr_min, dep_min

    examples: list[FifoViolation] = []
    by_class = {"nat": 0, "wraparound": 0, "overtaking": 0, "data_error": 0}
    n_violations = 0
    n_routes_checked = 0
    n_routes = int(routes_array.shape[0])

    for r in range(n_routes):
        n_tr = int(routes_array[r, 0])
        n_st = int(routes_array[r, 1])
        st_off = int(routes_array[r, 3])
        if n_tr < 2:
            continue
        n_routes_checked += 1

        # Collect first-valid departure minute per trip
        first_deps = np.full(n_tr, np.iinfo(np.int32).max, dtype=np.int32)
        for t in range(n_tr):
            for i in range(n_st):
                d = int(st_min[st_off + t * n_st + i, 1])
                if d >= 0:
                    first_deps[t] = d
                    break
        # Sort trips by first valid departure
        sort_idx = np.argsort(first_deps, kind="stable")

        # For each adjacent sorted pair, check overtaking at every stop pos
        for k in range(n_tr - 1):
            t_a = int(sort_idx[k])
            t_b = int(sort_idx[k + 1])
            for i in range(n_st):
                a_arr = int(st_min[st_off + t_a * n_st + i, 0])
                b_arr = int(st_min[st_off + t_b * n_st + i, 0])
                a_dep = int(st_min[st_off + t_a * n_st + i, 1])
                b_dep = int(st_min[st_off + t_b * n_st + i, 1])

                # Check if A overtakes B (we expect A <= B everywhere by FIFO)
                if a_arr >= 0 and b_arr >= 0 and a_arr > b_arr:
                    klass = _classify(a_dep, b_dep, a_arr, b_arr, wrap_threshold_min)
                    by_class[klass] += 1
                    n_violations += 1
                    if len(examples) < example_cap:
                        examples.append(FifoViolation(
                            route_int=r, stop_pos=i, trip_a=t_a, trip_b=t_b,
                            a_dep=a_dep, b_dep=b_dep, a_arr=a_arr, b_arr=b_arr,
                            klass=klass,
                        ))

    # Apply iter-2 mitigation decision tree (classification-only; no
    # percentage thresholds per B6+)
    fifo_clean = (n_violations == 0)
    if fifo_clean:
        decision_code = "proceed"
        rationale = "Clean (zero violations across all classes)."
    elif (by_class["overtaking"] == 0 and by_class["data_error"] == 0):
        # Only nat or wraparound — proceed with documented note,
        # CONDITIONAL on synthetic non-FIFO test passing (the test is
        # designed to surface algorithm-level FIFO assumptions; this
        # report only classifies the data-level violations).
        decision_code = "proceed_with_note"
        rationale = (
            f"NaT-only or wraparound-only violations "
            f"({by_class['nat']} nat, {by_class['wraparound']} wrap). "
            f"Proceed conditional on synthetic non-FIFO test (Stage 5 C4 oracle) passing."
        )
    else:
        decision_code = "block_reopen_stage_2"
        rationale = (
            f"Substantive overtaking ({by_class['overtaking']}) or data_error "
            f"({by_class['data_error']}) detected. Per iter-2 B6+ classification "
            f"tree, BLOCK Cycle 7 and reopen Stage 2 for mitigation design."
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
