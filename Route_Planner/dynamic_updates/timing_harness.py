"""Cycle 10 Day 4 — timing harness for dynamic-update experiments.

Per pre-reg §3 and Cycle 10 design:
- For each (scenario, intensity, seed, method) arm:
  - (a) Build edit_set
  - (b) Apply edit set to bundle → measure EDIT WALL (broadcast-once amortised cost)
  - (c) Identify affected ODs (the subset of 3000 ODs whose paths might change)
  - (d) Re-query path-plan for those ODs on the edited bundle → measure RE-PLAN WALL (per-query)
  - (e) Compare post-edit EA vs baseline EA (no-edit) → SOLUTION QUALITY DELTA

Scenarios locked: DELAY / CANCELLATION / ADDITION
Intensities locked: (1, 5, 25, 125)
Methods: CSA / RAPTOR-compiled / TEEG-ALT / MG-Dial / ULTRA
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

INTENSITIES = (1, 5, 25, 125)
SCENARIOS = ("DELAY", "CANCELLATION", "ADDITION")
METHODS = ("csa", "raptor_compiled", "teeg_alt", "mg_dial", "ultra")


@dataclass
class ArmResult:
    """One row in the timing harness output.

    edit_wall_s: graph edit + all broadcast-once data-prep (transfers_idx,
        ALT landmarks, etc.) amortized across all users.
    edit_wall_includes: human-readable description of what's included.
    """
    scenario: str
    intensity: int
    seed: int
    method: str
    edit_wall_s: float
    n_affected_ods: int
    replan_wall_total_s: float
    replan_wall_mean_ms: float
    replan_wall_median_ms: float
    n_quality_changed: int
    n_quality_better: int
    n_quality_worse: int
    n_became_unreachable: int
    n_became_reachable: int
    median_quality_delta_min: float
    max_quality_delta_min: float
    edit_set_short_id: str
    notes: str = ""
    edit_wall_includes: str = ""


def build_trip_maps(bundle):
    """Build the trip_id_to_mode and trip_id_to_route maps needed by edit_generation."""
    # trip_id_str = ['T1', 'T2', ...]
    # For Cycle 6+, each trip's route is determined by its first connection's route_int
    # and the mode is mapped from the route_int → mode via the modes_routes meta (not stored).
    # As a placeholder: assume all trips are 'bus' (most numerous in London).
    # TODO: use the actual mode mapping from data_layer when available.
    trip_id_to_mode = {t: "bus" for t in bundle.trip_id_str}
    return trip_id_to_mode


def build_addition_aux_maps(bundle):
    """For ADDITION scenario: trip_id_to_route + route_id_to_starts."""
    # Each trip belongs to one route_int (the route_int of its first connection)
    trip_to_route_int = {}
    route_int_to_starts = {}
    conns = bundle.connections
    for tidx, tname in enumerate(bundle.trip_id_str):
        mask = conns["trip_id"] == tidx
        if mask.any():
            r_int = int(conns[mask][0]["route_int"])
            start_dep = int(conns[mask]["dep_min"].min())
            route_id_str = f"R{r_int}"
            trip_to_route_int[tname] = route_id_str
            route_int_to_starts.setdefault(route_id_str, []).append(start_dep)
    return trip_to_route_int, route_int_to_starts


def compute_affected_ods(edit_set, queries: pd.DataFrame, bundle) -> list[int]:
    """Conservative: any OD whose path could possibly use an edited trip.

    Day-4 simplification: return ALL OD indices (re-plan all 3000). A future
    optimisation could prune to ODs near edited trips (Cycle 10.5).
    """
    return list(range(len(queries)))


def run_one_arm(
    bundle, queries: pd.DataFrame, baseline_eas: pd.Series,
    scenario: str, intensity: int, seed: int, method: str,
    *, cw_csr=None, ultra=None, mg=None, teeg_g=None, alt_labels=None,
    srp_csr=None, transfers_idx=None,
) -> ArmResult:
    """Run one (scenario, intensity, seed, method) arm and return timing+quality result."""
    from Route_Planner.dynamic_updates.edit_generation import build_edit_set
    from Route_Planner.dynamic_updates.schemas import EditType
    from Route_Planner.dynamic_updates.operators.csa_edit import apply_edit_set_to_bundle
    from Route_Planner.dynamic_updates.operators.raptor_compiled_edit import apply_edit_set_to_raptor_compiled
    from Route_Planner.dynamic_updates.operators.teeg_edit import apply_edit_set_to_teeg
    from Route_Planner.dynamic_updates.operators.mg_edit import apply_edit_set_to_mg
    from Route_Planner.dynamic_updates.operators.ultra_edit import apply_edit_set_to_ultra

    # Build trip_id maps
    trip_id_to_mode = build_trip_maps(bundle)
    trip_to_route = None
    route_to_starts = None
    if scenario == "ADDITION":
        trip_to_route, route_to_starts = build_addition_aux_maps(bundle)

    edit_type = EditType(scenario)
    edit_set = build_edit_set(
        edit_type, intensity, seed, trip_id_to_mode,
        trip_id_to_route=trip_to_route,
        route_id_to_starts=route_to_starts,
    )

    # Apply edit + measure edit wall
    t0 = time.perf_counter()
    if method == "csa":
        edited = apply_edit_set_to_bundle(bundle, edit_set)
        # Per rubber-duck audit (rd_1 fix 6): include transfers_idx rebuild in edit_wall
        # because the re-plan needs it (broadcast-once cost, comparable to TEEG's ALT rebuild).
        from Route_Planner.csa.csa import _build_transitively_closed_transfers
        edited_transfers_idx = _build_transitively_closed_transfers(
            edited.transfers_from, edited.n_stops, max_walk_min=15)
        edit_wall = time.perf_counter() - t0
        edit_wall_includes = "csa_edit + transitive_closure_rebuild"
    elif method == "raptor_compiled":
        # Per rubber-duck audit (rd_1 fix 7): handle ADDITION explicitly
        if scenario == "ADDITION":
            edit_wall = time.perf_counter() - t0
            return ArmResult(
                scenario=scenario, intensity=intensity, seed=seed, method=method,
                edit_wall_s=edit_wall, n_affected_ods=0,
                replan_wall_total_s=0, replan_wall_mean_ms=0, replan_wall_median_ms=0,
                n_quality_changed=0, n_quality_better=0, n_quality_worse=0,
                n_became_unreachable=0, n_became_reachable=0,
                median_quality_delta_min=0, max_quality_delta_min=0,
                edit_set_short_id=edit_set.short_id(),
                notes="SKIPPED: raptor_compiled does not support ADDITION at Day-3 (SRP CSR rebuild needs trip stop_sequence not in CSA conns)",
            )
        edited, srp_new, edit_wall, _ = apply_edit_set_to_raptor_compiled(bundle, edit_set)
        srp_csr_used = srp_new if srp_new is not None else srp_csr
        edit_wall_includes = "csa_edit + srp_csr_rebuild"
    elif method == "teeg_alt":
        from Route_Planner.csa.csa import closed_transfers_to_array
        if transfers_idx is None:
            from Route_Planner.csa.csa import _build_transitively_closed_transfers
            transfers_idx = _build_transitively_closed_transfers(
                bundle.transfers_from, bundle.n_stops, max_walk_min=15)
        closed_arr = closed_transfers_to_array(transfers_idx, bundle.n_stops)
        edited, teeg_new, _, _ = apply_edit_set_to_teeg(bundle, edit_set, closed_transfers_from=closed_arr)
        teeg_g = teeg_new
        # Rebuild ALT labels on edited TEEG (part of broadcast-once cost)
        from Route_Planner.teeg.query import build_lower_bound_graph, select_landmarks, LandmarkLabels
        lb = build_lower_bound_graph(teeg_g)
        landmarks = select_landmarks(lb, n_landmarks=16, seed=42)
        alt_labels = LandmarkLabels(lb, landmarks)
        edit_wall = time.perf_counter() - t0
        edit_wall_includes = "csa_edit + teeg_rebuild + alt_landmark_rebuild"
    elif method == "mg_dial":
        edited, mg_new, edit_wall, _ = apply_edit_set_to_mg(bundle, edit_set, cw_csr)
        mg = mg_new
        edit_wall_includes = "csa_edit + minute_grid_rebuild"
    elif method == "ultra":
        edited, ultra_new, edit_wall, _ = apply_edit_set_to_ultra(bundle, edit_set, cw_csr)
        ultra = ultra_new
        edit_wall_includes = "csa_edit + ultra_shortcut_rebuild"
    else:
        raise ValueError(f"unknown method: {method}")

    # Identify affected ODs (Day-4 simplification: ALL)
    affected = compute_affected_ods(edit_set, queries, bundle)

    # Re-query each affected OD on edited bundle/graph + measure per-OD wall
    post_eas: list[Any] = [None] * len(queries)
    per_od_walls: list[float] = []
    for i in affected:
        q = queries.iloc[i]
        s, d, t = int(q["src"]), int(q["dst"]), int(q["t_dep"])
        t1 = time.perf_counter()
        if method == "csa":
            from Route_Planner.csa.csa import csa_earliest_arrival
            ea, _ = csa_earliest_arrival(edited, s, d, t, transfers_idx=edited_transfers_idx)
        elif method == "raptor_compiled":
            from Route_Planner.raptor_compiled import raptor_earliest_arrival_compiled
            if srp_csr_used is None:
                # Cannot run if ADDITION + RAPTOR-compiled (Day-3 limitation)
                ea = None
            else:
                ea = raptor_earliest_arrival_compiled(edited, s, d, t, closed_walks_csr=cw_csr, srp_csr=srp_csr_used)
        elif method == "teeg_alt":
            from Route_Planner.teeg.query import teeg_alt
            ea = teeg_alt(teeg_g, s, d, t, alt_labels)
        elif method == "mg_dial":
            from Route_Planner.minute_grid import minute_grid_dial_earliest_arrival
            ea = minute_grid_dial_earliest_arrival(mg, s, d, t, closed_walks_csr=cw_csr)
        elif method == "ultra":
            from Route_Planner.ultra import ultra_raptor_earliest_arrival
            ea = ultra_raptor_earliest_arrival(edited, s, d, t, ultra)
        per_od_walls.append(time.perf_counter() - t1)
        post_eas[i] = ea

    per_od_ms = np.array([w * 1000 for w in per_od_walls])
    replan_total = float(sum(per_od_walls))
    replan_mean_ms = float(np.mean(per_od_ms)) if len(per_od_ms) > 0 else 0.0
    replan_median_ms = float(np.median(per_od_ms)) if len(per_od_ms) > 0 else 0.0

    # Quality delta (compare post_eas to baseline_eas)
    n_changed = 0; n_better = 0; n_worse = 0
    n_became_unreach = 0; n_became_reach = 0
    deltas = []
    # Normalize post-EAs with pd.isna (per rd_2 fix 10: don't rely on `is None`)
    for i in range(len(queries)):
        baseline = baseline_eas.iloc[i]
        post = post_eas[i]
        b_none = pd.isna(baseline)
        p_none = pd.isna(post) if post is not None else True
        if b_none and p_none:
            continue
        if (not b_none) and p_none:
            n_became_unreach += 1
            n_changed += 1
            n_worse += 1
            continue
        if b_none and (not p_none):
            n_became_reach += 1
            n_changed += 1
            n_better += 1
            continue
        # Both reachable
        b = int(baseline); p = int(post)
        if b != p:
            n_changed += 1
            deltas.append(abs(p - b))
            if p < b:
                n_better += 1
            else:
                n_worse += 1
    median_delta = float(np.median(deltas)) if deltas else 0.0
    max_delta = float(max(deltas)) if deltas else 0.0

    return ArmResult(
        scenario=scenario, intensity=intensity, seed=seed, method=method,
        edit_wall_s=edit_wall,
        n_affected_ods=len(affected),
        replan_wall_total_s=replan_total,
        replan_wall_mean_ms=replan_mean_ms,
        replan_wall_median_ms=replan_median_ms,
        n_quality_changed=n_changed,
        n_quality_better=n_better,
        n_quality_worse=n_worse,
        n_became_unreachable=n_became_unreach,
        n_became_reachable=n_became_reach,
        median_quality_delta_min=median_delta,
        max_quality_delta_min=max_delta,
        edit_set_short_id=edit_set.short_id(),
        edit_wall_includes=edit_wall_includes,
    )


def run_harness(
    bundle, queries: pd.DataFrame, baseline_eas: pd.Series,
    *, scenarios=SCENARIOS, intensities=INTENSITIES, methods=METHODS, seeds=(42,),
    cw_csr=None, ultra=None, mg=None, teeg_g=None, alt_labels=None,
    srp_csr=None, transfers_idx=None,
    progress_cb=None,
):
    """Run all (scenario × intensity × seed × method) arms; return DataFrame of results."""
    results = []
    n_arms = len(scenarios) * len(intensities) * len(seeds) * len(methods)
    i = 0
    for scenario in scenarios:
        for intensity in intensities:
            for seed in seeds:
                for method in methods:
                    i += 1
                    print(f"  [{i}/{n_arms}] {scenario}/N{intensity}/S{seed}/{method} ...", flush=True)
                    t0 = time.time()
                    try:
                        r = run_one_arm(
                            bundle, queries, baseline_eas,
                            scenario, intensity, seed, method,
                            cw_csr=cw_csr, ultra=ultra, mg=mg, teeg_g=teeg_g,
                            alt_labels=alt_labels, srp_csr=srp_csr, transfers_idx=transfers_idx,
                        )
                        results.append(asdict(r))
                        print(f"    edit={r.edit_wall_s*1000:.1f}ms, replan_mean={r.replan_wall_mean_ms:.2f}ms/q, "
                              f"changed={r.n_quality_changed}/{r.n_affected_ods} ({r.n_better}better/{r.n_worse}worse), wall={time.time()-t0:.1f}s")
                    except Exception as e:
                        print(f"    FAILED: {type(e).__name__}: {e}")
                        import traceback; traceback.print_exc()
                        results.append({
                            "scenario": scenario, "intensity": intensity, "seed": seed, "method": method,
                            "edit_wall_s": -1, "n_affected_ods": 0, "replan_wall_total_s": 0,
                            "replan_wall_mean_ms": 0, "replan_wall_median_ms": 0,
                            "n_quality_changed": 0, "n_quality_better": 0, "n_quality_worse": 0,
                            "n_became_unreachable": 0, "n_became_reachable": 0,
                            "median_quality_delta_min": 0, "max_quality_delta_min": 0,
                            "edit_set_short_id": "", "notes": f"FAILED: {e}",
                        })
                    if progress_cb:
                        progress_cb(i, n_arms)
    return pd.DataFrame(results)
