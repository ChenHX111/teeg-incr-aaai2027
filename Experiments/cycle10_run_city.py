r"""Cycle 10 -- TEEG-INCR dynamic-update experiments (parameterized by city).

Same 28-cell design as the London Cycle 10 runner. Reads city-specific
bundle location + queries CSV from CLI; writes results into a city-specific
RESULTS directory under cycle_10_dynamic_updates/.
See Experiments/cycle10_run_dynamic_updates.py for the original London runner.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-query-subset", type=int, default=300)
    return p.parse_args()


ARGS = _parse_args()
DATA_DIR = str(Path(ARGS.data_dir).resolve())
QUERIES_CSV = Path(ARGS.queries).resolve()
OUT_DIR = Path(ARGS.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_QUERY_SUBSET = ARGS.n_query_subset
CITY = ARGS.city

# Configure data dir BEFORE importing route_planner modules that use it
os.environ["MCRAPTOR_DATA_DIR"] = DATA_DIR

from Route_Planner.common import load_bundle  # noqa: E402
from Route_Planner.csa.csa import (  # noqa: E402
    _build_transitively_closed_transfers,
    closed_transfers_to_array,
)
from Route_Planner.teeg.builder import build_teeg  # noqa: E402
from Route_Planner.teeg.query import (  # noqa: E402
    LandmarkLabels, build_lower_bound_graph, select_landmarks,
)
from Route_Planner.teeg_incremental import TEEGIncremental  # noqa: E402

LOG_PATH = OUT_DIR / "run.log"
CSV_PATH = OUT_DIR / "cycle10_results.csv"
PER_QUERY_PATH = OUT_DIR / "cycle10_per_query.csv"
META_PATH = OUT_DIR / "cycle10_meta.json"

SCENARIOS = [
    *(("delay", n, m) for n in (1, 10, 100, 1000) for m in (1, 5, 10)),
    *(("cancel", n, 0) for n in (1, 10, 100, 1000)),
    *(("add", n, m) for n in (1, 10, 100, 1000) for m in (1, 5, 10)),
]

OVERLAY_CAP_NODES = 200_000
OVERLAY_CAP_EDGES = 2_000_000


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class CellResult:
    scenario: str
    N: int
    magnitude_min: int
    update_ms_total: float
    update_ms_per_edit_p50: float
    update_ms_per_edit_p99: float
    n_capacity_overflow: int
    query_ms_median: float
    query_ms_p95: float
    n_queries_run: int
    n_reachable: int
    n_match_baseline: int
    n_changed_vs_baseline: int
    overlay_n_nodes_after: int
    overlay_n_edges_after: int
    dead_edge_frac: float


def _select_trips_uniform(rng: np.random.Generator, n_trips: int, k: int) -> np.ndarray:
    k = min(k, n_trips)
    return rng.choice(n_trips, size=k, replace=False)


def _run_queries(inc: TEEGIncremental, queries: pd.DataFrame):
    latencies, eas = [], []
    for _, r in queries.iterrows():
        s, d, t = int(r["src"]), int(r["dst"]), int(r["t_dep"])
        t0 = time.perf_counter()
        ea = inc.query(s, d, t)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        eas.append(ea)
    return latencies, eas


def _cell(inc, queries, scenario, N, mag, rng, baseline_eas):
    n_trips = len(inc.state.trip_first_node) - 1
    if scenario == "delay":
        trips = _select_trips_uniform(rng, n_trips, N)
        edits = [("DELAY", int(t), int(mag)) for t in trips]
    elif scenario == "cancel":
        trips = _select_trips_uniform(rng, n_trips, N)
        edits = [("CANCEL", int(t)) for t in trips]
    elif scenario == "add":
        trips = _select_trips_uniform(rng, n_trips, N)
        edits = [("ADD", int(t), int(mag)) for t in trips]
    else:
        raise ValueError(scenario)

    t0 = time.perf_counter()
    stats = inc.apply_edits(edits, track_per_edit=True)
    update_ms_total = (time.perf_counter() - t0) * 1000.0
    per_edit_us = stats.per_edit_latencies_us or [0.0]
    per_edit_ms = [x / 1000.0 for x in per_edit_us]
    p50 = float(np.percentile(per_edit_ms, 50))
    p99 = float(np.percentile(per_edit_ms, 99))
    n_overflow = 1 if stats.capacity_overflow else 0

    latencies, eas = _run_queries(inc, queries)
    q_med = float(np.median(latencies))
    q_p95 = float(np.percentile(latencies, 95))
    n_reach = sum(1 for e in eas if e is not None)
    n_match = sum(1 for a, b in zip(eas, baseline_eas) if a == b)
    n_changed = sum(1 for a, b in zip(eas, baseline_eas) if a != b)
    dead_frac = (inc.state.dead_edge_count / max(1, inc.state.n_edges))

    return CellResult(
        scenario=scenario, N=N, magnitude_min=mag,
        update_ms_total=update_ms_total,
        update_ms_per_edit_p50=p50, update_ms_per_edit_p99=p99,
        n_capacity_overflow=n_overflow,
        query_ms_median=q_med, query_ms_p95=q_p95,
        n_queries_run=len(latencies),
        n_reachable=n_reach,
        n_match_baseline=n_match,
        n_changed_vs_baseline=n_changed,
        overlay_n_nodes_after=int(inc.state.overlay_n_nodes),
        overlay_n_edges_after=int(inc.state.overlay_n_edges),
        dead_edge_frac=float(dead_frac),
    ), latencies, eas


def main():
    _log("=" * 70)
    _log(f"Cycle 10 — TEEG-INCR dynamic-update experiments — CITY={CITY}")
    _log(f"  data:    {DATA_DIR}")
    _log(f"  queries: {QUERIES_CSV}")
    _log(f"  out:     {OUT_DIR}")
    _log(f"  scenarios: {len(SCENARIOS)}, n_query_subset={N_QUERY_SUBSET}")
    _log("=" * 70)

    t_rebuild_total0 = time.perf_counter()

    _log("[base] loading bundle ...")
    t0 = time.perf_counter()
    b = load_bundle(t_lo=0, t_hi=1440, load_raptor=True)
    bundle_s = time.perf_counter() - t0
    _log(f"  bundle in {bundle_s:.1f}s — {len(b.connections):,} conns, n_stops={b.n_stops:,}")

    _log("[base] transitively closing footpaths ...")
    t0 = time.perf_counter()
    transfers_idx = _build_transitively_closed_transfers(
        b.transfers_from, b.n_stops, max_walk_min=15)
    closed_arr = closed_transfers_to_array(transfers_idx, b.n_stops)
    walks_s = time.perf_counter() - t0
    _log(f"  walks in {walks_s:.1f}s")

    _log("[base] building TEEG ...")
    t0 = time.perf_counter()
    g = build_teeg(b, verbose=False, closed_transfers_from=closed_arr)
    teeg_s = time.perf_counter() - t0
    _log(f"  TEEG in {teeg_s:.1f}s — n_nodes={g.n_nodes:,} n_edges={g.n_edges:,}")

    _log("[base] ALT preprocessing (16 landmarks) ...")
    t0 = time.perf_counter()
    lb = build_lower_bound_graph(g)
    landmarks = select_landmarks(lb, n_landmarks=16, seed=42)
    labels = LandmarkLabels(lb, landmarks)
    alt_s = time.perf_counter() - t0
    _log(f"  ALT in {alt_s:.1f}s")

    rebuild_total_s = time.perf_counter() - t_rebuild_total0
    _log(f"[base] FULL REBUILD WALL = {rebuild_total_s:.1f}s ({rebuild_total_s/60:.1f} min)")

    _log("[incr] building TEEGIncremental state ...")
    t0 = time.perf_counter()
    inc = TEEGIncremental(
        g, labels=labels,
        overlay_capacity_nodes=OVERLAY_CAP_NODES,
        overlay_capacity_edges=OVERLAY_CAP_EDGES,
        auto_compact=False,
    )
    _log(f"  state built in {time.perf_counter()-t0:.1f}s")

    queries_all = pd.read_csv(QUERIES_CSV)
    queries = (queries_all.head(N_QUERY_SUBSET).reset_index(drop=True).copy()
               if N_QUERY_SUBSET < len(queries_all)
               else queries_all.reset_index(drop=True).copy())
    _log(f"  loaded {len(queries)} queries")

    _log("[incr] warming JIT ...")
    t0 = time.perf_counter()
    inc.apply_edits([("CANCEL", 0)])
    inc.apply_edits([("ADD", 1, 5)])
    inc.apply_edits([("DELAY", 2, 3)])
    q0 = queries.iloc[0]
    inc.query(int(q0["src"]), int(q0["dst"]), int(q0["t_dep"]))
    inc.query(int(q0["src"]), int(q0["dst"]), int(q0["t_dep"]), dijkstra=True)
    _log(f"  warm in {time.perf_counter()-t0:.1f}s")
    del inc; gc.collect()

    _log("[base] re-building fresh state for no-edit baseline ...")
    t0 = time.perf_counter()
    inc = TEEGIncremental(
        g, labels=labels,
        overlay_capacity_nodes=OVERLAY_CAP_NODES,
        overlay_capacity_edges=OVERLAY_CAP_EDGES,
        auto_compact=False,
    )
    _log(f"  rebuilt in {time.perf_counter()-t0:.1f}s")

    _log(f"[base] running {len(queries)}-OD baseline queries (no edits) ...")
    t0 = time.perf_counter()
    baseline_latencies, baseline_eas = _run_queries(inc, queries)
    bench_s = time.perf_counter() - t0
    base_med = float(np.median(baseline_latencies))
    base_p95 = float(np.percentile(baseline_latencies, 95))
    base_reach = sum(1 for e in baseline_eas if e is not None)
    _log(f"  baseline: median={base_med:.1f}ms p95={base_p95:.1f}ms "
         f"reachable={base_reach}/{len(queries)} ({100*base_reach/len(queries):.1f}%) "
         f"wall={bench_s:.0f}s")

    results: List[CellResult] = [CellResult(
        scenario="baseline", N=0, magnitude_min=0,
        update_ms_total=0.0,
        update_ms_per_edit_p50=0.0, update_ms_per_edit_p99=0.0,
        n_capacity_overflow=0,
        query_ms_median=base_med, query_ms_p95=base_p95,
        n_queries_run=len(baseline_latencies),
        n_reachable=base_reach,
        n_match_baseline=len(baseline_eas),
        n_changed_vs_baseline=0,
        overlay_n_nodes_after=0, overlay_n_edges_after=0,
        dead_edge_frac=0.0,
    )]
    per_query_rows: list[dict] = []
    for i, lat in enumerate(baseline_latencies):
        per_query_rows.append({
            "scenario": "baseline", "N": 0, "magnitude_min": 0,
            "qid": i, "latency_ms": lat,
            "ea": baseline_eas[i] if baseline_eas[i] is not None else -1,
        })
    del inc; gc.collect()

    rng = np.random.default_rng(2026_06_09)
    fresh_state_s = 0.0
    for k, (scenario, N, mag) in enumerate(SCENARIOS, start=1):
        _log(f"--- cell {k}/{len(SCENARIOS)}: scenario={scenario} N={N} mag={mag}min ---")
        t0 = time.perf_counter()
        inc = TEEGIncremental(
            g, labels=labels,
            overlay_capacity_nodes=OVERLAY_CAP_NODES,
            overlay_capacity_edges=OVERLAY_CAP_EDGES,
            auto_compact=False,
        )
        fresh_state_s = time.perf_counter() - t0
        _log(f"  fresh state in {fresh_state_s:.1f}s")

        try:
            cell, latencies, eas = _cell(inc, queries, scenario, N, mag, rng, baseline_eas)
        except Exception as e:
            _log(f"  !! cell FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            del inc; gc.collect()
            continue

        results.append(cell)
        _log(f"  update_total={cell.update_ms_total:.1f}ms "
             f"p50={cell.update_ms_per_edit_p50:.3f}ms p99={cell.update_ms_per_edit_p99:.3f}ms "
             f"overflow={cell.n_capacity_overflow}")
        _log(f"  query: median={cell.query_ms_median:.1f}ms p95={cell.query_ms_p95:.1f}ms "
             f"reachable={cell.n_reachable}/{cell.n_queries_run} "
             f"match={cell.n_match_baseline} changed={cell.n_changed_vs_baseline}")

        for i, lat in enumerate(latencies):
            per_query_rows.append({
                "scenario": scenario, "N": N, "magnitude_min": mag,
                "qid": i, "latency_ms": lat,
                "ea": eas[i] if eas[i] is not None else -1,
            })

        pd.DataFrame([asdict(r) for r in results]).to_csv(CSV_PATH, index=False)
        pd.DataFrame(per_query_rows).to_csv(PER_QUERY_PATH, index=False)
        del inc; gc.collect()

    meta = {
        "city": CITY,
        "data_dir": DATA_DIR,
        "n_query_subset": N_QUERY_SUBSET,
        "n_queries_total": int(len(queries)),
        "n_scenarios": len(SCENARIOS),
        "rebuild_baseline": {
            "bundle_s": bundle_s, "walks_s": walks_s, "teeg_s": teeg_s, "alt_s": alt_s,
            "total_s": rebuild_total_s, "total_ms": rebuild_total_s * 1000.0,
        },
        "fresh_state_s_first_cell": fresh_state_s,
        "overlay_cap_nodes": OVERLAY_CAP_NODES,
        "overlay_cap_edges": OVERLAY_CAP_EDGES,
        "n_query_baseline": N_QUERY_SUBSET,
        "baseline_median_ms": base_med,
        "baseline_p95_ms": base_p95,
        "baseline_reachable": base_reach,
        "g_n_nodes": int(g.n_nodes),
        "g_n_edges": int(g.n_edges),
        "g_n_stops": int(g.n_stops),
        "n_trips": int(len(g.node_trip) // 2) if hasattr(g, "node_trip") else None,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log(f"meta -> {META_PATH}")
    _log(f"csv  -> {CSV_PATH}")
    _log("DONE.")


if __name__ == "__main__":
    main()
