r"""Cycle 10 CSA-Rebuild Oracle — parameterised by city.

Same logic as Experiments/cycle10_rebuild_oracle.py (the London-only original)
but with --city / --data-dir / --queries / --out-dir CLI args, mirroring how
cycle10_run_city.py parameterises cycle10_run_dynamic_updates.py.

Cells, RNG seed, query subset size, overlay caps are identical to the London
run so that the Berlin/NYC results are directly comparable.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

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

os.environ["MCRAPTOR_DATA_DIR"] = DATA_DIR

from Route_Planner.common import load_bundle  # noqa: E402
from Route_Planner.csa.csa import (  # noqa: E402
    _build_transitively_closed_transfers,
    closed_transfers_to_array,
)
from Route_Planner.csa_compiled.core import (  # noqa: E402
    build_closed_walk_csr_from_dict,
    _csa_core_njit,
    INF_I32,
)
from Route_Planner.teeg.builder import build_teeg  # noqa: E402
from Route_Planner.teeg.query import (  # noqa: E402
    LandmarkLabels, build_lower_bound_graph, select_landmarks,
)
from Route_Planner.teeg_incremental import TEEGIncremental  # noqa: E402

OVERLAY_CAP_NODES = 200_000
OVERLAY_CAP_EDGES = 2_000_000

CELLS = [
    ("delay", 1, 1),
    ("delay", 1000, 5),
    ("cancel", 1000, 0),
    ("delay", 10, 5),
    ("delay", 100, 5),
    ("cancel", 1, 0),
    ("cancel", 10, 0),
    ("cancel", 100, 0),
    ("add", 1, 5),
    ("add", 10, 5),
    ("add", 100, 5),
    ("add", 1000, 5),
]

LOG_PATH = OUT_DIR / "run.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _select_trips_uniform(rng, n_trips, k):
    k = min(k, n_trips)
    return rng.choice(n_trips, size=k, replace=False)


def _make_edits(scenario, N, mag, rng, n_trips):
    trips = _select_trips_uniform(rng, n_trips, N)
    if scenario == "delay":
        return [("DELAY", int(t), int(mag)) for t in trips], trips
    elif scenario == "cancel":
        return [("CANCEL", int(t)) for t in trips], trips
    elif scenario == "add":
        return [("ADD", int(t), int(mag)) for t in trips], trips
    else:
        raise ValueError(scenario)


def _apply_edits_to_connections(conns_orig, scenario, trips, mag, trip_id_str):
    """Apply edits at the timetable (connection-array) level.

    Returns a NEW connection array reflecting the post-edit timetable, re-sorted
    by dep_min so the compiled CSA's binary-search start index is well-defined.
    """
    conns = conns_orig.copy()

    if scenario == "delay":
        for t in trips:
            mask = conns["trip_id"] == t
            conns["dep_min"][mask] = (conns["dep_min"][mask].astype(np.int32) + mag).astype(np.int16)
            conns["arr_min"][mask] = (conns["arr_min"][mask].astype(np.int32) + mag).astype(np.int16)
    elif scenario == "cancel":
        mask = np.ones(len(conns), dtype=bool)
        for t in trips:
            mask &= (conns["trip_id"] != t)
        conns = conns[mask]
    elif scenario == "add":
        new_rows = []
        max_trip_id = int(conns["trip_id"].max()) + 1
        for i, t in enumerate(trips):
            tmask = conns_orig["trip_id"] == t
            template_conns = conns_orig[tmask].copy()
            template_conns["dep_min"] = (template_conns["dep_min"].astype(np.int32) + mag).astype(np.int16)
            template_conns["arr_min"] = (template_conns["arr_min"].astype(np.int32) + mag).astype(np.int16)
            template_conns["trip_id"] = np.int32(max_trip_id + i)
            new_rows.append(template_conns)
        if new_rows:
            conns = np.concatenate([conns] + new_rows)

    order = np.argsort(conns["dep_min"], kind="stable")
    conns = conns[order]
    return conns


def _run_csa_on_modified(conns_modified, queries, n_stops, n_trips_total, cw_csr):
    dep_i32 = np.ascontiguousarray(conns_modified["dep_min"], dtype=np.int32)
    arr_i32 = np.ascontiguousarray(conns_modified["arr_min"], dtype=np.int32)
    fr_i32 = np.ascontiguousarray(conns_modified["from_stop"], dtype=np.int32)
    to_i32 = np.ascontiguousarray(conns_modified["to_stop"], dtype=np.int32)
    tr_i32 = np.ascontiguousarray(conns_modified["trip_id"], dtype=np.int32)

    cw_indptr, cw_indices, cw_wmins = cw_csr
    if cw_indptr.shape[0] < n_stops + 1:
        pad = np.full(n_stops + 1 - cw_indptr.shape[0], cw_indptr[-1], dtype=np.int32)
        cw_indptr = np.concatenate([cw_indptr, pad]).astype(np.int32)

    n_trips_arr = int(tr_i32.max()) + 1 if len(tr_i32) > 0 else n_trips_total

    eas = []
    for _, r in queries.iterrows():
        src, dst, t_dep = int(r["src"]), int(r["dst"]), int(r["t_dep"])
        start = int(np.searchsorted(dep_i32, t_dep, side="left"))
        result = _csa_core_njit(
            np.int32(src), np.int32(dst), np.int32(t_dep),
            np.int32(n_stops), np.int32(n_trips_arr),
            dep_i32, arr_i32, fr_i32, to_i32, tr_i32,
            np.int32(start),
            cw_indptr, cw_indices, cw_wmins,
        )
        if int(result) >= int(INF_I32) or int(result) >= 32_000:
            eas.append(None)
        else:
            eas.append(int(result))
    return eas


def _run_teeg_incr_queries(inc, queries):
    eas = []
    for _, r in queries.iterrows():
        src, dst, t_dep = int(r["src"]), int(r["dst"]), int(r["t_dep"])
        ea = inc.query(src, dst, t_dep)
        eas.append(ea)
    return eas


def main():
    _log("=" * 70)
    _log(f"Cycle 10 CSA-Rebuild Oracle — CITY={CITY}")
    _log(f"  data:    {DATA_DIR}")
    _log(f"  queries: {QUERIES_CSV}")
    _log(f"  out:     {OUT_DIR}")
    _log(f"  cells:   {len(CELLS)}, n_query_subset={N_QUERY_SUBSET}")
    _log("=" * 70)

    _log("[base] loading bundle ...")
    t0 = time.perf_counter()
    b = load_bundle(t_lo=0, t_hi=1440, load_raptor=True)
    _log(f"  bundle in {time.perf_counter()-t0:.1f}s — {len(b.connections):,} conns, n_stops={b.n_stops:,}")

    _log("[base] transitively closing footpaths ...")
    t0 = time.perf_counter()
    transfers_idx = _build_transitively_closed_transfers(
        b.transfers_from, b.n_stops, max_walk_min=15)
    cw_csr = build_closed_walk_csr_from_dict(transfers_idx, b.n_stops)
    _log(f"  walks in {time.perf_counter()-t0:.1f}s")
    closed_arr = closed_transfers_to_array(transfers_idx, b.n_stops)

    _log("[base] building TEEG ...")
    t0 = time.perf_counter()
    g = build_teeg(b, verbose=False, closed_transfers_from=closed_arr)
    _log(f"  TEEG in {time.perf_counter()-t0:.1f}s — n_nodes={g.n_nodes:,} n_edges={g.n_edges:,}")

    _log("[base] ALT preprocessing (16 landmarks) ...")
    t0 = time.perf_counter()
    lb = build_lower_bound_graph(g)
    landmarks = select_landmarks(lb, n_landmarks=16, seed=42)
    labels = LandmarkLabels(lb, landmarks)
    _log(f"  ALT in {time.perf_counter()-t0:.1f}s")

    _log("[jit] warming up CSA + TEEG-INCR ...")
    t0 = time.perf_counter()
    inc_warmup = TEEGIncremental(
        g, labels=labels,
        overlay_capacity_nodes=OVERLAY_CAP_NODES,
        overlay_capacity_edges=OVERLAY_CAP_EDGES,
        auto_compact=False,
    )
    inc_warmup.apply_edits([("CANCEL", 0)])
    inc_warmup.apply_edits([("ADD", 1, 5)])
    inc_warmup.apply_edits([("DELAY", 2, 3)])

    queries_all = pd.read_csv(QUERIES_CSV)
    queries = queries_all.head(N_QUERY_SUBSET).reset_index(drop=True).copy()
    _log(f"  loaded {len(queries)} queries")

    q0 = queries.iloc[0]
    inc_warmup.query(int(q0["src"]), int(q0["dst"]), int(q0["t_dep"]))
    _run_csa_on_modified(b.connections[:100], queries.head(1), b.n_stops,
                         len(b.trip_id_str), cw_csr)
    del inc_warmup; gc.collect()
    _log(f"  warmup in {time.perf_counter()-t0:.1f}s")

    n_trips = len(b.trip_id_str)
    rng = np.random.default_rng(2026_06_09)
    oracle_rows = []
    summary_rows = []
    oracle_csv = OUT_DIR / "oracle_results.csv"
    summary_csv = OUT_DIR / "summary.csv"

    for cell_idx, (scenario, N, mag) in enumerate(CELLS):
        cell_name = f"{scenario}_N{N}_mag{mag}"
        _log(f"\n--- Cell {cell_idx+1}/{len(CELLS)}: {cell_name} ---")

        edits, trips = _make_edits(scenario, N, mag, rng, n_trips)

        _log(f"  [teeg-incr] applying {len(edits)} edits ...")
        t0 = time.perf_counter()
        inc = TEEGIncremental(
            g, labels=labels,
            overlay_capacity_nodes=OVERLAY_CAP_NODES,
            overlay_capacity_edges=OVERLAY_CAP_EDGES,
            auto_compact=False,
        )
        stats = inc.apply_edits(edits, track_per_edit=True)
        _log(f"    edits applied in {(time.perf_counter()-t0)*1000:.0f}ms "
             f"(overflow={stats.capacity_overflow})")

        _log(f"  [teeg-incr] querying {len(queries)} OD pairs ...")
        t0 = time.perf_counter()
        teeg_eas = _run_teeg_incr_queries(inc, queries)
        _log(f"    queries in {time.perf_counter()-t0:.1f}s")
        del inc; gc.collect()

        _log(f"  [csa-rebuild] modifying timetable ...")
        t0 = time.perf_counter()
        conns_modified = _apply_edits_to_connections(
            b.connections, scenario, trips, mag, b.trip_id_str)
        _log(f"    timetable modified in {(time.perf_counter()-t0)*1000:.0f}ms "
             f"({len(conns_modified):,} conns)")

        _log(f"  [csa-rebuild] running CSA on {len(queries)} OD pairs ...")
        t0 = time.perf_counter()
        csa_eas = _run_csa_on_modified(
            conns_modified, queries, b.n_stops, n_trips + N, cw_csr)
        csa_wall = time.perf_counter() - t0
        _log(f"    CSA done in {csa_wall:.1f}s")
        del conns_modified; gc.collect()

        n_match = 0
        n_teeg_better = 0
        n_csa_better = 0
        n_both_none = 0
        deltas = []

        for qid in range(len(queries)):
            t_ea = teeg_eas[qid]
            c_ea = csa_eas[qid]

            if t_ea is None and c_ea is None:
                verdict = "both_unreachable"
                delta = 0
                n_both_none += 1
                n_match += 1
            elif t_ea is None and c_ea is not None:
                verdict = "csa_better"
                delta = 9999
                n_csa_better += 1
            elif t_ea is not None and c_ea is None:
                verdict = "teeg_better"
                delta = -9999
                n_teeg_better += 1
            elif t_ea == c_ea:
                verdict = "match"
                delta = 0
                n_match += 1
            elif t_ea < c_ea:
                verdict = "teeg_better"
                delta = t_ea - c_ea
                n_teeg_better += 1
            else:
                verdict = "csa_better"
                delta = t_ea - c_ea
                n_csa_better += 1

            deltas.append(delta)
            oracle_rows.append({
                "cell": cell_name,
                "q_id": qid,
                "csa_rebuild_ea": c_ea if c_ea is not None else -1,
                "teeg_incr_ea": t_ea if t_ea is not None else -1,
                "delta": delta,
                "verdict": verdict,
            })

        match_rate = n_match / len(queries) * 100.0
        abs_deltas = [abs(d) for d in deltas if d != 0 and abs(d) < 9000]
        mean_abs_delta = float(np.mean(abs_deltas)) if abs_deltas else 0.0
        max_abs_delta = max(abs_deltas) if abs_deltas else 0
        p95_abs_delta = float(np.percentile(abs_deltas, 95)) if abs_deltas else 0.0

        summary_rows.append({
            "city": CITY,
            "cell": cell_name,
            "scenario": scenario,
            "N": N,
            "magnitude": mag,
            "n_queries": len(queries),
            "n_match": n_match,
            "match_rate_pct": match_rate,
            "n_teeg_better": n_teeg_better,
            "n_csa_better": n_csa_better,
            "n_both_unreachable": n_both_none,
            "mean_abs_delta_min": mean_abs_delta,
            "max_abs_delta_min": max_abs_delta,
            "p95_abs_delta_min": p95_abs_delta,
        })

        _log(f"  RESULT: match={n_match}/{len(queries)} ({match_rate:.1f}%) "
             f"teeg_better={n_teeg_better} csa_better={n_csa_better}")
        if abs_deltas:
            _log(f"    |delta| stats: mean={mean_abs_delta:.1f} max={max_abs_delta} p95={p95_abs_delta:.1f}")

        # Incremental save after every cell so partial progress is preserved
        pd.DataFrame(oracle_rows).to_csv(oracle_csv, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    _log("\n" + "=" * 70)
    _log("Saving final per-city summary/verdict ...")

    total_csa_better = sum(r["n_csa_better"] for r in summary_rows)
    total_teeg_better = sum(r["n_teeg_better"] for r in summary_rows)
    total_queries = sum(r["n_queries"] for r in summary_rows)

    with open(OUT_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Cycle 10 CSA-Rebuild Oracle — {CITY} Summary\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**City:** {CITY}\n")
        f.write(f"**Queries:** {N_QUERY_SUBSET} OD pairs ({QUERIES_CSV.name}, stratified head)\n")
        f.write(f"**RNG seed:** 2026_06_09 (identical to cycle10_run_city.py)\n")
        f.write(f"**Cells:** {len(CELLS)} (delay/cancel/add × N∈{{1,10,100,1000}})\n\n")
        f.write("## Per-Cell Results\n\n")
        f.write("| Cell | Match Rate | TEEG-better | CSA-better | Mean |Δ| | Max |Δ| |\n")
        f.write("|------|-----------|-------------|------------|---------|--------|\n")
        for row in summary_rows:
            f.write(f"| {row['cell']} | {row['match_rate_pct']:.1f}% "
                    f"| {row['n_teeg_better']} | {row['n_csa_better']} "
                    f"| {row['mean_abs_delta_min']:.1f} | {row['max_abs_delta_min']} |\n")
        f.write(f"\n**Total CSA-better: {total_csa_better} / {total_queries} "
                f"({100*total_csa_better/total_queries:.2f}%)**\n")
        f.write(f"**Total TEEG-better: {total_teeg_better} / {total_queries} "
                f"({100*total_teeg_better/total_queries:.2f}%)**\n")

    with open(OUT_DIR / "VERDICT.md", "w", encoding="utf-8") as f:
        f.write(f"# Cycle 10 CSA-Rebuild Oracle — {CITY} VERDICT\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"- **{len(CELLS)} cells tested** "
                f"(delay/cancel/add × N∈{{1,10,100,1000}}), "
                f"{N_QUERY_SUBSET} queries each "
                f"= {total_queries} evaluations\n")
        f.write(f"- **CSA-better cases: {total_csa_better} / {total_queries} "
                f"({100*total_csa_better/total_queries:.2f}%)**\n")
        f.write(f"- **TEEG-better cases: {total_teeg_better} / {total_queries} "
                f"({100*total_teeg_better/total_queries:.2f}%)**\n")
        f.write("\n## Detailed Counts\n\n")
        f.write("| Cell | Queries | Match | TEEG-better | CSA-better | Max |Δ| |\n")
        f.write("|------|---------|-------|-------------|------------|---------|\n")
        for row in summary_rows:
            f.write(f"| {row['cell']} | {row['n_queries']} | {row['n_match']} "
                    f"| {row['n_teeg_better']} | {row['n_csa_better']} "
                    f"| {row['max_abs_delta_min']} |\n")

    _log(f"DONE. Wrote {OUT_DIR / 'oracle_results.csv'}, summary.csv, summary.md, VERDICT.md")


if __name__ == "__main__":
    main()
