"""Run multi-pass CSA Algorithm 2 on Berlin + NYC 500-OD subsets.

Answers iter-8 critic Q6: "is the 185-residual multi-pass blind spot
London-specific or general across cities?"
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
from Route_Planner.reference_oracle.parser import load_ref_bundle
from Route_Planner.reference_oracle.footpath_closure import transitive_closure_f15
from Route_Planner.reference_oracle.csa_core import csa_earliest_arrival
from Route_Planner.reference_oracle.csa_multipass import (
    csa_multipass_earliest_arrival, _closure_to_csr,
)

CITIES = [
    ("berlin", REPO / "Cities" / "Berlin" / "MCRaptor_Data_Berlin",
                REPO / "Cities" / "Berlin" / "queries_berlin_3000.csv"),
    ("nyc",    REPO / "MCRaptor_Data_NYC",
                REPO / "Cities" / "NYC" / "queries_nyc_3000.csv"),
]
N_QUERIES = 500


def run_city(name, bundle_dir, queries_csv):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    bundle = load_ref_bundle(str(bundle_dir))
    print(f"  bundle {time.time()-t0:.1f}s: {bundle.n_stops} stops, "
          f"{len(bundle.connections)} conns, {len(bundle.walks)} walks", flush=True)
    t1 = time.time()
    closed = transitive_closure_f15(bundle.walks, bundle.n_stops, max_walk_min=15)
    print(f"  closure {time.time()-t1:.1f}s: {len(closed)} sources", flush=True)
    cw_indptr, cw_indices, cw_wmins = _closure_to_csr(closed, bundle.n_stops)
    closed_walks_csr = (cw_indptr, cw_indices, cw_wmins)
    conns = bundle.connections
    dep = np.ascontiguousarray(conns["dep_min"], dtype=np.int32)
    arr = np.ascontiguousarray(conns["arr_min"], dtype=np.int32)
    fr_ = np.ascontiguousarray(conns["from_stop"], dtype=np.int32)
    to_ = np.ascontiguousarray(conns["to_stop"], dtype=np.int32)
    tr  = np.ascontiguousarray(conns["trip_id_int"], dtype=np.int32)
    conn_arrays = (dep, arr, fr_, to_, tr)

    queries = pd.read_csv(queries_csv).head(N_QUERIES)
    q0 = queries.iloc[0]
    _ = csa_multipass_earliest_arrival(bundle, src=int(q0["src"]), dst=int(q0["dst"]),
        t_dep=int(q0["t_dep"]), closed_walks_csr=closed_walks_csr,
        conn_arrays=conn_arrays)

    rows = []; n_disagree = 0; deltas = []
    t_sweep = time.time()
    for idx, row in queries.iterrows():
        q_id = int(row["q_id"]); src = int(row["src"]); dst = int(row["dst"])
        t_dep = int(row["t_dep"])
        try: alg1 = csa_earliest_arrival(bundle, src, dst, t_dep, closed_walks=closed)
        except: alg1 = None
        try:
            alg2 = csa_multipass_earliest_arrival(bundle, src=src, dst=dst,
                t_dep=t_dep, closed_walks_csr=closed_walks_csr,
                conn_arrays=conn_arrays)
        except: alg2 = None
        delta = None
        if alg1 is not None and alg2 is not None:
            delta = int(alg2) - int(alg1)
            if delta < 0:
                n_disagree += 1; deltas.append(delta)
        rows.append({"q_id": q_id, "src": src, "dst": dst, "t_dep": t_dep,
                      "alg1_ea": alg1, "alg2_ea": alg2, "delta_min": delta})

    pd.DataFrame(rows).to_csv(OUT_DIR / f"results_{name}.csv", index=False)
    summary = {
        "city": name, "n_queries": len(queries), "n_disagree": n_disagree,
        "disagree_rate_pct": round(100.0 * n_disagree / len(queries), 3),
        "delta_mean": float(np.mean(deltas)) if deltas else 0,
        "delta_median": float(np.median(deltas)) if deltas else 0,
        "delta_min": int(min(deltas)) if deltas else 0,
        "delta_max": int(max(deltas)) if deltas else 0,
        "sweep_wall_s": round(time.time() - t_sweep, 2),
    }
    print(f"  SUMMARY: {summary}", flush=True)
    return summary


def main():
    all_summaries = []
    for name, bundle, queries in CITIES:
        all_summaries.append(run_city(name, bundle, queries))
    (OUT_DIR / "summary.json").write_text(json.dumps(all_summaries, indent=2),
                                          encoding="utf-8")
    print("\nDONE — wrote summary.json")


if __name__ == "__main__":
    main()
