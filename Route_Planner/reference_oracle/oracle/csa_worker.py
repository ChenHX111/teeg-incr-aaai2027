"""Reference-CSA* subprocess oracle worker.

Designed for COLD-CACHE isolated invocation:
  python -I -B -m Route_Planner.reference_oracle.oracle.csa_worker \
      --data-dir <path> --queries <queries.csv> \
      --results <out.csv> --manifest <out.json>

Each invocation:
1. Cold-imports the reference_oracle package (no shared sys.modules).
2. Loads the RefBundle from --data-dir (independent parser).
3. Builds F_15 transitively-closed walks (independent multi-source Dijkstra).
4. Reads queries CSV (must have columns: q_id, src, dst, t_dep).
5. Runs csa_earliest_arrival per query; collects (q_id, ea_ref).
6. Writes results.csv (canonical UTF-8, LF line endings, sorted by q_id).
7. Writes manifest.json with: PID, hostname, walltime, bundle SHA, query SHA,
   results SHA, oracle version.

DETERMINISM CONTRACT (Day-4 gate): two cold-cache invocations on the same
(data_dir, queries.csv) MUST produce bitwise-identical results.csv.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Cold-import the package (no relative imports — invoked via -m)
from Route_Planner.reference_oracle.parser import load_ref_bundle
from Route_Planner.reference_oracle.footpath_closure import transitive_closure_f15
from Route_Planner.reference_oracle.csa_core import csa_earliest_arrival
from Route_Planner.reference_oracle.validate import validate_bundle, sha256_file, sha256_ndarray


ORACLE_VERSION = "ref_csa_star_v0_2_0_cycle9_day4"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical_results_str(results: list[dict]) -> str:
    """Canonical CSV string: sorted by q_id, LF line endings, no BOM, ea_ref="None" or int."""
    sorted_rows = sorted(results, key=lambda r: int(r["q_id"]))
    lines = ["q_id,src,dst,t_dep,ea_ref"]
    for r in sorted_rows:
        ea = "None" if r["ea_ref"] is None else str(int(r["ea_ref"]))
        lines.append(f"{int(r['q_id'])},{int(r['src'])},{int(r['dst'])},{int(r['t_dep'])},{ea}")
    return "\n".join(lines) + "\n"


def run(data_dir: str, queries_csv: str, results_out: str, manifest_out: str,
        max_walk_min: int = 15, max_queries: int | None = None) -> dict:
    pid = os.getpid()
    hostname = socket.gethostname()
    t0 = time.time()

    # Load bundle
    bundle = load_ref_bundle(data_dir)
    prov = validate_bundle(bundle)
    t_load = time.time() - t0
    print(f"[oracle] PID={pid} bundle loaded in {t_load:.2f}s: {prov['n_stops']} stops, {prov['n_connections']} conns, {prov['n_walks']} walks", flush=True)

    # Close walks
    t1 = time.time()
    closed = transitive_closure_f15(bundle.walks, bundle.n_stops, max_walk_min=max_walk_min)
    t_close = time.time() - t1
    print(f"[oracle] PID={pid} F_15 closure in {t_close:.2f}s: {len(closed)} non-empty sources", flush=True)

    # Read queries
    queries = pd.read_csv(queries_csv)
    required = {"q_id", "src", "dst", "t_dep"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"queries.csv missing columns: {missing}")
    if max_queries is not None:
        queries = queries.head(max_queries)
    print(f"[oracle] PID={pid} loaded {len(queries)} queries", flush=True)

    # Run
    t2 = time.time()
    results = []
    for _, q in queries.iterrows():
        ea = csa_earliest_arrival(bundle, int(q["src"]), int(q["dst"]), int(q["t_dep"]), closed_walks=closed)
        results.append({
            "q_id": int(q["q_id"]),
            "src": int(q["src"]),
            "dst": int(q["dst"]),
            "t_dep": int(q["t_dep"]),
            "ea_ref": ea,  # int or None
        })
    t_run = time.time() - t2
    print(f"[oracle] PID={pid} {len(results)} queries in {t_run:.2f}s ({t_run/len(results)*1000:.1f} ms/q)", flush=True)

    # Write results (canonical)
    results_str = _canonical_results_str(results)
    Path(results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(results_out).write_text(results_str, encoding="utf-8", newline="")
    results_sha = _sha256_str(results_str)
    print(f"[oracle] PID={pid} wrote results SHA-256={results_sha}", flush=True)

    # Compute provenance SHAs
    bundle_sha = sha256_ndarray(bundle.connections)
    walks_sha = sha256_ndarray(bundle.walks)
    queries_sha = sha256_file(queries_csv)

    # Write manifest
    manifest = {
        "oracle_version": ORACLE_VERSION,
        "role": "oracle",
        "method": "reference_csa_star",
        "pid": pid,
        "hostname": hostname,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "data_dir": str(data_dir),
        "queries_csv": str(queries_csv),
        "results_out": str(results_out),
        "n_stops": prov["n_stops"],
        "n_connections": prov["n_connections"],
        "n_walks_raw": prov["n_walks"],
        "n_closed_sources": len(closed),
        "n_queries": len(queries),
        "max_walk_min": max_walk_min,
        "service_date": prov["service_date"],
        "bundle_connections_sha256": bundle_sha,
        "bundle_walks_sha256": walks_sha,
        "queries_csv_sha256": queries_sha,
        "results_sha256": results_sha,
        "walltime_load_s": t_load,
        "walltime_closure_s": t_close,
        "walltime_run_s": t_run,
        "walltime_total_s": time.time() - t0,
        "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
    }
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[oracle] PID={pid} wrote manifest. Total wall: {manifest['walltime_total_s']:.2f}s", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to MCRaptor_Data_* directory.")
    ap.add_argument("--queries", required=True, help="Path to queries CSV (cols: q_id, src, dst, t_dep).")
    ap.add_argument("--results", required=True, help="Output path for results CSV.")
    ap.add_argument("--manifest", required=True, help="Output path for manifest JSON.")
    ap.add_argument("--max-walk-min", type=int, default=15, help="F_15 cap (default 15).")
    ap.add_argument("--max-queries", type=int, default=None, help="Limit query count (debug).")
    args = ap.parse_args()
    try:
        run(args.data_dir, args.queries, args.results, args.manifest,
            max_walk_min=args.max_walk_min, max_queries=args.max_queries)
    except Exception as e:
        # Surface error clearly to subprocess parent
        import traceback
        sys.stderr.write(f"[oracle PID={os.getpid()}] FAILED: {type(e).__name__}: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
