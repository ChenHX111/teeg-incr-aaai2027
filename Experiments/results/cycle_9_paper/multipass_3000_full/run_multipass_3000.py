"""Phase B — run Dibbelt 2018 Alg. 2 (multi-pass CSA) on ALL 3000 Full London queries.

Closes hostile attack #8 ("N=17 has zero statistical power"): we now report
multipass-vs-singlepass disagreement statistics on the full 3000-OD set.

Inputs
------
- Bundle:     MCRaptor_Data_Full/Full_London/MCRaptor_Data
- Queries:    Experiments/results/cycle_9_paper/stage_5_full_london/queries_3000.csv
- Alg.1 EAs:  Experiments/results/cycle_9_paper/stage_5_full_london/results_3000_oracle.csv

Outputs (this dir)
------------------
- results_3000.csv        — per-OD: q_id, src, dst, t_dep, alg1_ea, alg2_ea, delta, wall_s
- summary.json            — totals, residual rate, delta stats
- run.log                 — wall log (flush after each line)
- run.err                 — exceptions
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parent
LOG = OUT_DIR / "run.log"
ERR = OUT_DIR / "run.err"
RESULTS_CSV = OUT_DIR / "results_3000.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"

FULL_LONDON = Path(r"C:\Users\a-chenhaoxue\Documents\Design_Novel_Transport_Graph\MCRaptor_Data_Full\Full_London\MCRaptor_Data")
QUERIES_3000 = REPO / "Experiments" / "results" / "cycle_9_paper" / "stage_5_full_london" / "queries_3000.csv"
ALG1_RESULTS = REPO / "Experiments" / "results" / "cycle_9_paper" / "stage_5_full_london" / "results_3000_oracle.csv"


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    _log(f"PID={os.getpid()} cwd={os.getcwd()}")
    _log(f"REPO={REPO}")
    _log(f"FULL_LONDON exists: {FULL_LONDON.exists()}")
    _log(f"QUERIES_3000 exists: {QUERIES_3000.exists()}")
    _log(f"ALG1_RESULTS exists: {ALG1_RESULTS.exists()}")

    import numpy as np
    import pandas as pd
    from Route_Planner.reference_oracle.parser import load_ref_bundle
    from Route_Planner.reference_oracle.footpath_closure import transitive_closure_f15
    from Route_Planner.reference_oracle.csa_multipass import (
        csa_multipass_earliest_arrival, _closure_to_csr,
    )
    _log("imports OK")

    q_all = pd.read_csv(QUERIES_3000)
    alg1 = pd.read_csv(ALG1_RESULTS)
    alg1_map = dict(zip(alg1["q_id"].astype(int), alg1["ea_ref"]))
    _log(f"queries: {len(q_all)} rows, alg1 keys: {len(alg1_map)}")

    t0 = time.time()
    bundle = load_ref_bundle(str(FULL_LONDON))
    _log(f"bundle loaded in {time.time()-t0:.2f}s: {bundle.n_stops} stops, "
         f"{len(bundle.connections)} conns, {len(bundle.walks)} walks, "
         f"{bundle.n_trips} trips")

    t1 = time.time()
    closed = transitive_closure_f15(bundle.walks, bundle.n_stops, max_walk_min=15)
    _log(f"F_15 closure in {time.time()-t1:.2f}s: {len(closed)} non-empty sources")

    cw_indptr, cw_indices, cw_wmins = _closure_to_csr(closed, bundle.n_stops)
    closed_walks_csr = (cw_indptr, cw_indices, cw_wmins)
    _log(f"CSR closure: {len(cw_indices)} edges")

    conns = bundle.connections
    dep = np.ascontiguousarray(conns["dep_min"], dtype=np.int32)
    arr = np.ascontiguousarray(conns["arr_min"], dtype=np.int32)
    fr_ = np.ascontiguousarray(conns["from_stop"], dtype=np.int32)
    to_ = np.ascontiguousarray(conns["to_stop"], dtype=np.int32)
    tr = np.ascontiguousarray(conns["trip_id_int"], dtype=np.int32)
    conn_arrays = (dep, arr, fr_, to_, tr)

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["q_id", "src", "dst", "t_dep",
                    "alg1_ea", "alg2_ea", "delta_min", "wall_s"])

    _log("JIT warmup ...")
    t_jit = time.time()
    _ = csa_multipass_earliest_arrival(
        bundle, src=int(q_all.iloc[0]["src"]), dst=int(q_all.iloc[0]["dst"]),
        t_dep=int(q_all.iloc[0]["t_dep"]),
        closed_walks_csr=closed_walks_csr, conn_arrays=conn_arrays,
    )
    _log(f"JIT warmup done in {time.time()-t_jit:.2f}s")

    n = len(q_all)
    n_done = 0
    n_disagree = 0
    deltas = []
    t_start_sweep = time.time()
    for idx, row in q_all.iterrows():
        q_id = int(row["q_id"])
        src = int(row["src"])
        dst = int(row["dst"])
        t_dep = int(row["t_dep"])
        t_q = time.time()
        try:
            alg2 = csa_multipass_earliest_arrival(
                bundle, src=src, dst=dst, t_dep=t_dep,
                closed_walks_csr=closed_walks_csr, conn_arrays=conn_arrays,
            )
        except Exception as e:
            _log(f"q_id={q_id} FAILED: {type(e).__name__}: {e}")
            alg2 = None
        wall = time.time() - t_q
        alg1_ea = alg1_map.get(q_id)
        if alg2 is not None and alg1_ea is not None and not pd.isna(alg1_ea):
            delta = int(alg2) - int(alg1_ea)
        else:
            delta = None
        if delta is not None and delta != 0:
            n_disagree += 1
            deltas.append(delta)

        with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow([
                q_id, src, dst, t_dep,
                "" if alg1_ea is None or pd.isna(alg1_ea) else int(alg1_ea),
                "" if alg2 is None else int(alg2),
                "" if delta is None else delta,
                f"{wall:.3f}",
            ])
        n_done += 1
        if n_done % 200 == 0:
            elap = time.time() - t_start_sweep
            rate = n_done / elap if elap > 0 else 0
            eta = (n - n_done) / rate if rate > 0 else float('inf')
            _log(f"progress: {n_done}/{n} ({100*n_done/n:.1f}%) "
                 f"disagree={n_disagree} rate={rate:.1f} q/s eta={eta:.0f}s")

    _log(f"SWEEP DONE in {time.time()-t_start_sweep:.1f}s")

    summary = {
        "n_queries": n,
        "n_completed": n_done,
        "n_disagree": n_disagree,
        "disagree_rate_pct": round(100.0 * n_disagree / n_done if n_done else 0, 3),
        "delta_min_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_min_min": min(deltas) if deltas else 0,
        "delta_min_max": max(deltas) if deltas else 0,
        "sweep_wall_s": round(time.time() - t_start_sweep, 2),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"SUMMARY: {summary}")
    _log(f"DONE. results: {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        with ERR.open("a", encoding="utf-8") as f:
            f.write(f"{type(e).__name__}: {e}\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(2)
