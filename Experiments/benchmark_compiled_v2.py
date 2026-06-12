"""benchmark_compiled_v2.py — Run ONLY the 3 v2 methods on the same OD list as
benchmark_compiled_only.py. Output merges with the POSTFIX CSV for analysis.

Methods:
- teeg_alt_v2 (Ext 1 — tightened ALT with GH active landmarks + push-time f-pruning)
- mg_dial_v2 (Ext 2 — K=15 lane-aware Dial; defaults to K=infinity for parity, K=15 if BENCH_K env set)
- teeg_alt_incr (Ext 3 — TEEG incremental, no edits applied for this static bench)
"""
from __future__ import annotations
import os, sys, time, gc, argparse
from pathlib import Path
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS)
sys.path.insert(0, REPO)

from Route_Planner.common import load_bundle
from Route_Planner.csa.csa import (
    _build_transitively_closed_transfers, build_closed_walk_csr, closed_transfers_to_array,
)
from Route_Planner.teeg.builder import build_teeg
from Route_Planner.teeg.query import build_lower_bound_graph, select_landmarks, LandmarkLabels
from Route_Planner.minute_grid import (
    build_minute_grid, save_minute_grid, load_minute_grid,
)
# v2 methods
from Route_Planner.teeg_alt_v2 import (
    LandmarkLabelsV2, select_landmarks_v2, teeg_alt_v2_compiled,
)
from Route_Planner.mg_dial_v2 import build_minute_grid_v2, MGDialV2
from Route_Planner.teeg_incremental import IncrementalState, build_state
from Route_Planner.teeg_incremental import query_compiled as teeg_incr_query_compiled


def _to_none_or_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(v)
    except Exception:
        return None


def run_v2_bench(queries: pd.DataFrame, out_path: str,
                  bench_k: int | None = None):
    print(f"[bench-v2] loading bundle ...", flush=True)
    t0 = time.time()
    b = load_bundle(t_lo=0, t_hi=780, load_raptor=True)
    print(f"  bundle: {len(b.connections):,} conns, {b.n_stops:,} stops ({time.time()-t0:.1f}s)", flush=True)

    print(f"[bench-v2] closed walks + CSR ...", flush=True)
    t0 = time.time()
    transfers_idx = _build_transitively_closed_transfers(b.transfers_from, b.n_stops, max_walk_min=15)
    closed_arr = closed_transfers_to_array(transfers_idx, b.n_stops)
    cw_csr = build_closed_walk_csr(transfers_idx, b.n_stops)
    print(f"  done ({time.time()-t0:.1f}s; {len(cw_csr[1]):,} walks)", flush=True)

    print(f"[bench-v2] building TEEG ...", flush=True)
    t0 = time.time()
    g = build_teeg(b, verbose=False, closed_transfers_from=closed_arr)
    print(f"  TEEG: {g.n_nodes:,} nodes / {g.n_edges:,} edges ({time.time()-t0:.0f}s)", flush=True)

    print(f"[bench-v2] preprocessing v1 ALT landmarks ...", flush=True)
    t0 = time.time()
    lb = build_lower_bound_graph(g)
    landmarks = select_landmarks(lb, n_landmarks=16, seed=42)
    labels = LandmarkLabels(lb, landmarks)
    print(f"  v1 ALT: {time.time()-t0:.1f}s", flush=True)

    print(f"[bench-v2] preprocessing v2 ALT landmarks (multi-restart Maxmin, n=32) ...", flush=True)
    t0 = time.time()
    landmarks_v2 = select_landmarks_v2(lb, n_landmarks=32, n_restarts=4, seed=42)
    labels_v2 = LandmarkLabelsV2(lb, landmarks_v2)
    print(f"  v2 ALT: {time.time()-t0:.1f}s", flush=True)

    print(f"[bench-v2] loading/building minute-grid ...", flush=True)
    from Route_Planner.common.data_layer import get_data_tag
    data_tag = get_data_tag()
    mg_npz = os.path.join(REPO, "Graph_Builder", f"minute_grid_full_day_{data_tag}.npz")
    if os.path.exists(mg_npz):
        t0 = time.time(); mg = load_minute_grid(mg_npz); print(f"  loaded MG ({time.time()-t0:.1f}s)", flush=True)
    else:
        t0 = time.time(); mg = build_minute_grid(b, cw_csr, verbose=False)
        save_minute_grid(mg, mg_npz); print(f"  built MG ({time.time()-t0:.1f}s)", flush=True)

    print(f"[bench-v2] building MinuteGridV2 (with shadow nodes for CONTINUE) ...", flush=True)
    t0 = time.time()
    mg_v2 = build_minute_grid_v2(b, cw_csr, verbose=False)
    print(f"  MG-v2: {time.time()-t0:.0f}s", flush=True)

    print(f"[bench-v2] instantiating MGDialV2 runner ...", flush=True)
    mgd_v2 = MGDialV2(mg_v2, plan="C")  # F11 verdict: Plan C

    print(f"[bench-v2] building TEEGIncremental state ...", flush=True)
    t0 = time.time()
    incr_state = build_state(g)
    print(f"  Incr state: {time.time()-t0:.0f}s", flush=True)

    # Warm up
    print(f"[bench-v2] JIT warm-up ...", flush=True)
    qw = queries.iloc[0]
    s0, d0, t0_q = int(qw["src"]), int(qw["dst"]), int(qw["t_dep"])
    _ = teeg_alt_v2_compiled(g, s0, d0, t0_q, labels_v2)
    _ = mgd_v2.query(s0, d0, t0_q, closed_walks_csr=cw_csr, K=bench_k)
    _ = teeg_incr_query_compiled(incr_state, g, s0, d0, t0_q, labels)
    print(f"  warm-up complete", flush=True)

    print(f"[bench-v2] {len(queries)} queries × 3 v2 methods ...", flush=True)
    rows = []
    last_print = time.time()
    for i, qr in queries.iterrows():
        q_id, s, d, t = int(qr.q_id), int(qr.src), int(qr.dst), int(qr.t_dep)

        t0 = time.perf_counter()
        e_v2 = teeg_alt_v2_compiled(g, s, d, t, labels_v2)
        ms_v2 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        e_mgv2 = mgd_v2.query(s, d, t, closed_walks_csr=cw_csr, K=bench_k)
        ms_mgv2 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        e_incr = teeg_incr_query_compiled(incr_state, g, s, d, t, labels)
        ms_incr = (time.perf_counter() - t0) * 1000

        rows.append(dict(
            q_id=q_id, src=s, dst=d, t_dep=t,
            ea_teeg_alt_v2=_to_none_or_int(e_v2),
            ea_mg_dial_v2=_to_none_or_int(e_mgv2),
            ea_teeg_alt_incr=_to_none_or_int(e_incr),
            ms_teeg_alt_v2=ms_v2,
            ms_mg_dial_v2=ms_mgv2,
            ms_teeg_alt_incr=ms_incr,
        ))

        if time.time() - last_print > 10:
            print(f"  [{i+1}/{len(queries)}] in progress ...", flush=True)
            last_print = time.time()
            gc.collect()
        if (i + 1) % 200 == 0:
            df_chk = pd.DataFrame(rows)
            df_chk.to_csv(out_path, index=False)
            print(f"  checkpoint at q={i+1}", flush=True)

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[bench-v2] DONE: {len(df)} rows → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=None, help="MG-Dial-v2 K bound (None=unbounded for parity)")
    args = ap.parse_args()
    queries = pd.read_csv(args.queries)
    run_v2_bench(queries, args.out, bench_k=args.K)


if __name__ == "__main__":
    main()
