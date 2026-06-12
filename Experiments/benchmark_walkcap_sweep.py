"""Walk-cap sweep on Full London: vary `max_walk_min` ∈ {5, 10, 15, 20, 30} minutes
across 4 key methods (CSA, RAPTOR-compiled, TEEG-ALT-v2, HL) on the 3000-OD bench.

This tests robustness of each method's reachability + speed to walking cap.
TEEG-ALT-v2 expected to be more robust due to event-graph + ALT.

Each cap requires a full bundle/TEEG rebuild since closures change.
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
from Route_Planner.csa_compiled import csa_earliest_arrival_compiled
from Route_Planner.raptor_compiled import raptor_earliest_arrival_compiled, build_stop_route_position_csr
from Route_Planner.hl_compiled import hl_earliest_arrival_compiled
from Route_Planner.hl.labelling import build_stop_graph_from_connections
from Route_Planner.hl import build_hl_labels_stop_level
from Route_Planner.teeg.builder import build_teeg
from Route_Planner.teeg.query import build_lower_bound_graph
from Route_Planner.teeg_alt_v2 import (
    LandmarkLabelsV2, select_landmarks_v2, teeg_alt_v2_compiled,
)


def _to_none_or_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    try: return int(v)
    except Exception: return None


def run_cap(b_args, max_walk_min, queries, out_dir):
    """One walk-cap iteration. Rebuild closures + TEEG."""
    print(f"\n{'='*60}\n=== max_walk_min = {max_walk_min} ===\n{'='*60}", flush=True)

    t0 = time.time()
    b = load_bundle(t_lo=0, t_hi=780, load_raptor=True)
    print(f"  bundle: {len(b.connections):,} conns ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    transfers_idx = _build_transitively_closed_transfers(b.transfers_from, b.n_stops, max_walk_min=max_walk_min)
    closed_arr = closed_transfers_to_array(transfers_idx, b.n_stops)
    cw_csr = build_closed_walk_csr(transfers_idx, b.n_stops)
    cw_tuple = (
        np.ascontiguousarray(cw_csr[0], dtype=np.int32),
        np.ascontiguousarray(cw_csr[1], dtype=np.int32),
        np.ascontiguousarray(cw_csr[2], dtype=np.int32),
    )
    conns = b.connections
    csa_conn = (
        np.ascontiguousarray(conns["dep_min"], dtype=np.int32),
        np.ascontiguousarray(conns["arr_min"], dtype=np.int32),
        np.ascontiguousarray(conns["from_stop"], dtype=np.int32),
        np.ascontiguousarray(conns["to_stop"], dtype=np.int32),
        np.ascontiguousarray(conns["trip_id"], dtype=np.int32),
    )
    print(f"  closures: {len(cw_csr[1]):,} walks ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    srp_csr = build_stop_route_position_csr(b)
    print(f"  SRP CSR ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    g = build_teeg(b, verbose=False, closed_transfers_from=closed_arr)
    print(f"  TEEG: {g.n_nodes:,} nodes / {g.n_edges:,} edges ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    lb = build_lower_bound_graph(g)
    landmarks = select_landmarks_v2(lb, n_landmarks=32, n_restarts=4, seed=42)
    labels = LandmarkLabelsV2(lb, landmarks)
    print(f"  ALT-v2 ({time.time()-t0:.0f}s)", flush=True)

    # HL
    t0 = time.time()
    try:
        sg_indptr, sg_indices, sg_weights = build_stop_graph_from_connections(
            b.connections, b.transfers_from, b.n_stops, verbose=False)
        hl_index = build_hl_labels_stop_level(sg_indptr, sg_indices, sg_weights, b.n_stops, verbose=False)
        hl_index.conn_from = b.connections["from_stop"]
        hl_index.conn_to = b.connections["to_stop"]
        hl_index.conn_dep = b.connections["dep_min"]
        hl_index.conn_arr = b.connections["arr_min"]
        hl_index.conn_trip = b.connections["trip_id"]
        walk_rows = []
        for s in range(b.n_stops):
            for v, w in transfers_idx.get(s, []):
                walk_rows.append((s, v, w))
        if walk_rows:
            w_arr = np.array(walk_rows, dtype=np.int32)
            order = np.argsort(w_arr[:, 0])
            w_arr = w_arr[order]
            cnt = np.bincount(w_arr[:, 0], minlength=b.n_stops + 1).astype(np.int32)
            hl_index.walk_indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)[:b.n_stops + 1]
            hl_index.walk_indices = w_arr[:, 1].astype(np.int32)
            hl_index.walk_weights = w_arr[:, 2].astype(np.int32)
        hl_ok = True
        print(f"  HL ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  HL FAILED: {e}; skipping", flush=True)
        hl_index = None; hl_ok = False

    # Warm up
    qw = queries.iloc[0]
    s0, d0, t0_q = int(qw["src"]), int(qw["dst"]), int(qw["t_dep"])
    _ = csa_earliest_arrival_compiled(b, s0, d0, t0_q, closed_walks_csr=cw_tuple, conn_arrays=csa_conn)
    _ = raptor_earliest_arrival_compiled(b, s0, d0, t0_q, closed_walks_csr=cw_csr, srp_csr=srp_csr)
    _ = teeg_alt_v2_compiled(g, s0, d0, t0_q, labels)
    if hl_ok:
        _ = hl_earliest_arrival_compiled(hl_index, s0, d0, t0_q)

    print(f"  Running {len(queries)} queries × 4 methods ...", flush=True)
    rows = []
    last_print = time.time()
    for i, qr in queries.iterrows():
        q_id, s, d, t = int(qr.q_id), int(qr.src), int(qr.dst), int(qr.t_dep)
        t0 = time.perf_counter()
        e_csa = csa_earliest_arrival_compiled(b, s, d, t, closed_walks_csr=cw_tuple, conn_arrays=csa_conn)
        ms_csa = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        e_rap = raptor_earliest_arrival_compiled(b, s, d, t, closed_walks_csr=cw_csr, srp_csr=srp_csr)
        ms_rap = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        e_v2 = teeg_alt_v2_compiled(g, s, d, t, labels)
        ms_v2 = (time.perf_counter() - t0) * 1000
        if hl_ok:
            t0 = time.perf_counter()
            try: e_hl = hl_earliest_arrival_compiled(hl_index, s, d, t)
            except: e_hl = None
            ms_hl = (time.perf_counter() - t0) * 1000
        else:
            e_hl, ms_hl = None, 0.0
        rows.append(dict(
            q_id=q_id, src=s, dst=d, t_dep=t, max_walk_min=max_walk_min,
            ea_csa=_to_none_or_int(e_csa), ea_raptor=_to_none_or_int(e_rap),
            ea_teeg_alt_v2=_to_none_or_int(e_v2), ea_hl=_to_none_or_int(e_hl),
            ms_csa=ms_csa, ms_raptor=ms_rap, ms_teeg_alt_v2=ms_v2, ms_hl=ms_hl,
        ))
        if time.time() - last_print > 30:
            print(f"    [{i+1}/{len(queries)}]", flush=True); last_print = time.time(); gc.collect()

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"walkcap_{max_walk_min:02d}min.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--caps", default="5,10,15,20,30")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(args.queries)

    all_rows = []
    for cap in caps:
        df = run_cap(None, cap, queries, out_dir)
        all_rows.append(df)
        gc.collect()
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / "walkcap_combined.csv", index=False)
    print(f"\nALL CAPS DONE: {len(combined)} total rows → {out_dir / 'walkcap_combined.csv'}")


if __name__ == "__main__":
    main()
