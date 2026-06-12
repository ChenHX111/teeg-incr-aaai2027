"""benchmark_compiled_only.py — Run only the 9 Numba-JIT compiled methods.

This is the FAIR-COMPARISON benchmark: every method now runs through Numba JIT
(not just compiled RAPTOR), so timing differences reflect algorithm, not Python
overhead.

Reads the same query CSV as `benchmark.py` and saves a parallel results CSV with
`ea_*_compiled` and `ms_*_compiled` columns. The two CSVs can be merged on q_id
for side-by-side comparison.

Methods (all newly compiled this cycle unless noted):
- csa_compiled         (98.7× vs pure CSA)
- raptor_compiled      (374-400× vs pure RAPTOR — already shipped P0-D)
- teeg_alt_compiled    (11.6× vs pure TEEG-ALT)
- teeg_d_compiled      (16.6× vs pure TEEG-D)
- teeg_b_compiled      (16.5× vs pure TEEG-B)
- mg_dial_compiled     (17.7× vs pure MG-Dial)
- mg_dij_compiled      (9.6× vs pure MG-Dij)
- ultra_compiled       (483× vs pure ULTRA — same kernels as raptor_compiled)
- ch_compiled          (8.8× vs pure CH)
- hl_compiled          (158.8× vs pure HL)

Notes:
- TEEG-Dscipy intentionally NOT compiled (scipy.sparse.csgraph is already C).
- STATIC is excluded (it's a "wrong baseline" — not relevant for fair compare).
- Both MG-Dij implementations exist (in minute_grid_compiled/ and mg_dij_compiled/);
  we use the dedicated mg_dij_compiled per the user's two-team request.

Usage:
    python -m Experiments.benchmark_compiled_only \
        --queries Experiments/results/cycle_6/stage_5_design_experiments/queries_3000_stratified.csv \
        --out Experiments/results/cycle_9_paper/p1a_fullldn_bench/bench_3000_fullldn_compiled.csv
"""
from __future__ import annotations
import os, sys, time, gc, argparse
from pathlib import Path
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS)
sys.path.insert(0, REPO)

# ============================================================================
# Imports (lazy-style)
# ============================================================================
from Route_Planner.common import load_bundle
from Route_Planner.csa.csa import (
    _build_transitively_closed_transfers, build_closed_walk_csr,
)
# Pure-Python TEEG (only used for graph build + ALT label preprocessing).
from Route_Planner.teeg.builder import build_teeg
from Route_Planner.teeg.query import (
    build_lower_bound_graph, select_landmarks, LandmarkLabels,
)
# Compiled methods.
from Route_Planner.csa_compiled import (
    csa_earliest_arrival_compiled, build_closed_walk_csr_from_dict,
)
from Route_Planner.raptor_compiled import (
    raptor_earliest_arrival_compiled, build_stop_route_position_csr,
)
from Route_Planner.teeg_compiled import teeg_alt_compiled
from Route_Planner.teeg_d_compiled import teeg_dijkstra_compiled as teeg_d_compiled
from Route_Planner.teeg_b_compiled import teeg_bidirectional_compiled as teeg_b_compiled
from Route_Planner.minute_grid_compiled import (
    minute_grid_dial_earliest_arrival_compiled as mg_dial_compiled,
)
from Route_Planner.mg_dij_compiled import (
    minute_grid_dijkstra_earliest_arrival_compiled as mg_dij_compiled,
)
from Route_Planner.ultra_compiled import ultra_raptor_earliest_arrival_compiled
from Route_Planner.ch_compiled import ch_earliest_arrival_compiled
from Route_Planner.hl_compiled import hl_earliest_arrival_compiled

# Build helpers.
from Route_Planner.minute_grid import (
    build_minute_grid, save_minute_grid, load_minute_grid,
)
from Route_Planner.ultra import (
    build_ultra_shortcuts, save_ultra_shortcuts_npz, load_ultra_shortcuts_npz,
)
from Route_Planner.ultra.sidecar import CacheStaleness
from Route_Planner.ch import build_ch_from_teeg
from Route_Planner.hl.labelling import build_stop_graph_from_connections
from Route_Planner.hl import build_hl_labels_stop_level


# ============================================================================
# Helpers
# ============================================================================
def _to_none_or_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _now_ms():
    return time.perf_counter() * 1000.0


# ============================================================================
# Main benchmark
# ============================================================================
def run_compiled_bench(queries: pd.DataFrame, out_path: str,
                       skip: set[str] | None = None):
    skip = skip or set()

    print(f"[bench-compiled] loading bundle ...", flush=True)
    t0 = time.time()
    b = load_bundle(t_lo=0, t_hi=780, load_raptor=True)
    print(f"  bundle in {time.time()-t0:.1f}s — {len(b.connections):,} conns, "
          f"{b.n_stops:,} stops", flush=True)

    # Closed walks CSR (shared by CSA, ULTRA via raptor_compiled, etc.)
    print(f"[bench-compiled] building closed-walks CSR ...", flush=True)
    t0 = time.time()
    transfers_idx = _build_transitively_closed_transfers(
        b.transfers_from, b.n_stops, max_walk_min=15)
    cw_csr = build_closed_walk_csr(transfers_idx, b.n_stops)
    # Re-pack to (indptr, indices, wmins) tuple form expected by compiled CSA.
    cw_csr_tuple = (
        np.ascontiguousarray(cw_csr[0], dtype=np.int32),
        np.ascontiguousarray(cw_csr[1], dtype=np.int32),
        np.ascontiguousarray(cw_csr[2], dtype=np.int32),
    )
    n_walks = len(cw_csr_tuple[1])
    print(f"  closed walks: {n_walks:,} edges ({time.time()-t0:.1f}s)", flush=True)

    # Pre-extract connection arrays for compiled CSA (one-shot)
    print(f"[bench-compiled] preparing CSA connection arrays ...", flush=True)
    conns = b.connections
    csa_conn_arrays = (
        np.ascontiguousarray(conns["dep_min"], dtype=np.int32),
        np.ascontiguousarray(conns["arr_min"], dtype=np.int32),
        np.ascontiguousarray(conns["from_stop"], dtype=np.int32),
        np.ascontiguousarray(conns["to_stop"], dtype=np.int32),
        np.ascontiguousarray(conns["trip_id"], dtype=np.int32),
    )

    # Stop-route-position CSR for compiled RAPTOR/ULTRA
    srp_csr = None
    if "raptor" not in skip or "ultra" not in skip:
        print(f"[bench-compiled] building stop-route-position CSR ...", flush=True)
        t0 = time.time()
        try:
            srp_csr = build_stop_route_position_csr(b)
            print(f"  SRP CSR in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  SRP CSR FAILED: {e}; auto-skipping raptor + ultra", flush=True)
            skip.add("raptor"); skip.add("ultra")
            srp_csr = None
    else:
        print(f"[bench-compiled] SRP CSR skipped (raptor+ultra both in skip)", flush=True)

    # Build TEEG (needed for TEEG-D/B/ALT, CH)
    print(f"[bench-compiled] building TEEG (with closed walks) ...", flush=True)
    t0 = time.time()
    from Route_Planner.csa.csa import closed_transfers_to_array
    closed_arr = closed_transfers_to_array(transfers_idx, b.n_stops)
    g = build_teeg(b, verbose=False, closed_transfers_from=closed_arr)
    print(f"  TEEG in {time.time()-t0:.1f}s — {g.n_nodes:,} nodes, "
          f"{g.n_edges:,} edges", flush=True)

    # ALT landmarks
    print(f"[bench-compiled] preprocessing ALT landmarks ...", flush=True)
    t0 = time.time()
    lb = build_lower_bound_graph(g)
    landmarks = select_landmarks(lb, n_landmarks=16, seed=42)
    labels = LandmarkLabels(lb, landmarks)
    print(f"  ALT prep in {time.time()-t0:.1f}s", flush=True)

    # ULTRA (load cache or build)
    from Route_Planner.common.data_layer import get_data_tag
    data_tag = get_data_tag()
    ultra_cache_path = os.path.join(REPO, "Experiments", "results", "cycle_6",
                                     "cache", f"ultra_shortcuts_{data_tag}_v1.npz")
    walk_params = {"max_walk_min": 15, "walking_speed_ms": 1.4}
    ultra = None
    if os.path.exists(ultra_cache_path):
        try:
            print(f"[bench-compiled] loading ULTRA shortcuts from cache ...", flush=True)
            t0 = time.time()
            ultra_csr = load_ultra_shortcuts_npz(ultra_cache_path, b=b,
                                                  walk_params=walk_params, validate=True)
            from Route_Planner.ultra.preprocess import UltraShortcuts
            ultra = UltraShortcuts(csr=ultra_csr)
            print(f"  ULTRA loaded in {time.time()-t0:.1f}s "
                  f"({ultra_csr.n_edges:,} shortcuts)", flush=True)
        except (CacheStaleness, FileNotFoundError, AssertionError) as e:
            print(f"  cache stale: {e}; rebuilding ...", flush=True)
            ultra = None
    if ultra is None:
        print(f"[bench-compiled] building ULTRA shortcuts ...", flush=True)
        t0 = time.time()
        ultra = build_ultra_shortcuts(b, cw_csr, walk_params=walk_params, verbose=True)
        save_ultra_shortcuts_npz(ultra_cache_path, ultra.csr)
        print(f"  ULTRA built in {time.time()-t0:.1f}s "
              f"({ultra.n_edges:,} shortcuts)", flush=True)

    # Minute-grid
    mg_npz = os.path.join(REPO, "Graph_Builder", f"minute_grid_full_day_{data_tag}.npz")
    if os.path.exists(mg_npz):
        print(f"[bench-compiled] loading minute-grid from cache ...", flush=True)
        t0 = time.time()
        mg = load_minute_grid(mg_npz)
        print(f"  MG loaded in {time.time()-t0:.1f}s "
              f"({mg.n_nodes:,} nodes, {mg.n_edges:,} edges)", flush=True)
    else:
        print(f"[bench-compiled] building minute-grid ...", flush=True)
        t0 = time.time()
        mg = build_minute_grid(b, cw_csr, verbose=False)
        save_minute_grid(mg, mg_npz)
        print(f"  MG built in {time.time()-t0:.1f}s", flush=True)

    # CH (skip on failure)
    ch_graph = None
    ch_ok = False
    if "ch" not in skip:
        print(f"[bench-compiled] building CH-on-TEEG ...", flush=True)
        t0 = time.time()
        try:
            ch_graph = build_ch_from_teeg(g, mode="trivial")
            ch_ok = True
            print(f"  CH in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  CH build FAILED: {e}; skipping", flush=True)

    # HL (skip on failure)
    hl_index = None
    hl_ok = False
    if "hl" not in skip:
        print(f"[bench-compiled] building HL labels ...", flush=True)
        t0 = time.time()
        try:
            sg_indptr, sg_indices, sg_weights = build_stop_graph_from_connections(
                b.connections, b.transfers_from, b.n_stops, verbose=False)
            hl_index = build_hl_labels_stop_level(
                sg_indptr, sg_indices, sg_weights, b.n_stops, verbose=False)
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
                hl_index.walk_indptr = np.concatenate(
                    [[0], np.cumsum(cnt)]).astype(np.int32)[:b.n_stops + 1]
                hl_index.walk_indices = w_arr[:, 1].astype(np.int32)
                hl_index.walk_weights = w_arr[:, 2].astype(np.int32)
            hl_ok = True
            print(f"  HL in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"  HL build FAILED: {e}", flush=True)
            traceback.print_exc()

    # ========================================================================
    # JIT WARM-UP (untimed)
    # ========================================================================
    print(f"[bench-compiled] JIT warm-up (one query per compiled method) ...", flush=True)
    if len(queries) > 0:
        qw = queries.iloc[0]
        s0, d0, t0_q = int(qw["src"]), int(qw["dst"]), int(qw["t_dep"])
        _ = csa_earliest_arrival_compiled(
            b, s0, d0, t0_q, closed_walks_csr=cw_csr_tuple, conn_arrays=csa_conn_arrays)
        if "raptor" not in skip and srp_csr is not None:
            _ = raptor_earliest_arrival_compiled(
                b, s0, d0, t0_q, closed_walks_csr=cw_csr, srp_csr=srp_csr)
        _ = teeg_alt_compiled(g, s0, d0, t0_q, labels)
        _ = teeg_d_compiled(g, s0, d0, t0_q)
        _ = teeg_b_compiled(g, s0, d0, t0_q)
        _ = mg_dial_compiled(mg, s0, d0, t0_q, closed_walks_csr=cw_csr)
        _ = mg_dij_compiled(mg, s0, d0, t0_q, closed_walks_csr=cw_csr)
        if "ultra" not in skip and srp_csr is not None:
            _ = ultra_raptor_earliest_arrival_compiled(
                b, s0, d0, t0_q, ultra, srp_csr=srp_csr)
        if ch_ok:
            try: _ = ch_earliest_arrival_compiled(ch_graph, g, s0, d0, t0_q)
            except Exception as e: print(f"  CH warm-up failed: {e}"); ch_ok = False
        if hl_ok:
            try: _ = hl_earliest_arrival_compiled(hl_index, s0, d0, t0_q)
            except Exception as e: print(f"  HL warm-up failed: {e}"); hl_ok = False
    print(f"  warm-up complete", flush=True)

    # ========================================================================
    # Main loop
    # ========================================================================
    n_methods = 9 + (1 if ch_ok else 0) + (1 if hl_ok else 0) - 0
    print(f"[bench-compiled] {len(queries)} queries × {n_methods} compiled methods ...",
          flush=True)

    rows = []
    last_print = time.time()
    for i, qr in queries.iterrows():
        q_id, s, d, t = int(qr.q_id), int(qr.src), int(qr.dst), int(qr.t_dep)

        # CSA compiled
        t0 = time.perf_counter()
        e_csac = csa_earliest_arrival_compiled(
            b, s, d, t, closed_walks_csr=cw_csr_tuple, conn_arrays=csa_conn_arrays)
        ms_csac = (time.perf_counter() - t0) * 1000

        # RAPTOR compiled
        if "raptor" not in skip and srp_csr is not None:
            t0 = time.perf_counter()
            e_rapc = raptor_earliest_arrival_compiled(
                b, s, d, t, closed_walks_csr=cw_csr, srp_csr=srp_csr)
            ms_rapc = (time.perf_counter() - t0) * 1000
        else:
            e_rapc, ms_rapc = None, 0.0

        # TEEG-ALT compiled
        t0 = time.perf_counter()
        e_tac = teeg_alt_compiled(g, s, d, t, labels)
        ms_tac = (time.perf_counter() - t0) * 1000

        # TEEG-D compiled
        t0 = time.perf_counter()
        e_tdc = teeg_d_compiled(g, s, d, t)
        ms_tdc = (time.perf_counter() - t0) * 1000

        # TEEG-B compiled
        t0 = time.perf_counter()
        e_tbc = teeg_b_compiled(g, s, d, t)
        ms_tbc = (time.perf_counter() - t0) * 1000

        # MG-Dial compiled
        t0 = time.perf_counter()
        e_mgdc = mg_dial_compiled(mg, s, d, t, closed_walks_csr=cw_csr)
        ms_mgdc = (time.perf_counter() - t0) * 1000

        # MG-Dij compiled (dedicated impl)
        t0 = time.perf_counter()
        e_mgjc = mg_dij_compiled(mg, s, d, t, closed_walks_csr=cw_csr)
        ms_mgjc = (time.perf_counter() - t0) * 1000

        # ULTRA compiled
        if "ultra" not in skip and srp_csr is not None:
            t0 = time.perf_counter()
            e_ultrac = ultra_raptor_earliest_arrival_compiled(
                b, s, d, t, ultra, srp_csr=srp_csr)
            ms_ultrac = (time.perf_counter() - t0) * 1000
        else:
            e_ultrac, ms_ultrac = None, 0.0

        # CH compiled
        if ch_ok:
            t0 = time.perf_counter()
            try:
                e_chc = ch_earliest_arrival_compiled(ch_graph, g, s, d, t)
            except Exception:
                e_chc = None
            ms_chc = (time.perf_counter() - t0) * 1000
        else:
            e_chc, ms_chc = None, 0.0

        # HL compiled
        if hl_ok:
            t0 = time.perf_counter()
            try:
                e_hlc = hl_earliest_arrival_compiled(hl_index, s, d, t)
            except Exception:
                e_hlc = None
            ms_hlc = (time.perf_counter() - t0) * 1000
        else:
            e_hlc, ms_hlc = None, 0.0

        rows.append(dict(
            q_id=q_id, src=s, dst=d, t_dep=t,
            ea_csa_compiled=_to_none_or_int(e_csac),
            ea_raptor_compiled=_to_none_or_int(e_rapc),
            ea_teeg_alt_compiled=_to_none_or_int(e_tac),
            ea_teeg_d_compiled=_to_none_or_int(e_tdc),
            ea_teeg_b_compiled=_to_none_or_int(e_tbc),
            ea_mg_dial_compiled=_to_none_or_int(e_mgdc),
            ea_mg_dij_compiled=_to_none_or_int(e_mgjc),
            ea_ultra_compiled=_to_none_or_int(e_ultrac),
            ea_ch_compiled=_to_none_or_int(e_chc),
            ea_hl_compiled=_to_none_or_int(e_hlc),
            ms_csa_compiled=ms_csac,
            ms_raptor_compiled=ms_rapc,
            ms_teeg_alt_compiled=ms_tac,
            ms_teeg_d_compiled=ms_tdc,
            ms_teeg_b_compiled=ms_tbc,
            ms_mg_dial_compiled=ms_mgdc,
            ms_mg_dij_compiled=ms_mgjc,
            ms_ultra_compiled=ms_ultrac,
            ms_ch_compiled=ms_chc,
            ms_hl_compiled=ms_hlc,
        ))

        if time.time() - last_print > 10:
            print(f"  [{i+1}/{len(queries)}] in progress ...", flush=True)
            last_print = time.time()
            gc.collect()

        # Progressive checkpoint every 200 queries
        if (i + 1) % 200 == 0:
            df_chk = pd.DataFrame(rows)
            df_chk.to_csv(out_path, index=False)
            print(f"  checkpoint at q={i+1}: {len(df_chk)} rows → {out_path}",
                  flush=True)

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[bench-compiled] DONE: {len(df)} rows → {out_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="comma-sep methods to skip (ch,hl)")
    args = ap.parse_args()

    queries = pd.read_csv(args.queries)
    skip = set([x.strip() for x in args.skip.split(",") if x.strip()])
    run_compiled_bench(queries, args.out, skip=skip)


if __name__ == "__main__":
    main()
