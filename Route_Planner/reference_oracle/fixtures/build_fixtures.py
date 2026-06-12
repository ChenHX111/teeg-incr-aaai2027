"""Build the 18 toy regression fixtures for Reference-CSA*.

Per Day-3 audit (v1 → revised v2 per audit feedback):
- 18 fixtures across 7 equivalence classes (A walk-only, B single-trip, C multi-leg,
  D transfer, E walk-extends, F boundary, G tie-breaking — q_id=38 mechanism).
- Canonical JSON hashing (not raw numpy bytes).
- Each fixture's expected EA is computed by hand (justified) and asserted against
  Reference-CSA* at fixture-build time.
- Manifest is committed with `manifest_lock.sha256` to prevent silent regeneration.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from Route_Planner.reference_oracle.parser import RefBundle, REF_CONN_DTYPE, DEFAULT_SERVICE_DATE
from Route_Planner.reference_oracle.footpath_closure import transitive_closure_f15
from Route_Planner.reference_oracle.csa_core import csa_earliest_arrival
from Route_Planner.reference_oracle.fixtures.canonical_hash import (
    canonical_bundle_hash, canonical_closed_walks_hash, canonical_fixture_hash, SCHEMA_VERSION
)

OUT_DIR = REPO / "tests" / "reference_oracle" / "fixtures_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _mk_bundle(n_stops: int, conns_list, walks_list, n_trips: int, trip_str_map: dict):
    """Build a RefBundle from explicit Python lists.

    conns_list: list of dicts each with keys trip_id_int, stop_seq_dep, stop_seq_arr,
                dep_min, arr_min, from_stop, to_stop, route_int, mode.
    walks_list: list of (from, to, walk_min) triples.
    """
    conns = np.zeros(len(conns_list), dtype=REF_CONN_DTYPE)
    for i, c in enumerate(conns_list):
        conns[i] = (c["trip_id_int"], c["stop_seq_dep"], c["stop_seq_arr"],
                    c["dep_min"], c["arr_min"], c["from_stop"], c["to_stop"],
                    c.get("route_int", 0), c.get("mode", 0))
    # Canonical sort: by dep_min, then trip_id_int, then stop_seq_dep (matches parser.py:210)
    order = np.lexsort((conns["stop_seq_dep"], conns["trip_id_int"], conns["dep_min"]))
    conns = conns[order]
    walks = np.array(walks_list, dtype=np.int32) if walks_list else np.zeros((0, 3), dtype=np.int32)
    return RefBundle(
        n_stops=n_stops, connections=conns, walks=walks,
        service_date=DEFAULT_SERVICE_DATE, t_lo=0, t_hi=60,
        trip_id_str_to_int=trip_str_map,
        stop_int_to_name={i: chr(ord("A") + i) if i < 26 else f"S{i}" for i in range(n_stops)},
        stop_lat={i: 0.0 for i in range(n_stops)}, stop_lon={i: 0.0 for i in range(n_stops)},
        n_trips=n_trips, n_routes=1,
    )


def fixtures():
    """Return list of 18 fixture dicts. Each: id, description, n_stops, n_trips, n_walks_raw,
    bundle inputs (conns_list, walks_list), queries with expected_ea, justification."""
    fix = []

    # --- Class A: Walk-only ---
    fix.append({
        "id": "A1", "description": "walk-only 2-stop, asymmetric directed walk A->B (5 min)",
        "n_stops": 2, "n_trips": 0, "trip_str_map": {},
        "conns_list": [],
        "walks_list": [(0, 1, 5)],  # only A->B, NOT B->A (directed)
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": 5,
                     "just": "Direct walk A->B in 5 min; tau[1]=0+5=5."},
                    {"src": 1, "dst": 0, "t_dep": 0, "expected_ea": None,
                     "just": "No walk B->A in raw walks; unreachable."}],
    })
    fix.append({
        "id": "A2", "description": "walk-chain 3-stop A-B-C closes via Dijkstra",
        "n_stops": 3, "n_trips": 0, "trip_str_map": {},
        "conns_list": [],
        "walks_list": [(0, 1, 3), (1, 0, 3), (1, 2, 4), (2, 1, 4)],
        "queries": [{"src": 0, "dst": 2, "t_dep": 0, "expected_ea": 7,
                     "just": "F_15 closure: A->B(3) + B->C(4) = 7 ≤ 15 cap."}],
    })
    fix.append({
        "id": "A3", "description": "transitive-closure correctness: direct edge worse than 2-hop",
        "n_stops": 3, "n_trips": 0, "trip_str_map": {},
        # Direct A->C is 14, but A->B(10) + B->C(2) = 12 should win after closure
        "conns_list": [],
        "walks_list": [(0, 1, 10), (1, 0, 10), (1, 2, 2), (2, 1, 2), (0, 2, 14), (2, 0, 14)],
        "queries": [{"src": 0, "dst": 2, "t_dep": 0, "expected_ea": 12,
                     "just": "Closure: min(direct 14, A->B->C = 12) = 12. Pins multi-hop composition."}],
    })

    # --- Class B: Single-trip ---
    fix.append({
        "id": "B1", "description": "single trip A->B dep=10",
        "n_stops": 2, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 10, "arr_min": 15, "from_stop": 0, "to_stop": 1}],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": 15,
                     "just": "Wait at A 10 min, ride to B, arr=15."}],
    })
    fix.append({
        "id": "B2", "description": "two trips, dep query misses first, catches second",
        "n_stops": 2, "n_trips": 2, "trip_str_map": {"T1": 0, "T2": 1},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 1, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 15, "arr_min": 20, "from_stop": 0, "to_stop": 1},
        ],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 12, "expected_ea": 20,
                     "just": "T1 dep=5 < 12 = miss; T2 dep=15 >= 12 = catch, arr=20."}],
    })
    fix.append({
        "id": "B3", "description": "unreachable: trip dep < query t_dep, no other path",
        "n_stops": 2, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1}],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 100, "expected_ea": None,
                     "just": "Only trip dep=5 < 100, missed; no walk B; None."}],
    })

    # --- Class C: Multi-leg trip ---
    fix.append({
        "id": "C1", "description": "3-stop single trip stay-on through B",
        "n_stops": 3, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 0, "arr_min": 5, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 0, "stop_seq_dep": 1, "stop_seq_arr": 2,
             "dep_min": 5, "arr_min": 10, "from_stop": 1, "to_stop": 2},
        ],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 2, "t_dep": 0, "expected_ea": 10,
                     "just": "Board A at 0, stay on T1 through B (no re-board), arr C=10."}],
    })
    fix.append({
        "id": "C2", "description": "4-stop trip, board at intermediate stop B",
        "n_stops": 4, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 0, "arr_min": 5, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 0, "stop_seq_dep": 1, "stop_seq_arr": 2,
             "dep_min": 6, "arr_min": 11, "from_stop": 1, "to_stop": 2},
            {"trip_id_int": 0, "stop_seq_dep": 2, "stop_seq_arr": 3,
             "dep_min": 12, "arr_min": 17, "from_stop": 2, "to_stop": 3},
        ],
        "walks_list": [],
        "queries": [{"src": 1, "dst": 3, "t_dep": 0, "expected_ea": 17,
                     "just": "Board T1 at B (dep=6>=0), ride B->C->D, arr=17."}],
    })

    # --- Class D: Transfer with walk between trips ---
    fix.append({
        "id": "D1", "description": "trip + walk + trip; comfortable transfer",
        "n_stops": 4, "n_trips": 2, "trip_str_map": {"T1": 0, "T2": 1},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 0, "arr_min": 5, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 1, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 10, "arr_min": 15, "from_stop": 2, "to_stop": 3},
        ],
        "walks_list": [(1, 2, 3), (2, 1, 3)],
        "queries": [{"src": 0, "dst": 3, "t_dep": 0, "expected_ea": 15,
                     "just": "T1 A->B arr=5; walk B->C in 3 = arr C=8; T2 C->D dep=10>=8 catch; arr=15."}],
    })
    fix.append({
        "id": "D2a", "description": "transfer just missed (walk arrives 1 min late)",
        "n_stops": 4, "n_trips": 2, "trip_str_map": {"T1": 0, "T2": 1},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 0, "arr_min": 5, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 1, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 7, "arr_min": 12, "from_stop": 2, "to_stop": 3},
        ],
        "walks_list": [(1, 2, 3), (2, 1, 3)],
        "queries": [{"src": 0, "dst": 3, "t_dep": 0, "expected_ea": None,
                     "just": "Arr B=5; walk 3 = arr C=8 > T2 dep=7; miss only T2; no other path; None."}],
    })

    # --- Class E: Walk extends after trip arrival ---
    fix.append({
        "id": "E1", "description": "trip + walk extends to dst",
        "n_stops": 3, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 0, "arr_min": 5, "from_stop": 0, "to_stop": 1}],
        "walks_list": [(1, 2, 3), (2, 1, 3)],
        "queries": [{"src": 0, "dst": 2, "t_dep": 0, "expected_ea": 8,
                     "just": "T1 A->B arr=5; walk B->C in 3 = arr C=8."}],
    })

    # --- Class F: Boundary cases ---
    fix.append({
        "id": "F1", "description": "self-query src==dst returns t_dep",
        "n_stops": 2, "n_trips": 0, "trip_str_map": {},
        "conns_list": [],
        "walks_list": [(0, 1, 5)],  # walk exists but irrelevant
        "queries": [{"src": 0, "dst": 0, "t_dep": 7, "expected_ea": 7,
                     "just": "src==dst; tau[src] init = t_dep = 7; return tau[dst] = tau[0] = 7. Non-zero t_dep distinguishes from accidental-zero bug."}],
    })
    fix.append({
        "id": "F2", "description": "boundary: query t_dep == trip dep_min (inclusive boarding)",
        "n_stops": 2, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1}],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 5, "expected_ea": 10,
                     "just": "Boarding is INCLUSIVE: dep_min(5) >= tau[src]=t_dep(5); board; arr=10."}],
    })
    fix.append({
        "id": "F3", "description": "boundary: t_dep > trip dep_min by 1 min, miss",
        "n_stops": 2, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1}],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 6, "expected_ea": None,
                     "just": "Only trip dep=5 < t_dep=6 = miss; no walks; None."}],
    })
    fix.append({
        "id": "F4a", "description": "F_15 cap-boundary INCLUSIVE: walk of exactly 15 min OK",
        "n_stops": 2, "n_trips": 0, "trip_str_map": {},
        "conns_list": [],
        "walks_list": [(0, 1, 15)],
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": 15,
                     "just": "F_15 cap is INCLUSIVE; 15-min walk allowed; ea=0+15=15."}],
    })
    fix.append({
        "id": "F4b", "description": "F_15 cap-boundary EXCLUSIVE: walk of 16 min rejected",
        "n_stops": 2, "n_trips": 0, "trip_str_map": {},
        "conns_list": [],
        "walks_list": [(0, 1, 16)],
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": None,
                     "just": "F_15 cap rejects walk > 15 min; closure excludes; unreachable."}],
    })

    # --- Class G: q_id=38 tie-breaking mechanism (THE motivating case) ---
    fix.append({
        "id": "G1", "description": "two trips same dep_min/from_stop, different arr_min; earlier-arr wins",
        # Conn A: trip 0, dep=5, arr=10 (5-min ride)
        # Conn B: trip 1, dep=5, arr=8  (3-min ride, faster)
        # Both from src=0 to dst=1 at same dep_min.
        "n_stops": 2, "n_trips": 2, "trip_str_map": {"T1": 0, "T2": 1},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 1, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 5, "arr_min": 8, "from_stop": 0, "to_stop": 1},
        ],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": 8,
                     "just": "Both trips boardable at dep=5; T2 arrives at 8 < T1's 10; CSA must find 8 regardless of sort order. Pins tie-resolution correctness."}],
    })
    fix.append({
        "id": "G2", "description": "stay-on-trip vs new-board at identical dep_min",
        # Trip T1: A->B dep=0 arr=3; B->C dep=3 arr=7
        # Trip T2: B->C dep=3 arr=6  (faster B->C than T1's continuation)
        # Query A->C dep=0: must find 6 (board T1 at A, stay through B, but T2 from B is faster — should hop off T1 at B and board T2)
        "n_stops": 3, "n_trips": 2, "trip_str_map": {"T1": 0, "T2": 1},
        "conns_list": [
            {"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 0, "arr_min": 3, "from_stop": 0, "to_stop": 1},
            {"trip_id_int": 0, "stop_seq_dep": 1, "stop_seq_arr": 2,
             "dep_min": 3, "arr_min": 7, "from_stop": 1, "to_stop": 2},
            {"trip_id_int": 1, "stop_seq_dep": 0, "stop_seq_arr": 1,
             "dep_min": 3, "arr_min": 6, "from_stop": 1, "to_stop": 2},
        ],
        "walks_list": [],
        "queries": [{"src": 0, "dst": 2, "t_dep": 0, "expected_ea": 6,
                     "just": "T1 A->B arr=3; from B both T1's B->C (arr 7) and T2's B->C (arr 6) at dep=3. Better is T2: arr 6. CSA must pick the minimum, not just stay-on T1."}],
    })

    # --- Class H: walk vs trip dominance ---
    fix.append({
        "id": "H1", "description": "walk reaches dst earlier than any trip; trip must not overwrite",
        # Walk A->B in 2 min; trip A->B dep=5 arr=10. Walk wins (arr=2 vs 10).
        "n_stops": 2, "n_trips": 1, "trip_str_map": {"T1": 0},
        "conns_list": [{"trip_id_int": 0, "stop_seq_dep": 0, "stop_seq_arr": 1,
                        "dep_min": 5, "arr_min": 10, "from_stop": 0, "to_stop": 1}],
        "walks_list": [(0, 1, 2)],
        "queries": [{"src": 0, "dst": 1, "t_dep": 0, "expected_ea": 2,
                     "just": "Walk arr=0+2=2 < trip arr=10; CSA must not overwrite tau[1]=2 with 10. Pins min-relaxation correctness."}],
    })

    return fix


def build_and_verify():
    fix_list = fixtures()
    assert len(fix_list) == 19, f"expected 19 fixtures, got {len(fix_list)}"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "n_fixtures": len(fix_list),
        "fixtures": [],
    }

    failures = []
    for f in fix_list:
        bundle = _mk_bundle(f["n_stops"], f["conns_list"], f["walks_list"],
                             f["n_trips"], f["trip_str_map"])
        closed = transitive_closure_f15(bundle.walks, bundle.n_stops, max_walk_min=15)

        # Compute hashes
        # Reconstruct raw connection tuples from the (sorted) bundle
        conns_for_hash = [(int(c["from_stop"]), int(c["to_stop"]), int(c["dep_min"]),
                          int(c["arr_min"]), int(c["trip_id_int"]))
                         for c in bundle.connections]
        bundle_sha = canonical_bundle_hash(f["n_stops"], conns_for_hash, f["walks_list"])
        walks_closed_sha = canonical_closed_walks_hash(closed)

        # Verify each query
        for q in f["queries"]:
            ea = csa_earliest_arrival(bundle, q["src"], q["dst"], q["t_dep"], closed_walks=closed)
            if ea != q["expected_ea"]:
                failures.append(f"FAIL {f['id']} src={q['src']} dst={q['dst']} t_dep={q['t_dep']}: "
                                f"expected={q['expected_ea']} got={ea} ({q['just']})")
            else:
                print(f"  PASS {f['id']:6s} src={q['src']} dst={q['dst']} t_dep={q['t_dep']:3d} => ea={ea}")

        fixture_record = {
            "id": f["id"],
            "description": f["description"],
            "n_stops": f["n_stops"],
            "n_trips": f["n_trips"],
            "n_walks_raw": len(f["walks_list"]),
            "bundle_sha256": bundle_sha,
            "walks_closed_sha256": walks_closed_sha,
            "conns_list": [list(c) for c in conns_for_hash],
            "walks_list": [list(w) for w in f["walks_list"]],
            "trip_str_map": f["trip_str_map"],
            "queries": [{"src": int(q["src"]), "dst": int(q["dst"]),
                        "t_dep": int(q["t_dep"]),
                        "expected_ea": None if q["expected_ea"] is None else int(q["expected_ea"]),
                        "justification": q["just"]}
                       for q in f["queries"]],
        }
        fixture_record["fixture_sha256"] = canonical_fixture_hash(fixture_record)
        manifest["fixtures"].append(fixture_record)

    if failures:
        print(f"\n=== {len(failures)} VERIFICATION FAILURES ===")
        for f in failures:
            print(f"  {f}")
        return 1, manifest

    # Write manifest + lock
    manifest_str = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path = OUT_DIR / "fixtures_manifest.json"
    manifest_path.write_text(manifest_str, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    (OUT_DIR / "manifest_lock.sha256").write_text(manifest_sha + "\n", encoding="utf-8")

    print(f"\n=== 19 fixtures built and verified ===")
    print(f"manifest: {manifest_path}")
    print(f"manifest_sha256: {manifest_sha}")
    print(f"lock: {OUT_DIR / 'manifest_lock.sha256'}")
    return 0, manifest


if __name__ == "__main__":
    code, _ = build_and_verify()
    raise SystemExit(code)
