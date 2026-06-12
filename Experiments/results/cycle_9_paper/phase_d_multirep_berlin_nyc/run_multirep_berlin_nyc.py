"""Multi-rep Berlin + NYC bench for bootstrap CIs (Phase D extension).

Runs N=3 reps of benchmark_compiled_only + benchmark_compiled_v2 on Berlin
and NYC bundles, into a per-rep CSV layout matching Phase D's analyzer.

Walk-cap: defaults to 15 min (fits well under 64 GB on dev box for Berlin+NYC).
"""
from __future__ import annotations
import os
import sys
import subprocess
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT_BASE = Path(__file__).resolve().parent

BERLIN_DATA = REPO / "Cities" / "Berlin" / "MCRaptor_Data_Berlin"
NYC_DATA = REPO / "MCRaptor_Data_NYC"
BERLIN_Q = REPO / "Cities" / "Berlin" / "queries_berlin_3000.csv"
NYC_Q = REPO / "Cities" / "NYC" / "queries_nyc_3000.csv"

CITIES = [
    ("berlin", BERLIN_DATA, BERLIN_Q),
    ("nyc",    NYC_DATA,    NYC_Q),
]
REPS = 3
N_QUERIES = 500


def run_one_rep(city: str, data_dir: Path, queries: Path,
                rep_idx: int, rep_dir: Path) -> dict:
    rep_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MCRAPTOR_DATA_DIR"] = str(data_dir)
    env["PYTHONIOENCODING"] = "utf-8"

    queries_subset = rep_dir.parent / f"queries_{N_QUERIES}.csv"
    if not queries_subset.exists():
        import pandas as pd
        df = pd.read_csv(queries).head(N_QUERIES)
        df.to_csv(queries_subset, index=False)

    t0 = time.time()
    cmd1 = [sys.executable, "-m", "Experiments.benchmark_compiled_only",
            "--queries", str(queries_subset),
            "--out", str(rep_dir / "compiled.csv")]
    with open(rep_dir / "compiled.log", "w") as f:
        rc1 = subprocess.call(cmd1, cwd=str(REPO), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    t1 = time.time()
    cmd2 = [sys.executable, "-m", "Experiments.benchmark_compiled_v2",
            "--queries", str(queries_subset),
            "--out", str(rep_dir / "v2.csv")]
    with open(rep_dir / "v2.log", "w") as f:
        rc2 = subprocess.call(cmd2, cwd=str(REPO), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    t2 = time.time()
    return {
        "city": city, "rep": rep_idx,
        "compiled_rc": rc1, "compiled_wall_s": round(t1 - t0, 1),
        "v2_rc": rc2, "v2_wall_s": round(t2 - t1, 1),
        "total_wall_min": round((t2 - t0) / 60, 1),
    }


def main():
    results = []
    for city, data_dir, queries in CITIES:
        print(f"\n=== CITY: {city} ===", flush=True)
        city_dir = OUT_BASE / city
        city_dir.mkdir(parents=True, exist_ok=True)
        for i in range(REPS):
            print(f"  rep {i+1}/{REPS}", flush=True)
            rep_dir = city_dir / f"rep_{i:02d}"
            r = run_one_rep(city, data_dir, queries, i, rep_dir)
            results.append(r)
            print(f"    rc1={r['compiled_rc']} rc2={r['v2_rc']} total={r['total_wall_min']} min", flush=True)
            (OUT_BASE / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE. {OUT_BASE / 'manifest.json'}")


if __name__ == "__main__":
    main()
