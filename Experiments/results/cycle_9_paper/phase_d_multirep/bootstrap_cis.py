"""Phase D — bootstrap 95% CIs on the existing 3-rep Full London multi-rep data.

Inputs: 3 reps of fullldn_compiled.csv + fullldn_v2.csv (each one full
3000-OD benchmark) from results_multi_rep_20260612_0046.zip.

Method: for each (method) cell, pool per-OD times across reps, bootstrap
95% CI of the median (1000 resamples).

Outputs:
  bootstrap_cis.csv     — method, median_ms, ci95_lo, ci95_hi, n_reps, n_total_obs
  per_rep_medians.csv   — method, rep_idx, median_ms (raw per-rep)
  table1_ready.tex      — drop-in LaTeX snippet for Table 1
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
N_BOOT = 1000
RNG = np.random.default_rng(42)


def bootstrap_ci(values: np.ndarray, statistic=np.median, n=N_BOOT,
                 alpha=0.05) -> tuple[float, float, float]:
    """Return (statistic_point, ci_lo, ci_hi)."""
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    point = float(statistic(values))
    boot = np.empty(n)
    n_obs = len(values)
    for i in range(n):
        sample = RNG.choice(values, size=n_obs, replace=True)
        boot[i] = statistic(sample)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, lo, hi


def main() -> int:
    # The reps are nested under results/multi_rep_v2_<ts>/rep_NN/
    rep_root_candidates = list(OUT_DIR.glob("results/multi_rep_*"))
    if rep_root_candidates:
        rep_root = rep_root_candidates[0]
    else:
        rep_root = OUT_DIR
    reps = sorted([p for p in rep_root.glob("rep_*") if p.is_dir()])
    print(f"Rep root: {rep_root}")
    print(f"Found {len(reps)} reps: {[p.name for p in reps]}")

    # Per-method, accumulate per-OD ms across reps
    per_method_obs: dict[str, list[float]] = {}
    per_rep_medians: list[dict] = []

    for rep_dir in reps:
        rep_idx = int(rep_dir.name.split("_")[-1])
        for csv_name in ("fullldn_compiled.csv", "fullldn_v2.csv"):
            fp = rep_dir / csv_name
            if not fp.exists():
                print(f"  MISSING: {fp}")
                continue
            df = pd.read_csv(fp)
            # Each csv has columns like q_id, src, dst, t_dep, ms_<method>, ea_<method>
            ms_cols = [c for c in df.columns if c.startswith("ms_")]
            for c in ms_cols:
                method = c[3:]
                vals = df[c].dropna().astype(float).values
                vals = vals[vals > 0]  # drop sentinels
                if len(vals) == 0:
                    continue
                per_method_obs.setdefault(method, []).extend(vals.tolist())
                per_rep_medians.append({
                    "method": method, "rep_idx": rep_idx,
                    "csv": csv_name, "n": len(vals),
                    "median_ms": float(np.median(vals)),
                    "mean_ms": float(np.mean(vals)),
                    "p95_ms": float(np.quantile(vals, 0.95)),
                })
            print(f"  {rep_dir.name}/{csv_name}: {len(ms_cols)} method cols, {len(df)} OD rows")

    print(f"\nTotal methods: {len(per_method_obs)}")

    # Compute bootstrap CIs per method
    rows = []
    for method, obs_list in sorted(per_method_obs.items()):
        obs = np.asarray(obs_list, dtype=float)
        med, lo, hi = bootstrap_ci(obs, np.median)
        mean = float(np.mean(obs))
        p95 = float(np.quantile(obs, 0.95))
        rows.append({
            "method": method,
            "n_obs_total": len(obs),
            "n_reps": len(reps),
            "median_ms": round(med, 3),
            "ci95_lo_ms": round(lo, 3),
            "ci95_hi_ms": round(hi, 3),
            "ci_width_ms": round(hi - lo, 3),
            "mean_ms": round(mean, 3),
            "p95_ms": round(p95, 3),
        })

    df_ci = pd.DataFrame(rows).sort_values("median_ms")
    out_csv = OUT_DIR / "bootstrap_cis.csv"
    df_ci.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(df_ci.to_string(index=False))

    df_per_rep = pd.DataFrame(per_rep_medians).sort_values(["method", "rep_idx"])
    df_per_rep.to_csv(OUT_DIR / "per_rep_medians.csv", index=False)
    print(f"\nWrote per_rep_medians.csv")

    # Generate Table 1-ready LaTeX snippet (Full London column)
    tex_lines = [
        "% Phase D — bootstrap 95% CIs on existing 3-rep Full London data",
        "% Drop-in Full London column for Table 1 (3000 OD, N=3 reps)",
        "% Format: median (CI95_lo, CI95_hi) in ms",
        "%",
        "% method               | median_ms (CI95)",
        "%-------------------- | ----------------"]
    for _, r in df_ci.iterrows():
        tex_lines.append(
            f"% {r['method']:<24s} {r['median_ms']:>8.2f} "
            f"({r['ci95_lo_ms']:>6.2f},{r['ci95_hi_ms']:>6.2f}) "
            f"n={int(r['n_obs_total']):>5d}"
        )
    (OUT_DIR / "table1_full_london.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote table1_full_london.tex")

    summary = {
        "n_reps": len(reps),
        "n_methods": len(per_method_obs),
        "bootstrap_n_resamples": N_BOOT,
        "ci_alpha": 0.05,
        "city": "Full London",
        "n_ods_per_rep": 3000,
        "notes": [
            "Bootstrap on per-OD ms pooled across reps (not on per-rep medians).",
            "median statistic used (more robust than mean for skewed wall-time distributions).",
            "ci_width = ci95_hi - ci95_lo (smaller = tighter estimate).",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
