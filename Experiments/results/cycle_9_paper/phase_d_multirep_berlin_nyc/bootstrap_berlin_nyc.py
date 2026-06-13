"""Bootstrap CIs for Berlin + NYC multi-rep, per Phase D methodology."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
N_BOOT = 1000
RNG = np.random.default_rng(42)


def bootstrap_ci(values: np.ndarray, statistic=np.median, n=N_BOOT,
                 alpha=0.05) -> tuple[float, float, float]:
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    point = float(statistic(values))
    boot = np.empty(n)
    for i in range(n):
        sample = RNG.choice(values, size=len(values), replace=True)
        boot[i] = statistic(sample)
    return point, float(np.quantile(boot, alpha/2)), float(np.quantile(boot, 1-alpha/2))


def main() -> int:
    rows = []
    per_rep_rows = []
    for city in ("berlin", "nyc"):
        city_dir = OUT_DIR / city
        reps = sorted([p for p in city_dir.glob("rep_*") if p.is_dir()])
        print(f"\n=== {city}: {len(reps)} reps ===")
        per_method_obs: dict[str, list[float]] = {}
        for rep_dir in reps:
            rep_idx = int(rep_dir.name.split("_")[-1])
            for csv_name in ("compiled.csv", "v2.csv"):
                fp = rep_dir / csv_name
                if not fp.exists(): continue
                df = pd.read_csv(fp)
                for c in [x for x in df.columns if x.startswith("ms_")]:
                    method = c[3:]
                    vals = df[c].dropna().astype(float).values
                    vals = vals[vals > 0]
                    if len(vals) == 0: continue
                    per_method_obs.setdefault(method, []).extend(vals.tolist())
                    per_rep_rows.append({
                        "city": city, "method": method, "rep_idx": rep_idx,
                        "csv": csv_name, "n": len(vals),
                        "median_ms": float(np.median(vals)),
                        "mean_ms": float(np.mean(vals)),
                    })
        for method, obs_list in sorted(per_method_obs.items()):
            obs = np.asarray(obs_list, dtype=float)
            med, lo, hi = bootstrap_ci(obs, np.median)
            rows.append({
                "city": city, "method": method,
                "n_obs_total": len(obs), "n_reps": len(reps),
                "median_ms": round(med, 3),
                "ci95_lo_ms": round(lo, 3),
                "ci95_hi_ms": round(hi, 3),
                "ci_width_ms": round(hi - lo, 3),
                "mean_ms": round(float(np.mean(obs)), 3),
                "p95_ms": round(float(np.quantile(obs, 0.95)), 3),
            })

    df_ci = pd.DataFrame(rows).sort_values(["city", "median_ms"])
    df_ci.to_csv(OUT_DIR / "bootstrap_cis_berlin_nyc.csv", index=False)
    print("\n" + df_ci.to_string(index=False))

    pd.DataFrame(per_rep_rows).sort_values(["city", "method", "rep_idx"]).to_csv(
        OUT_DIR / "per_rep_medians.csv", index=False)
    print(f"\nWrote bootstrap_cis_berlin_nyc.csv + per_rep_medians.csv")

    # Headline summary
    summary = {}
    for city in ("berlin", "nyc"):
        city_df = df_ci[df_ci["city"] == city]
        if len(city_df) == 0: continue
        # Find csa_compiled and teeg_alt_v2 rows for ratio
        csa = city_df[city_df["method"] == "csa_compiled"]
        v2 = city_df[city_df["method"] == "teeg_alt_v2"]
        if len(csa) and len(v2):
            ratio_med = float(csa["median_ms"].iloc[0] / v2["median_ms"].iloc[0])
            summary[city] = {
                "csa_median_ms":   float(csa["median_ms"].iloc[0]),
                "csa_ci95":        [float(csa["ci95_lo_ms"].iloc[0]),
                                     float(csa["ci95_hi_ms"].iloc[0])],
                "teeg_alt_v2_median_ms": float(v2["median_ms"].iloc[0]),
                "teeg_alt_v2_ci95":      [float(v2["ci95_lo_ms"].iloc[0]),
                                            float(v2["ci95_hi_ms"].iloc[0])],
                "teeg_alt_v2_speedup_vs_csa": round(ratio_med, 3),
            }
    (OUT_DIR / "headline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== HEADLINE ===")
    for city, s in summary.items():
        print(f"  {city}: TEEG-ALT-v2 {s['teeg_alt_v2_median_ms']:.2f}ms "
              f"CI95% [{s['teeg_alt_v2_ci95'][0]:.2f},{s['teeg_alt_v2_ci95'][1]:.2f}] "
              f"vs CSA {s['csa_median_ms']:.2f}ms "
              f"CI95% [{s['csa_ci95'][0]:.2f},{s['csa_ci95'][1]:.2f}] "
              f"-> {s['teeg_alt_v2_speedup_vs_csa']}x")
    return 0


if __name__ == "__main__":
    main()
