"""Phase B follow-up: classify 185 multipass-CSA residuals by failure mode + cross-check vs 17."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
RES = OUT_DIR / "results_3000.csv"
SUMMARY = OUT_DIR / "summary.json"

REPO = OUT_DIR.parents[3]
TEEG_VERDICT = REPO / "Experiments" / "results" / "cycle_9_paper" / "RESIDUAL_17_DEEP_DIVE" / "verdict.csv"

ORIGINAL_17 = [102, 430, 431, 634, 827, 930, 1384, 1432, 1497, 1640,
               1800, 1811, 1858, 1893, 2038, 2415, 2759]


def classify_delta(delta):
    """Classify residuals by absolute size of correction."""
    if delta is None or delta == 0:
        return "none"
    a = abs(delta)
    if a <= 2:
        return "small_walk_from_walk"        # likely transitive 2-hop walk miss
    elif a <= 5:
        return "medium_transfer_chain"       # multi-hop walk chain
    elif a <= 15:
        return "large_src_walk_suppresses"   # src-walk-then-board pattern
    else:
        return "huge_alt_route"              # full alternative route found


def main():
    df = pd.read_csv(RES)
    print(f"Loaded {len(df)} rows")
    df_disagree = df[df["delta_min"].fillna(0).astype(int) < 0].copy()
    print(f"Disagreements: {len(df_disagree)}")

    df_disagree["mode"] = df_disagree["delta_min"].astype(int).apply(classify_delta)
    mode_counts = df_disagree["mode"].value_counts().to_dict()
    print(f"\nMode breakdown:")
    for m, n in mode_counts.items():
        pct = 100.0 * n / len(df_disagree)
        print(f"  {m:30s}: {n:4d} ({pct:5.1f}%)")

    # Cross-check: are the original 17 a subset?
    found_17 = set(df_disagree["q_id"].astype(int).tolist()) & set(ORIGINAL_17)
    missing_17 = set(ORIGINAL_17) - found_17
    print(f"\nOriginal 17 cross-check:")
    print(f"  Found in 185: {len(found_17)}/17 = {sorted(found_17)}")
    if missing_17:
        print(f"  MISSING from 185 (need investigation): {sorted(missing_17)}")
    else:
        print(f"  ✅ ALL 17 are subset of 185 — consistency confirmed")

    # Delta stats
    deltas = df_disagree["delta_min"].astype(int).values
    print(f"\nDelta stats (min): n={len(deltas)} mean={deltas.mean():.2f} "
          f"median={pd.Series(deltas).median():.0f} min={deltas.min()} max={deltas.max()}")

    # Wall stats
    walls = df["wall_s"].astype(float).values
    print(f"\nMultipass wall (s): mean={walls.mean():.3f} median={pd.Series(walls).median():.3f} "
          f"p95={pd.Series(walls).quantile(0.95):.3f} max={walls.max():.3f}")

    # Write classification CSV
    out_csv = OUT_DIR / "residual_classification.csv"
    df_disagree[["q_id", "src", "dst", "t_dep", "alg1_ea", "alg2_ea", "delta_min", "mode", "wall_s"]].to_csv(
        out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Augment summary.json
    summary = {
        "n_queries": int(len(df)),
        "n_disagree": int(len(df_disagree)),
        "disagree_rate_pct": round(100.0 * len(df_disagree) / len(df), 3),
        "delta_min_mean": float(deltas.mean()),
        "delta_min_median": float(pd.Series(deltas).median()),
        "delta_min_min": int(deltas.min()),
        "delta_min_max": int(deltas.max()),
        "multipass_wall_s_mean": float(walls.mean()),
        "multipass_wall_s_p95": float(pd.Series(walls).quantile(0.95)),
        "multipass_wall_s_max": float(walls.max()),
        "mode_counts": {k: int(v) for k, v in mode_counts.items()},
        "original_17_subset_check": {
            "original_count": 17,
            "found_in_185": len(found_17),
            "missing": sorted(missing_17),
            "consistency_confirmed": len(missing_17) == 0,
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Updated {SUMMARY}")
    return summary


if __name__ == "__main__":
    main()
