"""Phase C — Per-suspect attribution of the KIT-vs-oracle 58pp scan-side gap.

Approach: partition the existing 3000-OD KIT-vs-oracle disagreement data by
the empirical signature each suspect would leave (since re-running KIT under
3 ablations would require patching the C++ build).

Suspects and signatures (revised after first pass):
  S1 SERVICE-DAY FILTER  — KIT operates on a single calendar day; cannot
                            relax connections that cross midnight (oracle
                            uses full bundle including next-day prefixes).
                            Signature: |diff_csa_min| > 60 (cliff: trip
                            requires using post-midnight connection).
                            EXPECTED to dominate per Karlsruhe lit ~35-45pp.
  S3 MCC STOP-SET        — KIT runs reduceToMaximumConnectedComponent,
                            dropping ~155 isolated stops on Full London.
                            Signature: kit_dropped == True (47 ODs) OR
                            agree_csa == ORACLE_FINDS_KIT_BLIND.
                            EXPECTED ~5-10pp.
  S4 MULTIPASS BLIND     — both KIT and our oracle use single-pass CSA;
                            both inherit the 185-residual blind-spot.
                            HOWEVER: since both use single-pass, S4 cases
                            should AGREE between KIT and oracle. So S4
                            should NOT appear in disagreement. (Validates
                            classification: S4 count should be near zero.)
  S2 TIME-ENCODING / OTHER — KIT uses int16 or some quantization; signature
                              ambiguous in residual data. Reported as bulk
                              "other" rather than attributed.
  S5 RESIDUAL            — disagreements not attributed by S1+S3, broken
                            down by sign of diff:
                            S5a (diff<0): KIT terminates later than oracle
                                          (KIT slower); some are int16
                                          rounding or partial-day-window
                                          edge cases
                            S5b (diff>0): KIT terminates earlier than oracle
                                          (KIT faster); typically int16
                                          truncation or MCC indirect

Priority: S3 > S1 > (S5a/S5b by sign).
S2 not attributed cleanly; reported as part of S5.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parents[3]

KIT_VS_ORACLE = REPO / "Experiments" / "results" / "cycle_9_paper" / "karlsruhe_comparison" / "kit_ultra_3000_fullldn.csv"
MULTIPASS_RESID = REPO / "Experiments" / "results" / "cycle_9_paper" / "multipass_3000_full" / "residual_classification.csv"


def main() -> int:
    df = pd.read_csv(KIT_VS_ORACLE)
    mp = pd.read_csv(MULTIPASS_RESID)
    mp_qids = set(mp["q_id"].astype(int).tolist())
    print(f"Loaded {len(df)} KIT-vs-oracle rows, {len(mp_qids)} multipass-blind q_ids")

    def classify(row):
        agree = row["agree_csa"]
        if agree == "AGREE":
            return "AGREE"
        if agree == "NEAR":
            return "NEAR"
        if agree == "BOTH_UNREACHABLE":
            return "BOTH_UNREACHABLE"
        # S3 first (unreachability)
        if row["kit_dropped"] or agree == "ORACLE_FINDS_KIT_BLIND":
            return "S3_MCC"
        # S1 cliff
        diff = row["diff_csa_min"]
        if pd.notna(diff) and abs(diff) > 60:
            return "S1_SERVICE_DAY"
        # remaining DISAGREE by sign
        if agree in ("DISAGREE", "KIT_FINDS_ORACLE_BLIND"):
            if pd.isna(diff):
                return "S5_OTHER"
            return "S5a_KIT_SLOWER" if diff < 0 else "S5b_KIT_FASTER"
        return "UNCLASSIFIED"

    df["suspect"] = df.apply(classify, axis=1)
    counts = df["suspect"].value_counts().to_dict()
    print("\nSuspect breakdown (of 3000 ODs):")
    for s in sorted(counts, key=lambda x: -counts[x]):
        n = counts[s]
        pct = 100.0 * n / len(df)
        print(f"  {s:25s}: {n:5d} ({pct:5.1f}%)")

    # Cross-check S4 multipass-blind: how many of the multipass-blind q_ids
    # actually AGREE between KIT and oracle (both single-pass, both blind)?
    df_mp = df[df["q_id"].astype(int).isin(mp_qids)]
    print(f"\nS4 cross-check: of {len(mp_qids)} multipass-blind q_ids:")
    print(df_mp["agree_csa"].value_counts())

    # Contribution to 55.8pp scan-side gap
    n_misses = (len(df) - counts.get("AGREE", 0) - counts.get("NEAR", 0)
                - counts.get("BOTH_UNREACHABLE", 0))
    scan_gap_pp = 55.8
    print(f"\nTotal misses (non AGREE/NEAR/BOTH_UNREACHABLE): {n_misses}")
    print(f"Scan-side gap to attribute: {scan_gap_pp} pp\n")
    suspect_pp = {}
    for s in ("S3_MCC", "S1_SERVICE_DAY", "S5a_KIT_SLOWER",
              "S5b_KIT_FASTER", "S5_OTHER"):
        c = counts.get(s, 0)
        pp = round(scan_gap_pp * c / n_misses, 2) if n_misses else 0
        suspect_pp[s] = {"count": c, "pp_contribution": pp}
        print(f"  {s:25s}: {c:5d} -> {pp:5.2f} pp of {scan_gap_pp}")

    out_csv = OUT_DIR / "attribution_3000.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    summary = {
        "n_queries": int(len(df)),
        "scan_side_gap_pp_from_4way": scan_gap_pp,
        "suspect_counts": {k: int(v) for k, v in counts.items()},
        "suspect_pp_contribution": suspect_pp,
        "n_misses_total": int(n_misses),
        "priority": ["S3_MCC", "S1_SERVICE_DAY", "S5a/S5b by sign"],
        "S4_multipass_cross_check": {
            "n_multipass_blind_qids": len(mp_qids),
            "kit_oracle_agreement_breakdown": df_mp["agree_csa"].value_counts().to_dict(),
        },
        "notes": [
            "Each disagreement attributed to one suspect via priority S3>S1>S5(by sign).",
            "S1 cliff threshold: |diff_csa_min| > 60 min (crossing-midnight cliff).",
            "S2 'int16-seconds encoding' not given a clean signature in this data;",
            "  effects are folded into S5a/S5b (which are the bulk residual after S1+S3).",
            "S4 cross-check: multipass-blind q_ids should AGREE between KIT and oracle",
            "  (both use single-pass CSA); large AGREE share confirms classifier sanity.",
            "S5a (KIT slower than oracle) = KIT missed a fast path; likely partial-day",
            "  window or aggressive filtering that doesn't trigger the 60-min cliff.",
            "S5b (KIT faster than oracle) = KIT thinks it's faster; impossible if",
            "  oracle is correct, suggesting KIT uses pre-truncated bundle or rounds",
            "  arrival times down (int16-deci-seconds quantization).",
        ],
    }
    (OUT_DIR / "suspect_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote suspect_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
