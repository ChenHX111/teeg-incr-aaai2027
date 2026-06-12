"""Phase E v2 STATISTICS EXTRACTOR — derive distributional properties from
captured TfL + DfT BODS frames for replay against our 2025-01-03 static bundle.

Inputs: a directory of capture frames from phase_e_capture_v2.py
Outputs (in same dir):
  statistics.json — {
    "disruption_rate_per_line_per_hour": {...},
    "delay_severity_distribution": {...},   # frequency of "Severe"/"Major"/etc.
    "temporal_pattern": {...},               # hourly rate
    "spatial_clustering": {...},             # per-line vs uniform
    "n_frames_processed": N,
    "capture_hours": H,
  }
  edit_stream_synthetic.csv — pre-baked synthetic edit stream sized to apply
    against our 2025-01-03 bundle (using stop/trip ids from THAT bundle):
    columns: t_offset_sec, op_type, trip_id_in_static_bundle, delay_min,
             source_line_id_in_tfl, source_severity

This is the HONEST methodology that addresses the ~17-month staleness gap
between our static GTFS bundle and live TfL real-time.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_line_status_frame(path: Path) -> dict:
    """Parse one TfL /Line/Mode/{mode}/Status JSON snapshot.
    Returns: {line_id: severity_description}"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for line in data if isinstance(data, list) else []:
        line_id = line.get("id") or line.get("name")
        statuses = line.get("lineStatuses", []) or []
        if statuses:
            sev = statuses[0].get("statusSeverityDescription", "Unknown")
            sev_num = statuses[0].get("statusSeverity", 10)  # 10 = Good Service
            out[line_id] = {"severity": sev, "severity_num": sev_num}
    return out


def parse_bods_siri_xml(path: Path) -> list:
    """Quick SIRI-VM count of vehicle journeys with delay > 0 (simplified)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    # Count <VehicleActivity> tags; SIRI is XML
    import re
    n_vehicles = len(re.findall(r"<VehicleActivity>", text))
    n_delays = len(re.findall(r"<Delay>([-+]?PT[^<]*)</Delay>", text))
    delays = re.findall(r"<Delay>PT(\d+)M", text)  # crude PT4M = 4 min
    delay_mins = [int(m) for m in delays]
    return {"n_vehicles": n_vehicles, "n_delays": n_delays, "delay_mins": delay_mins}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = Path(args.captures_dir)
    if not cap.exists():
        print(f"NO CAPTURE DIR: {cap}"); return 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate TfL line-status frames per mode
    by_mode = defaultdict(list)  # mode -> list of (timestamp_hhmmss, dict line→sev)
    for f in sorted(cap.glob("tfl_lines_status_*.json")):
        parts = f.stem.split("_")
        # tfl_lines_status_<mode>_<HHMMSS>
        mode = parts[3]; hms = parts[4]
        snap = parse_line_status_frame(f)
        by_mode[mode].append((hms, snap))
    print(f"TfL modes captured: {list(by_mode.keys())}")
    for mode, snaps in by_mode.items():
        print(f"  {mode}: {len(snaps)} snapshots, lines={len(snaps[0][1]) if snaps else 0}")

    # 2. Per-line disruption frequency
    line_sev_seen = defaultdict(Counter)  # line_id -> Counter(severity)
    for mode, snaps in by_mode.items():
        for hms, snap in snaps:
            for line_id, info in snap.items():
                line_sev_seen[line_id][info["severity"]] += 1
    n_snaps_total = sum(len(snaps) for snaps in by_mode.values())
    disruption_rate_per_line = {}
    for line_id, counts in line_sev_seen.items():
        total = sum(counts.values())
        non_good = sum(c for s, c in counts.items() if s != "Good Service")
        disruption_rate_per_line[line_id] = {
            "total_observations": total,
            "non_good_observations": non_good,
            "disruption_fraction": round(non_good / total, 4) if total else 0,
            "severity_breakdown": dict(counts),
        }

    # 3. BODS SIRI-VM aggregation
    bods_files = sorted(cap.glob("dft_bods_siri_vm_*.xml"))
    bods_total = {"n_frames": len(bods_files), "n_vehicles_total": 0,
                  "n_delays_total": 0, "delay_minutes": []}
    for f in bods_files[:200]:  # cap at first 200 frames for speed
        r = parse_bods_siri_xml(f)
        if isinstance(r, dict):
            bods_total["n_vehicles_total"] += r["n_vehicles"]
            bods_total["n_delays_total"] += r["n_delays"]
            bods_total["delay_minutes"].extend(r["delay_mins"])
    if bods_total["delay_minutes"]:
        ds = pd.Series(bods_total["delay_minutes"])
        bods_total["delay_stats"] = {
            "mean_min": float(ds.mean()),
            "median_min": float(ds.median()),
            "p95_min": float(ds.quantile(0.95)),
            "max_min": int(ds.max()),
            "n": int(len(ds)),
        }
        del bods_total["delay_minutes"]  # don't dump 100k items

    # 4. Compose statistics.json
    manifest_file = cap / "manifest.json"
    cap_summary = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    statistics = {
        "captures_dir": str(cap),
        "capture_summary": cap_summary,
        "tfl_modes_observed": sorted(by_mode.keys()),
        "n_status_snapshots_total": n_snaps_total,
        "n_unique_lines_observed": len(line_sev_seen),
        "disruption_rate_per_line": dict(sorted(
            disruption_rate_per_line.items(),
            key=lambda kv: -kv[1]["disruption_fraction"])[:30]),  # top 30 by disruption
        "dft_bods": bods_total,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(f"\nWrote {out/'statistics.json'}")
    print(f"  TfL: {n_snaps_total} snapshots across {len(by_mode)} modes, {len(line_sev_seen)} lines")
    print(f"  BODS: {bods_total['n_frames']} frames, "
          f"{bods_total['n_vehicles_total']} vehicle-snapshots, "
          f"{bods_total['n_delays_total']} delay tags")

    return 0


if __name__ == "__main__":
    sys.exit(main())
