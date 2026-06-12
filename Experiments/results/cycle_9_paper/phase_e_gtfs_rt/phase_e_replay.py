"""Phase E — SIRI parse + replay through Cycle 10 incremental path.

Reads a TfL SIRI capture directory produced by phase_e_capture.py and:
  1. Maps each VM/SX entry to a (timestamp, primitive, args) edit tuple.
     Primitives: delay_trip(trip_id, delta_s), cancel_trip(trip_id),
                 add_trip(trip_id, conns_seq) — match Cycle 10's API.
  2. Replays the resulting edit stream through TEEG-INCR's update path.
  3. Records per-edit median + p95 wall, per-cell csa_better count vs
     post-edit CSA rebuild oracle.

Output:
  edit_stream.csv         — parsed edits
  replay_results.csv      — per-edit wall + oracle comparison
  summary.json            — headline real-trace metrics

When run for the first time (no captures yet), this script is a no-op
that documents the planned shape of the integration. Pending PI's TfL
API key registration.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True,
                    help="capture dir produced by phase_e_capture.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap_dir = Path(args.captures).resolve()
    if not cap_dir.exists():
        print(f"NO CAPTURE DIR: {cap_dir}; pending PI API-key + 24h run.")
        return 0
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vm_frames = sorted(cap_dir.glob("siri_vm_*.json"))
    sx_frames = sorted(cap_dir.glob("siri_sx_*.json"))
    print(f"VM frames: {len(vm_frames)}; SX frames: {len(sx_frames)}")

    # TODO: parse + map to (delay_trip / cancel_trip / add_trip) primitives
    # TODO: replay through Route_Planner.teeg.teeg_alt_incr.apply_edit
    # TODO: compare against post-edit CSA rebuild oracle
    # TODO: report per-edit median + p95 wall + per-cell csa_better count

    summary = {
        "status": "scaffold_only_pending_real_capture",
        "n_vm_frames": len(vm_frames),
        "n_sx_frames": len(sx_frames),
        "todo": [
            "parse VM frames to detect schedule deviations vs static GTFS",
            "classify each deviation as cancel/delay/add primitive",
            "replay through Cycle 10 TEEG-INCR.apply_edit path",
            "compare per-edit EA against post-edit CSA rebuild oracle",
            "write replay_results.csv + summary.json",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
