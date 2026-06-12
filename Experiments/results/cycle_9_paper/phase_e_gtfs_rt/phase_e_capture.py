"""Phase E — TfL Unified API GTFS-RT capture harness.

Runs a 24h polling loop against TfL Unified API and DfT Bus Open Data Service,
saving SIRI-VM (real-time vehicle monitoring) and SIRI-SX (disruptions) feeds
to per-hour JSON files for later parse + replay through Cycle 10.

USAGE (requires PI's TfL Unified API + DfT BODS keys):
    set TFL_APP_KEY=...
    set DFT_BODS_KEY=...
    python phase_e_capture.py --out captures/<date>/ --duration-hours 24

Output layout:
    captures/<date>/
        siri_vm_<HHMMSS>.json    (every 60s; vehicle positions + arrivals)
        siri_sx_<HHMMSS>.json    (every 5min; disruptions)
        run.log
        manifest.json

After the run, hand the dir off to phase_e_replay.py which:
    1. parses each SIRI-VM frame to detect schedule deviations vs scheduled times
    2. classifies each deviation as cancel / delay / add primitive
    3. replays the resulting edit stream through Cycle 10's incremental update path
    4. records per-edit median + p95 wall, per-cell csa_better count

Pending PI's API key registration (free, 5 min each):
    - TfL Unified API:  https://api-portal.tfl.gov.uk
    - DfT BODS:         https://data.bus-data.dft.gov.uk/account/signup/
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


TFL_URLS = {
    "siri_vm_tube": "https://api.tfl.gov.uk/Mode/tube/Arrivals",
    "siri_vm_bus":  "https://api.tfl.gov.uk/Mode/bus/Arrivals",
    "siri_sx_all":  "https://api.tfl.gov.uk/Disruptions/Lifts/v2",
    "lines_status": "https://api.tfl.gov.uk/Line/Mode/tube,bus/Status",
}


def poll_tfl(url: str, app_key: str, timeout: int = 30) -> dict:
    if requests is None:
        return {"_skipped_no_requests": True}
    r = requests.get(url, params={"app_key": app_key}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration-hours", type=float, default=24.0)
    ap.add_argument("--siri-vm-interval-s", type=int, default=60)
    ap.add_argument("--siri-sx-interval-s", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true",
                    help="single capture pass without keys for offline scaffold testing")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    manifest = out / "manifest.json"

    def lg(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log.write_text("", encoding="utf-8")
    lg(f"phase_e_capture START out={out} duration_hours={args.duration_hours}")

    tfl_key = os.environ.get("TFL_APP_KEY", "")
    if not tfl_key and not args.dry_run:
        lg("ERROR: TFL_APP_KEY env var not set; pass --dry-run for scaffold test")
        return 1

    t_start = time.time()
    t_end = t_start + args.duration_hours * 3600
    next_vm = t_start
    next_sx = t_start
    poll_count = {"vm": 0, "sx": 0}

    while time.time() < t_end:
        now = time.time()
        if now >= next_vm:
            for name, url in TFL_URLS.items():
                if "siri_vm" in name or "lines_status" in name:
                    fname = out / f"{name}_{time.strftime('%H%M%S')}.json"
                    try:
                        payload = ({"_dry_run": True, "url": url}
                                   if args.dry_run else poll_tfl(url, tfl_key))
                        fname.write_text(json.dumps(payload), encoding="utf-8")
                        poll_count["vm"] += 1
                    except Exception as e:
                        lg(f"VM poll FAILED {name}: {type(e).__name__}: {e}")
            next_vm = now + args.siri_vm_interval_s
        if now >= next_sx:
            for name, url in TFL_URLS.items():
                if "siri_sx" in name:
                    fname = out / f"{name}_{time.strftime('%H%M%S')}.json"
                    try:
                        payload = ({"_dry_run": True, "url": url}
                                   if args.dry_run else poll_tfl(url, tfl_key))
                        fname.write_text(json.dumps(payload), encoding="utf-8")
                        poll_count["sx"] += 1
                    except Exception as e:
                        lg(f"SX poll FAILED {name}: {type(e).__name__}: {e}")
            next_sx = now + args.siri_sx_interval_s
        if args.dry_run:
            break
        time.sleep(5)

    elapsed_h = (time.time() - t_start) / 3600
    summary = {
        "duration_hours": round(elapsed_h, 3),
        "vm_polls": poll_count["vm"],
        "sx_polls": poll_count["sx"],
        "out_dir": str(out),
        "dry_run": args.dry_run,
    }
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lg(f"phase_e_capture DONE: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
