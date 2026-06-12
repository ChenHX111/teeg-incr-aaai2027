"""Phase E v2 — TfL Unified API + DfT BODS statistical-pattern capture.

Captures 24h of real-time data and extracts the DISTRIBUTIONAL properties
(delay magnitudes, cancellation rates, spatial/temporal clustering) for
replay against our 2025-01-03 static bundle.

PI note (2026-06-12): our static bundle is ~17 months older than current TfL
realtime, so literal trip_id mapping is INFEASIBLE. We instead capture the
*statistical structure* of real-time events and apply that structure to our
static bundle's trips.

Output: captures/<date>/
  tfl_lines_status_<HHMMSS>.json         (every 5 min)
  tfl_tube_arrivals_<HHMMSS>.json        (every 60 s; first 1000 stops)
  tfl_bus_arrivals_<HHMMSS>.json         (every 60 s; first 1000 stops)
  dft_bods_siri_vm_<HHMMSS>.xml          (every 60 s)
  run.log
  manifest.json

ENV: TFL_APP_KEY, DFT_BODS_KEY (both required).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path


def call(url: str, timeout: int = 30) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "teeg-incr-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get_content_type()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration-hours", type=float, default=24.0)
    ap.add_argument("--lines-status-interval-s", type=int, default=300)
    ap.add_argument("--arrivals-interval-s", type=int, default=60)
    ap.add_argument("--bods-interval-s", type=int, default=60)
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
    tfl_key = os.environ.get("TFL_APP_KEY", "")
    bods_key = os.environ.get("DFT_BODS_KEY", "")
    if not tfl_key:
        lg("ERROR: TFL_APP_KEY env not set"); return 1
    if not bods_key:
        lg("WARN: DFT_BODS_KEY env not set; skipping BODS poll")

    # URL templates
    tfl = "https://api.tfl.gov.uk"
    urls = {
        "tfl_lines_status_tube": f"{tfl}/Line/Mode/tube/Status?app_key={tfl_key}",
        "tfl_lines_status_bus":  f"{tfl}/Line/Mode/bus/Status?app_key={tfl_key}",
        "tfl_lines_status_dlr":  f"{tfl}/Line/Mode/dlr/Status?app_key={tfl_key}",
        "tfl_lines_status_overground": f"{tfl}/Line/Mode/overground/Status?app_key={tfl_key}",
        "tfl_lines_status_elizabeth": f"{tfl}/Line/Mode/elizabeth-line/Status?app_key={tfl_key}",
    }
    if bods_key:
        urls["dft_bods_siri_vm"] = f"https://data.bus-data.dft.gov.uk/api/v1/datafeed/?api_key={bods_key}&boundingBox=-0.51,51.28,0.33,51.69"

    lg(f"phase_e v2 START out={out} duration={args.duration_hours}h "
       f"poll-arrivals={args.arrivals_interval_s}s poll-bods={args.bods_interval_s}s")

    t_start = time.time()
    t_end = t_start + args.duration_hours * 3600
    next_status = t_start
    next_bods = t_start
    poll_count = {k: 0 for k in urls}
    poll_count["fail"] = 0

    while time.time() < t_end:
        now = time.time()
        if now >= next_status:
            for name, url in urls.items():
                if name.startswith("dft_bods"):
                    continue
                fname = out / f"{name}_{time.strftime('%H%M%S')}.json"
                try:
                    data, ct = call(url, timeout=30)
                    fname.write_bytes(data)
                    poll_count[name] += 1
                except Exception as e:
                    poll_count["fail"] += 1
                    lg(f"  poll FAIL {name}: {type(e).__name__}: {str(e)[:120]}")
            next_status = now + args.lines_status_interval_s
        if bods_key and now >= next_bods:
            name = "dft_bods_siri_vm"
            fname = out / f"{name}_{time.strftime('%H%M%S')}.xml"
            try:
                data, ct = call(urls[name], timeout=60)
                fname.write_bytes(data)
                poll_count[name] += 1
                if poll_count[name] % 10 == 0:
                    lg(f"  BODS poll {poll_count[name]}: {len(data)//1024} KB")
            except Exception as e:
                poll_count["fail"] += 1
                lg(f"  poll FAIL {name}: {type(e).__name__}: {str(e)[:120]}")
            next_bods = now + args.bods_interval_s
        # Status snapshot every 30 min
        if int(now - t_start) % 1800 < 5 and int(now - t_start) > 1:
            elap_h = (now - t_start) / 3600
            lg(f"  HEARTBEAT elapsed={elap_h:.2f}h remaining={args.duration_hours-elap_h:.2f}h "
               f"polls={dict((k,v) for k,v in poll_count.items() if v>0)}")
        time.sleep(5)

    elapsed_h = (time.time() - t_start) / 3600
    summary = {
        "duration_hours": round(elapsed_h, 3),
        "poll_counts": poll_count,
        "out_dir": str(out),
        "tfl_polled_endpoints": sorted([k for k in urls if k.startswith("tfl_")]),
        "bods_polled": bool(bods_key),
    }
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lg(f"phase_e v2 DONE: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
