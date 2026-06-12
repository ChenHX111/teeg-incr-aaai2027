"""Validate / manifest emitter for Reference-CSA*.

- validate_bundle: post-Phase-I-bug-class assertion (mirrors dynamic_updates.validate_bundle
  but uses RefBundle's own field layout).
- emit_manifest: write a JSON manifest with full provenance + SHA-256s.
"""
from __future__ import annotations
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .parser import RefBundle


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path | str) -> str:
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256_bytes(canon.encode("utf-8"))


def sha256_ndarray(arr: np.ndarray) -> str:
    """SHA-256 over numpy array bytes (incl. dtype + shape header)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def validate_bundle(b: RefBundle) -> dict:
    """Assert bundle is sane. Returns provenance dict or raises ValueError."""
    if b is None:
        raise ValueError("validate_bundle: bundle is None")
    if b.n_stops <= 0:
        raise ValueError(f"validate_bundle: n_stops = {b.n_stops}")
    if len(b.connections) <= 0:
        raise ValueError(f"validate_bundle: n_connections = {len(b.connections)}")

    dep = b.connections["dep_min"]
    arr = b.connections["arr_min"]
    # Sanity: int16 range and no underflow signature
    if dep.dtype != np.int16:
        raise ValueError(f"validate_bundle: dep_min dtype {dep.dtype} != int16")
    if arr.dtype != np.int16:
        raise ValueError(f"validate_bundle: arr_min dtype {arr.dtype} != int16")
    valid_mask = (dep >= -1) & (dep <= b.t_hi + 200) & (arr >= -1) & (arr <= b.t_hi + 200)
    valid_frac = float(valid_mask.mean())
    if valid_frac < 0.9:
        raise ValueError(
            f"validate_bundle: only {valid_frac*100:.1f}% of dep/arr in [-1, {b.t_hi+200}]; "
            f"dep [{dep.min()}, {dep.max()}], arr [{arr.min()}, {arr.max()}] "
            f"— possible service-day mismatch (Phase-I-bug class)"
        )

    # CSA-core invariant: connections must be sorted by dep_min ascending.
    # Early termination `if c_dep >= tau_best[dst]: break` depends on this.
    if len(dep) >= 2:
        diff = np.diff(dep.astype(np.int32))
        n_inversions = int((diff < 0).sum())
        if n_inversions > 0:
            raise ValueError(
                f"validate_bundle: connections not sorted by dep_min "
                f"({n_inversions} inversions out of {len(dep)-1} adjacent pairs); "
                f"CSA early-termination invariant violated"
            )

    # Per-connection sanity: arr_min >= dep_min (no clock-wrap or service-date bug).
    # OVERNIGHT-WRAP HANDLING: GTFS data routinely includes services that cross
    # midnight (e.g., dep=23:55, arr=00:05 next day = stored as dep=1435, arr=-1435
    # post int16 clip). These are LEGITIMATE wrap-around services, not bugs.
    # We detect them: if arr < dep AND (arr + 1440 >= dep), it's overnight wrap.
    # Only true inversions (arr < dep AND arr + 1440 < dep) are flagged as bugs.
    raw_inverted_mask = (arr < dep)
    raw_inverted = int(raw_inverted_mask.sum())
    if raw_inverted > 0:
        # Among the raw inversions, count true bugs (arr+1440 < dep) vs overnight wrap
        arr_plus_day = arr.astype(np.int32) + 1440
        true_bug_mask = raw_inverted_mask & (arr_plus_day < dep.astype(np.int32))
        true_bugs = int(true_bug_mask.sum())
    else:
        true_bugs = 0
    bug_frac = float(true_bugs) / max(1, len(dep))
    if bug_frac > 0.01:
        raise ValueError(
            f"validate_bundle: {true_bugs}/{len(dep)} ({bug_frac*100:.2f}%) connections "
            f"have arr_min < dep_min EVEN AFTER adding 1440 (24h) — true clock-wrap / "
            f"service-date bug suspected (not overnight wraparound)"
        )

    return {
        "n_stops": int(b.n_stops),
        "n_connections": int(len(b.connections)),
        "n_trips": int(b.n_trips),
        "n_routes": int(b.n_routes),
        "n_walks": int(len(b.walks)),
        "service_date": str(b.service_date.date()),
        "valid_frac": valid_frac,
        "dep_min_range": [int(dep.min()), int(dep.max())],
        "arr_min_range": [int(arr.min()), int(arr.max())],
        "n_sort_inversions": 0,
        "n_arr_lt_dep_raw": raw_inverted,           # incl. overnight wrap (legitimate)
        "n_arr_lt_dep_true_bug": true_bugs,         # post-1440-offset inversions
    }


def emit_manifest(
    out_path: Path | str,
    *,
    role: str,
    method: str,
    bundle: RefBundle,
    n_queries: int,
    elapsed_s: float,
    extra: dict | None = None,
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_sha = sha256_json({
        "n_stops": bundle.n_stops,
        "n_connections": len(bundle.connections),
        "n_trips": bundle.n_trips,
        "service_date": str(bundle.service_date.date()),
        "connections_sha": sha256_ndarray(bundle.connections),
        "walks_sha": sha256_ndarray(bundle.walks),
    })
    manifest = {
        "role": role,
        "method": method,
        "pid": os.getpid(),
        "process_start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "bundle_sha": bundle_sha,
        "n_queries": n_queries,
        "elapsed_s": elapsed_s,
        "oracle_version": "ref_csa_star_v0_1_0_cycle9_day1",
    }
    if extra:
        manifest["extra"] = extra
    out_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest
