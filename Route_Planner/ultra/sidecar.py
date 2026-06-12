"""ULTRA cache .meta.json sidecar — schema-pinned reload validation.

Per Stage 2 F.3 freeze (sidecar schema) + B3 action ledger:
- 15 fields, all required, with sha256 canonical hashes over the
  underlying arrays that affect shortcut content.
- Validation on load: any mismatch raises CacheStaleness exception so the
  caller can decide to rebuild or fail loudly.

NO truncated hashes (full sha256 hex); walk_params explicitly hashed
(Stage 2 DO-NOT #8); route/stop/timetable/transfer arrays all in the
canonical-hash input (Stage 2 DO-NOT #7).
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "ultra_shortcuts_v1"


@dataclass
class UltraSidecar:
    """The .meta.json sidecar schema (Stage 2 F.3 frozen fields)."""
    schema_version: str        # SCHEMA_VERSION
    artifact_type: str         # "ultra_shortcuts_csr" (never "closed_walks_csr")
    n_stops: int
    stop_index_hash: str       # sha256 of bundle.used_stops (or stops_array contents)
    timetable_hash: str        # sha256 of (routes_array + route_stops + stop_routes + stop_times_min)
    transfer_hash: str         # sha256 of bundle.transfers_from
    walk_params: dict          # max_walk_min, walking_speed_ms, walk_cap_m (whatever the build used)
    walk_params_hash: str      # sha256 of canonicalised walk_params JSON
    build_version: str         # "v1.0-cycle6-stage4" or git SHA if available
    edge_count: int            # post-pruning shortcut count
    build_time_s: float        # wall-clock of build_ultra_shortcuts
    python_version: str
    numpy_version: str
    build_started_utc: str     # ISO 8601
    build_finished_utc: str    # ISO 8601


class CacheStaleness(Exception):
    """Raised when a sidecar's hash fields do not match the live bundle."""


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        # Use np.ascontiguousarray + tobytes for deterministic bytes
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
    return h.hexdigest()


def _sha256_canonical_json(obj: Any) -> str:
    """sha256 of a canonical JSON serialisation (sorted keys, no whitespace)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_hashes(b, walk_params: dict) -> dict:
    """Compute the 4 hash fields from a TimetableBundle.

    Used both at build time and at reload time; equality of all 4 hashes
    means the cached shortcut set is valid for the current bundle.
    """
    return {
        "stop_index_hash": _sha256_arrays(b.used_stops),
        "timetable_hash": _sha256_arrays(
            b.routes_array, b.route_stops, b.stop_routes, b.stop_times_min,
        ),
        "transfer_hash": _sha256_arrays(b.transfers_from),
        "walk_params_hash": _sha256_canonical_json(walk_params),
    }


def make_sidecar(b, walk_params: dict, edge_count: int,
                 build_started: float, build_finished: float,
                 build_version: str = "v1.0-cycle6-stage4") -> UltraSidecar:
    hashes = compute_hashes(b, walk_params)
    return UltraSidecar(
        schema_version=SCHEMA_VERSION,
        artifact_type="ultra_shortcuts_csr",
        n_stops=int(b.n_stops),
        stop_index_hash=hashes["stop_index_hash"],
        timetable_hash=hashes["timetable_hash"],
        transfer_hash=hashes["transfer_hash"],
        walk_params=walk_params,
        walk_params_hash=hashes["walk_params_hash"],
        build_version=build_version,
        edge_count=int(edge_count),
        build_time_s=float(build_finished - build_started),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        numpy_version=np.__version__,
        build_started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(build_started)),
        build_finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(build_finished)),
    )


def write_sidecar(path: Path, sidecar: UltraSidecar) -> None:
    Path(path).write_text(
        json.dumps(asdict(sidecar), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_sidecar(path: Path) -> UltraSidecar:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return UltraSidecar(**data)


def validate_sidecar_against_bundle(sidecar: UltraSidecar, b, walk_params: dict) -> None:
    """Raises CacheStaleness if the sidecar is not valid for this bundle.

    Checks all 4 hashes + walk_params dict equality. Per Stage 2 DO-NOT #7,
    DO-NOT #8, no truncation; full sha256 comparison.
    """
    live = compute_hashes(b, walk_params)
    mismatches = []
    if sidecar.schema_version != SCHEMA_VERSION:
        mismatches.append(f"schema_version: cache={sidecar.schema_version} live={SCHEMA_VERSION}")
    if sidecar.artifact_type != "ultra_shortcuts_csr":
        mismatches.append(f"artifact_type: cache={sidecar.artifact_type} (expected ultra_shortcuts_csr)")
    if sidecar.n_stops != int(b.n_stops):
        mismatches.append(f"n_stops: cache={sidecar.n_stops} live={int(b.n_stops)}")
    if sidecar.stop_index_hash != live["stop_index_hash"]:
        mismatches.append(f"stop_index_hash mismatch")
    if sidecar.timetable_hash != live["timetable_hash"]:
        mismatches.append(f"timetable_hash mismatch")
    if sidecar.transfer_hash != live["transfer_hash"]:
        mismatches.append(f"transfer_hash mismatch")
    if sidecar.walk_params != walk_params:
        mismatches.append(f"walk_params dict mismatch: cache={sidecar.walk_params} live={walk_params}")
    if sidecar.walk_params_hash != live["walk_params_hash"]:
        mismatches.append(f"walk_params_hash mismatch")
    if mismatches:
        raise CacheStaleness("ULTRA cache stale; " + "; ".join(mismatches))
