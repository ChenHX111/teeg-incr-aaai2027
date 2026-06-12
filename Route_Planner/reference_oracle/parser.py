"""Independent GTFS bundle parser for Reference-CSA*.

This module reads raw `.pkl` files from `MCRaptor_Data_*` directly using
only `pickle` + `pandas` + `numpy`. It does NOT import from
`Route_Planner.common.data_layer` or any other Route_Planner subpackage.

The output is a `RefBundle` dataclass — a minimal structured form that
the Reference-CSA* core (`csa_core.py`) consumes.

Independence properties:
- No `import Route_Planner.X` for X in {teeg, minute_grid, raptor, ultra, common}.
- Direct file I/O via pickle; CSV fallback for damaged pkl.
- All datetimes coerced to int16 minutes-since-SERVICE_START with explicit
  service-date detection (post-Phase-I-bug-class).
"""
from __future__ import annotations
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Constants — match the production data_layer for cross-validation but defined here independently
DEFAULT_SERVICE_DATE = pd.Timestamp("2025-01-03")  # subset reference day
DEFAULT_T_LO = 0
DEFAULT_T_HI = 780  # 13-hour service window in minutes


@dataclass(frozen=True)
class Connection:
    """One elementary timetable connection (vehicle leg between two stops)."""
    trip_id_int: int
    stop_seq_dep: int
    stop_seq_arr: int
    dep_min: int
    arr_min: int
    from_stop: int
    to_stop: int
    route_int: int
    mode: int  # 0=bus,1=subway,2=Tram,3=boat,4=walking


@dataclass(frozen=True)
class WalkEdge:
    """One direct walking footpath edge."""
    from_stop: int
    to_stop: int
    walk_min: int


@dataclass
class RefBundle:
    """The Reference-CSA* bundle. Independent of Route_Planner.common.data_layer.

    Connections are sorted by dep_min then by (trip_id_int, stop_seq_dep) for ties
    — this canonical order is what plain CSA assumes.
    """
    n_stops: int
    connections: np.ndarray  # structured dtype with the Connection fields
    walks: np.ndarray         # (W, 3) int32: from_stop, to_stop, walk_min
    service_date: pd.Timestamp
    t_lo: int
    t_hi: int
    trip_id_str_to_int: dict
    stop_int_to_name: dict
    stop_lat: dict
    stop_lon: dict
    n_trips: int
    n_routes: int
    bundle_sha: str = ""  # filled in by hashing.py after construction


# Canonical structured dtype for Connection — different field NAMES from
# Route_Planner.common.data_layer.CONN_DTYPE to make a co-import obviously wrong.
REF_CONN_DTYPE = np.dtype([
    ("trip_id_int", "i4"),
    ("stop_seq_dep", "i2"),
    ("stop_seq_arr", "i2"),
    ("dep_min", "i2"),
    ("arr_min", "i2"),
    ("from_stop", "i4"),
    ("to_stop", "i4"),
    ("route_int", "i4"),
    ("mode", "i1"),
])


def _detect_service_date(arr_dt: pd.Series, dep_dt: pd.Series) -> pd.Timestamp:
    """Detect the mode of valid (non-NaT) timestamps. Returns date at midnight."""
    combined = pd.concat([arr_dt.dropna(), dep_dt.dropna()])
    if len(combined) == 0:
        raise ValueError("parser: stop_times has all-NaT datetime columns")
    mode_date = pd.Series(combined.dt.normalize()).mode().iloc[0]
    return pd.Timestamp(mode_date)


def _pkl_load(p: Path) -> Any:
    """Direct pickle load, no shim."""
    with p.open("rb") as f:
        return pickle.load(f)


def load_ref_bundle(
    data_dir: str | Path,
    *,
    service_date: pd.Timestamp | None = None,
    t_lo: int = DEFAULT_T_LO,
    t_hi: int = DEFAULT_T_HI,
) -> RefBundle:
    """Independent loader. Reads:
    - stop_times_df.pkl  (DataFrame)
    - transfer_array_NewWalk.pkl  (numpy 2D array of (from, to, walk_seconds))

    DOES NOT call `Route_Planner.common.load_bundle()`.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"parser: data_dir {data_dir} not found")

    # Load DataFrame
    df = _pkl_load(data_dir / "stop_times_df.pkl")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"parser: expected DataFrame, got {type(df).__name__}")

    # Coerce datetime columns (post-Phase-I-bug-class: pkl may have str instead of datetime64)
    for col in ("arrival_time", "departure_time"):
        if df[col].dtype.kind != "M":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Detect service date (or use provided)
    detected_date = _detect_service_date(df["arrival_time"], df["departure_time"])
    if service_date is None:
        service_date = detected_date
        # Snap to subset's reference date to match production stop_times_min computation
        if service_date != DEFAULT_SERVICE_DATE:
            # We do NOT shift the DataFrame here — we just compute minutes relative to detected date
            pass
    service_start = pd.Timestamp(service_date) + pd.Timedelta(hours=7)  # SERVICE_START = date + 07:00

    # Build stop_int_to_name + lat/lon from the unique stops
    stops_df = df.drop_duplicates("stop_int")
    stop_int_to_name = {int(r["stop_int"]): str(r.get("stop_name", f"stop_{int(r['stop_int'])}"))
                        for _, r in stops_df.iterrows()}
    # lat/lon may be strings; coerce
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return float("nan")
    stop_lat = {int(r["stop_int"]): _to_float(r.get("stop_lat")) for _, r in stops_df.iterrows()}
    stop_lon = {int(r["stop_int"]): _to_float(r.get("stop_lon")) for _, r in stops_df.iterrows()}
    n_stops = max(stop_int_to_name) + 1

    # Build trip_id_str_to_int mapping (deterministic, sorted by first occurrence)
    seen = []
    seen_set = set()
    for tid in df["trip_id"].tolist():
        if tid not in seen_set:
            seen.append(tid)
            seen_set.add(tid)
    trip_id_str_to_int = {tid: i for i, tid in enumerate(seen)}
    n_trips = len(trip_id_str_to_int)

    # Extract connections (one per (trip, stop_sequence_i, stop_sequence_i+1))
    cols_needed = ["trip_id", "stop_sequence", "stop_int", "departure_time",
                   "arrival_time", "route_int", "Mode"]
    g = df[cols_needed].sort_values(["trip_id", "stop_sequence"],
                                     kind="mergesort").reset_index(drop=True)
    trip_arr = g["trip_id"].values
    next_same_trip = np.roll(trip_arr, -1) == trip_arr
    next_same_trip[-1] = False

    dep_curr = g["departure_time"].values
    arr_next = np.roll(g["arrival_time"].values, -1)
    seq_curr = g["stop_sequence"].values.astype(np.int32)
    seq_next = np.roll(seq_curr, -1)
    stop_curr = g["stop_int"].values.astype(np.int32)
    stop_next = np.roll(stop_curr, -1)
    route_curr = g["route_int"].values.astype(np.int32)
    MODE_MAP = {"bus": 0, "subway": 1, "Tram": 2, "boat": 3, "walking": 4}
    mode_curr = np.array([MODE_MAP.get(m, 0) for m in g["Mode"].values], dtype=np.int8)

    # Compute minutes since SERVICE_START
    dep_dt = pd.to_datetime(dep_curr)
    arr_dt = pd.to_datetime(arr_next)
    dep_min = ((dep_dt - service_start).total_seconds() // 60).values
    arr_min = ((arr_dt - service_start).total_seconds() // 60).values

    valid = next_same_trip & ~np.isnan(dep_min) & ~np.isnan(arr_min)

    # Build the connections structured array
    idx = np.where(valid)[0]
    n_conn = len(idx)
    conns = np.zeros(n_conn, dtype=REF_CONN_DTYPE)
    conns["trip_id_int"]  = np.array([trip_id_str_to_int[t] for t in trip_arr[idx]], dtype=np.int32)
    conns["stop_seq_dep"] = seq_curr[idx].astype(np.int16)
    conns["stop_seq_arr"] = seq_next[idx].astype(np.int16)
    conns["dep_min"]      = np.clip(dep_min[idx], -32768, 32767).astype(np.int16)
    conns["arr_min"]      = np.clip(arr_min[idx], -32768, 32767).astype(np.int16)
    conns["from_stop"]    = stop_curr[idx]
    conns["to_stop"]      = stop_next[idx]
    conns["route_int"]    = route_curr[idx]
    conns["mode"]         = mode_curr[idx]

    # Sort by (dep_min, trip_id_int, stop_seq_dep) — canonical CSA order
    order = np.lexsort((conns["stop_seq_dep"], conns["trip_id_int"], conns["dep_min"]))
    conns = conns[order]

    n_routes = int(g["route_int"].max() + 1) if len(g) > 0 else 0

    # Load walking transfers — file format is (W, 2) uint16 = (target, walk_seconds)
    # FROM-stop is encoded via the stops_array CSR-style index (cols 2,3 = tr_lo, tr_hi)
    transfers = _pkl_load(data_dir / "transfer_array_NewWalk.pkl")
    stops_array_csr = _pkl_load(data_dir / "stops_array_NewWalk.pkl")
    if transfers.ndim != 2 or transfers.shape[1] != 2:
        raise ValueError(f"parser: transfer_array shape {transfers.shape} != (W, 2)")
    if stops_array_csr.ndim != 2 or stops_array_csr.shape[1] != 4:
        raise ValueError(f"parser: stops_array shape {stops_array_csr.shape} != (S, 4)")
    # Reconstruct (from, to, walk_seconds) triples via CSR walk-back
    tr_rows = []
    for s in range(stops_array_csr.shape[0]):
        tr_lo = int(stops_array_csr[s, 2])
        tr_hi = int(stops_array_csr[s, 3])
        for k in range(tr_lo, tr_hi):
            tgt = int(transfers[k, 0])
            secs = int(transfers[k, 1])
            tr_rows.append((s, tgt, secs))
    walks_sec = np.asarray(tr_rows, dtype=np.int32) if tr_rows else np.zeros((0, 3), dtype=np.int32)
    walks_min = np.zeros_like(walks_sec)
    walks_min[:, 0] = walks_sec[:, 0]
    walks_min[:, 1] = walks_sec[:, 1]
    walks_min[:, 2] = (walks_sec[:, 2] + 59) // 60  # ceil to minutes

    return RefBundle(
        n_stops=int(n_stops),
        connections=conns,
        walks=walks_min,
        service_date=service_date,
        t_lo=t_lo,
        t_hi=t_hi,
        trip_id_str_to_int=trip_id_str_to_int,
        stop_int_to_name=stop_int_to_name,
        stop_lat=stop_lat,
        stop_lon=stop_lon,
        n_trips=n_trips,
        n_routes=n_routes,
    )
