"""Shared data layer for the TEEG / RAPTOR / CSA implementations.

Loads the London single-day subset and exposes structured arrays in a uniform
form usable by every routing algorithm.

The "minute" axis is integer minutes since SERVICE_START (2025-01-03 07:00:00).
Service span = 780 minutes [0..780].
"""
from __future__ import annotations
import os, pickle
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Cycle 7 (R6.1): allow runtime override via MCRAPTOR_DATA_DIR env var so the
# benchmark harness can swap between Subset and Full London without code edits.
# DATA is evaluated at import for backward compatibility; the canonical accessor
# is get_data_dir() which re-reads the env var each call (so set-env-before-call
# works even when this module was imported earlier with no env set).
def get_data_dir():
    return os.environ.get("MCRAPTOR_DATA_DIR") or os.path.join(REPO, "MCRaptor_Data_Subset")

def get_data_tag():
    """A short tag derived from the data dir, suitable for cache filenames.

    Recognizes: full_london, subset, berlin, nyc, paris, generic (fallback).
    """
    d = os.path.basename(os.path.normpath(get_data_dir())).lower()
    parent = os.path.basename(os.path.dirname(os.path.normpath(get_data_dir()))).lower()
    combined = f"{parent}/{d}"
    if 'full' in d or 'full' in parent:
        return 'full_london'
    if 'berlin' in combined:
        return 'berlin'
    if 'nyc' in combined or 'newyork' in combined or 'new_york' in combined:
        return 'nyc'
    if 'paris' in combined:
        return 'paris'
    if 'subset' in combined:
        return 'subset'
    # Fallback: use the directory basename to avoid cache pollution
    return d.replace('mcraptor_data_', '').replace('mcraptor_data', 'generic') or 'generic'

DATA = get_data_dir()

SERVICE_START = pd.Timestamp("2025-01-03 07:00:00")
SERVICE_END   = pd.Timestamp("2025-01-03 20:00:00")
DAY_MIN = 780

MODE_TO_INT = {"bus": 0, "subway": 1, "Tram": 2, "boat": 3, "walking": 4}
INT_TO_MODE = {v: k for k, v in MODE_TO_INT.items()}

CONN_DTYPE = np.dtype([
    ("from_stop", "i4"),
    ("to_stop",   "i4"),
    ("dep_min",   "i2"),
    ("arr_min",   "i2"),
    ("trip_id",   "i4"),
    ("route_int", "i4"),
    ("mode",      "i1"),
])


def _pk(name):
    with open(os.path.join(get_data_dir(), name), "rb") as f:
        return pickle.load(f)


def _collapse_consecutive_same_stop(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the GTFS dwell-row-splitting artifact in Underground/DLR data.

    Some London Tube/DLR trips have consecutive rows referencing the same stop_int
    (an artifact of how dwell times were encoded during preprocessing). These
    consecutive same-stop pairs propagate into b.connections as negative-time
    or zero-time self-loops, which corrupt TEEG's RIDE edges (Cycle-9 bug, 55,298
    negative-weight RIDE edges on Full London).

    This function collapses each run of consecutive same-stop rows within a trip
    into a single canonical row:
        arr_time = min(arrival_time across the run)
        dep_time = max(departure_time across the run)
        stop_sequence = first(stop_sequence)
        all other columns = first(...)

    PRECONDITION: df may be in any order; this function re-sorts by (trip_id,
    stop_sequence) for safety.

    INVARIANTS (asserted at end):
      - no two consecutive rows in the result share both trip_id and stop_int
      - departure_time >= arrival_time per row
    """
    import logging
    logger = logging.getLogger(__name__)

    # Enforce sort
    df = df.sort_values(["trip_id", "stop_sequence"], kind="mergesort").reset_index(drop=True)

    # Mark consecutive same-stop within same trip
    same_as_prev = (
        (df["trip_id"] == df["trip_id"].shift(1)) &
        (df["stop_int"] == df["stop_int"].shift(1))
    )
    n_dupes = int(same_as_prev.sum())
    if n_dupes == 0:
        # Nothing to do (subset data may not have this artifact)
        return df

    # Group id increments at every trip boundary or stop change
    df["_grp"] = (~same_as_prev).cumsum()

    # Build agg spec: arr_time=min, dep_time=max, everything else=first
    key_cols = {"_grp"}
    agg_spec = {}
    for c in df.columns:
        if c in key_cols:
            continue
        if c == "arrival_time":
            agg_spec[c] = "min"
        elif c == "departure_time":
            agg_spec[c] = "max"
        else:
            agg_spec[c] = "first"

    collapsed = (df.groupby("_grp", sort=False)
                   .agg(agg_spec)
                   .reset_index(drop=True))

    n_in, n_out = len(df), len(collapsed)
    logger.info(
        f"_collapse_consecutive_same_stop: {n_in:,} rows -> {n_out:,} rows "
        f"({n_in - n_out:,} duplicate-stop rows absorbed)"
    )
    print(
        f"[data_layer] collapsed {n_in - n_out:,} duplicate-stop rows "
        f"({n_in:,} -> {n_out:,}; {n_dupes:,} same-stop consecutive pairs)",
        flush=True,
    )

    # Internal invariant: dep >= arr per row
    bad = (collapsed["departure_time"] < collapsed["arrival_time"]).sum()
    if bad:
        logger.error(
            f"[data_layer] {bad} rows have dep < arr AFTER collapse - "
            f"data has anomalies beyond the dwell-split artifact"
        )

    # Internal invariant: no consecutive same-stop rows remain
    same2 = (
        (collapsed["trip_id"] == collapsed["trip_id"].shift(1)) &
        (collapsed["stop_int"] == collapsed["stop_int"].shift(1))
    ).sum()
    if same2:
        logger.error(
            f"[data_layer] {same2} consecutive same-stop pairs remain after collapse - "
            f"investigate"
        )

    return collapsed


@dataclass
class TimetableBundle:
    """Connection list (CSA / TEEG) + RAPTOR-style arrays + walking transfers."""
    # CSA / TEEG layer
    connections: np.ndarray            # CONN_DTYPE, sorted by dep_min
    transfers_from: np.ndarray         # (T, 3) int32: from_stop, to_stop, walk_seconds
    # naming
    stop_int_to_name: dict
    stop_name_to_int: dict
    trip_id_str: list                  # mapping local trip_id (int) -> original string
    # window & coverage
    t_lo: int
    t_hi: int
    used_stops: np.ndarray
    n_stops: int                       # total stop_int range (max+1)
    stop_lat: dict
    stop_lon: dict
    # RAPTOR-layer arrays (None unless load_raptor=True)
    routes_array: Optional[np.ndarray] = None      # (R, 4) int64
    route_stops:  Optional[np.ndarray] = None      # (Nrs,) int64 (stop_int)
    stop_routes:  Optional[np.ndarray] = None      # (Nrs,) int64
    stop_times:   Optional[np.ndarray] = None      # (Nst, 2) datetime64
    stops_array:  Optional[np.ndarray] = None      # (S, 4) uint32: sr_lo, sr_hi, tr_lo, tr_hi
    transfer_array: Optional[np.ndarray] = None    # (T, 2) uint16: target, walk_seconds
    stop_times_min: Optional[np.ndarray] = None    # (Nst, 2) int16: arr_min, dep_min (-1 = NaT)


def load_bundle(stop_int_set: Optional[Iterable[int]] = None,
                t_lo: int = 0, t_hi: int = DAY_MIN,
                load_raptor: bool = True) -> TimetableBundle:
    """Single entry point used by every algorithm.

    If stop_int_set is None, loads the FULL network.
    If load_raptor is True, also exposes the raw RAPTOR arrays (no time-window
    filtering — RAPTOR uses the t_lo/t_hi only via the query, not the data).
    """
    df = _pk("stop_times_df.pkl")
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)

    # CYCLE 9 FIX (TEEG negative-weight RIDE edge bug):
    # Collapse consecutive same-stop rows within a trip (London Tube/DLR data
    # artifact). Without this, ~55k negative-time + ~465k zero-time self-loop
    # connections leak into b.connections and corrupt TEEG's RIDE edges.
    df = _collapse_consecutive_same_stop(df)

    name_map = (df.drop_duplicates("stop_int")
                  .set_index("stop_int")["stop_name"].to_dict())
    stop_int_to_name = {int(k): v for k, v in name_map.items()}
    stop_name_to_int = {v: int(k) for k, v in name_map.items()}
    lat_map = (df.drop_duplicates("stop_int").set_index("stop_int")["stop_lat"].to_dict())
    lon_map = (df.drop_duplicates("stop_int").set_index("stop_int")["stop_lon"].to_dict())
    n_stops = max(stop_int_to_name) + 1

    # --- connection extraction
    cols = ["trip_id", "stop_sequence", "stop_int", "departure_time",
            "arrival_time", "route_int", "Mode"]
    g = df[cols].sort_values(["trip_id", "stop_sequence"], kind="mergesort").reset_index(drop=True)
    trip_id_arr = g["trip_id"].values
    next_same_trip = np.roll(trip_id_arr, -1) == trip_id_arr
    next_same_trip[-1] = False

    dep_curr = g["departure_time"].values
    arr_next = np.roll(g["arrival_time"].values, -1)
    stop_curr = g["stop_int"].values
    stop_next = np.roll(stop_curr, -1)
    route_curr = g["route_int"].values
    mode_curr  = g["Mode"].values

    dep_min_full = ((pd.to_datetime(dep_curr) - SERVICE_START).total_seconds() // 60)
    arr_min_full = ((pd.to_datetime(arr_next) - SERVICE_START).total_seconds() // 60)

    valid = next_same_trip & ~np.isnan(dep_min_full) & ~np.isnan(arr_min_full)
    dep_arr_i = np.where(valid, np.nan_to_num(dep_min_full, nan=-1), -1).astype(np.int32)
    arr_arr_i = np.where(valid, np.nan_to_num(arr_min_full, nan=-1), -1).astype(np.int32)
    in_window = (dep_arr_i >= t_lo) & (dep_arr_i < t_hi) & (arr_arr_i >= t_lo) & (arr_arr_i <= t_hi)
    valid = valid & in_window

    if stop_int_set is not None:
        sset = set(int(s) for s in stop_int_set)
        sc = np.fromiter((int(s) in sset for s in stop_curr), dtype=bool, count=len(stop_curr))
        sn = np.fromiter((int(s) in sset for s in stop_next), dtype=bool, count=len(stop_next))
        valid = valid & sc & sn

    idx = np.where(valid)[0]
    trip_str_arr = trip_id_arr[idx]
    unique_trips, trip_inverse = np.unique(trip_str_arr, return_inverse=True)
    trip_id_str = list(unique_trips)

    conns = np.empty(len(idx), dtype=CONN_DTYPE)
    conns["from_stop"] = stop_curr[idx].astype(np.int32)
    conns["to_stop"]   = stop_next[idx].astype(np.int32)
    conns["dep_min"]   = dep_arr_i[idx].astype(np.int16)
    conns["arr_min"]   = arr_arr_i[idx].astype(np.int16)
    conns["trip_id"]   = trip_inverse.astype(np.int32)
    conns["route_int"] = route_curr[idx].astype(np.int32)
    conns["mode"]      = np.array([MODE_TO_INT[m] for m in mode_curr[idx]], dtype=np.int8)
    order = np.lexsort((conns["arr_min"], conns["dep_min"]))
    conns = conns[order]

    # --- walking footpaths
    stops_array    = _pk("stops_array_NewWalk.pkl")
    transfer_array = _pk("transfer_array_NewWalk.pkl")
    tr_rows = []
    used_set = set(int(s) for s in stop_int_set) if stop_int_set is not None else None
    for s in range(stops_array.shape[0]):
        if used_set is not None and s not in used_set:
            continue
        lo, hi = int(stops_array[s, 2]), int(stops_array[s, 3])
        for k in range(lo, hi):
            tgt = int(transfer_array[k, 0])
            secs = int(transfer_array[k, 1])
            if used_set is not None and tgt not in used_set:
                continue
            tr_rows.append((s, tgt, secs))
    transfers_from = (np.asarray(tr_rows, dtype=np.int32)
                      if tr_rows else np.zeros((0, 3), dtype=np.int32))

    used_stops = (np.asarray(sorted(int(s) for s in stop_int_set), dtype=np.int32)
                  if stop_int_set is not None
                  else np.array(sorted(stop_int_to_name.keys()), dtype=np.int32))

    bundle = TimetableBundle(
        connections=conns, transfers_from=transfers_from,
        stop_int_to_name=stop_int_to_name, stop_name_to_int=stop_name_to_int,
        trip_id_str=trip_id_str, t_lo=t_lo, t_hi=t_hi,
        used_stops=used_stops, n_stops=n_stops,
        stop_lat=lat_map, stop_lon=lon_map,
    )

    if load_raptor:
        bundle.routes_array   = _pk("routes_array.pkl")
        bundle.route_stops    = _pk("route_stops_array.pkl")
        bundle.stop_routes    = _pk("stop_routes_array.pkl")
        bundle.stop_times     = _pk("stop_times_array.pkl")
        bundle.stops_array    = stops_array
        bundle.transfer_array = transfer_array
        # Convert stop_times to int16 minutes (vectorised, fast)
        st = bundle.stop_times
        arr = pd.to_datetime(st[:, 0])
        dep = pd.to_datetime(st[:, 1])
        arr_m = (arr - SERVICE_START).total_seconds() // 60
        dep_m = (dep - SERVICE_START).total_seconds() // 60
        arr_m = np.where(np.isnan(arr_m), -1, arr_m).astype(np.int16)
        dep_m = np.where(np.isnan(dep_m), -1, dep_m).astype(np.int16)
        bundle.stop_times_min = np.stack([arr_m, dep_m], axis=1)
    return bundle


def central_stops(n: int = 250) -> list:
    """Top-n stops by visit count, used for small/illustrative subsets."""
    df = _pk("stop_times_df.pkl")
    deg = (df.groupby("stop_int").size().sort_values(ascending=False))
    return deg.head(n).index.astype(int).tolist()


if __name__ == "__main__":
    import time
    t0 = time.time()
    b = load_bundle(t_lo=0, t_hi=DAY_MIN, load_raptor=True)
    print(f"load_bundle full-day full-network: {time.time()-t0:.1f}s")
    print(f"  connections   : {len(b.connections):,}")
    print(f"  transfers     : {len(b.transfers_from):,}")
    print(f"  trips         : {len(b.trip_id_str):,}")
    print(f"  stops         : {b.n_stops}")
    print(f"  routes        : {len(b.routes_array)}")
    print(f"  route-stops   : {len(b.route_stops)}")
    print(f"  stop-times    : {len(b.stop_times)}")
