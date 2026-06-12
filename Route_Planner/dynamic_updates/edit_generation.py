"""Edit-set generation per scenarios DELAY / CANCELLATION / ADDITION.

Pre-registered constants (CYCLE_10_PREREGISTRATION.md):
- DELAY: uniform per-trip integer Δ in [1, 5] minutes
- CANCELLATION: whole-trip
- ADDITION: window-construction (audit_3 reviewer-proof protocol)

Deterministic: given (scenario, intensity, seed, raw_gtfs_sha), the
returned EditSet is identical run-to-run.
"""
from __future__ import annotations
import numpy as np
from typing import Any

from .schemas import EditType, EditSet, TripEdit


DELAY_MIN_RANGE = (1, 5)
MIN_GAP_MIN = 5
MIN_WINDOW_MULTIPLE = 3
INTENSITIES_LOCKED = (1, 5, 25, 125)


def _stratified_sample_trips(
    rng: np.random.Generator,
    trip_id_to_mode: dict[str, str],
    n: int,
) -> list[str]:
    """Sample n trips stratified by mode, proportion-preserving."""
    by_mode: dict[str, list[str]] = {}
    for tid, mode in trip_id_to_mode.items():
        by_mode.setdefault(mode, []).append(tid)
    total = sum(len(v) for v in by_mode.values())
    if n > total:
        raise ValueError(f"intensity {n} > total trips {total}")
    out: list[str] = []
    for mode, tids in sorted(by_mode.items()):
        k = max(0, int(round(n * len(tids) / total)))
        if k > len(tids):
            k = len(tids)
        if k > 0:
            chosen = rng.choice(tids, size=k, replace=False)
            out.extend([str(x) for x in chosen.tolist()])
    if len(out) < n:
        remaining = [t for t in trip_id_to_mode if t not in set(out)]
        extra = rng.choice(remaining, size=n - len(out), replace=False)
        out.extend([str(x) for x in extra.tolist()])
    elif len(out) > n:
        out = list(out[:n])
    assert len(out) == n
    return out


def gen_delay_edits(
    rng: np.random.Generator,
    trip_id_to_mode: dict[str, str],
    intensity: int,
) -> list[TripEdit]:
    if intensity not in INTENSITIES_LOCKED:
        raise ValueError(f"intensity {intensity} not in pre-registered ladder {INTENSITIES_LOCKED}")
    chosen = _stratified_sample_trips(rng, trip_id_to_mode, intensity)
    out: list[TripEdit] = []
    for tid in chosen:
        delta = int(rng.integers(DELAY_MIN_RANGE[0], DELAY_MIN_RANGE[1] + 1))
        out.append(TripEdit(edit_type=EditType.DELAY, trip_id=tid, delta_min=delta))
    return out


def gen_cancel_edits(
    rng: np.random.Generator,
    trip_id_to_mode: dict[str, str],
    intensity: int,
) -> list[TripEdit]:
    if intensity not in INTENSITIES_LOCKED:
        raise ValueError(f"intensity {intensity} not in pre-registered ladder {INTENSITIES_LOCKED}")
    chosen = _stratified_sample_trips(rng, trip_id_to_mode, intensity)
    return [TripEdit(edit_type=EditType.CANCELLATION, trip_id=tid) for tid in chosen]


def gen_addition_edits(
    rng: np.random.Generator,
    trip_id_to_mode: dict[str, str],
    trip_id_to_route: dict[str, str],
    route_id_to_starts: dict[str, list[int]],
    intensity: int,
    t_lo: int = 0,
    t_hi: int = 780,
) -> list[TripEdit]:
    """ADDITION via inv_3's reviewer-proof window-construction."""
    if intensity not in INTENSITIES_LOCKED:
        raise ValueError(f"intensity {intensity} not in pre-registered ladder {INTENSITIES_LOCKED}")
    available_trips = list(trip_id_to_route.keys())
    out: list[TripEdit] = []
    attempts = 0
    max_attempts = intensity * 50
    synth_n = 0

    while len(out) < intensity and attempts < max_attempts:
        attempts += 1
        template_tid = str(rng.choice(available_trips))
        route_id = trip_id_to_route[template_tid]
        starts = sorted(route_id_to_starts.get(route_id, []))
        if len(starts) < 1:
            continue
        windows = []
        if starts[0] - MIN_GAP_MIN > t_lo + MIN_GAP_MIN:
            lo = t_lo + MIN_GAP_MIN
            hi = starts[0] - MIN_GAP_MIN
            if hi - lo >= MIN_WINDOW_MULTIPLE * MIN_GAP_MIN:
                windows.append((lo, hi))
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - starts[i]
            if gap >= MIN_WINDOW_MULTIPLE * MIN_GAP_MIN:
                lo = starts[i] + MIN_GAP_MIN
                hi = starts[i + 1] - MIN_GAP_MIN
                windows.append((lo, hi))
        if starts[-1] + MIN_GAP_MIN < t_hi - MIN_GAP_MIN:
            lo = starts[-1] + MIN_GAP_MIN
            hi = t_hi - MIN_GAP_MIN
            if hi - lo >= MIN_WINDOW_MULTIPLE * MIN_GAP_MIN:
                windows.append((lo, hi))
        if not windows:
            continue
        widths = np.array([hi - lo for (lo, hi) in windows], dtype=np.float64)
        widths = widths / widths.sum()
        w_idx = int(rng.choice(len(windows), p=widths))
        lo, hi = windows[w_idx]
        t_prime = int(rng.integers(lo, hi + 1))
        synth_n += 1
        out.append(TripEdit(
            edit_type=EditType.ADDITION,
            trip_id=f"synthetic_{synth_n}",
            template_trip_id=template_tid,
            start_minute_override=t_prime,
        ))

    if len(out) < intensity:
        raise RuntimeError(f"ADDITION: could not find {intensity} legal windows after {attempts} attempts")
    return out


def build_edit_set(
    scenario: EditType,
    intensity: int,
    seed: int,
    trip_id_to_mode: dict[str, str],
    *,
    trip_id_to_route: dict[str, str] | None = None,
    route_id_to_starts: dict[str, list[int]] | None = None,
    raw_gtfs_sha: str = "",
) -> EditSet:
    rng = np.random.default_rng(seed)
    if scenario == EditType.DELAY:
        edits = gen_delay_edits(rng, trip_id_to_mode, intensity)
    elif scenario == EditType.CANCELLATION:
        edits = gen_cancel_edits(rng, trip_id_to_mode, intensity)
    elif scenario == EditType.ADDITION:
        if trip_id_to_route is None or route_id_to_starts is None:
            raise ValueError("ADDITION requires trip_id_to_route and route_id_to_starts")
        edits = gen_addition_edits(rng, trip_id_to_mode, trip_id_to_route, route_id_to_starts, intensity)
    else:
        raise ValueError(f"unknown scenario {scenario}")
    return EditSet(
        scenario=scenario,
        intensity=intensity,
        seed=seed,
        edits=tuple(edits),
        raw_gtfs_sha=raw_gtfs_sha,
    )
