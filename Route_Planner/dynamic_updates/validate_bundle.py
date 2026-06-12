"""validate_bundle: post-Phase-I-bug-class assertion that the loaded bundle is sane."""
from __future__ import annotations
from typing import Any


def validate_bundle(b: Any, *, t_lo: int = 0, t_hi: int = 780) -> dict:
    """Assert bundle is sane. Returns provenance dict or raises ValueError."""
    if b is None:
        raise ValueError("validate_bundle: bundle is None")
    if not hasattr(b, "stop_times_min") or b.stop_times_min is None:
        raise ValueError("validate_bundle: no stop_times_min attribute")
    stm = b.stop_times_min
    if stm.ndim != 2 or stm.shape[1] != 2:
        raise ValueError(f"validate_bundle: stop_times_min wrong shape {stm.shape}")
    if str(stm.dtype) != "int16":
        raise ValueError(f"validate_bundle: stop_times_min dtype {stm.dtype} != int16 (overflow risk)")

    arr_min = stm[:, 0]
    dep_min = stm[:, 1]
    valid_mask = (arr_min >= -1) & (arr_min <= t_hi + 100) & (dep_min >= -1) & (dep_min <= t_hi + 100)
    valid_frac = float(valid_mask.mean())
    if valid_frac < 0.9:
        raise ValueError(
            f"validate_bundle: only {valid_frac*100:.1f}% of stop_times_min entries in [-1, {t_hi+100}]; "
            f"arr [{arr_min.min()}, {arr_min.max()}], dep [{dep_min.min()}, {dep_min.max()}] "
            f"— possible service-day mismatch (Phase-I-bug class)"
        )

    if b.n_stops <= 0:
        raise ValueError(f"validate_bundle: n_stops = {b.n_stops}")
    n_conns = len(b.connections) if hasattr(b, "connections") else 0
    if n_conns <= 0:
        raise ValueError(f"validate_bundle: n_connections = {n_conns}")

    return {
        "n_stops": int(b.n_stops),
        "n_connections": int(n_conns),
        "valid_frac": valid_frac,
        "arr_min_range": [int(arr_min.min()), int(arr_min.max())],
        "dep_min_range": [int(dep_min.min()), int(dep_min.max())],
    }
