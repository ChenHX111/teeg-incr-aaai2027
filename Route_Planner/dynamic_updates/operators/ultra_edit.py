"""ULTRA edit operator: apply EditSet, then rebuild ULTRA shortcuts.

Day-3 implementation: FULL REBUILD of ULTRA shortcuts (expensive — was 10-30 min
on Full London at Cycle 6). The expectation is that ULTRA's edit cost is HIGH,
which is exactly the comparison TEEG/MG want to win.

Returns (edited_bundle, ultra_shortcuts, edit_wall_s, stats).
"""
from __future__ import annotations
import time

from ..schemas import EditSet
from .csa_edit import apply_edit_set_to_bundle as apply_to_csa


def apply_edit_set_to_ultra(
    bundle,
    edit_set: EditSet,
    cw_csr,
    *,
    walk_params: dict | None = None,
    return_timing: bool = True,
    verbose: bool = False,
):
    """Apply EditSet, then rebuild ULTRA shortcuts from the edited bundle.

    NB: ULTRA shortcut preprocessing can take 10-30 min on Full London.
    This is the headline cost that TEEG/MG should beat for dynamic update
    workloads.
    """
    from ...ultra import build_ultra_shortcuts

    t0 = time.perf_counter()
    edited_csa = apply_to_csa(bundle, edit_set)
    csa_edit_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    ultra = build_ultra_shortcuts(
        edited_csa, cw_csr,
        walk_params=walk_params or {"max_walk_min": 15, "walking_speed_ms": 1.4},
        verbose=verbose,
    )
    ultra_build_wall = time.perf_counter() - t1

    total_wall = time.perf_counter() - t0
    stats = {
        "csa_edit_wall_s": csa_edit_wall,
        "ultra_rebuild_wall_s": ultra_build_wall,
        "total_wall_s": total_wall,
        "ultra_n_edges": int(ultra.n_edges),
        "n_conns_before": len(bundle.connections),
        "n_conns_after": len(edited_csa.connections),
    }
    if return_timing:
        return edited_csa, ultra, total_wall, stats
    return edited_csa, ultra
