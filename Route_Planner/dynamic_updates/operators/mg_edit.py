"""MG (Minute-Grid) edit operator: apply EditSet, then rebuild the minute-grid.

Day-3 implementation: FULL REBUILD from edited CSA bundle + closed-walks CSR.
Cycle 10.5 optimization: incremental bucket update.

Returns (edited_bundle, minute_grid, edit_wall_s, stats).
"""
from __future__ import annotations
import time

from ..schemas import EditSet
from .csa_edit import apply_edit_set_to_bundle as apply_to_csa


def apply_edit_set_to_mg(
    bundle,
    edit_set: EditSet,
    cw_csr,
    *,
    return_timing: bool = True,
    verbose: bool = False,
):
    """Apply EditSet, then rebuild minute-grid from the edited bundle."""
    from ...minute_grid import build_minute_grid

    t0 = time.perf_counter()
    edited_csa = apply_to_csa(bundle, edit_set)
    csa_edit_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    mg = build_minute_grid(edited_csa, cw_csr, verbose=verbose)
    mg_build_wall = time.perf_counter() - t1

    total_wall = time.perf_counter() - t0
    stats = {
        "csa_edit_wall_s": csa_edit_wall,
        "mg_rebuild_wall_s": mg_build_wall,
        "total_wall_s": total_wall,
        "mg_n_nodes": int(mg.n_nodes),
        "mg_n_edges": int(mg.n_edges),
        "n_conns_before": len(bundle.connections),
        "n_conns_after": len(edited_csa.connections),
    }
    if return_timing:
        return edited_csa, mg, total_wall, stats
    return edited_csa, mg
