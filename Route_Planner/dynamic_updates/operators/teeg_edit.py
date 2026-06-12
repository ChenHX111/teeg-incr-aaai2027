"""TEEG edit operator: apply EditSet, then rebuild the TEEG graph.

Day-3 implementation: FULL REBUILD from edited CSA bundle.
Cycle 10.5 optimization: incremental edge mutation.

Returns (edited_bundle, teeg_graph, edit_wall_s, stats).
"""
from __future__ import annotations
import time

from ..schemas import EditSet
from .csa_edit import apply_edit_set_to_bundle as apply_to_csa


def apply_edit_set_to_teeg(
    bundle,
    edit_set: EditSet,
    closed_transfers_from=None,
    *,
    return_timing: bool = True,
    verbose: bool = False,
):
    """Apply EditSet, then rebuild TEEG from the edited bundle.

    Total wall = CSA edit + TEEG rebuild. This is the honest "edit then prepare"
    wall time, comparable to baseline rebuild costs.
    """
    # Late import to avoid top-level cycle if TEEG package isn't fully loaded
    from ...teeg.builder import build_teeg

    t0 = time.perf_counter()
    edited_csa = apply_to_csa(bundle, edit_set)
    csa_edit_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    teeg_graph = build_teeg(edited_csa, verbose=verbose,
                             closed_transfers_from=closed_transfers_from)
    teeg_build_wall = time.perf_counter() - t1

    total_wall = time.perf_counter() - t0
    stats = {
        "csa_edit_wall_s": csa_edit_wall,
        "teeg_rebuild_wall_s": teeg_build_wall,
        "total_wall_s": total_wall,
        "teeg_n_nodes": int(teeg_graph.n_nodes),
        "teeg_n_edges": int(teeg_graph.n_edges),
        "n_conns_before": len(bundle.connections),
        "n_conns_after": len(edited_csa.connections),
    }
    if return_timing:
        return edited_csa, teeg_graph, total_wall, stats
    return edited_csa, teeg_graph
