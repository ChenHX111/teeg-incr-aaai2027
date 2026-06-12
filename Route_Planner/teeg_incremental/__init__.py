"""TEEG-ALT-INCR — Incremental event-graph patching for GTFS-RT (EXT3, AAAI 2027).

Public API:
    TEEGIncremental(g, labels=None)
        .apply_edits(edits, **kwargs) -> BatchStats
        .query(source, dest, t_dep, *, dijkstra=False) -> int | None
        .compact() -> None    (in-place; rebuilds g + state from alive subset)

Edits are tuples
    ('CANCEL', trip_id)
    ('DELAY',  trip_id, delta_min)
    ('ADD',    template_trip_id, t0_delta_min)

See `Experiments/results/cycle_9_paper/EXT3_GTFS_RT_DESIGN/DESIGN.md` for the
full theoretical specification and the §7.5b 50K-edit BLOCKING GATE that this
module is required to pass.
"""
from __future__ import annotations

from .state import IncrementalState, build_state
from .updates import (
    BatchStats, apply_edits, cancel_trip, add_trip, delay_trip, compact,
)
from .query import query_compiled, query_dijkstra_compiled


class TEEGIncremental:
    """High-level wrapper that owns a (g, state) pair and manages compaction."""

    def __init__(self, g, labels=None, *,
                 overlay_capacity_nodes: int | None = None,
                 overlay_capacity_edges: int | None = None,
                 auto_compact: bool = True):
        self.g = g
        self.labels = labels
        self.state = build_state(g,
                                 overlay_capacity_nodes=overlay_capacity_nodes,
                                 overlay_capacity_edges=overlay_capacity_edges)
        self.auto_compact = auto_compact

    def apply_edits(self, edits, **kwargs) -> BatchStats:
        cb = self._compaction_callback if self.auto_compact else None
        kwargs.setdefault('compaction_callback', cb)
        kwargs.setdefault('allow_compaction', self.auto_compact)
        return apply_edits(self.state, self.g, edits, **kwargs)

    def _compaction_callback(self, state, g):
        new_state, new_g = compact(state, g)
        new_state.compactions_run = state.compactions_run + 1
        # IMPORTANT: rebuild ALT landmark labels too (admissibility holds since
        # alive subset's LB graph is a subgraph of the original; existing labels
        # are still admissible — we keep them).  But d_to/d_from are sized to
        # the OLD n_stops which is unchanged, so we may keep `self.labels` as-is.
        self.g = new_g
        self.state = new_state

    def compact(self):
        new_state, new_g = compact(self.state, self.g)
        self.g = new_g
        self.state = new_state
        self.state.compactions_run += 1

    def query(self, source: int, dest: int, t_dep: int, *, dijkstra: bool = False):
        if dijkstra:
            return query_dijkstra_compiled(self.state, self.g, source, dest, t_dep)
        if self.labels is None:
            raise ValueError("ALT query requires landmark labels; pass `labels=` to constructor")
        return query_compiled(self.state, self.g, source, dest, t_dep, self.labels)


__all__ = [
    'TEEGIncremental',
    'IncrementalState',
    'BatchStats',
    'build_state',
    'apply_edits',
    'cancel_trip',
    'add_trip',
    'delay_trip',
    'compact',
    'query_compiled',
    'query_dijkstra_compiled',
]
