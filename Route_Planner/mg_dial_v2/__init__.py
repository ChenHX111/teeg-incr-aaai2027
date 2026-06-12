"""MG-Dial v2 — K-bounded lane-augmented Dial with dynamic tombstone overlay.

See ``Experiments/results/cycle_9_paper/EXT2_MGDIAL_DESIGN/DESIGN.md``
for the design and falsification gates.
"""
from .build import (
    MinuteGridV2,
    build_minute_grid_v2,
    verify_f9_shadow_uniqueness,
    ET_WAIT_R, ET_WAIT_W, ET_RIDE, ET_CONTINUE, ET_WALK, ET_EXIT,
)
from .query import MGDialV2
from .updates import (
    cancel_trip,
    restore_trip,
    add_trip,
    cancel_added_trip,
    delay_trip,
)


def query(runner: "MGDialV2", source: int, dest: int, t_dep: int, *,
          closed_walks_csr=None, K=None):
    """Functional facade for :meth:`MGDialV2.query`."""
    return runner.query(source, dest, t_dep,
                        closed_walks_csr=closed_walks_csr, K=K)


__all__ = [
    "MinuteGridV2",
    "build_minute_grid_v2",
    "verify_f9_shadow_uniqueness",
    "MGDialV2",
    "query",
    "cancel_trip",
    "restore_trip",
    "add_trip",
    "cancel_added_trip",
    "delay_trip",
    "ET_WAIT_R", "ET_WAIT_W", "ET_RIDE", "ET_CONTINUE", "ET_WALK", "ET_EXIT",
]
