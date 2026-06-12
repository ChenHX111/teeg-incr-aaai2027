"""Contraction Hierarchies (CH) baseline for transit routing.

Design choice: CH is built ON TOP OF the TEEG (Time-Expanded Event Graph)
rather than the raw timetable. TEEG is already a single non-negatively-weighted
DAG over time-expanded events, which is exactly the substrate CH expects.
CH-on-TEEG is therefore a defensible composition: it inherits CSA-correct
semantics from TEEG and adds bidirectional upward Dijkstra on top.

Scope of this MINIMAL VIABLE implementation:
  * `build_ch_from_teeg(teeg, mode=...)` -> CHGraph
      - mode="trivial"  : levels = node_time-based topological rank, no
                          shortcuts. Produces a CORRECT CH (every TEEG edge
                          is time-respecting so every edge is naturally
                          "upward"). Used for the 100-SHORT correctness gate.
      - mode="contract" : genuine edge-difference node ordering with
                          hop-limited witness Dijkstra and shortcut insertion.
                          Used for the 5-node toy correctness test.
  * `ch_earliest_arrival(ch, teeg, source, dest, t_dep)` — bidirectional
      upward Dijkstra with TEEG source/dest attachments.

Known incomplete pieces (flagged in DESIGN.md):
  * Full edge-difference contraction on a London-subset-sized TEEG (~10⁵ nodes)
    in pure Python is too slow for the 2-hour shipping window. Cycle 9
    delivers correctness on the toy graph + parity on the 100-SHORT set in
    trivial-CH mode. A Numba-JIT or networkit-backed contraction pass is the
    P1 follow-up.
  * Time-dependent CH (TD-CH; Geisberger 2010, Batz et al.) is the
    research-grade extension; not in scope for cycle 9.
"""
from .contraction import CHGraph, build_ch_from_teeg
from .query import ch_earliest_arrival

__all__ = ["CHGraph", "build_ch_from_teeg", "ch_earliest_arrival"]
