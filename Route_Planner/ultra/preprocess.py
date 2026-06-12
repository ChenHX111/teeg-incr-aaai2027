"""ULTRA preprocessing — build the stop-to-stop shortcut CSR.

This is the algorithmic core. Per `Experiments/results/cycle_6/stage4_scope.md`
this is a **single-criterion projection** of the published ULTRA-RAPTOR
(Baum et al., ESA 2019; Transportation Science 2023).

**Stage 4 / Cycle 6 implementation note (CRITICAL)**:

Real ULTRA pruning separates **source/target walks** (handled at query
time via raw Dijkstra in the transfer graph or per-query buckets) from
**intermediate-transfer shortcuts** (the pruned set computed in
preprocessing). Stage 2 wrapper-depth Option (a) substitutes the
shortcut CSR for `closed_walks_csr` at the existing
`raptor_earliest_arrival(...)` call site, but RAPTOR uses
`closed_walks_csr` for BOTH source initialisation AND intermediate
transfers (raptor.py lines 53-72 + 145-167). Any walk dropped from the
CSR is therefore unavailable as both an intermediate transfer AND as a
source/target walk — which breaks correctness for queries whose source
or destination is the dropped walk's endpoint.

For arbitrary-OD query workloads (our setup), the only walks safe to
drop without violating correctness are walks neither endpoint of which
can appear as a query source OR destination. In our subset, every stop
IS a candidate query source/dest, so NO walk is safe to drop under the
current Option-(a) wrapper-depth.

**Stage 4 ships a correctness-preserving identity transform**:
`ultra_shortcuts_csr := closed_walks_csr`. ULTRA-RAPTOR is then
`raptor_earliest_arrival(..., closed_walks_csr=ultra_shortcuts_csr)`,
which is bit-for-bit identical to RAPTOR-with-closed-walks. This:
- ✓ passes the triple-equality correctness gate (trivially).
- ✓ ships ULTRA as a named, cached, witness-audited method in the
  benchmark harness (sidecar metadata, cache I/O, FIFO preflight, etc.
  all functional).
- ✗ does NOT yield ULTRA's published speedup over RAPTOR.

**Stage 5+ TODO**: implement proper witness-search-based pruning that
either:
  (a) extends RAPTOR with separate source/target walk channels (e.g.,
      add `source_walks_csr` / `target_walks_csr` kwargs distinct from
      the intermediate-transfer `shortcuts_csr`), OR
  (b) implements the published ULTRA query-time source/target Dijkstra
      separately from the cached intermediate shortcut graph.

Deferred to Stage 5 (design experiments) or Stage 4 follow-up.

Per Stage 2 F.4 deterministic-build freeze, the canonicalisation
pipeline (lexsort + dedup) still runs over the input closed walks so
that the cached CSR is byte-identical on repeat builds.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np

from .cache import UltraShortcutsCSR
from .shortcuts import coo_to_csr_deterministic, csr_to_coo
from .sidecar import make_sidecar


@dataclass
class UltraShortcuts:
    """Container for the ULTRA shortcut CSR + sidecar metadata."""
    csr: UltraShortcutsCSR

    @property
    def indptr(self): return self.csr.indptr

    @property
    def indices(self): return self.csr.indices

    @property
    def wmins(self): return self.csr.wmins

    @property
    def sidecar(self): return self.csr.sidecar

    @property
    def n_edges(self): return self.csr.n_edges

    def as_tuple(self): return self.csr.as_tuple()


def build_ultra_shortcuts(b, closed_walks_csr: tuple,
                          walk_params: dict | None = None,
                          verbose: bool = False) -> UltraShortcuts:
    """Build the ULTRA stop-to-stop shortcut CSR.

    **Stage 4 / Cycle 6 implementation**: correctness-preserving identity
    transform of `closed_walks_csr` (see module docstring for the
    rationale). Output CSR is byte-identical to `closed_walks_csr` after
    canonicalisation (lexsort + dedup).

    Parameters
    ----------
    b : TimetableBundle.
    closed_walks_csr : (indptr, indices, wmins) tuple from
        `Route_Planner.csa.csa.build_closed_walk_csr`.
    walk_params : dict, optional. Default: {"max_walk_min": 15,
        "walking_speed_ms": 1.4}. **Note**: do NOT mutate the returned
        default; a fresh dict is constructed each call (no shared mutable
        default arg).
    verbose : print progress.

    Returns
    -------
    UltraShortcuts with byte-identical CSR (after canonicalisation) +
    sidecar metadata.

    Correctness guarantee: ULTRA-RAPTOR(b, ..., this_csr) ==
    raptor_earliest_arrival(b, ..., closed_walks_csr=closed_walks_csr)
    for ALL queries.
    """
    if walk_params is None:
        walk_params = {"max_walk_min": 15, "walking_speed_ms": 1.4}

    t0 = time.time()
    cw_indptr, cw_indices, cw_wmins = closed_walks_csr
    n_stops = int(b.n_stops)
    assert len(cw_indptr) == n_stops + 1, \
        f"closed_walks_csr indptr length {len(cw_indptr)} != n_stops+1 ({n_stops + 1})"

    if verbose:
        print(f"[ultra.build] starting; {n_stops} stops, "
              f"{len(cw_indices):,} closed walks input")

    # Decompress closed walks to COO, then canonicalise via the
    # deterministic CSR build pipeline. This produces a byte-identical
    # output on repeat invocations (Stage 2 F.4 freeze).
    src, dst, wmin = csr_to_coo(cw_indptr, cw_indices, cw_wmins)
    indptr, indices, wmins = coo_to_csr_deterministic(src, dst, wmin, n_stops)
    t1 = time.time()

    sidecar = make_sidecar(
        b, walk_params,
        edge_count=int(len(indices)),
        build_started=t0,
        build_finished=t1,
    )
    csr = UltraShortcutsCSR(indptr=indptr, indices=indices, wmins=wmins, sidecar=sidecar)

    if verbose:
        same = (len(indices) == len(cw_indices))
        print(f"[ultra.build] done in {t1-t0:.1f}s; "
              f"{len(indices):,} shortcuts "
              f"({'identity' if same else 'pruned'} vs {len(cw_indices):,} closed walks)")

    return UltraShortcuts(csr=csr)

