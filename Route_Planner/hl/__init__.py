"""Hub Labels (HL) / Public Transit Labelling baseline.

Implements two modes:
  1. Stop-level HL: APSP on 2300-stop graph + CSA scan (exact, fast build).
  2. TEEG-level HL: Pruned landmark labelling on event graph (research-grade).

For the minimal viable baseline, use build_hl_labels_stop_level + hl_earliest_arrival.
"""
from .labelling import (build_hl_labels, build_hl_labels_stop_level,
                        build_stop_graph_from_connections, HLIndex)
from .query import hl_earliest_arrival
