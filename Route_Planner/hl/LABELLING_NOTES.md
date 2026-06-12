# Hub Labels on TEEG — Labelling Notes

## Design Choice: Two-Tier HL Architecture

We implement Hub Labels in two tiers:

### Tier 1: Stop-Level HL (MINIMAL VIABLE — shipped)

- **Graph**: Stop-to-stop condensed graph (2922 nodes, 37K edges)
- **Edge weights**: Minimum travel time across all connections for each stop pair
- **Method**: All-Pairs Shortest Path via scipy (C-speed Dijkstra)
- **Query**: Precomputed distance matrix provides lower bound; CSA-style
  connection scan provides exact time-dependent answer
- **Build time**: 2.4 seconds
- **Memory**: 16.3 MB (int16 distance matrix)
- **Correctness**: 100% agreement with CSA on 100 SHORT queries

### Tier 2: TEEG-Level HL (research-grade — partially implemented)

- **Graph**: Full TEEG event graph (493K nodes, 2.5M edges)
- **Method**: Pruned Landmark Labelling (Akiba et al. 2013)
- **Status**: Implemented but too slow in pure Python for full-scale
- **Bottleneck**: O(n × label_size) pruning check per hub in Python
- **Fix needed**: C extension or Numba JIT for pruning loop

### Limitations of this Minimal Implementation

| Aspect | This version | Full PTL |
|--------|-------------|----------|
| Ordering | Degree heuristic | CH contraction order |
| Label compression | None (raw lists) | Bit-parallel, interval-based |
| Time-dependency | Via TEEG structure | Native time-dependent labels |
| Scalability | ~50K nodes OK | Millions with engineering |
| Memory | O(n × avg_label) Python lists | Compressed flat arrays |

### What's Incomplete (flagged for future cycles)

- [ ] CH-based vertex ordering (would reduce label sizes 3-5×)
- [ ] Flat array label storage (eliminate Python list overhead)
- [ ] Parallel construction (currently single-threaded)
- [ ] Incremental label updates for dynamic timetable changes
- [ ] Full London scale (requires above optimizations)
- [ ] Transfer-pattern integration (PTL = HL + transfer patterns)
