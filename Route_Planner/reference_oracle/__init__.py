"""Reference-CSA* — independent CSA oracle for the AAAI 2027 submission.

This package is INDEPENDENT from `Route_Planner.csa`, `Route_Planner.raptor`,
`Route_Planner.teeg`, `Route_Planner.minute_grid`, `Route_Planner.ultra`.

Independence is structural (Stage 1 v2 design §2 layers L1-L5):
- L1: package isolation (no imports from forbidden prefixes; enforced by AST audit
      test in `tests/test_import_independence.py`).
- L2: independent GTFS parser (`parser.py`) that reads raw `.pkl` files directly,
      not via `Route_Planner.common.data_layer`.
- L3: independent F_15 transitive closure (`footpath_closure.py`).
- L4: offline pre-flight walk-table uniformity check (NOT runtime).
- L5: every run emits a JSON manifest with full provenance + SHA-256s.

Pre-registration: see
`Experiments/results/cycle_9_paper/stage_2_plan_p0_a_implementation/CYCLE_9_PREREGISTRATION_PLACEHOLDER.md`
(the Cycle-9 pre-reg artifact will be created when the Day-1 deliverables are committed).

DO NOT import from forbidden prefixes at package-import time.
"""
__version__ = "0.1.0-cycle9-day1"
