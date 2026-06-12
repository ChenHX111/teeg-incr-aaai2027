"""Cycle 10 dynamic-updates package.

Implements the dynamic-timetable-update experiments per the SHIPPED
Stage 1 v2 design + Stage 2 v2 implementation plan.

Hash-locked pre-registration: see
`Experiments/results/cycle_10_dynamic_updates/stage_2_plan_implementation/CYCLE_10_PREREGISTRATION.md`
(SHA-256 in `.sha256` sibling file).

DO NOT import from Route_Planner.teeg / minute_grid / raptor / ultra at
package-import time — keep imports local to operators that need them so
this package can be inspected for independence violations.
"""
__version__ = "0.1.0-day1"
