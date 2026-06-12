"""Schemas for Cycle 10 dynamic-update experiments.

Locked at pre-registration (CYCLE_10_PREREGISTRATION.md §2). Any change
requires a new audit cycle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EditType(str, Enum):
    DELAY = "DELAY"
    CANCELLATION = "CANCELLATION"
    ADDITION = "ADDITION"


@dataclass(frozen=True)
class TripEdit:
    """A single per-trip edit."""
    edit_type: EditType
    trip_id: str
    delta_min: int = 0
    template_trip_id: str | None = None
    start_minute_override: int | None = None


@dataclass(frozen=True)
class EditSet:
    """An atomic edit set applied to a bundle."""
    scenario: EditType
    intensity: int
    seed: int
    edits: tuple[TripEdit, ...]
    raw_gtfs_sha: str = ""

    def short_id(self) -> str:
        return f"{self.scenario.value}-N{self.intensity}-S{self.seed}"


@dataclass(frozen=True)
class ArmConfig:
    scenario: EditType
    intensity: int
    seed: int
    method: str
    tier: int


@dataclass
class TimedState:
    state: Any
    elapsed_s: float


@dataclass(frozen=True)
class ConnectionRef:
    trip_id: str
    stop_seq_dep: int
    stop_seq_arr: int


@dataclass(frozen=True)
class QueryRecord:
    q_id: int
    src: int
    dst: int
    t_dep: int


@dataclass
class ArmManifest:
    role: str
    method: str
    pid: int
    process_start_time_iso: str
    git_sha: str
    bundle_sha: str
    edit_set_sha: str
    queries_sha: str
    preregistration_sha: str
    oracle_version: str
    elapsed_s: float
    warmup_runs: int
    timed_runs: int
    results_sha: str
    cache_hit: bool = False
