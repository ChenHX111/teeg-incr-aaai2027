"""Toy fixtures for Cycle 10 dynamic-update tests."""
from __future__ import annotations
import json
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent


def _fixture_path(fid: str) -> Path:
    return FIXTURES_DIR / f"{fid}.json"


F01 = {
    "fixture_id": "F01",
    "description": "Route R with 2 trips T1 (09:00-09:05 A->B) and T2 (09:10-09:15 A->B). Delay T1 by 3 min.",
    "stops": [
        {"stop_id": "A", "stop_int": 0, "stop_lat": 51.5, "stop_lon": -0.1},
        {"stop_id": "B", "stop_int": 1, "stop_lat": 51.51, "stop_lon": -0.1},
    ],
    "routes": [{"route_id": "R1", "route_int": 0, "mode": "bus"}],
    "trips": [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 0},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 5, "departure_minute": 5}
            ]
        },
        {
            "trip_id": "T2",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 10},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 15, "departure_minute": 15}
            ]
        }
    ],
    "footpaths": [],
    "service_window": {"t_lo": 0, "t_hi": 60},
    "edit": {
        "scenario": "DELAY",
        "intensity": 1,
        "seed": 1,
        "edits": [{"edit_type": "DELAY", "trip_id": "T1", "delta_min": 3}]
    },
    "expected_post_edit": {
        "queries": [
            {"q_id": 1, "src": 0, "dst": 1, "t_dep": 0, "expected_arrival": 8, "note": "T1 was 5min; now T1 starts at 3min and arrives 8min."},
            {"q_id": 2, "src": 0, "dst": 1, "t_dep": 6, "expected_arrival": 15, "note": "T1 dep=3<6 missed; take T2 dep=10 arr=15"}
        ]
    }
}

F02 = {
    "fixture_id": "F02",
    "description": "Walk-only A<->B (8 min). Add a synthetic trip A->B at 09:00, ride 5 min.",
    "stops": [
        {"stop_id": "A", "stop_int": 0, "stop_lat": 51.5, "stop_lon": -0.1},
        {"stop_id": "B", "stop_int": 1, "stop_lat": 51.508, "stop_lon": -0.1},
    ],
    "routes": [{"route_id": "R1", "route_int": 0, "mode": "bus"}],
    "trips": [
        {
            "trip_id": "T_ANCHOR",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 30},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 35, "departure_minute": 35}
            ]
        }
    ],
    "footpaths": [
        {"from_stop_int": 0, "to_stop_int": 1, "walk_min": 8},
        {"from_stop_int": 1, "to_stop_int": 0, "walk_min": 8}
    ],
    "service_window": {"t_lo": 0, "t_hi": 60},
    "edit": {
        "scenario": "ADDITION",
        "intensity": 1,
        "seed": 1,
        "edits": [{"edit_type": "ADDITION", "trip_id": "synthetic_1", "template_trip_id": "T_ANCHOR", "start_minute_override": 0}]
    },
    "expected_post_edit": {
        "queries": [
            {"q_id": 1, "src": 0, "dst": 1, "t_dep": 0, "expected_arrival": 5, "note": "synthetic trip dep=0 arr=5 beats 8-min walk"}
        ]
    }
}

F03 = {
    "fixture_id": "F03",
    "description": "Single trip T1 09:00-09:05 A->B. F_15 has A->B walk 7 min. Cancel T1.",
    "stops": [
        {"stop_id": "A", "stop_int": 0, "stop_lat": 51.5, "stop_lon": -0.1},
        {"stop_id": "B", "stop_int": 1, "stop_lat": 51.51, "stop_lon": -0.1},
    ],
    "routes": [{"route_id": "R1", "route_int": 0, "mode": "bus"}],
    "trips": [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 0},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 5, "departure_minute": 5}
            ]
        }
    ],
    "footpaths": [
        {"from_stop_int": 0, "to_stop_int": 1, "walk_min": 7},
        {"from_stop_int": 1, "to_stop_int": 0, "walk_min": 7}
    ],
    "service_window": {"t_lo": 0, "t_hi": 60},
    "edit": {
        "scenario": "CANCELLATION",
        "intensity": 1,
        "seed": 1,
        "edits": [{"edit_type": "CANCELLATION", "trip_id": "T1"}]
    },
    "expected_post_edit": {
        "queries": [
            {"q_id": 1, "src": 0, "dst": 1, "t_dep": 0, "expected_arrival": 7, "note": "T1 cancelled; walk fallback 7 min"}
        ]
    }
}

F04 = {
    "fixture_id": "F04",
    "description": "Route R with trips T1 at 09:00 and T2 at 09:30. Add synthetic trip at 09:15.",
    "stops": [
        {"stop_id": "A", "stop_int": 0, "stop_lat": 51.5, "stop_lon": -0.1},
        {"stop_id": "B", "stop_int": 1, "stop_lat": 51.51, "stop_lon": -0.1},
    ],
    "routes": [{"route_id": "R1", "route_int": 0, "mode": "bus"}],
    "trips": [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 0},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 5, "departure_minute": 5}
            ]
        },
        {
            "trip_id": "T2",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 30},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 35, "departure_minute": 35}
            ]
        }
    ],
    "footpaths": [],
    "service_window": {"t_lo": 0, "t_hi": 60},
    "edit": {
        "scenario": "ADDITION",
        "intensity": 1,
        "seed": 1,
        "edits": [{"edit_type": "ADDITION", "trip_id": "synthetic_1", "template_trip_id": "T1", "start_minute_override": 15}]
    },
    "expected_post_edit": {
        "queries": [
            {"q_id": 1, "src": 0, "dst": 1, "t_dep": 10, "expected_arrival": 20, "note": "synthetic dep=15 arr=20 beats T2 arr=35"}
        ]
    }
}

F05 = {
    "fixture_id": "F05",
    "description": "Trip T1 23:55-00:05 next day. Delay T1 by 10 min.",
    "stops": [
        {"stop_id": "A", "stop_int": 0, "stop_lat": 51.5, "stop_lon": -0.1},
        {"stop_id": "B", "stop_int": 1, "stop_lat": 51.51, "stop_lon": -0.1},
    ],
    "routes": [{"route_id": "R1", "route_int": 0, "mode": "bus"}],
    "trips": [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "stop_times": [
                {"stop_id": "A", "stop_int": 0, "stop_sequence": 0, "arrival_minute": -1, "departure_minute": 1435},
                {"stop_id": "B", "stop_int": 1, "stop_sequence": 1, "arrival_minute": 1445, "departure_minute": 1445}
            ]
        }
    ],
    "footpaths": [],
    "service_window": {"t_lo": 1400, "t_hi": 1500},
    "edit": {
        "scenario": "DELAY",
        "intensity": 1,
        "seed": 1,
        "edits": [{"edit_type": "DELAY", "trip_id": "T1", "delta_min": 10}]
    },
    "expected_post_edit": {
        "queries": [
            {"q_id": 1, "src": 0, "dst": 1, "t_dep": 1400, "expected_arrival": 1455, "note": "T1 dep+10=1445, arr+10=1455"}
        ]
    }
}


FIXTURES = {f["fixture_id"]: f for f in [F01, F02, F03, F04, F05]}


def write_all_fixtures() -> None:
    """Write the 5 fixture JSON files; deterministic (sort keys)."""
    for fid, payload in FIXTURES.items():
        path = _fixture_path(fid)
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def load_fixture(fid: str) -> dict:
    return json.loads(_fixture_path(fid).read_text(encoding="utf-8"))


if __name__ == "__main__":
    write_all_fixtures()
    print(f"Wrote {len(FIXTURES)} fixtures to {FIXTURES_DIR}")
    for fid in FIXTURES:
        p = _fixture_path(fid)
        print(f"  {fid}: {p.stat().st_size} B")
