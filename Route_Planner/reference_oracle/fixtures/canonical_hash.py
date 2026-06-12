"""Canonical hashing for Reference-CSA* fixtures.

Per Day-3 audit (3/4 GREEN-YELLOW with convergence on raw-bytes-is-fragile):
- Hash a canonical JSON representation, not numpy raw bytes.
- dtype-independent, endian-independent, numpy-version-independent.
- schema_version locked at 1.
"""
from __future__ import annotations
import hashlib
import json
from typing import Iterable

SCHEMA_VERSION = 1


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_bundle_hash(
    n_stops: int,
    connections: Iterable[tuple],  # (from_stop, to_stop, dep_min, arr_min, trip_id_int)
    walks_raw: Iterable[tuple],    # (from_stop, to_stop, walk_min)
) -> str:
    """SHA-256 of a canonical JSON representation of bundle inputs.

    Sorted lists of integer-tuples; no numpy dtype dependence.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "n_stops": int(n_stops),
        "connections": sorted([list(map(int, c)) for c in connections]),
        "walks_raw": sorted([list(map(int, w)) for w in walks_raw]),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return _sha256_str(canonical)


def canonical_closed_walks_hash(closed_walks: dict[int, dict[int, int]]) -> str:
    """SHA-256 of a canonical representation of {src: {dst: walk_min}}.

    Flatten to a sorted list of [src, dst, walk_min] integer triples.
    """
    triples = []
    for src, dsts in closed_walks.items():
        for dst, wmin in dsts.items():
            triples.append([int(src), int(dst), int(wmin)])
    triples.sort()
    record = {"schema_version": SCHEMA_VERSION, "closed_walks": triples}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return _sha256_str(canonical)


def canonical_query_hash(query: dict) -> str:
    """SHA-256 of a canonical query record {src, dst, t_dep, expected_ea}.

    expected_ea may be None (unreachable) or an integer.
    """
    rec = {
        "schema_version": SCHEMA_VERSION,
        "src": int(query["src"]),
        "dst": int(query["dst"]),
        "t_dep": int(query["t_dep"]),
        "expected_ea": None if query["expected_ea"] is None else int(query["expected_ea"]),
    }
    return _sha256_str(json.dumps(rec, sort_keys=True, separators=(",", ":")))


def canonical_fixture_hash(fixture: dict) -> str:
    """SHA-256 of the full fixture record (bundle + closed_walks + queries + meta).

    Includes schema_version so that any future hash-scheme change is detectable.
    Does NOT include the fixture id (so renaming doesn't change the hash).
    """
    rec = {
        "schema_version": SCHEMA_VERSION,
        "description": fixture["description"],
        "bundle_sha256": fixture["bundle_sha256"],
        "walks_closed_sha256": fixture["walks_closed_sha256"],
        "query_hashes": sorted([canonical_query_hash(q) for q in fixture["queries"]]),
        "n_stops": int(fixture["n_stops"]),
        "n_trips": int(fixture["n_trips"]),
        "n_walks_raw": int(fixture["n_walks_raw"]),
    }
    return _sha256_str(json.dumps(rec, sort_keys=True, separators=(",", ":")))
