"""SHA-256 hashing utilities for content-addressed artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path | str) -> str:
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Canonical-JSON SHA-256: sort keys, no whitespace, ascii-only."""
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256_bytes(canon.encode("utf-8"))


def sha256_dir(d: Path | str, *, follow_extensions: tuple[str, ...] = (".pkl", ".csv", ".npz", ".json")) -> str:
    d = Path(d)
    if not d.exists():
        raise FileNotFoundError(d)
    if d.is_file():
        return sha256_file(d)
    entries = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix in follow_extensions:
            rel = p.relative_to(d).as_posix()
            entries.append(f"{rel}={sha256_file(p)}")
    return sha256_bytes("\n".join(entries).encode("utf-8"))


def combined_cache_key(*parts: str) -> str:
    """Combine multiple SHA hashes into one cache key (the §5 cache key from STAGE_2_v2)."""
    return sha256_bytes("||".join(parts).encode("utf-8"))
