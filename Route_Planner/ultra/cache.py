"""ULTRA cache .npz I/O — raw int32 CSR triple via np.savez_compressed.

Per Stage 2 F.1 (cache format = .npz) + F.2 (in-memory type = raw int32
triple) + B2 action ledger.

NOT scipy.sparse.save_npz (Stage 2 DO-NOT #6). NOT parquet for the canonical
shortcut artifact (parquet is used only for Plan B chunked intermediates).
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sidecar import (
    UltraSidecar, make_sidecar, write_sidecar, read_sidecar,
    validate_sidecar_against_bundle, CacheStaleness,
)


@dataclass
class UltraShortcutsCSR:
    """Container for the ULTRA shortcut CSR triple + its sidecar metadata.

    The CSR triple (indptr, indices, wmins) is directly substitutable for
    the `closed_walks_csr=` argument of
    `Route_Planner.raptor.raptor.raptor_earliest_arrival` per the wrapper-
    depth Option (a) decision (Stage 2 §B).
    """
    indptr: np.ndarray   # int32 (n_stops + 1,)
    indices: np.ndarray  # int32 (n_edges,)
    wmins: np.ndarray    # int32 (n_edges,)
    sidecar: UltraSidecar | None = None

    def as_tuple(self) -> tuple:
        """The CSR triple in the (indptr, indices, wmins) shape expected by
        raptor_earliest_arrival(..., closed_walks_csr=)."""
        return (self.indptr, self.indices, self.wmins)

    @property
    def n_edges(self) -> int:
        return int(len(self.indices))


def save_ultra_shortcuts_npz(path: str | Path, csr: UltraShortcutsCSR) -> None:
    """Save the CSR triple as .npz + the sidecar as .meta.json.

    Per Stage 2 F.1: np.savez_compressed (canonical format). Sidecar file
    is `<path-without-.npz>.meta.json`.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"path must end in .npz; got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, indptr=csr.indptr, indices=csr.indices, wmins=csr.wmins)
    if csr.sidecar is not None:
        sidecar_path = path.with_suffix(".meta.json")
        write_sidecar(sidecar_path, csr.sidecar)


def load_ultra_shortcuts_npz(path: str | Path, b=None, walk_params: dict | None = None,
                             validate: bool = True) -> UltraShortcutsCSR:
    """Load the CSR triple + sidecar, optionally validating against a bundle.

    Parameters
    ----------
    path : path to the .npz file (sidecar inferred as `.meta.json` sibling).
    b : TimetableBundle, optional. Required if validate=True.
    walk_params : dict, optional. Required if validate=True.
    validate : bool. If True, raises CacheStaleness if sidecar hashes don't
               match the live bundle.

    Returns
    -------
    UltraShortcutsCSR with indptr/indices/wmins/sidecar populated.

    Raises
    ------
    FileNotFoundError, CacheStaleness, AssertionError (dtype/shape).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ULTRA cache not found: {path}")

    with np.load(path) as data:
        indptr = data["indptr"]
        indices = data["indices"]
        wmins = data["wmins"]
    # Asserts per Stage 2 B2 ("load asserts dtype/shape")
    assert indptr.dtype == np.int32, f"indptr dtype {indptr.dtype} (expected int32)"
    assert indices.dtype == np.int32, f"indices dtype {indices.dtype} (expected int32)"
    assert wmins.dtype == np.int32, f"wmins dtype {wmins.dtype} (expected int32)"
    assert len(indices) == len(wmins), \
        f"indices/wmins length mismatch: {len(indices)} vs {len(wmins)}"

    sidecar_path = path.with_suffix(".meta.json")
    sidecar: UltraSidecar | None = None
    if sidecar_path.exists():
        sidecar = read_sidecar(sidecar_path)
        assert sidecar.n_stops == len(indptr) - 1, \
            f"sidecar.n_stops {sidecar.n_stops} != indptr length - 1 ({len(indptr) - 1})"

    if validate:
        if b is None or walk_params is None:
            raise ValueError("validate=True requires both `b` (bundle) and `walk_params`")
        if sidecar is None:
            raise CacheStaleness(f"sidecar missing alongside {path}; cannot validate")
        validate_sidecar_against_bundle(sidecar, b, walk_params)

    return UltraShortcutsCSR(indptr=indptr, indices=indices, wmins=wmins, sidecar=sidecar)
