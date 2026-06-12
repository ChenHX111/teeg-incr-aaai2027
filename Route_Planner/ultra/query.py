"""ULTRA-RAPTOR query — thin wrapper substituting shortcut CSR into RAPTOR.

Per Stage 2 wrapper-depth Option (a) (3/3 investigator convergence):
ULTRA shortcuts are CSR-shape-compatible with the existing
`raptor_earliest_arrival(..., closed_walks_csr=...)` call site. The
wrapper just passes the shortcut CSR to the existing function unchanged.

The RAPTOR function is FIFO-safe (vectorised min-of-valid-departures)
and uses tau_snapshot to prevent intra-round walk chaining (raptor.py
step 3 footpath relaxation comment). Both invariants hold for ULTRA
shortcut CSR substitution (shortcuts are also one-hop per round).

This module exists for two reasons (not just for clarity):
1. It pins the public ULTRA query interface, so future single-criterion
   ULTRA enhancements (e.g., the witness-search-pruning of Stage 5) can
   evolve without touching the existing `raptor.py` contract.
2. It accepts an UltraShortcutsCSR (or a raw CSR tuple) and unpacks it
   to the format raptor.py expects, isolating callers from the CSR
   representation choice.
"""
from __future__ import annotations

from ..raptor import raptor_earliest_arrival
from .cache import UltraShortcutsCSR
from .preprocess import UltraShortcuts


def ultra_raptor_earliest_arrival(b, source: int, dest: int, t_dep: int,
                                  ultra_shortcuts,
                                  max_rounds: int = 8) -> int | None:
    """ULTRA-RAPTOR earliest arrival (single-criterion projection).

    Parameters
    ----------
    b : TimetableBundle
    source, dest, t_dep : query
    ultra_shortcuts : one of:
        - UltraShortcuts (or UltraShortcutsCSR) instance from build/load
        - raw (indptr, indices, wmins) tuple
    max_rounds : same as raptor.

    Returns
    -------
    int (earliest arrival minute) or None (unreachable).

    Semantics
    ---------
    Identical to RAPTOR-with-closed-walks-CSR, except `closed_walks_csr`
    is replaced by the (pruned) ULTRA shortcut CSR. By construction of
    `build_ultra_shortcuts`, every necessary shortcut is preserved, so
    `ea_ultra == ea_raptor_closed == ea_csa` is guaranteed (this is the
    triple-equality gate per Stage 2 F.7 + Stage 3 §3 witness-audit
    arrival-time-equality spec).
    """
    if isinstance(ultra_shortcuts, (UltraShortcuts, UltraShortcutsCSR)):
        csr = ultra_shortcuts.as_tuple()
    elif isinstance(ultra_shortcuts, tuple) and len(ultra_shortcuts) == 3:
        csr = ultra_shortcuts
    else:
        raise TypeError(
            f"ultra_shortcuts must be UltraShortcuts/UltraShortcutsCSR or "
            f"3-tuple (indptr, indices, wmins); got {type(ultra_shortcuts)}"
        )

    return raptor_earliest_arrival(b, source, dest, t_dep,
                                   max_rounds=max_rounds,
                                   closed_walks_csr=csr)
