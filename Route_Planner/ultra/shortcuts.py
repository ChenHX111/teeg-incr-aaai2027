"""ULTRA shortcut CSR utilities — COO → deduped → sorted → CSR pipeline.

Per Stage 2 F.4 freeze (deterministic tie-break) and B4 action ledger:
- Build COO triple `(src, dst, wmin)`.
- `np.lexsort((wmins, dsts, srcs))` to canonicalise.
- Dedup on `(src, dst)` keeping min `wmin`.
- Compact to CSR `(indptr, indices, wmins)` (raw int32 triples; matches the
  contract of `Route_Planner.raptor.raptor.raptor_earliest_arrival`'s
  `closed_walks_csr=` parameter exactly).

The CSR triple is NOT a scipy.sparse object; it is the same plain-numpy
triple that `Route_Planner.csa.csa.build_closed_walk_csr` returns. ULTRA
shortcuts are semantically DISTINCT from `closed_walks_csr` (one is the
necessary-shortcut set; the other is the transitive closure) but share the
on-disk and in-memory CSR shape so the RAPTOR call site is reusable
unchanged (Stage 2 wrapper-depth Option (a)).
"""
from __future__ import annotations
import numpy as np


def coo_to_csr_deterministic(src: np.ndarray, dst: np.ndarray, wmin: np.ndarray,
                              n_stops: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical COO → CSR with deterministic dedup (min wmin per (src, dst)).

    Parameters
    ----------
    src, dst : int32 arrays, same length, source / destination stop indices.
    wmin : int32 array, same length, walking minutes for each (src, dst).
    n_stops : total number of stops (CSR row dimension).

    Returns
    -------
    indptr : int32 (n_stops + 1,)
    indices : int32 (n_edges,)
    wmins : int32 (n_edges,)

    Determinism guarantee: repeat invocation on identical inputs returns
    bytewise-identical arrays. Achieved via np.lexsort((wmin, dst, src)) +
    dedup keeping the first (= min-wmin) per (src, dst).
    """
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    wmin = np.asarray(wmin, dtype=np.int32)
    if not (len(src) == len(dst) == len(wmin)):
        raise ValueError(f"COO triple length mismatch: {len(src)}, {len(dst)}, {len(wmin)}")

    if len(src) == 0:
        indptr = np.zeros(n_stops + 1, dtype=np.int32)
        return indptr, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    # Canonical sort: primary key src, secondary dst, tertiary wmin.
    # np.lexsort sorts by the LAST key first, so reverse:
    order = np.lexsort((wmin, dst, src))
    src_s = src[order]
    dst_s = dst[order]
    wmin_s = wmin[order]

    # Dedup: for each unique (src, dst), keep the first (which is min-wmin
    # by lexsort key ordering above).
    key_pairs = (src_s.astype(np.int64) << 32) | (dst_s.astype(np.int64) & 0xFFFFFFFF)
    keep = np.empty(len(key_pairs), dtype=bool)
    keep[0] = True
    keep[1:] = key_pairs[1:] != key_pairs[:-1]
    src_u = src_s[keep]
    dst_u = dst_s[keep]
    wmin_u = wmin_s[keep]

    # Build CSR.
    # Use np.bincount for indptr (faster than np.add.at on large inputs).
    counts = np.bincount(src_u, minlength=n_stops).astype(np.int32)
    indptr = np.zeros(n_stops + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])

    return indptr, dst_u.astype(np.int32), wmin_u.astype(np.int32)


def csr_to_coo(indptr: np.ndarray, indices: np.ndarray, wmins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of coo_to_csr_deterministic: CSR → (src, dst, wmin)."""
    n = len(indptr) - 1
    edge_counts = indptr[1:] - indptr[:-1]
    src = np.repeat(np.arange(n, dtype=np.int32), edge_counts)
    return src, indices.astype(np.int32), wmins.astype(np.int32)


def csr_equal_bytewise(a: tuple, b: tuple) -> bool:
    """Bitwise equality test on two CSR triples (used by deterministic-build test)."""
    if a is None or b is None:
        return a is b
    ai, ad, aw = a
    bi, bd, bw = b
    return (np.array_equal(ai, bi) and np.array_equal(ad, bd) and np.array_equal(aw, bw))


def csr_n_edges(csr: tuple) -> int:
    return int(len(csr[1]))
