"""Landmark selection and labels for TEEG-ALT v2.

select_landmarks_v2:
    Multi-restart farthest-point (Maxmin) sampling on the lower-bound graph.
    Each restart starts from a different random non-isolated stop; we score
    each candidate set by the average ALT lower-bound tightness on a sample
    of random (s, t) pairs (Goldberg-Harrelson §4.1).

LandmarkLabelsV2:
    Like the v1 LandmarkLabels but additionally exposes ``d_from_T`` —
    a C-contiguous transposed copy of d_from with shape (n_stops, L).
    This is the layout required by the v2 Numba kernels for cache-friendly
    row-major access during per-query active-landmark selection and
    h_stop computation.
"""
from __future__ import annotations

import numpy as np

LABEL_INF = np.int32(32_000)


def _maxmin_select(lb_graph, n_landmarks: int, first: int) -> np.ndarray:
    """Farthest-point (Maxmin) sampling starting from ``first``."""
    from scipy.sparse.csgraph import dijkstra
    landmarks = [int(first)]
    dist = dijkstra(lb_graph, directed=True, indices=int(first),
                    return_predecessors=False)
    dist = np.where(np.isinf(dist), -1.0, dist)
    for _ in range(n_landmarks - 1):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break
        landmarks.append(nxt)
        d2 = dijkstra(lb_graph, directed=True, indices=int(nxt),
                      return_predecessors=False)
        d2 = np.where(np.isinf(d2), -1.0, d2)
        # Maxmin: keep min distance to any landmark
        dist = np.minimum(dist, d2)
    return np.asarray(landmarks, dtype=np.int32)


def _evaluate_landmarks(lb_graph, landmarks: np.ndarray,
                        sample_pairs: int, seed: int) -> float:
    """Mean ALT-LB / true-LB ratio on ``sample_pairs`` random reachable (s,t)."""
    from scipy.sparse.csgraph import dijkstra
    n = lb_graph.shape[0]
    L = len(landmarks)
    if L == 0:
        return 0.0

    # Forward and backward labels for the candidate set
    d_from = np.zeros((L, n), dtype=np.int32)
    d_to = np.zeros((n, L), dtype=np.int32)
    for i, lm in enumerate(landmarks):
        d = dijkstra(lb_graph, directed=True, indices=int(lm),
                     return_predecessors=False)
        d_from[i] = np.where(np.isinf(d), LABEL_INF, d).astype(np.int32)
    lb_T = lb_graph.T.tocsr()
    for i, lm in enumerate(landmarks):
        d = dijkstra(lb_T, directed=True, indices=int(lm),
                     return_predecessors=False)
        d_to[:, i] = np.where(np.isinf(d), LABEL_INF, d).astype(np.int32)

    # Sample sources & target; compute true LB via Dijkstra-from-source
    rng = np.random.default_rng(seed)
    deg = np.asarray((lb_graph != 0).sum(axis=1)).ravel()
    cand = np.where(deg > 0)[0]
    if len(cand) < 2:
        return 0.0

    n_src = min(32, sample_pairs)  # cap unique sources to bound Dijkstras
    srcs = rng.choice(cand, size=n_src, replace=len(cand) < n_src)
    targets_per_src = max(1, sample_pairs // n_src)

    ratios = []
    for s in srcs:
        true_d = dijkstra(lb_graph, directed=True, indices=int(s),
                          return_predecessors=False)
        # pick reachable targets
        reachable = np.where(np.isfinite(true_d) & (true_d > 0))[0]
        if reachable.size == 0:
            continue
        ts = rng.choice(reachable,
                        size=min(targets_per_src, reachable.size),
                        replace=False)
        for t in ts:
            true_lb = float(true_d[int(t)])
            if true_lb <= 0:
                continue
            # GH formula across all candidate landmarks
            term1 = d_to[int(s)] - d_to[int(t)]
            term2 = d_from[:, int(t)] - d_from[:, int(s)]
            alt_lb = float(max(0, max(term1.max(), term2.max())))
            ratios.append(alt_lb / true_lb)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def select_landmarks_v2(lb_graph, n_landmarks: int = 16,
                        n_restarts: int = 5, seed: int = 42,
                        sample_pairs: int = 200) -> np.ndarray:
    """Multi-restart Maxmin landmark selection.

    Parameters
    ----------
    lb_graph : scipy.sparse.csr_matrix (n_stops, n_stops)
        Per-stop lower-bound travel-time graph (admissible).
    n_landmarks : {8, 16, 32, 64} (configurable)
    n_restarts : number of independent Maxmin runs.
    seed : RNG seed for the master RNG.
    sample_pairs : (s, t) pairs used to score each restart.

    Returns
    -------
    landmarks : int32[n_landmarks]
    """
    rng = np.random.default_rng(seed)
    deg = np.asarray((lb_graph != 0).sum(axis=1)).ravel()
    cand = np.where(deg > 0)[0]
    if len(cand) == 0:
        return np.zeros(0, dtype=np.int32)

    n_landmarks = int(min(n_landmarks, len(cand)))
    if n_landmarks <= 0:
        return np.zeros(0, dtype=np.int32)

    best = None
    best_score = -1.0
    for r in range(n_restarts):
        first = int(rng.choice(cand))
        cand_set = _maxmin_select(lb_graph, n_landmarks, first)
        if len(cand_set) == 0:
            continue
        score = _evaluate_landmarks(lb_graph, cand_set, sample_pairs,
                                    seed=int(rng.integers(0, 2**31 - 1)))
        if score > best_score:
            best_score = score
            best = cand_set

    if best is None:
        # Fallback: single Maxmin from a deterministic seed
        first = int(cand[0])
        best = _maxmin_select(lb_graph, n_landmarks, first)
    return np.asarray(best, dtype=np.int32)


class LandmarkLabelsV2:
    """Precomputed landmark distances with a transposed d_from view.

    Attributes
    ----------
    landmarks : int32[L]
    d_from    : int32[L, n_stops]   d(landmark, stop)
    d_to      : int32[n_stops, L]   d(stop, landmark)
    d_from_T  : int32[n_stops, L]   d(landmark, stop) transposed (cache view)
    """

    def __init__(self, lb_graph, landmarks: np.ndarray):
        from scipy.sparse.csgraph import dijkstra
        self.landmarks = np.ascontiguousarray(landmarks, dtype=np.int32)
        n = lb_graph.shape[0]
        L = int(len(self.landmarks))
        self.d_from = np.zeros((L, n), dtype=np.int32)
        self.d_to = np.zeros((n, L), dtype=np.int32)
        for i, lm in enumerate(self.landmarks):
            d = dijkstra(lb_graph, directed=True, indices=int(lm),
                         return_predecessors=False)
            self.d_from[i] = np.where(np.isinf(d), LABEL_INF, d).astype(np.int32)
        lb_T = lb_graph.T.tocsr()
        for i, lm in enumerate(self.landmarks):
            d = dijkstra(lb_T, directed=True, indices=int(lm),
                         return_predecessors=False)
            self.d_to[:, i] = np.where(np.isinf(d), LABEL_INF, d).astype(np.int32)
        # Transposed d_from for cache-friendly per-stop row-major access in kernels.
        self.d_from_T = np.ascontiguousarray(self.d_from.T, dtype=np.int32)

    # Convenience: same heuristic interface as v1 (used in unit tests / debug)
    def heuristic(self, stop_from: int, stop_to: int) -> int:
        if stop_from == stop_to:
            return 0
        term1 = self.d_to[stop_from] - self.d_to[stop_to]
        term2 = self.d_from_T[stop_to] - self.d_from_T[stop_from]
        return int(max(0, int(max(term1.max(), term2.max()))))
