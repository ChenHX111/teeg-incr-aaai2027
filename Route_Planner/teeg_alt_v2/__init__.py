"""TEEG-ALT v2 — tightened ALT with multi-restart landmarks, source-aware
Goldberg-Harrelson active landmark selection, and push-time f-pruning.

Public API:
  - select_landmarks_v2(lb_graph, n_landmarks, n_restarts, seed)
  - LandmarkLabelsV2(lb_graph, landmarks)            (adds .d_from_T)
  - teeg_alt_v2_compiled(g, source, dest, t_dep, labels_v2, n_active=6)
"""
from .landmarks import LandmarkLabelsV2, select_landmarks_v2
from .query import INF_I32, teeg_alt_v2_compiled

__all__ = [
    "INF_I32",
    "LandmarkLabelsV2",
    "select_landmarks_v2",
    "teeg_alt_v2_compiled",
]
