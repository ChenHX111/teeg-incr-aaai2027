"""Numba-compiled TEEG query algorithms."""
from .query import (
    INF_I32,
    teeg_alt_compiled,
    teeg_dijkstra_compiled,
)

__all__ = ["INF_I32", "teeg_alt_compiled", "teeg_dijkstra_compiled"]
