"""TEEG-D Compiled: Numba-JIT Dijkstra on the TEEG event graph.

Public API:
    teeg_dijkstra_compiled(g, source, dest, t_dep) -> int | None
"""
from .query import teeg_dijkstra_compiled, INF_I32

__all__ = ["teeg_dijkstra_compiled", "INF_I32"]
