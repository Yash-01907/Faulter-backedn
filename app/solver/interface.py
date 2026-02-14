"""
Solver Interface — abstract base for all solver strategies.

Design: Strategy pattern.  The GraphEngine delegates to whichever
SolverInterface implementation is configured, so you can swap in
MCSASolver, MatrixSolver, etc. without touching the rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.nodes import BaseNode
from app.core.state import SystemState


class SolverInterface(ABC):
    """
    Abstract solver that resolves execution order and runs the graph.
    """

    @abstractmethod
    def solve(
        self,
        nodes: dict[str, BaseNode],
        edges: list[tuple[str, str]],
        state: SystemState,
    ) -> tuple[SystemState, list[str]]:
        """
        Execute the computation graph.

        Parameters
        ----------
        nodes : dict mapping node_id → BaseNode instance
        edges : list of (source_id, target_id) tuples
        state : the initial SystemState

        Returns
        -------
        (final_state, execution_order)
            execution_order is a list of node_ids in the order they ran.
        """
        ...
