"""
GraphEngine — Top-level orchestrator.

Accepts React Flow JSON, builds node instances, constructs the
NetworkX graph, delegates to the configured Solver, and optionally
runs parameter sweeps for signature generation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from app.core.node_registry import create_node
from app.core.nodes import BaseNode, SweepNode
from app.core.state import SystemState
from app.solver.dag_solver import DAGSolver
from app.solver.interface import SolverInterface

logger = logging.getLogger(__name__)


class GraphEngine:
    """
    Main entry-point for graph computation.

    Usage
    -----
    >>> engine = GraphEngine()
    >>> result = engine.run(react_flow_json)
    >>> print(result["execution_order"])
    """

    def __init__(self, solver: SolverInterface | None = None) -> None:
        self.solver = solver or DAGSolver()
        self._nodes: dict[str, BaseNode] = {}
        self._edges: list[tuple[str, str]] = []

    # ── Public API ─────────────────────────────────────────────────

    def run(
        self,
        graph_json: dict[str, Any],
        initial_state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Parse a React Flow JSON payload, solve the graph, and return
        the computed state plus execution metadata.

        Parameters
        ----------
        graph_json : dict
            Must contain "nodes" and "edges" keys in React Flow format.
        initial_state : dict, optional
            Pre-set variable values (e.g. sensor inputs).

        Returns
        -------
        dict with keys: state, execution_order, node_count, edge_count
        """
        self._parse_graph(graph_json)
        state = SystemState(initial_state)

        final_state, execution_order = self.solver.solve(
            self._nodes, self._edges, state
        )

        return {
            "state": final_state.to_dict(),
            "execution_order": execution_order,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }

    def run_sweep(
        self,
        graph_json: dict[str, Any],
        sweep_node_id: str,
        initial_state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Run a parameter sweep using a specific SweepNode.

        For each value in the sweep range, re-solves the upstream graph
        and collects the output variable, producing a signature vector.

        Returns
        -------
        dict with keys: sweep_var, output_var, sweep_values, signature_vector
        """
        self._parse_graph(graph_json)

        sweep_node = self._nodes.get(sweep_node_id)
        if sweep_node is None:
            raise KeyError(f"Sweep node {sweep_node_id!r} not found.")
        if not isinstance(sweep_node, SweepNode):
            raise TypeError(
                f"Node {sweep_node_id!r} is {type(sweep_node).__name__}, "
                f"not a SweepNode."
            )

        sweep_values = sweep_node.get_sweep_range()
        results: list[float] = []

        for val in sweep_values:
            # Fresh state for each sweep point
            state = SystemState(initial_state)
            state.set(sweep_node.sweep_var, float(val))

            final_state, _ = self.solver.solve(
                self._nodes, self._edges, state
            )

            output_val = final_state.get(sweep_node.output_var)
            results.append(output_val)

        signature_vector = np.array(results)
        sweep_node.result_vector = signature_vector

        return {
            "sweep_var": sweep_node.sweep_var,
            "output_var": sweep_node.output_var,
            "sweep_values": sweep_values.tolist(),
            "signature_vector": signature_vector.tolist(),
            "steps": len(sweep_values),
        }

    def get_execution_order(
        self, graph_json: dict[str, Any]
    ) -> list[str]:
        """
        Parse a React Flow JSON and return ONLY the execution order
        (the node ids in topological order) without computing anything.
        This fulfills the 'Initial Task' from the spec.
        """
        self._parse_graph(graph_json)

        from app.solver.dag_solver import DAGSolver

        if isinstance(self.solver, DAGSolver):
            import networkx as nx

            g = nx.DiGraph()
            g.add_nodes_from(self._nodes.keys())
            g.add_edges_from(self._edges)
            return self.solver._kahns_sort(g)

        # Fallback: just solve with a dummy state and return the order
        state = SystemState()
        _, order = self.solver.solve(self._nodes, self._edges, state)
        return order

    # ── JSON parsing ───────────────────────────────────────────────

    def _parse_graph(self, graph_json: dict[str, Any]) -> None:
        """
        Convert React Flow JSON into internal node objects and edge lists.

        Expected JSON format:
        {
          "nodes": [
            {
              "id": "node-1",
              "type": "motor",
              "data": {
                "label": "Main Motor",
                "params": { "voltage": 230, "efficiency": 0.85 },
                "inputs": ["torque", "speed"],
                "outputs": ["motor_current"]
              }
            },
            ...
          ],
          "edges": [
            { "source": "node-1", "target": "node-2" },
            ...
          ]
        }
        """
        self._nodes.clear()
        self._edges.clear()

        raw_nodes = graph_json.get("nodes", [])
        raw_edges = graph_json.get("edges", [])

        for rn in raw_nodes:
            node_id = rn["id"]
            node_type = rn.get("type", "formula")
            data = rn.get("data", {})

            node = create_node(
                type_name=node_type,
                node_id=node_id,
                label=data.get("label", ""),
                params=data.get("params", {}),
                inputs=data.get("inputs", []),
                outputs=data.get("outputs", []),
            )
            self._nodes[node_id] = node

        for re_ in raw_edges:
            source = re_["source"]
            target = re_["target"]
            self._edges.append((source, target))

        logger.info(
            "Parsed graph: %d nodes, %d edges",
            len(self._nodes),
            len(self._edges),
        )
