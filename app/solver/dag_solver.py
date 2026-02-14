"""
DAGSolver — Directed Acyclic Graph solver using Kahn's algorithm,
with convergent iterative solving for feedback loops (cycles).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import networkx as nx

from app.core.nodes import BaseNode
from app.core.state import SystemState
from app.solver.interface import SolverInterface

logger = logging.getLogger(__name__)

# Convergence parameters
DEFAULT_CONVERGENCE_THRESHOLD = 0.001
DEFAULT_MAX_ITERATIONS = 100


class DAGSolver(SolverInterface):
    """
    Solver that uses Kahn's algorithm for acyclic topological sorting
    and iterative convergence for any detected cycles.
    """

    def __init__(
        self,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    # ── Public API ─────────────────────────────────────────────────

    def solve(
        self,
        nodes: dict[str, BaseNode],
        edges: list[tuple[str, str]],
        state: SystemState,
    ) -> tuple[SystemState, list[str]]:
        """
        Build a NetworkX DiGraph, detect cycles, and execute:
        1. Acyclic nodes via Kahn's topological sort.
        2. Cyclic sub-graphs via convergent iteration.
        """
        graph = self._build_graph(nodes, edges)
        execution_order: list[str] = []

        # Separate into acyclic and cyclic components
        cycles = list(nx.simple_cycles(graph))

        if not cycles:
            # Pure DAG — straightforward topological sort
            order = self._kahns_sort(graph)
            for node_id in order:
                if node_id in nodes:
                    nodes[node_id].compute(state)
                    execution_order.append(node_id)
            return state, execution_order

        # Mixed graph: some cycles exist
        cycle_node_ids: set[str] = set()
        for cycle in cycles:
            cycle_node_ids.update(cycle)

        acyclic_node_ids = set(graph.nodes) - cycle_node_ids

        # 1. Build a subgraph of only the acyclic nodes and sort them
        acyclic_subgraph = graph.subgraph(acyclic_node_ids).copy()
        # Remove edges pointing into cycle nodes
        acyclic_order = self._kahns_sort(acyclic_subgraph)

        # Execute acyclic nodes that come *before* cycle nodes
        # (i.e., they don't depend on any cycle output)
        pre_cycle: list[str] = []
        post_cycle: list[str] = []
        for nid in acyclic_order:
            depends_on_cycle = any(
                pred in cycle_node_ids for pred in graph.predecessors(nid)
            )
            if depends_on_cycle:
                post_cycle.append(nid)
            else:
                pre_cycle.append(nid)

        for node_id in pre_cycle:
            if node_id in nodes:
                nodes[node_id].compute(state)
                execution_order.append(node_id)

        # 2. Convergent iteration on cycle nodes
        cycle_order = self._resolve_cycle_order(graph, cycle_node_ids)
        state, cycle_exec = self._iterate_until_converged(
            nodes, cycle_order, state
        )
        execution_order.extend(cycle_exec)

        # 3. Execute post-cycle acyclic nodes
        for node_id in post_cycle:
            if node_id in nodes:
                nodes[node_id].compute(state)
                execution_order.append(node_id)

        return state, execution_order

    # ── Kahn's Algorithm ───────────────────────────────────────────

    def _kahns_sort(self, graph: nx.DiGraph) -> list[str]:
        """
        Kahn's algorithm: BFS-based topological sort.

        Process:
        1. Compute in-degree for all nodes.
        2. Enqueue all nodes with in-degree 0.
        3. Dequeue a node, add to result, and decrement in-degree of
           all its successors. Enqueue successors reaching in-degree 0.
        4. Repeat until queue is empty.
        """
        in_degree: dict[str, int] = {n: 0 for n in graph.nodes}
        for _u, v in graph.edges:
            in_degree[v] += 1

        queue: deque[str] = deque(
            node for node, deg in in_degree.items() if deg == 0
        )
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(graph.nodes):
            # Not all nodes were processed — there's a cycle that
            # wasn't caught (should not happen in the acyclic subgraph).
            missing = set(graph.nodes) - set(result)
            logger.warning(
                "Kahn's sort incomplete. Remaining nodes (likely cyclic): %s",
                missing,
            )
        return result

    # ── Cycle resolution ───────────────────────────────────────────

    def _resolve_cycle_order(
        self, graph: nx.DiGraph, cycle_nodes: set[str]
    ) -> list[str]:
        """
        Determine a reasonable execution order for cycle nodes.
        Uses a simple heuristic: sort by in-degree within the cycle subgraph.
        """
        subgraph = graph.subgraph(cycle_nodes)
        return sorted(
            cycle_nodes,
            key=lambda n: subgraph.in_degree(n),  # type: ignore[arg-type]
        )

    def _iterate_until_converged(
        self,
        nodes: dict[str, BaseNode],
        cycle_order: list[str],
        state: SystemState,
    ) -> tuple[SystemState, list[str]]:
        """
        Repeatedly execute cycle nodes until the state converges
        (max delta < threshold) or max_iterations is hit.
        """
        execution_log: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            snapshot = state.snapshot()

            for node_id in cycle_order:
                if node_id in nodes:
                    nodes[node_id].compute(state)
                    execution_log.append(node_id)

            delta = state.delta(snapshot)
            logger.debug(
                "Cycle iteration %d: max_delta=%.6f", iteration, delta
            )

            if delta < self.convergence_threshold:
                logger.info(
                    "Converged after %d iterations (delta=%.6f)",
                    iteration,
                    delta,
                )
                break
        else:
            logger.warning(
                "Cycle did NOT converge after %d iterations (delta=%.6f). "
                "Threshold was %.6f.",
                self.max_iterations,
                state.delta(snapshot),  # type: ignore[possibly-undefined]
                self.convergence_threshold,
            )

        return state, execution_log

    # ── Helper ─────────────────────────────────────────────────────

    @staticmethod
    def _build_graph(
        nodes: dict[str, BaseNode], edges: list[tuple[str, str]]
    ) -> nx.DiGraph:
        """Build a NetworkX DiGraph from node dict and edge list."""
        g = nx.DiGraph()
        g.add_nodes_from(nodes.keys())
        g.add_edges_from(edges)
        return g
