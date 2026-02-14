"""Tests for DAGSolver — topological sort and convergence."""

from app.core.nodes import FormulaNode
from app.core.state import SystemState
from app.solver.dag_solver import DAGSolver


def _make_simple_dag():
    """A → B → C  (linear chain)."""
    nodes = {
        "A": FormulaNode("A", outputs=["a_out"], params={"expression": "1 + 1"}),
        "B": FormulaNode("B", inputs=["a_out"], outputs=["b_out"], params={"expression": "a_out * 2"}),
        "C": FormulaNode("C", inputs=["b_out"], outputs=["c_out"], params={"expression": "b_out + 10"}),
    }
    edges = [("A", "B"), ("B", "C")]
    return nodes, edges


def test_topological_order():
    """Execution order must respect dependencies: A before B before C."""
    solver = DAGSolver()
    nodes, edges = _make_simple_dag()
    state = SystemState()

    final, order = solver.solve(nodes, edges, state)

    assert order.index("A") < order.index("B")
    assert order.index("B") < order.index("C")


def test_computed_values():
    """Values should propagate correctly through the DAG."""
    solver = DAGSolver()
    nodes, edges = _make_simple_dag()
    state = SystemState()

    final, _ = solver.solve(nodes, edges, state)

    assert abs(final.get("a_out") - 2.0) < 1e-6
    assert abs(final.get("b_out") - 4.0) < 1e-6
    assert abs(final.get("c_out") - 14.0) < 1e-6


def test_convergent_cycle():
    """
    Test convergent solving with a feedback loop.

    Setup: two nodes that feed into each other.
    - X computes: x_val = 0.5 * y_val + 1
    - Y computes: y_val = 0.3 * x_val + 2

    Analytical fixed point:
      x = 0.5y + 1
      y = 0.3x + 2
      => x = 0.5(0.3x + 2) + 1 = 0.15x + 2
      => 0.85x = 2 => x ≈ 2.3529
      => y = 0.3 * 2.3529 + 2 ≈ 2.7059
    """
    nodes = {
        "X": FormulaNode("X", inputs=["y_val"], outputs=["x_val"],
                         params={"expression": "0.5 * y_val + 1"}),
        "Y": FormulaNode("Y", inputs=["x_val"], outputs=["y_val"],
                         params={"expression": "0.3 * x_val + 2"}),
    }
    edges = [("X", "Y"), ("Y", "X")]

    solver = DAGSolver(convergence_threshold=0.001, max_iterations=200)
    state = SystemState({"x_val": 0.0, "y_val": 0.0})

    final, order = solver.solve(nodes, edges, state)

    assert abs(final.get("x_val") - 2.3529) < 0.01
    assert abs(final.get("y_val") - 2.7059) < 0.01
    assert len(order) > 2  # multiple iterations


def test_single_node_no_edges():
    """A single node with no edges should still execute."""
    solver = DAGSolver()
    nodes = {
        "alone": FormulaNode("alone", outputs=["val"], params={"expression": "42"}),
    }
    state = SystemState()
    final, order = solver.solve(nodes, [], state)

    assert abs(final.get("val") - 42.0) < 1e-6
    assert order == ["alone"]
