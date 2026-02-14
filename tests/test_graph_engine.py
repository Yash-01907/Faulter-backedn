"""Tests for GraphEngine — end-to-end with React Flow JSON."""

import json
import os

from app.engine.graph_engine import GraphEngine


SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "samples", "sample_graph.json"
)


def _load_sample() -> dict:
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def test_execution_order_from_sample():
    """The sample graph should resolve to a valid topological order."""
    engine = GraphEngine()
    graph = _load_sample()
    order = engine.get_execution_order(graph)

    # formula-total must come after motor-1, heater-1, hydraulic-1
    assert order.index("formula-total") > order.index("motor-1")
    assert order.index("formula-total") > order.index("heater-1")
    assert order.index("formula-total") > order.index("hydraulic-1")
    # sweep-tension must come last
    assert order.index("sweep-tension") > order.index("formula-total")


def test_solve_sample_graph():
    """Full solve should produce non-zero total_current."""
    engine = GraphEngine()
    graph = _load_sample()
    result = engine.run(
        graph,
        initial_state={
            "torque": 5.0,
            "speed": 1500,
            "temperature": 80.0,
            "pressure": 500.0,
            "flow_rate": 0.3,
        },
    )
    state = result["state"]

    assert state["motor_current"] > 0
    assert state["heater_current"] > 0
    assert state["hydraulic_current"] > 0
    assert state["total_current"] > 0
    assert result["node_count"] == 5
    assert result["edge_count"] == 4


def test_sweep():
    """Sweep should produce a vector of the correct length."""
    engine = GraphEngine()
    graph = _load_sample()
    result = engine.run_sweep(
        graph,
        sweep_node_id="sweep-tension",
        initial_state={
            "speed": 1500,
            "temperature": 80.0,
            "pressure": 500.0,
            "flow_rate": 0.3,
        },
    )
    assert result["sweep_var"] == "torque"
    assert result["output_var"] == "total_current"
    assert result["steps"] == 50
    assert len(result["signature_vector"]) == 50
    # As torque increases, motor current increases, so total current should increase
    assert result["signature_vector"][-1] > result["signature_vector"][0]
