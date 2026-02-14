"""Tests for node classes."""

import math

from app.core.nodes import (
    FormulaNode,
    HeaterNode,
    HydraulicNode,
    MotorNode,
    SweepNode,
)
from app.core.state import SystemState


def test_motor_node():
    node = MotorNode(
        node_id="m1",
        inputs=["torque", "speed"],
        outputs=["motor_current"],
        params={"voltage": 230, "efficiency": 0.85},
    )
    state = SystemState({"torque": 5.0, "speed": 1500})
    node.compute(state)

    expected_omega = 1500 * 2 * math.pi / 60
    expected_power = 5.0 * expected_omega
    expected_current = expected_power / (0.85 * 230)

    assert abs(state.get("motor_current") - expected_current) < 1e-6


def test_heater_node():
    node = HeaterNode(
        node_id="h1",
        inputs=["temperature"],
        outputs=["heater_resistance", "heater_current"],
        params={"r0": 10.0, "alpha": 0.004, "t0": 25.0, "voltage": 230},
    )
    state = SystemState({"temperature": 100.0})
    node.compute(state)

    expected_r = 10.0 * (1 + 0.004 * (100.0 - 25.0))
    expected_i = 230 / expected_r

    assert abs(state.get("heater_resistance") - expected_r) < 1e-6
    assert abs(state.get("heater_current") - expected_i) < 1e-6


def test_hydraulic_node():
    node = HydraulicNode(
        node_id="hyd1",
        inputs=["pressure", "flow_rate"],
        outputs=["hydraulic_current"],
        params={"voltage": 400, "efficiency": 0.8},
    )
    state = SystemState({"pressure": 1000.0, "flow_rate": 0.5})
    node.compute(state)

    expected_power = 1000.0 * 0.5
    expected_current = expected_power / (0.8 * 400)

    assert abs(state.get("hydraulic_current") - expected_current) < 1e-6


def test_formula_node():
    node = FormulaNode(
        node_id="f1",
        inputs=["a", "b"],
        outputs=["result"],
        params={"expression": "a + b * 2"},
    )
    state = SystemState({"a": 3.0, "b": 4.0})
    node.compute(state)
    assert abs(state.get("result") - 11.0) < 1e-6


def test_formula_node_with_math():
    node = FormulaNode(
        node_id="f2",
        outputs=["out"],
        params={"expression": "sqrt(16) + pi"},
    )
    state = SystemState()
    node.compute(state)
    assert abs(state.get("out") - (4.0 + math.pi)) < 1e-6


def test_sweep_node_properties():
    node = SweepNode(
        node_id="sw1",
        params={
            "sweep_var": "torque",
            "output_var": "current",
            "min_val": 1.0,
            "max_val": 10.0,
            "steps": 50,
        },
    )
    assert node.sweep_var == "torque"
    assert node.output_var == "current"
    assert node.min_val == 1.0
    assert node.max_val == 10.0
    assert node.steps == 50

    rng = node.get_sweep_range()
    assert len(rng) == 50
    assert abs(rng[0] - 1.0) < 1e-9
    assert abs(rng[-1] - 10.0) < 1e-9
