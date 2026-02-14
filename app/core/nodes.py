"""
Node classes — each maps to a physical component (or formula)
from the React Flow frontend.

Every node reads inputs from SystemState, computes its output(s),
writes the results back into SystemState, and returns the mutated state.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.core.state import SystemState


# ── Abstract base ──────────────────────────────────────────────────

class BaseNode(ABC):
    """
    Base class for all compute-graph nodes.

    Attributes
    ----------
    node_id : str
        Unique identifier (matches the React Flow node id).
    label : str
        Human-readable label.
    params : dict
        Node-specific configuration from the frontend JSON.
    inputs : list[str]
        Variable names this node reads from SystemState.
    outputs : list[str]
        Variable names this node writes to SystemState.
    """

    def __init__(
        self,
        node_id: str,
        label: str = "",
        params: dict[str, Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
    ) -> None:
        self.node_id = node_id
        self.label = label or node_id
        self.params = params or {}
        self.inputs = inputs or []
        self.outputs = outputs or []

    @abstractmethod
    def compute(self, state: SystemState) -> SystemState:
        """Execute this node's logic and return the updated state."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.node_id!r}>"


# ── Concrete nodes ─────────────────────────────────────────────────

class MotorNode(BaseNode):
    """
    Motor component.

    Converts *Torque* and *Speed* (RPM) into a *Current* draw using:
        Current = (Torque × Speed) / (Efficiency × Voltage)

    Params
    ------
    voltage : float   (default 230)
    efficiency : float (default 0.85, range 0-1)
    """

    def compute(self, state: SystemState) -> SystemState:
        torque = state.get(self.inputs[0] if self.inputs else "torque")
        speed = state.get(self.inputs[1] if len(self.inputs) > 1 else "speed")
        voltage = self.params.get("voltage", 230.0)
        efficiency = self.params.get("efficiency", 0.85)

        # Power (W) = Torque (N·m) × Angular velocity (rad/s)
        omega = speed * 2 * math.pi / 60  # RPM → rad/s
        power = torque * omega
        current = power / (efficiency * voltage) if (efficiency * voltage) != 0 else 0.0

        output_name = self.outputs[0] if self.outputs else "motor_current"
        state.set(output_name, current)
        return state


class HeaterNode(BaseNode):
    """
    Heater component.

    Models temperature-dependent resistance and current:
        R(T) = R0 × (1 + α × (T - T0))
        Current = Voltage / R(T)

    Params
    ------
    r0 : float         Base resistance at T0 (default 10 Ω)
    alpha : float       Temp coefficient (default 0.004 /°C)
    t0 : float          Reference temperature (default 25 °C)
    voltage : float     Supply voltage (default 230 V)
    """

    def compute(self, state: SystemState) -> SystemState:
        temperature = state.get(self.inputs[0] if self.inputs else "temperature")
        r0 = self.params.get("r0", 10.0)
        alpha = self.params.get("alpha", 0.004)
        t0 = self.params.get("t0", 25.0)
        voltage = self.params.get("voltage", 230.0)

        resistance = r0 * (1 + alpha * (temperature - t0))
        current = voltage / resistance if resistance != 0 else 0.0

        # Write outputs
        out_resistance = self.outputs[0] if self.outputs else "heater_resistance"
        out_current = self.outputs[1] if len(self.outputs) > 1 else "heater_current"
        state.set(out_resistance, resistance)
        state.set(out_current, current)
        return state


class HydraulicNode(BaseNode):
    """
    Hydraulic pump / actuator.

    Converts *Pressure* and *Flow Rate* into a *Current* draw:
        Power = Pressure × Flow_Rate
        Current = Power / (Efficiency × Voltage)

    Params
    ------
    voltage : float    (default 400)
    efficiency : float (default 0.80)
    """

    def compute(self, state: SystemState) -> SystemState:
        pressure = state.get(self.inputs[0] if self.inputs else "pressure")
        flow_rate = state.get(self.inputs[1] if len(self.inputs) > 1 else "flow_rate")
        voltage = self.params.get("voltage", 400.0)
        efficiency = self.params.get("efficiency", 0.80)

        power = pressure * flow_rate
        current = power / (efficiency * voltage) if (efficiency * voltage) != 0 else 0.0

        output_name = self.outputs[0] if self.outputs else "hydraulic_current"
        state.set(output_name, current)
        return state


class FormulaNode(BaseNode):
    """
    Generic formula evaluator.

    Evaluates an arithmetic expression string using current state values
    as variables.  E.g.  expression = "torque * 1.5 + temperature * 0.01"

    Params
    ------
    expression : str    The formula string to evaluate.
    """

    # Allowed builtins for safe eval
    _SAFE_BUILTINS = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
    }

    def compute(self, state: SystemState) -> SystemState:
        expression: str = self.params.get("expression", "0")

        # Build a local namespace from the current state + safe builtins
        namespace: dict[str, Any] = {**state.to_dict(), **self._SAFE_BUILTINS}

        try:
            result = float(eval(expression, {"__builtins__": {}}, namespace))  # noqa: S307
        except Exception as exc:
            raise ValueError(
                f"FormulaNode {self.node_id!r}: failed to evaluate "
                f"expression {expression!r} — {exc}"
            ) from exc

        output_name = self.outputs[0] if self.outputs else "formula_result"
        state.set(output_name, result)
        return state


class SweepNode(BaseNode):
    """
    Parameter sweep — generates a *signature vector*.

    Iterates a target parameter from *min_val* to *max_val* in *steps*,
    runs the upstream graph for each value, and collects the output
    variable into a NumPy vector (the "Stored Line").

    Params
    ------
    sweep_var : str       Variable name to sweep (e.g. "tension").
    output_var : str      Variable name to collect (e.g. "motor_current").
    min_val : float       Sweep start.
    max_val : float       Sweep end.
    steps : int           Number of sweep points (default 100).

    Note: The actual sweep is orchestrated by the GraphEngine, which
    re-solves the upstream graph for each sweep value. This node stores
    the configuration and the resulting vector.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.result_vector: np.ndarray | None = None

    @property
    def sweep_var(self) -> str:
        return self.params.get("sweep_var", "")

    @property
    def output_var(self) -> str:
        return self.params.get("output_var", "")

    @property
    def min_val(self) -> float:
        return float(self.params.get("min_val", 0.0))

    @property
    def max_val(self) -> float:
        return float(self.params.get("max_val", 100.0))

    @property
    def steps(self) -> int:
        return int(self.params.get("steps", 100))

    def compute(self, state: SystemState) -> SystemState:
        """
        When called standalone (no sweep orchestration), simply read the
        current output_var and store it as a single-element vector.
        The full sweep is handled by GraphEngine.run_sweep().
        """
        val = state.get(self.output_var)
        output_name = self.outputs[0] if self.outputs else "sweep_result"
        state.set(output_name, val)
        return state

    def get_sweep_range(self) -> np.ndarray:
        """Return the linspace array for this sweep configuration."""
        return np.linspace(self.min_val, self.max_val, self.steps)
