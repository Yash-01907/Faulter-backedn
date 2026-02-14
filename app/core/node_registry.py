"""
Node Registry — maps React Flow `type` strings to Python node classes.

This is the single extensibility point for adding new node types.
"""

from __future__ import annotations

from typing import Any, Type

from app.core.nodes import (
    BaseNode,
    FormulaNode,
    HeaterNode,
    HydraulicNode,
    MotorNode,
    SweepNode,
)

# ── Default registry ───────────────────────────────────────────────

_REGISTRY: dict[str, Type[BaseNode]] = {
    "motor": MotorNode,
    "heater": HeaterNode,
    "hydraulic": HydraulicNode,
    "formula": FormulaNode,
    "sweep": SweepNode,
}


def register_node_type(type_name: str, cls: Type[BaseNode]) -> None:
    """Register a new node type (or override an existing one)."""
    _REGISTRY[type_name] = cls


def get_node_class(type_name: str) -> Type[BaseNode]:
    """
    Look up the Python class for a React Flow node type.

    Raises KeyError if the type is not registered.
    """
    if type_name not in _REGISTRY:
        raise KeyError(
            f"Unknown node type {type_name!r}. "
            f"Registered types: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[type_name]


def list_node_types() -> list[str]:
    """Return all registered node type names."""
    return list(_REGISTRY.keys())


def create_node(
    type_name: str,
    node_id: str,
    label: str = "",
    params: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> BaseNode:
    """
    Factory: instantiate a node by its React Flow type string.
    """
    cls = get_node_class(type_name)
    return cls(
        node_id=node_id,
        label=label,
        params=params,
        inputs=inputs,
        outputs=outputs,
    )
