"""
SystemState — Central state container for the compute graph.

Tracks all variables (Torque, Temperature, Tension, Current, etc.)
across the graph during a solve pass, enabling convergence checks
for iterative feedback loops.
"""

from __future__ import annotations

import copy
from typing import Any


class SystemState:
    """
    A dictionary-like container that holds all named variables
    produced and consumed by nodes during graph execution.
    """

    def __init__(self, initial: dict[str, float] | None = None) -> None:
        self._variables: dict[str, float] = dict(initial) if initial else {}

    # ── Read / Write ───────────────────────────────────────────────

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the current value of *name*, or *default* if unset."""
        return self._variables.get(name, default)

    def set(self, name: str, value: float) -> None:
        """Set *name* to *value*."""
        self._variables[name] = value

    def has(self, name: str) -> bool:
        """Check if a variable exists in state."""
        return name in self._variables

    # ── Bulk operations ────────────────────────────────────────────

    def snapshot(self) -> dict[str, float]:
        """Return a deep-copy snapshot of all current values."""
        return copy.deepcopy(self._variables)

    def to_dict(self) -> dict[str, float]:
        """Return a shallow copy suitable for serialization."""
        return dict(self._variables)

    def update(self, mapping: dict[str, float]) -> None:
        """Merge *mapping* into the current state."""
        self._variables.update(mapping)

    # ── Convergence helpers ────────────────────────────────────────

    def delta(self, previous_snapshot: dict[str, float]) -> float:
        """
        Compute the maximum absolute difference between the current
        state and a *previous_snapshot*.

        Used to decide whether an iterative feedback loop has converged.
        Only compares keys that exist in **both** the current state and
        the snapshot.
        """
        if not previous_snapshot:
            return float("inf")

        max_delta = 0.0
        for key in self._variables:
            if key in previous_snapshot:
                diff = abs(self._variables[key] - previous_snapshot[key])
                if diff > max_delta:
                    max_delta = diff
        return max_delta

    # ── Dunder helpers ─────────────────────────────────────────────

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self._variables.items()))
        return f"SystemState({items})"

    def __len__(self) -> int:
        return len(self._variables)

    def __contains__(self, name: str) -> bool:
        return name in self._variables
