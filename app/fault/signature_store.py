"""
SignatureStore — In-memory library of named signature vectors.

Each entry is a "Stored Line" produced by a SweepNode.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SignatureStore:
    """
    Thread-safe, in-memory store for named current-signature vectors.
    """

    def __init__(self) -> None:
        self._library: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a signature vector under *name*."""
        self._library[name] = np.asarray(vector, dtype=np.float64)
        self._metadata[name] = metadata or {}

    def get(self, name: str) -> np.ndarray:
        """Retrieve a stored vector by name. Raises KeyError if missing."""
        if name not in self._library:
            raise KeyError(f"Signature {name!r} not in store.")
        return self._library[name]

    def remove(self, name: str) -> None:
        """Remove a stored signature."""
        self._library.pop(name, None)
        self._metadata.pop(name, None)

    def list_all(self) -> list[dict[str, Any]]:
        """Return a summary of all stored signatures."""
        return [
            {
                "name": name,
                "length": len(vec),
                "min": float(vec.min()),
                "max": float(vec.max()),
                "metadata": self._metadata.get(name, {}),
            }
            for name, vec in self._library.items()
        ]

    def get_library(self) -> dict[str, np.ndarray]:
        """Return the full library dict (for comparison)."""
        return dict(self._library)

    def __len__(self) -> int:
        return len(self._library)

    def __contains__(self, name: str) -> bool:
        return name in self._library
