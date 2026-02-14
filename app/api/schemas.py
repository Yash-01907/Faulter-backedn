"""
Pydantic schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Graph Computation ──────────────────────────────────────────────

class GraphInput(BaseModel):
    """React Flow JSON + optional initial state."""

    nodes: list[dict[str, Any]] = Field(..., description="React Flow nodes array")
    edges: list[dict[str, Any]] = Field(..., description="React Flow edges array")
    initial_state: dict[str, float] = Field(
        default_factory=dict,
        description="Pre-set variable values (e.g. sensor inputs)",
    )


class SolveResult(BaseModel):
    """Result of running the graph engine."""

    state: dict[str, float]
    execution_order: list[str]
    node_count: int
    edge_count: int


# ── Sweep ──────────────────────────────────────────────────────────

class SweepInput(BaseModel):
    """Sweep request: graph + which sweep node to execute."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    sweep_node_id: str
    initial_state: dict[str, float] = Field(default_factory=dict)


class SweepResult(BaseModel):
    """Result of a parameter sweep."""

    sweep_var: str
    output_var: str
    sweep_values: list[float]
    signature_vector: list[float]
    steps: int


# ── Signature Store ────────────────────────────────────────────────

class SignatureInput(BaseModel):
    """Store a named signature vector."""

    name: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignatureSummary(BaseModel):
    """Summary of a stored signature."""

    name: str
    length: int
    min: float
    max: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Fault Diagnosis ────────────────────────────────────────────────

class LiveVectorInput(BaseModel):
    """Live sensor data for fault diagnosis."""

    vector: list[float]
    threshold: float = Field(default=5.0, description="Residual threshold for fault flag")
    method: str = Field(default="euclidean", description="euclidean or cosine")


class ResidualInfo(BaseModel):
    residual_vector: list[float]
    max_residual: float
    mean_residual: float
    rms_residual: float


class ComparisonEntry(BaseModel):
    signature: str
    euclidean_distance: float
    cosine_similarity: float
    max_residual: float
    rms_residual: float


class DiagnosisResult(BaseModel):
    """Full fault diagnosis output."""

    closest_signature: str | None
    distance: float | None
    cosine_similarity: float | None
    residual: ResidualInfo | None
    fault_detected: bool
    message: str = ""
    all_comparisons: list[ComparisonEntry] = Field(default_factory=list)
