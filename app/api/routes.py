"""
FastAPI routes for the Faulter Core backend.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ComparisonEntry,
    DiagnosisResult,
    GraphInput,
    LiveVectorInput,
    ResidualInfo,
    SignatureInput,
    SignatureSummary,
    SolveResult,
    SweepInput,
    SweepResult,
)
from app.engine.graph_engine import GraphEngine
from app.fault.comparator import FaultComparator
from app.fault.signature_store import SignatureStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared instances ───────────────────────────────────────────────
# In production you'd use dependency injection; here we keep it simple.

engine = GraphEngine()
signature_store = SignatureStore()
comparator = FaultComparator(signature_store)


# ── Graph Computation ──────────────────────────────────────────────

@router.post("/solve", response_model=SolveResult)
async def solve_graph(payload: GraphInput) -> SolveResult:
    """
    Accept a React Flow graph and compute the full state.
    Returns the final state and execution order.
    """
    try:
        graph_json = {"nodes": payload.nodes, "edges": payload.edges}
        result = engine.run(graph_json, payload.initial_state or None)
        return SolveResult(**result)
    except Exception as exc:
        logger.exception("Solve failed")
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/execution-order")
async def get_execution_order(payload: GraphInput) -> dict[str, Any]:
    """
    Returns ONLY the execution order (topological sort) without
    computing node values. Useful for validation/debugging.
    """
    try:
        graph_json = {"nodes": payload.nodes, "edges": payload.edges}
        order = engine.get_execution_order(graph_json)
        return {"execution_order": order}
    except Exception as exc:
        logger.exception("Execution order failed")
        raise HTTPException(status_code=400, detail=str(exc))


# ── Sweeps / Signature Generation ──────────────────────────────────

@router.post("/sweep", response_model=SweepResult)
async def run_sweep(payload: SweepInput) -> SweepResult:
    """
    Run a parameter sweep on a specific SweepNode.
    Returns the sweep values and the resulting signature vector.
    """
    try:
        graph_json = {"nodes": payload.nodes, "edges": payload.edges}
        result = engine.run_sweep(
            graph_json, payload.sweep_node_id, payload.initial_state or None
        )
        return SweepResult(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Sweep failed")
        raise HTTPException(status_code=400, detail=str(exc))


# ── Signature Store ────────────────────────────────────────────────

@router.post("/signatures", status_code=201)
async def store_signature(payload: SignatureInput) -> dict[str, str]:
    """Store a named signature vector in the library."""
    signature_store.add(payload.name, payload.vector, payload.metadata)
    return {"message": f"Signature '{payload.name}' stored ({len(payload.vector)} points)."}


@router.get("/signatures", response_model=list[SignatureSummary])
async def list_signatures() -> list[SignatureSummary]:
    """List all stored signature vectors."""
    return [SignatureSummary(**s) for s in signature_store.list_all()]


@router.delete("/signatures/{name}")
async def delete_signature(name: str) -> dict[str, str]:
    """Remove a stored signature."""
    if name not in signature_store:
        raise HTTPException(status_code=404, detail=f"Signature '{name}' not found.")
    signature_store.remove(name)
    return {"message": f"Signature '{name}' removed."}


# ── Fault Diagnosis ────────────────────────────────────────────────

@router.post("/diagnose", response_model=DiagnosisResult)
async def diagnose_fault(payload: LiveVectorInput) -> DiagnosisResult:
    """
    Compare a live sensor vector against stored signatures.
    Returns the closest match, distance, residual, and fault flag.
    """
    result = comparator.diagnose(
        live_vector=payload.vector,
        threshold=payload.threshold,
        method=payload.method,
    )

    # Map raw dict to Pydantic models
    residual_info = None
    if result.get("residual"):
        residual_info = ResidualInfo(**result["residual"])

    comparisons = [
        ComparisonEntry(**c) for c in result.get("all_comparisons", [])
    ]

    return DiagnosisResult(
        closest_signature=result.get("closest_signature"),
        distance=result.get("distance"),
        cosine_similarity=result.get("cosine_similarity"),
        residual=residual_info,
        fault_detected=result.get("fault_detected", False),
        message=result.get("message", ""),
        all_comparisons=comparisons,
    )
