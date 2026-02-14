"""
FaultComparator — compares live sensor vectors against stored signatures
using Euclidean Distance, Cosine Similarity, and Residual Scoring.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.fault.signature_store import SignatureStore

logger = logging.getLogger(__name__)


class FaultComparator:
    """
    Accepts a live vector and compares it against every entry
    in a SignatureStore to find the closest fault signature and
    detect anomalies.
    """

    def __init__(self, store: SignatureStore) -> None:
        self.store = store

    # ── Distance metrics ───────────────────────────────────────────

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean (L2) distance between two vectors."""
        return float(np.linalg.norm(a - b))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity ∈ [-1, 1].
        Returns 1.0 for identical directions, 0.0 for orthogonal.
        """
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def compute_residual(
        live: np.ndarray, predicted: np.ndarray
    ) -> dict[str, Any]:
        """
        Element-wise residual (Live − Predicted).

        Returns
        -------
        dict with:
          residual_vector : list[float]
          max_residual    : float
          mean_residual   : float
          rms_residual    : float
        """
        residual = live - predicted
        return {
            "residual_vector": residual.tolist(),
            "max_residual": float(np.max(np.abs(residual))),
            "mean_residual": float(np.mean(np.abs(residual))),
            "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
        }

    # ── Full diagnosis ─────────────────────────────────────────────

    def diagnose(
        self,
        live_vector: list[float] | np.ndarray,
        threshold: float = 5.0,
        method: str = "euclidean",
    ) -> dict[str, Any]:
        """
        Compare a live sensor vector against every stored signature.

        Parameters
        ----------
        live_vector : array-like
            The current sensor reading vector.
        threshold : float
            If the residual score exceeds this, flag a fault.
        method : str
            "euclidean" or "cosine".

        Returns
        -------
        dict with: closest_signature, distance, cosine_sim,
                   residual, fault_detected, all_comparisons
        """
        live = np.asarray(live_vector, dtype=np.float64)
        library = self.store.get_library()

        if not library:
            return {
                "closest_signature": None,
                "distance": None,
                "cosine_sim": None,
                "residual": None,
                "fault_detected": True,
                "message": "No signatures in store for comparison.",
                "all_comparisons": [],
            }

        comparisons: list[dict[str, Any]] = []
        best_name: str | None = None
        best_score: float = float("inf")

        for name, stored in library.items():
            # Ensure matching lengths
            min_len = min(len(live), len(stored))
            l_vec = live[:min_len]
            s_vec = stored[:min_len]

            euc = self.euclidean_distance(l_vec, s_vec)
            cos = self.cosine_similarity(l_vec, s_vec)
            res = self.compute_residual(l_vec, s_vec)

            score = euc if method == "euclidean" else (1.0 - cos)

            comparisons.append({
                "signature": name,
                "euclidean_distance": euc,
                "cosine_similarity": cos,
                "max_residual": res["max_residual"],
                "rms_residual": res["rms_residual"],
            })

            if score < best_score:
                best_score = score
                best_name = name

        # Get residual for the closest match
        best_stored = library[best_name]  # type: ignore[index]
        min_len = min(len(live), len(best_stored))
        residual_info = self.compute_residual(
            live[:min_len], best_stored[:min_len]
        )

        fault_detected = residual_info["max_residual"] > threshold

        logger.info(
            "Diagnosis: closest=%s  dist=%.4f  fault=%s",
            best_name, best_score, fault_detected,
        )

        return {
            "closest_signature": best_name,
            "distance": best_score,
            "cosine_similarity": self.cosine_similarity(
                live[:min_len], best_stored[:min_len]
            ),
            "residual": residual_info,
            "fault_detected": fault_detected,
            "all_comparisons": sorted(
                comparisons, key=lambda c: c["euclidean_distance"]
            ),
        }
