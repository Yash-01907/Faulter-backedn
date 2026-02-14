"""Tests for FaultComparator and SignatureStore."""

import numpy as np

from app.fault.comparator import FaultComparator
from app.fault.signature_store import SignatureStore


def test_signature_store_crud():
    store = SignatureStore()
    store.add("normal", [1.0, 2.0, 3.0])
    store.add("fault_a", [1.5, 2.5, 3.5], {"description": "bearing wear"})

    assert len(store) == 2
    assert "normal" in store
    np.testing.assert_array_equal(store.get("normal"), [1.0, 2.0, 3.0])

    summaries = store.list_all()
    assert len(summaries) == 2

    store.remove("fault_a")
    assert len(store) == 1
    assert "fault_a" not in store


def test_euclidean_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    dist = FaultComparator.euclidean_distance(a, b)
    assert abs(dist - np.sqrt(2)) < 1e-6


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    sim = FaultComparator.cosine_similarity(a, a)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sim = FaultComparator.cosine_similarity(a, b)
    assert abs(sim) < 1e-6


def test_residual():
    live = np.array([10.0, 20.5, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])
    res = FaultComparator.compute_residual(live, predicted)
    assert abs(res["max_residual"] - 0.5) < 1e-6
    assert abs(res["mean_residual"] - 0.5 / 3) < 1e-6


def test_diagnose_no_fault():
    store = SignatureStore()
    store.add("normal", [1.0, 2.0, 3.0, 4.0, 5.0])
    store.add("fault_a", [10.0, 20.0, 30.0, 40.0, 50.0])

    comp = FaultComparator(store)
    result = comp.diagnose(
        live_vector=[1.1, 2.1, 3.1, 4.1, 5.1],
        threshold=5.0,
    )
    assert result["closest_signature"] == "normal"
    assert result["fault_detected"] is False


def test_diagnose_fault_detected():
    store = SignatureStore()
    store.add("normal", [1.0, 2.0, 3.0, 4.0, 5.0])

    comp = FaultComparator(store)
    result = comp.diagnose(
        live_vector=[10.0, 20.0, 30.0, 40.0, 50.0],
        threshold=5.0,
    )
    assert result["fault_detected"] is True


def test_diagnose_empty_store():
    store = SignatureStore()
    comp = FaultComparator(store)
    result = comp.diagnose([1.0, 2.0])
    assert result["closest_signature"] is None
    assert result["fault_detected"] is True
