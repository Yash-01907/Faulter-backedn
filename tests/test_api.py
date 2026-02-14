"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Faulter Core"
    assert data["status"] == "running"


SAMPLE_GRAPH = {
    "nodes": [
        {
            "id": "m1",
            "type": "motor",
            "data": {
                "label": "Motor",
                "params": {"voltage": 230, "efficiency": 0.85},
                "inputs": ["torque", "speed"],
                "outputs": ["motor_current"],
            },
        },
        {
            "id": "f1",
            "type": "formula",
            "data": {
                "label": "Double",
                "params": {"expression": "motor_current * 2"},
                "inputs": ["motor_current"],
                "outputs": ["doubled"],
            },
        },
    ],
    "edges": [{"source": "m1", "target": "f1"}],
    "initial_state": {"torque": 5.0, "speed": 1500},
}


def test_solve():
    resp = client.post("/api/solve", json=SAMPLE_GRAPH)
    assert resp.status_code == 200
    data = resp.json()
    assert "motor_current" in data["state"]
    assert "doubled" in data["state"]
    assert data["state"]["doubled"] > 0
    assert data["node_count"] == 2


def test_store_and_list_signatures():
    # Store
    resp = client.post(
        "/api/signatures",
        json={"name": "test_sig", "vector": [1.0, 2.0, 3.0]},
    )
    assert resp.status_code == 201

    # List
    resp = client.get("/api/signatures")
    assert resp.status_code == 200
    sigs = resp.json()
    names = [s["name"] for s in sigs]
    assert "test_sig" in names


def test_diagnose():
    # First store a reference signature
    client.post(
        "/api/signatures",
        json={"name": "ref", "vector": [1.0, 2.0, 3.0, 4.0]},
    )

    # Diagnose with similar vector (no fault)
    resp = client.post(
        "/api/diagnose",
        json={"vector": [1.1, 2.1, 3.1, 4.1], "threshold": 5.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fault_detected"] is False

    # Diagnose with very different vector (fault)
    resp = client.post(
        "/api/diagnose",
        json={"vector": [100.0, 200.0, 300.0, 400.0], "threshold": 5.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fault_detected"] is True


def test_delete_signature():
    client.post(
        "/api/signatures",
        json={"name": "to_delete", "vector": [1.0]},
    )
    resp = client.delete("/api/signatures/to_delete")
    assert resp.status_code == 200

    resp = client.delete("/api/signatures/nonexistent")
    assert resp.status_code == 404
