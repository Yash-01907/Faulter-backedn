"""Tests for SystemState."""

from app.core.state import SystemState


def test_get_set():
    s = SystemState()
    s.set("torque", 5.0)
    assert s.get("torque") == 5.0
    assert s.get("nonexistent") == 0.0
    assert s.get("nonexistent", -1.0) == -1.0


def test_has_and_contains():
    s = SystemState({"x": 1.0})
    assert s.has("x")
    assert "x" in s
    assert not s.has("y")


def test_snapshot_is_independent():
    s = SystemState({"a": 1.0, "b": 2.0})
    snap = s.snapshot()
    s.set("a", 99.0)
    assert snap["a"] == 1.0  # snapshot unchanged


def test_delta():
    s = SystemState({"a": 10.0, "b": 20.0})
    snap = s.snapshot()
    s.set("a", 10.5)
    s.set("b", 19.8)
    assert abs(s.delta(snap) - 0.5) < 1e-9


def test_delta_empty_snapshot():
    s = SystemState({"a": 1.0})
    assert s.delta({}) == float("inf")


def test_update():
    s = SystemState({"a": 1.0})
    s.update({"b": 2.0, "c": 3.0})
    assert s.get("b") == 2.0
    assert s.get("c") == 3.0


def test_to_dict():
    s = SystemState({"x": 42.0})
    d = s.to_dict()
    assert d == {"x": 42.0}
    d["x"] = 0  # mutating the returned dict shouldn't affect state
    assert s.get("x") == 42.0


def test_len_and_repr():
    s = SystemState({"a": 1.0, "b": 2.0})
    assert len(s) == 2
    assert "a=" in repr(s)
