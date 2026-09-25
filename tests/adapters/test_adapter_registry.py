"""P0-2 — adapter registry mechanism + registration of existing adapters."""

from __future__ import annotations

import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


def test_registry_surface():
    import cases.adapters
    assert hasattr(cases.adapters, "register_adapter")
    assert hasattr(cases.adapters, "get_adapter")
    assert hasattr(cases.adapters, "create_adapter_from_config")


def test_hypospace_get_adapter_can_construct():
    import cases.adapters as A
    assert "hypospace" in A.available_adapters()
    factory = A.get_adapter("hypospace")
    assert callable(factory)
    adapter = A.create_adapter_from_config("hypospace")
    from cases.adapters.hypospace.boolean import BooleanAdapter
    assert isinstance(adapter, BooleanAdapter)
    assert adapter.task_id == "hypospace_boolean"


def test_unknown_adapter_raises():
    import cases.adapters as A
    with pytest.raises(KeyError):
        A.get_adapter("no-such-adapter")
    with pytest.raises(KeyError):
        A.create_adapter_from_config("no-such-adapter")


def test_hypospace_domain_dispatch():
    import cases.adapters as A
    from cases.adapters.hypospace.causal import CausalAdapter
    from cases.adapters.hypospace.voxel3d import Voxel3DAdapter

    causal = A.create_adapter_from_config(
        "hypospace",
        {"domain": "causal", "task": {
            "dataset": "c", "observation_set_id": "o",
            "nodes": ("A", "B"), "max_edges": 1,
            "observations": (), "n_observations": 0,
        }},
    )
    assert isinstance(causal, CausalAdapter)
