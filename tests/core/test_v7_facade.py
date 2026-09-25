"""this three-operation facade regression tests (Block A).

Pure-computation path only — no LLM gateway required.  Covered:
  - import surface + exports
  - update -> recover -> readout end-to-end (posterior non-None, non-empty text)
  - empty-evidence graceful degradation
  - default-backend bare ``recover()`` must not crash (graceful fallback)
  - token_budget validation
  - idempotent re-update
"""

from __future__ import annotations

import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


def _boolean_adapter():
    from cases.adapters.hypospace.boolean import BooleanAdapter
    from cases.adapters.hypospace.tasks import BooleanTask

    task = BooleanTask(
        dataset="boolean_test",
        observation_set_id="t",
        variables=("x", "y"),
        operators=frozenset({"AND", "OR", "NOT"}),
        max_depth=2,
        mechanistic_opts={
            "apply_commutativity": True,
            "apply_idempotence_and_or": True,
            "flatten_associativity": True,
        },
        observations=(),
        n_observations=0,
    )
    return BooleanAdapter(task)


def _batch(n=4):
    from cases.core.model import EvidenceRecord
    from cases.core.types import LanguageHypothesis

    texts = ["x AND y", "x OR y", "NOT x", "x AND NOT y"]
    return [
        EvidenceRecord(
            kind="hypothesis",
            hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t),
        )
        for i, t in enumerate(texts[:n])
    ]


def test_import_surface():
    import cases

    assert hasattr(cases, "update") and hasattr(cases, "recover") and hasattr(cases, "readout")
    from cases.core.model import CASESModel  # noqa: F401

    import cases.adapters  # noqa: F401


def test_update_recover_readout_end_to_end():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    snap = m.update(_batch())
    assert snap["n_objects"] == 4
    st = m.recover()
    assert st.posterior is not None
    text = m.readout(st, token_budget=1800)
    assert text.strip()
    assert "Target" in text  # frontier semantics keyword
    assert len(text) <= 1800 * 4  # frozen render cap (4 chars/token approx)


def test_default_backend_bare_recover_no_crash():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())  # default backend = laplacian
    m.update(_batch())
    st = m.recover()  # no features
    assert st.posterior is not None
    assert st.metadata.get("recovery_backend") in ("laplacian", "mean", "rbf_gp")


def test_empty_evidence_graceful():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    snap = m.update([])
    assert snap["n_objects"] == 0
    st = m.recover()
    assert st.posterior is not None
    assert m.readout(st, token_budget=1800).strip()


def test_token_budget_validation():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    m.update(_batch())
    st = m.recover()
    with pytest.raises(ValueError):
        m.readout(st, token_budget=0)
    with pytest.raises(ValueError):
        m.readout(st, token_budget=-5)


def test_update_idempotent_unless_logged():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    batch = _batch()
    m.update(batch)
    assert m.state is not None
    n_objects = len(m.state.objects)
    m.update(batch)  # re-apply same batch
    assert len(m.state.objects) == n_objects  # dict keyed by id, idempotent
    # evidence log is a RAW append counter (documented), so it grows
    assert len(m.evidence_log) == 2 * len(batch)


def test_language_correspondence_wraps_adapter():
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())
    gamma = m.language_correspondence
    assert callable(gamma.descriptor)
    assert callable(gamma.target_instruction)
    assert callable(gamma.render_object)

def test_readout_leakage_discipline():
    """Readout must expose NO evaluator-only ground truth (no GT, no |H_O|,
    no admissible set) - HypoSpace boolean built from PUBLIC grammar only."""
    from cases.core.model import CASESModel

    m = CASESModel(adapter=_boolean_adapter())  # empty observations, no GT
    m.update(_batch())
    st = m.recover()
    text = m.readout(st, token_budget=1800)
    packet = m.readout_packet(st)
    blob = text + "\n" + str(packet)
    for tok in ("ground_truth", "|H_O|", "admissible set", "full solution",
                "n_compatible", "recovery="):
        assert tok.lower() not in blob.lower(), "leak token in readout: %s" % tok
    # boolean has no scalar observations; nothing evaluator-only may enter state
    assert len(m.state.observations) == 0
    assert packet.constructed_count == len(_batch())
    # structure present: covered region + at least one frontier target
    assert packet.covered_regions
    assert packet.frontier_targets
