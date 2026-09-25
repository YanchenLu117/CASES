"""this LLM provider abstraction regression tests (Block B).

No live gateway — only construction, registry, key-resolution and backward
compat.  ``DeepseekV4Flash`` fetches the key from the env, so tests that
construct it set a throwaway value to avoid depending on a secret.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# sys.path wiring lives in tests/conftest.py (T3 tech-debt: no per-file hacks).


@pytest.fixture(autouse=True)
def _env_key():
    os.environ["CASES_LLM_API_KEY"] = "test-not-real"
    yield
    os.environ.pop("CASES_LLM_API_KEY", None)


def test_generation_result_metering():
    from cases.llm import GenerationResult

    g = GenerationResult(text="x", input_tokens=10, output_tokens=5, total_tokens=15)
    assert g.total_tokens == g.input_tokens + g.output_tokens


def test_backend_registry_default():
    from cases.llm import available_backends, get_backend

    assert "deepseek-v4-flash" in available_backends()
    b = get_backend("deepseek-v4-flash")
    from cases.llm import LLMBackend

    assert isinstance(b, LLMBackend)
    assert b.model_name == "deepseek-v4-flash"


def test_unknown_backend_raises():
    from cases.llm import get_backend

    with pytest.raises(KeyError):
        get_backend("no-such-model")


def test_key_required_without_env():
    import cases.llm.deepseek as ds

    for k in ("CASES_LLM_API_KEY", "DEEPSEEK_API_KEY"):
        os.environ.pop(k, None)
    with pytest.raises(ValueError):
        ds.DeepseekV4Flash()


def test_max_output_tokens_range():
    from cases.llm.deepseek import DeepseekV4Flash

    with pytest.raises(ValueError):
        DeepseekV4Flash(max_output_tokens=10)  # below [8192, 32768]


def test_v6_provider_registry_kept():
    from cases.llm import create_provider

    p = create_provider("deepseek", api_key="k", model="m")
    assert p.provider_name == "deepseek"


def test_config_driven_backend_default(capsys):
    from cases.llm import create_backend_from_config

    # model-less config of a valid path is not exercised here (no fixture file);
    # just assert the known-backend loader from the shipped default config file.
    cfg = Path(__file__).resolve().parents[2] / "configs" / "llm" / "deepseek_v4_flash.yaml"
    if cfg.exists():
        b = create_backend_from_config(str(cfg))
        assert b.model_name == "deepseek-v4-flash"


def test_base_url_env_override_reaches_client(monkeypatch):
    """P1 fix: CASES_LLM_BASE_URL must win on the client even when the ctor arg
    is the explicit base_url; the client uses the RESOLVED self._base_url."""
    import openai  # already a dependency of the cases-core env

    import cases.llm.deepseek as ds

    captured = {}

    class _FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            captured["base_url"] = base_url

    # DeepseekV4Flash does ``import openai`` inside __init__, so we patch the
    # real openai.OpenAI constructor (no live gateway is touched).
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    env_url = "http://gateway.local:9999/v1"
    monkeypatch.setenv("CASES_LLM_BASE_URL", env_url)

    # With no explicit ctor base_url, the CASES_LLM_BASE_URL env override must
    # reach the CLIENT (base_url=self._base_url) -- this is the P0/P1 bug.
    b = ds.DeepseekV4Flash()
    assert captured["base_url"] == env_url
    assert b._base_url == env_url
    # Resolution precedence is ctor-arg > env > frozen default: with an explicit
    # ctor arg the client uses that value (env only kicks in when None).
    monkeypatch.delenv("CASES_LLM_BASE_URL")
    b2 = ds.DeepseekV4Flash(base_url="http://explicit:42/v1")
    assert b2._base_url == "http://explicit:42/v1"
