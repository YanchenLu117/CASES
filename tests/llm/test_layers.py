"""V9 layer registry + context-budget precheck contracts."""

from __future__ import annotations

import pytest

from cases.llm.context_budget import (
    TokenStats,
    check_layer,
    estimate_tokens,
    measure_texts,
)
from cases.llm.layers import (
    LayerNotConfigured,
    get_layer,
    layer_status,
    roster,
)


# ---------------------------------------------------------------------------
# roster + env resolution
# ---------------------------------------------------------------------------


def test_roster_has_six_layers_in_batch_order():
    names = roster()
    assert len(names) == 6
    # batch 1 = gpt_oss + qwen36 + glm53flash (user ruling 2026-09-04);
    # everything else is batch 2 (nemotron35 removed per r31, kept in roster
    # for history; gemma4 reinstated as regression layer; qwen38 runs last)
    batch1 = {n for n in names if get_layer(n, strict=False).batch == 1}
    assert batch1 == {"gpt_oss_20b", "qwen36_35b_a3b", "glm53_flash"}
    assert names[-1] == "qwen38_27b"  # dense 27B layer runs last
    assert "nemotron35_30b_a3b" in names  # r31-removed, roster slot kept
    # batches are non-decreasing in roster order
    batches = [get_layer(n, strict=False).batch for n in names]
    assert batches == sorted(batches)


def test_layer_resolves_from_env(monkeypatch):
    monkeypatch.setenv("CASES_LAYER_GPT_OSS_20B_BASE_URL", "http://10.0.0.1:8000/v1")
    monkeypatch.setenv("CASES_LAYER_GPT_OSS_20B_API_KEY", "sk-test")
    monkeypatch.setenv("CASES_LAYER_GPT_OSS_20B_MODEL", "gpt-oss-20b")
    monkeypatch.setenv("CASES_LAYER_GPT_OSS_20B_MAX_MODEL_LEN", "131072")
    cfg = get_layer("gpt_oss_20b")
    assert cfg.base_url == "http://10.0.0.1:8000/v1"
    assert cfg.model == "gpt-oss-20b"
    assert cfg.max_model_len == 131072
    assert cfg.prefix == "gpt_oss_20b_"
    assert cfg.local is True


def test_unwired_layer_refuses_with_missing_names(monkeypatch, tmp_path):
    monkeypatch.delenv("CASES_LAYER_GLM53_FLASH_BASE_URL", raising=False)
    monkeypatch.delenv("CASES_LAYER_GLM53_FLASH_API_KEY", raising=False)
    monkeypatch.delenv("CASES_LAYER_GLM53_FLASH_MODEL", raising=False)
    # hermetic: isolate from any experiments/v9_layers/<name>/layer.env fallback
    import cases.llm.layers as layers
    monkeypatch.setattr(layers, "_REPO_ROOT", tmp_path)
    with pytest.raises(LayerNotConfigured) as err:
        get_layer("glm53_flash")
    assert "BASE_URL" in str(err.value)


def test_status_reports_wiring_without_raising(monkeypatch, tmp_path):
    monkeypatch.delenv("CASES_LAYER_QWEN36_35B_A3B_BASE_URL", raising=False)
    monkeypatch.delenv("CASES_LAYER_QWEN36_35B_A3B_API_KEY", raising=False)
    monkeypatch.delenv("CASES_LAYER_QWEN36_35B_A3B_MODEL", raising=False)
    # hermetic: isolate from any experiments/v9_layers/<name>/layer.env fallback
    import cases.llm.layers as layers
    monkeypatch.setattr(layers, "_REPO_ROOT", tmp_path)
    rows = layer_status()
    assert len(rows) == 6
    by_name = {r["name"]: r for r in rows}
    assert by_name["qwen36_35b_a3b"]["wired"] is False


def test_unknown_layer_rejected():
    with pytest.raises(KeyError):
        get_layer("does_not_exist")


# ---------------------------------------------------------------------------
# context budget
# ---------------------------------------------------------------------------


def test_estimate_tokens_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("a" * 100) == 25


def test_measure_texts_percentiles():
    # 10 texts of 400..4000 chars -> 100..1000 tokens
    texts = ["x" * (400 * (i + 1)) for i in range(10)]
    stats = measure_texts(texts)
    assert stats.n == 10
    assert stats.p50 == 500  # nearest-rank: 5th of 10 sorted
    assert stats.p90 == 900  # nearest-rank: 9th of 10 sorted
    assert stats.max == 1000


def test_measure_empty_refuses():
    with pytest.raises(ValueError):
        measure_texts([])


def test_check_layer_gate_math():
    # p90=1000 tokens; window 8192, gen budget 2000, headroom 5% -> usable 5782
    texts = ["x" * 4000] * 10
    ok, stats, reason = check_layer(texts, max_model_len=8192, max_tokens=2000)
    assert ok is True
    assert "p90=1000" in reason

    # shrink the window so p90 no longer fits: usable = 2048*0.95 - 1500 = 445
    ok2, _stats2, reason2 = check_layer(texts, max_model_len=2048, max_tokens=1500)
    assert ok2 is False
    assert "REFUSED" in reason2


def test_tokenstats_fits_headroom():
    stats = TokenStats(n=1, p50=900, p90=950, max=1000)
    # usable = 4096*0.95 - 3000 = 891 -> 950 does not fit
    assert stats.fits(4096, 3000) is False
    assert stats.fits(8192, 3000) is True


def test_layer_env_file_fallback(tmp_path, monkeypatch):
    """Layer resolution falls back to experiments/v9_layers/<name>/layer.env
    when the environment variables are unset; env vars win when set."""
    import cases.llm.layers as layers

    name = roster()[0]
    envs = layers._env_stem(name)
    for var in envs.values():
        monkeypatch.delenv(var, raising=False)
    d = tmp_path.joinpath(*layers._LAYER_ENV_RELPATH, name)
    d.mkdir(parents=True)
    lines = [
        f"{envs['base_url']}=http://localhost:19999/v1",
        f"{envs['api_key']}=sk-test-file-fallback",
        f"{envs['model']}=test-model-id",
        f"{envs['max_tokens']}=1234",
        "# a comment line that must be ignored",
    ]
    (d / "layer.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(layers, "_REPO_ROOT", tmp_path)

    cfg = layers.get_layer(name, strict=True)
    assert cfg.base_url == "http://localhost:19999/v1"
    assert cfg.model == "test-model-id"
    assert cfg.max_tokens == 1234
    assert layers.get_layer_secret(cfg) == "sk-test-file-fallback"

    # environment variables win over the file
    monkeypatch.setenv(envs["base_url"], "http://env-wins.example/v1")
    cfg2 = layers.get_layer(name, strict=True)
    assert cfg2.base_url == "http://env-wins.example/v1"
    assert cfg2.model == "test-model-id"  # still from the file
