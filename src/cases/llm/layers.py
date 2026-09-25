"""Model-layer registry — one disciplined layer per LLM backbone.

A *layer* bundles everything a model-layer campaign needs: the endpoint, the
model id, the context budget, rate caps, and the file-prefix discipline
(separate ledgers / canaries / run dirs per layer; layers never pool — the
M8-deepseek / qwen35 layered-presentation rule).

All secrets resolve from environment variables so no key ever lands in git::

    CASES_LAYER_<NAME>_BASE_URL      # e.g. http://127.0.0.1:8000/v1 (user-built gateway)
    CASES_LAYER_<NAME>_API_KEY       # the gateway key
    CASES_LAYER_<NAME>_MODEL         # model field for the chat request

Optional per-layer knobs (with sane defaults)::

    CASES_LAYER_<NAME>_MAX_MODEL_LEN # context window (default: probe the endpoint)
    CASES_LAYER_<NAME>_MAX_TOKENS    # generation budget (reasoning models need >= 2500)
    CASES_LAYER_<NAME>_RPM           # rate cap, requests per minute
    CASES_LAYER_<NAME>_CONCURRENCY   # parallel in-flight requests

The endpoint/gateway itself is built and operated by the user (2026-09-02
decision); this module only *consumes* it.  GLM-5.3-Flash is an API layer with
a low rate limit and runs as batch 4, ahead of the still-downloading Qwen3.8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """One model layer of the layer matrix."""

    name: str                 # canonical layer name (dir / prefix / env stem)
    label: str                # human description (roster from the deployment plan)
    batch: int                # launch batch (1..4) per the anti-overload plan
    base_url: str
    model: str
    api_key_env: str
    max_model_len: int | None = None
    max_tokens: int = 4096
    rpm: int | None = None
    concurrency: int = 4
    local: bool = True        # False for the GLM-5.3-Flash API layer

    @property
    def prefix(self) -> str:
        """File-prefix discipline: every artifact of this layer starts here."""
        return f"{self.name}_"

    def env_names(self) -> dict[str, str]:
        stem = f"CASES_LAYER_{self.name.upper()}"
        return {
            "base_url": f"{stem}_BASE_URL",
            "api_key": f"{stem}_API_KEY",
            "model": f"{stem}_MODEL",
            "max_model_len": f"{stem}_MAX_MODEL_LEN",
            "max_tokens": f"{stem}_MAX_TOKENS",
            "rpm": f"{stem}_RPM",
            "concurrency": f"{stem}_CONCURRENCY",
        }


# ---------------------------------------------------------------------------
# Canonical layer roster
# 批次（user ruling 2026-09-04，唯一权威口径，与 V9_TEAM_DISPATCH 活规则一致）:
#   batch 1 = gpt_oss_20b + qwen36_35b_a3b + glm53_flash (API)  第一批
#   batch 2 = 其余模型组（nemotron35 已按 r31 整体移除；gemma4_12b 回归层
#             2026-09-03 regrade 保留；qwen38_27b 停放至其窗口，runs last）
# roster 顺序仍按 dispatch 六队表：qwen36, gpt_oss, nemotron35(removed),
# gemma4, glm53, qwen38 — batch 号单调不减（tests/llm/test_layers.py 契约）。
# ---------------------------------------------------------------------------
_ROSTER: tuple[dict, ...] = (
    dict(name="qwen36_35b_a3b", label="Qwen3.6-35B-A3B — 35B / 3B active MoE, research workhorse", batch=1),
    dict(name="gpt_oss_20b", label="GPT-oss-20b — MoE 3.6B active, local llama.cpp Q8_0", batch=1),
    dict(name="nemotron35_30b_a3b", label="Nemotron 3.5 Lightning 30B-A3B — 30B / 3B active MoE, agent/efficiency (REMOVED r31, kept for history)", batch=2),
    dict(name="gemma4_12b", label="Gemma-4-12B dense — Q4_K_M, local llama.cpp GPU 2, ruling-5 regression layer", batch=2),
    dict(name="glm53_flash", label="GLM-5.3-Flash — remote API, low rate limit", batch=1, local=False),
    dict(name="qwen38_27b", label="Qwen3.8-27B dense — unsloth UD-Q4_K_M, local llama.cpp, runs last", batch=2),
)


class LayerNotConfigured(RuntimeError):
    """A roster layer is missing its endpoint environment variables."""


_LAYER_ENV_RELPATH = ("experiments", "v9_layers")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _layer_env_file_values(name: str) -> dict[str, str]:
    """Config-file fallback: read ``experiments/v9_layers/<name>/layer.env``.

    Real environment variables win; the file exists so runners resolve layers
    without a manual ``source`` step. Never raises -- an absent or unreadable
    file simply means "no fallback values".
    """
    path = _REPO_ROOT.joinpath(*_LAYER_ENV_RELPATH, name, "layer.env")
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def get_layer_secret(cfg: LayerConfig) -> str:
    """The layer API key value: environment variable first, then the layer.env file."""
    return os.environ.get(cfg.api_key_env, "") or _layer_env_file_values(cfg.name).get(cfg.api_key_env, "")


def get_layer(name: str, *, strict: bool = True) -> LayerConfig:
    """Resolve a roster layer from the environment.

    ``strict=True`` (default) raises :class:`LayerNotConfigured` when the
    endpoint is not wired yet — the right behaviour for launch paths.  Pass
    ``strict=False`` for status reporting (fields left empty).
    """
    spec = next((s for s in _ROSTER if s["name"] == name), None)
    if spec is None:
        raise KeyError(f"unknown layer: {name!r} (roster: {[s['name'] for s in _ROSTER]})")
    envs = _env_stem(spec["name"])
    file_vals = _layer_env_file_values(name)

    def _var(var: str) -> str:
        return os.environ.get(var, "") or file_vals.get(var, "")

    def _var_int(var: str) -> int | None:
        raw = _var(var)
        return int(raw) if raw else None

    base_url = _var(envs["base_url"])
    model = _var(envs["model"])
    has_key = bool(_var(envs["api_key"]))
    if strict and not (base_url and model and has_key):
        missing = [
            label
            for label, var in (("BASE_URL", envs["base_url"]), ("API_KEY", envs["api_key"]), ("MODEL", envs["model"]))
            if not _var(var)
        ]
        raise LayerNotConfigured(
            f"layer {name!r} not wired: set {', '.join(missing)} (see experiments/v9_layers/{name}/layer.env.example)"
        )
    return LayerConfig(
        name=spec["name"],
        label=spec["label"],
        batch=spec["batch"],
        local=spec.get("local", True),
        base_url=base_url,
        model=model,
        api_key_env=envs["api_key"],
        max_model_len=_var_int(envs["max_model_len"]),
        max_tokens=_var_int(envs["max_tokens"]) or 4096,
        rpm=_var_int(envs["rpm"]),
        concurrency=_var_int(envs["concurrency"]) or 4,
    )


def layer_status() -> list[dict]:
    """Roster table with wiring state — for status_gen / dispatch preflight."""
    rows = []
    for spec in _ROSTER:
        try:
            cfg = get_layer(spec["name"], strict=True)
            wired = True
        except LayerNotConfigured:
            cfg = get_layer(spec["name"], strict=False)
            wired = False
        rows.append(
            {
                "name": cfg.name,
                "batch": cfg.batch,
                "label": cfg.label,
                "wired": wired,
                "base_url": cfg.base_url or "-",
                "model": cfg.model or "-",
                "max_model_len": cfg.max_model_len or "probe",
                "rpm": cfg.rpm or "-",
            }
        )
    return rows


def roster() -> tuple[str, ...]:
    """Canonical layer names in batch order."""
    return tuple(s["name"] for s in sorted(_ROSTER, key=lambda s: (s["batch"], s["name"])))


def _env_stem(name: str) -> dict[str, str]:
    stem = f"CASES_LAYER_{name.upper()}"
    return {
        "base_url": f"{stem}_BASE_URL",
        "api_key": f"{stem}_API_KEY",
        "model": f"{stem}_MODEL",
        "max_model_len": f"{stem}_MAX_MODEL_LEN",
        "max_tokens": f"{stem}_MAX_TOKENS",
        "rpm": f"{stem}_RPM",
        "concurrency": f"{stem}_CONCURRENCY",
    }


def _opt_int(var: str) -> int | None:
    raw = os.environ.get(var, "").strip()
    return int(raw) if raw.isdigit() else None
