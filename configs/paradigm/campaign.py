"""Campaign config schema — dataclass validation via OmegaConf.structured.

We use OmegaConf + dataclasses (strict typed validation, YAML/CLI merge)
WITHOUT the full Hydra launcher stack: the paradigm needs supervisor-driven
batch loops over a frozen preregistry, not per-run dynamic composition, and
a plain `OmegaConf.merge(structured, yaml)` keeps the dependency surface
minimal on air-gapped containers (c89). Rationale recorded in README / ADR-0001.
"""
from dataclasses import dataclass, field


@dataclass
class CapEnvelope:
    llm_calls: int = 40
    generated_tokens: int = 40960
    state_tokens: int = 4096
    tool_calls: int = 400
    oracle: int = 40
    wall_s: int = 3600


@dataclass
class Endpoint:
    base_url: str = "https://example.invalid/v1"
    model: str = "mock-model"


@dataclass
class CampaignConfig:
    name: str = "demo"
    arms: list[str] = field(default_factory=lambda: ["native", "cases_state"])
    instances: list[str] = field(default_factory=lambda: ["inst_a"])
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    endpoint: Endpoint = field(default_factory=Endpoint)
    cap_envelope: CapEnvelope = field(default_factory=CapEnvelope)
    wandb_project: str | None = None  # None => NullRun (offline ledger only)

    def cells(self) -> list[tuple[str, str, int]]:
        return [(a, i, s) for a in self.arms
                for i in self.instances for s in self.seeds]


def load_config(path: str) -> CampaignConfig:
    """Load a YAML campaign config, validated against the dataclass schema."""
    from omegaconf import OmegaConf
    base = OmegaConf.structured(CampaignConfig)
    if path:
        user = OmegaConf.load(path)
        merged = OmegaConf.merge(base, user)
    else:
        merged = base
    cfg: CampaignConfig = OmegaConf.to_object(merged)
    if not cfg.name or not cfg.arms or not cfg.instances:
        raise ValueError(f"invalid campaign config: {cfg}")
    return cfg
