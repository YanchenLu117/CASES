"""Campaign-API boundary adapter template (CASES baselines接入四件套之一).

每个 S/X 基线系统的接入都必须经由本边界类与 benchmark 交互：
官方系统代码零改动；本类是唯一的翻译层。

契约（EXPERIMENT_PLAN_FINAL_V1 §5.1，逐条强制）：
- scientific content byte-identical：传给官方系统的任务/观测内容不得改写；
- no hidden labels/threshold/cardinality：任何接口调用不得暴露 GT、top-percent 阈值、解空间基数；
- live retrieval disabled；
- invalid/duplicate proposals 也消耗预注册的 proposal/tool budget（不许静默重试白嫖）；
- terminal oracle budget 冻结决策与分数；
- 每个请求/响应必须落日志且可重放（见 run_manifest.template.json 的 log 字段）。

复制本模板到 src/cases/adapters/<system>/ 后，把所有 TODO 替换为该系统的具体实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class BoundaryLog:
    """Append-only request/response log (replayable)."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, payload: Mapping[str, Any]) -> None:
        # TODO(system): attach timestamp, call args hash, response digest
        self.entries.append({"kind": kind, "payload": dict(payload)})


class CampaignBoundaryAdapter:
    """Template: route an official system through the frozen campaign API.

    Subclass per system (e.g. S2DiscoveryAdapter, X3PhysicsAdapter) and fill TODOs.
    """

    system_name = "TEMPLATE"
    official_repo_path = "baselines/<system>/"  # TODO(system): set
    pin_file = "baselines/<system>/PIN.txt"  # TODO(system): set

    def __init__(self, campaign_api: Any, config: Mapping[str, Any] | None = None) -> None:
        self.api = campaign_api  # cases.api.campaign shared B3/B4 API handle
        self.cfg = dict(config or {})
        self.log = BoundaryLog()

    # -- campaign API surface -------------------------------------------------

    def get_task_dossier(self) -> Mapping[str, Any]:
        """Public task, constraints, schema, metric definitions. No GT/threshold."""
        dossier = self.api.get_task_dossier()
        self.log.add("get_task_dossier", {"keys": sorted(dossier)})
        return dossier

    def get_observations(self) -> Sequence[Mapping[str, Any]]:
        """Queried observations only — never the full oracle table."""
        obs = self.api.get_observations()
        self.log.add("get_observations", {"n": len(obs)})
        return obs

    def get_candidate_space(self) -> Mapping[str, Any]:
        """Legal IDs / grammar / official generator handle."""
        space = self.api.get_candidate_space()
        self.log.add("get_candidate_space", {"keys": sorted(space)})
        return space

    def submit_batch(self, candidate_ids: Sequence[str], hypotheses: Sequence[str] | None = None,
                     rationale: Sequence[str] | None = None) -> Sequence[Mapping[str, Any]]:
        """Submit a batch; invalid/duplicate entries consume budget (no free retries)."""
        obs = self.api.submit_batch(list(candidate_ids), list(hypotheses or []),
                                    list(rationale or []))
        self.log.add("submit_batch", {"n": len(candidate_ids)})
        return obs

    def remaining_budget(self) -> Mapping[str, int]:
        return self.api.remaining_budget()

    def save_artifact(self, kind: str, payload: Mapping[str, Any]) -> str:
        return self.api.save_artifact(kind, payload)

    # -- identity contract ----------------------------------------------------

    def identity_checklist(self) -> dict[str, bool]:
        """Per §5.7: the frozen identity components this adapter must preserve.

        TODO(system): override with the system's real frozen components, e.g.
        for S2: tree search / experiment manager / reflection preserved;
        no CASES state injection; human interventions == 0.
        """
        return {
            "official_loop_unmodified": True,   # TODO(system): verify by diff
            "no_hidden_info_injected": True,
            "budget_charged_for_invalid": True,
            "logs_replayable": bool(self.log.entries),
        }
