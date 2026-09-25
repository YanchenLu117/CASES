"""P0-c tests part 2: representation induction (Detail §3.3) and controller (§3.4)."""

import pytest

from cases.protocol import (
    READOUT_ONLY_POLICY,
    ControllerWeights,
    FrozenController,
    InductionCallBudget,
    InductionOutcome,
    InductionPrompt,
    RepresentationInductor,
    ValidatorReport,
    check_proposal_schema,
)

PROMPT = InductionPrompt("induction_v1", "sha256:prompt")
SCHEMA = {
    "required": {"generators": list, "relations": list},
    "optional": {"notes": str},
}


class FakeValidator:
    validator_id = "fake_v1"

    def __init__(self, pass_on: set[str]):
        self.pass_on = pass_on
        self.checked: list[dict] = []

    def check(self, proposal):
        self.checked.append(dict(proposal))
        gens = proposal.get("generators", [])
        ok = any(g in self.pass_on for g in gens)
        report = ValidatorReport(
            passed=ok,
            problems=() if ok else ("no_recognized_generator",),
            actor_visible={"validator_id": self.validator_id, "problems": [] if ok else ["no_recognized_generator"]},
        )
        return report


class FakeProposer:
    def __init__(self, first, repair=None):
        self.first = first
        self.repair_result = repair
        self.propose_inputs: list[dict] = []
        self.repair_inputs: list[dict] = []

    def propose(self, serialized_context):
        self.propose_inputs.append(dict(serialized_context))
        return self.first

    def repair(self, actor_visible_report):
        self.repair_inputs.append(dict(actor_visible_report))
        return self.repair_result or {}


def _valid(gens=("g_bool_interaction",)):
    return {"generators": list(gens), "relations": []}


def test_admitted_on_first_proposal() -> None:
    proposer = FakeProposer(_valid())
    validator = FakeValidator({"g_bool_interaction"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, SCHEMA)
    result = inductor.run(dossier_ref="bh://main", visible_evidence={"obs": [1]})
    assert result.outcome is InductionOutcome.ADMITTED
    assert result.call_budget == InductionCallBudget(proposal_calls=1, repair_calls=0)
    assert result.contract == {"generators": ["g_bool_interaction"], "relations": []}
    # serialized context carries dossier + evidence + prompt identity
    ctx = proposer.propose_inputs[0]
    assert ctx["dossier_ref"] == "bh://main"
    assert ctx["prompt_id"] == "induction_v1"
    assert ctx["previous_contract"] is None


def test_schema_violation_skips_validator_and_goes_to_repair() -> None:
    proposer = FakeProposer({"generators": "not_a_list"}, repair=_valid())
    validator = FakeValidator({"g_bool_interaction"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, SCHEMA)
    result = inductor.run(dossier_ref="d", visible_evidence={})
    assert result.outcome is InductionOutcome.ADMITTED_AFTER_REPAIR
    # the rejected first proposal never reached the validator (schema failed
    # first); the repaired proposal did get checked
    assert len(validator.checked) == 1
    assert validator.checked[0]["generators"] == ["g_bool_interaction"]
    assert result.first_report.actor_visible["schema_problems"]


def test_repair_receives_actor_visible_report_only() -> None:
    proposer = FakeProposer(_valid(gens=()), repair=_valid())
    validator = FakeValidator({"g_bool_interaction"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, SCHEMA)
    inductor.run(dossier_ref="d", visible_evidence={})
    repair_input = proposer.repair_inputs[0]
    # exactly the actor-visible payload: validator_id + problems; nothing else
    assert repair_input == {"validator_id": "fake_v1", "problems": ["no_recognized_generator"]}


def test_repair_failure_retains_previous_contract() -> None:
    previous = {"generators": ["g_old"], "relations": []}
    proposer = FakeProposer(_valid(gens=()), repair=_valid(gens=()))
    validator = FakeValidator({"g_bool_interaction"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, SCHEMA)
    result = inductor.run(
        dossier_ref="d", visible_evidence={}, previous_contract=previous, failure_signal={"kind": "FIDELITY"}
    )
    assert result.outcome is InductionOutcome.RETAINED_PREVIOUS
    assert result.contract == previous
    assert result.call_budget.total_calls == 2
    assert result.repair_report is not None and not result.repair_report.passed


def test_repair_failure_without_previous_terminates() -> None:
    proposer = FakeProposer(_valid(gens=()), repair=_valid(gens=()))
    validator = FakeValidator({"g_bool_interaction"})
    inductor = RepresentationInductor(proposer, validator, PROMPT, SCHEMA)
    result = inductor.run(dossier_ref="d", visible_evidence={})
    assert result.outcome is InductionOutcome.TERMINATED_NO_CONTRACT
    assert result.contract is None


def test_schema_check_rules() -> None:
    problems = check_proposal_schema({"generators": [], "relations": [], "surprise": 1}, SCHEMA)
    assert any("unknown keys: surprise" in p for p in problems)
    problems = check_proposal_schema({"generators": [], "relations": True}, SCHEMA)
    assert any("relations" in p for p in problems)
    ok = check_proposal_schema({"generators": [], "relations": [], "notes": "n"}, SCHEMA)
    assert ok == []


# ------------------------------------------------------------- controller


def test_controller_score_formula_exact() -> None:
    w = ControllerWeights(lambda_s=0.4, lambda_u=0.3, lambda_c=0.2, lambda_o=0.1)
    ctrl = FrozenController(w)
    scores = ctrl.score({"z1": {"p_sat": 1.0, "utility": 0.5, "coverage_gap": 0.0, "open_gap": 0.2}})
    assert scores["z1"] == pytest.approx(0.4 * 1.0 + 0.3 * 0.5 + 0.2 * 0.0 + 0.1 * 0.2)


def test_controller_skips_unavailable_candidates() -> None:
    w = ControllerWeights(lambda_s=1.0, lambda_u=0.0, lambda_c=0.0, lambda_o=0.0)
    ctrl = FrozenController(w)
    scores = ctrl.score(
        {
            "z_ok": {"p_sat": 0.9, "utility": 0.1, "coverage_gap": 0.0, "open_gap": 0.0},
            "z_broken": {"p_sat": 0.9},  # missing terms -> unavailable
        }
    )
    assert "z_ok" in scores and "z_broken" not in scores
    decision = ctrl.select_batch(scores, 2, unavailable=["z_ok"])
    assert decision.batch == ()
    assert decision.skipped_unavailable == ("z_ok",)


def test_controller_batch_order_and_tie_break() -> None:
    w = ControllerWeights(lambda_s=1.0, lambda_u=0.0, lambda_c=0.0, lambda_o=0.0)
    ctrl = FrozenController(w)
    scores = ctrl.score(
        {
            "b2": {"p_sat": 0.9, "utility": 0, "coverage_gap": 0, "open_gap": 0},
            "a1": {"p_sat": 0.9, "utility": 0, "coverage_gap": 0, "open_gap": 0},
            "c3": {"p_sat": 0.5, "utility": 0, "coverage_gap": 0, "open_gap": 0},
        }
    )
    decision = ctrl.select_batch(scores, 2)
    assert decision.batch == ("a1", "b2")  # tie -> candidate id ascending
    assert "0.9" in decision.tie_breaks and decision.tie_breaks["0.9"] == ("a1", "b2")


def test_controller_diversity_rule_hook() -> None:
    w = ControllerWeights(lambda_s=1.0, lambda_u=0.0, lambda_c=0.0, lambda_o=0.0)
    spread = lambda ordered, scores: [ordered[-1], *ordered[:-1]]  # noqa: E731
    ctrl = FrozenController(w, diversity_rule=spread, diversity_rule_id="round_robin_v1")
    scores = ctrl.score(
        {
            "a1": {"p_sat": 0.9, "utility": 0, "coverage_gap": 0, "open_gap": 0},
            "b2": {"p_sat": 0.8, "utility": 0, "coverage_gap": 0, "open_gap": 0},
            "c3": {"p_sat": 0.7, "utility": 0, "coverage_gap": 0, "open_gap": 0},
        }
    )
    decision = ctrl.select_batch(scores, 2)
    assert decision.batch == ("c3", "a1")
    assert decision.diversity_rule_id == "round_robin_v1"


def test_controller_weight_validation() -> None:
    with pytest.raises(ValueError):
        FrozenController(ControllerWeights(lambda_s=True, lambda_u=0.0, lambda_c=0.0, lambda_o=0.0))
    with pytest.raises(ValueError):
        FrozenController(ControllerWeights(lambda_s=float("nan"), lambda_u=0.0, lambda_c=0.0, lambda_o=0.0))


def test_cases_state_uses_readout_only_policy() -> None:
    # Detail §3.4: CASES-State must NOT use the controller — the constant is the
    # machine-checkable marker campaigns assert on.
    assert READOUT_ONLY_POLICY == "cases_state_uses_frozen_readout_schema_no_controller"
