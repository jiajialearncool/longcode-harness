from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from longcode.models import ManagerDecision, TaskContract, TaskState
from longcode.routing import (
    AgentAdaptiveManager,
    EvidenceDrivenManager,
    HeuristicAdaptiveManager,
    capabilities_for,
)


class ManagerBackend:
    def __init__(self, decision: ManagerDecision):
        self.decision = decision
        self.calls = 0

    def manage(self, workspace, contract, state):
        self.calls += 1
        class Result:
            ok = True

        return Result(), self.decision


class RoutingTests(unittest.TestCase):
    def test_static_page_semantic_review_does_not_require_live_browser_or_gui(self):
        capabilities = capabilities_for(
            "构建 onboarding 页面，并适配手机和桌面宽度",
            "semantic",
        )
        self.assertEqual(capabilities, ["cli"])
        self.assertIn(
            "browser",
            capabilities_for("构建 onboarding 页面", "product_flow"),
        )

    def test_evidence_manager_calls_model_once_per_new_replan(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = TaskContract.create("Goal", ["Fix it"], checks=["true"])
            state = TaskState.create(contract)
            proposed = ManagerDecision(
                action="execute",
                criterion_id="AC-001",
                subtask_goal="Fix it with a revised approach",
                executor_tier="E2",
                risk_level="medium",
                verification_profile="default",
                required_capabilities=["cli"],
                reason="replanned after repeated evidence",
            )
            backend = ManagerBackend(proposed)
            manager = EvidenceDrivenManager(backend, Path(directory))

            manager.decide(contract, state)
            self.assertEqual(backend.calls, 0)
            state.criteria["AC-001"].replan_count = 1
            manager.decide(contract, state)
            manager.decide(contract, state)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(state.criteria["AC-001"].manager_replans_handled, 1)
    def test_explicit_low_risk_recipe_makes_e0_reachable(self):
        contract = TaskContract.create(
            "Generate file",
            ["Generated metadata exists"],
            checks=["test -f metadata.json"],
            deterministic_recipes={"AC-001": ["printf '{}\\n' > metadata.json"]},
        )
        contract.acceptance_criteria[0] = replace(
            contract.acceptance_criteria[0],
            risk_level="low",
            recipe_id="AC-001",
        )
        decision = HeuristicAdaptiveManager().decide(contract, TaskState.create(contract))
        self.assertEqual(decision.executor_tier, "E0")
        self.assertEqual(decision.recipe_id, "AC-001")

    def test_agent_cannot_invent_e0_without_declared_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = TaskContract.create("Goal", ["Simple"], checks=["true"])
            state = TaskState.create(contract)
            proposed = ManagerDecision(
                action="execute",
                criterion_id="AC-001",
                subtask_goal="Simple",
                executor_tier="E0",
                risk_level="low",
                verification_profile="default",
                required_capabilities=["cli"],
                reason="claims deterministic",
                recipe_id="invented",
            )
            decision = AgentAdaptiveManager(
                ManagerBackend(proposed), Path(directory)
            ).decide(contract, state)
            self.assertEqual(decision.executor_tier, "E2")
            self.assertIsNone(decision.recipe_id)

    def test_heuristic_manager_enforces_high_risk_expert_tier(self):
        contract = TaskContract.create(
            "Safely deploy to production",
            ["Deploy the production database migration"],
            checks=["true"],
        )
        contract.acceptance_criteria[0] = replace(
            contract.acceptance_criteria[0], risk_level="high"
        )
        state = TaskState.create(contract)
        decision = HeuristicAdaptiveManager().decide(contract, state)
        self.assertEqual(decision.action, "execute")
        self.assertEqual(decision.executor_tier, "E3")
        self.assertEqual(decision.risk_level, "high")

    def test_agent_manager_cannot_select_non_ready_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = TaskContract.create("Goal", ["First", "Second"], checks=["true"])
            state = TaskState.create(contract)
            proposed = ManagerDecision(
                action="execute",
                criterion_id="AC-999",
                subtask_goal="Wrong",
                executor_tier="E1",
                risk_level="low",
                verification_profile="default",
                required_capabilities=["cli"],
                reason="bad proposal",
            )
            manager = AgentAdaptiveManager(ManagerBackend(proposed), Path(directory))
            decision = manager.decide(contract, state)
            self.assertEqual(decision.criterion_id, "AC-001")
            self.assertIsNotNone(manager.last_fallback_reason)

    def test_agent_manager_risk_floor_overrides_underpowered_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = TaskContract.create("Goal", ["Critical change"], checks=["true"])
            state = TaskState.create(contract)
            proposed = ManagerDecision(
                action="execute",
                criterion_id="AC-001",
                subtask_goal="Critical change",
                executor_tier="E1",
                risk_level="high",
                verification_profile="security",
                required_capabilities=["cli"],
                reason="underestimated tier",
                complexity=2,
                uncertainty=2,
                criticality=3,
                verification_difficulty=3,
            )
            manager = AgentAdaptiveManager(ManagerBackend(proposed), Path(directory))
            decision = manager.decide(contract, state)
            self.assertEqual(decision.executor_tier, "E3")

    def test_agent_manager_cannot_downgrade_contract_verifier_or_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = TaskContract.create(
                "Build product",
                ["User can open the dashboard page"],
                checks=["true"],
            )
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0],
                verification_profile="product_flow",
            )
            proposed = ManagerDecision(
                action="execute",
                criterion_id="AC-001",
                subtask_goal="Implement dashboard",
                executor_tier="E2",
                risk_level="medium",
                verification_profile="default",
                required_capabilities=["cli"],
                reason="attempted downgrade",
            )
            decision = AgentAdaptiveManager(
                ManagerBackend(proposed), Path(directory)
            ).decide(contract, TaskState.create(contract))
            self.assertEqual(decision.verification_profile, "product_flow")
            self.assertIn("browser", decision.required_capabilities)


if __name__ == "__main__":
    unittest.main()
