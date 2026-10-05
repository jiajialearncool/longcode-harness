from __future__ import annotations

import unittest

from longcode.models import Subtask, TaskContract, TaskState
from longcode.policy import PolicyEngine


class PolicyTests(unittest.TestCase):
    def _subtask(self, *, capabilities: list[str], criterion: str = "Implement feature") -> Subtask:
        return Subtask(
            id="ST-0001",
            round=1,
            criterion_id="AC-001",
            criterion=criterion,
            objective="Goal",
            constraints=[],
            allowed_paths=["**"],
            forbidden_paths=[".git/**"],
            checks=["true"],
            attempt=1,
            max_attempts=2,
            required_capabilities=capabilities,
        )

    def test_missing_browser_capability_requests_input_before_execution(self):
        contract = TaskContract.create("Goal", ["Flow works"], checks=["true"])
        decision = PolicyEngine().authorize(
            contract,
            TaskState.create(contract),
            self._subtask(capabilities=["cli", "browser"]),
            available_capabilities={"cli", "filesystem"},
        )
        self.assertEqual(decision.verdict, "ask")
        self.assertEqual(decision.missing_capabilities, ("browser",))

    def test_irreversible_action_needs_explicit_gate(self):
        contract = TaskContract.create("Goal", ["Deploy production"], checks=["true"])
        decision = PolicyEngine().authorize(
            contract,
            TaskState.create(contract),
            self._subtask(capabilities=["cli"], criterion="Deploy to production"),
            available_capabilities={"cli"},
        )
        self.assertEqual(decision.verdict, "ask")
        self.assertIn("explicit human gate", decision.reason)


if __name__ == "__main__":
    unittest.main()
