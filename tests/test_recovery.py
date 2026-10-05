from __future__ import annotations

import unittest

from longcode.models import CriterionState, Subtask
from longcode.recovery import FaultClassifier, SafetyBelt, register_fault


def subtask() -> Subtask:
    return Subtask(
        id="ST-0001",
        round=1,
        criterion_id="AC-001",
        criterion="Works",
        objective="Goal",
        constraints=[],
        allowed_paths=["**"],
        forbidden_paths=[".git/**"],
        checks=["true"],
        attempt=1,
        max_attempts=2,
        executor_tier="E1",
    )


class RecoveryTests(unittest.TestCase):
    def test_same_verified_failure_escalates_only_after_repetition(self):
        classifier = FaultClassifier()
        fault = classifier.check_failure(subtask(), ["pytest"], ["evidence.json"])
        state = CriterionState()
        belt = SafetyBelt(escalate_after_same_failure=2)
        first = belt.decide(
            fault,
            state,
            current_tier="E1",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=True,
            recovery_budget_remaining=True,
        )
        self.assertEqual(first.action, "retry")
        register_fault(state, fault, first)
        second = belt.decide(
            fault,
            state,
            current_tier="E1",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=False,
            recovery_budget_remaining=True,
        )
        self.assertEqual(second.action, "escalate")
        self.assertEqual(second.next_tier, "E2")

    def test_transient_environment_failure_does_not_escalate_model(self):
        fault = FaultClassifier().environment_failure(
            code="filesystem_race", summary="directory changed during snapshot", subtask=subtask()
        )
        decision = SafetyBelt().decide(
            fault,
            CriterionState(),
            current_tier="E2",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=True,
            recovery_budget_remaining=True,
        )
        self.assertEqual(decision.action, "retry")
        self.assertIsNone(decision.next_tier)

    def test_policy_violation_cannot_be_bypassed_by_escalation(self):
        fault = FaultClassifier().scope_violation(subtask(), ["forbidden:secrets"], [])
        decision = SafetyBelt().decide(
            fault,
            CriterionState(),
            current_tier="E1",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=True,
            recovery_budget_remaining=True,
        )
        self.assertEqual(decision.action, "rollback")

    def test_repeated_failure_at_strongest_tier_requests_replan(self):
        classifier = FaultClassifier()
        fault = classifier.check_failure(subtask(), ["pytest"], ["evidence.json"])
        state = CriterionState(fault_fingerprints=[fault.fingerprint])
        decision = SafetyBelt(escalate_after_same_failure=2).decide(
            fault,
            state,
            current_tier="E3",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=False,
            recovery_budget_remaining=True,
        )
        self.assertEqual(decision.action, "replan")

    def test_reported_goal_fault_requests_user_input(self):
        report = {
            "fault": {
                "category": "goal",
                "code": "ambiguous_requirement",
                "summary": "Two incompatible deletion semantics remain",
                "retryable": False,
                "scope": "run",
                "severity": "error",
            }
        }
        fault = FaultClassifier().reported_failure(
            report,
            subtask=subtask(),
            evidence=["executor.json"],
        )
        decision = SafetyBelt().decide(
            fault,
            CriterionState(),
            current_tier="E2",
            allowed_tiers=["E1", "E2", "E3"],
            attempts_remaining=True,
            recovery_budget_remaining=True,
        )
        self.assertEqual(fault.category, "goal")
        self.assertEqual(decision.action, "ask")


if __name__ == "__main__":
    unittest.main()
