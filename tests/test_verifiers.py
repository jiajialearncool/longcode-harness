from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.models import Subtask, TaskContract, TaskState, VerifierResult
from longcode.verifiers import ProductFlowCommandVerifier, VerificationContext, VerifierMesh
from tests.helpers import FakeBackend


class ProductFlowPlugin:
    verifier_id = "fake-browser-flow"
    kind = "browser"
    profiles = {"product_flow"}

    def __init__(self, verdict: str = "pass"):
        self.verdict = verdict

    def verify(self, context: VerificationContext) -> VerifierResult:
        return VerifierResult(
            verifier_id=self.verifier_id,
            kind=self.kind,
            verdict=self.verdict,
            required=True,
            summary="hidden user flow executed",
            evidence=["browser-trace.zip"],
            fault_code=None if self.verdict == "pass" else "product_flow_failed",
        )


class VerifierMeshTests(unittest.TestCase):
    def test_default_profile_skips_semantic_auditor_when_checks_are_conclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = TaskContract.create("Goal", ["Done"], checks=["true"])
            state = TaskState.create(contract)
            auditor = FakeBackend()
            outcome = VerifierMesh(auditor=auditor).verify(
                root,
                contract,
                state,
                subtask=None,
                changed_paths=[],
                executor_report=None,
            )
            self.assertTrue(outcome.passed)
            self.assertFalse(outcome.auditor_invoked)
            self.assertEqual(auditor.audits, [])

    def test_semantic_profile_invokes_auditor_after_deterministic_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = TaskContract.create("Goal", ["Looks correct"], checks=["true"])
            contract.default_verification_profile = "semantic"
            state = TaskState.create(contract)
            auditor = FakeBackend()
            outcome = VerifierMesh(auditor=auditor).verify(
                root,
                contract,
                state,
                subtask=None,
                changed_paths=[],
                executor_report=None,
            )
            self.assertTrue(outcome.passed)
            self.assertTrue(outcome.auditor_invoked)
            self.assertEqual(len(auditor.audits), 1)

    def _context(self, root: Path):
        contract = TaskContract.create("Build flow", ["User can finish flow"], checks=["true"])
        state = TaskState.create(contract)
        subtask = Subtask(
            id="ST-0001",
            round=1,
            criterion_id="AC-001",
            criterion="User can finish flow",
            objective=contract.objective,
            constraints=[],
            allowed_paths=["**"],
            forbidden_paths=[".git/**"],
            checks=["true"],
            attempt=1,
            max_attempts=2,
            verification_profile="product_flow",
        )
        return contract, state, subtask

    def test_product_claim_fails_closed_without_environment_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, state, subtask = self._context(root)
            outcome = VerifierMesh().verify(
                root,
                contract,
                state,
                subtask=subtask,
                changed_paths=[],
                executor_report={},
            )
            self.assertEqual(outcome.verdict, "uncertain")
            self.assertIn("product_verifier_missing", [item.fault_code for item in outcome.results])

    def test_product_claim_passes_with_required_browser_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, state, subtask = self._context(root)
            outcome = VerifierMesh(plugins=[ProductFlowPlugin()]).verify(
                root,
                contract,
                state,
                subtask=subtask,
                changed_paths=[],
                executor_report={},
            )
            self.assertTrue(outcome.passed)
            self.assertIn("browser", [item.kind for item in outcome.results])

    def test_explicit_product_command_is_environment_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, state, subtask = self._context(root)
            outcome = VerifierMesh(
                plugins=[ProductFlowCommandVerifier(["test -d ."])]
            ).verify(
                root,
                contract,
                state,
                subtask=subtask,
                changed_paths=[],
                executor_report={},
            )
            self.assertTrue(outcome.passed)
            result = next(
                item for item in outcome.results if item.verifier_id == "product-flow-command"
            )
            self.assertEqual(result.kind, "browser")
            self.assertTrue(result.evidence)


if __name__ == "__main__":
    unittest.main()
