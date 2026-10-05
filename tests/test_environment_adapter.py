from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from longcode.engine import LongCodeEngine
from longcode.environment_adapter import JsonProcessEnvironmentAdapter
from longcode.models import TaskContract
from longcode.product import build_product_spec, product_acceptance
from longcode.storage import RuntimeStore
from longcode.verifiers import EnvironmentEvidenceVerifier

from tests.helpers import FakeBackend


def write_environment_controller(
    path: Path,
    *,
    fail_commit: bool = False,
    fail_commit_once: bool = False,
    fail_inspect_once: bool = False,
    reconcile_status: str = "unknown",
) -> Path:
    log_path = path.with_suffix(".log")
    path.write_text(
        "#!/usr/bin/env python3\n"
        + textwrap.dedent(
            f"""\
            import argparse
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--operation", required=True)
            args = parser.parse_args()
            request = json.load(sys.stdin)
            log = Path({str(log_path)!r})
            previous = log.read_text().splitlines() if log.exists() else []
            operation_count = previous.count(args.operation) + 1
            with log.open("a") as handle:
                handle.write(args.operation + "\\n")
            if args.operation == "begin":
                response = {{
                    "ok": True,
                    "session_id": "browser-session-001",
                    "context": {{"endpoint": "ws://controller/session", "access_token": "secret-value"}},
                    "evidence": [],
                    "summary": "isolated browser context created"
                }}
            elif args.operation == "inspect" and {fail_inspect_once!r} and operation_count == 1:
                response = {{"ok": False, "summary": "simulated inspect failure"}}
            elif args.operation == "inspect":
                response = {{
                    "ok": True,
                    "context": {{"page": "dashboard"}},
                    "evidence": [{{"kind": "browser", "verdict": "pass", "path": "traces/flow.zip", "summary": "critical flow passed"}}],
                    "summary": "browser flow inspected"
                }}
            elif args.operation == "commit" and ({fail_commit!r} or ({fail_commit_once!r} and operation_count == 1)):
                response = {{"ok": False, "summary": "simulated external commit failure"}}
            elif args.operation == "reconcile":
                response = {{
                    "ok": True,
                    "status": {reconcile_status!r},
                    "summary": "durable controller lookup"
                }}
            else:
                response = {{"ok": True, "summary": args.operation + " ok"}}
            print(json.dumps(response))
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return log_path


class EnvironmentAdapterTests(unittest.TestCase):
    def test_external_reconciliation_queries_controller_without_begin_or_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            controller = root / "controller"
            log_path = write_environment_controller(
                controller, reconcile_status="committed"
            )
            adapter = JsonProcessEnvironmentAdapter(
                str(controller), root / "runtime", capabilities=["browser"]
            )

            result = adapter.reconcile(
                {
                    "candidate_id": "CAND-interrupted",
                    "changed_paths": ["app.py"],
                    "environment_context": {"session_id": "browser-session-001"},
                },
                workspace,
            )

            self.assertEqual(result["status"], "committed")
            self.assertEqual(log_path.read_text().splitlines(), ["reconcile"])

    def test_external_environment_uses_two_phase_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("OLD = True\n", encoding="utf-8")
            controller = root / "controller"
            log_path = write_environment_controller(controller)
            adapter = JsonProcessEnvironmentAdapter(
                str(controller), root / "runtime", capabilities=["browser"]
            )
            transaction = adapter.begin(workspace, round_number=1)
            self.assertEqual(transaction.context["session_id"], "browser-session-001")
            (transaction.candidate_workspace / "app.py").write_text(
                "NEW = True\n", encoding="utf-8"
            )
            paths = transaction.diff()
            self.assertEqual(transaction.context["evidence"][0]["kind"], "browser")
            result = transaction.promote(paths)
            transaction.close()

            self.assertTrue(result.passed)
            self.assertEqual((workspace / "app.py").read_text(), "NEW = True\n")
            self.assertEqual(
                log_path.read_text().splitlines(),
                ["begin", "inspect", "prepare_commit", "commit", "close"],
            )

    def test_external_commit_failure_compensates_local_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "app.py"
            target.write_text("ORIGINAL = True\n", encoding="utf-8")
            controller = root / "controller"
            write_environment_controller(controller, fail_commit=True)
            transaction = JsonProcessEnvironmentAdapter(
                str(controller), root / "runtime", capabilities=["browser"]
            ).begin(workspace, round_number=1)
            (transaction.candidate_workspace / "app.py").write_text(
                "UNCOMMITTED = True\n", encoding="utf-8"
            )
            result = transaction.promote(transaction.diff())

            self.assertFalse(result.passed)
            self.assertTrue(result.restored_after_failure)
            self.assertEqual(target.read_text(), "ORIGINAL = True\n")

    def test_product_engine_consumes_environment_evidence_without_persisting_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            controller = root / "controller"
            write_environment_controller(controller)
            product = build_product_spec(
                problem="Show a dashboard",
                target_users=["manager"],
                key_flows=["open the dashboard page"],
            )
            criteria = product_acceptance(product)
            contract = TaskContract.create(
                "Build dashboard",
                [item.description for item in criteria],
                checks=["test -f app.py"],
                allowed_paths=["app.py"],
                mode="product",
                product=product,
            )
            contract.acceptance_criteria = criteria
            contract.default_verification_profile = "product_flow"
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)

            def execute(workspace, contract, subtask):
                self.assertEqual(
                    subtask.environment_context["access_token"], "secret-value"
                )
                (workspace / "app.py").write_text("DASHBOARD = True\n", encoding="utf-8")

            backend = FakeBackend(execute)
            state = LongCodeEngine(
                store,
                backend,
                auditor=None,
                verifier_plugins=[EnvironmentEvidenceVerifier("browser")],
                transaction_manager=JsonProcessEnvironmentAdapter(
                    str(controller), store.root, capabilities=["browser"]
                ),
            ).run(max_rounds=2)

            self.assertEqual(state.status, "completed")
            self.assertEqual(state.environment_evidence[0]["verdict"], "pass")
            serialized_events = json.dumps(store.iter_events())
            serialized_state = json.dumps(state.to_dict())
            self.assertNotIn("secret-value", serialized_events)
            self.assertNotIn("secret-value", serialized_state)
            self.assertIn("<redacted>", serialized_events)

    def test_environment_commit_fault_recovers_same_tier_without_losing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "app.py"
            target.write_text("STATE = 'original'\n", encoding="utf-8")
            controller = root / "controller"
            write_environment_controller(controller, fail_commit_once=True)
            contract = TaskContract.create(
                "Update app safely",
                ["App contains committed state"],
                checks=["grep -q committed app.py"],
                allowed_paths=["app.py"],
                max_attempts=2,
            )
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)
            calls = 0

            def execute(candidate, contract, subtask):
                nonlocal calls
                calls += 1
                if calls == 2:
                    self.assertEqual(target.read_text(), "STATE = 'original'\n")
                (candidate / "app.py").write_text(
                    "STATE = 'committed'\n", encoding="utf-8"
                )

            backend = FakeBackend(execute)
            state = LongCodeEngine(
                store,
                backend,
                auditor=None,
                transaction_manager=JsonProcessEnvironmentAdapter(
                    str(controller), store.root, capabilities=["browser"]
                ),
            ).run(max_rounds=2)

            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E2", "E2"])
            self.assertEqual(target.read_text(), "STATE = 'committed'\n")
            self.assertNotIn(
                "executor_tier_escalated",
                [item["type"] for item in store.iter_events()],
            )
            fault = store.iter_faults()[0]
            self.assertEqual(fault["category"], "environment")

    def test_environment_inspect_fault_restarts_candidate_without_model_escalation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            controller = root / "controller"
            write_environment_controller(controller, fail_inspect_once=True)
            contract = TaskContract.create(
                "Create app",
                ["App exists"],
                checks=["test -f app.py"],
                allowed_paths=["app.py"],
                max_attempts=2,
            )
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)

            def execute(candidate, contract, subtask):
                (candidate / "app.py").write_text("READY = True\n", encoding="utf-8")

            backend = FakeBackend(execute)
            state = LongCodeEngine(
                store,
                backend,
                auditor=None,
                transaction_manager=JsonProcessEnvironmentAdapter(
                    str(controller), store.root, capabilities=["browser"]
                ),
            ).run(max_rounds=2)

            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E2", "E2"])
            self.assertEqual(store.iter_faults()[0]["code"], "environment_inspect_failed")


if __name__ == "__main__":
    unittest.main()
