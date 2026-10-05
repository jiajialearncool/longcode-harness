from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from longcode.models import Subtask, TaskContract, TaskState
from longcode.process_adapter import JsonProcessAgentAdapter


def write_adapter(path: Path, *, invalid: bool = False) -> None:
    if invalid:
        body = "print('not-json')"
    else:
        body = textwrap.dedent(
            """\
            import argparse
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--role", required=True)
            args = parser.parse_args()
            request = json.load(sys.stdin)
            if args.role == "executor":
                Path("adapter-output.py").write_text("ADAPTER = True\\n")
                response = {"summary": "external adapter executed", "claimed_complete": True, "changed_files": ["adapter-output.py"], "tests_run": [], "remaining_risks": [], "fault": None}
            elif args.role == "manager":
                criterion = request["contract"]["acceptance_criteria"][0]["id"]
                verified = request["state"]["criteria"][criterion]["status"] == "verified"
                response = {
                    "action": "done" if verified else "execute",
                    "criterion_id": "" if verified else criterion,
                    "subtask_goal": "" if verified else "Create adapter output",
                    "executor_tier": "E2",
                    "risk_level": "medium",
                    "verification_profile": "default",
                    "required_capabilities": ["cli"],
                    "reason": "external manager decision",
                    "recipe_id": "",
                    "complexity": 2,
                    "uncertainty": 1,
                    "criticality": 1,
                    "verification_difficulty": 1
                }
            else:
                response = {"verdict": "pass", "summary": "external audit passed", "evidence": ["adapter-output.py"], "risks": []}
            print(json.dumps(response))
            """
        )
    path.write_text("#!/usr/bin/env python3\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


class ProcessAdapterTests(unittest.TestCase):
    def test_external_process_supports_all_agent_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            command = root / "adapter"
            write_adapter(command)
            adapter = JsonProcessAgentAdapter(str(command), capabilities=["browser"])
            contract = TaskContract.create(
                "Use external agent",
                ["Output exists"],
                checks=["test -f adapter-output.py"],
            )
            state = TaskState.create(contract)
            subtask = Subtask(
                id="ST-0001",
                round=1,
                criterion_id="AC-001",
                criterion="Output exists",
                objective=contract.objective,
                constraints=[],
                allowed_paths=["**"],
                forbidden_paths=[".git/**"],
                checks=contract.checks,
                attempt=1,
                max_attempts=2,
            )

            manager_result, decision = adapter.manage(workspace, contract, state)
            execution = adapter.execute(workspace, contract, subtask)
            audit_result, audit = adapter.audit(
                workspace,
                contract,
                state,
                subtask=subtask,
                changed_paths=["adapter-output.py"],
                checks=[],
                executor_report=execution.report,
            )

            self.assertTrue(manager_result.ok)
            self.assertEqual(decision.criterion_id, "AC-001")
            self.assertTrue(execution.ok)
            self.assertTrue(audit_result.ok)
            self.assertTrue(audit.passed)
            self.assertIn("browser", adapter.capabilities)
            self.assertTrue((workspace / "adapter-output.py").exists())

    def test_invalid_stdout_is_a_structured_backend_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            command = root / "invalid-adapter"
            write_adapter(command, invalid=True)
            adapter = JsonProcessAgentAdapter(str(command))
            contract = TaskContract.create("Goal", ["Done"], checks=["true"])
            subtask = Subtask(
                id="ST-0001", round=1, criterion_id="AC-001", criterion="Done",
                objective="Goal", constraints=[], allowed_paths=["**"],
                forbidden_paths=[".git/**"], checks=["true"], attempt=1, max_attempts=2,
            )
            result = adapter.execute(workspace, contract, subtask)
            self.assertFalse(result.ok)
            self.assertIn("did not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
