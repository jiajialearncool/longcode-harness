from __future__ import annotations

import json
import unittest
import tempfile
import textwrap
from pathlib import Path

from longcode.backends import (
    AUDITOR_SCHEMA,
    CodexBackend,
    EXECUTOR_SCHEMA,
    _codex_jsonl_usage,
    _matches_schema,
    _parse_json_object,
)
from longcode.models import Subtask, TaskContract, TaskState


class BackendTests(unittest.TestCase):
    def test_current_codex_cli_top_level_usage_fields_are_recorded(self):
        usage = _codex_jsonl_usage(
            json.dumps({"type": "turn.started"})
            + "\n"
            + json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 12848,
                        "cached_input_tokens": 9984,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 3,
                    },
                }
            )
        )

        self.assertTrue(usage["token_metrics_available"])
        self.assertEqual(usage["input_tokens"], 12848)
        self.assertEqual(usage["cached_input_tokens"], 9984)
        self.assertEqual(usage["output_tokens"], 5)
        self.assertEqual(usage["reasoning_tokens"], 3)

    def test_codex_backend_uses_fresh_sandboxed_subprocesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            codex_home = root / "codex-home"
            workspace.mkdir()
            codex_home.mkdir()
            fake_codex = root / "fake-codex"
            fake_codex.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    sandbox = args[args.index("--sandbox") + 1]
                    output = Path(args[args.index("--output-last-message") + 1])
                    if sandbox == "read-only":
                        report = {"verdict": "pass", "summary": "audited", "evidence": ["repo"], "risks": []}
                    else:
                        report = {"summary": "executed", "claimed_complete": True, "changed_files": [], "tests_run": [], "remaining_risks": [], "fault": None}
                    output.write_text(json.dumps(report))
                    print(json.dumps({"sandbox": sandbox, "codex_home": os.environ.get("CODEX_HOME")}))
                    print(json.dumps({
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 40,
                            "input_tokens_details": {"cached_tokens": 5},
                            "output_tokens": 10,
                            "output_tokens_details": {"reasoning_tokens": 3}
                        }
                    }))
                    """
                )
            )
            fake_codex.chmod(0o755)
            contract = TaskContract.create("Durable objective", ["Works"], checks=["true"])
            state = TaskState.create(contract)
            subtask = Subtask(
                id="ST-0001",
                round=1,
                criterion_id="AC-001",
                criterion="Works",
                objective=contract.objective,
                constraints=[],
                allowed_paths=["**"],
                forbidden_paths=[".git/**"],
                checks=["true"],
                attempt=1,
                max_attempts=2,
            )
            backend = CodexBackend(executable=str(fake_codex), codex_home=codex_home)
            executed = backend.execute(workspace, contract, subtask)
            audited_backend, audit = backend.audit(
                workspace,
                contract,
                state,
                subtask=subtask,
                changed_paths=[],
                checks=[],
                executor_report=executed.report,
            )
            self.assertTrue(executed.ok)
            self.assertTrue(audited_backend.ok)
            self.assertTrue(audit.passed)
            self.assertIn('"sandbox": "workspace-write"', executed.stdout)
            self.assertIn('"sandbox": "read-only"', audited_backend.stdout)
            self.assertIn(str(codex_home), executed.stdout)
            self.assertTrue(executed.token_metrics_available)
            self.assertEqual(executed.input_tokens, 40)
            self.assertEqual(executed.cached_input_tokens, 5)
            self.assertEqual(executed.output_tokens, 10)
            self.assertEqual(executed.reasoning_tokens, 3)
            self.assertEqual(backend.usage_totals["model_calls"], 2)
            self.assertEqual(backend.usage_totals["input_tokens"], 80)

    def test_parse_json_object_accepts_plain_and_fenced_text(self):
        self.assertEqual(_parse_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(_parse_json_object('report:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_parse_json_object("not json"), {})

    def test_executor_report_must_match_schema(self):
        valid = {
            "summary": "done",
            "claimed_complete": True,
            "changed_files": ["a.py"],
            "tests_run": ["unit"],
            "remaining_risks": [],
            "fault": None,
        }
        self.assertTrue(_matches_schema(valid, EXECUTOR_SCHEMA))
        self.assertFalse(_matches_schema({"summary": "done"}, EXECUTOR_SCHEMA))
        self.assertFalse(_matches_schema({**valid, "extra": "no"}, EXECUTOR_SCHEMA))
        self.assertFalse(_matches_schema({**valid, "claimed_complete": "yes"}, EXECUTOR_SCHEMA))

    def test_auditor_verdict_is_constrained(self):
        valid = {
            "verdict": "pass",
            "summary": "verified",
            "evidence": ["test output"],
            "risks": [],
        }
        self.assertTrue(_matches_schema(valid, AUDITOR_SCHEMA))
        self.assertFalse(_matches_schema({**valid, "verdict": "maybe"}, AUDITOR_SCHEMA))


if __name__ == "__main__":
    unittest.main()
