from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from longcode.backends import EXECUTOR_SCHEMA
from longcode.claude_backend import ClaudeCodeBackend, _claude_structured_output, _claude_usage
from longcode.models import Subtask, TaskContract


class ClaudeCodeBackendTests(unittest.TestCase):
    def test_extracts_structured_output_and_usage(self):
        report = {
            "summary": "done",
            "claimed_complete": False,
            "changed_files": ["answer.txt"],
            "tests_run": ["check"],
            "remaining_risks": [],
            "fault": None,
        }
        outer = {
            "type": "result",
            "structured_output": report,
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
            },
        }
        self.assertEqual(_claude_structured_output(outer), report)
        self.assertEqual(
            _claude_usage(outer),
            {
                "input_tokens": 15,
                "cached_input_tokens": 5,
                "output_tokens": 7,
                "reasoning_tokens": 0,
                "token_metrics_available": True,
            },
        )

    def test_executor_accepts_claude_json_envelope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "claude"
            report = {
                "summary": "changed one file",
                "claimed_complete": False,
                "changed_files": [],
                "tests_run": [],
                "remaining_risks": [],
                "fault": None,
            }
            payload = {
                "type": "result",
                "structured_output": report,
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
            fake.write_text(
                "#!/bin/sh\nprintf '%s\\n' " + repr(json.dumps(payload)) + "\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            contract = TaskContract.create("do work", ["work is correct"], checks=["true"])
            criterion = contract.acceptance_criteria[0]
            subtask = Subtask(
                id="ST-1",
                round=1,
                criterion_id=criterion.id,
                criterion=criterion.description,
                objective=contract.objective,
                constraints=list(contract.constraints),
                allowed_paths=list(contract.allowed_paths),
                forbidden_paths=list(contract.forbidden_paths),
                checks=list(contract.checks),
                attempt=1,
                max_attempts=1,
            )
            result = ClaudeCodeBackend(executable=str(fake)).execute(root, contract, subtask)
            self.assertTrue(result.ok)
            self.assertEqual(result.report, report)
            self.assertEqual(result.input_tokens, 4)
            self.assertEqual(result.output_tokens, 2)
            self.assertEqual(ClaudeCodeBackend(executable=str(fake)).usage_totals["model_calls"], 0)
            self.assertEqual(set(result.report), set(EXECUTOR_SCHEMA["required"]))


if __name__ == "__main__":
    unittest.main()
