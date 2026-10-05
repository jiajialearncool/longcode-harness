from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from longcode.lh_benchmark_arm import (
    _lh_codex_usage,
    _write_codex_wrapper,
    run_lh_arm,
)


class LhBenchmarkArmTests(unittest.TestCase):
    def test_lh_codex_trajectories_are_aggregated_without_counting_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round_dir = root / "role_orchestration" / "rounds" / "round_001"
            round_dir.mkdir(parents=True)
            (round_dir / "manager_raw_trajectory.jsonl").write_text(
                json.dumps({"type": "turn.started"})
                + "\n"
                + json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 7,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # LH also writes episode copies. The aggregator intentionally reads
            # only the canonical round trajectories.
            duplicate = root / "manager_episodes" / "ep001"
            duplicate.mkdir(parents=True)
            (duplicate / "codex_stream.jsonl").write_text(
                (round_dir / "manager_raw_trajectory.jsonl").read_text(),
                encoding="utf-8",
            )

            usage = _lh_codex_usage(root)

            self.assertTrue(usage["token_metrics_available"])
            self.assertEqual(usage["model_calls"], 1)
            self.assertEqual(usage["input_tokens"], 100)
            self.assertEqual(usage["cached_input_tokens"], 40)
            self.assertEqual(usage["output_tokens"], 20)
            self.assertEqual(usage["reasoning_tokens"], 7)

    def test_lh_codex_wrapper_injects_matched_reasoning_without_copying_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "codex-real"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
            fake.chmod(0o755)

            wrapper = _write_codex_wrapper(
                root,
                environment={"LH_HARNESS_CODEX_BINARY": str(fake)},
                reasoning_effort="xhigh",
                ignore_user_config=True,
            )
            completed = __import__("subprocess").run(
                [str(wrapper), "exec", "--json", "-"],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("--ignore-user-config", completed.stdout)
            self.assertIn('model_reasoning_effort="xhigh"', completed.stdout)
            self.assertIn("--json", completed.stdout)
            self.assertFalse((root / "auth.json").exists())

    def _fake_lh(self, root: Path) -> Path:
        executable = root / "lh-harness"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            + textwrap.dedent(
                """\
                import json
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                def value(flag):
                    return args[args.index(flag) + 1]
                workspace = Path(value("--workspace"))
                log_dir = Path(value("--log-dir"))
                log_dir.mkdir(parents=True, exist_ok=True)
                (workspace / "solution.txt").write_text("done\\n")
                (log_dir / "report.json").write_text(json.dumps({
                    "status": "complete",
                    "completion_satisfied": True,
                    "rounds_run": 2,
                    "max_rounds": int(value("--max-rounds")),
                    "abort_reason": ""
                }))
                (log_dir / "args.json").write_text(json.dumps(args))
                print("fake LH completed")
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def test_official_cli_contract_is_wrapped_and_completion_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            result_file = root / "arm-result.json"
            executable = self._fake_lh(root)
            environment = {
                "HARNESS_WORKSPACE": str(workspace),
                "HARNESS_RESULT_FILE": str(result_file),
                "HARNESS_RUN_ID": "case#r1",
                "HARNESS_TASK_JSON": json.dumps(
                    {"goal": "write solution", "acceptance": ["solution exists"]}
                ),
                "HARNESS_BUDGET_JSON": json.dumps({"max_rounds": 4}),
            }

            exit_code = run_lh_arm(
                executable=str(executable),
                agent="codex",
                model="same-model",
                codex_mcp_config=str(root / "empty-mcp.toml"),
                environ=environment,
            )

            result = json.loads(result_file.read_text())
            args = json.loads((root / "lh-arm" / "logs" / "args.json").read_text())
            self.assertEqual(exit_code, 0)
            self.assertTrue(result["claimed_completed"])
            self.assertTrue(result["budget_enforced"])
            self.assertFalse(result["token_metrics_available"])
            self.assertEqual(result["lh"]["rounds_run"], 2)
            self.assertEqual(args[args.index("--max-rounds") + 1], "4")
            self.assertIn("--no-dashboard", args)
            self.assertIn("--codex-mcp-config", args)
            self.assertEqual((workspace / "solution.txt").read_text(), "done\n")

    def test_token_budget_without_broker_is_not_misreported_as_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            result_file = root / "arm-result.json"
            environment = {
                "HARNESS_WORKSPACE": str(workspace),
                "HARNESS_RESULT_FILE": str(result_file),
                "HARNESS_TASK_JSON": json.dumps({"goal": "write solution"}),
                "HARNESS_BUDGET_JSON": json.dumps(
                    {"max_rounds": 3, "max_output_tokens": 1000}
                ),
            }

            run_lh_arm(executable=str(self._fake_lh(root)), environ=environment)

            result = json.loads(result_file.read_text())
            self.assertFalse(result["budget_enforced"])
            self.assertEqual(result["budget_enforcement"]["unsupported_keys"], [])
            self.assertTrue(
                result["budget_enforcement"]["broker_required_but_missing"]
            )


if __name__ == "__main__":
    unittest.main()
