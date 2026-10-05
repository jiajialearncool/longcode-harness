from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from longcode.native_benchmark_arms import (
    _contract_from_task,
    run_direct_arm,
    run_longcode_arm,
)
from longcode.benchmark_runner import run_benchmark


class NativeBenchmarkArmTests(unittest.TestCase):
    def test_matched_task_acceptance_is_one_atomic_execution_unit(self):
        contract = _contract_from_task(
            {
                "goal": "Complete the whole task",
                "acceptance": ["first condition", "second condition"],
                "checks": ["true"],
            },
            {},
        )
        self.assertEqual(len(contract.acceptance_criteria), 1)
        description = contract.acceptance_criteria[0].description
        self.assertIn("first condition", description)
        self.assertIn("second condition", description)

    def _fake_codex(self, root: Path) -> Path:
        executable = root / "fake-codex"
        executable.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                workspace = Path(args[args.index("--cd") + 1])
                output = Path(args[args.index("--output-last-message") + 1])
                schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
                properties = schema["properties"]
                if "action" in properties:
                    if (workspace / "solution.txt").exists():
                        report = {
                            "action": "done", "criterion_id": "", "subtask_goal": "",
                            "executor_tier": "E2", "risk_level": "medium",
                            "verification_profile": "default", "required_capabilities": [],
                            "reason": "all acceptance criteria are verified", "recipe_id": "",
                            "complexity": 0, "uncertainty": 0, "criticality": 0,
                            "verification_difficulty": 0
                        }
                    else:
                        report = {
                            "action": "execute", "criterion_id": "AC-001",
                            "subtask_goal": "create the required solution", "executor_tier": "E2",
                            "risk_level": "medium", "verification_profile": "default",
                            "required_capabilities": ["cli", "filesystem"],
                            "reason": "the acceptance criterion is still pending", "recipe_id": "",
                            "complexity": 1, "uncertainty": 1, "criticality": 1,
                            "verification_difficulty": 1
                        }
                elif "verdict" in properties:
                    report = {
                        "verdict": "pass", "summary": "verified from the workspace",
                        "evidence": ["solution.txt"], "risks": []
                    }
                else:
                    (workspace / "solution.txt").write_text("done\\n")
                    report = {
                        "summary": "created solution", "claimed_complete": True,
                        "changed_files": ["solution.txt"], "tests_run": [],
                        "remaining_risks": [], "fault": None
                    }
                output.write_text(json.dumps(report))
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def _fake_lh(self, root: Path) -> Path:
        executable = root / "fake-lh"
        executable.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
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
                    "status": "complete", "completion_satisfied": True,
                    "rounds_run": 1, "max_rounds": int(value("--max-rounds")),
                    "abort_reason": ""
                }))
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    @staticmethod
    def _environment(root: Path, workspace: Path) -> dict[str, str]:
        return {
            "HARNESS_WORKSPACE": str(workspace),
            "HARNESS_RESULT_FILE": str(root / "arm-result.json"),
            "HARNESS_TASK_JSON": json.dumps(
                {
                    "goal": "Create solution.txt",
                    "acceptance": ["solution.txt exists"],
                    "checks": ["test -f solution.txt"],
                    "allowed_paths": ["solution.txt"],
                }
            ),
            "HARNESS_BUDGET_JSON": "{}",
        }

    def test_direct_arm_runs_one_codex_episode_and_records_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            environment = self._environment(root, workspace)

            exit_code = run_direct_arm(
                codex=str(self._fake_codex(root)),
                model="same-model",
                environ=environment,
            )

            result = json.loads((root / "arm-result.json").read_text())
            self.assertEqual(exit_code, 0)
            self.assertTrue(result["claimed_completed"])
            self.assertTrue(result["budget_enforced"])
            self.assertFalse(result["token_metrics_available"])
            self.assertEqual((workspace / "solution.txt").read_text(), "done\n")

    def test_longcode_arm_runs_manager_executor_auditor_to_verified_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            environment = self._environment(root, workspace)

            exit_code = run_longcode_arm(
                codex=str(self._fake_codex(root)),
                model="same-model",
                max_rounds=2,
                environ=environment,
            )

            result = json.loads((root / "arm-result.json").read_text())
            self.assertEqual(exit_code, 0)
            self.assertTrue(result["claimed_completed"])
            self.assertEqual(result["longcode"]["status"], "completed")
            self.assertTrue(result["longcode"]["final_evidence"])
            self.assertTrue((root / "longcode-arm" / "runtime" / "state.json").exists())
            self.assertFalse((workspace / ".longcode").exists())

    def test_eval_run_executes_all_builtin_arms_and_preserves_internal_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            hidden = root / "hidden"
            hidden.mkdir()
            (hidden / "verify.py").write_text(
                "from pathlib import Path\n"
                "raise SystemExit(0 if Path('solution.txt').is_file() else 1)\n",
                encoding="utf-8",
            )
            fake_codex = self._fake_codex(root)
            fake_lh = self._fake_lh(root)
            launcher = Path(__file__).resolve().parents[1] / "longcode_cli.py"
            manifest = root / "manifest.json"
            common_task = {
                "goal": "Create solution.txt",
                "acceptance": ["solution.txt exists"],
                "checks": ["test -f solution.txt"],
                "allowed_paths": ["solution.txt"],
            }
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "repetitions": 1,
                        "timeout_seconds": 20,
                        "hidden_check_timeout_seconds": 5,
                        "budget": {},
                        "arms": {
                            "direct": {
                                "command": [
                                    sys.executable, str(launcher), "eval-arm-direct",
                                    "--codex", str(fake_codex), "--model", "same-model",
                                ]
                            },
                            "lh": {
                                "command": [
                                    sys.executable, str(launcher), "eval-arm-lh",
                                    "--lh-harness", str(fake_lh), "--agent", "codex",
                                    "--model", "same-model",
                                ]
                            },
                            "longcode": {
                                "command": [
                                    sys.executable, str(launcher), "eval-arm-longcode",
                                    "--codex", str(fake_codex), "--model", "same-model",
                                    "--max-rounds", "2",
                                ]
                            },
                        },
                        "cases": [
                            {
                                "id": "builtin-arms-01",
                                "suite": "coding",
                                "source": str(source),
                                "hidden_root": str(hidden),
                                "task": common_task,
                                "hidden_checks": ["python3 {HIDDEN_ROOT}/verify.py"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "results.jsonl"
            preserved = root / "preserved"

            records = run_benchmark(
                manifest,
                output,
                seed=3,
                preserve_runs=preserved,
            )

            self.assertEqual({item.arm for item in records}, {"direct", "lh", "longcode"})
            self.assertTrue(all(item.success for item in records))
            self.assertTrue(all(item.budget_enforced for item in records))
            self.assertFalse(
                (
                    preserved
                    / "builtin-arms-01-r1"
                    / "direct"
                    / "workspace"
                    / "verify.py"
                ).exists()
            )
            self.assertTrue(
                (
                    preserved
                    / "builtin-arms-01-r1"
                    / "lh"
                    / "lh-arm"
                    / "logs"
                    / "report.json"
                ).exists()
            )
            self.assertTrue(
                (
                    preserved
                    / "builtin-arms-01-r1"
                    / "longcode"
                    / "longcode-arm"
                    / "runtime"
                    / "state.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
