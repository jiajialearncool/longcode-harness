from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from longcode.benchmark_runner import (
    _is_loopback_provider_url,
    _run_arm_process,
    _validate_manifest,
    run_benchmark,
)
from longcode.evaluation import comparison_report, load_records


class BenchmarkRunnerTests(unittest.TestCase):
    def test_external_hidden_acceptance_is_primary_even_when_arm_does_not_claim_done(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            arm = root / "arm.py"
            arm.write_text(
                "import json, os\n"
                "from pathlib import Path\n"
                "Path('solution.txt').write_text('correct\\n')\n"
                "Path(os.environ['HARNESS_RESULT_FILE']).write_text(json.dumps({"
                "'claimed_completed': False, 'token_metrics_available': False, "
                "'budget_enforced': True}))\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "version": 1,
                    "arms": {
                        name: {"command": [sys.executable, str(arm)]}
                        for name in ("direct", "lh", "longcode")
                    },
                    "cases": [{
                        "id": "external-primary",
                        "suite": "coding",
                        "source": str(source),
                        "hidden_checks": ["grep -q '^correct$' solution.txt"],
                    }],
                }),
                encoding="utf-8",
            )

            records = run_benchmark(manifest, root / "results.jsonl")

            self.assertTrue(all(item.success for item in records))
            self.assertTrue(all(item.run_exit_code == 1 for item in records))
            self.assertTrue(all(not item.false_completed for item in records))

    def test_chatgpt_subscription_manifest_uses_observed_tokens_without_api_proxy(self):
        model = "gpt-5.6-sol"
        manifest = {
            "version": 1,
            "budget": {"max_wall_seconds": 900, "max_rounds": 6},
            "authentication": {
                "mode": "chatgpt_subscription",
                "ignore_user_config": True,
                "strip_environment_keys": ["OPENAI_API_KEY", "CODEX_API_KEY"],
            },
            "process_environment": {
                "HTTPS_PROXY": "http://127.0.0.1:7897",
            },
            "matched_conditions": {
                "model": model,
                "reasoning_effort": "xhigh",
                "tools": {},
                "environment": {},
            },
            "arms": {
                name: {
                    "command": [
                        sys.executable,
                        "-c",
                        "pass",
                        "--model",
                        model,
                    ]
                }
                for name in ("direct", "lh", "longcode")
            },
            "cases": [
                {
                    "id": "oauth",
                    "suite": "coding",
                    "source": ".",
                    "hidden_checks": ["true"],
                }
            ],
        }

        _validate_manifest(manifest)

    def test_local_provider_guard_accepts_only_literal_loopback_http(self):
        self.assertTrue(_is_loopback_provider_url("http://127.0.0.1:1234/v1"))
        self.assertTrue(_is_loopback_provider_url("http://[::1]:1234/v1"))
        self.assertFalse(_is_loopback_provider_url("https://api.openai.com/v1"))
        self.assertFalse(_is_loopback_provider_url("http://localhost:1234/v1"))

    def test_manifest_rejects_nonlocal_upstream_when_loopback_is_required(self):
        manifest = {
            "version": 1,
            "budget": {
                "max_model_calls": 2,
                "max_input_tokens": 100,
                "max_output_tokens": 20,
            },
            "provider_proxy": {
                "upstream_base_url": "https://api.openai.com/v1",
                "require_loopback": True,
                "strip_environment_keys": ["OPENAI_API_KEY"],
            },
            "arms": {
                name: {"command": [sys.executable, "-c", "pass"]}
                for name in ("direct", "lh", "longcode")
            },
            "cases": [
                {
                    "id": "local-only",
                    "suite": "coding",
                    "source": ".",
                    "hidden_checks": ["true"],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "only permits"):
            _validate_manifest(manifest)

    def test_matched_runner_clones_one_snapshot_and_scores_hidden_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "seed.txt").write_text("same snapshot\n", encoding="utf-8")
            arm = root / "arm"
            arm.write_text(
                "#!/usr/bin/env python3\n"
                + textwrap.dedent(
                    """\
                    import json
                    import os
                    from pathlib import Path

                    name = os.environ["HARNESS_ARM"]
                    task = json.loads(os.environ["HARNESS_TASK_JSON"])
                    Path(task["origin_seed"]).write_text("mutated during benchmark\\n")
                    Path("result.txt").write_text("wrong\\n" if name == "direct" else "correct\\n")
                    tokens = {"direct": 160, "lh": 200, "longcode": 100}[name]
                    Path(os.environ["HARNESS_RESULT_FILE"]).write_text(json.dumps({
                        "claimed_completed": True,
                        "input_tokens": tokens // 2,
                        "output_tokens": tokens // 2,
                        "recovered": False,
                        "committed_progress_lost": False,
                        "token_metrics_available": True,
                        "budget_enforced": True
                    }))
                    """
                ),
                encoding="utf-8",
            )
            arm.chmod(0o755)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "repetitions": 1,
                        "timeout_seconds": 30,
                        "hidden_check_timeout_seconds": 10,
                        "budget": {"max_calls": 3},
                        "arms": {
                            "direct": {"command": [str(arm)]},
                            "lh": {"command": [str(arm)]},
                            "longcode": {"command": [str(arm)]},
                        },
                        "cases": [
                            {
                                "id": "matched-01",
                                "suite": "coding",
                                "source": str(source),
                                "task": {
                                    "goal": "write correct result",
                                    "origin_seed": str(source / "seed.txt"),
                                },
                                "hidden_checks": [
                                    "grep -q '^correct$' result.txt",
                                    "grep -q '^same snapshot$' seed.txt",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "results.jsonl"
            records = run_benchmark(manifest, output, seed=7)

            self.assertEqual(len(records), 3)
            self.assertEqual(len({item.snapshot_id for item in records}), 1)
            by_arm = {item.arm: item for item in records}
            self.assertFalse(by_arm["direct"].success)
            self.assertTrue(by_arm["direct"].false_completed)
            self.assertTrue(by_arm["lh"].success)
            self.assertTrue(by_arm["longcode"].success)
            self.assertEqual(len(load_records(output)), 3)
            report = comparison_report(records)
            self.assertTrue(report["claims"]["success_higher_or_cost_20pct_lower"])
            self.assertTrue(report["all_testable_claims_pass"])
            evidence = output.with_suffix(".jsonl.evidence")
            self.assertEqual(len(list(evidence.rglob("run.json"))), 3)

    def test_runner_refuses_to_overwrite_prior_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = root / "results.jsonl"
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_benchmark(manifest, output)

    def test_runner_selects_exact_pairs_and_overrides_wall_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            arm = root / "arm.py"
            arm.write_text(
                "import json, os\n"
                "from pathlib import Path\n"
                "Path(os.environ['HARNESS_RESULT_FILE']).write_text(json.dumps({"
                "'claimed_completed': True, 'input_tokens': 1, 'output_tokens': 1, "
                "'model_calls': 1, 'token_metrics_available': True, 'budget_enforced': True}))\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "timeout_seconds": 1,
                        "budget": {"max_wall_seconds": 1, "max_rounds": 1},
                        "arms": {
                            name: {"command": [sys.executable, str(arm)]}
                            for name in ("direct", "lh", "longcode")
                        },
                        "cases": [
                            {
                                "id": case_id,
                                "suite": "subset",
                                "source": str(source),
                                "hidden_checks": ["true"],
                            }
                            for case_id in ("case-a", "case-b")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "selected.jsonl"
            records = run_benchmark(
                manifest,
                output,
                timeout_seconds=17,
                only_runs={("case-a", "lh"), ("case-b", "direct")},
            )

            self.assertEqual(
                {(item.base_case_id, item.arm) for item in records},
                {("case-a", "lh"), ("case-b", "direct")},
            )
            for item in records:
                evidence = json.loads(
                    (
                        output.with_suffix(".jsonl.evidence")
                        / f"{item.base_case_id}-r1"
                        / item.arm
                        / "run.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(evidence["timeout_seconds"], 17)
                self.assertEqual(evidence["budget"]["max_wall_seconds"], 17)

            with self.assertRaisesRegex(ValueError, "unknown benchmark run selector"):
                run_benchmark(
                    manifest,
                    root / "invalid.jsonl",
                    only_runs={("missing", "lh")},
                )

    def test_hidden_pass_after_wall_timeout_is_not_counted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            arm = root / "slow-arm.py"
            arm.write_text(
                "from pathlib import Path\n"
                "import time\n"
                "Path('result.txt').write_text('correct\\n')\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "timeout_seconds": 1,
                        "arms": {
                            name: {"command": [sys.executable, str(arm)]}
                            for name in ("direct", "lh", "longcode")
                        },
                        "cases": [
                            {
                                "id": "timeout",
                                "suite": "coding",
                                "source": str(source),
                                "hidden_checks": ["grep -q '^correct$' result.txt"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = run_benchmark(manifest, root / "results.jsonl")

            self.assertEqual(len(records), 3)
            self.assertTrue(all(item.timed_out for item in records))
            self.assertTrue(all(not item.success for item in records))
            self.assertTrue(all(item.budget_enforced for item in records))
            self.assertTrue(all(item.budget_exhausted for item in records))

    def test_strict_token_budget_requires_provider_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "budget": {
                            "max_model_calls": 2,
                            "max_input_tokens": 100,
                            "max_output_tokens": 20,
                        },
                        "arms": {
                            name: {"command": [sys.executable, "-c", "pass"]}
                            for name in ("direct", "lh", "longcode")
                        },
                        "cases": [
                            {
                                "id": "strict-budget",
                                "suite": "coding",
                                "source": str(source),
                                "hidden_checks": ["true"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "require provider_proxy"):
                run_benchmark(manifest, root / "results.jsonl")

    def test_hidden_root_inside_source_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            hidden = source / "hidden"
            hidden.mkdir(parents=True)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "arms": {
                            name: {"command": [sys.executable, "-c", "pass"]}
                            for name in ("direct", "lh", "longcode")
                        },
                        "cases": [
                            {
                                "id": "leaked-hidden-check",
                                "suite": "coding",
                                "source": str(source),
                                "hidden_root": str(hidden),
                                "hidden_checks": ["python3 {HIDDEN_ROOT}/verify.py"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "results.jsonl"
            with self.assertRaisesRegex(ValueError, "outside the source fixture"):
                run_benchmark(manifest, output)
            self.assertFalse(output.with_suffix(".jsonl.evidence").exists())

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    def test_arm_timeout_terminates_descendant_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "orphan-wrote.txt"
            child = (
                "import time; from pathlib import Path; "
                f"time.sleep(1); Path({str(marker)!r}).write_text('orphan')"
            )
            parent = root / "parent.py"
            parent.write_text(
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )

            exit_code, timed_out, _, _ = _run_arm_process(
                [sys.executable, str(parent)],
                cwd=root,
                environment=dict(os.environ),
                timeout=0.2,
            )
            time.sleep(1.1)

            self.assertIsNone(exit_code)
            self.assertTrue(timed_out)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
