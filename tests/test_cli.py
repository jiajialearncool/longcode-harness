from __future__ import annotations

import io
import subprocess
import textwrap
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from longcode.cli import main
from longcode.storage import RuntimeStore


class CliTests(unittest.TestCase):
    def test_full_cli_run_with_local_fake_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = workspace / ".longcode"
            fake_codex = root / "fake-codex"
            fake_codex.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    sandbox = args[args.index("--sandbox") + 1]
                    workspace = Path(args[args.index("--cd") + 1])
                    output = Path(args[args.index("--output-last-message") + 1])
                    if sandbox == "workspace-write":
                        (workspace / "done.py").write_text("DONE = True\\n")
                        report = {"summary": "implemented", "claimed_complete": True, "changed_files": ["done.py"], "tests_run": [], "remaining_risks": [], "fault": None}
                    else:
                        report = {"verdict": "pass", "summary": "verified", "evidence": ["done.py"], "risks": []}
                    output.write_text(json.dumps(report))
                    """
                )
            )
            fake_codex.chmod(0o755)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(
                    main(
                        [
                            "init",
                            "--workspace",
                            str(workspace),
                            "--runtime",
                            str(runtime),
                            "--goal",
                            "Create done.py",
                            "--acceptance",
                            "done.py exists",
                            "--allowed-path",
                            "done.py",
                            "--check",
                            "test -f done.py",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--runtime",
                            str(runtime),
                            "--codex",
                            str(fake_codex),
                            "--max-rounds",
                            "2",
                        ]
                    ),
                    0,
                )
            self.assertTrue((workspace / "done.py").exists())
            self.assertIn("Status: completed", output.getvalue())

    def test_installation_free_launcher(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            ["python3", str(project_root / "longcode_cli.py"), "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "1.2.0")
        shell_launcher = subprocess.run(
            [str(project_root / "longcode"), "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(shell_launcher.returncode, 0, shell_launcher.stderr)
        self.assertEqual(shell_launcher.stdout.strip(), "1.2.0")

    def test_init_status_revise_and_verify_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            runtime = root / "runtime"
            workspace.mkdir()
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(
                    main(
                        [
                            "init",
                            "--workspace",
                            str(workspace),
                            "--runtime",
                            str(runtime),
                            "--goal",
                            "CLI durable goal",
                            "--acceptance",
                            "CLI criterion",
                            "--check",
                            "true",
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["status", "--runtime", str(runtime)]), 0)
                self.assertEqual(
                    main(
                        [
                            "revise",
                            "--runtime",
                            str(runtime),
                            "--add-constraint",
                            "Keep changes small",
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["verify", "--runtime", str(runtime)]), 0)
            rendered = output.getvalue()
            self.assertIn("CLI durable goal", rendered)
            self.assertIn("Created active goal version 2", rendered)

    def test_product_cli_persists_flow_verifier_and_completes_with_fake_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            fake_codex = root / "fake-codex"
            fake_codex.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    sandbox = args[args.index("--sandbox") + 1]
                    workspace = Path(args[args.index("--cd") + 1])
                    output = Path(args[args.index("--output-last-message") + 1])
                    if sandbox == "workspace-write":
                        (workspace / "app.py").write_text("READY = True\\n")
                        report = {"summary": "built flow", "claimed_complete": True, "changed_files": ["app.py"], "tests_run": [], "remaining_risks": [], "fault": None}
                    else:
                        report = {"verdict": "pass", "summary": "verified", "evidence": ["app.py"], "risks": []}
                    output.write_text(json.dumps(report))
                    """
                )
            )
            fake_codex.chmod(0o755)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(
                    main(
                        [
                            "init",
                            "--workspace", str(workspace),
                            "--runtime", str(runtime),
                            "--mode", "product",
                            "--goal", "Build a small dashboard",
                            "--problem", "Users need a summary",
                            "--target-user", "manager",
                            "--key-flow", "open the dashboard",
                            "--product-check", "test -f app.py",
                            "--check", "test -f app.py",
                            "--allowed-path", "app.py",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "run", "--runtime", str(runtime), "--codex", str(fake_codex),
                            "--no-auditor", "--max-rounds", "2",
                        ]
                    ),
                    0,
                )
            contract = RuntimeStore(runtime).load_contract()
            self.assertEqual(contract.default_verification_profile, "product_flow")
            self.assertEqual(contract.product.product_checks, ["test -f app.py"])
            self.assertEqual(RuntimeStore(runtime).load_state().status, "completed")
            self.assertTrue((workspace / "app.py").exists())

    def test_cli_runs_without_codex_through_json_process_adapter(self):
        from tests.test_process_adapter import write_adapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            adapter = root / "adapter"
            write_adapter(adapter)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(
                    main(
                        [
                            "init", "--workspace", str(workspace), "--runtime", str(runtime),
                            "--goal", "Use external adapter", "--acceptance", "Output exists",
                            "--allowed-path", "adapter-output.py",
                            "--check", "test -f adapter-output.py",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "run", "--runtime", str(runtime),
                            "--agent-command", str(adapter), "--max-rounds", "2",
                        ]
                    ),
                    0,
                )
            self.assertTrue((workspace / "adapter-output.py").exists())
            event_types = [
                item["type"] for item in RuntimeStore(runtime).iter_events()
            ]
            self.assertNotIn("manager_backend_completed", event_types)
            self.assertIn("control_level_changed", event_types)

    def test_cli_product_run_uses_external_environment_evidence(self):
        from tests.test_environment_adapter import write_environment_controller

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            agent = root / "agent"
            environment = root / "environment"
            from tests.test_process_adapter import write_adapter

            write_adapter(agent)
            write_environment_controller(environment)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(
                    main(
                        [
                            "init", "--workspace", str(workspace), "--runtime", str(runtime),
                            "--mode", "product", "--goal", "Build dashboard",
                            "--problem", "Managers need a summary", "--target-user", "manager",
                            "--key-flow", "open the dashboard page",
                            "--allowed-path", "adapter-output.py",
                            "--check", "test -f adapter-output.py",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "run", "--runtime", str(runtime),
                            "--agent-command", str(agent),
                            "--environment-command", str(environment),
                            "--environment-capability", "browser",
                            "--no-auditor", "--max-rounds", "2",
                        ]
                    ),
                    0,
                )
            state = RuntimeStore(runtime).load_state()
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.environment_evidence[0]["kind"], "browser")


if __name__ == "__main__":
    unittest.main()
