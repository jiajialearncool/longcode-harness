from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from longcode.agent_config import DEFAULT_SETTINGS, load_settings, save_settings, role_settings
from longcode.agent_runtime import Cancellation, Cancelled, project_lock, run_process
from longcode.agent_service import AgentService, discover_checks, validate_checks
from longcode.agent_tools import Tools
from longcode.backends import EXECUTOR_SCHEMA, BackendResult
from longcode.engine import LongCodeEngine
from longcode.models import TaskContract, TaskState, Subtask
from longcode.native_agent import NativeAgentBackend, user_message
from longcode.session_store import SessionStore, reconcile_messages
from longcode.storage import RuntimeStore
from longcode.verifiers import VerifierMesh


def assistant(*blocks):
    return {"role": "assistant", "content": list(blocks), "stopReason": "stop", "timestamp": 1,
            "usage": {"input": 12, "output": 3, "cacheRead": 2}}


def call(name, arguments, id="call-1"):
    return {"type": "toolCall", "id": id, "name": name, "arguments": arguments}


class ScriptedBridge:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def request(self, op, **kwargs):
        kwargs["context"] = json.loads(json.dumps(kwargs["context"]))
        self.requests.append((op, kwargs))
        return next(self.replies)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "project"
        self.workspace.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.settings = {**DEFAULT_SETTINGS, "model": "test-model"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_loop_edits_without_external_cli(self):
        bridge = ScriptedBridge([
            assistant(call("write_file", {"path": "hello.py", "content": "answer = 42\n"})),
            assistant({"type": "text", "text": "已创建文件。尚未运行测试。"}),
        ])
        events = []
        with patch("shutil.which", side_effect=AssertionError("Must not locate an external agent")):
            backend = NativeAgentBackend(self.settings, home=self.home, bridge=bridge, emit=lambda *e: events.append(e))
            text, messages, usage = backend.converse(self.workspace, [user_message("创建文件")])
        self.assertEqual((self.workspace / "hello.py").read_text(), "answer = 42\n")
        self.assertIn("尚未", text)
        self.assertTrue(any(m["role"] == "toolResult" for m in messages))
        self.assertEqual(usage["input_tokens"], 28)
        self.assertEqual(usage["cached_input_tokens"], 4)
        self.assertIn("file_changed", [x[0] for x in events])

    def test_auditor_cannot_invoke_write_or_shell(self):
        tools = Tools(self.workspace, role="auditor")
        for name, args in (("write_file", {"path": "p", "content": "bad"}), ("shell", {"command": "touch p"})):
            with self.assertRaises(PermissionError):
                tools.call(name, args)
        self.assertFalse((self.workspace / "p").exists())

    def test_native_schema_report_is_checked(self):
        valid = {"summary": "done", "claimed_complete": True, "changed_files": [], "tests_run": [], "remaining_risks": [], "fault": None}
        bridge = ScriptedBridge([assistant(call("submit_report", {"summary": "incomplete"})), assistant(call("submit_report", valid))])
        backend = NativeAgentBackend(self.settings, home=self.home, bridge=bridge)
        result = backend._run(self.workspace, "task", EXECUTOR_SCHEMA, sandbox="workspace-write")
        self.assertTrue(result.ok)
        self.assertEqual(result.report, valid)
        self.assertTrue(bridge.requests[1][1]["context"]["messages"][-1]["isError"])

    def test_role_contexts_start_fresh(self):
        valid = {"summary": "done", "claimed_complete": False, "changed_files": [], "tests_run": [], "remaining_risks": [], "fault": None}
        bridge = ScriptedBridge([assistant(call("submit_report", valid)), assistant(call("submit_report", valid))])
        backend = NativeAgentBackend(self.settings, home=self.home, bridge=bridge)
        for prompt in ("first-task", "second-task"):
            backend._run(self.workspace, prompt, EXECUTOR_SCHEMA, sandbox="workspace-write")
        self.assertNotIn("first-task", json.dumps(bridge.requests[1][1]["context"]))

    def test_paths_block_traversal_symlinks_and_secrets(self):
        tools = Tools(self.workspace, home=self.home)
        (self.workspace / "outside").symlink_to(self.root)
        for path in ("../secret", "outside/secret", ".env", ".env.local", ".git/config", str(self.home / "auth.json")):
            with self.assertRaises(PermissionError, msg=path):
                tools.call("read_file", {"path": path})

    def test_nested_instructions_must_be_read_before_edit(self):
        nested = self.workspace / "src"
        nested.mkdir()
        (nested / "AGENTS.md").write_text("Use tabs")
        tools = Tools(self.workspace)
        value = tools.call("write_file", {"path": "src/x.py", "content": "ok"})
        self.assertIn("Use tabs", value)
        self.assertFalse((nested / "x.py").exists())
        tools.call("write_file", {"path": "src/x.py", "content": "ok"})
        self.assertTrue((nested / "x.py").exists())

    def test_edit_requires_exact_single_match(self):
        (self.workspace / "x").write_text("a a")
        with self.assertRaises(ValueError):
            Tools(self.workspace).call("edit_file", {"path": "x", "old": "a", "new": "b"})
        self.assertEqual((self.workspace / "x").read_text(), "a a")

    def test_denied_shell_is_never_started(self):
        with patch("longcode.agent_tools.run_process") as process:
            with self.assertRaises(PermissionError):
                Tools(self.workspace, ask=lambda _: "deny").call("shell", {"command": "touch x"})
            process.assert_not_called()

    def test_cancel_process_group(self):
        cancel = Cancellation()
        timer = threading.Timer(.2, cancel.cancel)
        timer.start()
        started = time.monotonic()
        with self.assertRaises(Cancelled):
            run_process(["/bin/sh", "-c", "sleep 20 & wait"], cwd=self.workspace, cancel=cancel, timeout=30)
        timer.join()
        self.assertLess(time.monotonic() - started, 4)

    def test_project_lock_cross_home_and_release(self):
        with project_lock(self.home, self.workspace):
            with self.assertRaises(RuntimeError):
                with project_lock(self.root / "other-home", self.workspace):
                    pass
        with project_lock(self.home, self.workspace):
            pass

    def test_settings_roles_and_secret_rejection(self):
        save_settings({**self.settings, "roles": {"auditor": {"model": "review"}}}, self.home)
        self.assertEqual(role_settings(load_settings(self.home), "auditor")["model"], "review")
        self.assertEqual((self.home / "settings.json").stat().st_mode & 0o777, 0o600)
        for change in ({"api_key": "secret"}, {"base_url": "http://evil.example/v1"}, {"roles": {"auditor": {"password": "secret"}}}):
            with self.assertRaises(ValueError):
                save_settings({**self.settings, **change}, self.home)

    def test_no_checks_cannot_pass_even_with_auditor(self):
        contract = TaskContract.create("goal", ["done"], checks=["python3 -m unittest"])
        contract.checks = []
        outcome = VerifierMesh().verify(self.workspace, contract, TaskState.create(contract), subtask=None, changed_paths=[], executor_report={})
        self.assertEqual(outcome.verdict, "uncertain")

    def test_check_discovery_and_no_fake_verification(self):
        (self.workspace / "package.json").write_text(json.dumps({"scripts": {"test": "echo no test specified && exit 1", "build": "vite build"}}))
        self.assertEqual(discover_checks(self.workspace), ["npm run build"])
        for checks in ([], ["true"], [":"], ["tests || true"]):
            with self.assertRaises(ValueError):
                validate_checks(checks)

    def test_session_trace_redaction_export_and_recovery(self):
        store = SessionStore(self.home)
        session = store.create(str(self.workspace))
        sid = session["id"]
        store.append(sid, "tool", {"api_key": "secret", "text": "sk-abcdefghijklmnop"})
        self.assertNotIn("secret", store.export(sid))
        self.assertNotIn("sk-abcdefghijklmnop", store.export(sid))
        self.assertEqual(store.events(sid)[0]["sequence"], 1)
        # Partial record after a process crash is ignored and does not poison the next append.
        with (store.folder(sid) / "trace.jsonl").open("a") as stream:
            stream.write('{"partial":')
        recovered = SessionStore(self.home)
        recovered.append(sid, "next", {})
        self.assertEqual([e["sequence"] for e in recovered.events(sid)], [1, 2])
        store.update(sid, status="running")
        service = AgentService(self.home)
        self.assertEqual(service.session(sid)["status"], "interrupted")

    def test_interrupted_tools_are_not_replayed(self):
        messages = reconcile_messages([user_message("work"), assistant(call("shell", {"command": "side_effect"}))])
        self.assertEqual(messages[-1]["role"], "toolResult")
        self.assertTrue(messages[-1]["isError"])
        self.assertIn("结果未知", messages[-1]["content"][0]["text"])

    def test_session_paths_and_broad_workspaces_rejected(self):
        store = SessionStore(self.home)
        for target in ("../auth", "abc", "a" * 33):
            with self.assertRaises(ValueError):
                store.folder(target)
        for target in (str(self.home), str(self.root), "/", str(Path.home())):
            with self.assertRaises(ValueError):
                store.create(target)

    def test_service_deduplicates_requests_and_locks_project(self):
        release = threading.Event()
        calls = []
        class Backend:
            def converse(_, workspace, messages):
                calls.append(messages)
                release.wait(3)
                return "ok", messages + [assistant({"type": "text", "text": "ok"})], {}
        service = AgentService(self.home, backend_factory=lambda *a, **k: Backend())
        sid = service.store.create(str(self.workspace))["id"]
        other = service.store.create(str(self.workspace))["id"]
        try:
            service.message(sid, "hello", "request-123")
            self.assertTrue(service.message(sid, "hello", "request-123")["duplicate"])
            with self.assertRaises(RuntimeError):
                service.message(other, "other", "request-456")
            self.assertNotIn(other, service.jobs)
        finally:
            release.set()
            for _ in range(100):
                if service.jobs[sid].status != "running":
                    break
                time.sleep(.02)
        self.assertEqual(len(calls), 1)
        self.assertEqual(service.session(sid)["status"], "idle")

    def test_engine_cancel_does_not_promote_or_complete(self):
        cancel = Cancellation()
        class Backend:
            capabilities = frozenset({"cli", "filesystem"})
            def execute(_, workspace, contract, subtask):
                (workspace / "candidate").write_text("unverified")
                cancel.cancel()
                cancel.check()
        contract = TaskContract.create("implement feature", ["feature works"], checks=["python3 -m unittest"])
        store = RuntimeStore(self.root / "runtime")
        store.initialize(contract, self.workspace)
        result = LongCodeEngine(store, Backend(), auditor=None, cancellation=cancel).run()
        self.assertEqual(result.status, "paused")
        self.assertFalse((self.workspace / "candidate").exists())
        self.assertNotEqual(next(iter(result.criteria.values())).status, "verified")

    def test_task_question_answer_is_saved_before_resume(self):
        service = AgentService(self.home, backend_factory=lambda *a, **k: object())
        sid = service.store.create(str(self.workspace))["id"]
        class Engine:
            def __init__(_, store, *args, **kwargs):
                _.store, _.calls = store, 0
            def run(_, **kwargs):
                _.calls += 1
                state = _.store.load_state()
                state.status = "waiting_input" if _.calls == 1 else "paused"
                state.blocker = "应保留哪些字段？" if _.calls == 1 else None
                _.store.save_state(state)
                return state
        with patch('longcode.agent_service.LongCodeEngine', Engine):
            service.start_task(sid, {"objective":"修改字段", "acceptance":["保留所需字段"],
                "checks":["python3 -m unittest"]}, 'request-question-1')
            for _ in range(100):
                pending = service.session(sid).get('pending', [])
                if pending:
                    break
                time.sleep(.01)
            self.assertTrue(pending)
            service.jobs[sid].answer(pending[0]['id'], '保留姓名和邮箱')
            service.jobs[sid].thread.join(5)
        self.assertEqual(service.jobs[sid].status, 'idle', service.jobs[sid].error)
        contract = RuntimeStore(service.store.folder(sid) / 'task').load_contract()
        self.assertIn('保留姓名和邮箱', '\n'.join(contract.constraints))
        self.assertEqual(service.session(sid)['task_state']['status'], 'paused')


@unittest.skipUnless(os.environ.get("LONGCODE_TEST_SANDBOX") == "1", "requires real macOS sandbox; run explicit sandbox suite")
class SandboxTests(unittest.TestCase):
    setUp = AgentTests.setUp
    tearDown = AgentTests.tearDown
    def test_checks_write_only_to_disposable_copy(self):
        from longcode.agent_checks import make_check_runner
        (self.workspace / "business.txt").write_text("original")
        results = make_check_runner(self.home, Cancellation(), lambda *_: None)(self.workspace,
            ["printf changed > business.txt; printf temp > generated.txt"], 10)
        self.assertTrue(results[0].passed)
        self.assertEqual((self.workspace / "business.txt").read_text(), "original")
        self.assertFalse((self.workspace / "generated.txt").exists())

    def test_actual_sandbox_allows_project_denies_outside_and_credentials(self):
        (self.root / "outside").write_text("private")
        (self.workspace / ".env").write_text("private")
        tools = Tools(self.workspace, home=self.home, ask=lambda _: "allow")
        result = tools.command("printf ok > allowed.txt; cat allowed.txt")
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual(result["stdout"], "ok")
        for command in (f"cat {self.root / 'outside'}", "cat .env", f"printf bad > {self.root / 'outside'}"):
            result = tools.command(command)
            self.assertNotEqual(result["exit_code"], 0, result)
        self.assertEqual((self.root / "outside").read_text(), "private")

    def test_candidate_under_private_home_remains_usable(self):
        candidate = self.home / "sessions" / "candidate"
        candidate.mkdir(parents=True)
        (self.home / "auth.json").write_text("private")
        tools = Tools(candidate, home=self.home, ask=lambda _: "allow")
        tools.call("write_file", {"path": "x", "content": "allowed"})
        self.assertEqual(tools.command("cat x")["exit_code"], 0)
        self.assertNotEqual(tools.command(f"cat {self.home / 'auth.json'}")["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
