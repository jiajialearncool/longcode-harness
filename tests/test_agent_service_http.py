from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from longcode.agent_config import DEFAULT_SETTINGS
from longcode.agent_runtime import Cancellation, Cancelled
from longcode.agent_service import AgentService
from longcode.local_server import LocalServer, api
from longcode.model_bridge import ModelBridge
from longcode.native_agent import NativeAgentBackend, user_message
from longcode.mcp_client import discover_mcp
from longcode.agent_tools import Tools

from tests import test_autonomous_agent as fixtures
from tests.test_autonomous_agent import assistant


class HttpTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    tearDown = fixtures.AgentTests.tearDown

    def start_server(self):
        class Backend:
            def converse(_, workspace, messages):
                return "测试回复", [*messages, assistant({"type": "text", "text": "测试回复"})], {}
        service = AgentService(self.home, backend_factory=lambda *a, **k: Backend())
        try:
            self.server = LocalServer(self.home, service=service)
        except PermissionError:
            self.skipTest("环境不允许本机监听；需在允许本机网络的环境运行 HTTP 测试")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        return {"url": f"http://127.0.0.1:{self.server.server_port}", "token": self.server.token}

    def test_no_token_cross_origin_and_bad_host_are_denied(self):
        descriptor = self.start_server()
        for headers, status in (({}, 401), ({"Authorization": "Bearer " + self.server.token, "Origin": "https://evil.example"}, 403),
                                ({"Authorization": "Bearer " + self.server.token, "Host": "evil.example"}, 403)):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(urllib.request.Request(descriptor["url"] + "/api/sessions", headers=headers))
            self.assertEqual(raised.exception.code, status)
        self.assertEqual(api(descriptor, "/api/health")["status"], "ok")

    def test_public_page_contains_no_token_and_serves_assets(self):
        descriptor = self.start_server()
        for path in ("/", "/app.js", "/style.css"):
            with urllib.request.urlopen(descriptor["url"] + path) as response:
                text = response.read().decode()
                self.assertNotIn(self.server.token, text)
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_http_conversation_reconnect_export_and_dedup(self):
        descriptor = self.start_server()
        session = api(descriptor, "/api/sessions", {"workspace": str(self.workspace)})
        sid = session["id"]
        payload = {"text": "你好", "request_id": "request-http-1"}
        api(descriptor, f"/api/sessions/{sid}/message", payload)
        for _ in range(100):
            state = api(descriptor, f"/api/sessions/{sid}")
            if state["status"] != "running":
                break
            time.sleep(.02)
        self.assertEqual(state["messages"][-1]["content"][0]["text"], "测试回复")
        self.assertTrue(api(descriptor, f"/api/sessions/{sid}/message", payload)["duplicate"])
        events = api(descriptor, f"/api/sessions/{sid}/events")
        cursor = events[-1]["sequence"]
        self.assertEqual(api(descriptor, f"/api/sessions/{sid}/events?after={cursor}"), [])
        req = urllib.request.Request(descriptor["url"] + f"/api/sessions/{sid}/export", headers={"Authorization": "Bearer " + descriptor["token"]})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode()
            self.assertIn("你好", text)
            self.assertIn("测试回复", text)


class BridgeTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    tearDown = fixtures.AgentTests.tearDown

    def test_bridge_status_key_logout_and_refresh_failure_preserves_credential(self):
        bridge = ModelBridge(self.home)
        result = bridge.request("status", provider="openai")
        self.assertGreater(len(result["models"]), 0)
        self.assertEqual(result["credentials"], [])
        bridge.request("key", provider="openai", key="fixture-fake-key")
        self.assertNotIn("fixture", json.dumps(bridge.request("status", provider="openai")))
        self.assertEqual((self.home / "auth.json").stat().st_mode & 0o777, 0o600)
        bridge.request("logout", provider="openai")
        self.assertEqual(bridge.request("status", provider="openai")["credentials"], [])

    def test_cancel_while_waiting_for_auth_lock(self):
        import fcntl
        bridge = ModelBridge(self.home)
        cancel = Cancellation()
        with (self.home / "auth.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            timer = threading.Timer(.15, cancel.cancel)
            timer.start()
            try:
                with self.assertRaises(Cancelled):
                    bridge.request("status", provider="openai", cancel=cancel)
            finally:
                timer.join()

    def fixture_provider(self, *, shell=False):
        received = []
        class FakeEndpoint(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(body)
                turn = len(received)
                if turn == 1:
                    delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "fixture-call-1", "type": "function", "function": {
                        "name": "write_file", "arguments": json.dumps({"path": "answer.py", "content": "def answer():\n    return 42\n"})}}]}
                    reason = "tool_calls"
                elif turn == 2 and shell:
                    delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "fixture-call-2", "type": "function", "function": {
                        "name": "shell", "arguments": json.dumps({"command": "python3 -m unittest discover -s tests"})}}]}
                    reason = "tool_calls"
                else:
                    delta, reason = {"role": "assistant", "content": "文件已经修改。"}, "stop"
                chunks = [{"id": "fixture", "object": "chat.completion.chunk", "created": 1, "model": "fixture-model", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"id": "fixture", "object": "chat.completion.chunk", "created": 1, "model": "fixture-model", "choices": [{"index": 0, "delta": {}, "finish_reason": reason}], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}}]
                payload = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload.encode())))
                self.end_headers(); self.wfile.write(payload.encode())
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint)
        except PermissionError:
            self.skipTest("环境不允许本机监听；需在允许本机网络的环境运行协议测试")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        settings = {**self.settings, "provider": "compatible", "model": "fixture-model", "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        ModelBridge(self.home).request("key", provider="compatible", key="fixture-only")
        return settings, received

    def test_actual_pi_protocol_and_native_tool_loop_without_codex_or_claude(self):
        settings, received = self.fixture_provider()
        backend = NativeAgentBackend(settings, home=self.home)
        # The Node bridge is real. Only the remote model is replaced by a local fixture.
        original_which = __import__("shutil").which
        with patch("shutil.which", side_effect=lambda name: None if name in {"codex", "claude"} else original_which(name)):
            text, messages, usage = backend.converse(self.workspace, [user_message("创建 answer.py")])
        self.assertEqual((self.workspace / "answer.py").read_text(), "def answer():\n    return 42\n")
        self.assertEqual(len(received), 2)
        self.assertIn("已修改", json.dumps(received[1], ensure_ascii=False))
        self.assertTrue(usage["token_metrics_available"])

    @unittest.skipUnless(os.environ.get("LONGCODE_TEST_SANDBOX") == "1", "requires macOS sandbox")
    def test_actual_pi_protocol_runs_real_project_tests(self):
        settings, received = self.fixture_provider(shell=True)
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_answer.py").write_text("import unittest\nfrom answer import answer\nclass TestAnswer(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(answer(),42)\n")
        backend = NativeAgentBackend(settings, home=self.home, ask=lambda _: "allow")
        backend.converse(self.workspace, [user_message("实现答案并运行测试")])
        self.assertEqual(len(received), 3)
        self.assertIn("Ran 1 test", json.dumps(received[2]))
        tool_result = [m for m in received[2]["messages"] if m["role"] == "tool"][-1]
        self.assertIn('exit_code', tool_result["content"])
        self.assertNotIn('Permission denied', tool_result["content"])


@unittest.skipUnless(os.environ.get("LONGCODE_TEST_SANDBOX") == "1", "requires real macOS sandbox")
class McpTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    tearDown = fixtures.AgentTests.tearDown

    def test_stdio_server_requires_approval_and_auditor_gets_no_external_tools(self):
        import sys
        script = self.workspace / "fixture_mcp.py"
        script.write_text('''import sys,json
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r: continue
 method=r['method']
 value={'protocolVersion':'2024-11-05','capabilities':{},'serverInfo':{'name':'fixture','version':'1'}} if method=='initialize' else ({'tools':[{'name':'echo','description':'echo','inputSchema':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}]} if method=='tools/list' else {'content':[{'type':'text','text':r['params']['arguments']['text']}]})
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':value}),flush=True)
''')
        settings = {"mcp": [{"command": [sys.executable, str(script)]}]}
        asked = []
        tools = Tools(self.workspace, settings=settings, ask=lambda q: asked.append(q) or "allow")
        from longcode.mcp_client import close_mcp
        self.addCleanup(close_mcp, tools)
        discover_mcp(tools)
        process = tools.mcp_connections[0].process
        self.assertEqual(len(tools.mcp_tools), 1)
        result = tools.call(next(iter(tools.mcp_tools)), {"text": "hello"})
        self.assertEqual(result["content"][0]["text"], "hello")
        self.assertEqual(len(asked), 2)
        self.assertIs(tools.mcp_connections[0].process, process)
        self.assertIsNone(process.poll())
        audit = Tools(self.workspace, role="auditor", settings=settings, ask=lambda _: self.fail("Audit cannot launch MCP"))
        discover_mcp(audit)
        self.assertEqual(audit.mcp_tools, {})


@unittest.skipUnless(os.environ.get("LONGCODE_TEST_SANDBOX") == "1", "requires real macOS sandbox")
class LongTaskTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    tearDown = fixtures.AgentTests.tearDown

    def test_native_task_failed_candidate_repairs_then_promotes(self):
        (self.workspace / "tests").mkdir()
        (self.workspace / "answer.py").write_text("def answer():\n    return 0\n")
        (self.workspace / "tests" / "test_answer.py").write_text("import unittest\nfrom answer import answer\nclass TestAnswer(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(answer(),42)\n")
        report = {"summary": "fixed", "claimed_complete": True, "changed_files": ["answer.py"], "tests_run": [], "remaining_risks": [], "fault": None}
        bridge = fixtures.ScriptedBridge([
            assistant(fixtures.call("write_file", {"path": "answer.py", "content": "def answer():\n    return 41\n"})),
            assistant(fixtures.call("submit_report", report)),
            assistant(fixtures.call("write_file", {"path": "answer.py", "content": "def answer():\n    return 42\n"})),
            assistant(fixtures.call("submit_report", report)),
        ])
        def factory(settings, **kwargs):
            return NativeAgentBackend({**settings, "model": "fixture"}, bridge=bridge, **kwargs)
        service = AgentService(self.home, backend_factory=factory)
        sid = service.store.create(str(self.workspace))["id"]
        draft = {"objective": "Make answer return the expected value", "acceptance": ["answer returns 42"],
                 "checks": ["python3 -m unittest discover -s tests"]}
        service.start_task(sid, draft, "long-task-request-1")
        for _ in range(500):
            if service.jobs[sid].status != "running":
                break
            time.sleep(.02)
        state = service.session(sid)
        self.assertEqual(state["status"], "idle", state.get("error"))
        self.assertEqual(state["task_state"]["status"], "completed", state["task_state"].get("blocker"))
        self.assertIn("return 42", (self.workspace / "answer.py").read_text())
        results = [e["data"] for e in service.store.events(sid) if e["type"] == "check_finished"]
        self.assertFalse(results[0]["passed"])
        self.assertTrue(results[-1]["passed"])
        self.assertNotIn("return 41", (self.workspace / "answer.py").read_text())


if __name__ == "__main__":
    unittest.main()
