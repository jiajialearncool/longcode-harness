"""Explicitly configured MCP tools; each call requires approval, including reads.

Supports newline stdio and Streamable HTTP JSON/SSE responses. No server is
started or connected merely because it appears in a project file.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import sys
import tempfile
from pathlib import Path
import urllib.request
from contextlib import contextmanager

from .agent_runtime import run_process, stop_process
from .agent_tools import sandbox_profile


class Connection:
    def __init__(self, config, tools):
        self.config, self.tools = config, tools
        self.process = None
        self.session = None
        self.sequence = 0
        self.lines = queue.Queue()
        self.temporary = None

    def start(self):
        if self.config.get("command"):
            command = self.config["command"]
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError("MCP command 必须为命令和参数组成的列表")
            if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
                raise RuntimeError("本地 MCP 命令隔离目前仅支持 macOS")
            self.temporary = tempfile.TemporaryDirectory(prefix="longcode-mcp-")
            temporary = Path(self.temporary.name).resolve()
            profile = sandbox_profile(self.tools.root, temporary, home=self.tools.home,
                                      network=self.config.get("network", False))
            env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TMPDIR"}}
            env.update(HOME=str(temporary), TMPDIR=str(temporary))
            for key in self.config.get("env_names", []):
                if key in os.environ:
                    env[key] = os.environ[key]
            self.process = subprocess.Popen(["/usr/bin/sandbox-exec", "-p", profile, *command], cwd=self.tools.root, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
            def read():
                for line in self.process.stdout:
                    if len(line) <= 1000000:
                        self.lines.put(line)
                self.lines.put(None)
            threading.Thread(target=read, daemon=True).start()
        elif not self.config.get("url"):
            raise ValueError("MCP 服务需要 command 或 url")
        result = self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                            "clientInfo": {"name": "longcode", "version": "2.0.0"}})
        self.version = result.get("protocolVersion", "2024-11-05")
        self.rpc("notifications/initialized", {}, notification=True)

    def rpc(self, method, params, *, notification=False):
        self.tools.cancel.check()
        self.sequence += 1
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            body["id"] = self.sequence
        if self.process:
            self.process.stdin.write(json.dumps(body) + "\n"); self.process.stdin.flush()
            if notification:
                return {}
            deadline = time.monotonic() + 60
            while True:
                self.tools.cancel.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("MCP 服务响应超时")
                try:
                    line = self.lines.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("MCP 服务已退出")
                value = json.loads(line)
                if value.get("id") == self.sequence and ("result" in value or "error" in value):
                    break
                if "method" in value and "id" in value:
                    self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": value["id"],
                        "error": {"code": -32601, "message": "Server requests not supported"}}) + "\n")
                    self.process.stdin.flush()
        else:
            url = self.config["url"]
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.username or parsed.password or parsed.scheme not in {"http", "https"}:
                raise ValueError("MCP 地址必须为不含凭据的 HTTP(S) 地址")
            if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("非本机 MCP 服务必须使用 HTTPS")
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            if self.session:
                headers["Mcp-Session-Id"] = self.session
                headers["MCP-Protocol-Version"] = self.version
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                self.session = response.headers.get("Mcp-Session-Id", self.session)
                if notification:
                    return {}
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    value = None
                    size = 0
                    for line in response:
                        self.tools.cancel.check()
                        size += len(line)
                        if size > 1000000:
                            raise ValueError("MCP 结果过大")
                        if line.startswith(b"data:"):
                            candidate = json.loads(line[5:])
                            if candidate.get("id") == self.sequence:
                                value = candidate; break
                    if value is None:
                        raise RuntimeError("MCP 未返回对应结果")
                else:
                    raw = response.read(1000001)
                    if len(raw) > 1000000:
                        raise ValueError("MCP 结果过大")
                    value = json.loads(raw)
        if "error" in value:
            raise RuntimeError("MCP 调用失败：" + str(value["error"])[:2000])
        return value.get("result", {})

    def close(self):
        if self.process:
            stop_process(self.process)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
        if self.temporary:
            self.temporary.cleanup()


@contextmanager
def connect(config, tools):
    client = Connection(config, tools)
    try:
        client.start()
        yield client
    finally:
        client.close()


def discover_mcp(tools):
    tools.mcp_mapping = {}
    tools.mcp_connections = {}
    for index, config in enumerate(tools.settings.get("mcp", [])):
        if not isinstance(config, dict):
            raise ValueError("每项 MCP 配置必须为对象")
        if tools.role in {"manager", "auditor"}:
            # External servers can lie about readOnlyHint; do not expose them to audit.
            continue
        if not tools.ask or tools.ask({"kind": "permission", "message": "连接此 MCP 服务？本地进程受项目文件隔离；远程服务会收到随后批准的工具参数。", "server": config}) != "allow":
            continue
        client = Connection(config, tools)
        tools.mcp_connections[index] = client
        try:
            client.start()
            result = client.rpc("tools/list", {})
        except BaseException:
            close_mcp(tools)
            raise
        for item in result.get("tools", [])[:50]:
            name = f"mcp_{index}_{len(tools.mcp_mapping)}"
            tools.mcp_tools[name] = {"name": name, "description": item.get("description", item["name"]),
                                     "parameters": item.get("inputSchema", {"type": "object"})}
            tools.mcp_mapping[name] = (index, item["name"])


def close_mcp(tools):
    for client in getattr(tools, "mcp_connections", {}).values():
        client.close()
    tools.mcp_connections = {}


def call_mcp_tool(tools, name, arguments):
    index, remote_name = tools.mcp_mapping[name]
    if not tools.ask or tools.ask({"kind": "permission", "message": "允许调用外部 MCP 工具？", "tool": remote_name, "arguments": arguments}) != "allow":
        raise PermissionError("MCP 工具未获批准")
    client = tools.mcp_connections[index]
    result = client.rpc("tools/call", {"name": remote_name, "arguments": arguments})
    if result.get("isError"):
        raise RuntimeError(json.dumps(result, ensure_ascii=False)[:20000])
    return result
