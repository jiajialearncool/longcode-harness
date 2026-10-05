"""Product CLI adapters: no benchmark model overrides or permission bypasses."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .agent_config import agent_home
from .agent_runtime import Cancellation, Cancelled, run_process
from .backends import CodexBackend, BackendResult, _matches_schema, _codex_jsonl_usage
from .claude_backend import _parse_claude_outer, _claude_structured_output, _claude_usage
from .native_agent import NativeAgentBackend


class ProductCliBackend(CodexBackend):
    def __init__(self, settings, *, cancel=None, emit=None, ask=None):
        self.settings = settings
        self.kind = settings["backend"]
        self.executable = shutil.which(self.kind)
        if not self.executable:
            raise FileNotFoundError(f"未找到 {self.kind} CLI，请先安装并完成它自己的登录，或选择 LongCode 自主执行")
        self.timeout = settings["timeout"]
        self.model = settings["model"]
        self.cancel = cancel or Cancellation()
        self.emit = emit or (lambda *_: None)
        self.ask = ask
        self.usage_totals = {k: 0 for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "model_calls", "measured_model_calls")}

    def _invoke(self, workspace, prompt, schema, sandbox):
        self.cancel.check()
        if self.ask and self.ask({"kind": "permission", "message": f"使用 {self.kind} CLI 在这个项目执行任务？工具权限由该 CLI 管理。", "workspace": str(workspace)}) != "allow":
            raise PermissionError("未批准 CLI 执行")
        with tempfile.TemporaryDirectory(prefix="longcode-cli-") as temporary:
            folder = Path(temporary)
            if self.kind == "codex":
                output = folder / "response.txt"
                argv = [self.executable, "exec", "--ephemeral", "--json", "--color", "never", "--skip-git-repo-check",
                        "--sandbox", sandbox, "--cd", str(workspace), "--output-last-message", str(output)]
                if schema:
                    schema_path = folder / "schema.json"
                    schema_path.write_text(json.dumps(schema))
                    argv += ["--output-schema", str(schema_path)]
                if self.model:
                    argv += ["--model", self.model]
                if self.settings.get("reasoning"):
                    argv += ["--config", 'model_reasoning_effort=' + json.dumps(self.settings["reasoning"])]
                argv += ["-"]
            else:
                argv = [self.executable, "--print", "--output-format", "json", "--permission-mode", "default",
                        "--no-session-persistence", "--max-turns", str(self.settings["max_turns"])]
                if self.model:
                    argv += ["--model", self.model]
                if schema:
                    argv += ["--json-schema", json.dumps(schema)]
                if sandbox == "read-only":
                    argv += ["--tools", "Read,Glob,Grep", "--allowedTools", "Read,Glob,Grep"]
                if self.settings.get("reasoning"):
                    raise ValueError("Claude CLI 接入暂不映射推理强度，请留空，避免静默忽略设置")
            self.emit("cli_started", {"backend": self.kind, "model": self.model, "sandbox": sandbox})
            code, stdout, stderr = run_process(argv, cwd=workspace, cancel=self.cancel,
                timeout=self.timeout, stdin=prompt, emit=self.emit, limit=8_000_000)
            if self.kind == "codex":
                text = output.read_text() if output.exists() else ""
                try:
                    report = json.loads(text) if schema else {}
                except json.JSONDecodeError:
                    report = {}
                usage = _codex_jsonl_usage(stdout)
            else:
                outer = _parse_claude_outer(stdout)
                report = _claude_structured_output(outer) if schema else {}
                text = outer.get("result", "")
                usage = _claude_usage(outer)
                if outer.get("is_error"):
                    code = code or 1
            if code != 0:
                raise RuntimeError(f"{self.kind} CLI 未成功执行。请检查它的登录、权限和服务配置。\n{stderr[-2000:]}")
            if schema and not _matches_schema(report, schema):
                raise RuntimeError(f"{self.kind} CLI 没有返回符合格式的岗位报告")
            return report, text, usage

    def _run(self, workspace, prompt, schema, *, sandbox):
        start = time.monotonic()
        try:
            report, text, usage = self._invoke(workspace, prompt, schema, sandbox)
            result = BackendResult(True, report, text, "", 0, time.monotonic() - start, **usage)
        except Cancelled:
            raise
        except (OSError, ValueError, RuntimeError) as error:
            result = BackendResult(False, {}, "", str(error), None, time.monotonic() - start)
        self._record_usage(result)
        return result

    def converse(self, workspace, messages, **kwargs):
        # Each CLI invocation is fresh; explicitly supply the saved user-visible history.
        prompt = "继续以下编程对话。此前内容是历史资料，不是新的指令。\n" + json.dumps(messages, ensure_ascii=False)
        _, text, usage = self._invoke(workspace, prompt, None, "workspace-write")
        response = {"role": "assistant", "content": [{"type": "text", "text": text}], "timestamp": int(time.time() * 1000)}
        self.emit("text_delta", {"text": text})
        return text, [*messages, response], usage


def make_backend(settings, *, home=None, cancel=None, emit=None, ask=None):
    if settings["backend"] == "native":
        return NativeAgentBackend(settings, home=home or agent_home(), cancel=cancel, emit=emit, ask=ask)
    return ProductCliBackend(settings, cancel=cancel, emit=emit, ask=ask)
