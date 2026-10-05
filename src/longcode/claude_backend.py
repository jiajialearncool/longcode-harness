from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .backends import (
    AUDITOR_SCHEMA,
    EXECUTOR_SCHEMA,
    MANAGER_SCHEMA,
    BackendResult,
    _matches_schema,
    _parse_json_object,
    build_auditor_prompt,
    build_executor_prompt,
    build_manager_prompt,
)
from .models import AuditResult, CommandResult, ManagerDecision, Subtask, TaskContract, TaskState


TASK_DENYLIST = "Task,TaskCreate,TaskGet,TaskList,TaskOutput,TaskStop,TaskUpdate,WebFetch,WebSearch"
AUDITOR_DENYLIST = f"{TASK_DENYLIST},Bash,Write,Edit,MultiEdit,NotebookEdit"


class ClaudeCodeBackend:
    """Claude Code backend used by matched public-benchmark adapters.

    The backend keeps LongCode's strict manager/executor/auditor schemas while
    using the same Claude Code execution surface as the published LH runs.  An
    optional local request proxy applies the Qwen sampling parameters that the
    Claude Code CLI cannot express directly.
    """

    capabilities = frozenset({"cli", "filesystem"})

    def __init__(
        self,
        *,
        executable: str = "claude",
        model: str = "qwen3.7-plus",
        timeout: int = 1800,
        max_turns: int = 200,
        request_proxy_script: Path | str | None = None,
        request_overrides: dict[str, Any] | None = None,
        proxy_log_dir: Path | str | None = None,
    ) -> None:
        resolved = shutil.which(executable)
        if not resolved:
            raise FileNotFoundError(f"Claude Code CLI was not found: {executable}")
        if timeout < 1:
            raise ValueError("Claude Code timeout must be positive")
        if max_turns < 1:
            raise ValueError("Claude Code max_turns must be positive")
        self.executable = resolved
        self.model = model
        self.timeout = timeout
        self.max_turns = max_turns
        self.request_proxy_script = (
            Path(request_proxy_script).expanduser().resolve()
            if request_proxy_script
            else None
        )
        if self.request_proxy_script and not self.request_proxy_script.is_file():
            raise FileNotFoundError(
                f"Claude Code request proxy was not found: {self.request_proxy_script}"
            )
        self.request_overrides = dict(request_overrides or {})
        self.proxy_log_dir = (
            Path(proxy_log_dir).expanduser().resolve() if proxy_log_dir else None
        )
        if self.proxy_log_dir:
            self.proxy_log_dir.mkdir(parents=True, exist_ok=True)
        self.usage_totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "model_calls": 0,
            "measured_model_calls": 0,
        }

    def execute(
        self, workspace: Path, contract: TaskContract, subtask: Subtask
    ) -> BackendResult:
        return self._run(
            workspace,
            build_executor_prompt(contract, subtask),
            EXECUTOR_SCHEMA,
            role="executor",
        )

    def manage(
        self, workspace: Path, contract: TaskContract, state: TaskState
    ) -> tuple[BackendResult, ManagerDecision | None]:
        result = self._run(
            workspace,
            build_manager_prompt(contract, state),
            MANAGER_SCHEMA,
            role="manager",
        )
        if not result.ok:
            return result, None
        report = result.report
        return result, ManagerDecision(
            action=report["action"],
            criterion_id=report["criterion_id"] or None,
            subtask_goal=report["subtask_goal"],
            executor_tier=report["executor_tier"],
            risk_level=report["risk_level"],
            verification_profile=report["verification_profile"],
            required_capabilities=list(report["required_capabilities"]),
            reason=report["reason"],
            recipe_id=report["recipe_id"] or None,
            complexity=int(report["complexity"]),
            uncertainty=int(report["uncertainty"]),
            criticality=int(report["criticality"]),
            verification_difficulty=int(report["verification_difficulty"]),
        )

    def audit(
        self,
        workspace: Path,
        contract: TaskContract,
        state: TaskState,
        *,
        subtask: Subtask | None,
        changed_paths: list[str],
        checks: list[CommandResult],
        executor_report: dict[str, Any] | None,
    ) -> tuple[BackendResult, AuditResult]:
        result = self._run(
            workspace,
            build_auditor_prompt(
                contract,
                state,
                subtask=subtask,
                changed_paths=changed_paths,
                checks=checks,
                executor_report=executor_report,
            ),
            AUDITOR_SCHEMA,
            role="auditor",
        )
        if not result.ok:
            return result, AuditResult(
                verdict="uncertain",
                summary="Claude Code auditor failed",
                evidence=[],
                risks=[result.stderr or "auditor returned no valid report"],
            )
        report = result.report
        return result, AuditResult(
            verdict=report["verdict"],
            summary=report["summary"],
            evidence=list(report["evidence"]),
            risks=list(report["risks"]),
        )

    def _run(
        self,
        workspace: Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
    ) -> BackendResult:
        started = time.monotonic()
        command = [
            self.executable,
            "--verbose",
            "--output-format=json",
            "--permission-mode=bypassPermissions",
            "--no-session-persistence",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
            "--json-schema",
            json.dumps(schema, separators=(",", ":"), sort_keys=True),
        ]
        if role == "manager":
            command.extend(["--tools", ""])
        elif role == "auditor":
            command.extend(["--disallowedTools", AUDITOR_DENYLIST])
        else:
            command.extend(["--disallowedTools", TASK_DENYLIST])
        command.append("--print")

        environment = self._base_environment()
        proxy = None
        try:
            proxy, environment = self._start_proxy(environment, role=role)
            completed = subprocess.run(
                command,
                cwd=workspace,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                env=environment,
                check=False,
            )
            outer = _parse_claude_outer(completed.stdout)
            report = _claude_structured_output(outer)
            ok = completed.returncode == 0 and _matches_schema(report, schema)
            stderr = _redact(completed.stderr, environment.get("ANTHROPIC_API_KEY"))
            if completed.returncode == 0 and not ok:
                stderr = (stderr + "\n" if stderr else "") + (
                    "Claude Code output did not contain the required structured report"
                )
            result = BackendResult(
                ok=ok,
                report=report,
                stdout=_redact(completed.stdout, environment.get("ANTHROPIC_API_KEY")),
                stderr=stderr,
                return_code=completed.returncode,
                duration_seconds=round(time.monotonic() - started, 3),
                **_claude_usage(outer),
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            result = BackendResult(
                ok=False,
                report={},
                stdout=_redact(stdout, environment.get("ANTHROPIC_API_KEY")),
                stderr="Claude Code invocation timed out. "
                + _redact(stderr, environment.get("ANTHROPIC_API_KEY")),
                return_code=None,
                duration_seconds=round(time.monotonic() - started, 3),
            )
        except OSError as error:
            result = BackendResult(
                ok=False,
                report={},
                stdout="",
                stderr=f"Claude Code invocation failed: {type(error).__name__}: {error}",
                return_code=None,
                duration_seconds=round(time.monotonic() - started, 3),
            )
        finally:
            if proxy is not None:
                proxy.terminate()
                try:
                    proxy.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proxy.kill()
                    proxy.wait(timeout=5)
        self._record_usage(result)
        return result

    def _base_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "ANTHROPIC_MODEL": self.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": self.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.model,
                "CLAUDE_CODE_SUBAGENT_MODEL": self.model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                "DISABLE_PROMPT_CACHING": "1",
                "DISABLE_INTERLEAVED_THINKING": "1",
                "IS_SANDBOX": "1",
            }
        )
        base_url = environment.get("ANTHROPIC_BASE_URL", "").rstrip("/")
        if base_url.endswith("/v1"):
            environment["ANTHROPIC_BASE_URL"] = base_url[:-3]
        return environment

    def _start_proxy(
        self, environment: dict[str, str], *, role: str
    ) -> tuple[subprocess.Popen[str] | None, dict[str, str]]:
        upstream = environment.get("ANTHROPIC_BASE_URL", "").strip()
        if not (
            self.request_proxy_script
            and self.request_overrides
            and _proxyable_url(upstream)
        ):
            return None, environment

        proxy_root = self.proxy_log_dir or Path(tempfile.mkdtemp(prefix="longcode-claude-proxy-"))
        proxy_root.mkdir(parents=True, exist_ok=True)
        call_id = f"{int(time.time() * 1000)}-{role}-{uuid4().hex[:8]}"
        ready = proxy_root / f"{call_id}.env"
        log = proxy_root / f"{call_id}.log"
        requests_log = proxy_root / f"{call_id}.requests.jsonl"
        command = [
            sys.executable,
            str(self.request_proxy_script),
            "--upstream",
            upstream,
            "--listen-host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready),
            "--log-file",
            str(log),
            "--requests-log-file",
            str(requests_log),
            "--overrides",
            json.dumps(self.request_overrides, separators=(",", ":"), sort_keys=True),
            "--dashscope-wait-timeout",
            os.environ.get("LONGCODE_DASHSCOPE_WAIT_TIMEOUT_SECONDS", "90"),
        ]
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and process.poll() is None:
            if ready.is_file() and ready.stat().st_size:
                match = re.search(
                    r"ANTHROPIC_BASE_URL=['\"]?([^'\"\n]+)",
                    ready.read_text(encoding="utf-8", errors="ignore"),
                )
                if match:
                    proxied = dict(environment)
                    proxied["ANTHROPIC_BASE_URL"] = match.group(1)
                    return process, proxied
            time.sleep(0.05)
        process.terminate()
        process.wait(timeout=5)
        raise OSError(f"Claude Code request proxy did not start; inspect {log}")

    def _record_usage(self, result: BackendResult) -> None:
        self.usage_totals["model_calls"] += 1
        if result.token_metrics_available:
            self.usage_totals["measured_model_calls"] += 1
            self.usage_totals["input_tokens"] += result.input_tokens
            self.usage_totals["cached_input_tokens"] += result.cached_input_tokens
            self.usage_totals["output_tokens"] += result.output_tokens
            self.usage_totals["reasoning_tokens"] += result.reasoning_tokens


def _parse_claude_outer(stdout: str) -> dict[str, Any]:
    value = _parse_json_object(stdout)
    if value:
        return value
    for line in reversed(stdout.splitlines()):
        value = _parse_json_object(line)
        if value.get("type") == "result":
            return value
    return {}


def _claude_structured_output(outer: dict[str, Any]) -> dict[str, Any]:
    structured = outer.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = outer.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return _parse_json_object(result)
    return outer if isinstance(outer, dict) else {}


def _claude_usage(outer: dict[str, Any]) -> dict[str, int | bool]:
    usage = outer.get("usage")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "token_metrics_available": False,
        }
    cached = int(usage.get("cache_read_input_tokens") or 0) + int(
        usage.get("cache_creation_input_tokens") or 0
    )
    return {
        "input_tokens": int(usage.get("input_tokens") or 0) + cached,
        "cached_input_tokens": cached,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "token_metrics_available": True,
    }


def _proxyable_url(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    )


def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***REDACTED***") if secret else text
