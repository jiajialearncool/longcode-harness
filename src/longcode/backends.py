from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import AuditResult, CommandResult, ManagerDecision, Subtask, TaskContract, TaskState


@dataclass(frozen=True)
class BackendResult:
    ok: bool
    report: dict[str, Any]
    stdout: str
    stderr: str
    return_code: int | None
    duration_seconds: float
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    token_metrics_available: bool = False


class AgentBackend(Protocol):
    def execute(
        self, workspace: Path, contract: TaskContract, subtask: Subtask
    ) -> BackendResult: ...

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
    ) -> tuple[BackendResult, AuditResult]: ...


EXECUTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "claimed_complete": {"type": "boolean"},
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "remaining_risks": {"type": "array", "items": {"type": "string"}},
        "fault": {
            "type": ["object", "null"],
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "provider",
                        "agent_runtime",
                        "tool",
                        "environment",
                        "execution",
                        "verification",
                        "policy",
                        "control_plane",
                        "goal",
                    ],
                },
                "code": {"type": "string"},
                "summary": {"type": "string"},
                "retryable": {"type": "boolean"},
                "scope": {
                    "type": "string",
                    "enum": [
                        "tool_call",
                        "episode",
                        "subtask",
                        "branch",
                        "milestone",
                        "run",
                    ],
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "fatal"],
                },
            },
            "required": [
                "category",
                "code",
                "summary",
                "retryable",
                "scope",
                "severity",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "summary",
        "claimed_complete",
        "changed_files",
        "tests_run",
        "remaining_risks",
        "fault",
    ],
    "additionalProperties": False,
}


AUDITOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "evidence", "risks"],
    "additionalProperties": False,
}


MANAGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["execute", "done", "blocked", "ask"]},
        "criterion_id": {"type": "string"},
        "subtask_goal": {"type": "string"},
        "executor_tier": {"type": "string", "enum": ["E0", "E1", "E2", "E3"]},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "verification_profile": {"type": "string"},
        "required_capabilities": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["cli", "filesystem", "browser", "gui"],
            },
        },
        "reason": {"type": "string"},
        "recipe_id": {"type": "string"},
        "complexity": {"type": "integer"},
        "uncertainty": {"type": "integer"},
        "criticality": {"type": "integer"},
        "verification_difficulty": {"type": "integer"},
    },
    "required": [
        "action", "criterion_id", "subtask_goal", "executor_tier", "risk_level",
        "verification_profile", "required_capabilities", "reason", "complexity",
        "uncertainty", "criticality", "verification_difficulty", "recipe_id",
    ],
    "additionalProperties": False,
}


class CodexBackend:
    """Fresh Codex CLI sessions for execution and independent read-only audit."""

    capabilities = frozenset({"cli", "filesystem"})

    def __init__(
        self,
        *,
        executable: str = "codex",
        model: str | None = None,
        timeout: int = 1800,
        codex_home: Path | str | None = None,
        ignore_user_config: bool = False,
        reasoning_effort: str | None = None,
        externally_sandboxed: bool = False,
    ):
        resolved = shutil.which(executable)
        if not resolved:
            raise FileNotFoundError(f"Codex CLI was not found: {executable}")
        self.executable = resolved
        self.model = model
        self.timeout = timeout
        self.ignore_user_config = ignore_user_config
        self.reasoning_effort = reasoning_effort
        self.externally_sandboxed = externally_sandboxed
        self.usage_totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "model_calls": 0,
            "measured_model_calls": 0,
        }
        self.codex_home = Path(codex_home).expanduser().resolve() if codex_home else None
        if self.codex_home and not self.codex_home.is_dir():
            raise NotADirectoryError(f"CODEX_HOME does not exist: {self.codex_home}")

    def execute(
        self, workspace: Path, contract: TaskContract, subtask: Subtask
    ) -> BackendResult:
        prompt = build_executor_prompt(contract, subtask)
        return self._run(workspace, prompt, EXECUTOR_SCHEMA, sandbox="workspace-write")

    def execute_direct(
        self, workspace: Path, contract: TaskContract
    ) -> BackendResult:
        """Run the whole public contract as one unmediated Codex CLI episode."""
        prompt = build_direct_prompt(contract)
        return self._run(workspace, prompt, EXECUTOR_SCHEMA, sandbox="workspace-write")

    def manage(
        self, workspace: Path, contract: TaskContract, state: TaskState
    ) -> tuple[BackendResult, ManagerDecision | None]:
        prompt = build_manager_prompt(contract, state)
        result = self._run(workspace, prompt, MANAGER_SCHEMA, sandbox="read-only")
        if not result.ok:
            return result, None
        report = result.report
        decision = ManagerDecision(
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
        return result, decision

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
        prompt = build_auditor_prompt(
            contract,
            state,
            subtask=subtask,
            changed_paths=changed_paths,
            checks=checks,
            executor_report=executor_report,
        )
        backend_result = self._run(workspace, prompt, AUDITOR_SCHEMA, sandbox="read-only")
        report = backend_result.report
        if not backend_result.ok:
            audit = AuditResult(
                verdict="uncertain",
                summary="Auditor backend failed",
                evidence=[],
                risks=[backend_result.stderr or "auditor returned no valid report"],
            )
        else:
            audit = AuditResult(
                verdict=report["verdict"],
                summary=report["summary"],
                evidence=list(report["evidence"]),
                risks=list(report["risks"]),
            )
        return backend_result, audit

    def _run(
        self, workspace: Path, prompt: str, schema: dict[str, Any], *, sandbox: str
    ) -> BackendResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="longcode-codex-") as temp_dir:
            temp = Path(temp_dir)
            schema_path = temp / "schema.json"
            output_path = temp / "last-message.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--json",
                "--color",
                "never",
                "--skip-git-repo-check",
            ]
            # The benchmark runner already wraps each complete arm in a macOS
            # sandbox that denies access to hidden checks. macOS does not allow
            # Codex to apply a second Seatbelt profile inside that process. LH's
            # official Codex adapter likewise bypasses its inner sandbox, so do
            # the same for Direct/LongCode only when the outer isolation gate is
            # explicitly present.
            if self.externally_sandboxed:
                command.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                command.extend(["--sandbox", sandbox])
            command.extend(
                [
                    "--cd",
                    str(workspace),
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                ]
            )
            if self.ignore_user_config:
                command.append("--ignore-user-config")
            proxy_url = os.environ.get("HARNESS_BUDGET_PROXY_URL", "").strip()
            proxy_key_env = os.environ.get(
                "HARNESS_BUDGET_PROXY_KEY_ENV", "LONGCODE_BUDGET_PROXY_TOKEN"
            ).strip()
            if proxy_url:
                provider = "longcode_budget_proxy"
                command.extend(
                    [
                        "--config",
                        f'model_provider="{provider}"',
                        "--config",
                        f'model_providers.{provider}.name="LongCode Budget Proxy"',
                        "--config",
                        f"model_providers.{provider}.base_url={json.dumps(proxy_url)}",
                        "--config",
                        f"model_providers.{provider}.env_key={json.dumps(proxy_key_env)}",
                        "--config",
                        f'model_providers.{provider}.wire_api="responses"',
                        "--config",
                        f"model_providers.{provider}.request_max_retries=0",
                        "--config",
                        f"model_providers.{provider}.stream_max_retries=0",
                    ]
                )
            if self.reasoning_effort:
                command.extend(
                    [
                        "--config",
                        f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
                    ]
                )
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")
            try:
                environment = os.environ.copy()
                if self.codex_home:
                    environment["CODEX_HOME"] = str(self.codex_home)
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    env=environment,
                )
                raw_report = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
                report = _parse_json_object(raw_report)
                ok = completed.returncode == 0 and _matches_schema(report, schema)
                usage = _codex_jsonl_usage(completed.stdout)
                result = BackendResult(
                    ok=ok,
                    report=report,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    return_code=completed.returncode,
                    duration_seconds=round(time.monotonic() - started, 3),
                    **usage,
                )
                self._record_usage(result)
                return result
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
                stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
                result = BackendResult(
                    ok=False,
                    report={},
                    stdout=stdout,
                    stderr=f"Codex invocation timed out. {stderr}",
                    return_code=None,
                    duration_seconds=round(time.monotonic() - started, 3),
                    **_codex_jsonl_usage(stdout),
                )
                self._record_usage(result)
                return result

    def _record_usage(self, result: BackendResult) -> None:
        self.usage_totals["model_calls"] += 1
        if result.token_metrics_available:
            self.usage_totals["measured_model_calls"] += 1
            self.usage_totals["input_tokens"] += result.input_tokens
            self.usage_totals["cached_input_tokens"] += result.cached_input_tokens
            self.usage_totals["output_tokens"] += result.output_tokens
            self.usage_totals["reasoning_tokens"] += result.reasoning_tokens


def _codex_jsonl_usage(stdout: str) -> dict[str, int | bool]:
    latest: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("type") != "turn.completed":
            continue
        usage = value.get("usage")
        if isinstance(usage, dict):
            latest = usage
    if latest is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "token_metrics_available": False,
        }
    input_details = latest.get("input_tokens_details", {})
    output_details = latest.get("output_tokens_details", {})
    # Newer Codex CLI builds expose cached/reasoning usage as top-level
    # fields, while older OpenAI-compatible providers nest them under the
    # corresponding details object.  Accept both formats so subscription
    # (ChatGPT OAuth) runs and provider-proxy runs use the same accounting.
    cached_input = latest.get("cached_input_tokens")
    if cached_input is None and isinstance(input_details, dict):
        cached_input = input_details.get("cached_tokens")
    reasoning_output = latest.get("reasoning_output_tokens")
    if reasoning_output is None and isinstance(output_details, dict):
        reasoning_output = output_details.get("reasoning_tokens")
    return {
        "input_tokens": _safe_token_int(latest.get("input_tokens")),
        "cached_input_tokens": _safe_token_int(cached_input),
        "output_tokens": _safe_token_int(latest.get("output_tokens")),
        "reasoning_tokens": _safe_token_int(reasoning_output),
        "token_metrics_available": True,
    }


def _safe_token_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def build_executor_prompt(contract: TaskContract, subtask: Subtask) -> str:
    repair_guidance = ""
    if subtask.repair_context:
        repair_guidance = f"""

VERIFIED REPAIR PACKET:
{json.dumps(subtask.repair_context, ensure_ascii=False, indent=2)}

This is an incremental repair, not a restart. The workspace already contains the previous
candidate. Reproduce the listed failure, preserve every passing behavior, make the smallest
corrective change, run the targeted behavioral check, and then run all regression checks.
"""
    return f"""You are the Executor in a verification-first coding harness.

The immutable goal contract and bounded subtask below are the source of truth. Work on exactly one
subtask. Inspect the repository, make the smallest useful in-scope code changes, and run relevant
checks when practical. Never expand scope, edit forbidden paths, weaken tests, or claim that the
overall goal is complete. Your final JSON report is only an execution report; an independent
verifier decides completion.

GOAL CONTRACT:
{json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)}

CURRENT SUBTASK:
{json.dumps(subtask.to_dict(), ensure_ascii=False, indent=2)}
{repair_guidance}

Return the required structured report after the work. If blocked, explain the blocker in summary
and remaining_risks. Set fault to null after a normal execution. When execution cannot continue,
include a fault object with exactly
category, code, summary, retryable, scope, and severity. Valid categories are provider,
agent_runtime, tool, environment, execution, verification, policy, control_plane, and goal;
valid scopes are tool_call, episode, subtask, branch, milestone, and run. A fault is a signal for
the deterministic recovery controller, not permission to expand scope. Do not modify files merely
to make a verification command vacuously pass.
"""


def build_direct_prompt(contract: TaskContract) -> str:
    return f"""You are the sole coding agent in a direct Codex CLI baseline run.

Complete the entire public task contract below in this one episode. Inspect the repository, make
the smallest coherent in-scope changes, and run relevant checks. Do not weaken tests, edit forbidden
paths, or invent access to hidden checks. In the final structured report, set claimed_complete to
true only if you believe the full public contract is complete; otherwise set it to false and list
the remaining risks. A separate external runner decides actual completion after this episode.

PUBLIC TASK CONTRACT:
{json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)}
"""


def build_manager_prompt(contract: TaskContract, state: TaskState) -> str:
    return f"""You are the Manager in an adaptive long-horizon harness.

You cannot modify or inspect the environment. Decide from the immutable goal contract, typed task
state, verified evidence references, failure fingerprints, and remaining work. Select exactly one
ready acceptance criterion, or return done/blocked/ask. For execute, choose the cheapest executor
tier that is still reliable:

- E0: deterministic recipe, no open-ended model work;
- E1: clear, low-risk, easy-to-verify work;
- E2: normal multi-step work;
- E3: high-risk, architecture-critical, highly uncertain, or capability-intensive work.

Rate complexity, uncertainty, criticality, and verification difficulty from 0 to 3. Never use a low
tier to bypass risk. Do not mark done unless every current requirement has independent evidence.
Use an empty criterion_id for done, blocked, or ask.
Use E0 only when the selected criterion already has a deterministic recipe_id in the contract;
otherwise leave recipe_id empty. Never invent or modify a recipe.
For required_capabilities, use only cli, filesystem, browser, or gui. Ordinary repository coding,
editing, inspection, and test execution require cli and filesystem; do not invent capability names.

GOAL CONTRACT:
{json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)}

TYPED DURABLE STATE:
{json.dumps(state.to_dict(), ensure_ascii=False, indent=2)}
"""


def build_auditor_prompt(
    contract: TaskContract,
    state: TaskState,
    *,
    subtask: Subtask | None,
    changed_paths: list[str],
    checks: list[CommandResult],
    executor_report: dict[str, Any] | None,
) -> str:
    scope = (
        "Audit the entire goal for final completion."
        if subtask is None
        else f"Audit only acceptance criterion {subtask.criterion_id}: {subtask.criterion}"
    )
    return f"""You are an independent, read-only Auditor in a verification-first coding harness.

{scope} Inspect the repository directly. Treat the executor's report as an untrusted claim. Use the
goal contract, actual files, changed paths, and deterministic check results as evidence. Return pass
only when the requested scope is demonstrably satisfied without violating constraints. Return fail
when evidence contradicts completion and uncertain when required evidence is missing. Do not edit
files or suggest expanding the approved scope.

GOAL CONTRACT:
{json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)}

DURABLE STATE:
{json.dumps(state.to_dict(), ensure_ascii=False, indent=2)}

SUBTASK:
{json.dumps(subtask.to_dict() if subtask else None, ensure_ascii=False, indent=2)}

CHANGED PATHS:
{json.dumps(changed_paths, ensure_ascii=False, indent=2)}

DETERMINISTIC CHECKS:
{json.dumps([item.to_dict() for item in checks], ensure_ascii=False, indent=2)}

UNTRUSTED EXECUTOR REPORT:
{json.dumps(executor_report, ensure_ascii=False, indent=2)}
"""


def _parse_json_object(value: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _matches_schema(value: dict[str, Any], schema: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    properties = schema.get("properties", {})
    if any(key not in value for key in schema.get("required", [])):
        return False
    if schema.get("additionalProperties") is False and any(key not in properties for key in value):
        return False
    expected_types = {
        "string": str,
        "boolean": bool,
        "array": list,
        "object": dict,
        "integer": int,
    }
    for key, definition in properties.items():
        if key not in value:
            continue
        declared_type = definition.get("type")
        declared_types = declared_type if isinstance(declared_type, list) else [declared_type]
        if value[key] is None:
            if "null" not in declared_types:
                return False
        else:
            expected = tuple(
                expected_types[item]
                for item in declared_types
                if item in expected_types
            )
            if expected and not isinstance(value[key], expected):
                return False
            if "integer" in declared_types and isinstance(value[key], bool):
                return False
        if "enum" in definition and value[key] not in definition["enum"]:
            return False
        if "array" in declared_types and isinstance(value[key], list):
            item_type = expected_types.get(definition.get("items", {}).get("type"))
            if item_type and any(not isinstance(item, item_type) for item in value[key]):
                return False
            item_enum = definition.get("items", {}).get("enum")
            if item_enum and any(item not in item_enum for item in value[key]):
                return False
    return True
