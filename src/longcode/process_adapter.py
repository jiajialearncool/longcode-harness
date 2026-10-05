from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .backends import (
    AUDITOR_SCHEMA,
    EXECUTOR_SCHEMA,
    MANAGER_SCHEMA,
    BackendResult,
    _matches_schema,
    _parse_json_object,
)
from .models import AuditResult, CommandResult, ManagerDecision, Subtask, TaskContract, TaskState


PROCESS_ADAPTER_PROTOCOL_VERSION = 1


class JsonProcessAgentAdapter:
    """Provider-neutral Agent adapter using strict JSON over a fresh local process."""

    def __init__(
        self,
        command: str | list[str],
        *,
        timeout: int = 1800,
        capabilities: list[str] | set[str] | None = None,
    ):
        parts = shlex.split(command) if isinstance(command, str) else list(command)
        if not parts:
            raise ValueError("agent adapter command must not be empty")
        resolved = shutil.which(parts[0])
        if resolved is None:
            candidate = Path(parts[0]).expanduser()
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if resolved is None:
            raise FileNotFoundError(f"agent adapter executable was not found: {parts[0]}")
        if timeout < 1:
            raise ValueError("agent adapter timeout must be at least 1 second")
        self.command = [resolved, *parts[1:]]
        self.timeout = timeout
        self.capabilities = frozenset({"cli", "filesystem", *(capabilities or [])})

    def execute(
        self,
        workspace: Path,
        contract: TaskContract,
        subtask: Subtask,
    ) -> BackendResult:
        return self._invoke(
            "executor",
            workspace,
            {
                "contract": contract.to_dict(),
                "subtask": subtask.to_dict(),
            },
            EXECUTOR_SCHEMA,
        )

    def manage(
        self,
        workspace: Path,
        contract: TaskContract,
        state: TaskState,
    ) -> tuple[BackendResult, ManagerDecision | None]:
        result = self._invoke(
            "manager",
            workspace,
            {"contract": contract.to_dict(), "state": state.to_dict()},
            MANAGER_SCHEMA,
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
        result = self._invoke(
            "auditor",
            workspace,
            {
                "contract": contract.to_dict(),
                "state": state.to_dict(),
                "subtask": subtask.to_dict() if subtask else None,
                "changed_paths": list(changed_paths),
                "checks": [item.to_dict() for item in checks],
                "executor_report": executor_report,
            },
            AUDITOR_SCHEMA,
        )
        if not result.ok:
            return result, AuditResult(
                verdict="uncertain",
                summary="External auditor adapter failed",
                evidence=[],
                risks=[result.stderr or "adapter returned invalid output"],
            )
        report = result.report
        return result, AuditResult(
            verdict=report["verdict"],
            summary=report["summary"],
            evidence=list(report["evidence"]),
            risks=list(report["risks"]),
        )

    def _invoke(
        self,
        role: str,
        workspace: Path,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> BackendResult:
        started = time.monotonic()
        envelope = {
            "protocol_version": PROCESS_ADAPTER_PROTOCOL_VERSION,
            "role": role,
            "workspace": str(workspace.resolve()),
            **payload,
        }
        try:
            completed = subprocess.run(
                [*self.command, "--role", role],
                cwd=workspace,
                input=json.dumps(envelope, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            report = _parse_json_object(completed.stdout)
            valid = _matches_schema(report, schema)
            stderr = completed.stderr
            if completed.returncode == 0 and not valid:
                stderr = (stderr + "\n" if stderr else "") + "Adapter stdout did not match role schema"
            return BackendResult(
                ok=completed.returncode == 0 and valid,
                report=report,
                stdout=completed.stdout,
                stderr=stderr,
                return_code=completed.returncode,
                duration_seconds=round(time.monotonic() - started, 3),
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            return BackendResult(
                ok=False,
                report={},
                stdout=stdout,
                stderr=f"Agent adapter timed out. {stderr}",
                return_code=None,
                duration_seconds=round(time.monotonic() - started, 3),
            )
