from __future__ import annotations

from pathlib import Path
from typing import Callable

from longcode.backends import BackendResult
from longcode.models import AuditResult, Subtask, TaskContract, TaskState


class FakeBackend:
    def __init__(
        self,
        action: Callable[[Path, TaskContract, Subtask], None] | None = None,
        *,
        executor_ok: bool = True,
        audit_verdict: str = "pass",
        claimed_complete: bool = True,
    ):
        self.action = action
        self.executor_ok = executor_ok
        self.audit_verdict = audit_verdict
        self.claimed_complete = claimed_complete
        self.executions: list[tuple[str, str]] = []
        self.audits: list[tuple[str, str | None]] = []

    def execute(self, workspace: Path, contract: TaskContract, subtask: Subtask) -> BackendResult:
        self.executions.append((contract.objective, subtask.criterion_id))
        if self.action:
            self.action(workspace, contract, subtask)
        return BackendResult(
            ok=self.executor_ok,
            report={
                "summary": "fake execution",
                "claimed_complete": self.claimed_complete,
                "changed_files": [],
                "tests_run": [],
                "remaining_risks": [],
            },
            stdout="fake stdout",
            stderr="" if self.executor_ok else "fake failure",
            return_code=0 if self.executor_ok else 1,
            duration_seconds=0.01,
        )

    def audit(
        self,
        workspace: Path,
        contract: TaskContract,
        state: TaskState,
        *,
        subtask: Subtask | None,
        changed_paths,
        checks,
        executor_report,
    ):
        self.audits.append((contract.objective, subtask.criterion_id if subtask else None))
        report = {
            "verdict": self.audit_verdict,
            "summary": "fake audit",
            "evidence": ["fake evidence"],
            "risks": [],
        }
        backend = BackendResult(
            ok=True,
            report=report,
            stdout="fake auditor stdout",
            stderr="",
            return_code=0,
            duration_seconds=0.01,
        )
        return backend, AuditResult(**report)
