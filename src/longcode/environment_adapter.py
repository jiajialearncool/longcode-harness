from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import _parse_json_object
from .transactions import PromotionResult, TransactionManager, WorkspaceTransaction


ENVIRONMENT_ADAPTER_PROTOCOL_VERSION = 1


class EnvironmentAdapterError(OSError):
    pass


class JsonProcessEnvironmentAdapter:
    """Two-phase external environment controller wrapped around a local candidate workspace."""

    def __init__(
        self,
        command: str | list[str],
        runtime_root: Path | str,
        *,
        timeout: int = 300,
        capabilities: list[str] | set[str] | None = None,
    ):
        parts = shlex.split(command) if isinstance(command, str) else list(command)
        if not parts:
            raise ValueError("environment adapter command must not be empty")
        resolved = shutil.which(parts[0])
        if resolved is None:
            candidate = Path(parts[0]).expanduser()
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if resolved is None:
            raise FileNotFoundError(
                f"environment adapter executable was not found: {parts[0]}"
            )
        if timeout < 1:
            raise ValueError("environment adapter timeout must be at least 1 second")
        self.command = [resolved, *parts[1:]]
        self.timeout = timeout
        self.local = TransactionManager(runtime_root)
        self.capabilities = frozenset(
            {"cli", "filesystem", "transaction", "rollback", "external_environment", *(capabilities or [])}
        )

    def begin(
        self,
        workspace: Path | str,
        *,
        round_number: int,
    ) -> "ExternalEnvironmentTransaction":
        local = self.local.begin(workspace, round_number=round_number)
        transaction = ExternalEnvironmentTransaction(
            command=self.command,
            timeout=self.timeout,
            local=local,
            capabilities=set(self.capabilities),
        )
        try:
            response = transaction._call(
                "begin",
                {"round_number": round_number, "capabilities": sorted(self.capabilities)},
            )
            session_id = str(response.get("session_id", "")).strip()
            if not session_id:
                raise EnvironmentAdapterError("environment begin response has no session_id")
            transaction.session_id = session_id
            transaction._merge_response(response)
            return transaction
        except OSError:
            local.rollback()
            raise

    def reconcile(self, journal: dict[str, Any], workspace: Path | str) -> dict[str, Any]:
        """Query an external controller without replaying the interrupted action."""
        source = Path(workspace).expanduser().resolve()
        candidate_id = str(journal.get("candidate_id", ""))
        environment_context = journal.get("environment_context", {})
        if not isinstance(environment_context, dict):
            environment_context = {}
        envelope = {
            "protocol_version": ENVIRONMENT_ADAPTER_PROTOCOL_VERSION,
            "operation": "reconcile",
            "candidate_id": candidate_id,
            "session_id": environment_context.get("session_id"),
            "source_workspace": str(source),
            "candidate_workspace": str(
                self.local.candidates_root / candidate_id / "workspace"
            ),
            "context": environment_context,
            "changed_paths": list(journal.get("changed_paths", [])),
        }
        try:
            response = _invoke_controller(
                self.command,
                self.timeout,
                "reconcile",
                envelope,
                cwd=source,
            )
        except OSError as error:
            return {"status": "unknown", "summary": str(error)}
        status = str(response.get("status", "unknown"))
        if status not in {"committed", "not_committed", "rolled_back", "unknown"}:
            status = "unknown"
        return {**response, "status": status}


@dataclass
class ExternalEnvironmentTransaction:
    command: list[str]
    timeout: int
    local: WorkspaceTransaction
    capabilities: set[str]
    session_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    controller_errors: list[str] = field(default_factory=list)

    @property
    def candidate_id(self) -> str:
        return self.local.candidate_id

    @property
    def candidate_workspace(self) -> Path:
        return self.local.candidate_workspace

    @property
    def transaction_root(self) -> Path:
        return self.local.transaction_root

    def diff(self) -> list[str]:
        response = self._call("inspect", {})
        self._merge_response(response)
        return self.local.diff()

    def promote(self, paths: list[str]) -> PromotionResult:
        try:
            prepared = self._call("prepare_commit", {"changed_paths": list(paths)})
            self._merge_response(prepared)
        except OSError as error:
            return PromotionResult(
                passed=False,
                changed_paths=list(paths),
                summary=f"External environment prepare_commit failed: {error}",
            )
        local_result = self.local.promote(paths)
        if not local_result.passed:
            self._best_effort("rollback", {"reason": local_result.summary})
            return local_result
        try:
            committed = self._call("commit", {"changed_paths": list(paths)})
            self._merge_response(committed)
        except OSError as error:
            restored = self.local.compensate_promotion()
            self._best_effort("rollback", {"reason": str(error)})
            return PromotionResult(
                passed=False,
                changed_paths=list(paths),
                summary=f"External environment commit failed: {error}",
                restored_after_failure=restored,
            )
        return PromotionResult(
            passed=True,
            changed_paths=list(paths),
            summary=(
                f"{local_result.summary}; external session {self.session_id} committed"
            ),
        )

    def rollback(self) -> None:
        self._best_effort("rollback", {"reason": "candidate rejected"})
        self.local.rollback()

    def close(self, *, preserve: bool = False) -> None:
        self._best_effort("close", {"preserve": preserve})
        self.local.close(preserve=preserve)

    def _call(self, operation: str, extra: dict[str, Any]) -> dict[str, Any]:
        envelope = {
            "protocol_version": ENVIRONMENT_ADAPTER_PROTOCOL_VERSION,
            "operation": operation,
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "source_workspace": str(self.local.source_workspace),
            "candidate_workspace": str(self.local.candidate_workspace),
            "context": self.context,
            **extra,
        }
        return _invoke_controller(
            self.command,
            self.timeout,
            operation,
            envelope,
            cwd=self.candidate_workspace,
        )

    def _merge_response(self, response: dict[str, Any]) -> None:
        update = response.get("context", {})
        if isinstance(update, dict):
            self.context.update(update)
        if self.session_id:
            self.context["session_id"] = self.session_id
        self.context["capabilities"] = sorted(self.capabilities)
        evidence = self.context.setdefault("evidence", [])
        for item in response.get("evidence", []):
            if isinstance(item, dict) and item not in evidence:
                evidence.append(item)
        summary = response.get("summary")
        if summary:
            self.context["last_summary"] = str(summary)

    def _best_effort(self, operation: str, extra: dict[str, Any]) -> None:
        try:
            response = self._call(operation, extra)
            self._merge_response(response)
        except OSError as error:
            self.controller_errors.append(f"{operation}: {error}")
            self.context["controller_errors"] = list(self.controller_errors)


def _invoke_controller(
    command: list[str],
    timeout: int,
    operation: str,
    envelope: dict[str, Any],
    *,
    cwd: Path,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [*command, "--operation", operation],
            cwd=cwd,
            input=json.dumps(envelope, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise EnvironmentAdapterError(
            f"environment operation {operation} timed out"
        ) from error
    response = _parse_json_object(completed.stdout)
    if completed.returncode != 0:
        raise EnvironmentAdapterError(
            f"environment operation {operation} exited {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    if response.get("ok") is not True:
        summary = response.get("summary") or "invalid/non-ok controller response"
        raise EnvironmentAdapterError(
            f"environment operation {operation} rejected: {summary}"
        )
    return response
