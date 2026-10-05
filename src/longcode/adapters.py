from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .backends import BackendResult
from .models import Subtask, TaskContract
from .transactions import TransactionManager, WorkspaceTransaction


@runtime_checkable
class AgentAdapter(Protocol):
    """Provider-neutral execution boundary used by executor tiers."""

    capabilities: frozenset[str]

    def execute(
        self, workspace: Path, contract: TaskContract, subtask: Subtask
    ) -> BackendResult: ...


@runtime_checkable
class EnvironmentTransaction(Protocol):
    candidate_id: str
    candidate_workspace: Path
    context: dict[str, Any]

    def diff(self) -> list[str]: ...

    def promote(self, paths: list[str]): ...

    def rollback(self) -> None: ...

    def close(self, *, preserve: bool = False) -> None: ...


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Transactional boundary for local, container, VM, browser, or GUI environments."""

    capabilities: frozenset[str]

    def begin(self, workspace: Path, *, round_number: int) -> EnvironmentTransaction: ...


class LocalWorkspaceAdapter(TransactionManager):
    capabilities = frozenset({"cli", "filesystem", "transaction", "rollback"})


def adapter_capabilities(adapter: object) -> set[str]:
    values = getattr(adapter, "capabilities", ())
    return {str(value) for value in values}
