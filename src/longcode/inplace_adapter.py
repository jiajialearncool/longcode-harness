from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .scope import DEFAULT_IGNORES, changed_paths, runtime_ignore_for, snapshot_tree
from .transactions import PromotionResult


@dataclass
class InPlaceWorkspaceTransaction:
    """Recoverable workspace checkpoint that executes at the benchmark's real cwd."""

    candidate_id: str
    candidate_workspace: Path
    backup_workspace: Path
    transaction_root: Path
    base_snapshot: dict[str, str]
    context: dict[str, object] = field(default_factory=dict)
    status: str = "prepared"

    def diff(self) -> list[str]:
        return changed_paths(self.base_snapshot, snapshot_tree(self.candidate_workspace))

    def promote(self, paths: list[str]) -> PromotionResult:
        self.status = "promoted"
        return PromotionResult(
            passed=True,
            changed_paths=list(paths),
            summary=f"Validated {len(paths)} in-place workspace paths",
        )

    def rollback(self) -> None:
        current = snapshot_tree(self.candidate_workspace)
        paths = changed_paths(self.base_snapshot, current)
        for relative in paths:
            target = self.candidate_workspace / relative
            backup = self.backup_workspace / relative
            _remove(target)
            if backup.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.readlink(backup))
            elif backup.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
            elif backup.is_dir():
                shutil.copytree(backup, target, symlinks=True)
        self.status = "rolled_back"

    def close(self, *, preserve: bool = False) -> None:
        if not preserve and self.transaction_root.exists():
            shutil.rmtree(self.transaction_root)


class InPlaceWorkspaceAdapter:
    """Checkpoint files while keeping the real task workspace as the agent cwd.

    Terminal-Bench instructions often refer to absolute workspace paths. A
    detached copy would silently change task semantics, so this adapter backs up
    the workspace but lets the agent operate in place.
    """

    capabilities = frozenset({"cli", "filesystem", "transaction", "rollback", "in_place"})

    def __init__(self, runtime_root: Path | str):
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.candidates_root = self.runtime_root / "candidates"

    def begin(
        self, source_workspace: Path | str, *, round_number: int
    ) -> InPlaceWorkspaceTransaction:
        source = Path(source_workspace).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"workspace does not exist: {source}")
        candidate_id = f"INPLACE-{round_number:04d}-{uuid4().hex[:10]}"
        transaction_root = self.candidates_root / candidate_id
        backup = transaction_root / "workspace-backup"
        transaction_root.mkdir(parents=True, exist_ok=False)
        ignore_names = set(DEFAULT_IGNORES)
        ignore_names.update(runtime_ignore_for(source, self.runtime_root))

        def ignore(directory: str, names: list[str]) -> set[str]:
            base = Path(directory)
            try:
                relative = base.resolve().relative_to(source).as_posix()
            except ValueError:
                relative = ""
            ignored: set[str] = set()
            for name in names:
                item = (Path(relative) / name).as_posix().lstrip("./")
                if name in ignore_names or any(
                    item == value or item.startswith(value.rstrip("/") + "/")
                    for value in ignore_names
                ):
                    ignored.add(name)
            return ignored

        try:
            shutil.copytree(source, backup, symlinks=True, ignore=ignore)
        except OSError:
            shutil.rmtree(transaction_root, ignore_errors=True)
            raise
        return InPlaceWorkspaceTransaction(
            candidate_id=candidate_id,
            candidate_workspace=source,
            backup_workspace=backup,
            transaction_root=transaction_root,
            base_snapshot=snapshot_tree(source),
        )


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
