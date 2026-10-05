from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .scope import DEFAULT_IGNORES, changed_paths, runtime_ignore_for, snapshot_tree


class UnsafeCandidatePathError(OSError):
    pass


@dataclass
class PromotionResult:
    passed: bool
    changed_paths: list[str]
    summary: str
    restored_after_failure: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "changed_paths": self.changed_paths,
            "summary": self.summary,
            "restored_after_failure": self.restored_after_failure,
        }


@dataclass
class WorkspaceTransaction:
    candidate_id: str
    source_workspace: Path
    candidate_workspace: Path
    transaction_root: Path
    base_snapshot: dict[str, str]
    status: str = "prepared"
    promoted_paths: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    promotion_backup_root: Path | None = None
    promotion_missing_targets: set[str] = field(default_factory=set)

    def diff(self) -> list[str]:
        _ensure_symlinks_contained(self.candidate_workspace)
        after = snapshot_tree(self.candidate_workspace)
        return changed_paths(self.base_snapshot, after)

    def promote(self, paths: list[str]) -> PromotionResult:
        validated = [_validate_relative(path) for path in paths]
        _ensure_symlinks_contained(self.candidate_workspace)
        current_source = snapshot_tree(self.source_workspace)
        conflicts = [
            path
            for path in validated
            if current_source.get(path) != self.base_snapshot.get(path)
        ]
        if conflicts:
            self.status = "promotion_conflict"
            return PromotionResult(
                passed=False,
                changed_paths=validated,
                summary=(
                    "Source workspace changed after candidate creation: "
                    + ", ".join(conflicts)
                ),
            )
        backup_root = self.transaction_root / "promotion-backup"
        missing_targets: set[str] = set()
        backup_root.mkdir(parents=True, exist_ok=True)
        self.promotion_backup_root = backup_root
        self.promotion_missing_targets = missing_targets
        try:
            for path in validated:
                target = self.source_workspace / path
                backup = backup_root / path
                if target.is_symlink():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.symlink_to(os.readlink(target))
                elif target.is_file():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                elif target.is_dir():
                    shutil.copytree(target, backup, symlinks=True)
                else:
                    missing_targets.add(path)

            for path in validated:
                candidate = self.candidate_workspace / path
                target = self.source_workspace / path
                _replace_target(candidate, target)
        except OSError as error:
            restored = _restore_targets(
                self.source_workspace, backup_root, validated, missing_targets
            )
            self.status = "promotion_failed"
            return PromotionResult(
                passed=False,
                changed_paths=validated,
                summary=f"Promotion failed: {error}",
                restored_after_failure=restored,
            )
        self.status = "promoted"
        self.promoted_paths = validated
        return PromotionResult(
            passed=True,
            changed_paths=validated,
            summary=f"Promoted {len(validated)} validated paths",
        )

    def compensate_promotion(self) -> bool:
        if self.status != "promoted" or self.promotion_backup_root is None:
            return False
        restored = _restore_targets(
            self.source_workspace,
            self.promotion_backup_root,
            list(self.promoted_paths),
            set(self.promotion_missing_targets),
        )
        self.status = "compensated" if restored else "compensation_failed"
        return restored

    def rollback(self) -> None:
        self.status = "rolled_back"
        if self.transaction_root.exists():
            shutil.rmtree(self.transaction_root)

    def close(self, *, preserve: bool = False) -> None:
        if not preserve and self.transaction_root.exists():
            shutil.rmtree(self.transaction_root)


class TransactionManager:
    """Portable copy-on-candidate implementation of the transactional environment protocol."""

    def __init__(self, runtime_root: Path | str):
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.candidates_root = self.runtime_root / "candidates"

    def begin(self, source_workspace: Path | str, *, round_number: int) -> WorkspaceTransaction:
        return self._begin(source_workspace, round_number=round_number, seed_workspace=None)

    def begin_repair(
        self,
        source_workspace: Path | str,
        *,
        seed_workspace: Path | str,
        round_number: int,
    ) -> WorkspaceTransaction:
        """Create a new candidate from a failed candidate while retaining the trusted base.

        The candidate contains the prior implementation, but diff/promotion conflict checks
        remain anchored to the formal workspace. This makes repair incremental without
        promoting an unverified change.
        """

        return self._begin(
            source_workspace,
            round_number=round_number,
            seed_workspace=seed_workspace,
        )

    def _begin(
        self,
        source_workspace: Path | str,
        *,
        round_number: int,
        seed_workspace: Path | str | None,
    ) -> WorkspaceTransaction:
        source = Path(source_workspace).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"workspace does not exist: {source}")
        seed = (
            Path(seed_workspace).expanduser().resolve()
            if seed_workspace is not None
            else source
        )
        if not seed.is_dir():
            raise NotADirectoryError(f"repair seed workspace does not exist: {seed}")
        candidate_id = f"CAND-{round_number:04d}-{uuid4().hex[:10]}"
        transaction_root = self.candidates_root / candidate_id
        candidate = transaction_root / "workspace"
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
            shutil.copytree(seed, candidate, symlinks=True, ignore=ignore)
            _ensure_symlinks_contained(candidate)
        except OSError:
            shutil.rmtree(transaction_root, ignore_errors=True)
            raise
        return WorkspaceTransaction(
            candidate_id=candidate_id,
            source_workspace=source,
            candidate_workspace=candidate,
            transaction_root=transaction_root,
            base_snapshot=snapshot_tree(source),
            context={
                "repair_seed": str(seed)
            } if seed_workspace is not None else {},
        )


def _ensure_symlinks_contained(root: Path) -> None:
    resolved_root = root.resolve()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirs, *files]:
            path = base / name
            if not path.is_symlink():
                continue
            resolved = path.resolve(strict=False)
            try:
                resolved.relative_to(resolved_root)
            except ValueError as error:
                relative = path.relative_to(root).as_posix()
                raise UnsafeCandidatePathError(
                    f"candidate symlink escapes workspace: {relative} -> {os.readlink(path)}"
                ) from error


def _validate_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or not path or ".." in pure.parts:
        raise ValueError(f"unsafe candidate path: {path}")
    return pure.as_posix()


def _replace_target(candidate: Path, target: Path) -> None:
    if candidate.is_symlink():
        _remove_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.readlink(candidate))
    elif candidate.is_file():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.is_symlink():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.longcode-promote.tmp")
        shutil.copy2(candidate, temporary)
        os.replace(temporary, target)
    elif candidate.is_dir():
        _remove_target(target)
        shutil.copytree(candidate, target, symlinks=True)
    else:
        _remove_target(target)


def _remove_target(target: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    elif target.is_dir():
        shutil.rmtree(target)


def _restore_targets(
    workspace: Path,
    backup_root: Path,
    paths: list[str],
    missing_targets: set[str],
) -> bool:
    try:
        for path in paths:
            target = workspace / path
            backup = backup_root / path
            _remove_target(target)
            if path in missing_targets:
                continue
            if backup.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.readlink(backup))
            elif backup.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
            elif backup.is_dir():
                shutil.copytree(backup, target, symlinks=True)
        return True
    except OSError:
        return False
