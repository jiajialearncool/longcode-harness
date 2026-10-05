from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_IGNORES = {
    ".git",
    ".longcode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}


@dataclass(frozen=True)
class ScopeResult:
    passed: bool
    changed_paths: list[str]
    violations: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "changed_paths": self.changed_paths,
            "violations": self.violations,
        }


def snapshot_tree(root: Path, *, extra_ignores: set[str] | None = None) -> dict[str, str]:
    root = root.resolve()
    ignores = DEFAULT_IGNORES | set(extra_ignores or set())
    result: dict[str, str] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        relative_dir = base.relative_to(root)
        dirs[:] = [
            name
            for name in dirs
            if not _ignored((relative_dir / name).as_posix(), name, ignores)
        ]
        for name in files:
            relative = (relative_dir / name).as_posix()
            if _ignored(relative, name, ignores):
                continue
            path = base / name
            try:
                if path.is_symlink():
                    payload = f"symlink:{os.readlink(path)}".encode()
                else:
                    payload = path.read_bytes()
                result[relative] = hashlib.sha256(payload).hexdigest()
            except (FileNotFoundError, PermissionError, OSError):
                result[relative] = "<unreadable>"
    return result


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    all_paths = set(before) | set(after)
    return sorted(path for path in all_paths if before.get(path) != after.get(path))


def check_scope(
    paths: list[str], allowed_patterns: list[str], forbidden_patterns: list[str]
) -> ScopeResult:
    violations: list[str] = []
    for path in paths:
        if any(path_matches(path, pattern) for pattern in forbidden_patterns):
            violations.append(f"forbidden:{path}")
            continue
        if not any(path_matches(path, pattern) for pattern in allowed_patterns):
            violations.append(f"outside-allowed-scope:{path}")
    return ScopeResult(passed=not violations, changed_paths=paths, violations=violations)


def path_matches(path: str, pattern: str) -> bool:
    path = path.lstrip("./")
    pattern = pattern.strip().lstrip("./")
    if not pattern:
        return False
    if pattern in {"*", "**"}:
        return True
    if pattern.endswith("/"):
        return path == pattern[:-1] or path.startswith(pattern)
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return fnmatch.fnmatchcase(path, pattern) or PurePosixPath(path).match(pattern)


def runtime_ignore_for(workspace: Path, runtime: Path) -> set[str]:
    try:
        relative = runtime.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return set()
    return {relative, relative.split("/", 1)[0]}


def _ignored(relative: str, name: str, ignores: set[str]) -> bool:
    return name in ignores or any(
        relative == item or relative.startswith(item.rstrip("/") + "/") for item in ignores
    )
