from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .models import CommandResult


MAX_CAPTURE_CHARS = 40_000


def run_checks(
    workspace: Path, commands: list[str], timeout_seconds: int
) -> list[CommandResult]:
    return [run_check(workspace, command, timeout_seconds) for command in commands]


def run_check(workspace: Path, command: str, timeout_seconds: int) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            executable="/bin/sh",
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return CommandResult(
            command=command,
            passed=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=_tail(completed.stdout),
            stderr=_tail(completed.stderr),
            duration_seconds=round(time.monotonic() - started, 3),
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        return CommandResult(
            command=command,
            passed=False,
            exit_code=None,
            stdout=_tail(stdout),
            stderr=_tail(stderr),
            duration_seconds=round(time.monotonic() - started, 3),
            timed_out=True,
        )


def _tail(value: str) -> str:
    if len(value) <= MAX_CAPTURE_CHARS:
        return value
    return "<truncated>\n" + value[-MAX_CAPTURE_CHARS:]
