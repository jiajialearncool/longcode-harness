"""Run accepted checks in a disposable copy, never in the business workspace."""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from .agent_runtime import Cancelled
from .agent_tools import Tools
from .models import CommandResult


def make_check_runner(home, cancel, emit):
    def run(workspace, commands, timeout):
        results = []
        with tempfile.TemporaryDirectory(prefix="longcode-check-") as directory:
            copy = Path(directory).resolve() / "project"
            def ignore(folder, names):
                cancel.check()
                return [name for name in names if name in {".git", ".longcode", ".codex", ".agents", ".env"}
                        or name.startswith('.env.')]
            # Retain symlinks as symlinks: the OS sandbox refuses their external targets.
            shutil.copytree(workspace, copy, symlinks=True, ignore=ignore)
            tools = Tools(copy, role="verifier", cancel=cancel, home=home, emit=emit)
            for command in commands:
                cancel.check()
                begin = time.monotonic()
                try:
                    value = tools.command(command, approved=True, timeout=timeout)
                    result = CommandResult(command, value["exit_code"] == 0, value["exit_code"],
                        value["stdout"], value["stderr"], time.monotonic() - begin)
                except (OSError, RuntimeError, TimeoutError) as exc:
                    if isinstance(exc, Cancelled):
                        raise
                    result = CommandResult(command, False, None, "", str(exc), time.monotonic() - begin,
                                           timed_out=isinstance(exc, TimeoutError))
                emit("check_finished", result.to_dict())
                results.append(result)
        return results
    return run
