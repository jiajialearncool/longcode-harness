from __future__ import annotations

from pathlib import Path

from .backends import BackendResult
from .models import Subtask, TaskContract
from .verifier import run_checks


class RecipeBackend:
    """Execute only user-declared deterministic recipes for the E0 tier."""

    capabilities = frozenset({"cli", "filesystem", "deterministic_recipe"})

    def execute(
        self,
        workspace: Path,
        contract: TaskContract,
        subtask: Subtask,
    ) -> BackendResult:
        recipe_id = subtask.recipe_id
        commands = contract.deterministic_recipes.get(recipe_id or "", [])
        if not recipe_id or not commands:
            return BackendResult(
                ok=False,
                report={},
                stdout="",
                stderr="E0 routing requires a user-declared deterministic recipe",
                return_code=None,
                duration_seconds=0.0,
            )
        results = run_checks(workspace, commands, contract.command_timeout_seconds)
        passed = all(item.passed for item in results)
        stdout = "\n".join(
            f"$ {item.command}\n{item.stdout}" for item in results if item.stdout
        )
        stderr = "\n".join(
            f"$ {item.command}\n{item.stderr}" for item in results if item.stderr
        )
        if not passed and not stderr:
            stderr = "Failed deterministic recipes: " + ", ".join(
                item.command for item in results if not item.passed
            )
        return BackendResult(
            ok=passed,
            report={
                "summary": (
                    f"Executed deterministic recipe {recipe_id}"
                    if passed
                    else f"Deterministic recipe {recipe_id} failed"
                ),
                "claimed_complete": passed,
                "changed_files": [],
                "tests_run": list(commands),
                "remaining_risks": [] if passed else ["Recipe command failed"],
            },
            stdout=stdout,
            stderr=stderr,
            return_code=0 if passed else next(
                (item.exit_code for item in results if not item.passed),
                None,
            ),
            duration_seconds=round(sum(item.duration_seconds for item in results), 3),
        )
