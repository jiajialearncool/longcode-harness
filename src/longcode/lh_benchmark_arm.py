from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .backends import _codex_jsonl_usage
from .budget import BROKER_BUDGET_KEYS, RUNNER_BUDGET_KEYS


SUPPORTED_BUDGET_KEYS = {
    "max_rounds",
    "manager_timeout_seconds",
    "executor_timeout_seconds",
    "auditor_timeout_seconds",
}


def run_lh_arm(
    *,
    executable: str = "lh-harness",
    agent: str = "codex",
    model: str | None = None,
    prompt_language: str = "en",
    max_rounds: int = 30,
    manager_timeout: int = 600,
    executor_timeout: int = 1800,
    auditor_timeout: int = 600,
    codex_mcp_config: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run the official LH CLI as one matched-benchmark arm."""
    environment = dict(os.environ)
    if environ is not None:
        environment.update(environ)
    workspace = _required_path(environment, "HARNESS_WORKSPACE", directory=True)
    result_file = _required_path(environment, "HARNESS_RESULT_FILE", directory=False)
    task = _json_environment(environment, "HARNESS_TASK_JSON", default={})
    budget = _json_environment(environment, "HARNESS_BUDGET_JSON", default={})
    if not isinstance(budget, dict):
        raise ValueError("HARNESS_BUDGET_JSON must contain an object")
    if isinstance(budget.get("max_rounds"), int):
        max_rounds = int(budget["max_rounds"])
    manager_timeout = _budget_int(
        budget, "manager_timeout_seconds", manager_timeout
    )
    executor_timeout = _budget_int(
        budget, "executor_timeout_seconds", executor_timeout
    )
    auditor_timeout = _budget_int(
        budget, "auditor_timeout_seconds", auditor_timeout
    )
    if max_rounds < 1:
        raise ValueError("LH max_rounds must be at least 1")

    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
    if resolved is None:
        raise FileNotFoundError(f"LH executable was not found: {executable}")

    arm_root = result_file.parent / "lh-arm"
    runs_root = arm_root / "runs"
    log_dir = arm_root / "logs"
    task_file = arm_root / "task.md"
    arm_root.mkdir(parents=True, exist_ok=True)
    task_file.write_text(_task_text(task), encoding="utf-8")
    run_id = "matched-" + _safe_name(environment.get("HARNESS_RUN_ID", "run"))
    reasoning_effort = environment.get("HARNESS_REASONING_EFFORT", "").strip()
    ignore_user_config = _environment_flag(
        environment, "HARNESS_IGNORE_USER_CONFIG"
    )
    codex_wrapper: Path | None = None
    if agent == "codex" and (reasoning_effort or ignore_user_config):
        codex_wrapper = _write_codex_wrapper(
            arm_root,
            environment=environment,
            reasoning_effort=reasoning_effort,
            ignore_user_config=ignore_user_config,
        )
        environment["LH_HARNESS_CODEX_BINARY"] = str(codex_wrapper)
    command = [
        resolved,
        "run",
        "--task",
        f"@{task_file}",
        "--agent",
        agent,
        "--workspace",
        str(workspace),
        "--runs-root",
        str(runs_root),
        "--run-id",
        run_id,
        "--log-dir",
        str(log_dir),
        "--max-rounds",
        str(max_rounds),
        "--manager-timeout",
        str(manager_timeout),
        "--gui-executor-timeout",
        str(executor_timeout),
        "--cli-executor-timeout",
        str(executor_timeout),
        "--auditor-timeout",
        str(auditor_timeout),
        "--prompt-language",
        prompt_language,
        "--no-dashboard",
    ]
    if model:
        command.extend(["--model", model])
    if codex_mcp_config:
        command.extend(
            ["--codex-mcp-config", str(Path(codex_mcp_config).expanduser().resolve())]
        )
    proxy_url = environment.get("HARNESS_BUDGET_PROXY_URL", "").strip()
    proxy_key_env = environment.get(
        "HARNESS_BUDGET_PROXY_KEY_ENV", "LONGCODE_BUDGET_PROXY_TOKEN"
    ).strip()
    if proxy_url:
        proxy_token = environment.get(proxy_key_env, "").strip()
        if not proxy_token:
            raise ValueError(f"budget proxy key environment variable is unset: {proxy_key_env}")
        command.extend(["--base-url", proxy_url, "--api-key", proxy_token])

    completed = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    report_path = log_dir / "report.json"
    report = _read_json(report_path)
    usage = _lh_codex_usage(log_dir) if agent == "codex" else _empty_usage()
    unsupported_budget_keys = sorted(
        set(budget) - SUPPORTED_BUDGET_KEYS - BROKER_BUDGET_KEYS - RUNNER_BUDGET_KEYS
    )
    broker_required_but_missing = bool(set(budget) & BROKER_BUDGET_KEYS) and not proxy_url
    payload = {
        "claimed_completed": bool(report.get("completion_satisfied", False)),
        "recovered": False,
        "committed_progress_lost": None,
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "model_calls": usage["model_calls"],
        "token_metrics_available": usage["token_metrics_available"],
        "observed_models": (
            [model] if model and usage["token_metrics_available"] else []
        ),
        "model_observation_source": "successful_codex_cli_turns",
        "budget_enforced": not unsupported_budget_keys and not broker_required_but_missing,
        "budget_enforcement": {
            "max_rounds": max_rounds,
            "manager_timeout_seconds": manager_timeout,
            "executor_timeout_seconds": executor_timeout,
            "auditor_timeout_seconds": auditor_timeout,
            "unsupported_keys": unsupported_budget_keys,
            "broker_route_requested": bool(proxy_url),
            "broker_required_but_missing": broker_required_but_missing,
            "reasoning_effort": reasoning_effort or None,
            "ignore_user_config": ignore_user_config,
            "wall_timeout": "enforced by outer matched runner",
        },
        "lh": {
            "status": report.get("status"),
            "completion_satisfied": report.get("completion_satisfied"),
            "rounds_run": report.get("rounds_run"),
            "max_rounds": report.get("max_rounds"),
            "abort_reason": report.get("abort_reason"),
            "report_path": str(report_path),
            "exit_code": completed.returncode,
            "codex_wrapper": str(codex_wrapper) if codex_wrapper else None,
            "codex_usage_files": usage["usage_files"],
            "codex_measured_calls": usage["measured_model_calls"],
            "role_call_counts": usage["role_call_counts"],
        },
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return completed.returncode
    return 0 if payload["claimed_completed"] else 1


def _required_path(
    environment: Mapping[str, str], key: str, *, directory: bool
) -> Path:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    path = Path(value).expanduser().resolve()
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{key} is not a directory: {path}")
    return path


def _json_environment(
    environment: Mapping[str, str], key: str, *, default: Any
) -> Any:
    raw = environment.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{key} is not valid JSON") from error


def _budget_int(budget: dict[str, Any], key: str, fallback: int) -> int:
    value = budget.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"benchmark budget {key} must be a positive integer")
    return value


def _task_text(task: Any) -> str:
    if isinstance(task, str):
        return task.strip() + "\n"
    if not isinstance(task, dict):
        raise ValueError("HARNESS_TASK_JSON must contain a string or object")
    goal = str(task.get("goal") or task.get("objective") or "").strip()
    if not goal:
        raise ValueError("benchmark task needs goal or objective")
    return (
        f"Goal:\n{goal}\n\n"
        "Public task contract (hidden checks are intentionally absent):\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
        + "\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_codex_wrapper(
    arm_root: Path,
    *,
    environment: Mapping[str, str],
    reasoning_effort: str,
    ignore_user_config: bool,
) -> Path:
    declared = environment.get("LH_HARNESS_CODEX_BINARY", "").strip()
    real_binary = declared or shutil.which("codex")
    if not real_binary:
        desktop = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        if desktop.is_file():
            real_binary = str(desktop)
    if not real_binary:
        raise FileNotFoundError("Codex CLI was not found for the LH benchmark wrapper")
    real_path = Path(real_binary).expanduser().resolve()
    if not real_path.is_file():
        raise FileNotFoundError(f"Codex CLI was not found: {real_path}")
    injected: list[str] = []
    if ignore_user_config:
        injected.append("--ignore-user-config")
    if reasoning_effort:
        injected.extend(
            ["--config", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
        )
    quoted_injected = " ".join(shlex.quote(item) for item in injected)
    wrapper = arm_root / "codex-matched-wrapper.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "$#" -gt 0 ] && [ "$1" = "exec" ]; then\n'
        "  shift\n"
        f"  exec {shlex.quote(str(real_path))} exec {quoted_injected} \"$@\"\n"
        "fi\n"
        f"exec {shlex.quote(str(real_path))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return wrapper


def _lh_codex_usage(log_dir: Path) -> dict[str, Any]:
    totals = _empty_usage()
    usage_files: list[str] = []
    trajectories = sorted(
        (log_dir / "role_orchestration" / "rounds").glob(
            "round_*/*_raw_trajectory.jsonl"
        )
    )
    for path in trajectories:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if '"type":"turn.started"' not in raw and '"type": "turn.started"' not in raw:
            continue
        totals["model_calls"] += 1
        role = path.name.split("_raw_trajectory", 1)[0]
        totals["role_call_counts"][role] = (
            totals["role_call_counts"].get(role, 0) + 1
        )
        parsed = _codex_jsonl_usage(raw)
        if not parsed["token_metrics_available"]:
            continue
        totals["measured_model_calls"] += 1
        usage_files.append(str(path))
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            totals[key] += int(parsed[key])
    totals["token_metrics_available"] = bool(
        totals["model_calls"] > 0
        and totals["measured_model_calls"] == totals["model_calls"]
    )
    totals["usage_files"] = usage_files
    return totals


def _empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "model_calls": 0,
        "measured_model_calls": 0,
        "token_metrics_available": False,
        "usage_files": [],
        "role_call_counts": {},
    }


def _environment_flag(environment: Mapping[str, str], key: str) -> bool:
    return str(environment.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    return result[:100] or "run"
