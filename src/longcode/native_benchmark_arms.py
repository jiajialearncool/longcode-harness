from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .backends import CodexBackend
from .budget import BROKER_BUDGET_KEYS, RUNNER_BUDGET_KEYS
from .engine import LongCodeEngine
from .models import Subtask, TaskContract
from .routing import EvidenceDrivenManager
from .semantic_probes import SemanticContractVerifier
from .storage import RuntimeStore


DIRECT_BUDGET_KEYS = {"backend_timeout_seconds", "max_agent_calls", "max_rounds"}
LONGCODE_BUDGET_KEYS = {
    "max_rounds",
    "backend_timeout_seconds",
    "max_attempts",
    "command_timeout_seconds",
}


def run_direct_arm(
    *,
    codex: str = "codex",
    model: str | None = None,
    backend_timeout: int = 1800,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment, workspace, result_file, task, budget = _arm_inputs(environ)
    backend_timeout = _budget_int(
        budget, "backend_timeout_seconds", backend_timeout
    )
    max_calls = _budget_int(budget, "max_agent_calls", 1)
    # A direct run is intrinsically one round.  Treat a shared max-rounds
    # declaration as enforced when that upper bound permits the single turn.
    declared_max_rounds = _budget_int(budget, "max_rounds", 1)
    contract = _contract_from_task(task, budget)
    criterion = contract.acceptance_criteria[0]
    subtask = Subtask(
        id="DIRECT-0001",
        round=1,
        criterion_id=criterion.id,
        criterion="; ".join(
            item.description for item in contract.acceptance_criteria
        ),
        objective=contract.objective,
        constraints=list(contract.constraints),
        allowed_paths=list(contract.allowed_paths),
        forbidden_paths=list(contract.forbidden_paths),
        checks=list(contract.checks),
        attempt=1,
        max_attempts=1,
        executor_tier="direct",
    )
    proxy_requested = bool(environment.get("HARNESS_BUDGET_PROXY_URL"))
    ignore_user_config = proxy_requested or _environment_flag(
        environment, "HARNESS_IGNORE_USER_CONFIG"
    )
    externally_sandboxed = _environment_flag(
        environment, "HARNESS_OUTER_SANDBOXED"
    )
    result = CodexBackend(
        executable=codex,
        model=model,
        timeout=backend_timeout,
        ignore_user_config=ignore_user_config,
        reasoning_effort=environment.get("HARNESS_REASONING_EFFORT") or None,
        externally_sandboxed=externally_sandboxed,
    ).execute_direct(workspace, contract)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    claimed = bool(result.ok and result.report.get("claimed_complete", False))
    unsupported = sorted(
        set(budget) - DIRECT_BUDGET_KEYS - BROKER_BUDGET_KEYS - RUNNER_BUDGET_KEYS
    )
    broker_required_but_missing = bool(set(budget) & BROKER_BUDGET_KEYS) and not proxy_requested
    payload = {
        "claimed_completed": claimed,
        "recovered": False,
        "committed_progress_lost": None,
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "model_calls": 1,
        "token_metrics_available": result.token_metrics_available,
        "observed_models": [model] if model and result.token_metrics_available else [],
        "model_observation_source": "successful_codex_cli_turn",
        "budget_enforced": (
            not unsupported
            and not broker_required_but_missing
            and max_calls >= 1
            and declared_max_rounds >= 1
        ),
        "budget_enforcement": {
            "max_agent_calls": 1,
            "declared_max_agent_calls": max_calls,
            "max_rounds": 1,
            "declared_max_rounds": declared_max_rounds,
            "backend_timeout_seconds": backend_timeout,
            "unsupported_keys": unsupported,
            "broker_route_requested": proxy_requested,
            "broker_required_but_missing": broker_required_but_missing,
            "ignore_user_config": ignore_user_config,
            "externally_sandboxed": externally_sandboxed,
            "wall_timeout": "enforced by outer matched runner",
        },
        "direct": {
            "backend_ok": result.ok,
            "return_code": result.return_code,
            "report": result.report,
        },
    }
    _write_result(result_file, payload)
    return 0 if claimed else 1


def run_longcode_arm(
    *,
    codex: str = "codex",
    model: str | None = None,
    backend_timeout: int = 1800,
    max_rounds: int = 10,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment, workspace, result_file, task, budget = _arm_inputs(environ)
    backend_timeout = _budget_int(
        budget, "backend_timeout_seconds", backend_timeout
    )
    max_rounds = _budget_int(budget, "max_rounds", max_rounds)
    contract = _contract_from_task(task, budget)
    runtime = result_file.parent / "longcode-arm" / "runtime"
    store = RuntimeStore(runtime)
    store.initialize(contract, workspace)
    proxy_requested = bool(environment.get("HARNESS_BUDGET_PROXY_URL"))
    ignore_user_config = proxy_requested or _environment_flag(
        environment, "HARNESS_IGNORE_USER_CONFIG"
    )
    externally_sandboxed = _environment_flag(
        environment, "HARNESS_OUTER_SANDBOXED"
    )
    backend = CodexBackend(
        executable=codex,
        model=model,
        timeout=backend_timeout,
        ignore_user_config=ignore_user_config,
        reasoning_effort=environment.get("HARNESS_REASONING_EFFORT") or None,
        externally_sandboxed=externally_sandboxed,
    )
    state = LongCodeEngine(
        store,
        backend,
        auditor=backend,
        manager=EvidenceDrivenManager(backend, workspace),
        executor_tiers={"E1": backend, "E3": backend},
        verifier_plugins=[SemanticContractVerifier()],
    ).run(max_rounds=max_rounds)
    completed = state.status == "completed"
    faults = store.iter_faults()
    verified_criteria = {
        str(item.get("data", {}).get("criterion_id"))
        for item in store.iter_events()
        if item.get("type") == "criterion_verified"
        and item.get("data", {}).get("criterion_id")
    }
    committed_progress_lost = any(
        criterion_id not in state.criteria
        or state.criteria[criterion_id].status not in {"verified", "completed"}
        for criterion_id in verified_criteria
    )
    unsupported = sorted(
        set(budget) - LONGCODE_BUDGET_KEYS - BROKER_BUDGET_KEYS - RUNNER_BUDGET_KEYS
    )
    broker_required_but_missing = bool(set(budget) & BROKER_BUDGET_KEYS) and not proxy_requested
    usage = backend.usage_totals
    events = store.iter_events()
    control_level_counts = {
        level: sum(
            1
            for item in events
            if item.get("type") == "control_level_changed"
            and item.get("data", {}).get("to") == level
        )
        for level in ("L0", "L1", "L2", "L3")
    }
    escalation_reasons = [
        str(item.get("data", {}).get("reason"))
        for item in events
        if item.get("type") == "control_level_changed"
        and item.get("data", {}).get("to") in {"L2", "L3"}
    ]
    payload = {
        "claimed_completed": completed,
        "recovered": bool(faults) and completed,
        "committed_progress_lost": committed_progress_lost,
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "model_calls": usage["model_calls"],
        "token_metrics_available": (
            usage["model_calls"] > 0
            and usage["measured_model_calls"] == usage["model_calls"]
        ),
        "observed_models": (
            [model]
            if model
            and usage["model_calls"] > 0
            and usage["measured_model_calls"] == usage["model_calls"]
            else []
        ),
        "model_observation_source": "successful_codex_cli_turns",
        "budget_enforced": not unsupported and not broker_required_but_missing,
        "budget_enforcement": {
            "max_rounds": max_rounds,
            "backend_timeout_seconds": backend_timeout,
            "max_attempts": contract.max_attempts_per_criterion,
            "command_timeout_seconds": contract.command_timeout_seconds,
            "unsupported_keys": unsupported,
            "broker_route_requested": proxy_requested,
            "broker_required_but_missing": broker_required_but_missing,
            "ignore_user_config": ignore_user_config,
            "externally_sandboxed": externally_sandboxed,
            "wall_timeout": "enforced by outer matched runner",
        },
        "longcode": {
            "status": state.status,
            "rounds_run": state.round,
            "blocker": state.blocker,
            "fault_count": len(faults),
            "runtime": str(runtime),
            "final_evidence": list(state.final_evidence),
            "control_level_counts": control_level_counts,
            "deterministic_check_batches": sum(
                item.get("type") == "deterministic_verification" for item in events
            ),
            "auditor_calls": sum(
                item.get("type") == "audit_completed" for item in events
            ),
            "manager_calls": sum(
                item.get("type") == "manager_backend_completed" for item in events
            ),
            "escalation_reasons": escalation_reasons,
            "semantic_probe_batches": state.semantic_probe_batches,
            "repair_packets_created": state.repair_packets_created,
            "repair_attempts": state.repair_attempts,
            "repairs_succeeded": state.repairs_succeeded,
            "repair_conversion_rate": (
                state.repairs_succeeded / state.repair_attempts
                if state.repair_attempts
                else None
            ),
        },
    }
    _write_result(result_file, payload)
    return 0 if completed else 1


def _arm_inputs(
    environ: Mapping[str, str] | None,
) -> tuple[dict[str, str], Path, Path, dict[str, Any], dict[str, Any]]:
    environment = dict(os.environ)
    if environ is not None:
        environment.update(environ)
    workspace_value = environment.get("HARNESS_WORKSPACE", "").strip()
    result_value = environment.get("HARNESS_RESULT_FILE", "").strip()
    if not workspace_value or not result_value:
        raise ValueError("HARNESS_WORKSPACE and HARNESS_RESULT_FILE are required")
    workspace = Path(workspace_value).expanduser().resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"HARNESS_WORKSPACE is not a directory: {workspace}")
    result_file = Path(result_value).expanduser().resolve()
    task = _json_object(
        environment.get("HARNESS_TASK_JSON", "{}"), "HARNESS_TASK_JSON"
    )
    budget = _json_object(
        environment.get("HARNESS_BUDGET_JSON", "{}"), "HARNESS_BUDGET_JSON"
    )
    return environment, workspace, result_file, task, budget


def _environment_flag(environment: Mapping[str, str], key: str) -> bool:
    return str(environment.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def _contract_from_task(
    task: dict[str, Any], budget: dict[str, Any]
) -> TaskContract:
    objective = str(task.get("goal") or task.get("objective") or "").strip()
    if not objective:
        raise ValueError("benchmark task needs goal or objective")
    acceptance_value = task.get("acceptance") or task.get("acceptance_criteria")
    if not isinstance(acceptance_value, list) or not acceptance_value:
        raise ValueError("benchmark task needs a non-empty acceptance list")
    acceptance = [
        str(value).strip() for value in acceptance_value if str(value).strip()
    ]
    if not acceptance:
        raise ValueError("benchmark task acceptance list is empty")
    # A matched arm receives the whole benchmark task as one execution unit.
    # Keeping each bullet as a separate engine criterion would spend one fresh
    # Codex episode per bullet, while Direct and LH both receive the full task in
    # one episode/round. Preserve every bullet in one atomic criterion instead.
    atomic_acceptance = ["\n".join(f"- {item}" for item in acceptance)]
    contract = TaskContract.create(
        objective,
        atomic_acceptance,
        constraints=_string_list(task.get("constraints")),
        non_goals=_string_list(task.get("non_goals")),
        allowed_paths=_string_list(task.get("allowed_paths")) or ["**"],
        forbidden_paths=_string_list(task.get("forbidden_paths")) or [".git/**"],
        checks=_string_list(task.get("checks")) or ["true"],
        max_attempts=_budget_int(budget, "max_attempts", 2),
        command_timeout=_budget_int(budget, "command_timeout_seconds", 300),
    )
    profile = str(task.get("verification_profile") or "default").strip()
    contract.default_verification_profile = profile
    contract.acceptance_criteria = [
        replace(item, verification_profile=profile)
        for item in contract.acceptance_criteria
    ]
    return contract


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _json_object(raw: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _budget_int(budget: dict[str, Any], key: str, fallback: int) -> int:
    value = budget.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"benchmark budget {key} must be a positive integer")
    return value


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
