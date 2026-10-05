from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import socket
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .budget import (
    BROKER_BUDGET_KEYS,
    BudgetBroker,
    BudgetLimits,
    broker_result,
)
from .backends import _codex_jsonl_usage
from .evaluation import EvaluationRecord
from .lh_benchmark_arm import _lh_codex_usage
from .scope import snapshot_tree
from .verifier import run_checks


BENCHMARK_MANIFEST_VERSION = 1
REQUIRED_ARMS = {"direct", "lh", "longcode"}
CHATGPT_SUBSCRIPTION_AUTH = "chatgpt_subscription"


def _apply_timeout_override(
    manifest: dict[str, Any], timeout_seconds: int | None
) -> dict[str, Any]:
    if timeout_seconds is None:
        return manifest
    if isinstance(timeout_seconds, bool) or timeout_seconds < 1:
        raise ValueError("benchmark timeout override must be a positive integer")
    return {
        **manifest,
        "timeout_seconds": timeout_seconds,
        "budget": {
            **dict(manifest.get("budget", {})),
            "max_wall_seconds": timeout_seconds,
        },
    }


def benchmark_preflight(
    manifest_path: Path | str, *, timeout_seconds: int | None = None
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        manifest = _apply_timeout_override(
            json.loads(manifest_file.read_text(encoding="utf-8")),
            timeout_seconds,
        )
        _validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        add("manifest_valid", False, str(error))
        return {
            "ready": False,
            "manifest": str(manifest_file),
            "checks": checks,
            "blocking_checks": ["manifest_valid"],
        }
    add("manifest_valid", True, "schema and matched-condition declarations are valid")
    outer_errors: list[str] = []
    for arm, config in manifest["arms"].items():
        try:
            _command_parts(config["command"], base=manifest_file.parent)
        except (ValueError, FileNotFoundError) as error:
            outer_errors.append(f"{arm}: {error}")
    add("outer_arm_commands_available", not outer_errors, outer_errors or "all found")

    raw_commands = {
        arm: (
            shlex.split(config["command"])
            if isinstance(config["command"], str)
            else list(config["command"])
        )
        for arm, config in manifest["arms"].items()
    }
    codex_paths: list[str] = []
    codex_errors: list[str] = []
    for arm in ("direct", "longcode"):
        declared = _flag_value(raw_commands[arm], "--codex") or "codex"
        resolved = shutil.which(declared)
        if resolved is None and Path(declared).expanduser().is_file():
            resolved = str(Path(declared).expanduser().resolve())
        if resolved is None:
            codex_errors.append(f"{arm}: {declared} not found")
        else:
            codex_paths.append(str(Path(resolved).resolve()))
    same_codex = not codex_errors and len(set(codex_paths)) == 1
    detail: dict[str, Any] = {"paths": sorted(set(codex_paths)), "errors": codex_errors}
    declared_tools = manifest.get("matched_conditions", {}).get("tools", {})
    expected_hash = (
        declared_tools.get("codex_sha256")
        if isinstance(declared_tools, dict)
        else None
    )
    pinned_binary = (
        declared_tools.get("codex_binary")
        if isinstance(declared_tools, dict)
        else None
    )
    if pinned_binary:
        pinned_path = Path(str(pinned_binary)).expanduser().resolve()
        detail["declared_binary"] = str(pinned_path)
        if not pinned_path.is_file():
            codex_errors.append(f"matched_conditions: {pinned_path} not found")
            same_codex = False
        elif codex_paths and str(pinned_path) != codex_paths[0]:
            codex_errors.append(
                "matched_conditions codex_binary differs from the direct/longcode binary"
            )
            same_codex = False
    if same_codex:
        observed_hash = _file_sha256(Path(codex_paths[0]))
        detail["sha256"] = observed_hash
        detail["declared_sha256"] = expected_hash
        same_codex = expected_hash in (None, observed_hash)
    add("same_codex_binary", same_codex, detail)

    lh_declared = _flag_value(raw_commands["lh"], "--lh-harness") or "lh-harness"
    lh_resolved = shutil.which(lh_declared)
    if lh_resolved is None and Path(lh_declared).expanduser().is_file():
        lh_resolved = str(Path(lh_declared).expanduser().resolve())
    add(
        "lh_harness_available",
        lh_resolved is not None,
        str(Path(lh_resolved).resolve()) if lh_resolved else f"not found: {lh_declared}",
    )

    authentication = manifest.get("authentication", {})
    auth_mode = (
        str(authentication.get("mode", "")).strip()
        if isinstance(authentication, dict)
        else ""
    )
    if auth_mode == CHATGPT_SUBSCRIPTION_AUTH:
        login_binary = str(pinned_binary or (codex_paths[0] if codex_paths else "codex"))
        login_environment = os.environ.copy()
        login_environment.update(_process_environment(manifest))
        for key in authentication.get("strip_environment_keys", []):
            login_environment.pop(str(key), None)
        try:
            login = subprocess.run(
                [login_binary, "login", "status"],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
                env=login_environment,
            )
            login_output = ((login.stdout or "") + (login.stderr or "")).strip()
            login_ready = login.returncode == 0 and "Logged in using ChatGPT" in login_output
        except (OSError, subprocess.TimeoutExpired) as error:
            login_ready = False
            login_output = str(error)
        add(
            "chatgpt_subscription_login_available",
            login_ready,
            "Logged in using ChatGPT" if login_ready else login_output[-500:],
        )
        add(
            "api_credentials_excluded",
            bool(authentication.get("strip_environment_keys")),
            sorted(str(key) for key in authentication.get("strip_environment_keys", [])),
        )

    provider = manifest.get("provider_proxy")
    key_env = provider.get("api_key_env") if isinstance(provider, dict) else None
    key_required = bool(key_env)
    key_ready = not key_required or bool(os.environ.get(str(key_env)))
    add(
        "provider_api_key_available",
        key_ready,
        (
            f"{key_env} is {'set' if key_ready else 'unset'}"
            if key_env
            else "no API key required by the declared provider"
        ),
    )
    loopback_required = bool(
        isinstance(provider, dict) and provider.get("require_loopback", False)
    )
    loopback_ready = bool(
        not loopback_required
        or _is_loopback_provider_url(str(provider.get("upstream_base_url", "")))
    )
    add(
        "provider_upstream_loopback_only",
        loopback_ready,
        (
            provider.get("upstream_base_url")
            if isinstance(provider, dict)
            else "no provider_proxy declared"
        ),
    )
    if loopback_required and loopback_ready:
        provider_ready, provider_detail = _probe_local_provider(
            str(provider["upstream_base_url"]),
            expected_model=str(manifest.get("matched_conditions", {}).get("model", "")),
        )
        add("local_model_server_ready", provider_ready, provider_detail)
    socket_ready = False
    socket_detail = "not checked"
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        socket_ready = True
        socket_detail = "loopback bind succeeded"
    except OSError as error:
        socket_detail = str(error)
    finally:
        probe.close()
    add("local_budget_broker_socket", socket_ready, socket_detail)

    fixture_errors: list[str] = []
    snapshot_ids: dict[str, str] = {}
    hidden_roots: set[Path] = set()
    for case in manifest["cases"]:
        try:
            source = _resolve_path(manifest_file.parent, case["source"])
            hidden_root = _resolve_hidden_root(
                manifest_file.parent, case, source=source
            )
            if hidden_root is not None:
                hidden_roots.add(hidden_root)
            snapshot_ids[str(case["id"])] = _snapshot_id(source)
        except (ValueError, NotADirectoryError) as error:
            fixture_errors.append(f"{case.get('id')}: {error}")
    add(
        "fixtures_and_hidden_paths_separate",
        not fixture_errors,
        fixture_errors or f"{len(snapshot_ids)} source snapshots resolved",
    )
    isolation_passed, isolation_detail = _preflight_arm_isolation(
        manifest,
        base=manifest_file.parent,
        hidden_roots=hidden_roots,
    )
    add(
        "hidden_checks_inaccessible_to_arms",
        isolation_passed and not fixture_errors,
        isolation_detail,
    )
    budget = manifest.get("budget", {})
    strict_token_budget = bool(set(budget) & BROKER_BUDGET_KEYS) and provider is not None
    subscription_execution_budget = (
        auth_mode == CHATGPT_SUBSCRIPTION_AUTH
        and isinstance(budget.get("max_wall_seconds"), int)
        and int(budget["max_wall_seconds"]) > 0
        and isinstance(budget.get("max_rounds"), int)
        and int(budget["max_rounds"]) > 0
        and not bool(set(budget) & BROKER_BUDGET_KEYS)
    )
    strict_budget = strict_token_budget or subscription_execution_budget
    add(
        "strict_budget_declared",
        strict_budget,
        (
            {
                "mode": "provider_proxy_token_cap",
                **{
                    key: budget.get(key)
                    for key in sorted(BROKER_BUDGET_KEYS)
                    if key in budget
                },
            }
            if strict_token_budget
            else {
                "mode": "subscription_wall_and_round_cap",
                "max_wall_seconds": budget.get("max_wall_seconds"),
                "max_rounds": budget.get("max_rounds"),
                "token_budget": "post-hoc observation; OAuth traffic is not proxy-capped",
            }
        ),
    )
    planned_runs = (
        len(manifest["cases"])
        * int(manifest.get("repetitions", 1))
        * len(manifest["arms"])
    )
    blocking = [item["name"] for item in checks if not item["passed"]]
    return {
        "ready": not blocking,
        "manifest": str(manifest_file),
        "task_count": len(manifest["cases"]),
        "repetitions": int(manifest.get("repetitions", 1)),
        "planned_arm_runs": planned_runs,
        "checks": checks,
        "blocking_checks": blocking,
    }


def run_benchmark(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    seed: int = 0,
    preserve_runs: Path | str | None = None,
    timeout_seconds: int | None = None,
    only_runs: set[tuple[str, str]] | None = None,
) -> list[EvaluationRecord]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _apply_timeout_override(
        json.loads(manifest_file.read_text(encoding="utf-8")),
        timeout_seconds,
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    _validate_manifest(manifest)
    selected_runs = set(only_runs) if only_runs is not None else None
    if selected_runs is not None:
        available_runs = {
            (str(case["id"]), arm)
            for case in manifest["cases"]
            for arm in manifest["arms"]
        }
        unknown_runs = selected_runs - available_runs
        if unknown_runs:
            rendered = ", ".join(
                f"{case_id}:{arm}" for case_id, arm in sorted(unknown_runs)
            )
            raise ValueError(f"unknown benchmark run selector(s): {rendered}")
        if not selected_runs:
            raise ValueError("benchmark run selection cannot be empty")
    for config in manifest["arms"].values():
        _command_parts(config["command"], base=manifest_file.parent)
    for case in manifest["cases"]:
        source = _resolve_path(manifest_file.parent, case["source"])
        _resolve_hidden_root(manifest_file.parent, case, source=source)
    isolation_prefix = _arm_isolation_prefix(
        manifest.get("arm_isolation"), base=manifest_file.parent
    )
    hidden_roots = {
        root
        for case in manifest["cases"]
        if (
            root := _resolve_hidden_root(
                manifest_file.parent,
                case,
                source=_resolve_path(manifest_file.parent, case["source"]),
            )
        )
        is not None
    }
    if isolation_prefix is not None:
        isolation_passed, isolation_detail = _probe_hidden_read_denied(
            isolation_prefix, hidden_roots
        )
        if not isolation_passed:
            raise RuntimeError(
                "arm isolation did not protect hidden checks: " + isolation_detail
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_root = output.with_suffix(output.suffix + ".evidence")
    if evidence_root.exists():
        raise FileExistsError(f"benchmark evidence directory already exists: {evidence_root}")
    evidence_root.mkdir(parents=True, exist_ok=True)
    preserve_root = Path(preserve_runs).expanduser().resolve() if preserve_runs else None
    if preserve_root:
        preserve_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[EvaluationRecord] = []
    repetitions = int(manifest.get("repetitions", 1))
    default_timeout = int(manifest.get("timeout_seconds", 1800))
    check_timeout = int(manifest.get("hidden_check_timeout_seconds", 300))
    budget = dict(manifest.get("budget", {}))
    provider_proxy = manifest.get("provider_proxy")
    matched_conditions = dict(manifest.get("matched_conditions", {}))
    authentication = dict(manifest.get("authentication", {}))
    process_environment = _process_environment(manifest)
    condition_id = (
        hashlib.sha256(
            json.dumps(
                matched_conditions, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if matched_conditions
        else None
    )

    frozen_temporary = tempfile.TemporaryDirectory(prefix="longcode-benchmark-baseline-")
    frozen_root = Path(frozen_temporary.name)
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            source = _resolve_path(manifest_file.parent, case["source"])
            hidden_root = _resolve_hidden_root(
                manifest_file.parent,
                case,
                source=source,
            )
            frozen_source = frozen_root / f"case-{case_index:04d}"
            shutil.copytree(source, frozen_source, symlinks=True)
            snapshot_id = _snapshot_id(frozen_source)
            for repetition in range(1, repetitions + 1):
                arms = list(manifest["arms"])
                rng.shuffle(arms)
                for arm in arms:
                    if selected_runs is not None and (str(case["id"]), arm) not in selected_runs:
                        continue
                    record = _run_one(
                        manifest_file=manifest_file,
                        case=case,
                        arm=arm,
                        arm_config=manifest["arms"][arm],
                        source=frozen_source,
                        hidden_root=hidden_root,
                        snapshot_id=snapshot_id,
                        repetition=repetition,
                        timeout=(
                            min(
                                int(case.get("timeout_seconds", default_timeout)),
                                int(budget["max_wall_seconds"]),
                            )
                            if "max_wall_seconds" in budget
                            else int(case.get("timeout_seconds", default_timeout))
                        ),
                        check_timeout=check_timeout,
                        budget=budget,
                        provider_proxy=provider_proxy,
                        matched_conditions=matched_conditions,
                        authentication=authentication,
                        process_environment=process_environment,
                        isolation_prefix=isolation_prefix,
                        condition_id=condition_id,
                        evidence_root=evidence_root,
                        preserve_root=preserve_root,
                    )
                    records.append(record)
                    _append_jsonl(output, record.__dict__)
    finally:
        frozen_temporary.cleanup()
    return records


def _run_one(
    *,
    manifest_file: Path,
    case: dict[str, Any],
    arm: str,
    arm_config: dict[str, Any],
    source: Path,
    hidden_root: Path | None,
    snapshot_id: str,
    repetition: int,
    timeout: int,
    check_timeout: int,
    budget: dict[str, Any],
    provider_proxy: dict[str, Any] | None,
    matched_conditions: dict[str, Any],
    authentication: dict[str, Any],
    process_environment: dict[str, str],
    isolation_prefix: list[str] | None,
    condition_id: str | None,
    evidence_root: Path,
    preserve_root: Path | None,
) -> EvaluationRecord:
    run_id = f"{case['id']}#r{repetition}"
    safe_run = _safe_name(run_id)
    evidence_dir = evidence_root / safe_run / arm
    evidence_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=f"longcode-bench-{arm}-")
    run_root = Path(temporary.name)
    workspace = run_root / "workspace"
    shutil.copytree(source, workspace, symlinks=True)
    result_file = run_root / "arm-result.json"
    arm_command = _command_parts(arm_config["command"], base=manifest_file.parent)
    command = [*(isolation_prefix or []), *arm_command]
    environment = os.environ.copy()
    environment.update(process_environment)
    if isinstance(provider_proxy, dict):
        for key in provider_proxy.get("strip_environment_keys", []):
            environment.pop(str(key), None)
    for key in authentication.get("strip_environment_keys", []):
        environment.pop(str(key), None)
    environment.update(
        {
            "HARNESS_ARM": arm,
            "HARNESS_CASE_ID": str(case["id"]),
            "HARNESS_RUN_ID": run_id,
            "HARNESS_REPETITION": str(repetition),
            "HARNESS_WORKSPACE": str(workspace),
            "HARNESS_RESULT_FILE": str(result_file),
            "HARNESS_TASK_JSON": json.dumps(case.get("task", {}), ensure_ascii=False),
            "HARNESS_BUDGET_JSON": json.dumps(budget, ensure_ascii=False),
            "HARNESS_SNAPSHOT_ID": snapshot_id,
            "HARNESS_REASONING_EFFORT": str(
                matched_conditions.get("reasoning_effort", "")
            ),
            "HARNESS_IGNORE_USER_CONFIG": (
                "1" if authentication.get("ignore_user_config", False) else "0"
            ),
            "HARNESS_OUTER_SANDBOXED": "1" if isolation_prefix else "0",
        }
    )
    auth_mode = str(authentication.get("mode", "")).strip()
    if auth_mode != CHATGPT_SUBSCRIPTION_AUTH:
        codex_home = run_root / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        reasoning_effort = str(matched_conditions.get("reasoning_effort", "")).strip()
        if reasoning_effort:
            (codex_home / "config.toml").write_text(
                f"model_reasoning_effort = {json.dumps(reasoning_effort)}\n",
                encoding="utf-8",
            )
        environment["CODEX_HOME"] = str(codex_home)
    tools_condition = matched_conditions.get("tools", {})
    if isinstance(tools_condition, dict) and tools_condition.get("codex_binary"):
        pinned_codex = Path(str(tools_condition["codex_binary"])).expanduser().resolve()
        environment["PATH"] = os.pathsep.join(
            [str(pinned_codex.parent), environment.get("PATH", "")]
        ).rstrip(os.pathsep)
        # The PATH pin covers the official LH subprocess lookup. This explicit
        # variable is harmless for versions that do not consume it and makes the
        # intended tool identity visible in preserved evidence.
        environment["LH_HARNESS_CODEX_BINARY"] = str(pinned_codex)
    broker: BudgetBroker | None = None
    broker_snapshot: dict[str, Any] | None = None
    ledger_path = run_root / "budget-ledger.json"
    if provider_proxy is not None:
        broker = BudgetBroker(
            limits=BudgetLimits.from_mapping(budget),
            upstream_base_url=str(provider_proxy["upstream_base_url"]),
            upstream_api_key_env=(
                str(provider_proxy["api_key_env"])
                if provider_proxy.get("api_key_env")
                else None
            ),
            upstream_timeout_seconds=int(
                provider_proxy.get("upstream_timeout_seconds", 900)
            ),
            ledger_path=ledger_path,
        )
        try:
            broker.start()
        except OSError as error:
            raise RuntimeError(
                f"could not start the local matched-budget broker: {error}"
            ) from error
        environment.update(
            {
                "HARNESS_BUDGET_PROXY_URL": broker.base_url,
                "HARNESS_BUDGET_PROXY_KEY_ENV": "LONGCODE_BUDGET_PROXY_TOKEN",
                "LONGCODE_BUDGET_PROXY_TOKEN": "local-matched-budget-token",
            }
        )
    started = time.monotonic()
    timed_out = False
    try:
        exit_code, timed_out, stdout, stderr = _run_arm_process(
            command,
            cwd=workspace,
            environment=environment,
            timeout=timeout,
        )
    finally:
        if broker is not None:
            broker_snapshot = broker.ledger.snapshot()
            broker.close()
    wall_seconds = round(time.monotonic() - started, 3)
    hidden_commands = [
        command.replace(
            "{HIDDEN_ROOT}",
            shlex.quote(str(hidden_root)) if hidden_root else "",
        )
        for command in case.get("hidden_checks", [])
    ]
    hidden_checks = run_checks(
        workspace,
        hidden_commands,
        check_timeout,
    )
    hidden_passed = bool(hidden_checks) and all(item.passed for item in hidden_checks)
    arm_result = _read_optional_json(result_file)
    if timed_out and arm == "lh" and not arm_result:
        # The outer matched wall clock can terminate LH between role turns,
        # before its adapter writes arm-result.json. Completed Codex turns are
        # already durable in LH's canonical trajectories, so retain those
        # observed tokens as an honest lower bound while leaving coverage false
        # if the interrupted turn has no completion usage event.
        usage = _lh_codex_usage(run_root / "lh-arm" / "logs")
        observed_model = str(matched_conditions.get("model", "")).strip()
        arm_result.update(
            {
                "claimed_completed": False,
                "input_tokens": usage["input_tokens"],
                "cached_input_tokens": usage["cached_input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "model_calls": usage["model_calls"],
                "token_metrics_available": usage["token_metrics_available"],
                "observed_models": (
                    [observed_model]
                    if observed_model and usage["measured_model_calls"] > 0
                    else []
                ),
                "model_observation_source": "completed_lh_codex_turns_before_timeout",
                "role_call_counts": usage["role_call_counts"],
                "budget_enforced": True,
                "budget_exhausted": True,
                "budget_breach": False,
                "usage_files": usage["usage_files"],
            }
        )
    elif timed_out and arm == "longcode" and not arm_result:
        runtime = run_root / "longcode-arm" / "runtime"
        usage = _longcode_evidence_usage(runtime)
        controls = _longcode_control_snapshot(runtime)
        observed_model = str(matched_conditions.get("model", "")).strip()
        arm_result.update(
            {
                "claimed_completed": False,
                "input_tokens": usage["input_tokens"],
                "cached_input_tokens": usage["cached_input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                # A timeout without arm-result.json means the final backend
                # turn did not durably report usage. Keep completed calls plus
                # one interrupted call and mark coverage incomplete.
                "model_calls": usage["model_calls"] + 1,
                "token_metrics_available": False,
                "observed_models": (
                    [observed_model]
                    if observed_model and usage["model_calls"] > 0
                    else []
                ),
                "model_observation_source": "completed_longcode_turns_before_timeout",
                "longcode": controls,
                "committed_progress_lost": _longcode_committed_progress_lost(runtime),
                "budget_enforced": True,
                "budget_exhausted": True,
                "budget_breach": False,
                "usage_files": usage["usage_files"],
            }
        )
    if broker_snapshot is not None:
        arm_result.update(broker_result(broker_snapshot))
    success = bool(
        hidden_passed
        and not timed_out
        and not arm_result.get("budget_breach", False)
    )
    claimed_completed = bool(arm_result.get("claimed_completed", False))
    record = EvaluationRecord(
        case_id=run_id,
        base_case_id=str(case["id"]),
        repetition=repetition,
        arm=arm,
        suite=str(case["suite"]),
        success=success,
        false_completed=claimed_completed and not hidden_passed,
        recoverable_fault=bool(case.get("recoverable_fault", False)),
        recovered=(
            success
            if case.get("recoverable_fault", False)
            else (
                arm_result.get("recovered")
                if isinstance(arm_result.get("recovered"), bool)
                else None
            )
        ),
        committed_progress_lost=(
            arm_result.get("committed_progress_lost")
            if isinstance(arm_result.get("committed_progress_lost"), bool)
            else None
        ),
        input_tokens=int(arm_result.get("input_tokens", 0)),
        cached_input_tokens=int(arm_result.get("cached_input_tokens", 0)),
        output_tokens=int(arm_result.get("output_tokens", 0)),
        reasoning_tokens=int(arm_result.get("reasoning_tokens", 0)),
        model_calls=int(arm_result.get("model_calls", 0)),
        wall_seconds=wall_seconds,
        hidden_flow_pass=(
            hidden_passed if case.get("hidden_flow", False) else None
        ),
        human_acceptance=arm_result.get("human_acceptance"),
        run_exit_code=exit_code,
        timed_out=timed_out,
        snapshot_id=snapshot_id,
        token_metrics_available=bool(
            arm_result.get("token_metrics_available", False)
        ),
        budget_enforced=bool(
            arm_result.get("budget_enforced", False) or timed_out
        ),
        budget_exhausted=bool(
            arm_result.get("budget_exhausted", False) or timed_out
        ),
        budget_breach=bool(arm_result.get("budget_breach", False)),
        observed_models=list(arm_result.get("observed_models", [])),
        system_fingerprints=list(arm_result.get("system_fingerprints", [])),
        provider_route_requests=int(arm_result.get("provider_route_requests", 0)),
        condition_id=condition_id,
        hidden_checks_isolated=bool(hidden_root is not None and isolation_prefix),
    )
    evidence_payload = {
        "manifest": str(manifest_file),
        "case": case["id"],
        "arm": arm,
        "repetition": repetition,
        "snapshot_id": snapshot_id,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "budget": budget,
        "stdout": stdout,
        "stderr": stderr,
        "arm_result": arm_result,
        "matched_conditions": matched_conditions,
        "authentication": {
            "mode": auth_mode or None,
            "ignore_user_config": bool(authentication.get("ignore_user_config", False)),
            "stripped_environment_keys": sorted(
                str(key) for key in authentication.get("strip_environment_keys", [])
            ),
            "codex_home": "inherited" if auth_mode == CHATGPT_SUBSCRIPTION_AUTH else "isolated",
        },
        "process_environment": _redacted_process_environment(process_environment),
        "condition_id": condition_id,
        "provider_proxy": _redacted_provider_proxy(provider_proxy),
        "arm_isolation": {
            "enabled": bool(isolation_prefix),
            "command_prefix": isolation_prefix,
        },
        "hidden_checks": [item.to_dict() for item in hidden_checks],
        "record": record.__dict__,
    }
    (evidence_dir / "run.json").write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result_file.is_file():
        shutil.copy2(result_file, evidence_dir / "arm-result.json")
    if ledger_path.is_file():
        shutil.copy2(ledger_path, evidence_dir / "budget-ledger.json")
    if preserve_root:
        destination = preserve_root / safe_run / arm
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(run_root, destination, symlinks=True)
    temporary.cleanup()
    return record


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("version") != BENCHMARK_MANIFEST_VERSION:
        raise ValueError("benchmark manifest version must be 1")
    arms = set(manifest.get("arms", {}))
    if arms != REQUIRED_ARMS:
        raise ValueError("benchmark arms must be exactly direct, lh, and longcode")
    if not manifest.get("cases"):
        raise ValueError("benchmark manifest must contain cases")
    if int(manifest.get("repetitions", 1)) < 1:
        raise ValueError("benchmark repetitions must be at least 1")
    budget = manifest.get("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("benchmark budget must be an object")
    broker_keys = set(budget) & BROKER_BUDGET_KEYS
    provider_proxy = manifest.get("provider_proxy")
    if broker_keys and provider_proxy is None:
        raise ValueError(
            "strict model/token budget keys require provider_proxy"
        )
    if provider_proxy is not None:
        if not isinstance(provider_proxy, dict):
            raise ValueError("provider_proxy must be an object")
        if not provider_proxy.get("upstream_base_url"):
            raise ValueError("provider_proxy.upstream_base_url is required")
        if provider_proxy.get("require_loopback", False) and not _is_loopback_provider_url(
            str(provider_proxy["upstream_base_url"])
        ):
            raise ValueError(
                "provider_proxy.require_loopback only permits 127.0.0.1 or ::1"
            )
        strip_keys = provider_proxy.get("strip_environment_keys", [])
        if not isinstance(strip_keys, list) or any(
            not isinstance(key, str) or not key.strip() for key in strip_keys
        ):
            raise ValueError(
                "provider_proxy.strip_environment_keys must be a list of names"
            )
        BudgetLimits.from_mapping(budget)
    matched_conditions = manifest.get("matched_conditions", {})
    if matched_conditions and not isinstance(matched_conditions, dict):
        raise ValueError("matched_conditions must be an object")
    if isinstance(matched_conditions, dict) and matched_conditions:
        required_conditions = {"model", "reasoning_effort", "tools", "environment"}
        missing_conditions = required_conditions - set(matched_conditions)
        if missing_conditions:
            raise ValueError(
                "matched_conditions is missing: "
                + ", ".join(sorted(missing_conditions))
            )
        declared_model = str(matched_conditions["model"])
        for arm, config in manifest["arms"].items():
            raw_parts = (
                shlex.split(config["command"])
                if isinstance(config["command"], str)
                else list(config["command"])
            )
            if _flag_value(raw_parts, "--model") != declared_model:
                raise ValueError(
                    f"benchmark arm {arm} does not declare matched model {declared_model}"
                )
    authentication = manifest.get("authentication", {})
    if authentication and not isinstance(authentication, dict):
        raise ValueError("authentication must be an object")
    if isinstance(authentication, dict) and authentication:
        auth_mode = str(authentication.get("mode", "")).strip()
        if auth_mode not in {CHATGPT_SUBSCRIPTION_AUTH, "provider_api_key"}:
            raise ValueError(f"unsupported authentication mode: {auth_mode}")
        strip_keys = authentication.get("strip_environment_keys", [])
        if not isinstance(strip_keys, list) or any(
            not isinstance(key, str) or not key.strip() for key in strip_keys
        ):
            raise ValueError(
                "authentication.strip_environment_keys must be a list of names"
            )
        if auth_mode == CHATGPT_SUBSCRIPTION_AUTH and provider_proxy is not None:
            raise ValueError(
                "chatgpt_subscription authentication cannot use provider_proxy"
            )
    _process_environment(manifest)
    for arm, config in manifest["arms"].items():
        if not config.get("command"):
            raise ValueError(f"benchmark arm {arm} has no command")
    for case in manifest["cases"]:
        for field in ("id", "suite", "source", "hidden_checks"):
            if field not in case:
                raise ValueError(f"benchmark case is missing {field}")
        if not case["hidden_checks"]:
            raise ValueError(f"benchmark case {case['id']} needs hidden checks")
        uses_hidden_root = any(
            "{HIDDEN_ROOT}" in command for command in case["hidden_checks"]
        )
        if uses_hidden_root and not case.get("hidden_root"):
            raise ValueError(
                f"benchmark case {case['id']} uses HIDDEN_ROOT without hidden_root"
            )


def _snapshot_id(source: Path) -> str:
    snapshot = snapshot_tree(source)
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"benchmark source does not exist: {resolved}")
    return resolved


def _resolve_hidden_root(
    base: Path,
    case: dict[str, Any],
    *,
    source: Path,
) -> Path | None:
    value = case.get("hidden_root")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"hidden_root does not exist: {resolved}")
    try:
        resolved.relative_to(source)
    except ValueError:
        return resolved
    raise ValueError(
        f"benchmark case {case['id']} hidden_root must be outside the source fixture"
    )


def _command_parts(value: str | list[str], *, base: Path) -> list[str]:
    parts = shlex.split(value) if isinstance(value, str) else list(value)
    if not parts:
        raise ValueError("benchmark command must not be empty")
    executable = shutil.which(parts[0])
    if executable is None:
        candidate = Path(parts[0]).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            executable = str(candidate.resolve())
    if executable is None:
        raise FileNotFoundError(f"benchmark executable was not found: {parts[0]}")
    return [executable, *parts[1:]]


def _arm_isolation_prefix(
    value: dict[str, Any] | None, *, base: Path
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("arm_isolation must be an object")
    prefix = value.get("command_prefix")
    if not prefix:
        raise ValueError("arm_isolation.command_prefix is required")
    return _command_parts(prefix, base=base)


def _preflight_arm_isolation(
    manifest: dict[str, Any], *, base: Path, hidden_roots: set[Path]
) -> tuple[bool, str]:
    if not hidden_roots:
        return True, "no external hidden root is declared"
    try:
        prefix = _arm_isolation_prefix(manifest.get("arm_isolation"), base=base)
    except (ValueError, FileNotFoundError) as error:
        return False, str(error)
    if prefix is None:
        return (
            False,
            "external hidden checks exist but no inherited arm isolation is configured",
        )
    return _probe_hidden_read_denied(prefix, hidden_roots)


def _probe_hidden_read_denied(
    prefix: list[str], hidden_roots: set[Path]
) -> tuple[bool, str]:
    probe_code = (
        "import pathlib,sys\n"
        "try:\n pathlib.Path(sys.argv[1]).read_bytes()\n"
        "except OSError:\n raise SystemExit(0)\n"
        "raise SystemExit(41)\n"
    )
    checked: list[str] = []
    for root in sorted(hidden_roots):
        try:
            target = next(path for path in root.rglob("*") if path.is_file())
        except StopIteration:
            return False, f"hidden root has no probeable file: {root}"
        try:
            completed = subprocess.run(
                [*prefix, sys.executable, "-c", probe_code, str(target)],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"isolation probe failed to run: {error}"
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if completed.returncode == 41:
                detail = "hidden file remained readable"
            return (
                False,
                f"{target}: {detail or f'isolation exit {completed.returncode}'}",
            )
        checked.append(str(root))
    return True, f"read denial verified for {len(checked)} hidden root(s)"


def _longcode_evidence_usage(runtime: Path) -> dict[str, Any]:
    evidence = runtime / "evidence"
    paths = [
        *evidence.glob("manager/*/decision-*.json"),
        *evidence.glob("round-*/executor.json"),
        *evidence.glob("round-*/auditor.json"),
    ]
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "model_calls": 0,
        "usage_files": [],
    }
    for path in sorted(paths):
        payload = _read_optional_json(path)
        stdout = payload.get("stdout")
        if not isinstance(stdout, str) or not stdout:
            continue
        usage = _codex_jsonl_usage(stdout)
        if not usage["token_metrics_available"]:
            continue
        totals["model_calls"] += 1
        totals["usage_files"].append(str(path))
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            totals[key] += int(usage[key])
    return totals


def _longcode_control_snapshot(runtime: Path) -> dict[str, Any]:
    state = _read_optional_json(runtime / "state.json")
    return {
        key: state.get(key)
        for key in (
            "status",
            "rounds_run",
            "deterministic_check_batches",
            "auditor_calls",
            "manager_calls",
            "control_level_counts",
            "escalation_reasons",
        )
        if key in state
    }


def _longcode_committed_progress_lost(runtime: Path) -> bool | None:
    state = _read_optional_json(runtime / "state.json")
    criteria = state.get("criteria")
    events_path = runtime / "events.jsonl"
    if not isinstance(criteria, dict) or not events_path.is_file():
        return None
    verified: set[str] = set()
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "criterion_verified":
            continue
        criterion_id = event.get("data", {}).get("criterion_id")
        if criterion_id:
            verified.add(str(criterion_id))
    if not verified:
        return False
    for criterion_id in verified:
        criterion = criteria.get(criterion_id)
        if not isinstance(criterion, dict) or criterion.get("status") not in {
            "verified",
            "completed",
        }:
            return True
    return False


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_arm_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
) -> tuple[int | None, bool, str, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, False, stdout, stderr
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
        return None, True, stdout, stderr


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value)


def _flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _redacted_provider_proxy(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "upstream_base_url": value.get("upstream_base_url"),
        "api_key_env": value.get("api_key_env"),
        "upstream_timeout_seconds": value.get("upstream_timeout_seconds", 900),
        "require_loopback": bool(value.get("require_loopback", False)),
        "strip_environment_keys": list(value.get("strip_environment_keys", [])),
    }


def _process_environment(manifest: dict[str, Any]) -> dict[str, str]:
    value = manifest.get("process_environment", {})
    if not isinstance(value, dict):
        raise ValueError("process_environment must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("process_environment keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"process_environment.{key} must be a string")
        result[key] = item
    return result


def _redacted_process_environment(value: dict[str, str]) -> dict[str, str]:
    sensitive = ("key", "token", "secret", "password", "authorization", "cookie")
    return {
        key: ("***REDACTED***" if any(part in key.lower() for part in sensitive) else item)
        for key, item in value.items()
    }


def _is_loopback_provider_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}


def _probe_local_provider(
    base_url: str, *, expected_model: str
) -> tuple[bool, dict[str, Any]]:
    models_url = base_url.rstrip("/") + "/models"
    detail: dict[str, Any] = {
        "url": models_url,
        "expected_model": expected_model or None,
    }
    try:
        with urllib.request.urlopen(models_url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            detail["http_status"] = response.status
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        detail["error"] = str(error)
        return False, detail
    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = sorted(
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    )
    detail["available_models"] = models
    if expected_model and expected_model not in models:
        detail["error"] = "declared matched model is not loaded by the local server"
        return False, detail
    return True, detail


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
