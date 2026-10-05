from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .backends import CodexBackend
from .benchmark_runner import benchmark_preflight, run_benchmark
from .engine import LongCodeEngine
from .environment_adapter import JsonProcessEnvironmentAdapter
from .evaluation import aa_validation_report, comparison_report, load_records
from .goals import answer_product_question, approve_product_gate, revise_goal
from .lh_benchmark_arm import run_lh_arm
from .memory import MemoryStore
from .models import TaskContract
from .native_benchmark_arms import run_direct_arm, run_longcode_arm
from .product import build_product_spec, product_acceptance
from .process_adapter import JsonProcessAgentAdapter
from .routing import AgentAdaptiveManager, EvidenceDrivenManager
from .storage import RuntimeStore
from .verifier import run_checks
from .verifiers import EnvironmentEvidenceVerifier, ProductFlowCommandVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longcode",
        description="Adaptive, verification-first harness for long-running agents",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .agent_cli import add_commands
    add_commands(commands)

    init = commands.add_parser("init", help="create an immutable, versioned goal contract")
    init.add_argument("--workspace", default=".", help="repository or project directory")
    init.add_argument("--runtime", default=".longcode", help="durable runtime directory")
    init.add_argument("--goal", required=True, help="one durable objective")
    init.add_argument("--acceptance", action="append", default=[], help="repeatable criterion")
    init.add_argument("--constraint", action="append", default=[])
    init.add_argument("--non-goal", action="append", default=[])
    init.add_argument("--allowed-path", action="append", default=[])
    init.add_argument("--forbidden-path", action="append", default=[])
    init.add_argument("--check", action="append", default=[], help="trusted verification command")
    init.add_argument("--max-attempts", type=int, default=2)
    init.add_argument("--command-timeout", type=int, default=300)
    init.add_argument("--mode", choices=["task", "product", "auto"], default="task")
    init.add_argument("--problem")
    init.add_argument("--target-user", action="append", default=[])
    init.add_argument("--key-flow", action="append", default=[])
    init.add_argument("--must-have", action="append", default=[])
    init.add_argument("--visual-expectation", action="append", default=[])
    init.add_argument("--quality-attribute", action="append", default=[])
    init.add_argument("--assumption", action="append", default=[])
    init.add_argument("--open-question", action="append", default=[])
    init.add_argument("--human-gate", action="append", default=[])
    init.add_argument(
        "--product-check",
        action="append",
        default=[],
        help="trusted browser/E2E command that proves a product flow",
    )
    init.add_argument(
        "--e0-recipe",
        action="append",
        default=[],
        metavar="AC-ID=COMMAND",
        help="trusted deterministic command for an acceptance criterion; repeat for multiple commands",
    )

    run = commands.add_parser("run", help="resume the manager-executor-verifier loop")
    _runtime_argument(run)
    run.add_argument("--backend", choices=["codex", "claude", "native"], default="codex",
                     help="legacy run defaults to codex; chat/web default to native")
    run.add_argument("--provider", choices=["openai-codex", "openai", "anthropic", "compatible"])
    run.add_argument("--base-url")
    run.add_argument("--max-rounds", type=int, default=10)
    run.add_argument("--model", help="optional Codex model override")
    run.add_argument(
        "--agent-command",
        help="default external Agent adapter command using the JSON process protocol",
    )
    run.add_argument("--e1-command", help="external E1 executor adapter command")
    run.add_argument("--e2-command", help="external E2 executor adapter command")
    run.add_argument("--e3-command", help="external E3 executor adapter command")
    run.add_argument("--manager-command", help="external Manager adapter command")
    run.add_argument("--auditor-command", help="external Auditor adapter command")
    run.add_argument(
        "--agent-capability",
        action="append",
        default=[],
        help="capability exposed by external Agent adapters, for example browser or gui",
    )
    run.add_argument(
        "--environment-command",
        help="external transactional Browser/GUI/VM controller command",
    )
    run.add_argument(
        "--environment-capability",
        action="append",
        default=[],
        help="capability exposed by the external environment, for example browser, gui, or vm",
    )
    run.add_argument("--environment-timeout", type=int, default=300)
    run.add_argument("--e1-model", help="economy executor model")
    run.add_argument("--e2-model", help="standard executor model")
    run.add_argument("--e3-model", help="expert executor model")
    run.add_argument("--manager-model", help="manager model; defaults to --model")
    run.add_argument("--auditor-model", help="auditor model; defaults to --model")
    run.add_argument(
        "--manager-mode",
        choices=["evidence", "agent", "heuristic"],
        default="evidence",
        help="evidence calls the model Manager only after a bounded replan trigger",
    )
    run.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="Codex reasoning effort for executor, Auditor, and Manager calls",
    )
    run.add_argument("--codex", default="codex", help="Codex CLI executable")
    run.add_argument("--backend-timeout", type=int, default=1800)
    run.add_argument(
        "--codex-home",
        help="optional writable CODEX_HOME that already contains valid auth/config",
    )
    run.add_argument(
        "--no-auditor",
        action="store_true",
        help="skip independent semantic audit; deterministic checks still gate completion",
    )

    status = commands.add_parser("status", help="show durable goal and verification state")
    _runtime_argument(status)
    status.add_argument("--json", action="store_true")

    show = commands.add_parser("show-goal", help="show original and active goal contracts")
    _runtime_argument(show)

    revise = commands.add_parser("revise", help="create a new goal version without losing history")
    _runtime_argument(revise)
    revise.add_argument("--objective")
    revise.add_argument("--add-constraint", action="append", default=[])
    revise.add_argument("--remove-constraint", action="append", default=[])
    revise.add_argument("--add-acceptance", action="append", default=[])
    revise.add_argument("--remove-acceptance", action="append", default=[])

    answer = commands.add_parser("answer", help="resolve a blocking product question")
    _runtime_argument(answer)
    answer.add_argument("--question", required=True)
    answer.add_argument("--answer", required=True)

    approve = commands.add_parser("approve", help="approve an explicit product human gate")
    _runtime_argument(approve)
    approve.add_argument("--gate", required=True)
    approve.add_argument("--note", required=True)

    verify = commands.add_parser("verify", help="run deterministic checks without an executor")
    _runtime_argument(verify)

    events = commands.add_parser("events", help="inspect the append-only event trace")
    _runtime_argument(events)
    events.add_argument("--tail", type=int, default=20)

    faults = commands.add_parser("faults", help="inspect classified faults and recovery decisions")
    _runtime_argument(faults)
    faults.add_argument("--tail", type=int, default=20)

    graph = commands.add_parser("graph", help="show the typed durable task graph")
    _runtime_argument(graph)

    memory = commands.add_parser("memory", help="inspect trusted and recent episodic memory")
    _runtime_argument(memory)
    memory.add_argument("--episode-tail", type=int, default=10)

    evaluate = commands.add_parser("eval-report", help="summarize matched Direct/LH/LongCode runs")
    evaluate.add_argument("--input", required=True, help="JSON array or JSONL evaluation records")
    evaluate.add_argument("--candidate", default="longcode")
    evaluate.add_argument("--baseline", default="lh")

    eval_run = commands.add_parser(
        "eval-run",
        help="run Direct/LH/LongCode on isolated copies and score hidden checks",
    )
    eval_run.add_argument("--manifest", required=True)
    eval_run.add_argument("--output", required=True)
    eval_run.add_argument("--seed", type=int, default=0)
    eval_run.add_argument("--preserve-runs")
    eval_run.add_argument(
        "--timeout-seconds",
        type=int,
        help="override both the whole-run timeout and subscription wall budget",
    )
    eval_run.add_argument(
        "--only-run",
        action="append",
        default=[],
        metavar="CASE_ID:ARM",
        help="run one case/arm pair; repeat to run an exact subset",
    )
    eval_run.add_argument(
        "--descriptive-only",
        action="store_true",
        help="return success after execution without applying formal claim gates",
    )

    eval_preflight = commands.add_parser(
        "eval-preflight",
        help="audit matched model, tools, fixtures, budget broker, and credentials",
    )
    eval_preflight.add_argument("--manifest", required=True)
    eval_preflight.add_argument("--output")
    eval_preflight.add_argument("--timeout-seconds", type=int)

    eval_aa_run = commands.add_parser(
        "eval-aa-run",
        help="run an identical-arm A/A benchmark and check runner neutrality",
    )
    eval_aa_run.add_argument("--manifest", required=True)
    eval_aa_run.add_argument("--output", required=True)
    eval_aa_run.add_argument("--seed", type=int, default=0)
    eval_aa_run.add_argument("--preserve-runs")
    eval_aa_run.add_argument("--token-tolerance", type=int, default=0)
    eval_aa_run.add_argument("--report-output")

    eval_aa_report = commands.add_parser(
        "eval-aa-report", help="validate an existing three-label A/A result"
    )
    eval_aa_report.add_argument("--input", required=True)
    eval_aa_report.add_argument("--token-tolerance", type=int, default=0)
    eval_aa_report.add_argument("--report-output")

    eval_lh = commands.add_parser(
        "eval-arm-lh",
        help="run official LongHorizon-Harness using eval-run environment inputs",
    )
    eval_lh.add_argument("--lh-harness", default="lh-harness")
    eval_lh.add_argument("--agent", default="codex")
    eval_lh.add_argument("--model")
    eval_lh.add_argument("--prompt-language", choices=["en", "zh"], default="en")
    eval_lh.add_argument("--max-rounds", type=int, default=30)
    eval_lh.add_argument("--manager-timeout", type=int, default=600)
    eval_lh.add_argument("--executor-timeout", type=int, default=1800)
    eval_lh.add_argument("--auditor-timeout", type=int, default=600)
    eval_lh.add_argument("--codex-mcp-config")

    eval_direct = commands.add_parser(
        "eval-arm-direct",
        help="run one direct Codex episode using eval-run environment inputs",
    )
    eval_direct.add_argument("--codex", default="codex")
    eval_direct.add_argument("--model")
    eval_direct.add_argument("--backend-timeout", type=int, default=1800)

    eval_longcode = commands.add_parser(
        "eval-arm-longcode",
        help="run LongCode using eval-run environment inputs",
    )
    eval_longcode.add_argument("--codex", default="codex")
    eval_longcode.add_argument("--model")
    eval_longcode.add_argument("--backend-timeout", type=int, default=1800)
    eval_longcode.add_argument("--max-rounds", type=int, default=10)

    doctor = commands.add_parser("doctor", help="check local runtime prerequisites")
    doctor.add_argument("--codex", default="codex")
    doctor.add_argument("--backend", choices=["native", "codex", "claude"], default="codex")
    doctor.add_argument("--agent-command")
    doctor.add_argument("--runtime")
    return parser


def _eval_run_selectors(values: list[str]) -> set[tuple[str, str]] | None:
    if not values:
        return None
    selected: set[tuple[str, str]] = set()
    for raw in values:
        case_id, separator, arm = raw.rpartition(":")
        if not separator or not case_id or arm not in {"direct", "lh", "longcode"}:
            raise ValueError(
                f"invalid --only-run value {raw!r}; expected CASE_ID:direct, CASE_ID:lh, or CASE_ID:longcode"
            )
        selected.add((case_id, arm))
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, NotADirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    from .agent_cli import COMMANDS, dispatch
    if args.command in COMMANDS:
        return dispatch(args)
    if args.command == "eval-preflight":
        report = benchmark_preflight(
            args.manifest,
            timeout_seconds=args.timeout_seconds,
        )
        _write_optional_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 1

    if args.command == "eval-aa-run":
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        commands = [
            json.dumps(config.get("command"), sort_keys=True)
            for config in manifest.get("arms", {}).values()
        ]
        if len(commands) != 3 or len(set(commands)) != 1:
            raise ValueError("A/A manifest must use the identical command for all three arm labels")
        records = run_benchmark(
            manifest_path,
            args.output,
            seed=args.seed,
            preserve_runs=args.preserve_runs,
        )
        report = aa_validation_report(records, token_tolerance=args.token_tolerance)
        _write_optional_report(args.report_output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "eval-aa-report":
        report = aa_validation_report(
            load_records(args.input), token_tolerance=args.token_tolerance
        )
        _write_optional_report(args.report_output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "eval-arm-direct":
        return run_direct_arm(
            codex=args.codex,
            model=args.model,
            backend_timeout=args.backend_timeout,
        )

    if args.command == "eval-arm-longcode":
        return run_longcode_arm(
            codex=args.codex,
            model=args.model,
            backend_timeout=args.backend_timeout,
            max_rounds=args.max_rounds,
        )

    if args.command == "eval-arm-lh":
        return run_lh_arm(
            executable=args.lh_harness,
            agent=args.agent,
            model=args.model,
            prompt_language=args.prompt_language,
            max_rounds=args.max_rounds,
            manager_timeout=args.manager_timeout,
            executor_timeout=args.executor_timeout,
            auditor_timeout=args.auditor_timeout,
            codex_mcp_config=args.codex_mcp_config,
        )

    if args.command == "eval-run":
        records = run_benchmark(
            args.manifest,
            args.output,
            seed=args.seed,
            preserve_runs=args.preserve_runs,
            timeout_seconds=args.timeout_seconds,
            only_runs=_eval_run_selectors(args.only_run),
        )
        report = comparison_report(records)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if args.descriptive_only or report["all_testable_claims_pass"] else 1

    if args.command == "eval-report":
        report = comparison_report(
            load_records(args.input),
            candidate=args.candidate,
            baseline=args.baseline,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["all_testable_claims_pass"] else 1

    if args.command == "init":
        mode = "product" if args.mode == "auto" and (
            args.problem or args.target_user or args.key_flow or args.must_have
        ) else ("task" if args.mode == "auto" else args.mode)
        product = None
        acceptance = list(args.acceptance)
        checks = list(args.check)
        if mode == "product":
            product = build_product_spec(
                problem=args.problem or args.goal,
                target_users=args.target_user,
                key_flows=args.key_flow,
                must_haves=args.must_have,
                visual_expectations=args.visual_expectation,
                quality_attributes=args.quality_attribute,
                assumptions=args.assumption,
                open_questions=args.open_question,
                required_human_gates=args.human_gate,
                product_checks=args.product_check,
            )
            generated = product_acceptance(product, start_index=len(acceptance) + 1)
            acceptance.extend(item.description for item in generated)
            checks = checks or ["true"]
        else:
            if not acceptance:
                raise ValueError("task mode requires at least one --acceptance")
            if not checks:
                raise ValueError("task mode requires at least one --check")
        contract = TaskContract.create(
            args.goal,
            acceptance,
            constraints=args.constraint,
            non_goals=args.non_goal,
            allowed_paths=args.allowed_path or ["**"],
            forbidden_paths=args.forbidden_path or [".git/**"],
            checks=checks,
            max_attempts=args.max_attempts,
            command_timeout=args.command_timeout,
            mode=mode,
            product=product,
        )
        if product:
            generated_by_description = {
                item.description: item for item in product_acceptance(product, start_index=len(args.acceptance) + 1)
            }
            contract.acceptance_criteria = [
                generated_by_description.get(item.description, item)
                for item in contract.acceptance_criteria
            ]
            contract.default_verification_profile = "product_flow"
        _apply_e0_recipes(contract, args.e0_recipe)
        store = RuntimeStore(args.runtime)
        store.initialize(contract, args.workspace)
        print(f"Initialized goal {contract.goal_id} v1 at {store.root}")
        print("Run: longcode run --runtime", store.root)
        return 0

    if args.command == "doctor":
        if args.backend == "native":
            import platform
            from .model_bridge import ModelBridge
            print(f"Python: {sys.version.split()[0]} (required: >=3.11)")
            print(f"Node.js: {shutil.which('node') or 'NOT FOUND'} (required: >=22.19)")
            isolated = platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists()
            print(f"Command isolation: {'macOS sandbox-exec' if isolated else 'NOT AVAILABLE'}")
            try:
                result = ModelBridge().request("status", provider="openai-codex", timeout=20)
                print(f"Pi model component: ready ({len(result['models'])} subscription models)")
                print("Login status is local only; this check does not verify account access with a model request.")
            except (OSError, RuntimeError) as exc:
                print(f"Pi model component: {exc}")
                return 1
            return 0 if isolated and sys.version_info >= (3, 11) else 1
        if args.backend == "claude":
            found = shutil.which("claude")
            print(f"Claude CLI: {found or 'NOT FOUND; install and run claude auth login'}")
            return 0 if found else 1
        codex_path = shutil.which(args.codex)
        adapter_ok = False
        adapter_error = None
        if args.agent_command:
            try:
                JsonProcessAgentAdapter(args.agent_command)
                adapter_ok = True
            except (ValueError, FileNotFoundError) as error:
                adapter_error = str(error)
        print(f"Python: {sys.version.split()[0]} (required: >=3.11)")
        print(f"Codex CLI: {codex_path or 'NOT FOUND'}")
        if args.agent_command:
            print(f"External Agent adapter: {'valid' if adapter_ok else adapter_error}")
        if args.runtime:
            store = RuntimeStore(args.runtime)
            print(f"Runtime: {store.root} ({'valid' if store.exists else 'not initialized'})")
        return 0 if sys.version_info >= (3, 11) and (codex_path or adapter_ok) else 1

    store = RuntimeStore(args.runtime)
    if not store.exists:
        raise FileNotFoundError(f"runtime is not initialized: {store.root}")

    if args.command == "run":
        if args.environment_capability and not args.environment_command:
            raise ValueError(
                "--environment-capability requires --environment-command"
            )
        backend = _build_agent_backend(
            args,
            command=args.e2_command or args.agent_command,
            model=args.e2_model or args.model,
        )
        executor_tiers = {}
        for tier, command, model in (
            ("E1", args.e1_command, args.e1_model),
            ("E3", args.e3_command, args.e3_model),
        ):
            if command or model:
                executor_tiers[tier] = _build_agent_backend(
                    args,
                    command=command,
                    model=model,
                )
        auditor_backend = None if args.no_auditor else _build_agent_backend(
            args,
            command=args.auditor_command or args.agent_command,
            model=args.auditor_model or args.model,
        )
        manager = None
        if args.manager_mode in {"evidence", "agent"}:
            manager_backend = _build_agent_backend(
                args,
                command=args.manager_command or args.agent_command,
                model=args.manager_model or args.model,
            )
            manager_type = (
                EvidenceDrivenManager
                if args.manager_mode == "evidence"
                else AgentAdaptiveManager
            )
            manager = manager_type(manager_backend, store.workspace())
        active_contract = store.load_contract()
        verifier_plugins = (
            [ProductFlowCommandVerifier(active_contract.product.product_checks)]
            if active_contract.product and active_contract.product.product_checks
            else []
        )
        for capability in args.environment_capability:
            if capability in {"browser", "gui", "product"}:
                verifier_plugins.append(EnvironmentEvidenceVerifier(capability))
        transaction_manager = (
            JsonProcessEnvironmentAdapter(
                args.environment_command,
                store.root,
                timeout=args.environment_timeout,
                capabilities=args.environment_capability,
            )
            if args.environment_command
            else None
        )
        check_runner = None
        if args.backend in {"native", "claude"}:
            from .agent_checks import make_check_runner
            from .agent_runtime import Cancellation
            from .agent_config import agent_home
            from .agent_service import validate_checks
            validate_checks(active_contract.checks)
            check_runner = make_check_runner(agent_home(), Cancellation(), lambda *_: None)
        engine = LongCodeEngine(
            store,
            backend,
            auditor=auditor_backend,
            manager=manager,
            executor_tiers=executor_tiers,
            verifier_plugins=verifier_plugins,
            transaction_manager=transaction_manager,
            check_runner=check_runner,
        )
        from .agent_runtime import project_lock
        from .agent_config import agent_home
        with project_lock(agent_home(), store.workspace()):
            state = engine.run(max_rounds=args.max_rounds)
        _print_status(store, as_json=False)
        return 0 if state.status == "completed" else 1

    if args.command == "status":
        _print_status(store, as_json=args.json)
        return 0

    if args.command == "show-goal":
        print(
            json.dumps(
                {
                    "original": store.load_original_contract().to_dict(),
                    "active": store.load_contract().to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "revise":
        contract = revise_goal(
            store,
            objective=args.objective,
            add_constraints=args.add_constraint,
            remove_constraints=args.remove_constraint,
            add_acceptance=args.add_acceptance,
            remove_acceptance_ids=args.remove_acceptance,
        )
        print(f"Created active goal version {contract.version}")
        _print_status(store, as_json=False)
        return 0

    if args.command == "answer":
        contract = answer_product_question(
            store,
            question=args.question,
            answer=args.answer,
        )
        print(f"Resolved product question in goal version {contract.version}")
        _print_status(store, as_json=False)
        return 0

    if args.command == "approve":
        approve_product_gate(store, gate=args.gate, note=args.note)
        print(f"Approved product gate: {args.gate}")
        _print_status(store, as_json=False)
        return 0

    if args.command == "verify":
        contract = store.load_contract()
        results = run_checks(
            store.workspace(), contract.checks, contract.command_timeout_seconds
        )
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return 0 if all(item.passed for item in results) else 1

    if args.command == "events":
        events = store.iter_events()
        for item in events[-max(0, args.tail) :]:
            print(json.dumps(item, ensure_ascii=False))
        return 0


    if args.command == "faults":
        faults = store.iter_faults()
        for item in faults[-max(0, args.tail) :]:
            print(json.dumps(item, ensure_ascii=False))
        return 0

    if args.command == "graph":
        state = store.load_state()
        print(json.dumps(state.graph.to_dict() if state.graph else None, ensure_ascii=False, indent=2))
        return 0

    if args.command == "memory":
        memory_store = MemoryStore(store.root)
        print(
            json.dumps(
                memory_store.context(episode_tail=args.episode_tail),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    raise RuntimeError(f"unknown command: {args.command}")


def _print_status(store: RuntimeStore, *, as_json: bool) -> None:
    contract = store.load_contract()
    state = store.load_state()
    if as_json:
        print(
            json.dumps(
                {"goal": contract.to_dict(), "state": state.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"Goal v{contract.version}: {contract.objective}")
    print(f"Status: {state.status}; round: {state.round}")
    for criterion in contract.acceptance_criteria:
        item = state.criteria[criterion.id]
        print(
            f"  {criterion.id} [{item.status}] attempts={item.attempts}: "
            f"{criterion.description}"
        )
        if item.last_error:
            label = item.last_error_type or "ERROR"
            print(f"    last error [{label}]: {item.last_error}")
        if item.tier_history:
            print(f"    tiers: {' -> '.join(item.tier_history)}")
    if state.blocker:
        print(f"Blocker: {state.blocker}")
    if state.final_evidence:
        print("Final evidence:", ", ".join(state.final_evidence))


def _write_optional_report(path: str | None, report: dict) -> None:
    if not path:
        return
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"report output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _runtime_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime", default=".longcode")


def _apply_e0_recipes(contract: TaskContract, values: list[str]) -> None:
    if not values:
        return
    known = {item.id for item in contract.acceptance_criteria}
    parsed: dict[str, list[str]] = {}
    for value in values:
        target, separator, command = value.partition("=")
        target = target.strip()
        command = command.strip()
        if not separator or not target or not command:
            raise ValueError("--e0-recipe must use AC-ID=COMMAND")
        if target.isdigit():
            target = f"AC-{int(target):03d}"
        if target not in known:
            raise ValueError(f"unknown acceptance criterion for E0 recipe: {target}")
        parsed.setdefault(target, []).append(command)
    contract.deterministic_recipes.update(parsed)
    contract.acceptance_criteria = [
        replace(
            criterion,
            recipe_id=criterion.id,
            risk_level="low",
        )
        if criterion.id in parsed
        else criterion
        for criterion in contract.acceptance_criteria
    ]


def _build_agent_backend(
    args: argparse.Namespace,
    *,
    command: str | None,
    model: str | None,
):
    if command:
        return JsonProcessAgentAdapter(
            command,
            timeout=args.backend_timeout,
            capabilities=args.agent_capability,
        )
    if getattr(args, "backend", "codex") in {"native", "claude"}:
        from .agent_config import load_settings, validate_settings
        from .agent_cli import terminal_answer
        from .product_backends import make_backend
        settings = load_settings()
        settings["backend"] = args.backend
        for key, value in (("provider", args.provider), ("base_url", args.base_url),
                           ("model", model), ("reasoning", args.reasoning_effort)):
            if value is not None:
                settings[key] = value
        settings["timeout"] = args.backend_timeout
        return make_backend(validate_settings(settings), ask=terminal_answer)
    return CodexBackend(
        executable=args.codex,
        model=model,
        timeout=args.backend_timeout,
        codex_home=args.codex_home,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    raise SystemExit(main())
