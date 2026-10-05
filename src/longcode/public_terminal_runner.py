from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .claude_backend import ClaudeCodeBackend
from .engine import LongCodeEngine
from .inplace_adapter import InPlaceWorkspaceAdapter
from .models import TaskContract
from .routing import AgentAdaptiveManager
from .storage import RuntimeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LongCode inside a Terminal-Bench task")
    parser.add_argument("--workspace", default="/app")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--model", default=os.environ.get("LONGCODE_PUBLIC_MODEL", "qwen3.7-plus"))
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--backend-timeout", type=int, default=1800)
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    runtime = Path(args.runtime).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    instruction = Path(args.instruction_file).read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError("Terminal-Bench instruction is empty")

    contract = TaskContract.create(
        instruction,
        [
            "Fully satisfy the original task instruction and leave all required externally visible artifacts and behavior correct"
        ],
        constraints=[
            "Do not read, alter, or bypass benchmark hidden tests or ground-truth artifacts",
            "Do not weaken validation or fabricate completion evidence",
            "Preserve correct progress across recovery attempts",
        ],
        allowed_paths=["**"],
        forbidden_paths=[".git/**", ".longcode/**", ".harness/**", "gt/**"],
        checks=["true"],
        max_attempts=args.max_rounds,
        command_timeout=args.command_timeout,
    )
    store = RuntimeStore(runtime)
    store.initialize(contract, workspace)
    proxy_script = os.environ.get("LONGCODE_CLAUDE_REQUEST_PROXY") or None
    overrides_raw = os.environ.get("LONGCODE_CLAUDE_REQUEST_OVERRIDES", "{}")
    overrides = json.loads(overrides_raw)
    if not isinstance(overrides, dict):
        raise ValueError("LONGCODE_CLAUDE_REQUEST_OVERRIDES must be a JSON object")
    backend = ClaudeCodeBackend(
        executable=os.environ.get("LONGCODE_CLAUDE_BINARY", "claude"),
        model=args.model,
        timeout=args.backend_timeout,
        max_turns=args.max_turns,
        request_proxy_script=proxy_script,
        request_overrides=overrides,
        proxy_log_dir=runtime / "proxy-logs",
    )
    state = LongCodeEngine(
        store,
        backend,
        auditor=backend,
        manager=AgentAdaptiveManager(backend, workspace),
        executor_tiers={"E1": backend, "E2": backend, "E3": backend},
        transaction_manager=InPlaceWorkspaceAdapter(runtime),
    ).run(max_rounds=args.max_rounds)
    payload = {
        "schema_version": 1,
        "benchmark": "terminal-bench-2.1",
        "candidate": "longcode-harness",
        "model": args.model,
        "status": state.status,
        "claimed_completed": state.status == "completed",
        "rounds_run": state.round,
        "blocker": state.blocker,
        "usage": dict(backend.usage_totals),
        "runtime": str(runtime),
        "completion_authority": "longcode verifier mesh; final score remains Harbor's external verifier",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
