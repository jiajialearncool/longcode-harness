from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .models import VerifierResult
from .verifiers import VerificationContext


class SemanticContractVerifier:
    """Compile supported public requirements into deterministic black-box probes.

    V1.2 deliberately keeps this layer evidence-driven: a probe is activated only
    when both the public contract and the workspace expose the required interface.
    It never reads benchmark hidden checks or relies on executor self-report.
    """

    verifier_id = "semantic-contract-probe"
    kind = "behavioral"
    profiles = {"*"}

    def verify(self, context: VerificationContext) -> VerifierResult:
        compiled = _compile_json_migration_probe(context)
        if compiled is None:
            return VerifierResult(
                verifier_id=self.verifier_id,
                kind=self.kind,
                verdict="pass",
                required=False,
                summary="No supported executable semantic probe was implied by the public contract",
                evidence=[],
                details={"applicable": False},
            )
        script, source = compiled
        try:
            obligations = _run_json_migration_probe(
                context.workspace,
                script,
                source,
                timeout_seconds=min(context.contract.command_timeout_seconds, 30),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            return VerifierResult(
                verifier_id=self.verifier_id,
                kind=self.kind,
                verdict="uncertain",
                required=True,
                summary=f"Could not execute the compiled JSON migration probe: {error}",
                evidence=[f"entrypoint={script.name}", f"fixture={source.name}"],
                fault_code="semantic_probe_unavailable",
                details={
                    "applicable": True,
                    "probe_kind": "json_migration_cli",
                    "entrypoint": script.name,
                    "fixture": source.name,
                },
            )

        failed = [item for item in obligations if item["status"] != "pass"]
        summary = (
            "All public JSON migration behaviors were demonstrated"
            if not failed
            else "; ".join(
                f"{item['id']}: expected {item['expected']}; observed {item['observed']}"
                for item in failed
            )
        )
        return VerifierResult(
            verifier_id=self.verifier_id,
            kind=self.kind,
            verdict="pass" if not failed else "fail",
            required=True,
            summary=summary,
            evidence=[
                f"{item['id']}={item['status']}: {item['observed']}"
                for item in obligations
            ],
            fault_code=None if not failed else "semantic_behavior_failed",
            details={
                "applicable": True,
                "probe_kind": "json_migration_cli",
                "entrypoint": script.name,
                "fixture": source.name,
                "reproduction": (
                    f"{sys.executable} {script.name} --input {source.name} "
                    "--output <temporary-output.json>"
                ),
                "obligations": obligations,
            },
        )


def _compile_json_migration_probe(
    context: VerificationContext,
) -> tuple[Path, Path] | None:
    descriptions = [
        context.contract.objective,
        *(item.description for item in context.contract.acceptance_criteria),
    ]
    text = "\n".join(descriptions).lower()
    signals = {
        "migration": any(token in text for token in ("migration", "migrate", "迁移")),
        "json": "json" in text,
        "input_output": (
            ("--input" in text and "--output" in text)
            or ("读取" in text and "写入" in text)
            or ("input" in text and "output" in text)
        ),
        "dry_run": "dry-run" in text or "dry run" in text,
        "idempotent": any(token in text for token in ("idempot", "重复", "两次", "rerun")),
    }
    if not all(signals.values()):
        return None

    fixtures = sorted(context.workspace.glob("*.json"))
    if not fixtures:
        return None
    scripts: list[Path] = []
    for path in sorted(context.workspace.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if all(flag in source for flag in ("--input", "--output", "--dry-run")):
            scripts.append(path)
    if len(scripts) != 1:
        return None
    return scripts[0], fixtures[0]


def _run_json_migration_probe(
    workspace: Path,
    script: Path,
    source: Path,
    *,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    if not isinstance(original, dict):
        raise ValueError("JSON migration fixture must contain an object")

    obligations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="longcode-semantic-probe-") as directory:
        root = Path(directory)
        probe_input = root / "input.json"
        output = root / "output.json"
        probe_input.write_bytes(original_bytes)

        dry = _run_cli(
            workspace,
            script,
            probe_input,
            output,
            timeout_seconds=timeout_seconds,
            dry_run=True,
        )
        _obligation(
            obligations,
            "dry_run_no_mutation",
            dry.returncode == 0
            and not output.exists()
            and probe_input.read_bytes() == original_bytes,
            "dry-run exits successfully without creating output or changing input",
            (
                f"exit={dry.returncode}, output_exists={output.exists()}, "
                f"input_unchanged={probe_input.read_bytes() == original_bytes}"
            ),
        )

        real = _run_cli(
            workspace,
            script,
            probe_input,
            output,
            timeout_seconds=timeout_seconds,
            dry_run=False,
        )
        output_exists = output.is_file()
        parsed: Any = None
        parse_error = ""
        if output_exists:
            try:
                parsed = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                parse_error = str(error)
        _obligation(
            obligations,
            "real_run_writes_json",
            real.returncode == 0 and output_exists and isinstance(parsed, dict),
            "real run exits successfully and writes a JSON object",
            f"exit={real.returncode}, output_exists={output_exists}, parse_error={parse_error or 'none'}",
        )

        transformed = isinstance(parsed, dict) and parsed != original
        _obligation(
            obligations,
            "migration_transforms_semantics",
            transformed,
            "a migration changes the input document's JSON semantics",
            (
                "output document differs from input"
                if transformed
                else "output document is semantically identical to input"
            ),
        )

        preservation_keys = [
            key
            for key in original
            if any(
                marker in key.lower()
                for marker in ("unknown", "extra", "extension", "custom", "metadata", "keep")
            )
        ]
        preserved = isinstance(parsed, dict) and all(
            key in parsed and parsed[key] == original[key] for key in preservation_keys
        )
        _obligation(
            obligations,
            "unknown_fields_preserved",
            preserved,
            "fields explicitly marked unknown/custom/extra are retained unchanged",
            (
                "preserved keys: " + ", ".join(preservation_keys)
                if preserved
                else "missing or changed keys: "
                + ", ".join(
                    key
                    for key in preservation_keys
                    if not isinstance(parsed, dict)
                    or key not in parsed
                    or parsed[key] != original[key]
                )
            ),
        )

        first_bytes = output.read_bytes() if output_exists else b""
        again = _run_cli(
            workspace,
            script,
            probe_input,
            output,
            timeout_seconds=timeout_seconds,
            dry_run=False,
        )
        stable = (
            output.is_file()
            and output.read_bytes() == first_bytes
            and probe_input.read_bytes() == original_bytes
        )
        _obligation(
            obligations,
            "rerun_is_stable",
            again.returncode == 0 and stable,
            "a second real run produces identical output and never changes input",
            f"exit={again.returncode}, stable={stable}",
        )

        conflict_output = root / "conflict.json"
        conflict_output.write_text('{"occupied": true}\n', encoding="utf-8")
        conflict = _run_cli(
            workspace,
            script,
            probe_input,
            conflict_output,
            timeout_seconds=timeout_seconds,
            dry_run=False,
        )
        conflict_text = f"{conflict.stdout}\n{conflict.stderr}".lower()
        conflict_reported = conflict.returncode != 0 and "conflict" in conflict_text
        _obligation(
            obligations,
            "conflict_is_reported",
            conflict_reported,
            "an existing incompatible output is rejected and reported as a conflict",
            f"exit={conflict.returncode}, mentions_conflict={'conflict' in conflict_text}",
        )
    return obligations


def _run_cli(
    workspace: Path,
    script: Path,
    source: Path,
    output: Path,
    *,
    timeout_seconds: int,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


def _obligation(
    target: list[dict[str, Any]],
    identifier: str,
    passed: bool,
    expected: str,
    observed: str,
) -> None:
    target.append(
        {
            "id": identifier,
            "status": "pass" if passed else "fail",
            "expected": expected,
            "observed": observed,
        }
    )
