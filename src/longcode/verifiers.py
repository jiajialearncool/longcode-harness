from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import (
    AuditResult,
    CommandResult,
    Subtask,
    TaskContract,
    TaskState,
    VerifierResult,
)
from .scope import ScopeResult, check_scope
from .verifier import run_checks


class VerifierPlugin(Protocol):
    verifier_id: str
    kind: str

    def verify(self, context: "VerificationContext") -> VerifierResult: ...


@dataclass(frozen=True)
class VerificationContext:
    workspace: Path
    contract: TaskContract
    state: TaskState
    subtask: Subtask | None
    changed_paths: list[str]
    executor_report: dict | None
    command_results: list[CommandResult] = field(default_factory=list)


@dataclass
class VerificationOutcome:
    verdict: str
    results: list[VerifierResult]
    scope: ScopeResult
    commands: list[CommandResult]
    audit: AuditResult | None = None
    auditor_backend_result: object | None = None
    deterministic_verdict: str = "pass"
    auditor_invoked: bool = False
    escalation_reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "results": [item.to_dict() for item in self.results],
            "scope": self.scope.to_dict(),
            "commands": [item.to_dict() for item in self.commands],
            "audit": self.audit.to_dict() if self.audit else None,
            "deterministic_verdict": self.deterministic_verdict,
            "auditor_invoked": self.auditor_invoked,
            "escalation_reason": self.escalation_reason,
        }


class VerifierMesh:
    """Compose deterministic, semantic, product, GUI, and policy verifiers."""

    def __init__(self, *, auditor=None, plugins: list[VerifierPlugin] | None = None, check_runner=None):
        self.auditor = auditor
        self.plugins = list(plugins or [])
        self.check_runner = check_runner or run_checks

    def verify(
        self,
        workspace: Path,
        contract: TaskContract,
        state: TaskState,
        *,
        subtask: Subtask | None,
        changed_paths: list[str],
        executor_report: dict | None,
    ) -> VerificationOutcome:
        allowed = contract.allowed_paths if subtask is None else subtask.allowed_paths
        forbidden = contract.forbidden_paths if subtask is None else subtask.forbidden_paths
        scope = check_scope(changed_paths, allowed, forbidden)
        commands_to_run = contract.checks if subtask is None else subtask.checks
        commands = self.check_runner(workspace, commands_to_run, contract.command_timeout_seconds)
        results = [
            VerifierResult(
                verifier_id="scope",
                kind="policy",
                verdict="pass" if scope.passed else "fail",
                required=True,
                summary="Candidate changes are within scope" if scope.passed else "; ".join(scope.violations),
                evidence=list(scope.changed_paths),
                fault_code=None if scope.passed else "scope_violation",
            ),
            VerifierResult(
                verifier_id="commands",
                kind="deterministic",
                verdict=("pass" if all(item.passed for item in commands) else "fail") if commands else "uncertain",
                required=True,
                summary=(
                    "No executable verification commands" if not commands else "All configured commands passed"
                    if all(item.passed for item in commands)
                    else "Failed checks: " + ", ".join(item.command for item in commands if not item.passed)
                ),
                evidence=[item.command for item in commands],
                fault_code=None if all(item.passed for item in commands) else "check_failed",
            ),
        ]
        context = VerificationContext(
            workspace=workspace,
            contract=contract,
            state=state,
            subtask=subtask,
            changed_paths=changed_paths,
            executor_report=executor_report,
            command_results=commands,
        )
        profile = contract.default_verification_profile if subtask is None else subtask.verification_profile
        selected_plugins = [
            plugin
            for plugin in self.plugins
            if _plugin_applies(plugin, profile)
        ]
        for plugin in selected_plugins:
            results.append(plugin.verify(context))

        audit = None
        backend_result = None
        deterministic_verdict = _aggregate(results)
        hard_failure = deterministic_verdict == "fail"
        semantic_profiles = {
            "semantic",
            "design",
            "ux",
            "security",
            "security_strict",
            "irreversible",
        }
        claim_conflict = bool(
            deterministic_verdict == "pass"
            and subtask is None
            and isinstance(executor_report, dict)
            and executor_report.get("claimed_complete") is False
        )
        if hard_failure:
            escalation_reason = None
        elif deterministic_verdict == "uncertain":
            escalation_reason = "required verifier returned uncertain"
        elif profile in semantic_profiles:
            escalation_reason = f"verification profile requires semantic review: {profile}"
        elif claim_conflict:
            escalation_reason = "executor completion claim conflicts with deterministic evidence"
        else:
            escalation_reason = None
        auditor_required = escalation_reason is not None

        if auditor_required and self.auditor is not None:
            backend_result, audit = self.auditor.audit(
                workspace,
                contract,
                state,
                subtask=subtask,
                changed_paths=changed_paths,
                checks=commands,
                executor_report=executor_report,
            )
            results.append(
                VerifierResult(
                    verifier_id="semantic-auditor",
                    kind="semantic",
                    verdict=audit.verdict,
                    required=True,
                    summary=audit.summary,
                    evidence=list(audit.evidence),
                    fault_code=audit.fault_code or (
                        None if audit.verdict == "pass" else (
                            "audit_uncertain" if audit.verdict == "uncertain" else "audit_rejected"
                        )
                    ),
                )
            )
        elif auditor_required:
            results.append(
                VerifierResult(
                    verifier_id="semantic-auditor",
                    kind="semantic",
                    verdict="uncertain",
                    required=True,
                    summary="Semantic review is required but no auditor is configured",
                    evidence=[],
                    fault_code="auditor_missing",
                )
            )
        else:
            results.append(
                VerifierResult(
                    verifier_id="semantic-auditor",
                    kind="semantic",
                    verdict="skipped",
                    required=False,
                    summary=(
                        "Skipped because deterministic verification failed"
                        if hard_failure
                        else "Skipped because deterministic evidence is conclusive"
                    ),
                    evidence=[],
                )
            )

        # Product-flow claims require a non-semantic environment verifier. An LLM opinion alone is
        # insufficient evidence that a user flow actually ran.
        if profile in {"product_flow", "visual", "visual_strict"} and not any(
            item.kind in {"browser", "gui", "product"} and item.required
            for item in results
        ):
            results.append(
                VerifierResult(
                    verifier_id="product-flow-evidence",
                    kind="product",
                    verdict="uncertain",
                    required=True,
                    summary="Product/visual claim requires a browser/GUI/product verifier",
                    evidence=[],
                    fault_code="product_verifier_missing",
                )
            )
        return VerificationOutcome(
            verdict=_aggregate(results),
            results=results,
            scope=scope,
            commands=commands,
            audit=audit,
            auditor_backend_result=backend_result,
            deterministic_verdict=deterministic_verdict,
            auditor_invoked=backend_result is not None,
            escalation_reason=escalation_reason,
        )


def _plugin_applies(plugin: VerifierPlugin, profile: str) -> bool:
    profiles = getattr(plugin, "profiles", None)
    return profiles is None or profile in profiles or "*" in profiles


def _aggregate(results: list[VerifierResult]) -> str:
    required = [item for item in results if item.required]
    if any(item.verdict == "fail" for item in required):
        return "fail"
    if any(item.verdict != "pass" for item in required):
        return "uncertain"
    return "pass"


class ProductFlowCommandVerifier:
    """Treat an explicit browser/E2E command as environment evidence, not an LLM opinion."""

    verifier_id = "product-flow-command"
    kind = "browser"
    profiles = {"product_flow", "visual", "visual_strict"}

    def __init__(self, commands: list[str], *, required: bool = True):
        self.commands = [item.strip() for item in commands if item.strip()]
        if not self.commands:
            raise ValueError("product flow verifier requires at least one command")
        self.required = required

    def verify(self, context: VerificationContext) -> VerifierResult:
        results = run_checks(
            context.workspace,
            self.commands,
            context.contract.command_timeout_seconds,
        )
        failed = [item for item in results if not item.passed]
        return VerifierResult(
            verifier_id=self.verifier_id,
            kind=self.kind,
            verdict="fail" if failed else "pass",
            required=self.required,
            summary=(
                "Product/browser flow commands passed"
                if not failed
                else "Product/browser flow commands failed: "
                + ", ".join(item.command for item in failed)
            ),
            evidence=[
                f"{item.command} exit={item.exit_code} timeout={item.timed_out}"
                for item in results
            ],
            fault_code="product_flow_failed" if failed else None,
        )


class EnvironmentEvidenceVerifier:
    """Verify signed-off observations emitted by a configured external environment controller."""

    def __init__(
        self,
        evidence_kind: str,
        *,
        profiles: set[str] | None = None,
        required: bool = True,
    ):
        evidence_kind = evidence_kind.strip()
        if not evidence_kind:
            raise ValueError("environment evidence kind must not be empty")
        self.verifier_id = f"environment-evidence-{evidence_kind}"
        self.kind = evidence_kind
        self.profiles = profiles or {"product_flow", "visual", "visual_strict"}
        self.required = required

    def verify(self, context: VerificationContext) -> VerifierResult:
        if (
            context.subtask is not None
            and self.kind not in context.subtask.required_capabilities
        ):
            return VerifierResult(
                verifier_id=self.verifier_id,
                kind=self.kind,
                verdict="pass",
                required=False,
                summary=f"{self.kind} evidence is not required by this subtask",
                evidence=[],
            )
        source = (
            context.subtask.environment_context.get("evidence", [])
            if context.subtask is not None
            else context.state.environment_evidence
        )
        matching = [
            item
            for item in source
            if isinstance(item, dict) and item.get("kind") == self.kind
        ]
        if context.subtask is None and not matching:
            return VerifierResult(
                verifier_id=self.verifier_id,
                kind=self.kind,
                verdict="pass",
                required=False,
                summary=f"No completed subtask used the {self.kind} environment",
                evidence=[],
            )
        failed = [item for item in matching if item.get("verdict") == "fail"]
        passed = [item for item in matching if item.get("verdict") == "pass"]
        if failed:
            verdict = "fail"
            summary = f"External {self.kind} environment reported a failed observation"
            fault_code = f"{self.kind}_environment_failed"
        elif passed:
            verdict = "pass"
            summary = f"External {self.kind} environment supplied passing evidence"
            fault_code = None
        else:
            verdict = "uncertain"
            summary = f"No passing {self.kind} environment evidence was supplied"
            fault_code = f"{self.kind}_environment_evidence_missing"
        evidence = [
            str(item.get("path") or item.get("summary") or item.get("id") or item)
            for item in matching
        ]
        return VerifierResult(
            verifier_id=self.verifier_id,
            kind=self.kind,
            verdict=verdict,
            required=self.required,
            summary=summary,
            evidence=evidence,
            fault_code=fault_code,
        )
