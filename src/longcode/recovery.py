from __future__ import annotations

import hashlib
import re
from collections import Counter
from uuid import uuid4

from .models import CriterionState, FaultEvent, RecoveryDecision, Subtask
from .routing import next_tier


class FaultClassifier:
    def reported_failure(
        self,
        report: dict,
        *,
        subtask: Subtask,
        evidence: list[str],
    ) -> FaultEvent:
        value = report.get("fault", {})
        allowed_categories = {
            "provider", "agent_runtime", "tool", "environment", "execution",
            "verification", "policy", "control_plane", "goal",
        }
        allowed_scopes = {"tool_call", "episode", "subtask", "branch", "milestone", "run"}
        category = str(value.get("category", "")).strip()
        code = str(value.get("code", "")).strip()
        summary = str(value.get("summary", "")).strip()
        retryable = value.get("retryable")
        scope = str(value.get("scope", "subtask")).strip()
        severity = str(value.get("severity", "error")).strip()
        if (
            category not in allowed_categories
            or not code
            or not summary
            or not isinstance(retryable, bool)
            or scope not in allowed_scopes
            or severity not in {"info", "warning", "error", "fatal"}
        ):
            return _fault(
                "agent_runtime",
                "invalid_fault_report",
                "error",
                True,
                "episode",
                "executor",
                subtask,
                "Executor returned an invalid structured fault report",
                evidence,
            )
        return _fault(
            category,
            code,
            severity,
            retryable,
            scope,
            "executor",
            subtask,
            summary,
            evidence,
        )

    def backend_failure(self, result, *, role: str, subtask: Subtask | None) -> FaultEvent:
        message = f"{getattr(result, 'stderr', '')}\n{getattr(result, 'stdout', '')}".strip()
        lowered = message.lower()
        if any(token in lowered for token in ("rate limit", "429", "too many requests")):
            category, code, severity, retryable, scope = "provider", "rate_limit", "warning", True, "episode"
        elif any(token in lowered for token in ("unauthorized", "authentication", "invalid api key", "401")):
            category, code, severity, retryable, scope = "provider", "auth", "fatal", False, "run"
        elif any(token in lowered for token in ("network", "dns", "connection reset", "connection refused")):
            category, code, severity, retryable, scope = "provider", "network", "warning", True, "episode"
        elif any(token in lowered for token in ("timed out", "timeout")):
            category, code, severity, retryable, scope = "agent_runtime", "timeout", "warning", True, "episode"
        elif getattr(result, "return_code", None) not in (0, None):
            category, code, severity, retryable, scope = "agent_runtime", "crash", "error", True, "episode"
        else:
            category, code, severity, retryable, scope = "agent_runtime", "invalid_output", "error", True, "episode"
        return _fault(
            category,
            code,
            severity,
            retryable,
            scope,
            role,
            subtask,
            message or f"{role} backend returned no valid result",
            [],
        )

    def scope_violation(self, subtask: Subtask, violations: list[str], evidence: list[str]) -> FaultEvent:
        return _fault(
            "policy", "scope_violation", "fatal", False, "subtask", "executor", subtask,
            "scope violations: " + "; ".join(violations), evidence,
        )

    def check_failure(self, subtask: Subtask, failed_checks: list[str], evidence: list[str]) -> FaultEvent:
        return _fault(
            "verification", "check_failed", "error", True, "subtask", "verifier", subtask,
            "; ".join(failed_checks), evidence,
        )

    def audit_failure(self, subtask: Subtask, verdict: str, summary: str, evidence: list[str]) -> FaultEvent:
        code = "audit_uncertain" if verdict == "uncertain" else "audit_rejected"
        return _fault(
            "verification", code, "error", True, "subtask", "auditor", subtask,
            summary, evidence,
        )

    def verifier_failure(
        self,
        subtask: Subtask,
        *,
        code: str,
        summary: str,
        evidence: list[str],
    ) -> FaultEvent:
        return _fault(
            "verification",
            code or "verification_rejected",
            "error",
            True,
            "subtask",
            "verifier",
            subtask,
            summary,
            evidence,
        )

    def environment_failure(self, *, code: str, summary: str, subtask: Subtask | None = None) -> FaultEvent:
        return _fault(
            "environment", code, "warning", True, "episode", "environment", subtask, summary, [],
        )


class SafetyBelt:
    """Deterministic recovery and escalation policy."""

    def __init__(self, *, escalate_after_same_failure: int = 2):
        if escalate_after_same_failure < 1:
            raise ValueError("escalate_after_same_failure must be at least 1")
        self.escalate_after_same_failure = escalate_after_same_failure

    def decide(
        self,
        fault: FaultEvent,
        criterion_state: CriterionState,
        *,
        current_tier: str,
        allowed_tiers: list[str],
        attempts_remaining: bool,
        recovery_budget_remaining: bool,
    ) -> RecoveryDecision:
        if not recovery_budget_remaining:
            return RecoveryDecision("block", "Recovery budget exhausted", recovery_scope="run")
        if fault.category == "policy":
            return RecoveryDecision(
                "rollback", "Policy violations cannot be bypassed by model escalation", recovery_scope="subtask"
            )
        if fault.category == "goal":
            return RecoveryDecision("ask", fault.summary, recovery_scope="run")
        if not fault.retryable:
            return RecoveryDecision("block", f"Non-retryable {fault.category}:{fault.code}", recovery_scope=fault.scope)
        counts = Counter(criterion_state.fault_fingerprints)
        same_failures = counts[fault.fingerprint] + 1
        if fault.category in {"provider", "tool", "environment", "control_plane"}:
            if not attempts_remaining:
                return RecoveryDecision("block", "Transient retry budget exhausted", recovery_scope=fault.scope)
            action = (
                "retry_backoff"
                if fault.category == "provider"
                else ("restart_tool" if fault.category == "tool" else "retry")
            )
            return RecoveryDecision(
                action,
                f"Retry transient {fault.category}:{fault.code} without changing executor capability",
                backoff_seconds=2 if action == "retry_backoff" else 0,
                recovery_scope=fault.scope,
            )
        if fault.code == "audit_uncertain":
            if not attempts_remaining:
                return RecoveryDecision("block", "Verifier strengthening budget exhausted", recovery_scope="subtask")
            return RecoveryDecision(
                "strengthen_verifier",
                "Missing audit evidence should strengthen verification before executor escalation",
                recovery_scope="subtask",
            )
        if same_failures >= self.escalate_after_same_failure:
            stronger = next_tier(current_tier, allowed_tiers)
            if stronger:
                return RecoveryDecision(
                    "escalate",
                    f"Same verified failure repeated {same_failures} times",
                    next_tier=stronger,
                    recovery_scope="subtask",
                )
            return RecoveryDecision(
                "replan",
                "Failure repeated at the strongest configured executor tier",
                recovery_scope="subtask",
            )
        if not attempts_remaining:
            return RecoveryDecision("block", "Current executor tier attempt budget exhausted", recovery_scope="subtask")
        return RecoveryDecision(
            "retry",
            "Allow one evidence-driven repair at the current executor tier",
            recovery_scope=fault.scope,
        )


def register_fault(state: CriterionState, fault: FaultEvent, decision: RecoveryDecision) -> None:
    state.fault_fingerprints.append(fault.fingerprint)
    state.last_error_type = f"{fault.category.upper()}_{fault.code.upper()}"
    state.last_error = fault.summary
    state.last_recovery_action = decision.action
    state.next_tier = decision.next_tier or state.next_tier


def _fault(
    category: str,
    code: str,
    severity: str,
    retryable: bool,
    scope: str,
    role: str,
    subtask: Subtask | None,
    summary: str,
    evidence: list[str],
) -> FaultEvent:
    normalized = re.sub(r"\d+", "#", " ".join(summary.lower().split()))[:500]
    payload = f"{category}|{code}|{normalized}"
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return FaultEvent(
        id=f"FAULT-{uuid4()}",
        category=category,
        code=code,
        severity=severity,
        retryable=retryable,
        scope=scope,
        source_role=role,
        subtask_id=subtask.id if subtask else None,
        fingerprint=fingerprint,
        summary=summary,
        evidence=list(evidence),
    )
